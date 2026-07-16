import torch
import torch.nn as nn
import torch.nn.functional as F
import time
from typing import Tuple

# Ensure CUDA is available
if not torch.cuda.is_available():
    print("[ERROR] CUDA is not available. This diagnostics script must run on a GPU.")
    exit(1)

device = torch.device("cuda")
device_name = torch.cuda.get_device_name(device)
major, minor = torch.cuda.get_device_capability(device)
print("======================================================================")
print("              GPU Hardware & Environment Diagnostics                  ")
print("======================================================================")
print(f"Device Name: {device_name}")
print(f"Compute Capability: {major}.{minor} (sm_{major}{minor})")
print(f"PyTorch Version: {torch.__version__}")
print(f"Float8 E4M3 Support: {hasattr(torch, 'float8_e4m3fn')}")
print(f"Float8 E5M2 Support: {hasattr(torch, 'float8_e5m2')}")
print(f"scaled_mm Support: {hasattr(torch, '_scaled_mm')}")
print("======================================================================\n")

# Warmup helper
def warmup_gpu():
    dummy = torch.randn(1000, 1000, device=device)
    for _ in range(50):
        _ = dummy @ dummy
    torch.cuda.synchronize()

# Time measurement helper
def measure_time_ms(fn, num_iters: int = 100) -> float:
    # Warmup
    for _ in range(20):
        fn()
    torch.cuda.synchronize()
    
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    
    start_event.record()
    for _ in range(num_iters):
        fn()
    end_event.record()
    
    torch.cuda.synchronize()
    return start_event.elapsed_time(end_event) / num_iters

# 1. GEMM Microbenchmarks
def run_gemm_diagnostics():
    print("--- 1. GEMM Microbenchmarks (FP8 vs BF16) ---")
    # Shapes: (M, N, K) where A is (M, K), B is (N, K) transposed to (K, N)
    gemm_shapes = [
        (4096, 4096, 4096),
        (4096, 1000, 4096),
        (512, 4096, 25088)
    ]
    
    for M, N, K in gemm_shapes:
        print(f"\nShape: M={M}, N={N}, K={K}")
        
        # BF16 setup
        a_bf16 = torch.randn(M, K, dtype=torch.bfloat16, device=device)
        b_bf16 = torch.randn(N, K, dtype=torch.bfloat16, device=device)
        
        # FP8 setup
        # Quantize inputs
        a_fp8 = a_bf16.clamp(-448.0, 448.0).to(torch.float8_e4m3fn)
        b_fp8 = b_bf16.clamp(-448.0, 448.0).to(torch.float8_e4m3fn)
        
        # Scaling factors (inverse scales)
        scale_a = torch.tensor([1.0], device=device, dtype=torch.float32)
        scale_b = torch.tensor([1.0], device=device, dtype=torch.float32)
        
        # Define functions
        def run_bf16():
            return a_bf16.matmul(b_bf16.t())
            
        def run_fp8():
            res = torch._scaled_mm(a_fp8, b_fp8.t(), scale_a=scale_a, scale_b=scale_b, out_dtype=torch.bfloat16)
            if isinstance(res, tuple):
                res = res[0]
            return res

        # Run timings
        bf16_time = measure_time_ms(run_bf16)
        fp8_time = measure_time_ms(run_fp8)
        
        speedup = bf16_time / fp8_time if fp8_time > 0 else 0
        print(f"  BF16 matmul time : {bf16_time:.4f} ms")
        print(f"  FP8 scaled_mm time: {fp8_time:.4f} ms")
        print(f"  Speedup           : {speedup:.2f}x ({'Faster' if speedup > 1.0 else 'Slower'})")

# 2. Conv2d Microbenchmarks
def run_conv_diagnostics():
    print("\n--- 2. Conv2d Microbenchmarks (FP8 via im2col+scaled_mm vs BF16) ---")
    # Shapes: (Batch, C_in, C_out, H, W, kernel, padding)
    conv_shapes = [
        # (Batch, C_in, C_out, H, W, K_h, K_w, stride, padding)
        (64, 64, 64, 32, 32, 3, 3, 1, 1),   # Early layer
        (64, 512, 512, 7, 7, 3, 3, 1, 1)    # Late layer
    ]
    
    for N, C_in, C_out, H, W, kh, kw, stride, padding in conv_shapes:
        print(f"\nShape: Batch={N}, C_in={C_in}, C_out={C_out}, H={H}, W={W}, Kernel={kh}x{kw}")
        
        # BF16 inputs
        x_bf16 = torch.randn(N, C_in, H, W, dtype=torch.bfloat16, device=device)
        w_bf16 = torch.randn(C_out, C_in, kh, kw, dtype=torch.bfloat16, device=device)
        
        # Standard BF16 Conv2d
        def run_bf16_conv():
            return F.conv2d(x_bf16, w_bf16, stride=stride, padding=padding)
            
        # FP8 Conv2d via im2col (unfold) + scaled_mm
        # We need to replicate the exact steps taken in layers.py
        def run_fp8_conv():
            # Spatial unfolding
            x_unfold = F.unfold(x_bf16, kernel_size=(kh, kw), padding=padding, stride=stride)
            x_cols = x_unfold.transpose(1, 2).reshape(-1, C_in * kh * kw)
            w_mat = w_bf16.reshape(C_out, -1)
            
            # Scale & quantize
            x_cols_fp8 = x_cols.clamp(-448.0, 448.0).to(torch.float8_e4m3fn)
            w_mat_fp8 = w_mat.clamp(-448.0, 448.0).to(torch.float8_e4m3fn)
            
            scale_a = torch.tensor([1.0], device=device, dtype=torch.float32)
            scale_b = torch.tensor([1.0], device=device, dtype=torch.float32)
            
            out_mat = torch._scaled_mm(x_cols_fp8, w_mat_fp8.t(), scale_a=scale_a, scale_b=scale_b, out_dtype=torch.bfloat16)
            if isinstance(out_mat, tuple):
                out_mat = out_mat[0]
                
            out_h = (H + 2 * padding - kh) // stride + 1
            out_w = (W + 2 * padding - kw) // stride + 1
            out = out_mat.reshape(N, out_h * out_w, C_out).transpose(1, 2).reshape(N, C_out, out_h, out_w)
            return out

        # Measure memory overhead
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
        mem_start = torch.cuda.memory_allocated(device)
        _ = run_bf16_conv()
        mem_bf16 = torch.cuda.max_memory_allocated(device) - mem_start
        
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
        mem_start = torch.cuda.memory_allocated(device)
        _ = run_fp8_conv()
        mem_fp8 = torch.cuda.max_memory_allocated(device) - mem_start

        # Run timings
        bf16_time = measure_time_ms(run_bf16_conv)
        fp8_time = measure_time_ms(run_fp8_conv)
        
        speedup = bf16_time / fp8_time if fp8_time > 0 else 0
        mem_overhead_ratio = mem_fp8 / mem_bf16 if mem_bf16 > 0 else 0
        
        print(f"  BF16 Conv time  : {bf16_time:.4f} ms (Peak Memory: {mem_bf16 / (1024*1024):.2f} MB)")
        print(f"  FP8 Conv time   : {fp8_time:.4f} ms (Peak Memory: {mem_fp8 / (1024*1024):.2f} MB)")
        print(f"  Speedup         : {speedup:.2f}x ({'Faster' if speedup > 1.0 else 'Slower'})")
        print(f"  Memory Overhead : {mem_overhead_ratio:.2f}x ({mem_fp8 / (1024*1024):.2f} MB vs {mem_bf16 / (1024*1024):.2f} MB)")

if __name__ == '__main__':
    warmup_gpu()
    run_gemm_diagnostics()
    run_conv_diagnostics()
