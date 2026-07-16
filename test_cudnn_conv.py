import torch
import torch.nn.functional as F
import sys

print("==========================================================")
print("       cuDNN Frontend Convolution Execution Test          ")
print("==========================================================\n")

if not torch.cuda.is_available():
    print("[ERROR] CUDA is not available. This script must run on a GPU.")
    sys.exit(1)

device = torch.device("cuda")
print(f"Target GPU: {torch.cuda.get_device_name(0)} (Capability: {torch.cuda.get_device_capability(0)})\n")

try:
    import cudnn
    print("[OK] cuDNN Python package imported successfully.")
except ImportError:
    print("[ERROR] cudnn-frontend is not installed. Run: pip install nvidia-cudnn-frontend")
    sys.exit(1)

# Helper to inspect cudnn data types
print("\nAvailable cuDNN data types:")
for attr in dir(cudnn.data_type):
    if not attr.startswith("__"):
        print(f"  cudnn.data_type.{attr}")

# Shape setup
# Batch=1, C_in=16, C_out=16, H=8, W=8, Kernel=3x3, Padding=1, Stride=1
N, C_in, C_out, H_global, W_global, kh, kw = 1, 16, 16, 8, 8, 3, 3
padding = [1, 1]
stride = [1, 1]
dilation = [1, 1]

# 1. FP16 Graph Conv Test
def run_fp16_cudnn_conv():
    print("\n--- 1. Testing FP16 cuDNN Graph Convolution ---")
    
    # We must use channels_last (NHWC) layout as preferred by cuDNN
    x_pt = torch.randn(N, C_in, H_global, W_global, device=device, dtype=torch.float16).to(memory_format=torch.channels_last)
    w_pt = torch.randn(C_out, C_in, kh, kw, device=device, dtype=torch.float16).to(memory_format=torch.channels_last)
    y_pt = torch.empty(N, C_out, H_global, W_global, device=device, dtype=torch.float16).to(memory_format=torch.channels_last)
    
    try:
        # Create cuDNN graph
        graph = cudnn.pygraph(
            io_data_type=cudnn.data_type.HALF,
            intermediate_data_type=cudnn.data_type.FLOAT,
            compute_data_type=cudnn.data_type.FLOAT
        )
        
        # Define symbolic tensors (using distinct names to avoid name collisions)
        graph_x = graph.tensor_like(x_pt)
        graph_w = graph.tensor_like(w_pt)
        
        # Add convolution node
        graph_y = graph.conv_fprop(
            image=graph_x,
            weight=graph_w,
            padding=padding,
            stride=stride,
            dilation=dilation
        )
        graph_y.set_output(True)
        
        # Build graph plans using Heuristic Mode A
        print("Building cuDNN graph execution plans...")
        graph.build([cudnn.heur_mode.A])
        
        # Allocate workspace
        workspace_size = graph.get_workspace_size()
        print(f"Graph built successfully. Workspace size: {workspace_size / 1024:.2f} KB")
        workspace = torch.empty(workspace_size, device=device, dtype=torch.uint8)
        
        # Execute
        variant_pack = {graph_x: x_pt, graph_w: w_pt, graph_y: y_pt}
        print("Executing cuDNN graph...")
        graph.execute(variant_pack, workspace)
        torch.cuda.synchronize()
        print("[OK] cuDNN FP16 Graph executed successfully.")
        
        # Verify numerically with F.conv2d
        # Convert standard PyTorch result to channels_last for direct comparison
        ref_out = F.conv2d(x_pt.float(), w_pt.float(), padding=1, stride=1).half().to(memory_format=torch.channels_last)
        diff = torch.max(torch.abs(y_pt - ref_out)).item()
        print(f"Max numerical difference vs PyTorch F.conv2d: {diff:.6f}")
        if diff < 1e-2:
            print("[SUCCESS] FP16 cuDNN conv output matches PyTorch reference!")
        else:
            print("[WARNING] Significant numerical difference detected.")
            
    except Exception as e:
        print(f"[FAIL] FP16 cuDNN Graph Conv failed: {e}")
        import traceback
        traceback.print_exc()

# 2. FP8 Graph Conv Test (If FP8 is supported)
def run_fp8_cudnn_conv():
    print("\n--- 2. Testing FP8 cuDNN Graph Convolution (E4M3) ---")
    
    # Check if FP8 E4M3 is in cuDNN data types
    if not hasattr(cudnn.data_type, "FP8_E4M3"):
        print("[INFO] cudnn.data_type.FP8_E4M3 is not defined in this cuDNN frontend version. Skipping.")
        return
        
    try:
        # Create input tensors in float16 first
        x_pt_f16 = torch.randn(N, C_in, H_global, W_global, device=device, dtype=torch.float16).to(memory_format=torch.channels_last)
        w_pt_f16 = torch.randn(C_out, C_in, kh, kw, device=device, dtype=torch.float16).to(memory_format=torch.channels_last)
        
        # Cast to FP8 (float8_e4m3fn)
        x_pt_fp8 = x_pt_f16.clamp(-448.0, 448.0).to(torch.float8_e4m3fn)
        w_pt_fp8 = w_pt_f16.clamp(-448.0, 448.0).to(torch.float8_e4m3fn)
        
        # Output will be collected in float16
        y_pt_f16 = torch.empty(N, C_out, H_global, W_global, device=device, dtype=torch.float16).to(memory_format=torch.channels_last)
        
        # Create cuDNN graph specifying FP8 input and HALF output
        graph = cudnn.pygraph(
            io_data_type=cudnn.data_type.FP8_E4M3,
            intermediate_data_type=cudnn.data_type.FLOAT,
            compute_data_type=cudnn.data_type.FLOAT
        )
        
        # Define tensors manually to ensure correct type matching
        graph_x = graph.tensor(
            name="graph_x",
            dim=list(x_pt_fp8.shape),
            stride=list(x_pt_fp8.stride()),
            data_type=cudnn.data_type.FP8_E4M3
        )
        graph_w = graph.tensor(
            name="graph_w",
            dim=list(w_pt_fp8.shape),
            stride=list(w_pt_fp8.stride()),
            data_type=cudnn.data_type.FP8_E4M3
        )
        
        # Add convolution
        graph_y = graph.conv_fprop(
            image=graph_x,
            weight=graph_w,
            padding=padding,
            stride=stride,
            dilation=dilation
        )
        
        # Explicitly set the output data type of Y to HALF so we can collect it in y_pt_f16
        graph_y.set_data_type(cudnn.data_type.HALF)
        graph_y.set_output(True)
        
        print("Building cuDNN FP8 graph execution plans...")
        graph.build([cudnn.heur_mode.A])
        
        workspace_size = graph.get_workspace_size()
        print(f"FP8 Graph built successfully. Workspace size: {workspace_size / 1024:.2f} KB")
        workspace = torch.empty(workspace_size, device=device, dtype=torch.uint8)
        
        # Execute
        variant_pack = {graph_x: x_pt_fp8, graph_w: w_pt_fp8, graph_y: y_pt_f16}
        print("Executing cuDNN FP8 graph...")
        graph.execute(variant_pack, workspace)
        torch.cuda.synchronize()
        print("[OK] cuDNN FP8 Graph executed successfully.")
        
        # Compare with F.conv2d (simulated FP8 via dequantize or standard float16)
        ref_out = F.conv2d(x_pt_f16.float(), w_pt_f16.float(), padding=1, stride=1).half().to(memory_format=torch.channels_last)
        diff = torch.max(torch.abs(y_pt_f16 - ref_out)).item()
        print(f"Max difference vs standard float16 F.conv2d: {diff:.6f} (FP8 precision difference is expected)")
        
    except Exception as e:
        print(f"[FAIL] FP8 cuDNN Graph Conv failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    run_fp16_cudnn_conv()
    run_fp8_cudnn_conv()
