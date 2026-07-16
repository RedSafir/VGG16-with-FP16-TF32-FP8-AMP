import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset
import numpy as np
import time
from typing import Dict, Any

from .model import VGG16_FP8, convert_model_to_fp8
from .train import run_training
from .utils import check_gpu_fp8_support

def run_benchmark():
    print("======================================================================")
    print("      VGG16 FP8 vs BF16 vs FP32 Baseline Training Benchmark           ")
    print("======================================================================\n")

    # 1. Hardware Check
    has_native_fp8, fp8_desc = check_gpu_fp8_support()
    print(f"[HW INFO] FP8 Support Status: {fp8_desc}")
    print(f"[HW INFO] Native/Simulated Mode: {'NATIVE' if has_native_fp8 else 'SIMULATED'}")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[HW INFO] Target Device: {device}\n")

    # 2. CIFAR-10 Datasets & Subsets (Small subset for fast benchmarking)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])

    print("[DATA] Loading CIFAR-10 dataset...")
    full_train = torchvision.datasets.CIFAR10(root='././dataset/cifar10', train=True, download=True, transform=transform)
    full_test = torchvision.datasets.CIFAR10(root='././dataset/cifar10', train=False, download=True, transform=transform)

    # Use subset for benchmark speed
    train_subset_idx = list(range(2000))
    test_subset_idx = list(range(1000))
    
    train_dataset = Subset(full_train, train_subset_idx)
    test_dataset = Subset(full_test, test_subset_idx)

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=0)

    print(f"[DATA] Subset loaded: {len(train_dataset)} train samples, {len(test_dataset)} test samples.\n")

    # 3. Parameters
    epochs = 3
    lr = 0.01
    results = {}

    # Define benchmark configs
    configs = ['fp32', 'bf16', 'fp8']

    for precision in configs:
        print(f"\n--- Running {precision.upper()} Training Run ---")
        
        # Initialize standard VGG16 model
        model = VGG16_FP8(num_classes=10, batch_norm=True).to(device)

        # Apply torchao conversion if precision is fp8
        if precision == 'fp8':
            print("[INFO] Converting FC layers to FP8 using torchao...")
            convert_model_to_fp8(model)

        # Identify which layers are Float8Linear
        from torchao.float8.float8_linear import Float8Linear
        fp8_layers = [name for name, m in model.named_modules() if isinstance(m, Float8Linear)]
        fp8_layers_str = ", ".join(fp8_layers) if fp8_layers else "None"
        print(f"[INFO] FP8 layers active: {fp8_layers_str}")

        # We disable torch.compile by default to avoid known PyTorch 2.11 Inductor backend bugs
        # (e.g. TypeError: '_InsertPoint' object is not iterable).
        # If your PyTorch version has a stable compiler, feel free to set compile_model to True.
        compile_model = False
        if compile_model:
            try:
                print("[INFO] Compiling model with torch.compile...")
                compiled_model = torch.compile(model)
            except Exception as e:
                print(f"[WARNING] torch.compile failed/unavailable ({e}). Running in eager mode.")
                compiled_model = model
        else:
            print("[INFO] Running in Eager Mode (torch.compile disabled).")
            compiled_model = model

        # Run training
        history = run_training(
            model=compiled_model,
            train_loader=train_loader,
            test_loader=test_loader,
            epochs=epochs,
            lr=lr,
            device=device,
            precision=precision
        )

        history['fp8_layers'] = fp8_layers_str
        results[precision] = history

    # 4. Summarize results
    print("\n" + "=" * 115)
    print(f"{'PRECISION':<12} | {'FINAL LOSS':<12} | {'FINAL ACC':<12} | {'AVG TIME/EPOCH':<15} | {'PEAK VRAM (EST)':<15} | {'FP8 LAYERS USED':<25}")
    print("-" * 115)
    
    for precision, history in results.items():
        avg_time = np.mean(history['epoch_times'])
        final_loss = history['train_loss'][-1]
        final_acc = history['train_acc'][-1]
        vram_display = "N/A"
        if device.type == 'cuda':
            vram_display = f"{np.max(history.get('vram_usage', [0.0])):.2f} MB" if 'vram_usage' in history and len(history['vram_usage']) > 0 else "Tracked Above"
            
        fp8_used = history.get('fp8_layers', 'None')
        print(f"{precision.upper():<12} | {final_loss:<12.4f} | {final_acc:<11.2f}% | {avg_time:<13.2f}s | {vram_display:<15} | {fp8_used:<25}")
    
    print("=" * 115)
    print("\nBenchmark completed successfully!")

if __name__ == '__main__':
    run_benchmark()
