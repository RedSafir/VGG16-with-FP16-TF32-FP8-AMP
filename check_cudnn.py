import torch
import sys

print("==========================================================")
print("             cuDNN Frontend Compatibility Check           ")
print("==========================================================\n")

print(f"PyTorch Version: {torch.__version__}")
print(f"CUDA Available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU Name: {torch.cuda.get_device_name(0)}")
    print(f"Compute Capability: {torch.cuda.get_device_capability(0)}")
    print(f"PyTorch cuDNN Version: {torch.backends.cudnn.version()}")
    print(f"PyTorch cuDNN Enabled: {torch.backends.cudnn.enabled}")
print("")

try:
    import cudnn
    print("[OK] 'cudnn' python package (nvidia-cudnn-frontend) is INSTALLED.")
    try:
        version = cudnn.backend_version()
        print(f"cuDNN Backend Version: {version}")
    except Exception as e:
        print(f"[WARNING] Could not get cudnn backend version: {e}")
        
    # Quick capability probe
    try:
        print("\nAttempting to create cuDNN graph handle...")
        graph = cudnn.pygraph()
        print("[OK] cuDNN Graph API is available.")
    except Exception as e:
        print(f"[FAIL] cuDNN Graph API initialization failed: {e}")
        
except ImportError:
    print("[INFO] 'cudnn' python package (nvidia-cudnn-frontend) is NOT installed.")
    print("To install it, run: pip install nvidia-cudnn-frontend")

print("\n==========================================================")
