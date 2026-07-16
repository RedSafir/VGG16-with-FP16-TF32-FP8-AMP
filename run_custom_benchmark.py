import sys
import os

# Add the current directory to sys.path to ensure local imports resolve correctly
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from vgg16_fp8_custom.test_layers import test_torchao_float8_linear
from vgg16_fp8_custom.benchmark import run_benchmark

def main():
    print("==========================================================")
    print("     VGG16 FP8 Training Framework: Run Tests & Benchmarks ")
    print("==========================================================\n")

    # 1. Run Unit Tests
    print(">>> Running Unit Tests for torchao Float8 Integration...")
    try:
        test_torchao_float8_linear()
        print(">>> [OK] torchao Float8 unit tests passed successfully!\n")
    except Exception as e:
        print(f">>> [FAIL] Unit tests encountered an error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    # 2. Run Benchmark
    print(">>> Starting Training Benchmark...")
    try:
        run_benchmark()
    except Exception as e:
        print(f">>> [FAIL] Benchmark encountered an error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == '__main__':
    main()
