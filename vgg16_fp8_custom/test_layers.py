import torch
import torch.nn as nn
from .layers import FP8Linear, FP8Conv2d
from .scaling import DelayedScalingManager, FP8Config

def test_fp8_linear():
    print("Testing FP8Linear...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    scaling_manager = DelayedScalingManager()

    # Create dummy data
    x = torch.randn(4, 8, device=device, requires_grad=True)
    
    # Initialize FP8Linear
    linear = FP8Linear(
        in_features=8, 
        out_features=4, 
        bias=True, 
        scaling_manager=scaling_manager, 
        name="test_linear", 
        fallback_mode=True
    ).to(device)

    # Forward pass
    out = linear(x)
    print(f"Forward output shape: {out.shape} (Expected: [4, 4])")
    assert out.shape == (4, 4), "Output shape mismatch in FP8Linear forward"

    # Backward pass
    loss = out.sum()
    loss.backward()
    
    print(f"Input grad shape: {x.grad.shape if x.grad is not None else None} (Expected: [4, 8])")
    print(f"Weight grad shape: {linear.weight.grad.shape if linear.weight.grad is not None else None} (Expected: [4, 8])")
    
    assert x.grad is not None, "Input gradient is None"
    assert linear.weight.grad is not None, "Weight gradient is None"
    assert linear.bias.grad is not None, "Bias gradient is None"
    
    # Check if scales were tracked
    assert f"test_linear_x_fwd" in scaling_manager.current_scale, "Scaling manager missed input forward scale"
    assert f"test_linear_w_fwd" in scaling_manager.current_scale, "Scaling manager missed weight forward scale"
    assert f"test_linear_grad_bwd" in scaling_manager.current_scale, "Scaling manager missed backward gradient scale"
    print("FP8Linear tests passed successfully!\n")


def test_fp8_conv2d():
    print("Testing FP8Conv2d...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    scaling_manager = DelayedScalingManager()

    # Create dummy data
    x = torch.randn(2, 3, 16, 16, device=device, requires_grad=True)

    # Initialize FP8Conv2d
    conv = FP8Conv2d(
        in_channels=3, 
        out_channels=8, 
        kernel_size=3, 
        padding=1, 
        bias=True, 
        scaling_manager=scaling_manager, 
        name="test_conv", 
        fallback_mode=True
    ).to(device)

    # Forward pass
    out = conv(x)
    print(f"Forward output shape: {out.shape} (Expected: [2, 8, 16, 16])")
    assert out.shape == (2, 8, 16, 16), "Output shape mismatch in FP8Conv2d forward"

    # Backward pass
    loss = out.sum()
    loss.backward()

    print(f"Input grad shape: {x.grad.shape if x.grad is not None else None} (Expected: [2, 3, 16, 16])")
    print(f"Weight grad shape: {conv.weight.grad.shape if conv.weight.grad is not None else None} (Expected: [8, 3, 3, 3])")

    assert x.grad is not None, "Input gradient is None"
    assert conv.weight.grad is not None, "Weight gradient is None"
    assert conv.bias.grad is not None, "Bias gradient is None"

    # Check if scales were tracked
    assert f"test_conv_x_fwd" in scaling_manager.current_scale, "Scaling manager missed input forward scale"
    assert f"test_conv_w_fwd" in scaling_manager.current_scale, "Scaling manager missed weight forward scale"
    assert f"test_conv_grad_bwd" in scaling_manager.current_scale, "Scaling manager missed backward gradient scale"
    print("FP8Conv2d tests passed successfully!\n")

if __name__ == '__main__':
    test_fp8_linear()
    test_fp8_conv2d()
