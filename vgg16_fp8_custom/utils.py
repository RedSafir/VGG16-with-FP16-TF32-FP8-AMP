import torch
import warnings
from typing import Dict, Any, Tuple, Optional

def check_gpu_fp8_support() -> Tuple[bool, str]:
    """
    Checks if the current system has hardware capability for native FP8 training.
    FP8 is supported on SM89 (Ada Lovelace) and SM90+ (Hopper/Blackwell).
    Returns:
        bool: True if natively supported.
        str: Reason or description.
    """
    if not torch.cuda.is_available():
        return False, "CUDA is not available. GPU is required for native FP8."

    # Check PyTorch version support
    try:
        _ = torch.float8_e4m3fn
        _ = torch.float8_e5m2
    except AttributeError:
        return False, "PyTorch version does not support float8 dtypes (requires PyTorch >= 2.1)."

    # Check GPU compute capability
    try:
        device = torch.cuda.current_device()
        major, minor = torch.cuda.get_device_capability(device)
        capability = major * 10 + minor
        device_name = torch.cuda.get_device_name(device)

        if capability >= 89:
            return True, f"Natively supported on {device_name} (SM{major}.{minor} >= SM8.9)."
        else:
            return False, f"Hardware fallback: {device_name} (SM{major}.{minor} < SM8.9). FP8 will run in simulated mode."
    except Exception as e:
        return False, f"Error checking GPU capability: {e}. Defaulting to simulated mode."


class CustomDynamicLossScaler:
    """
    Manually tracks gradient overflow and handles dynamic loss scaling to keep 
    gradients within the representable range of E5M2 (max 57344.0).
    """
    def __init__(self, init_scale: float = 65536.0, scale_factor: float = 2.0, 
                 growth_interval: int = 1000, min_scale: float = 1e-4, max_scale: float = 1e20):
        self.current_scale = init_scale
        self.scale_factor = scale_factor
        self.growth_interval = growth_interval
        self.min_scale = min_scale
        self.max_scale = max_scale
        self.steps_since_overflow = 0

    def scale(self, loss: torch.Tensor) -> torch.Tensor:
        """Scales the loss by the current scale factor."""
        return loss * self.current_scale

    def step_and_update(self, optimizer: torch.optim.Optimizer) -> bool:
        """
        Unscales gradients, checks for NaN/Inf overflow, and steps the optimizer.
        Returns:
            bool: True if step was successfully taken, False if overflow occurred and step was skipped.
        """
        # Step 1: Check for gradient overflow
        has_overflow = False
        for group in optimizer.param_groups:
            for param in group['params']:
                if param.grad is not None:
                    if torch.isnan(param.grad).any() or torch.isinf(param.grad).any():
                        has_overflow = True
                        break
            if has_overflow:
                break

        # Step 2: Handle overflow
        if has_overflow:
            # Scale down
            self.current_scale = max(self.min_scale, self.current_scale / self.scale_factor)
            self.steps_since_overflow = 0
            optimizer.zero_grad()  # Clear bad gradients
            return False

        # Step 3: Unscale gradients
        inv_scale = 1.0 / self.current_scale
        for group in optimizer.param_groups:
            for param in group['params']:
                if param.grad is not None:
                    param.grad.mul_(inv_scale)

        # Step 4: Step optimizer
        optimizer.step()

        # Step 5: Scale growth check
        self.steps_since_overflow += 1
        if self.steps_since_overflow >= self.growth_interval:
            self.current_scale = min(self.max_scale, self.current_scale * self.scale_factor)
            self.steps_since_overflow = 0

        return True
