import torch
import torch.nn as nn

class APAManager:
    """
    Adaptive Precision Layering Manager.
    Handles dynamic precision adjustments and on-the-fly overflow detection.
    """
    def __init__(self, model, initial_precision="fp8", overflow_threshold=3):
        self.model = model
        self.precision = initial_precision
        self.overflow_threshold = overflow_threshold
        self.overflow_count = 0

    def check_overflow(self, loss):
        """
        Checks for NaN or Inf in the loss and parameter gradients on-the-fly.
        """
        # 1. Check loss stability
        if not torch.isfinite(loss):
            return True

        # 2. Check gradients stability
        for p in self.model.parameters():
            if p.grad is not None:
                if torch.isinf(p.grad).any() or torch.isnan(p.grad).any():
                    return True
        return False

    def step(self, loss):
        """
        Performs a check step. If consecutive overflows exceed threshold,
        demotes model precision dynamically.
        """
        if self.check_overflow(loss):
            self.overflow_count += 1
            print(f"[APA Manager] Overflow detected! Count: {self.overflow_count}/{self.overflow_threshold}")
            if self.overflow_count >= self.overflow_threshold:
                self.demote_precision()
                self.overflow_count = 0
                return True  # Precision was adjusted
        else:
            # Decay overflow count slowly to reward stability
            if self.overflow_count > 0:
                self.overflow_count -= 1
        return False

    def demote_precision(self):
        """
        Demotes the precision level to stabilize training.
        """
        if self.precision == "fp8":
            self.precision = "fp16"
            print("[APA Manager] Demoting model precision from FP8 to FP16 for numerical stability.")
            self._apply_precision("fp16")
        elif self.precision == "fp16":
            self.precision = "fp32"
            print("[APA Manager] Demoting model precision from FP16 to FP32 for numerical stability.")
            self._apply_precision("fp32")

    def _apply_precision(self, precision):
        """
        Recursively alters layers configuration dynamically.
        """
        from ext3.nn.nn_native import NativeConv2d, NativeLinear
        for m in self.model.modules():
            if isinstance(m, (NativeConv2d, NativeLinear)):
                if precision == "fp8":
                    m.dtype_fwd = torch.float8_e4m3fn
                else:
                    m.dtype_fwd = None  # Reverts forward pass back to baseline model precision

        if precision == "fp16":
            self.model.half()
        elif precision == "fp32":
            self.model.float()
