import torch
from collections import deque
from typing import Dict, Tuple, Union, Optional

class FP8Config:
    # Native float8 types
    FWD_DTYPE = torch.float8_e4m3fn   # For weights & activations
    BWD_DTYPE = torch.float8_e5m2     # For gradients

    # Maximum representable values (standard limits)
    E4M3_MAX = 448.0
    E5M2_MAX = 57344.0

    DEFAULT_HISTORY_LEN = 16
    DEFAULT_MARGIN = 0.0  # Log2 safety margin
    SCALE_MIN = 1e-12
    SCALE_MAX = 1e12

class DelayedScalingManager:
    """
    Manages scaling factors for tensors using Delayed Scaling algorithm.
    Stores the maximum absolute values (amax) in a history window
    and computes the scale factor as:
        scale = max_representable_value / max(amax_history)
    """
    def __init__(self, history_len: int = FP8Config.DEFAULT_HISTORY_LEN, margin: float = FP8Config.DEFAULT_MARGIN):
        self.history_len = history_len
        self.margin = margin
        self.history: Dict[str, deque] = {}  # key -> deque of float or torch.Tensor
        self.current_scale: Dict[str, torch.Tensor] = {}

    def _get_max_representable(self, dtype: torch.dtype) -> float:
        if dtype == FP8Config.FWD_DTYPE:
            return FP8Config.E4M3_MAX
        elif dtype == FP8Config.BWD_DTYPE:
            return FP8Config.E5M2_MAX
        else:
            raise ValueError(f"Unsupported FP8 dtype: {dtype}")

    def update_and_get_scale(self, name: str, tensor: torch.Tensor, dtype: torch.dtype, per_channel: bool = False, channel_dim: int = 0) -> torch.Tensor:
        """
        Updates the amax history with the current tensor and computes/returns the scale factor.
        For per-channel scaling, computes scale factors along the specified dim.
        """
        device = tensor.device
        dtype_max = self._get_max_representable(dtype)

        # Compute current amax
        with torch.no_grad():
            if per_channel:
                # We need to keep dims for proper broadcasting later
                # For conv weights (out_channels, in_channels, kh, kw), channel_dim=0, reduce dims (1, 2, 3)
                # For activations (batch_size, channels, h, w), channel_dim=1, reduce dims (0, 2, 3)
                reduce_dims = list(range(tensor.dim()))
                reduce_dims.remove(channel_dim)
                
                # Compute absolute max along channel dimension
                temp = tensor.abs()
                for d in sorted(reduce_dims, reverse=True):
                    temp = temp.max(dim=d, keepdim=True).values
                
                amax = temp.clamp(min=1e-12)
            else:
                amax = tensor.abs().max().clamp(min=1e-12)

        # Initialize history if not present
        if name not in self.history:
            # Seed history with dtype_max so initial scale is 1.0
            initial_val = torch.full_like(amax, dtype_max) if isinstance(amax, torch.Tensor) else dtype_max
            self.history[name] = deque([initial_val] * self.history_len, maxlen=self.history_len)

        # Append current amax to history
        self.history[name].append(amax)

        # Compute maximum amax in history window
        history_tensors = list(self.history[name])
        max_amax = torch.stack(history_tensors).max(dim=0).values if isinstance(amax, torch.Tensor) else max(history_tensors)

        # Compute new scale factor
        scale = dtype_max / (max_amax * (2.0 ** self.margin))
        
        # Clamp scale factor
        if isinstance(scale, torch.Tensor):
            scale = scale.clamp(FP8Config.SCALE_MIN, FP8Config.SCALE_MAX)
        else:
            scale = min(FP8Config.SCALE_MAX, max(FP8Config.SCALE_MIN, scale))
            scale = torch.tensor(scale, device=device, dtype=tensor.dtype)

        self.current_scale[name] = scale
        return scale

    def get_scale(self, name: str, default_device: torch.device = torch.device('cpu')) -> torch.Tensor:
        """Returns the current scale factor. If it doesn't exist, returns a default scale of 1.0."""
        if name in self.current_scale:
            return self.current_scale[name]
        return torch.tensor(1.0, device=default_device)
