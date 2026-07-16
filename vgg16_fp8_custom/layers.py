import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
from .scaling import FP8Config, DelayedScalingManager

class FP8LinearFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, weight: torch.Tensor, bias: Optional[torch.Tensor], 
                scaling_manager: DelayedScalingManager, name: str, fallback_mode: bool) -> torch.Tensor:
        # Save variables for backward
        ctx.save_for_backward(x, weight, bias)
        ctx.scaling_manager = scaling_manager
        ctx.name = name
        ctx.fallback_mode = fallback_mode

        # Get/Update forward scale factors (Per-tensor for Linear)
        scale_x = scaling_manager.update_and_get_scale(f"{name}_x_fwd", x, FP8Config.FWD_DTYPE)
        scale_w = scaling_manager.update_and_get_scale(f"{name}_w_fwd", weight, FP8Config.FWD_DTYPE)

        # Scale and cast to E4M3
        x_fp8 = (x * scale_x).clamp(-FP8Config.E4M3_MAX, FP8Config.E4M3_MAX).to(FP8Config.FWD_DTYPE)
        w_fp8 = (weight * scale_w).clamp(-FP8Config.E4M3_MAX, FP8Config.E4M3_MAX).to(FP8Config.FWD_DTYPE)

        # Compute GEMM
        if fallback_mode or not x.is_cuda or not hasattr(torch, '_scaled_mm'):
            # Fallback Simulation: dequantize and perform standard GEMM
            x_dequant = x_fp8.to(x.dtype) / scale_x
            w_dequant = w_fp8.to(weight.dtype) / scale_w
            out = F.linear(x_dequant, w_dequant, bias)
        else:
            # Native FP8 GEMM via torch._scaled_mm
            try:
                inv_scale_x = 1.0 / scale_x
                inv_scale_w = 1.0 / scale_w
                
                scale_x_tensor = torch.tensor([inv_scale_x], device=x.device, dtype=torch.float32)
                scale_w_tensor = torch.tensor([inv_scale_w], device=x.device, dtype=torch.float32)
                
                # torch._scaled_mm expects right matrix to be transposed, i.e., shape (K, N)
                # weight shape is (out_features, in_features) -> transposed inside
                out = torch._scaled_mm(
                    x_fp8,
                    w_fp8.t(),
                    scale_a=scale_x_tensor,
                    scale_b=scale_w_tensor,
                    out_dtype=x.dtype
                )
                if isinstance(out, tuple):
                    out = out[0]
                if bias is not None:
                    out = out + bias
            except Exception as e:
                # Fallback on failure
                x_dequant = x_fp8.to(x.dtype) / scale_x
                w_dequant = w_fp8.to(weight.dtype) / scale_w
                out = F.linear(x_dequant, w_dequant, bias)

        return out

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> Tuple[Optional[torch.Tensor], ...]:
        x, weight, bias = ctx.saved_tensors
        scaling_manager = ctx.scaling_manager
        name = ctx.name
        fallback_mode = ctx.fallback_mode

        # Get/Update backward scale factor for grad_output (E5M2)
        scale_grad = scaling_manager.update_and_get_scale(f"{name}_grad_bwd", grad_output, FP8Config.BWD_DTYPE)

        # Scale and cast gradient to E5M2
        grad_scaled = (grad_output * scale_grad).clamp(-FP8Config.E5M2_MAX, FP8Config.E5M2_MAX).to(FP8Config.BWD_DTYPE)
        grad_dequant = grad_scaled.to(grad_output.dtype) / scale_grad

        grad_x = None
        grad_weight = None
        grad_bias = None

        # Compute gradients
        if ctx.needs_input_grad[0]:
            grad_x = grad_dequant.matmul(weight.to(grad_output.dtype))
        if ctx.needs_input_grad[1]:
            grad_weight = grad_dequant.t().matmul(x.to(grad_output.dtype))
        if ctx.needs_input_grad[2] and bias is not None:
            grad_bias = grad_dequant.sum(dim=0)

        return grad_x, grad_weight, grad_bias, None, None, None


class FP8Conv2dFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, weight: torch.Tensor, bias: Optional[torch.Tensor],
                stride: Tuple[int, int], padding: Tuple[int, int], dilation: Tuple[int, int], groups: int,
                scaling_manager: DelayedScalingManager, name: str, fallback_mode: bool) -> torch.Tensor:
        ctx.save_for_backward(x, weight, bias)
        ctx.stride = stride
        ctx.padding = padding
        ctx.dilation = dilation
        ctx.groups = groups
        ctx.scaling_manager = scaling_manager
        ctx.name = name
        ctx.fallback_mode = fallback_mode

        # Compute per-channel scale factors
        # Input channel dim is 1 (batch_size, channels, h, w) -> scale shape (1, C, 1, 1)
        scale_x = scaling_manager.update_and_get_scale(f"{name}_x_fwd", x, FP8Config.FWD_DTYPE, per_channel=True, channel_dim=1)
        # Weight out_channels dim is 0 (out_channels, in_channels, kh, kw) -> scale shape (C_out, 1, 1, 1)
        scale_w = scaling_manager.update_and_get_scale(f"{name}_w_fwd", weight, FP8Config.FWD_DTYPE, per_channel=True, channel_dim=0)

        # Scale and cast to E4M3
        x_fp8 = (x * scale_x).clamp(-FP8Config.E4M3_MAX, FP8Config.E4M3_MAX).to(FP8Config.FWD_DTYPE)
        w_fp8 = (weight * scale_w).clamp(-FP8Config.E4M3_MAX, FP8Config.E4M3_MAX).to(FP8Config.FWD_DTYPE)

        # Compute Convolution
        if fallback_mode or not x.is_cuda or not hasattr(torch, '_scaled_mm'):
            # Fallback Simulation: dequantize and perform standard Conv2d
            x_dequant = x_fp8.to(x.dtype) / scale_x
            w_dequant = w_fp8.to(weight.dtype) / scale_w
            out = F.conv2d(x_dequant, w_dequant, bias, stride, padding, dilation, groups)
        else:
            # Native FP8 Conv via im2col (unfold) + torch._scaled_mm
            try:
                batch_size, in_channels, in_h, in_w = x.shape
                out_channels, _, kernel_h, kernel_w = weight.shape
                
                # Calculate spatial output dimensions
                out_h = (in_h + 2 * padding[0] - dilation[0] * (kernel_h - 1) - 1) // stride[0] + 1
                out_w = (in_w + 2 * padding[1] - dilation[1] * (kernel_w - 1) - 1) // stride[1] + 1

                # Unfold input to 2D: (batch_size * out_h * out_w, in_channels * kh * kw)
                x_unfold = F.unfold(x_fp8.to(x.dtype) / scale_x, kernel_size=(kernel_h, kernel_w), 
                                    dilation=dilation, padding=padding, stride=stride)
                x_cols = x_unfold.transpose(1, 2).reshape(-1, in_channels * kernel_h * kernel_w)
                
                # Reshape weight to 2D: (out_channels, in_channels * kh * kw)
                w_mat = (w_fp8.to(weight.dtype) / scale_w).reshape(out_channels, -1)

                # Re-quantize unfolded matrices to FP8 for matmul
                scale_x_cols = scaling_manager.update_and_get_scale(f"{name}_x_cols_fwd", x_cols, FP8Config.FWD_DTYPE)
                scale_w_mat = scaling_manager.update_and_get_scale(f"{name}_w_mat_fwd", w_mat, FP8Config.FWD_DTYPE)

                x_cols_fp8 = (x_cols * scale_x_cols).clamp(-FP8Config.E4M3_MAX, FP8Config.E4M3_MAX).to(FP8Config.FWD_DTYPE)
                w_mat_fp8 = (w_mat * scale_w_mat).clamp(-FP8Config.E4M3_MAX, FP8Config.E4M3_MAX).to(FP8Config.FWD_DTYPE)

                inv_scale_x_cols = 1.0 / scale_x_cols
                inv_scale_w_mat = 1.0 / scale_w_mat

                scale_x_tensor = torch.tensor([inv_scale_x_cols], device=x.device, dtype=torch.float32)
                scale_w_tensor = torch.tensor([inv_scale_w_mat], device=x.device, dtype=torch.float32)

                out_mat = torch._scaled_mm(
                    x_cols_fp8,
                    w_mat_fp8.t(),
                    scale_a=scale_x_tensor,
                    scale_b=scale_w_tensor,
                    out_dtype=x.dtype
                )
                if isinstance(out_mat, tuple):
                    out_mat = out_mat[0]

                # Reshape back to 4D tensor
                out = out_mat.reshape(batch_size, out_h * out_w, out_channels).transpose(1, 2).reshape(batch_size, out_channels, out_h, out_w)
                if bias is not None:
                    out = out + bias.view(1, -1, 1, 1)
            except Exception as e:
                # Fallback on failure
                x_dequant = x_fp8.to(x.dtype) / scale_x
                w_dequant = w_fp8.to(weight.dtype) / scale_w
                out = F.conv2d(x_dequant, w_dequant, bias, stride, padding, dilation, groups)

        return out

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> Tuple[Optional[torch.Tensor], ...]:
        x, weight, bias = ctx.saved_tensors
        stride = ctx.stride
        padding = ctx.padding
        dilation = ctx.dilation
        groups = ctx.groups
        scaling_manager = ctx.scaling_manager
        name = ctx.name
        fallback_mode = ctx.fallback_mode

        # Compute per-channel backward scale factor for grad_output (dim 1: channels) -> scale shape (1, C_out, 1, 1)
        scale_grad = scaling_manager.update_and_get_scale(f"{name}_grad_bwd", grad_output, FP8Config.BWD_DTYPE, per_channel=True, channel_dim=1)

        # Scale, cast to E5M2 and dequantize
        grad_scaled = (grad_output * scale_grad).clamp(-FP8Config.E5M2_MAX, FP8Config.E5M2_MAX).to(FP8Config.BWD_DTYPE)
        grad_dequant = grad_scaled.to(grad_output.dtype) / scale_grad

        grad_x = None
        grad_weight = None
        grad_bias = None

        target_dtype = grad_output.dtype

        # Compute gradients using stable PyTorch backward functions
        if ctx.needs_input_grad[0]:
            grad_x = torch.nn.grad.conv2d_input(
                x.shape, weight.to(target_dtype), grad_dequant,
                stride=stride, padding=padding, dilation=dilation, groups=groups
            )
        if ctx.needs_input_grad[1]:
            grad_weight = torch.nn.grad.conv2d_weight(
                x.to(target_dtype), weight.shape, grad_dequant,
                stride=stride, padding=padding, dilation=dilation, groups=groups
            )
        if ctx.needs_input_grad[2] and bias is not None:
            grad_bias = grad_dequant.sum(dim=(0, 2, 3))

        return grad_x, grad_weight, grad_bias, None, None, None, None, None, None, None


class FP8Linear(nn.Module):
    """Custom Linear layer running in FP8 precision."""
    def __init__(self, in_features: int, out_features: int, bias: bool = True, 
                 scaling_manager: Optional[DelayedScalingManager] = None, name: str = "", fallback_mode: bool = True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.name = name
        self.fallback_mode = fallback_mode
        self.scaling_manager = scaling_manager if scaling_manager is not None else DelayedScalingManager()

        # Keep master weights in FP32
        self.weight = nn.Parameter(torch.empty(out_features, in_features, dtype=torch.float32))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features, dtype=torch.float32))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5) if hasattr(self, 'reset_parameters') else 1.0)
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return FP8LinearFunction.apply(x, self.weight, self.bias, self.scaling_manager, self.name, self.fallback_mode)


class FP8Conv2d(nn.Module):
    """Custom Conv2d layer running in FP8 precision with per-channel scaling."""
    def __init__(self, in_channels: int, out_channels: int, kernel_size: Union[int, Tuple[int, int]],
                 stride: Union[int, Tuple[int, int]] = 1, padding: Union[int, Tuple[int, int]] = 0,
                 dilation: Union[int, Tuple[int, int]] = 1, groups: int = 1, bias: bool = True,
                 scaling_manager: Optional[DelayedScalingManager] = None, name: str = "", fallback_mode: bool = True):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        # Convert kernel_size, stride, padding, dilation to tuples
        self.kernel_size = kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size)
        self.stride = stride if isinstance(stride, tuple) else (stride, stride)
        self.padding = padding if isinstance(padding, tuple) else (padding, padding)
        self.dilation = dilation if isinstance(dilation, tuple) else (dilation, dilation)
        self.groups = groups
        self.name = name
        self.fallback_mode = fallback_mode
        self.scaling_manager = scaling_manager if scaling_manager is not None else DelayedScalingManager()

        # Keep master weights in FP32
        self.weight = nn.Parameter(torch.empty(out_channels, in_channels // groups, *self.kernel_size, dtype=torch.float32))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_channels, dtype=torch.float32))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return FP8Conv2dFunction.apply(
            x, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups,
            self.scaling_manager, self.name, self.fallback_mode
        )

import math
