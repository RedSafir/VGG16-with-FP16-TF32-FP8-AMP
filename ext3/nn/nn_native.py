import torch
import torch.nn as nn
import torch.nn.functional as F

class NativeConv2d(nn.Conv2d):
    """
    Custom Conv2d layer optimized for FP8 Tensor Cores in PyTorch 2.1+.
    Uses GEMM-based convolution (im2col + torch._scaled_mm) for FP8 execution.
    """
    def __init__(self, *args, dtype_fwd=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.dtype_fwd = dtype_fwd  # e.g., torch.float8_e4m3fn

    def forward(self, x):
        # Fallback to standard Conv2d if not on CUDA or FP8 dtype is not configured
        if not x.is_cuda or self.dtype_fwd is None:
            return super().forward(x)

        # FP8 cuBLAS/Tensor Core Alignment check:
        # mat_a shape: (N * L, C * kh * kw) -> K = C * kh * kw must be divisible by 16.
        # mat_b shape: (C * kh * kw, out_channels) -> out_channels must be divisible by 16.
        kh, kw = self.kernel_size
        K = x.shape[1] * kh * kw  # C * kh * kw
        if K % 16 != 0 or self.out_channels % 16 != 0:
            # Fallback secara otomatis ke baseline precision jika bentuk matriks tidak sejajar (e.g., input layer dengan C=3)
            return super().forward(x)

        try:
            # 1. Get input dimensions and setup parameters
            N, C, H, W = x.shape
            ph, pw = self.padding
            sh, sw = self.stride
            dh, dw = self.dilation

            out_h = (H + 2 * ph - dh * (kh - 1) - 1) // sh + 1
            out_w = (W + 2 * pw - dw * (kw - 1) - 1) // sw + 1

            # 2. Extract patches (im2col / unfold)
            # Shape of x_unfold: (N, C * kh * kw, L) where L = out_h * out_w
            x_unfold = F.unfold(x, self.kernel_size, self.dilation, self.padding, self.stride)
            L = x_unfold.shape[2]

            # Transpose and reshape to 2D for GEMM: (N * L, C * kh * kw)
            mat_a = x_unfold.transpose(1, 2).reshape(N * L, C * kh * kw)

            # Weight reshape: (out_channels, C * kh * kw) -> Transpose to (C * kh * kw, out_channels)
            mat_b = self.weight.reshape(self.out_channels, C * kh * kw).t()

            # 3. FP8 Scaling and Quantization (e4m3fn max representable value is 448.0)
            max_val = 448.0
            
            # Find amax for activations and weights
            amax_a = torch.max(torch.abs(mat_a)).clamp(min=1e-12)
            amax_b = torch.max(torch.abs(mat_b)).clamp(min=1e-12)

            # Calculate scaling factors
            scale_a = max_val / amax_a
            scale_b = max_val / amax_b

            # Quantize matrices to FP8
            mat_a_fp8 = (mat_a * scale_a).clamp(-max_val, max_val).to(self.dtype_fwd)
            mat_b_fp8 = (mat_b * scale_b).clamp(-max_val, max_val).to(self.dtype_fwd)

            # Reciprocal scaling factors passed to dequantize in _scaled_mm
            scale_a_inv = (1.0 / scale_a).view(1)
            scale_b_inv = (1.0 / scale_b).view(1)

            # 4. Execute scaled MM
            bias_val = self.bias if self.bias is not None else None
            out_gemm, _ = torch._scaled_mm(
                mat_a_fp8,
                mat_b_fp8,
                scale_a=scale_a_inv,
                scale_b=scale_b_inv,
                bias=bias_val,
                out_dtype=x.dtype
            )

            # 5. Reshape 2D GEMM output back to 4D Conv format: (N, out_channels, out_h, out_w)
            out = out_gemm.reshape(N, L, self.out_channels).transpose(1, 2).reshape(N, self.out_channels, out_h, out_w)
            return out

        except Exception as e:
            # Safe runtime fallback if GPU architecture doesn't support scaled mm or FP8 casting
            return super().forward(x)


class NativeLinear(nn.Linear):
    """
    Custom Linear layer optimized for FP8 Tensor Cores in PyTorch 2.1+.
    Uses torch._scaled_mm for FP8 execution.
    """
    def __init__(self, *args, dtype_fwd=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.dtype_fwd = dtype_fwd  # e.g., torch.float8_e4m3fn

    def forward(self, x):
        # Fallback to standard Linear if not on CUDA or FP8 dtype is not configured
        if not x.is_cuda or self.dtype_fwd is None:
            return super().forward(x)

        # FP8 cuBLAS Alignment check: in_features and out_features must be divisible by 16
        if self.in_features % 16 != 0 or self.out_features % 16 != 0:
            return super().forward(x)

        try:
            # 1. FP8 Scaling and Quantization
            max_val = 448.0
            
            amax_x = torch.max(torch.abs(x)).clamp(min=1e-12)
            amax_w = torch.max(torch.abs(self.weight)).clamp(min=1e-12)

            scale_x = max_val / amax_x
            scale_w = max_val / amax_w

            x_fp8 = (x * scale_x).clamp(-max_val, max_val).to(self.dtype_fwd)
            w_fp8 = (self.weight * scale_w).clamp(-max_val, max_val).to(self.dtype_fwd)

            scale_x_inv = (1.0 / scale_x).view(1)
            scale_w_inv = (1.0 / scale_w).view(1)

            # 2. Execute scaled MM (weight transpose passed as mat2)
            bias_val = self.bias if self.bias is not None else None
            out, _ = torch._scaled_mm(
                x_fp8,
                w_fp8.t(),
                scale_a=scale_x_inv,
                scale_b=scale_w_inv,
                bias=bias_val,
                out_dtype=x.dtype
            )
            return out

        except Exception as e:
            return super().forward(x)
