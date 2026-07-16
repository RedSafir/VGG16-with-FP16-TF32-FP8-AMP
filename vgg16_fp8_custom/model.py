import torch
import torch.nn as nn
from typing import Dict, Any, List, Optional
from .layers import FP8Conv2d, FP8Linear
from .scaling import DelayedScalingManager

class VGG16_FP8(nn.Module):
    """
    VGG16 Model with support for standard (FP32/BF16) or custom FP8 layers.
    - First Conv layer and last classifier linear layer are typically kept in 
      higher precision (FP32/BF16) to preserve input features and logit distribution.
    - Master weights are stored in FP32 and cast on-the-fly.
    """
    def __init__(self, num_classes: int = 10, use_fp8: bool = True, batch_norm: bool = True,
                 scaling_manager: Optional[DelayedScalingManager] = None, fallback_mode: bool = True):
        super().__init__()
        self.num_classes = num_classes
        self.use_fp8 = use_fp8
        self.batch_norm = batch_norm
        self.fallback_mode = fallback_mode
        
        # Instantiate or share a global scaling manager
        self.scaling_manager = scaling_manager if scaling_manager is not None else DelayedScalingManager()

        # Config: layer size configurations for VGG16
        self.cfg = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M']
        
        self.features = self._make_layers()
        
        # Classifier: input is 512 (due to CIFAR-10 32x32 size being reduced by 5 maxpools to 1x1)
        if use_fp8:
            self.classifier = nn.Sequential(
                FP8Linear(512, 4096, bias=True, scaling_manager=self.scaling_manager, name="fc1", fallback_mode=fallback_mode),
                nn.ReLU(True),
                nn.Dropout(),
                FP8Linear(4096, 4096, bias=True, scaling_manager=self.scaling_manager, name="fc2", fallback_mode=fallback_mode),
                nn.ReLU(True),
                nn.Dropout(),
                # Keep output layer in FP32 for stability
                nn.Linear(4096, num_classes)
            )
        else:
            self.classifier = nn.Sequential(
                nn.Linear(512, 4096),
                nn.ReLU(True),
                nn.Dropout(),
                nn.Linear(4096, 4096),
                nn.ReLU(True),
                nn.Dropout(),
                nn.Linear(4096, num_classes)
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.features(x)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out

    def _make_layers(self) -> nn.Sequential:
        layers = []
        in_channels = 3
        conv_idx = 0
        
        for x in self.cfg:
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                conv_idx += 1
                name = f"conv_{conv_idx}"
                
                # Check if this layer should run in FP8
                # First layer is usually kept in FP32 to avoid quantizing low-channel inputs
                if self.use_fp8 and conv_idx > 1:
                    conv = FP8Conv2d(
                        in_channels, x, kernel_size=3, padding=1, 
                        scaling_manager=self.scaling_manager, name=name, fallback_mode=self.fallback_mode
                    )
                else:
                    conv = nn.Conv2d(in_channels, x, kernel_size=3, padding=1)

                layers += [conv]
                if self.batch_norm:
                    # BatchNorm layers are always kept in FP32/higher precision
                    layers += [nn.BatchNorm2d(x)]
                layers += [nn.ReLU(True)]
                in_channels = x
                
        return nn.Sequential(*layers)
