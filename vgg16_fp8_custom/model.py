import torch
import torch.nn as nn
from typing import Optional

class VGG16_FP8(nn.Module):
    """
    Standard VGG16 model for CIFAR-10.
    Built entirely using standard PyTorch nn.Conv2d and nn.Linear modules.
    Can be dynamically converted to float8 precision using torchao.
    """
    def __init__(self, num_classes: int = 10, batch_norm: bool = True):
        super().__init__()
        self.num_classes = num_classes
        self.batch_norm = batch_norm
        
        # Config: VGG16 layer structure
        self.cfg = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M']
        self.features = self._make_layers()
        
        # Standard Linear classifier
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
        for x in self.cfg:
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1)]
                if self.batch_norm:
                    layers += [nn.BatchNorm2d(x)]
                layers += [nn.ReLU(True)]
                in_channels = x
        return nn.Sequential(*layers)


# Helper function to convert VGG16 model to use torchao float8 training
def convert_model_to_fp8(model: nn.Module) -> nn.Module:
    """
    Converts Linear layers of the model to torchao Float8Linear if dimensions are divisible by 16.
    """
    from torchao.float8 import convert_to_float8_training
    
    def module_filter_fn(module: nn.Module, fqn: str) -> bool:
        # Only convert Linear modules whose input and output dimensions are divisible by 16
        if isinstance(module, nn.Linear):
            if module.in_features % 16 == 0 and module.out_features % 16 == 0:
                return True
        return False

    # Apply torchao conversion inplace
    convert_to_float8_training(model, module_filter_fn=module_filter_fn)
    return model
