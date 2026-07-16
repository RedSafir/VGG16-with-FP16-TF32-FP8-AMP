import torch
import torch.nn as nn
from vgg16_fp8_custom.model import convert_model_to_fp8

def test_torchao_float8_linear():
    print("Testing torchao Float8Linear integration...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Create dummy model with linear layers (some divisible by 16, some not)
    class DummyModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1 = nn.Linear(32, 64)   # Divisible by 16 -> should convert
            self.fc2 = nn.Linear(64, 10)   # 10 is not divisible by 16 -> should skip

        def forward(self, x):
            return self.fc2(torch.relu(self.fc1(x)))

    model = DummyModel().to(device)
    
    # Check initial types
    assert isinstance(model.fc1, nn.Linear)
    assert isinstance(model.fc2, nn.Linear)

    # Convert model to FP8 via torchao helper
    convert_model_to_fp8(model)

    # Verify types after conversion
    from torchao.float8 import Float8Linear
    
    print(f"fc1 class after conversion: {model.fc1.__class__.__name__} (Expected: Float8Linear)")
    print(f"fc2 class after conversion: {model.fc2.__class__.__name__} (Expected: Linear)")

    assert isinstance(model.fc1, Float8Linear), "fc1 was not converted to Float8Linear"
    assert isinstance(model.fc2, nn.Linear) and not isinstance(model.fc2, Float8Linear), "fc2 should have been skipped"

    # Forward & Backward Pass Verification
    x = torch.randn(4, 32, device=device)
    
    # Use bfloat16 autocast as recommended for Float8Linear training
    with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
        out = model(x)
        loss = out.sum()
        
    loss.backward()

    print(f"fc1 weight grad calculated: {model.fc1.weight.grad is not None}")
    assert model.fc1.weight.grad is not None, "Gradient of fc1 weight is None after backward pass"
    
    print("torchao Float8Linear integration tests passed successfully!\n")

if __name__ == '__main__':
    test_torchao_float8_linear()
