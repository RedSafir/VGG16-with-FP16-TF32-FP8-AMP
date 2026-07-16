import torch
import torch.nn as nn
import torch.optim as optim
import time
from typing import Dict, Any, Tuple
from .model import VGG16_FP8
from .utils import CustomDynamicLossScaler

def train_epoch(model: nn.Module, loader: torch.utils.data.DataLoader, criterion: nn.Module, 
                optimizer: optim.Optimizer, device: torch.device, precision: str, 
                scaler: Optional[CustomDynamicLossScaler] = None) -> Tuple[float, float]:
    """Trains the model for one epoch and returns (average_loss, accuracy)."""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for inputs, targets in loader:
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()

        if precision == 'fp8':
            # FP8 training: forward pass is executed, custom layers handle E4M3 scaling.
            # Backward pass handles E5M2. We use our CustomDynamicLossScaler.
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            if scaler is not None:
                scaled_loss = scaler.scale(loss)
                scaled_loss.backward()
                scaler.step_and_update(optimizer)
            else:
                loss.backward()
                optimizer.step()
                
        elif precision == 'bf16':
            # BF16 training via standard PyTorch autocast
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
                outputs = model(inputs)
                loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
        else:  # fp32
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

    avg_loss = running_loss / total
    accuracy = 100.0 * correct / total
    return avg_loss, accuracy


def evaluate(model: nn.Module, loader: torch.utils.data.DataLoader, criterion: nn.Module, 
             device: torch.device) -> Tuple[float, float]:
    """Evaluates the model on test dataset and returns (average_loss, accuracy)."""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, targets in loader:
            inputs, targets = inputs.to(device), targets.to(device)
            # Evaluation is run in standard FP32 for numerical stability
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            running_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    avg_loss = running_loss / total
    accuracy = 100.0 * correct / total
    return avg_loss, accuracy


def run_training(model: nn.Module, train_loader: torch.utils.data.DataLoader, 
                 test_loader: torch.utils.data.DataLoader, epochs: int, lr: float, 
                 device: torch.device, precision: str, fallback_mode: bool) -> Dict[str, Any]:
    """Runs a complete training run and logs history (losses, accuracies, durations)."""
    criterion = nn.CrossEntropyLoss()
    
    # Define optimizer
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
    
    scaler = CustomDynamicLossScaler() if precision == 'fp8' else None
    
    history = {
        'train_loss': [],
        'train_acc': [],
        'test_loss': [],
        'test_acc': [],
        'epoch_times': []
    }

    for epoch in range(1, epochs + 1):
        if device.type == 'cuda':
            torch.cuda.reset_peak_memory_stats(device)

        start_time = time.time()
        
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device, precision, scaler)
        test_acc, test_loss = evaluate(model, test_loader, criterion, device)
        
        epoch_time = time.time() - start_time
        
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['test_loss'].append(test_loss)
        history['test_acc'].append(test_acc)
        history['epoch_times'].append(epoch_time)
        
        vram_str = ""
        if device.type == 'cuda':
            peak_memory = torch.cuda.max_memory_allocated(device) / (1024 * 1024)
            vram_str = f" - Peak VRAM: {peak_memory:.2f} MB"
            
        print(f"Epoch [{epoch}/{epochs}] - Time: {epoch_time:.2f}s{vram_str} - "
              f"Train Loss: {train_loss:.4f} - Train Acc: {train_acc:.2f}% - "
              f"Test Loss: {test_loss:.4f} - Test Acc: {test_acc:.2f}%")

    return history

from typing import Optional
