import torch
import pandas as pd
import time
from typing import Tuple, List, Dict
from torch.utils.data import DataLoader
from tqdm import tqdm

from .model import SonarCNN
from . import config
from . import adversarial

def train_one_epoch(model: SonarCNN, dataloader: DataLoader, optimizer: torch.optim.Optimizer, device: str) -> float:
    """Trains the model for one epoch."""
    model.train()
    running_loss = 0.0
    
    for inputs, labels in dataloader:
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = model.criterion(outputs, labels)
        
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    return running_loss / len(dataloader)

def evaluate(model: SonarCNN, dataloader: DataLoader, device: str) -> Tuple[float, float, List[List[int]]]:
    """
    Evaluates the model on a dataset.
    Returns: Average Loss, Accuracy, Confusion Matrix
    """
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    num_classes = model.num_classes
    confusion_matrix = [[0] * num_classes for _ in range(num_classes)]

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            outputs = model(inputs)
            loss = model.criterion(outputs, labels)
            running_loss += loss.item()
            
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            # Update confusion matrix (Batch size 1 assumption handled, but works for batch > 1 too)
            for t, p in zip(labels.view(-1), predicted.view(-1)):
                confusion_matrix[t.long()][p.long()] += 1

    accuracy = 100 * correct / total
    avg_loss = running_loss / len(dataloader)
    
    return avg_loss, accuracy, confusion_matrix

def evaluate_adversarial(model: SonarCNN, dataloader: DataLoader, epsilon: float, device: str) -> Tuple[float, float]:
    """
    Evaluates robustness against FGSM attacks.
    """
    model.eval()
    correct = 0
    total = 0
    running_loss = 0.0

    # Needed for gradient calculation in FGSM
    # We loop manually to enable grad
    for inputs, labels in dataloader:
        inputs, labels = inputs.to(device), labels.to(device)
        inputs.requires_grad = True

        outputs = model(inputs)
        loss = model.criterion(outputs, labels)
        
        model.zero_grad()
        loss.backward()
        
        data_grad = inputs.grad.data
        perturbed_data = adversarial.fgsm_attack(inputs, epsilon, data_grad)
        
        # Re-classify the perturbed image
        output_adv = model(perturbed_data)
        running_loss += model.criterion(output_adv, labels).item()
        
        _, predicted = torch.max(output_adv.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    avg_loss = running_loss / len(dataloader)
    return avg_loss, accuracy

def training_loop(
    model: SonarCNN, 
    train_loader: DataLoader, 
    val_loader: DataLoader, 
    epochs: int,
    save_path: str,
    device: str
) -> pd.DataFrame:
    
    optimizer = torch.optim.Adam(model.parameters(), lr=config.LEARNING_RATE)
    history = []
    
    best_loss = float('inf')
    
    print(f"Starting training on {device}...")
    
    for epoch in range(epochs):
        start_time = time.time()
        
        train_loss = train_one_epoch(model, train_loader, optimizer, device)
        val_loss, val_acc, _ = evaluate(model, val_loader, device)
        
        # Check robustness during training
        adv_loss_weak, adv_acc_weak = evaluate_adversarial(model, val_loader, epsilon=0.001, device=device)
        adv_loss_strong, adv_acc_strong = evaluate_adversarial(model, val_loader, epsilon=0.01, device=device)
        
        duration = time.time() - start_time
        
        print(f"Epoch [{epoch+1}/{epochs}] | "
              f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | "
              f"Val Acc: {val_acc:.2f}% | Time: {duration:.1f}s")
        
        # Save logs
        history.append({
            'epoch': epoch + 1,
            'loss_train': train_loss,
            'loss_val': val_loss,
            'acc_val': val_acc,
            'loss_adv_weak': adv_loss_weak,
            'acc_adv_weak': adv_acc_weak,
            'loss_adv_strong': adv_loss_strong,
            'acc_adv_strong': adv_acc_strong
        })
        
        # Save best model
        if val_loss < best_loss:
            best_loss = val_loss
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': val_loss,
                'accuracy': val_acc
            }
            torch.save(checkpoint, f"{save_path}/best_model.pth")
            
    return pd.DataFrame(history)
