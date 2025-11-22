import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from typing import List

def save_training_logs(df: pd.DataFrame, filepath: str):
    """Saves training history to CSV."""
    df.to_csv(filepath, index=False)
    print(f"Logs saved to {filepath}")

def plot_confusion_matrix(matrix: List[List[int]], classes: List[str], save_path: str = None):
    """Plots and saves confusion matrix."""
    plt.figure(figsize=(10, 8))
    sns.heatmap(matrix, annot=True, fmt='d', cmap='cividis', 
                xticklabels=classes, yticklabels=classes)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    
    if save_path:
        plt.savefig(save_path)
        print(f"Confusion matrix saved to {save_path}")
    
    plt.close()

def plot_losses(df: pd.DataFrame, save_path: str = None):
    """Plots training and validation loss."""
    plt.figure(figsize=(10, 6))
    plt.plot(df['epoch'], df['loss_train'], label='Train Loss')
    plt.plot(df['epoch'], df['loss_val'], label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training Convergence')
    
    if save_path:
        plt.savefig(save_path)
    plt.close()
