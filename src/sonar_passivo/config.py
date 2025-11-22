import os
from pathlib import Path

# Project Root
PROJECT_ROOT = Path(__file__).parent.parent.parent

# Data Paths
DATA_DIR = PROJECT_ROOT / "data"
DATASET_DIR = DATA_DIR / "Datasets" / "DadosSonar"
NETWORKS_DIR = DATA_DIR / "Networks"
LOGS_DIR = DATA_DIR / "Training_data"

# Model Hyperparameters
NUM_EPOCHS = 75
BATCH_SIZE = 1
LEARNING_RATE = 1e-3
NUM_FILTERS = 128
KERNEL_SIZE = 71
POOLING_SIZE = 4
DROPOUT_RATE = 0.5
NEURONS_DENSE = 75

# Hardware
import torch
DEVICE = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

def ensure_dirs():
    """Ensures that necessary directories exist."""
    NETWORKS_DIR.mkdir(parents=True, exist_ok=True)
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    (NETWORKS_DIR / "robust").mkdir(exist_ok=True)
