import torch
from torch import nn
import torch.nn.functional as F
from . import config

class SonarCNN(nn.Module):
    """
    1D Convolutional Neural Network for Sonar Signal Classification.
    """

    def __init__(self, num_classes: int):
        super().__init__()
        
        self.num_classes = num_classes
        
        # Configurable parameters from config.py
        self.conv1d = nn.Conv1d(in_channels=1, out_channels=config.NUM_FILTERS, kernel_size=config.KERNEL_SIZE)
        self.maxpooling1d = nn.MaxPool1d(config.POOLING_SIZE)
        self.dropout = nn.Dropout(config.DROPOUT_RATE)
        self.flatten = nn.Flatten()

        # Calculate input features for dense layer dynamically or use hardcoded calculation
        # Original: 121 * 128
        self.dense = nn.Sequential(
            nn.Linear(121 * config.NUM_FILTERS, config.NEURONS_DENSE),
            nn.ReLU(),
            nn.Linear(config.NEURONS_DENSE, num_classes)
        )

        self.criterion = nn.CrossEntropyLoss()
        # Optimizer is usually initialized outside, but we can keep a reference here if desired
        # or initialized in the training loop.

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        Shape input: (Batch, 1, Length)
        """
        x = self.conv1d(x)
        x = F.relu(x)
        x = self.maxpooling1d(x)
        x = self.dropout(x)
        x = self.flatten(x)
        logits = self.dense(x)
        return logits
