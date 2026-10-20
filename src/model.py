"""A small 1D convolutional neural network for heartbeat classification."""

import torch
from torch import nn


class ECGConvNet(nn.Module):
    """Two convolution blocks followed by global average pooling and a classifier."""

    def __init__(self, number_of_classes: int) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=7, padding=3),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            nn.Conv1d(16, 32, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.classifier = nn.Linear(32, number_of_classes)

    def forward(self, signal: torch.Tensor) -> torch.Tensor:
        features = self.features(signal).squeeze(-1)
        return self.classifier(features)
