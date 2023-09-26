import torch
from torch import nn


class Encoder3D(nn.Module):
    def __init__(self, out_channels: int) -> None:
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv3d(1, 32, 3, 1, 1),
            nn.ReLU(),
            nn.Conv3d(32, 64, 3, 1, 1),
            nn.ReLU(),
            nn.Conv3d(64, out_channels, 3, 1, 1),
            nn.ReLU(),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)
