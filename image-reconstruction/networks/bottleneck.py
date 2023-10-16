import torch
from torch import nn


class Bottleneck(nn.Module):
    def __init__(self,
                 in_channels: int,
                 bottleneck_channels: int,
                 ) -> None:
        super().__init__()
        self.downsample = nn.Sequential(
            nn.Conv3d(in_channels, bottleneck_channels, 4, 2, 1),
            nn.ReLU(),
            nn.BatchNorm3d(bottleneck_channels)
        )

        self.upsample = nn.Sequential(
            nn.ConvTranspose3d(bottleneck_channels, in_channels, 4, 2, 1),
            nn.ReLU(),
            nn.BatchNorm3d(in_channels)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.downsample(x)
        x = self.upsample(x)
        return x
