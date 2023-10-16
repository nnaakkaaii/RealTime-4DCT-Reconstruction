import torch
from torch import nn
from torch.nn import functional as F


class ResidualBlock3D(nn.Module):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 inner_channels: int,
                 ) -> None:
        super().__init__()

        self.conv1 = nn.Conv3d(in_channels, inner_channels, 3, 1, 1)
        self.bn1 = nn.BatchNorm3d(inner_channels)

        self.conv2 = nn.Conv3d(inner_channels, out_channels, 3, 1, 1)
        self.bn2 = nn.BatchNorm3d(out_channels)

        self.skip = nn.Conv3d(in_channels, out_channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.skip(x)

        x = F.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))

        return F.relu(x + residual)
