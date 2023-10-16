import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock2D(nn.Module):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 inner_channels: int,
                 ) -> None:
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels, inner_channels, 3, 1, 1)
        self.bn1 = nn.BatchNorm2d(inner_channels)

        self.conv2 = nn.Conv2d(inner_channels, out_channels, 3, 1, 1)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.skip = nn.Conv2d(in_channels, out_channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.skip(x)

        x = F.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))

        return F.relu(x + residual)
