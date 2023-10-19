import torch
import torch.nn as nn
import torch.nn.functional as F

from .utils import CONV, BN, MODE


class ResidualIdentity(nn.Module):
    def __init__(self,
                 in_channels: int,
                 use_3d: bool,
                 ) -> None:
        super().__init__()

        conv = CONV[use_3d]
        bn = BN[use_3d]

        self.conv1 = conv(in_channels, in_channels, 3, 1, 1)
        self.bn1 = bn(in_channels)

        self.conv2 = conv(in_channels, in_channels, 3, 1, 1)
        self.bn2 = bn(in_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x

        x = F.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))

        return F.relu(x + residual)


class ResidualDownSample(nn.Module):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 use_3d: bool,
                 ) -> None:
        super().__init__()

        conv = CONV[use_3d]
        bn = BN[use_3d]

        self.conv1 = conv(in_channels, out_channels, 4, 2, 1)
        self.bn1 = bn(out_channels)

        self.conv2 = conv(out_channels, out_channels, 3, 1, 1)
        self.bn2 = bn(out_channels)

        self.skip = nn.Sequential(
            conv(in_channels, out_channels, 1, 2, 0),  # Adjusted stride and padding
            bn(out_channels)  # Added batch normalization
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.skip(x)

        x = F.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))

        return F.relu(x + residual)


class ResidualUpSample(nn.Module):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 use_3d: bool,
                 ) -> None:
        super().__init__()

        conv = CONV[use_3d]
        bn = BN[use_3d]
        mode = MODE[use_3d]

        self.conv1 = conv(in_channels, out_channels, 3, 1, 1)
        self.bn1 = bn(out_channels)

        self.conv2 = conv(out_channels, out_channels, 3, 1, 1)
        self.bn2 = bn(out_channels)

        self.skip = nn.Sequential(
            nn.Upsample(scale_factor=2, mode=mode, align_corners=True),
            conv(in_channels, out_channels, 1),
        )

        self.upsample = nn.Upsample(scale_factor=2, mode=mode, align_corners=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.skip(x)

        x = self.upsample(x)
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))

        return F.relu(x + residual)
