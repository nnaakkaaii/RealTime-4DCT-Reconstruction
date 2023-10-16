import torch
from torch import nn

from .resnet3d import ResidualBlock3D


class Decoder3D(nn.Module):
    def __init__(self,
                 input_dim: int,
                 num_layers: int,
                 ) -> None:
        super().__init__()

        layers = []
        start, end = input_dim, 32 * 2 ** (num_layers - 1)

        for i in range(num_layers):
            if i == num_layers - 1:
                end = 1
            layers.append(nn.ConvTranspose3d(start, end, 3, 1, 1))
            layers.append(nn.ReLU())
            start, end = end, end // 2

        self.layers = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


class ResNetDecoder3D(nn.Module):
    def __init__(self,
                 input_dim: int,
                 num_layers: int,
                 inner_channels: int,
                 ) -> None:
        super().__init__()

        layers = []
        start, end = input_dim, 32 * 2 ** (num_layers - 1)

        for i in range(num_layers - 1):
            layers.append(ResidualBlock3D(start, end, inner_channels))
            layers.append(nn.Conv3d(end, end, 3, 1, 1))
            layers.append(nn.ReLU())
            start, end = end, end // 2
        layers.append(ResidualBlock3D(start, 1, inner_channels))

        self.layers = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)
