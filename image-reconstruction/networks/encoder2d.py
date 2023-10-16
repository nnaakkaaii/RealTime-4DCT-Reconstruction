import torch
from torch import nn

from .resnet2d import ResidualBlock2D


class Encoder2D(nn.Module):
    def __init__(self,
                 use_batch_norm: bool,
                 num_layers: int,
                 ) -> None:
        super().__init__()

        start, end = 1, 32
        layers = []
        for _ in range(num_layers):
            layers.append(nn.Conv2d(start, end, 3, 1, 1))
            if use_batch_norm:
                layers.append(nn.BatchNorm2d(end))
            layers.append(nn.ReLU())
            start, end = end, end * 2
        self.layers = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


class ResNetEncoder2D(nn.Module):
    def __init__(self,
                 num_layers: int,
                 inner_channels: int,
                 ) -> None:
        super().__init__()

        start, end = 1, 32
        layers = []
        for _ in range(num_layers):
            layers.append(ResidualBlock2D(start, end, inner_channels))
            start, end = end, end * 2
        self.layers = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)
