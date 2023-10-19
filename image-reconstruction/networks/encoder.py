import torch
from torch import nn

from .resnet import ResidualIdentity, ResidualDownSample
from .utils import CONV, BN


class Encoder(nn.Module):
    def __init__(self,
                 use_batch_norm: bool,
                 num_layers: int,
                 use_3d: bool,
                 ) -> None:
        super().__init__()

        conv = CONV[use_3d]
        bn = BN[use_3d]

        start, end = 1, 32
        layers = []
        for i in range(num_layers):
            layers.append(conv(start, end, 3, 1, 1))
            if use_batch_norm:
                layers.append(bn(end))
            layers.append(nn.ReLU())
            start, end = end, end * 2
        self.layers = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


class ResNetEncoder(nn.Module):
    def __init__(self,
                 num_layers: int,
                 num_inner_layers: int,
                 use_3d: bool,
                 ) -> None:
        super().__init__()

        start, end = 1, 32
        layers = []
        for _ in range(num_layers):
            for _ in range(num_inner_layers - 1):
                layers.append(ResidualIdentity(start, use_3d))
            layers.append(ResidualDownSample(start, end, use_3d))
            start, end = end, end * 2
        self.layers = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)
