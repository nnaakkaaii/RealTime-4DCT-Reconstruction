import torch
from torch import nn

from .resnet import ResidualIdentity, ResidualUpSample
from .utils import CONVT


class Decoder(nn.Module):
    def __init__(self,
                 input_dim: int,
                 num_layers: int,
                 use_3d: bool,
                 ) -> None:
        super().__init__()

        convt = CONVT[use_3d]

        layers = []
        start, end = input_dim, 32 * 2 ** (num_layers - 1)

        for i in range(num_layers):
            if i == num_layers - 1:
                end = 1
            layers.append(convt(start, end, 3, 1, 1))
            layers.append(nn.ReLU())
            start, end = end, end // 2

        self.layers = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


class ResNetDecoder(nn.Module):
    def __init__(self,
                 input_dim: int,
                 num_layers: int,
                 num_inner_layers: int,
                 use_3d: bool,
                 ) -> None:
        super().__init__()

        layers = []
        start, end = input_dim, 32 * 2 ** (num_layers - 1)

        for i in range(num_layers):
            if i == num_layers - 1:
                end = 1
            for _ in range(num_inner_layers - 1):
                layers.append(ResidualIdentity(start, use_3d))
            layers.append(ResidualUpSample(start, end, use_3d))
            start, end = end, end // 2

        self.layers = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)
