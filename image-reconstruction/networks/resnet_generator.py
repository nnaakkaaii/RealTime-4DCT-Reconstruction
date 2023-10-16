from typing import Optional

import torch
from torch import nn

from .simple_generator import SimpleGenerator
from .encoder2d import ResNetEncoder2D
from .encoder3d import ResNetEncoder3D
from .decoder3d import ResNetDecoder3D
from .bottleneck import Bottleneck


class ResNetGenerator(SimpleGenerator):
    def __init__(self,
                 num_layers: int = 3,
                 inner_channels: int = 256,
                 bottleneck_channels: Optional[int] = None,
                 ) -> None:
        super().__init__()

        self.encoder_x_2d_ct = ResNetEncoder2D(num_layers, inner_channels)
        self.encoder_exhale_2d_ct = ResNetEncoder2D(num_layers, inner_channels)
        self.encoder_inhale_2d_ct = ResNetEncoder2D(num_layers, inner_channels)
        self.encoder_exhale_3d_ct = ResNetEncoder3D(num_layers, inner_channels)
        self.encoder_inhale_3d_ct = ResNetEncoder3D(num_layers, inner_channels)

        dim = 160 * 2 ** (num_layers - 1)
        if bottleneck_channels is not None:
            self.deconv = nn.Sequential(
                Bottleneck(dim, bottleneck_channels),
                ResNetDecoder3D(dim, num_layers, inner_channels),
            )
        else:
            self.deconv = ResNetDecoder3D(dim, num_layers, inner_channels)


if __name__ == '__main__':
    x = (
        torch.randn(1, 1, 50, 128),
        torch.randn(1, 1, 50, 128, 128),
        torch.randn(1, 1, 50, 128, 128),
        torch.randn(1, 1, 50, 128),
        torch.randn(1, 1, 50, 128),
    )

    g = ResNetGenerator(num_layers=2,
                        inner_channels=64,
                        )
    print(g)
    print(g(*x).shape)

    g = ResNetGenerator(num_layers=2,
                        inner_channels=64,
                        bottleneck_channels=128)
    print(g)
    print(g(*x).shape)
