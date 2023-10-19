from typing import Optional

import torch
from torch import nn

from .simple_generator import SimpleGenerator
from .encoder import ResNetEncoder
from .decoder import ResNetDecoder
from .bottleneck import Bottleneck


class ResNetGenerator(SimpleGenerator):
    def __init__(self,
                 num_layers: int = 3,
                 num_inner_layers: int = 3,
                 bottleneck_channels: Optional[int] = None,
                 ) -> None:
        super().__init__()

        self.encoder_x_2d_ct = ResNetEncoder(num_layers, num_inner_layers, use_3d=False)
        self.encoder_exhale_2d_ct = ResNetEncoder(num_layers, num_inner_layers, use_3d=False)
        self.encoder_inhale_2d_ct = ResNetEncoder(num_layers, num_inner_layers, use_3d=False)
        self.encoder_exhale_3d_ct = ResNetEncoder(num_layers, num_inner_layers, use_3d=True)
        self.encoder_inhale_3d_ct = ResNetEncoder(num_layers, num_inner_layers, use_3d=True)

        dim = 160 * 2 ** (num_layers - 1)
        if bottleneck_channels is not None:
            self.deconv = nn.Sequential(
                Bottleneck(dim, bottleneck_channels),
                ResNetDecoder(dim, num_layers, num_inner_layers, use_3d=True),
            )
        else:
            self.deconv = ResNetDecoder(dim, num_layers, num_inner_layers, use_3d=True)


if __name__ == '__main__':
    x = (
        torch.randn(1, 1, 64, 64),
        torch.randn(1, 1, 64, 64, 64),
        torch.randn(1, 1, 64, 64, 64),
        torch.randn(1, 1, 64, 64),
        torch.randn(1, 1, 64, 64),
    )

    g = ResNetGenerator(num_layers=2,
                        num_inner_layers=3,
                        )
    print(g)
    print(g(*x).shape)

    g = ResNetGenerator(num_layers=2,
                        num_inner_layers=3,
                        bottleneck_channels=128)
    print(g)
    print(g(*x).shape)
