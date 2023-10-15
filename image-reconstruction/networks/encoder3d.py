import torch
from torch import nn


class Encoder3D(nn.Module):
    def __init__(self,
                 use_batch_norm: bool,
                 num_layers: int,
                 ) -> None:
        super().__init__()

        start, end = 1, 32
        layers = []
        for i in range(num_layers):
            layers.append(nn.Conv3d(start, end, 3, 1, 1))
            if use_batch_norm:
                layers.append(nn.BatchNorm3d(end))
            layers.append(nn.ReLU())
            start, end = end, end * 2
        self.layers = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)
