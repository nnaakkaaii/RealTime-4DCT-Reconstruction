import torch
from torch import nn


class Decoder3D(nn.Module):
    def __init__(self,
                 input_dim: int,
                 num_layers: int = 3,
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
