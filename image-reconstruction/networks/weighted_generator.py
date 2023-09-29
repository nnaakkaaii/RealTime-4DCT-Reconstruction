import torch
from torch import nn
from torch.nn import functional as F

from .simple_generator import SimpleGenerator


class WeightedGenerator(SimpleGenerator):
    def __init__(self) -> None:
        super().__init__()

        self.weights = nn.Parameter(torch.ones(5))

        self.deconv = nn.Sequential(
            nn.ConvTranspose3d(128, 64, 3, 1, 1),
            nn.ReLU(),
            nn.ConvTranspose3d(64, 1, 3, 1, 1),
        )

    def aggregate(self,
                  x_2d_ct: torch.Tensor,
                  exhale_3d_ct: torch.Tensor,
                  inhale_3d_ct: torch.Tensor,
                  exhale_2d_ct: torch.Tensor,
                  inhale_2d_ct: torch.Tensor,
                  ) -> torch.Tensor:
        normalized_weights = F.softmax(self.weights, dim=0)

        x_2d_ct *= normalized_weights[0]
        exhale_2d_ct *= normalized_weights[1]
        inhale_2d_ct *= normalized_weights[2]
        exhale_3d_ct *= normalized_weights[3]
        inhale_3d_ct *= normalized_weights[4]

        return x_2d_ct + exhale_2d_ct + inhale_2d_ct + exhale_3d_ct + inhale_3d_ct
