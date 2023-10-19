import torch
from torch import nn
from torch.nn import functional as F

from .simple_generator import SimpleGenerator
from .decoder import Decoder


class WeightedGenerator(SimpleGenerator):
    def __init__(self,
                 use_batch_norm: bool = True,
                 num_layers: int = 3,
                 ) -> None:
        super().__init__(use_batch_norm, num_layers)

        self.weights = nn.Parameter(torch.ones(5))

        self.deconv = Decoder(128, 2, use_3d=True)

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
