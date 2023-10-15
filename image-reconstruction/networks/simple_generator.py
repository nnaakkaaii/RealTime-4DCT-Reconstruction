from typing import Tuple

import torch
from torch import nn

from .encoder2d import Encoder2D
from .encoder3d import Encoder3D
from .decoder3d import Decoder3D


class SimpleGenerator(nn.Module):
    def __init__(self,
                 use_batch_size: bool = True,
                 num_layers: int = 3,
                 ) -> None:
        super().__init__()

        self.encoder_x_2d_ct = Encoder2D(use_batch_size, num_layers)
        self.encoder_exhale_2d_ct = Encoder2D(use_batch_size, num_layers)
        self.encoder_inhale_2d_ct = Encoder2D(use_batch_size, num_layers)
        self.encoder_exhale_3d_ct = Encoder3D(use_batch_size, num_layers)
        self.encoder_inhale_3d_ct = Encoder3D(use_batch_size, num_layers)
        self.deconv = Decoder3D(160 * 2 ** (num_layers - 1), num_layers)

    def encode(self,
               x_2d_ct: torch.Tensor,
               exhale_3d_ct: torch.Tensor,
               inhale_3d_ct: torch.Tensor,
               exhale_2d_ct: torch.Tensor,
               inhale_2d_ct: torch.Tensor,
               ) -> Tuple[
                    torch.Tensor,
                    torch.Tensor,
                    torch.Tensor,
                    torch.Tensor,
                    torch.Tensor,
                    ]:
        exhale_3d_ct = self.encoder_exhale_3d_ct(exhale_3d_ct)
        inhale_3d_ct = self.encoder_inhale_3d_ct(inhale_3d_ct)
        w_dim = exhale_3d_ct.shape[-1]
        x_2d_ct = self.encoder_x_2d_ct(x_2d_ct).unsqueeze(4).repeat(1, 1, 1, 1, w_dim)
        exhale_2d_ct = self.encoder_exhale_2d_ct(exhale_2d_ct).unsqueeze(4).repeat(1, 1, 1, 1, w_dim)
        inhale_2d_ct = self.encoder_inhale_2d_ct(inhale_2d_ct).unsqueeze(4).repeat(1, 1, 1, 1, w_dim)
        return x_2d_ct, exhale_3d_ct, inhale_3d_ct, exhale_2d_ct, inhale_2d_ct

    def aggregate(self,
                  x_2d_ct: torch.Tensor,
                  exhale_3d_ct: torch.Tensor,
                  inhale_3d_ct: torch.Tensor,
                  exhale_2d_ct: torch.Tensor,
                  inhale_2d_ct: torch.Tensor,
                  ) -> torch.Tensor:
        return torch.cat([x_2d_ct,
                          exhale_2d_ct,
                          inhale_2d_ct,
                          exhale_3d_ct,
                          inhale_3d_ct,
                          ],
                         dim=1)

    def decode(self,
               x: torch.Tensor) -> torch.Tensor:
        out = self.deconv(x)

        return out

    def forward(self,
                x_2d_ct: torch.Tensor,
                exhale_3d_ct: torch.Tensor,
                inhale_3d_ct: torch.Tensor,
                exhale_2d_ct: torch.Tensor,
                inhale_2d_ct: torch.Tensor,
                ) -> torch.Tensor:
        x_2d_ct, exhale_3d_ct, inhale_3d_ct, exhale_2d_ct, inhale_2d_ct = self.encode(
            x_2d_ct,
            exhale_3d_ct,
            inhale_3d_ct,
            exhale_2d_ct,
            inhale_2d_ct,
            )

        combined = self.aggregate(
            x_2d_ct,
            exhale_3d_ct,
            inhale_3d_ct,
            exhale_2d_ct,
            inhale_2d_ct,
            )

        out = self.deconv(combined)

        return out


if __name__ == '__main__':
    g = SimpleGenerator()
    print(g)
    print(g(
        torch.randn(1, 1, 50, 128),
        torch.randn(1, 1, 50, 128, 128),
        torch.randn(1, 1, 50, 128, 128),
        torch.randn(1, 1, 50, 128),
        torch.randn(1, 1, 50, 128),
    ).shape)
