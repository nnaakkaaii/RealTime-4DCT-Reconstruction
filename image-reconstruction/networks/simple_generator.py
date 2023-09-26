import torch
from torch import nn

from .encoder2d import Encoder2D
from .encoder3d import Encoder3D


class SimpleGenerator(nn.Module):
    def __init__(self) -> None:
        super().__init__()

        self.encoder_x_2d_ct = Encoder2D(128)
        self.encoder_exhale_2d_ct = Encoder2D(128)
        self.encoder_inhale_2d_ct = Encoder2D(128)
        self.encoder_exhale_3d_ct = Encoder3D(128)
        self.encoder_inhale_3d_ct = Encoder3D(128)

        self.deconv = nn.Sequential(
            nn.ConvTranspose3d(640, 128, 3, 1, 1),
            nn.ReLU(),
            nn.ConvTranspose3d(128, 64, 3, 1, 1),
            nn.ReLU(),
            nn.ConvTranspose3d(64, 1, 3, 1, 1),
        )

    def forward(self,
                x_2d_ct: torch.Tensor,
                exhale_3d_ct: torch.Tensor,
                inhale_3d_ct: torch.Tensor,
                exhale_2d_ct: torch.Tensor,
                inhale_2d_ct: torch.Tensor,
                ) -> torch.Tensor:
        x_2d_ct = self.encoder_x_2d_ct(x_2d_ct).unsqueeze(4).repeat(1, 1, 1, 1, 64)
        exhale_2d_ct = self.encoder_exhale_2d_ct(exhale_2d_ct).unsqueeze(4).repeat(1, 1, 1, 1, 64)
        inhale_2d_ct = self.encoder_inhale_2d_ct(inhale_2d_ct).unsqueeze(4).repeat(1, 1, 1, 1, 64)
        exhale_3d_ct = self.encoder_exhale_3d_ct(exhale_3d_ct)
        inhale_3d_ct = self.encoder_inhale_3d_ct(inhale_3d_ct)

        combined = torch.cat([x_2d_ct,
                              exhale_2d_ct,
                              inhale_2d_ct,
                              exhale_3d_ct,
                              inhale_3d_ct,
                              ],
                              dim=1)
        
        out = self.deconv(combined)

        return out
