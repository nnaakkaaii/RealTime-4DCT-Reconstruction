import torch
from torch import nn


class SimpleDiscriminator(nn.Module):
    def __init__(self) -> None:
        super().__init__()

        self.net = nn.Sequential(
            nn.Conv3d(1, 64, 4, 2, 1),  # (1, 50, 512, 512) -> (64, 25, 256, 256)
            # nn.Conv3d(1, 64, 4, 2, 1),  # (1, 50, 64, 64) -> (64, 25, 32, 32)
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv3d(64, 128, 4, 2, 1),  # (64, 25, 256, 256) -> (128, 12, 128, 128)
            # nn.Conv3d(64, 128, 4, 2, 1),  # (64, 25, 32, 32) -> (128, 12, 16, 16)
            nn.BatchNorm3d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv3d(128, 256, 4, 2, 1),  # (128, 12, 128, 128) -> (256, 6, 64, 64)
            # nn.Conv3d(128, 256, 4, 2, 1),  # (128, 12, 16, 16) -> (256, 6, 8, 8)
            nn.BatchNorm3d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv3d(256, 512, 4, 2, 1),  # (256, 6, 64, 64) -> (512, 3, 32, 32)
            # nn.Conv3d(256, 512, 4, 2, 1),  # (256, 6, 8, 8) -> (512, 3, 4, 4)
            nn.BatchNorm3d(512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv3d(512, 1024, 4, 2, 1),  # (512, 3, 32, 32) -> (1024, 1, 16, 16)
            # nn.Conv3d(512, 1024, 4, 2, 1),  # (512, 3, 4, 4) -> (1024, 1, 2, 2)
            nn.BatchNorm3d(1024),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv3d(1024, 1, (1, 16, 16), 1, 0),  # (1024, 1, 16, 16) -> (1, 1, 1, 1)
            # nn.Conv3d(1024, 1, (1, 2, 2), 1, 0),  # (1024, 1, 2, 2) -> (1, 1, 1, 1)
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).view(-1, 1)
