import torch
from torch import nn


class SimpleDiscriminator(nn.Module):
    def __init__(self, pool_size: int) -> None:
        super().__init__()

        self.net = nn.Sequential(
            nn.Conv3d(1, 64, 4, 2, 1),  # (1, 50, 128, 128) -> (64, 25, 64, 64)
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv3d(64, 128, 4, 2, 1),  # (64, 25, 64, 64) -> (128, 12, 32, 32)
            nn.BatchNorm3d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv3d(128, 256, 4, 2, 1),  # (128, 12, 32, 32) -> (256, 6, 16, 16)
            nn.BatchNorm3d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv3d(256, 512, 4, 2, 1),  # (256, 6, 16, 16) -> (512, 3, 8, 8)
            nn.BatchNorm3d(512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv3d(512, 1024, 4, 2, 1),  # (512, 3, 8, 8) -> (1024, 1, 4, 4)
            nn.BatchNorm3d(1024),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv3d(1024, 1, (1, 16 // pool_size, 16 // pool_size), 1, 0),  # (1024, 1, 4, 4) -> (1, 1, 1, 1)
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).view(-1, 1)
