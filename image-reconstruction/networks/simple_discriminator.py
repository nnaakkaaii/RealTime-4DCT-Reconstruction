import torch
from torch import nn


class SimpleDiscriminator(nn.Module):
    def __init__(self, pool_size: int) -> None:
        assert pool_size == 8
        super().__init__()

        self.net = nn.Sequential(
            nn.Conv3d(1, 64, 4, 2, 1),  # (1, 64, 64, 64) -> (64, 32, 32, 32)
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv3d(64, 128, 4, 2, 1),  # (64, 32, 32, 32) -> (128, 16, 16, 16)
            nn.BatchNorm3d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv3d(128, 256, 4, 2, 1),  # (128, 16, 16, 16) -> (256, 8, 8, 8)
            nn.BatchNorm3d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv3d(256, 512, 4, 2, 1),  # (256, 8, 8, 8) -> (512, 4, 4, 4)
            nn.BatchNorm3d(512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv3d(512, 1024, 4, 2, 1),  # (512, 4, 4, 4) -> (1024, 2, 2, 2)
            nn.BatchNorm3d(1024),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv3d(1024, 1, 4, 2, 1),  # (1024, 2, 2, 2) -> (1, 1, 1, 1)
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).view(-1, 1)


if __name__ == "__main__":
    x = torch.randn(32, 1, 64, 64, 64)
    d = SimpleDiscriminator(pool_size=8)
    print(d(x).shape)
