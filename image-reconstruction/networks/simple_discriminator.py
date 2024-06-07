import torch
from torch import nn


class SimpleDiscriminator(nn.Module):
    def __init__(self, pool_size: int, method: str = 'cat') -> None:
        assert pool_size == 8
        super().__init__()

        self.method = method
        in_channels = 1
        if self.method == 'cat':
            in_channels = 3

        self.net = nn.Sequential(
            nn.Conv3d(in_channels, 64, 4, 2, 1),  # (1, 64, 64, 64) -> (64, 32, 32, 32)
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

    def forward(self, x: torch.Tensor, x_exhale: torch.Tensor, x_inhale: torch.Tensor) -> torch.Tensor:
        if self.method == 'cat':
            x = torch.cat([x, x_exhale, x_inhale], dim=1)
        return self.net(x).view(-1, 1)


if __name__ == "__main__":
    _x, _x_exhale, _x_inhale = [torch.randn(32, 1, 64, 64, 64) for _ in range(3)]
    d = SimpleDiscriminator(pool_size=8)
    print(d(_x, _x_exhale, _x_inhale).shape)
