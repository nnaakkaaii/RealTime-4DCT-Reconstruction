import random

import torch
from torch.nn import functional as F

from .base import PreTransform


class Shift(PreTransform):
    def __init__(self,
                 ds: tuple[int, int, int] = (5, 30, 30),
                 ) -> None:
        self.dz, self.dx, self.dy = ds

    def __call__(self,
                    x: torch.Tensor,
                    ) -> torch.Tensor:
        min_val = torch.min(x)

        # N dimension (dz)
        dz = random.randint(-self.dz, self.dz)
        if dz != 0:
            if dz > 0:
                x = F.pad(x, (0, 0, 0, 0, dz, 0), value=min_val)
                x = x[:, dz:, :, :]
            else:
                x = F.pad(x, (0, 0, 0, 0, 0, -dz), value=min_val)
                x = x[:, :dz, :, :]
        
        # H dimension (dy)
        dy = random.randint(-self.dy, self.dy)
        if dy != 0:
            if dy > 0:
                x = F.pad(x, (0, 0, dy, 0, 0, 0), value=min_val)
                x = x[:, :, dy:, :]
            else:
                x = F.pad(x, (0, 0, 0, -dy, 0, 0), value=min_val)
                x = x[:, :, :dy, :]
        
        # W dimension (dx)
        dx = random.randint(-self.dx, self.dx)
        if dx != 0:
            if dx > 0:
                x = F.pad(x, (dx, 0, 0, 0, 0, 0), value=min_val)
                x = x[:, :, :, dx:]
            else:
                x = F.pad(x, (0, -dx, 0, 0, 0, 0), value=min_val)
                x = x[:, :, :, :dx]
        
        return x
