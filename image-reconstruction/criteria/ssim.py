import torch
from torch.nn import functional as F


def ssim(x: torch.Tensor,
         y: torch.Tensor,
         window_size: int = 11,
         size_average: bool = True,
         ) -> torch.Tensor:
    c1 = 0.01 ** 2
    c2 = 0.03 ** 2
    
    mu_x = F.avg_pool3d(x, window_size, 1)
    mu_y = F.avg_pool3d(y, window_size, 1)
    
    sigma_x = F.avg_pool3d(x ** 2, window_size, 1) - mu_x ** 2
    sigma_y = F.avg_pool3d(y ** 2, window_size, 1) - mu_y ** 2
    sigma_xy = F.avg_pool3d(x * y, window_size, 1) - mu_x * mu_y
    
    s = ((2 * mu_x * mu_y + c1) * (2 * sigma_xy + c2)) / ((mu_x ** 2 + mu_y ** 2 + c1) * (sigma_x + sigma_y + c2))
    
    if size_average:
        return -s.mean()
    else:
        return -s.mean(1).mean(1).mean(1)
