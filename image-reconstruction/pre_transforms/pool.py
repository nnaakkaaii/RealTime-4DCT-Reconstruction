import torch
from torch.nn import functional as F

from .base import PreTransform


class Pool(PreTransform):
    def __init__(self,
                 size: int = 8,
                 ) -> None:
        self.__size = (1, size, size)

    def __call__(self,
                 x: torch.Tensor,
                 ) -> torch.Tensor:
        return F.avg_pool3d(x,
                            kernel_size=self.__size,
                            stride=self.__size,
                            )
