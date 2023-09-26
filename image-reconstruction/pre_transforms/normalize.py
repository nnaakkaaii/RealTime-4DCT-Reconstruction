import torch

from .base import PreTransform


class Normalize(PreTransform):
    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return (x - x.min()) / (x.max() - x.min())
