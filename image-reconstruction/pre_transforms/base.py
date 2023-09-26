from abc import ABCMeta, abstractmethod

import torch


class PreTransform(metaclass=ABCMeta):
    @abstractmethod
    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        pass
