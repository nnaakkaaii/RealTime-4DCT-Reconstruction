from abc import ABCMeta, abstractmethod

import torch


class Transform(metaclass=ABCMeta):
    @abstractmethod
    def __call__(self,
                 data: dict[str, torch.Tensor],
                 ) -> dict[str, torch.Tensor]:
        pass
