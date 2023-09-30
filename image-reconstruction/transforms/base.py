from abc import ABCMeta, abstractmethod
from typing import Dict

import torch


class Transform(metaclass=ABCMeta):
    @abstractmethod
    def __call__(self,
                 data: Dict[str, torch.Tensor],
                 ) -> Dict[str, torch.Tensor]:
        pass
