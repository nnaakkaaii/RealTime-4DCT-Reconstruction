import torch
from torch.nn import functional as F

from .base import PreTransform


class UniformShape(PreTransform):
    def __init__(self,
                 target_shape: tuple[int, int, int] = (50, 512, 512),
                 ) -> None:
        self.__target_shape = target_shape

    def __call__(self,
                 x: torch.Tensor,
                 ) -> torch.Tensor:
        _, n, h, w = x.shape
        target_n, target_h, target_w = self.__target_shape

        min_val = torch.min(x)

        # N dimension
        if n < target_n:
            pad_size = target_n - n
            pad_top = pad_size // 2
            pad_bottom = pad_size - pad_top
            x = F.pad(x, (0, 0, 0, 0, pad_top, pad_bottom), value=min_val)
        elif n > target_n:
            crop_size = n - target_n
            crop_top = crop_size // 2
            crop_bottom = crop_size - crop_top
            x = x[:, crop_top:n-crop_bottom, :, :]

        # H dimension
        if h < target_h:
            pad_size = target_h - h
            pad_top = pad_size // 2
            pad_bottom = pad_size - pad_top
            x = F.pad(x, (0, 0, pad_top, pad_bottom, 0, 0), value=min_val)
        elif h > target_h:
            crop_size = h - target_h
            crop_top = crop_size // 2
            crop_bottom = crop_size - crop_top
            x = x[:, :, crop_top:h-crop_bottom, :]

        # W dimension
        if w < target_w:
            pad_size = target_w - w
            pad_left = pad_size // 2
            pad_right = pad_size - pad_left
            x = F.pad(x, (pad_left, pad_right, 0, 0, 0, 0), value=min_val)
        elif w > target_w:
            crop_size = w - target_w
            crop_left = crop_size // 2
            crop_right = crop_size - crop_left
            x = x[:, :, :, crop_left:w-crop_right]

        return x
