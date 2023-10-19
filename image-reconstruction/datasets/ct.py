import random
from pathlib import Path
from typing import Optional, Callable, List, Dict

import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm import tqdm

from ..pre_transforms.base import PreTransform
from ..transforms.base import Transform


class CT(Dataset):

    NUM_TIME_STEPS = 10
    TRAIN_PER_VAL = 4

    def __init__(self,
                 directory: Path,
                 slice_indexing_func: Callable[[torch.Tensor], int],
                 transforms: Optional[List[Transform]] = None,
                 pre_transforms: Optional[List[PreTransform]] = None,
                 in_memory: bool = True,
                 phase: str = "train",
                 max_data: Optional[int] = None,
                 ) -> None:
        super().__init__()

        self.__paths = []
        for i, path in enumerate(self.__read_all(directory)):
            if max_data is not None and len(self.__paths) >= max_data:
                break
            if phase == "train" and i % (1 + self.TRAIN_PER_VAL) != 0:
                self.__paths.append(path)
            elif phase == "val" and i % (1 + self.TRAIN_PER_VAL) == 0:
                self.__paths.append(path)

        self.__data: List[torch.Tensor] = []
        if in_memory:
            for path in tqdm(self.__paths, desc="loading datasets..."):
                x_4d_np = np.load(path)["arr_0"]
                x_4d_tensor = torch.from_numpy(x_4d_np)

                for transform in pre_transforms:
                    x_4d_tensor = transform(x_4d_tensor)

                self.__data.append(x_4d_tensor)
        
        self.__slice_indexing_func = slice_indexing_func

        self.transforms = []
        if transforms is not None:
            self.transforms = transforms

        self.pre_transforms = []
        if pre_transforms is not None:
            self.pre_transforms = pre_transforms

    @staticmethod
    def get_normal_indexing_func(min_occupancy: float = 0.2,
                                 threshold: float = 0.1,
                                 ) -> Callable[[torch.Tensor], int]:

        def _normal_indexing_func(x: torch.Tensor) -> int:
            mask = torch.where(x > threshold, 1, 0)
            choices = []
            _, n, h, w = x.shape
            for i in range(w):
                if mask[:, :, i].sum() < min_occupancy * n * h:
                    continue
                choices.append(i)
            
            if len(choices) > 0:
                return random.choice(choices)

            return int(mask.sum(dim=(0, 1)).argmax())

        return _normal_indexing_func

    @staticmethod
    def __read_all(directory: Path) -> List[Path]:
        return list(sorted(directory.glob("**/*")))

    def __len__(self) -> int:
        return self.NUM_TIME_STEPS * len(self.__paths)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        idx = index // self.NUM_TIME_STEPS
        timestep_idx = index % self.NUM_TIME_STEPS

        if len(self.__data) > 0:
            x_4d_tensor = self.__data[idx]
        else:
            x_4d_np = np.load(self.__paths[idx])["arr_0"]
            x_4d_tensor = torch.from_numpy(x_4d_np)

            for transform in self.pre_transforms:
                x_4d_tensor = transform(x_4d_tensor)

        # channelを追加
        x_4d_tensor = x_4d_tensor.unsqueeze(0).float()

        x_3d_tensor = x_4d_tensor[:, timestep_idx]
        
        slice_idx = self.__slice_indexing_func(x_3d_tensor)
        # x_cine_2d_tensor = x_4d_tensor[:, :(1 + timestep_idx), :, :, slice_idx]
        x_2d_tensor = x_4d_tensor[:, timestep_idx, :, :, slice_idx]

        data = {
            "2d": x_2d_tensor,
            "3d": x_3d_tensor,
            "exhale_3d": x_4d_tensor[:, 0],
            "inhale_3d": x_4d_tensor[:, self.NUM_TIME_STEPS // 2],
            "exhale_2d": x_4d_tensor[:, 0, :, :, slice_idx],
            "inhale_2d": x_4d_tensor[:, self.NUM_TIME_STEPS // 2, :, :, slice_idx],
            "4d": x_4d_tensor,
            "slice_idx": torch.tensor(slice_idx, dtype=int),
            "timestep_idx": torch.tensor(timestep_idx, dtype=int),
        }

        for transform in self.transforms:
            data = transform(data)

        return data
