import torch
from torch.nn import functional as F


def projection_consistency_loss(reconstructed_3d: torch.Tensor,
                                input_2d: torch.Tensor,
                                slice_idx: torch.Tensor,
                                ) -> torch.Tensor:
    sliced_tensors = []
    for i, idx in enumerate(slice_idx):
        sliced_tensor = reconstructed_3d[i, :, :, idx].unsqueeze(0)
        sliced_tensors.append(sliced_tensor)
    loss = F.mse_loss(torch.cat(sliced_tensors, dim=0), input_2d)
    return loss
