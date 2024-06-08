import os
import json
from pathlib import Path
from collections import OrderedDict

import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader

from .criteria.ssim import ssim
from .criteria.pjc import projection_consistency_loss


def update_keys(state_dict):
    new_state_dict = OrderedDict()
    for key, value in state_dict.items():
        # encoder_*.convをencoder_*.layersに変更
        if "encoder_" in key and ".conv." in key:
            new_key = key.replace(".conv.", ".layers.")
        # deconv.*をdeconv.layers.*に変更
        elif "deconv." in key and "deconv.layers." not in key:
            new_key = key.replace("deconv.", "deconv.layers.")
        else:
            new_key = key
        new_state_dict[new_key] = value
    return new_state_dict


def test(val_set: Dataset,
         generator: nn.Module,
         save_dir: Path,
         device: str = "cuda:0",
         ) -> None:
    if device == "cuda:0":
        generator = nn.DataParallel(generator)

    val_loader = DataLoader(val_set, batch_size=1, shuffle=False)
    criterion_mse = nn.MSELoss()

    for epoch_save_dir in save_dir.glob("epoch_*"):
        if not epoch_save_dir.is_dir():
            continue

        if not (epoch_save_dir / "generator.pth").exists():
            continue

        test_save_dir = epoch_save_dir / "test_results"

        if test_save_dir.exists():
            continue

        os.makedirs(test_save_dir)

        generator_state_dict = torch.load(epoch_save_dir / "generator.pth")
        generator_state_dict = update_keys(generator_state_dict)
        generator.module.load_state_dict(generator_state_dict)

        generator.to(device)
        generator.eval()

        fakes = []
        reals = []
        metrics = []
        last_idx = None
        last_timestep_idx = None
        with torch.no_grad():
            for data in val_loader:
                if data['idx'] == last_idx:
                    assert data['timestep_idx'] == last_timestep_idx + 1, f'expected timestep idx {last_timestep_idx + 1}, got {data["timestep_idx"]}'
                    last_timestep_idx = int(data['timestep_idx'])
                elif last_idx is None:
                    last_idx = int(data['idx'])
                    last_timestep_idx = int(data['timestep_idx'])
                else:
                    assert data['idx'] == last_idx + 1, f'expected idx {last_idx + 1}, got {data["idx"]}'
                    # save data
                    fake = torch.cat(fakes, dim=1)
                    np.savez(test_save_dir / f"fake_{last_idx}.npz", fake.numpy())
                    real = torch.cat(reals, dim=1)
                    np.savez(test_save_dir / f"real_{last_idx}.npz", real.numpy())
                    # save metrics
                    with open(test_save_dir / f"metrics_{last_idx}.json", "w") as f:
                        json.dump(metrics, f)
                    metrics = []
                    fakes = []
                    reals = []

                    last_idx = int(data['idx'])
                    last_timestep_idx = int(data['timestep_idx'])

                real_3d_ct = data["3d"].to(device)
                real_2d_ct = data["2d"].to(device)

                fake_3d_ct = generator(real_2d_ct,
                                       data["exhale_3d"].to(device),
                                       data["inhale_3d"].to(device),
                                       data["exhale_2d"].to(device),
                                       data["inhale_2d"].to(device),
                                       )

                loss_mse = criterion_mse(fake_3d_ct, real_3d_ct)
                loss_ssim = ssim(fake_3d_ct, real_3d_ct)
                loss_pjc = projection_consistency_loss(fake_3d_ct,
                                                       real_2d_ct,
                                                       slice_idx=data["slice_idx"],
                                                       )

                fakes.append(fake_3d_ct.cpu().clone().detach())
                reals.append(real_3d_ct.cpu().clone().detach())
                metrics.append({
                    "val_mse": loss_mse.item(),
                    "val_ssim": loss_ssim.item(),
                    "val_pjc": loss_pjc.item(),
                })

    return
