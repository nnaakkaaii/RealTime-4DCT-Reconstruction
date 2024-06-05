import os
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset

from .criteria.ssim import ssim
from .criteria.pjc import projection_consistency_loss


def test(val_set: Dataset,
         save_dir: Path,
         device: str = "cuda:0",
         ) -> None:
    criterion_mse = nn.MSELoss()

    for epoch_save_dir in save_dir.glob("epoch_*"):
        if not epoch_save_dir.is_dir():
            continue

        if not (epoch_save_dir / "generator.pth").exists():
            continue
        generator = torch.load(epoch_save_dir / "generator.pth")
        if device == "cuda:0":
            generator = nn.DataParallel(generator)

        generator.to(device)
        generator.eval()

        test_save_dir = epoch_save_dir / "test_results"
        os.makedirs(test_save_dir, exist_ok=True)

        fakes = []
        reals = []
        metrics = []
        last_idx = None
        last_timestep_idx = None
        with torch.no_grad():
            for data in val_set:
                if last_idx is None and last_timestep_idx is None:
                    last_idx = data['idx']
                    last_timestep_idx = data['timestep_idx']
                else:
                    if data['idx'] == last_idx:
                        assert data['timestep_idx'] == last_timestep_idx + 1, f'expected timestep idx {last_timestep_idx + 1}, got {data["timestep_idx"]}'
                        last_timestep_idx = data['timestep_idx']
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

                        last_idx = data['idx']

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
