import os
import json
from pathlib import Path
from collections import defaultdict
from typing import Tuple

import torch
import numpy as np
from torch import nn
from torch.optim import Adam
from torch.utils.data import Dataset, DataLoader

from .criteria.ssim import ssim
from .criteria.pjc import projection_consistency_loss


def train(train_set: Dataset,
          val_set: Dataset,
          generator: nn.Module,
          discriminator: nn.Module,
          num_epochs: int,
          batch_size: int,
          lr_g: float,
          lr_d: float,
          betas_g: Tuple[float, float],
          betas_d: Tuple[float, float],
          c_adv: float,
          c_mse: float,
          c_ssim: float,
          c_pjc: float,
          save_dir: Path,
          save_graph_per_idx: int = 10,
          save_weight_per_epoch: int = 10,
          target: str = "mse",
          device: str = "cuda:0",
          ) -> float:
    min_target_value = 10**6

    if device == "cuda:0":
        generator = nn.DataParallel(generator)
        discriminator = nn.DataParallel(discriminator)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)

    generator.to(device)
    discriminator.to(device)

    optim_g = Adam(generator.parameters(), lr=lr_g, betas=betas_g)
    optim_d = Adam(discriminator.parameters(), lr=lr_d, betas=betas_d)

    criterion_adv = nn.BCELoss()
    criterion_mse = nn.MSELoss()

    metrics_history = []

    for epoch in range(num_epochs):
        generator.train()
        discriminator.train()

        metrics = defaultdict(float)

        for data in train_loader:
            batch_size = data["3d"].size(0)

            real_3d_ct = data["3d"].to(device)
            real_2d_ct = data["2d"].to(device)

            # update discriminator
            label_real = torch.ones(batch_size).to(device)
            label_fake = torch.zeros(batch_size).to(device)

            optim_d.zero_grad()

            output_real = discriminator(real_3d_ct).view(-1)
            loss_real = criterion_adv(output_real, label_real)

            fake_3d_ct = generator(real_2d_ct,
                                   data["exhale_3d"].to(device),
                                   data["inhale_3d"].to(device),
                                   data["exhale_2d"].to(device),
                                   data["inhale_2d"].to(device),
                                   )
            
            output_fake = discriminator(fake_3d_ct.detach()).view(-1)
            loss_fake = criterion_adv(output_fake, label_fake)

            loss_d = loss_real + loss_fake
            loss_d.backward()

            optim_d.step()

            metrics["train_d_total"] += loss_d.item()

            # udpate generator
            optim_g.zero_grad()

            output_fake = discriminator(fake_3d_ct).view(-1)
            loss_adv = criterion_adv(output_fake, label_real)

            loss_mse = criterion_mse(fake_3d_ct, real_3d_ct)
            loss_ssim = ssim(fake_3d_ct, real_3d_ct)
            loss_pjc = projection_consistency_loss(fake_3d_ct,
                                                   real_2d_ct,
                                                   slice_idx=data["slice_idx"],
                                                   )
            
            loss_g = (
                c_adv * loss_adv
                + c_mse * loss_mse
                + c_ssim * loss_ssim
                + c_pjc * loss_pjc
            )
            loss_g.backward()

            optim_g.step()

            metrics["train_adv"] += loss_adv.item()
            metrics["train_mse"] += loss_mse.item()
            metrics["train_ssim"] += loss_ssim.item()
            metrics["train_pjc"] += loss_pjc.item()
            metrics["train_g_total"] += loss_g.item()

        generator.eval()
        discriminator.eval()

        epoch_save_dir = save_dir / f"epoch_{epoch}"

        with torch.no_grad():
            for idx, data in enumerate(val_loader):
                batch_size = data["3d"].size(0)

                real_3d_ct = data["3d"].to(device)
                real_2d_ct = data["2d"].to(device)

                # discriminator
                label_real = torch.ones(batch_size).to(device)
                label_fake = torch.zeros(batch_size).to(device)

                output_real = discriminator(real_3d_ct).view(-1)
                loss_real = criterion_adv(output_real, label_real)

                fake_3d_ct = generator(real_2d_ct,
                                       data["exhale_3d"].to(device),
                                       data["inhale_3d"].to(device),
                                       data["exhale_2d"].to(device),
                                       data["inhale_2d"].to(device),
                                       )

                output_fake = discriminator(fake_3d_ct.detach()).view(-1)
                loss_fake = criterion_adv(output_fake, label_fake)

                loss_d = loss_real + loss_fake

                metrics["val_d_total"] += loss_d.item()

                output_fake = discriminator(fake_3d_ct.detach()).view(-1)
                loss_adv = criterion_adv(output_fake, label_real)

                loss_mse = criterion_mse(fake_3d_ct, real_3d_ct)
                loss_ssim = ssim(fake_3d_ct, real_3d_ct)
                loss_pjc = projection_consistency_loss(fake_3d_ct, real_2d_ct, slice_idx=data["slice_idx"])

                loss_g = (
                    c_adv * loss_adv
                    + c_mse * loss_mse
                    + c_ssim * loss_ssim
                    + c_pjc * loss_pjc
                )

                metrics["val_adv"] += loss_adv.item()
                metrics["val_mse"] += loss_mse.item()
                metrics["val_ssim"] += loss_ssim.item()
                metrics["val_pjc"] += loss_pjc.item()
                metrics["val_g_total"] += loss_g.item()

                if idx % save_graph_per_idx == 0:
                    os.makedirs(epoch_save_dir, exist_ok=True)
                    np.savez(epoch_save_dir / f"real_3d_ct_batch_{idx}.npz", real_3d_ct.cpu().numpy())
                    np.savez(epoch_save_dir / f"fake_3d_ct_batch_{idx}.npz", fake_3d_ct.cpu().numpy())
                
                if metrics[f"val_{target}"] < min_target_value:
                    min_target_value = metrics[f"val_{target}"]
        
        if epoch % save_weight_per_epoch == 0:
            os.makedirs(epoch_save_dir, exist_ok=True)
            
            if device == "cuda:0":
                torch.save(generator.module.state_dict(), epoch_save_dir / "generator.pth")
                torch.save(discriminator.module.state_dict(), epoch_save_dir / "discriminator.pth")
            else:
                torch.save(generator.state_dict(), epoch_save_dir / "generator.pth")
                torch.save(discriminator.state_dict(), epoch_save_dir / "discriminator.pth")

        print(f"Epoch {epoch}/{num_epochs}")
        print(f"Train Losses: "
              f"MSE: {metrics['train_mse'] / len(train_loader)}, "
              f"SSIM: {metrics['train_ssim'] / len(train_loader)}, "
              f"ADV: {metrics['train_adv'] / len(train_loader)}, "
              f"PJC: {metrics['train_pjc'] / len(train_loader)}, "
              f"Total G: {metrics['train_g_total'] / len(train_loader)}, "
              f"Total D: {metrics['train_d_total'] / len(train_loader)}"
              )
        print(f"Val Losses: "
              f"MSE: {metrics['val_mse'] / len(train_loader)}, "
              f"SSIM: {metrics['val_ssim'] / len(train_loader)}, "
              f"ADV: {metrics['val_adv'] / len(train_loader)}, "
              f"PJC: {metrics['val_pjc'] / len(train_loader)}, "
              f"Total G: {metrics['val_g_total'] / len(train_loader)}, "
              f"Total D: {metrics['val_d_total'] / len(train_loader)}"
              )
        print("="*50)

        metrics_history.append(dict(metrics))
        with open(save_dir / "metrics.json", "wb") as f:
            json.dump(metrics_history, f, indent=2)

    return min_target_value
