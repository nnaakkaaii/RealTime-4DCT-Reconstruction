from pathlib import Path
from typing import Tuple, Optional

from .train import train
from .test import test
from .datasets.ct import CT
from .pre_transforms.normalize import Normalize
from .pre_transforms.shift import Shift
from .pre_transforms.uniform_shape import UniformShape
from .pre_transforms.pool import Pool
from .networks.simple_generator import SimpleGenerator
from .networks.weighted_generator import WeightedGenerator
from .networks.resnet_generator import ResNetGenerator
from .networks.simple_discriminator import SimpleDiscriminator


def main(phase: str,
         data_dir: Path,
         in_memory: bool,
         dataset_slice_indexing_min_occupancy: float,
         dataset_slice_indexing_threshold: float,
         pool_size: int,
         use_shift_pre_transform: bool,
         shift_pre_transform_n: int,
         shift_pre_transform_h: int,
         shift_pre_transform_w: int,
         generator_name: str,
         generator_use_batch_norm: bool,
         generator_num_layers: int,
         generator_num_inner_layers: int,
         generator_bottleneck_channels: Optional[int],
         discriminator_name: str,
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
         max_iter: Optional[int] = None,
         ) -> None:
    train_pre_transforms = [Normalize()]
    if use_shift_pre_transform:
        train_pre_transforms.append(Shift(
            (shift_pre_transform_n,
             shift_pre_transform_h,
             shift_pre_transform_w,
             ),
            ))
    train_pre_transforms.append(UniformShape())
    if pool_size != 1:
        train_pre_transforms.append(Pool(pool_size))
    
    val_pre_transforms = [
        Normalize(),
        UniformShape(),
        ]
    if pool_size != 1:
        val_pre_transforms.append(Pool(pool_size))
    val_dataset = CT(
        directory=data_dir,
        slice_indexing_func=CT.get_normal_indexing_func(
            dataset_slice_indexing_min_occupancy,
            dataset_slice_indexing_threshold,
            ),
        pre_transforms=val_pre_transforms,
        phase="val",
        in_memory=in_memory,
        max_data=None if max_iter is None else batch_size * max_iter,
        )
    if phase == "test":
        test(val_set=val_dataset,
             save_dir=save_dir,
             device=device,
             )
        return

    assert phaes == "train"

    train_dataset = CT(
        directory=data_dir,
        slice_indexing_func=CT.get_normal_indexing_func(
            dataset_slice_indexing_min_occupancy,
            dataset_slice_indexing_threshold,
        ),
        pre_transforms=train_pre_transforms,
        phase="train",
        in_memory=in_memory,
        max_data=None if max_iter is None else batch_size * max_iter,
    )

    if generator_name == "simple":
        generator = SimpleGenerator(
            generator_use_batch_norm,
            generator_num_layers)
    elif generator_name == "weighted":
        generator = WeightedGenerator(
            generator_use_batch_norm,
            generator_num_layers)
    elif generator_name == "resnet":
        generator = ResNetGenerator(
            generator_num_layers,
            generator_num_inner_layers,
            generator_bottleneck_channels)
    else:
        raise KeyError(f"unknown generator {generator_name}")
    
    if discriminator_name == "simple":
        discriminator = SimpleDiscriminator(pool_size=pool_size)
    else:
        raise KeyError(f"unknown discriminator {discriminator_name}")

    train(train_dataset,
          val_dataset,
          generator,
          discriminator,
          num_epochs=num_epochs,
          batch_size=batch_size,
          lr_g=lr_g,
          lr_d=lr_d,
          betas_g=betas_g,
          betas_d=betas_d,
          c_adv=c_adv,
          c_mse=c_mse,
          c_ssim=c_ssim,
          c_pjc=c_pjc,
          save_dir=save_dir,
          save_graph_per_idx=save_graph_per_idx,
          save_weight_per_epoch=save_weight_per_epoch,
          target=target,
          device=device,
          max_iter=max_iter,
          )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument('--phase', type=str, default='train', help='Phase of the program (default: train).')
    parser.add_argument('--data_dir', type=Path, required=True, help='Directory containing the CT dataset.')
    parser.add_argument('--in_memory', action='store_true', help='Place dataset on memory')
    parser.add_argument('--dataset_slice_indexing_min_occupancy', type=float, default=0.2, help='Minimum occupancy for slice indexing.')
    parser.add_argument('--dataset_slice_indexing_threshold', type=float, default=0.1, help='Threshold for slice indexing.')
    parser.add_argument('--pool_size', type=int, default=8, help='Size of pooling.')
    parser.add_argument('--use_shift_pre_transform', action='store_true', help='Use shift pre-transform or not.')
    parser.add_argument('--shift_pre_transform_n', type=int, default=5, help='Shift along the n axis for pre-transform.')
    parser.add_argument('--shift_pre_transform_h', type=int, default=30, help='Shift along the h axis for pre-transform.')
    parser.add_argument('--shift_pre_transform_w', type=int, default=30, help='Shift along the w axis for pre-transform.')
    parser.add_argument('--generator_name', type=str, choices=['simple', 'weighted', 'resnet'], default='simple', help='Name of the generator to use.')
    parser.add_argument('--generator_use_batch_norm', action='store_true', help='Use batch norm')
    parser.add_argument('--generator_num_layers', type=int, default=3, help='Number of layers')
    parser.add_argument('--generator_num_inner_layers', type=int, default=3, help='Number of inner layers')
    parser.add_argument('--generator_bottleneck_channels', type=int, default=None, help='Number of bottleneck channels')
    parser.add_argument('--discriminator_name', type=str, choices=['simple'], default='simple', help='Name of the discriminator to use.')
    parser.add_argument('--num_epochs', type=int, default=100, help='Number of epochs for training.')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training.')
    parser.add_argument('--lr_g', type=float, default=0.0002, help='Learning rate for the generator.')
    parser.add_argument('--lr_d', type=float, default=0.0002, help='Learning rate for the discriminator.')
    parser.add_argument('--betas_g', type=tuple, default=(0.5, 0.999), help='Betas for Adam optimizer for the generator.')
    parser.add_argument('--betas_d', type=tuple, default=(0.5, 0.999), help='Betas for Adam optimizer for the discriminator.')
    parser.add_argument('--c_adv', type=float, default=1.0, help='Weight for adversarial loss.')
    parser.add_argument('--c_mse', type=float, default=1.0, help='Weight for MSE loss.')
    parser.add_argument('--c_ssim', type=float, default=1.0, help='Weight for SSIM loss.')
    parser.add_argument('--c_pjc', type=float, default=1.0, help='Weight for projection consistency loss.')
    parser.add_argument('--save_dir', type=Path, required=True, help='Directory to save the results.')
    parser.add_argument('--save_graph_per_idx', type=int, default=10, help='Frequency to save the generated CT graphs.')
    parser.add_argument('--save_weight_per_epoch', type=int, default=10, help='Frequency to save the model weights.')
    parser.add_argument('--target', type=str, default='mse', help='Target loss for training.')
    parser.add_argument('--device', type=str, choices=['cpu', 'cuda:0'], default='cuda:0', help='Name of the generator to use.')
    parser.add_argument('--max_iter', type=int, default=None, help='Max iterations per epoch')

    args = parser.parse_args()

    # Call the main function with the parsed arguments
    main(**vars(args))
