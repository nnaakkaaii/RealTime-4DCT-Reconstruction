from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm


def main(result_dir: Path) -> None:
    for fake_path in tqdm(list(sorted(result_dir.glob("**/fake_*.npz")))):
        name = fake_path.stem.lstrip("fake_")
        directory = fake_path.parent

        real_path = directory / ("real_" + name + ".npz")
        image_path = directory / ("image_" + name + ".jpg")

        if image_path.exists():
            continue

        fake = np.load(fake_path)['arr_0']
        if real_path.exists():
            real = np.load(real_path)['arr_0']
        else:
            real = None

        b_size = len(fake)

        fig, axs = plt.subplots(
            b_size,
            4,
            figsize=(10, b_size*2),
            gridspec_kw={'width_ratios': [128, 128, 128, 128]},
            )

        _, _, z, h, w = fake.shape

        for i in range(b_size):
            # 列1: x[:, 0, :, :, 64]
            axs[i, 0].imshow(fake[i, 0, :, :, w // 2], cmap='gray')
            axs[i, 0].axis('off')

            # 列2: y[:, 0, :, :, 64] または空白
            if real is not None:
                axs[i, 1].imshow(real[i, 0, :, :, w // 2], cmap='gray')
            axs[i, 1].axis('off')

            # 列3: x[:, 0, 25, :, :]
            axs[i, 2].imshow(fake[i, 0, z // 2, :, :], cmap='gray')
            axs[i, 2].axis('off')

            # 列4: y[:, 0, 25, :, :] または空白
            if real is not None:
                axs[i, 3].imshow(real[i, 0, z // 2, :, :], cmap='gray')
            axs[i, 3].axis('off')

        plt.subplots_adjust(wspace=0, hspace=0)
        plt.savefig(image_path, bbox_inches='tight', pad_inches=0)
        plt.close()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--result_dir', type=Path, required=True)
    args = parser.parse_args()

    main(result_dir=args.result_dir)
