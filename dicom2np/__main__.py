from pathlib import Path

from .utils import process


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=Path, required=True)
    parser.add_argument('--save_dir', type=Path, required=True)
    parser.add_argument('--upper', type=int, default=-400)
    parser.add_argument('--lower', type=int, default=-900)
    args = parser.parse_args()

    process(args.data_dir.glob('**/*.dcm'),
            args.save_dir,
            args.lower,
            args.upper,
            )
