import argparse
from pathlib import Path


def directory_path(path):
    path = Path(path)
    if not path.is_dir():
        raise argparse.ArgumentTypeError('{} is not a valid directory'.format(path))
    return path


def training_args_parser():
    parser = argparse.ArgumentParser(description='Run training.')
    parser.add_argument('--weights', type=Path, help='weights file')
    parser.add_argument('--steps-per-epoch', type=int, default=2000)
    parser.add_argument('--epochs', type=int, default=80)
    parser.add_argument('--validation-steps', type=int, default=20)
    parser.add_argument('--dry', action='store_true')
    parser.add_argument('dataset', type=directory_path, help='dataset folder')
    return parser
