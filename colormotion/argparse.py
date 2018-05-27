import argparse
from pathlib import Path


def directory_path(path):
    path = Path(path)
    if not path.is_dir():
        raise argparse.ArgumentTypeError('{} is not a valid directory'.format(path))
    return path
