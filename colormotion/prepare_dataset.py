#!/usr/bin/env python3
import argparse
import hashlib
from pathlib import Path
import sys

import ffmpy


def parse_args():
    parser = argparse.ArgumentParser(description='Prepares a dataset for training.')
    parser.add_argument('source', help='source video')
    parser.add_argument('destination', help='destination directory')
    return parser.parse_args()


def hash_file(filename):
    digest = hashlib.blake2b(digest_size=20)
    with open(filename, 'rb') as f:
        for chunk in iter(lambda: f.read(8192), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main(args):
    file_hash = hash_file(args.source)
    destination = Path(args.destination)

    if (destination / '{}_{:06}.png'.format(file_hash, 1)).exists():
        print('Error: This video has already been added to folder {}'.format(destination), file=sys.stderr)
        raise SystemExit(1)
    output = destination / '{}_%06d.png'.format(file_hash)  # %06d will be processed by ffmpeg
    ffmpy.FFmpeg(
        inputs={args.source: None},
        outputs={str(output): None},
    ).run()


if __name__ == '__main__':
    main(parse_args())
