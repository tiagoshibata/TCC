#!/usr/bin/env python3
import argparse
from os import SEEK_END
from pathlib import Path
import uuid

import cv2

from colormotion.argparse import directory_path
from colormotion.threading import ConsumerPool


def parse_args():
    parser = argparse.ArgumentParser(description='Remove invalid image files.')
    parser.add_argument('dataset', type=directory_path, help='dataset directory')
    return parser.parse_args()


def validate_jpg(path):
    # Check for premature end of JPEG
    with path.open('rb') as f:
        try:
            f.seek(-2, SEEK_END)
        except OSError:
            raise RuntimeError('File too small')
        if f.read(2) != b'\xff\xd9':
            raise RuntimeError('Premature end of image')


def validate_images(dataset_path):
    print('Validating images in dataset...')
    def validate(path):
        if path.suffix[1:].isnumeric():
            new_path = Path('duplicate{}.{}'.format(path.suffix, path.stem))
            path.rename(new_path)
            path = new_path
        if path.suffix.lower() not in ('.bmp', '.gif', '.jpeg', '.jpg', '.pjpeg', '.pjpg', '.png', '.tiff'):
            return
        try:
            if path.suffix.lower() in ('.jpeg', '.jpg'):
                validate_jpg(path)
            try:
                image = cv2.imread(str(path))
            except UnicodeEncodeError as e:
                print('Filename has invalid UTF-8 characters')
                try:
                    str(path.suffix)
                except UnicodeEncodeError as e:
                    print('Extension has invalid UTF-8 characters, skipping')
                    return
                new_path = Path('{}.{}'.format(uuid.uuid4(), path.suffix))
                path.rename(new_path)
                path = new_path
                image = cv2.imread(str(path))
            if image is None:
                raise RuntimeError('Cannot read image')
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            histogram = sorted(cv2.calcHist([gray], [0], None, [64], [0, 256]) / (gray.shape[0] * gray.shape[1]))
            if histogram[-1] + histogram[-2] >= 0.9:
                raise RuntimeError('Histogram shows few light variation')
        except RuntimeError as e:
            path.unlink()
            try:
                print('{}: {}'.format(path, e))
            except UnicodeEncodeError:
                print(e)

    with ConsumerPool(validate) as validate_consumer_pool:
        for image_path in dataset_path.rglob('*'):
            validate_consumer_pool.put(image_path)


def main(args):
    validate_images(args.dataset)


if __name__ == '__main__':
    main(parse_args())
