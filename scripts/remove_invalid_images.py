#!/usr/bin/env python3
import argparse

import cv2

from colormotion.argparse import directory_path
from colormotion.threading import ConsumerPool


def parse_args():
    parser = argparse.ArgumentParser(description='Remove invalid image files.')
    parser.add_argument('dataset', type=directory_path, help='dataset directory')
    return parser.parse_args()


def validate_images(dataset_path):
    print('Validating images in dataset...')
    def validate(path):
        if path.suffix[1:].isnumeric():
            new_path = 'duplicate{}.{}'.format(path.suffix, path.stem)
            path.rename(new_path)
            path = new_path
        if cv2.imread(str(path)) is None:
            print('Cannot read image {}'.format(path))
            path.unlink()

    with ConsumerPool(validate) as validate_consumer_pool:
        for image_path in dataset_path.rglob('*'):
            validate_consumer_pool.put(image_path)


def main(args):
    validate_images(args.dataset)


if __name__ == '__main__':
    main(parse_args())
