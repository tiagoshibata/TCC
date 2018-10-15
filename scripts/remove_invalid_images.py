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
        image = cv2.imread(str(path))
        try:
            if image is None:
                raise RuntimeError('Cannot read image {}'.format(path))
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            histogram = sorted(cv2.calcHist([gray], [0], None, [64], [0, 256]) / (gray.shape[0] * gray.shape[1]))
            if histogram[-1] + histogram[-2] >= 0.9:
                raise RuntimeError('Histogram shows few light variation')
        except RuntimeError as e:
            print(e)
            path.unlink()

    with ConsumerPool(validate) as validate_consumer_pool:
        for image_path in dataset_path.rglob('*'):
            validate_consumer_pool.put(image_path)


def main(args):
    validate_images(args.dataset)


if __name__ == '__main__':
    main(parse_args())
