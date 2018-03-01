#!/usr/bin/env python3
import argparse
import hashlib
from itertools import count
import logging
from pathlib import Path
import sys
import threading
import time

import cv2
from skimage.measure import compare_ssim

from colormotion.threading import ConsumerPool


def parse_args():
    parser = argparse.ArgumentParser(description='Prepares a dataset for training.')
    parser.add_argument('source', help='source video')
    parser.add_argument('destination', help='destination directory')
    parser.add_argument('--verbose', '-v', action='count',
                        help='verbose mode (GUI output of each frame). If given twice, stop at each frame')
    parser.add_argument('--loglevel', choices=['debug', 'info', 'warning', 'error', 'critical'], default='warning', help='log level')
    return parser.parse_args()


def fail(message, *args, **kwargs):
    print(message, file=sys.stderr, *args, **kwargs)
    raise SystemExit(1)


def hash_file(filename):
    digest = hashlib.blake2b(digest_size=20)
    with open(filename, 'rb') as f:
        for chunk in iter(lambda: f.read(8192), b""):
            digest.update(chunk)
    return digest.hexdigest()


def get_destination_folder(source, destination):
    destination = Path(destination)
    if not destination.exists():
        fail('Folder {} does not exist'.format(destination))
    file_hash = hash_file(source)
    if any(destination.glob('{}_*'.format(file_hash))):
        fail('Video has already been processed (folders {}/{}_* exist)'.format(destination, file_hash))
    return str(destination / file_hash)


class FrameValidationException(Exception):
    pass


def validate_frame(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    histogram = cv2.calcHist([gray], [0], None, [64], [0, 256]) / (gray.shape[0] * gray.shape[1])
    if max(histogram) >= 0.96:
        raise FrameValidationException('Histogram shows few light variation')


def is_new_scene(frame, previous):
    if previous is None:
        return True
    ssim = compare_ssim(frame, previous, multichannel=True)
    scene_changed = ssim < 0.4
    logging.info('SSIM is {} ({} scene)'.format(ssim, scene_changed and 'NEW' or 'SAME'))
    return scene_changed


def build_dataset_from_video(video, destination, verbose=0):
    frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT)) or '<Unknown frame count>'
    start_time = time.time()
    frame = None
    encoding_pool = ConsumerPool(lambda x: cv2.imwrite(*x))
    try:
        for i in count():
            previous = frame
            has_frame, frame = video.read()
            if not has_frame:
                break
            try:
                if verbose:
                    cv2.imshow('Video', frame)
                    cv2.waitKey(verbose <= 1 and 1 or 0)
                validate_frame(frame)
                if is_new_scene(frame, previous):
                    print('Frame {} is a new scene'.format(i))
                    scene_destination = Path('{}_{:06d}'.format(destination, i))
                    scene_destination.mkdir()
                output = str(scene_destination / '{:06d}.png'.format(i))
                encoding_pool.put((output, frame))
                if not i % 1000:
                    print('Frame {}/{}: {} s'.format(i, frame_count, time.time() - start_time))
            except FrameValidationException as e:
                print('Invalid frame {}: {}'.format(i, e))
                frame = None
    finally:
        video.release()
        cv2.destroyAllWindows()
        encoding_pool.join()


def main(args):
    logging.basicConfig(level=getattr(logging, args.loglevel.upper()), format='%(asctime)s:%(levelname)s:%(message)s')
    destination = get_destination_folder(args.source, args.destination)
    video = cv2.VideoCapture(args.source)
    build_dataset_from_video(video, destination, args.verbose)


if __name__ == '__main__':
    main(parse_args())
