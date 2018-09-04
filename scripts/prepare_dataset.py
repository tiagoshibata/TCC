#!/usr/bin/env python3
import argparse
from itertools import count
import logging
import threading
import time

import cv2
from skimage.measure import compare_ssim

from colormotion.argparse import directory_path
from colormotion import dataset
from colormotion.threading import ConsumerPool, ProducerPool


def parse_args():
    parser = argparse.ArgumentParser(description='Prepares a dataset for training.')
    parser.add_argument('source', help='source video')
    parser.add_argument('destination', type=directory_path, help='destination directory')
    parser.add_argument('--resolution', type=int, nargs=2, help='output resolution')
    parser.add_argument('--verbose', '-v', action='count',
                        help='verbose mode (GUI output of each frame). If given twice, stop at each frame')
    parser.add_argument('--loglevel', choices=['debug', 'info', 'warning', 'error', 'critical'],
                        default='warning', help='log level')
    return parser.parse_args()


class FrameValidationException(Exception):
    pass


def validate_frame(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    histogram = sorted(cv2.calcHist([gray], [0], None, [64], [0, 256]) / (gray.shape[0] * gray.shape[1]))
    if histogram[-1] + histogram[-2] >= 0.85:
        raise FrameValidationException('Histogram shows few light variation')


def is_new_scene(frame, previous):
    if previous is None:
        return True
    ssim = compare_ssim(frame, previous, multichannel=True)
    scene_changed = ssim < 0.35
    logging.info('SSIM is {:.4f} ({} scene)'.format(ssim, scene_changed and 'NEW' or 'same'))
    return scene_changed


def build_dataset_from_video(video, destination, verbose=0, resolution=None):
    frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT)) or '<Unknown frame count>'
    start_time = time.time()

    def decode_function():
        while True:
            has_frame, frame = video.read()
            if not has_frame:
                return
            if resolution:
                frame = cv2.resize(frame, (resolution[0], resolution[1]), interpolation=cv2.INTER_AREA)
            yield frame

    def consume_function(args):
        frame_data, previous_data, i = args
        frame, ready_event, _ = frame_data
        previous_frame, previous_ready, _ = previous_data or (None,) * 3
        if is_new_scene(frame, previous_frame):
            print('Frame {} is a new scene'.format(i))
            scene_destination = dataset.get_scene_directory(destination, i)
        else:
            # If this frame is a continuation of the previous scene, save to the same folder
            previous_ready.wait()
            scene_destination = previous_data[2]
        frame_data[2] = scene_destination
        # Destination has been processed and folder created if needed, so it's ready to save the next frame
        ready_event.set()
        cv2.imwrite(str(dataset.get_frame_path(scene_destination, i)), frame, [cv2.IMWRITE_PNG_COMPRESSION, 9])

    decoding_thread = ProducerPool(decode_function, num_workers=1)
    consume_pool = ConsumerPool(consume_function)
    previous_frame_data = None
    try:
        for i, frame in zip(count(), decoding_thread):
            if verbose:
                cv2.imshow('Video', frame)
                cv2.waitKey(verbose <= 1 and 1 or 0)
            try:
                validate_frame(frame)
                current_frame_data = [frame, threading.Event(), destination]
                consume_pool.put((current_frame_data, previous_frame_data, i))
                previous_frame_data = current_frame_data
            except FrameValidationException as e:
                print('Invalid frame {}: {}'.format(i, e))
                previous_frame_data = None
            if not i % 1000:
                print('Frame {}/{}: {:.2f} s'.format(i, frame_count, time.time() - start_time))
    finally:
        video.release()
        cv2.destroyAllWindows()
        decoding_thread.join()
        consume_pool.join()


def main(args):
    logging.basicConfig(level=getattr(logging, args.loglevel.upper()), format='%(asctime)s:%(levelname)s:%(message)s')
    destination = dataset.create_video_destination_folder(args.source, args.destination)
    video = cv2.VideoCapture(args.source)
    build_dataset_from_video(video, destination, args.verbose, args.resolution)


if __name__ == '__main__':
    main(parse_args())
