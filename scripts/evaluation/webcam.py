#!/usr/bin/env python3
import argparse
from pathlib import Path
import random

import cv2
import numpy as np

from colormotion import dataset
from colormotion.argparse import directory_path
from colormotion.nn.layers import load_weights
from colormotion.nn.model.filters_optical_flow import model, warp_features
from colormotion.user_guided import ab_and_mask_matrix


def parse_args():
    parser = argparse.ArgumentParser(
        description='Evaluate a colorization model using optical flow for coherent colorization of videos.')
    parser.add_argument('--save', action='store_true', help='save results to a video file')
    parser.add_argument('--ui', action='store_true', help='show UI')
    parser.add_argument('encoder', type=Path, help='encoder weights')
    parser.add_argument('decoder', type=Path, help='decoder weights')
    parser.add_argument('video', help='video file or webcam id')
    return parser.parse_args()

mask_coverage = 16 / 100000


def capture_generator(capture):
    while True:
        success, frame = capture.read()
        if not success:
            return
        yield frame


def directory_generator(path):
    frames = sorted(Path(path).iterdir())
    for filename in frames:
        image = cv2.imread(str(filename))
        if image is not None:
            yield image


def open_video(video):
    if video.isdigit():
        # Ints are threated as camera indexes
        video = int(video)
    elif Path(video).is_dir():
        return directory_generator(video)
    return capture_generator(cv2.VideoCapture(video))


def main(args):  # pylint: disable=too-many-locals
    m = model()
    load_weights(m, args.encoder, by_name=True)
    load_weights(m, args.decoder, by_name=True)

    video = args.video
    capture = open_video(video)

    writer = None
    if args.save:
        truth = random.choice(['L', 'R'])
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        if video.isdigit():
            stem = video
        else:
            stem = Path(video).stem
        filename = 'output_{}_{}.avi'.format(stem, truth)
        print('Saving to {}'.format(filename))
        writer = cv2.VideoWriter(filename, fourcc, 30.0, (512, 256))
        truth = cv2.VideoWriter('truth_{}'.format(filename), fourcc, 30.0, (256, 256))
        colormotion = cv2.VideoWriter('colormotion_{}'.format(filename), fourcc, 30.0, (256, 256))

    def on_trackbar(val):
        global mask_coverage
        mask_coverage = val / 100000

    if args.ui:
        cv2.namedWindow('ColorMotion')
        cv2.createTrackbar('Mask percentage * 0.1%', 'ColorMotion' , 16, 100, on_trackbar)
        on_trackbar(16)

    l_tm1 = None
    prev = None
    interpolated_features_tm1 = None
    prev_mask = None
    # while True:
    for _, frame in zip(range(300), capture):
        frame = cv2.resize(frame, (256, 256), interpolation=cv2.INTER_AREA)
        l, ab = dataset.bgr_to_lab((frame / 255).astype(np.float32))
        if l_tm1 is None:
            # Set warped_features = encoded_features on the first frame
            _, warped_features, _ = m.predict([
                np.array([ab_and_mask_matrix(ab, mask_coverage)]), np.array([l]), np.empty((1, 32, 32, 512))], verbose=1)
        else:
            warped_features = warp_features(l_tm1, l, interpolated_features_tm1)[np.newaxis]

        mask = np.array([ab_and_mask_matrix(ab, mask_coverage)])
        if prev_mask is not None:
            prev_mask[:, :, 2] *= .8
            # mask_valid = mask[:, :, :, 2:3]
            # condition = np.stack((mask_valid, ) * 3, axis=-1)
            # mask = np.where(condition, mask, prev_mask)
            mask += prev_mask
        x, _, interpolated_features = m.predict([
            np.array([ab_and_mask_matrix(ab, mask_coverage)]), np.array([l]), warped_features], verbose=1)
        prev_mask = mask

        ab = x[0]
        if prev is not None:
            ab = (ab + 2 * prev) / 3
        prev = ab
        bgr = np.round(255 * dataset.lab_to_bgr(l, ab)).astype('uint8')
        if writer:
            if truth == 'L':
                output = (frame, bgr)
            else:
                output = (bgr, frame)
            output = np.concatenate(output, axis=1)
            writer.write(output)
            truth.write(frame)
            colormotion.write(bgr)
        if args.ui:
            cv2.imshow('Original stream', frame)
            cv2.imshow('ColorMotion', bgr)
        key = cv2.waitKey(1) & 255
        if key == ord('q'):
            break

        interpolated_features_tm1 = interpolated_features[0]
        l_tm1 = l

    if writer:
        writer.release()
        truth.release()
        colormotion.release()


if __name__ == '__main__':
    main(parse_args())
