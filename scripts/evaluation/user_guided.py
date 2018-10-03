#!/usr/bin/env python3
import argparse
from pathlib import Path

import cv2
import numpy as np

from colormotion import dataset
from colormotion.argparse import directory_path
from colormotion.nn.layers import load_weights
from colormotion.nn.model.user_guided import model
from colormotion.user_guided import ab_and_mask_matrix


def parse_args():
    parser = argparse.ArgumentParser(
        description='Evaluate a user guided colorization model.')
    parser.add_argument('model', type=Path, help='model weights')
    parser.add_argument('dataset', type=directory_path, help='dataset directory')
    parser.add_argument('output', type=directory_path, help='output directory')
    return parser.parse_args()


def main(args):
    scenes = dataset.get_all_scenes(args.dataset)
    m = model()
    load_weights(m, args.model)

    for scene, frames in scenes.items():
        def read_image_lab(scene, frame_number):
            bgr_image = dataset.read_image(dataset.get_frame_path(scene, frame_number), resolution=(256, 256))
            return dataset.bgr_to_lab(bgr_image)

        def predict(grayscale_input, ab_and_mask_input):
            return m.predict({
                'grayscale_input': np.expand_dims(grayscale_input, axis=0),
                'ab_and_mask_input': np.expand_dims(ab_and_mask_input, axis=0),
            }, verbose=1)

        for frame in frames:
            l, ab = read_image_lab(scene, frame)
            x = predict(l, ab_and_mask_matrix(ab, .1))
            image = np.round(255 * dataset.lab_to_bgr(l, x[0])).astype('uint8')
            cv2.imshow('Video', image)
            cv2.waitKey(1)


if __name__ == '__main__':
    main(parse_args())
