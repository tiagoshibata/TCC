#!/usr/bin/env python3
import argparse

import cv2
from keras.models import load_model
import numpy as np

from colormotion.argparse import directory_path
import colormotion.dataset as dataset


def parse_args():
    parser = argparse.ArgumentParser(
        description='Evaluate model based on Real-Time User-Guided Image Colorization '
        'with Learned Deep Priors (R. Zhang et al).')
    parser.add_argument('model', help='model file')
    parser.add_argument('dataset', type=directory_path, help='dataset directory')
    parser.add_argument('output', type=directory_path, help='output directory')
    return parser.parse_args()


def main(args):
    model = load_model(str(args.model))
    scenes = dataset.get_all_scenes(args.dataset)
    for scene, frames in scenes.items():
        def read_image_lab(scene, frame_number):
            bgr_image = dataset.read_image(dataset.get_frame_path(scene, frame_number), resolution=(256, 256))
            return dataset.bgr_to_lab(bgr_image)

        def write_image_lab(filename, l, ab):
            image = np.round(255 * dataset.lab_to_bgr(l, ab)).astype('uint8')
            if not cv2.imwrite(filename, image):
                raise RuntimeError('Failed to write {}'.format(filename))

        def predict(state, grayscale_input):
            return model.predict({
                'state_input': np.expand_dims(state, axis=0),
                'grayscale_input': np.expand_dims(grayscale_input, axis=0)
            }, verbose=1)

        scene_number = scene.stem
        state = np.dstack(read_image_lab(scene, frames[0]))
        for i, frame in enumerate(frames[1:]):
            l, _ = read_image_lab(scene, frame)

            output = predict(state, l)
            write_image_lab('output_{:04d}_{:04d}.png'.format(scene_number, i), l, output[0])
            state = np.dstack((l, output[0]))


if __name__ == '__main__':
    main(parse_args())