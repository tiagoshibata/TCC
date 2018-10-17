#!/usr/bin/env python3
import argparse
from pathlib import Path

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
    parser.add_argument('encoder', type=Path, help='encoder weights')
    parser.add_argument('decoder', type=Path, help='decoder weights')
    parser.add_argument('dataset', type=directory_path, help='dataset directory')
    parser.add_argument('output', type=directory_path, help='output directory')
    return parser.parse_args()


def main(args):  # pylint: disable=too-many-locals
    scenes = dataset.get_all_scenes(args.dataset)
    m = model()
    load_weights(m, args.encoder, by_name=True)
    load_weights(m, args.decoder, by_name=True)

    for scene, frames in scenes.items():
        l_tm1 = None
        for frame in frames:
            l, ab = dataset.read_frame_lab(scene, frame, (256, 256))
            if l_tm1 is None:
                # Set warped_features = encoded_features on the first frame
                _, warped_features, _ = m.predict([
                    np.array([ab_and_mask_matrix(ab, .00016)]), np.array([l]), np.empty((1, 32, 32, 512))], verbose=1)
            else:
                warped_features = warp_features(l_tm1, l, interpolated_features_tm1)[np.newaxis]

            x, _, interpolated_features = m.predict([
                np.array([ab_and_mask_matrix(ab, .00016)]), np.array([l]), warped_features], verbose=1)

            image = np.round(255 * dataset.lab_to_bgr(l, x[0])).astype('uint8')
            cv2.imshow('Video', image)
            cv2.waitKey(1)

            interpolated_features_tm1 = interpolated_features[0]
            l_tm1 = l


if __name__ == '__main__':
    main(parse_args())
