#!/usr/bin/env python3
import argparse
from pathlib import Path

import cv2
import numpy as np

from colormotion import dataset
from colormotion.argparse import directory_path
from colormotion.nn.graph import new_model_session
from colormotion.nn.layers import load_weights
from colormotion.nn.model.filters_optical_flow import interpolate_and_decode, warp_features
from colormotion.nn.model.user_guided import encoder_model
from colormotion.user_guided import ab_and_mask_matrix


def parse_args():
    parser = argparse.ArgumentParser(
        description='Evaluate a colorization model using optical flow for coherent colorization of videos.')
    parser.add_argument('encoder', type=Path, help='encoder weights')
    parser.add_argument('decoder', type=Path, help='decoder weights')
    parser.add_argument('dataset', type=directory_path, help='dataset directory')
    parser.add_argument('output', type=directory_path, help='output directory')
    return parser.parse_args()


def predict(session, model, inputs):
    with session():
        return model.predict([
            np.expand_dims(x, axis=0) for x in inputs
        ], verbose=1)


def main(args):  # pylint: disable=too-many-locals
    scenes = dataset.get_all_scenes(args.dataset)
    with new_model_session() as decoder_session:
        decoder = interpolate_and_decode()
        load_weights(decoder, args.decoder)
    with new_model_session() as encoder_session:
        encoder = encoder_model()
        load_weights(encoder, args.encoder, by_name=True)

    for scene, frames in scenes.items():
        for frame_tm1, frame in zip((None, *frames[1:]), frames):
            l_tm1, ab_tm1 = dataset.read_frame_lab(scene, frame_tm1 or frame, (256, 256))
            l, ab = dataset.read_frame_lab(scene, frame, (256, 256))

            features_tm1, _, _, _ = predict(encoder_session, encoder, [l_tm1, ab_and_mask_matrix(ab_tm1, .00008)])
            features, conv1_2norm, conv2_2norm, conv3_3norm = predict(
                encoder_session, encoder, [l, ab_and_mask_matrix(ab, .00008)])
            warped_features = warp_features(l_tm1, l, features_tm1[0])
            x = predict(decoder_session, decoder,
                        [np.array([warped_features]), features, conv1_2norm, conv2_2norm, conv3_3norm])

            image = np.round(255 * dataset.lab_to_bgr(l, x[0])).astype('uint8')
            cv2.imshow('Video', image)
            cv2.waitKey(1)


if __name__ == '__main__':
    main(parse_args())
