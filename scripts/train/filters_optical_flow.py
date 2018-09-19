#!/usr/bin/env python3
import argparse
from pathlib import Path
import random
import sys

from keras.callbacks import ModelCheckpoint
import numpy as np

from colormotion import dataset
from colormotion.argparse import directory_path
from colormotion.nn.generators import VideoFramesGenerator
from colormotion.nn.layers import load_weights
from colormotion.nn.model.filters_optical_flow import model


def parse_args():
    parser = argparse.ArgumentParser(description='Run training.')
    parser.add_argument('--weights', type=Path, help='weights file')
    parser.add_argument('--steps-per-epoch', default=2000)
    parser.add_argument('--epochs', default=80)
    parser.add_argument('--validation-steps', default=20)
    parser.add_argument('dataset', type=directory_path, help='dataset folder')
    return parser.parse_args()


class Generator(VideoFramesGenerator):  # pylint: disable=too-few-public-methods
    '''Generate groups of contiguous frames from a dataset, with the L*a*b* channels of previous frames as the state.'''
    def load_sample(self, scene, start_frame, target_size):
        '''Load a sample to build a batch.'''
        # y = expected colorization in last frame
        # state = previous frames colorized and current frame in grayscale
        grayscale, y = dataset.read_frame_lab(scene, start_frame + self.contiguous_count, target_size)
        state = [
            np.dstack(dataset.read_frame_lab(scene, start_frame + i, target_size))
            for i in range(self.contiguous_count)
        ]
        return state + [grayscale, y]


def data_generators(dataset_folder):
    flow_params = {
        'batch_size': 8,
        'target_size': (256, 256),
        'seed': random.randrange(sys.maxsize),
    }
    # TODO Split train and test datasets
    train = Generator().flow_from_directory(dataset_folder, **flow_params)
    test = Generator().flow_from_directory(dataset_folder, **flow_params)
    return train, test


def frames_to_model_input(m, generator):
    '''Convert a pair of frames to the model inputs.'''
    # The model has inputs [l_input, l_input_tm1, features_tm1, ab_and_mask_input] and
    # outputs [x, encoded_features, interpolated_features]
    while True:
        state, y = next(generator)
        l_tm1, l = state
        features_tm1_placeholder = np.empty((32, 32, 512))
        ab_and_mask_input_placeholder = np.empty((256, 256, 3))
        _, features_tm1, _ = m.predict(l_tm1, l_tm1, features_tm1_placeholder, ab_and_mask_input_placeholder)
        yield [l, l_tm1, features_tm1, ab_and_mask_input_placeholder], y



def main(args):
    train_generator, test_generator = data_generators(args.dataset)
    m = model()
    if args.weights:
        load_weights(m, args.weights)
    checkpoint = ModelCheckpoint('epoch-{epoch:03d}-{val_loss:.3f}.hdf5', verbose=1, period=5)
    fit = m.fit_generator(
        frames_to_model_input(m, train_generator),
        steps_per_epoch=args.steps_per_epoch,
        epochs=args.epochs,
        validation_data=frames_to_model_input(m, test_generator),
        validation_steps=args.validation_steps,
        callbacks=[checkpoint])
    print(fit.history)
    m.save('optical_flow_model.h5')
    # score = m.evaluate(...)
    # print('Test loss:', score[0])
    # print('Test accuracy:', score[1])


if __name__ == '__main__':
    main(parse_args())
