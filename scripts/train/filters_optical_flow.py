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


class Generator(VideoFramesGenerator):
    '''Generate groups of contiguous frames from a dataset.

    The generated data has inputs [l_input, l_input_tm1, features_tm1, ab_and_mask_input].'''
    def __init__(self, nn_model, **kwargs):
        self.model = nn_model
        super().__init__(**kwargs)

    def load_batch(self, start_frames, target_size):  # pylint: disable=too-many-locals
        assert self.contiguous_count == 1
        x_batch = [[], [], [], []]
        y_batch = []
        for scene, frame in start_frames:
            # y = expected colorization in last frame
            # state = previous frames colorized and current frame in grayscale
            l, ab = dataset.read_frame_lab(scene, frame + self.contiguous_count, target_size)
            x_batch[0].append(l)
            y_batch.append(ab)
            l_tm1, ab_tm1 = dataset.read_frame_lab(scene, frame, target_size)
            x_batch[1].append(l_tm1)
            features_tm1_placeholder = np.empty((32, 32, 512))
            ab_and_mask_input_placeholder = np.zeros((256, 256, 3))
            _, features_tm1, _ = self.model.predict(l_tm1, l_tm1,
                                                    features_tm1_placeholder, ab_and_mask_input_placeholder)
            x_batch[2].append(features_tm1)
            # TODO create ab_and_mask_input using ab_tm1 instead of returning a placeholder
            x_batch[3].append(ab_and_mask_input_placeholder)
        return [np.array(model_input) for model_input in x_batch], np.array(y_batch)

    def load_sample(self, scene, start_frame, target_size):
        pass  # unused


def data_generators(dataset_folder, nn_model):
    flow_params = {
        'batch_size': 8,
        'target_size': (256, 256),
        'seed': random.randrange(sys.maxsize),
    }
    # TODO Split train and test datasets
    train = Generator(nn_model).flow_from_directory(dataset_folder, **flow_params)
    test = Generator(nn_model).flow_from_directory(dataset_folder, **flow_params)
    return train, test


def main(args):
    m = model()
    train_generator, test_generator = data_generators(args.dataset, m)
    if args.weights:
        load_weights(m, args.weights)
    checkpoint = ModelCheckpoint('epoch-{epoch:03d}-{val_loss:.3f}.hdf5', verbose=1, period=5)
    fit = m.fit_generator(
        train_generator,
        steps_per_epoch=args.steps_per_epoch,
        epochs=args.epochs,
        validation_data=test_generator,
        validation_steps=args.validation_steps,
        callbacks=[checkpoint])
    print(fit.history)
    m.save('optical_flow_model.h5')
    # score = m.evaluate(...)
    # print('Test loss:', score[0])
    # print('Test accuracy:', score[1])


if __name__ == '__main__':
    main(parse_args())
