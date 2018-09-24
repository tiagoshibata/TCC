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
from colormotion.nn.model.user_guided import model
from colormotion.user_guided import ab_and_mask_matrix


def parse_args():
    parser = argparse.ArgumentParser(description='Run training.')
    parser.add_argument('--weights', type=Path, help='weights file')
    parser.add_argument('--steps-per-epoch', default=2000)
    parser.add_argument('--epochs', default=80)
    parser.add_argument('--validation-steps', default=20)
    parser.add_argument('--dry', action='store_true')
    parser.add_argument('dataset', type=directory_path, help='dataset folder')
    return parser.parse_args()


class Generator(VideoFramesGenerator):
    '''Generate groups of contiguous frames from a dataset.

    The generated data has inputs [l_input, ab_and_mask_input].'''
    def load_batch(self, start_frames, target_size):
        assert self.contiguous_count == 1
        x_batch = [[], []]
        y_batch = []
        for scene, frame in start_frames:
            l, ab = dataset.read_frame_lab(scene, frame + self.contiguous_count, target_size)
            x_batch[0].append(ab_and_mask_matrix(ab, .05))
            x_batch[1].append(l)
            y_batch.append(ab)
        return [np.array(model_input) for model_input in x_batch], np.array(y_batch)

    def load_sample(self, scene, start_frame, target_size):
        pass  # unused


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


def main(args):
    train_generator, test_generator = data_generators(args.dataset)
    m = model()
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
    if not args.dry:
        m.save('user_guided_model.h5')
    # score = m.evaluate(...)
    # print('Test loss:', score[0])
    # print('Test accuracy:', score[1])


if __name__ == '__main__':
    main(parse_args())
