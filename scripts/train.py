#!/usr/bin/env python3
import argparse
from pathlib import Path
import random
import sys

from colormotion.argparse import directory_path
from colormotion.nn.generators import VideoFramesDataGenerator
import colormotion.nn.model as model


def parse_args():
    parser = argparse.ArgumentParser(description='Run training.')
    parser.add_argument('--weights', type=Path, help='weights file')
    parser.add_argument('dataset', type=directory_path, help='dataset folder')
    return parser.parse_args()


def data_generators(dataset_folder):
    flow_params = {
        'batch_size': 8,
        'target_size': (256, 256),
        'seed': random.randrange(sys.maxsize),
    }
    # TODO Split train and test datasets
    train = VideoFramesDataGenerator().flow_from_directory(dataset_folder, **flow_params)
    test = VideoFramesDataGenerator().flow_from_directory(dataset_folder, **flow_params)
    return train, test


def main(args):
    train_generator, test_generator = data_generators(args.dataset)
    m = model.interactive_colorization(weights_path=args.weights)
    fit = m.fit_generator(
        train_generator,
        steps_per_epoch=2000,
        epochs=80,
        validation_data=test_generator,
        validation_steps=10)
    print(fit.history)
    m.save('Colorful_model.h5')
    # score = m.evaluate(...)
    # print('Test loss:', score[0])
    # print('Test accuracy:', score[1])


if __name__ == '__main__':
    main(parse_args())
