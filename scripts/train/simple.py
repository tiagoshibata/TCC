#!/usr/bin/env python3
import argparse
from pathlib import Path

from keras.callbacks import ModelCheckpoint

from colormotion.argparse import directory_path
from colormotion.nn.generators import VideoFramesDataGenerator
from colormotion.nn.layers import load_weights
from colormotion.nn.model.simple import model


def parse_args():
    parser = argparse.ArgumentParser(description='Run training.')
    parser.add_argument('--weights', type=Path, help='weights file')
    parser.add_argument('dataset', type=directory_path, help='dataset folder')
    return parser.parse_args()


def data_generators(dataset_folder):
    flow_params = {
        'batch_size': 10,
        'target_size': (256, 256),
    }
    # TODO Split train and test datasets
    train = VideoFramesDataGenerator(contiguous_count=0).flow_from_directory(dataset_folder, **flow_params)
    test = VideoFramesDataGenerator(contiguous_count=0).flow_from_directory(dataset_folder, **flow_params)
    return train, test


def main(args):
    train_generator, test_generator = data_generators(args.dataset)
    m = model()
    if args.weights:
        load_weights(m, args.weights)
    checkpoint = ModelCheckpoint('epoch-{epoch:03d}-{val_loss:.3f}.hdf5', verbose=1, period=5)
    fit = m.fit_generator(
        train_generator,
        steps_per_epoch=2000,
        epochs=80,
        validation_data=test_generator,
        validation_steps=10,
        callbacks=[checkpoint])
    print(fit.history)
    m.save('simple_model.h5')
    # score = m.evaluate(...)
    # print('Test loss:', score[0])
    # print('Test accuracy:', score[1])


if __name__ == '__main__':
    main(parse_args())
