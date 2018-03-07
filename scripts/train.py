#!/usr/bin/env python3
import argparse

from keras.preprocessing.image import ImageDataGenerator

from colormotion.model import model


def parse_args():
    parser = argparse.ArgumentParser(description='Run training.')
    parser.add_argument('dataset', help='dataset folder')
    return parser.parse_args()


def data_generators(dataset_folder):
    params = {
        'data_format': 'channels_last',
        'rescale': 1 / 255,
    }
    flow_params = {
        'batch_size': 32,
        'class_mode': None,
        'target_size': (720, 1280),
        'seed': 14,
    }
    # TODO Split train and validation datasets
    # FIXME color_mode='grayscale' calls Pillow.Image.convert('L'),
    # which uses the ITU-R 601-2 luma transform, but L*a*b* seems more suitable.
    # See: https://github.com/keras-team/keras/blob/master/keras/preprocessing/image.py
    # http://pillow.readthedocs.io/en/3.1.x/reference/Image.html#PIL.Image.Image.convert
    train_params = {
        'shear_range': 0.2,
        'horizontal_flip': True,
        'zoom_range': 0.2,
    }
    train = ImageDataGenerator(
        **params,
        **train_params
    ).flow_from_directory(
        dataset_folder,
        color_mode='grayscale',
        **flow_params)

    train_ground_truth = ImageDataGenerator(
        **params,
        **train_params
    ).flow_from_directory(
        dataset_folder,
        **flow_params)

    test = ImageDataGenerator(
        **params
    ).flow_from_directory(
        dataset_folder,
        color_mode='grayscale',
        **flow_params)

    test_ground_truth = ImageDataGenerator(
        **params
    ).flow_from_directory(
        dataset_folder,
        **flow_params)

    return zip(train, train_ground_truth), zip(test, test_ground_truth)


def main(args):
    train_generator, test_generator = data_generators(args.dataset)
    m = model((720, 1280, 1))  # (height, width, channels)
    m.fit_generator(
        train_generator,
        steps_per_epoch=2000,
        epochs=50,
        validation_data=test_generator,
        validation_steps=800)
    # score = m.evaluate(...)
    # print('Test loss:', score[0])
    # print('Test accuracy:', score[1])


if __name__ == '__main__':
    main(parse_args())
