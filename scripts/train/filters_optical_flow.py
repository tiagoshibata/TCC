#!/usr/bin/env python3
import argparse
from pathlib import Path
import random
import sys

from keras.callbacks import ModelCheckpoint

from colormotion.argparse import directory_path
from colormotion.nn.generators import VideoFramesWithLabStateGenerator
from colormotion.nn.layers import load_weights
from colormotion.nn.model.filters_optical_flow import model, model_encoder


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
    train = VideoFramesWithLabStateGenerator().flow_from_directory(dataset_folder, **flow_params)
    test = VideoFramesWithLabStateGenerator().flow_from_directory(dataset_folder, **flow_params)
    return train, test


# def frames_to_model_input(m, generator):
#     '''Convert a pair of frames to the model inputs.'''
#     # The model has inputs [l_input, l_input_tm1, features_tm1] and outputs [x, features]
#     encoder = model_encoder()
#     while True:
#         state, y = next(generator)
#         l_tm1, l = state
#         features_tm1 = encoder.predict(l_tm1)
#         yield (l, l_tm1, features_tm1), y



def main(args):
    # m = Model(inputs=[l_input, l_input_tm1, features_tm1], outputs=[x, features])
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
    m.save('optical_flow_model.h5')
    # score = m.evaluate(...)
    # print('Test loss:', score[0])
    # print('Test accuracy:', score[1])


if __name__ == '__main__':
    main(parse_args())
