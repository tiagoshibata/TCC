#!/usr/bin/env python3
import random
import sys

from keras.callbacks import ModelCheckpoint
import numpy as np

from colormotion import dataset
from colormotion.argparse import training_args_parser
from colormotion.nn.generators import VideoFramesGenerator
from colormotion.nn.layers import load_weights
from colormotion.nn.model.filters_optical_flow import model, warp_features
from colormotion.user_guided import ab_and_mask_matrix


class Generator(VideoFramesGenerator):
    '''Generate groups of contiguous frames from a dataset.

    The generated data has inputs [ab_and_mask_input, l_input, warped_features].'''
    def __init__(self, m, **kwargs):
        self.model = m
        super().__init__(**kwargs)

    def load_batch(self, start_frames, target_size):  # pylint: disable=too-many-locals
        assert self.contiguous_count == 1
        x_batch = [[], [], []]
        y_batch = []
        for scene, frame in start_frames:
            l, ab = dataset.read_frame_lab(scene, frame + self.contiguous_count, target_size)
            l_tm1, ab_tm1 = dataset.read_frame_lab(scene, frame, target_size)
            # TODO implement augmentation
            # TODO generate artificial flow data from ImageNet

            features_tm1 = self.model.predict([
                np.array([ab_and_mask_matrix(ab_tm1, .00016)]),
                np.array([l_tm1]),
                np.empty((1, 32, 32, 512)),
            ])[1]

            warped_features = warp_features(l_tm1, l, features_tm1)

            x_batch[0].append(ab_and_mask_matrix(ab, .00016))
            x_batch[1].append(l)
            x_batch[2].append(warped_features)
            y_batch.append(ab)
        return [np.array(model_input) for model_input in x_batch], np.array(y_batch)


    def load_sample(self, scene, start_frame, target_size):
        pass  # unused


def data_generators(m, dataset_folder):
    flow_params = {
        'batch_size': 4,
        'target_size': (256, 256),
        'seed': random.randrange(sys.maxsize),
    }
    # TODO Split train and test datasets
    train = Generator(m).flow_from_directory(dataset_folder, **flow_params)
    # test = Generator(encoded_features_path, skip_connections_pipe).flow_from_directory(dataset_folder, **flow_params)
    return train, None


def main(args):
    m = model()
    train_generator, _ = data_generators(m, args.dataset)
    checkpoint = ModelCheckpoint('epoch-{epoch:03d}-{loss:.3f}.h5', verbose=1, period=1)
    load_weights(m, args.weights, by_name=True)
    fit = m.fit_generator(
        train_generator,
        steps_per_epoch=args.steps_per_epoch,
        epochs=args.epochs,
        # validation_data=test_generator,
        # validation_steps=args.validation_steps,
        callbacks=[checkpoint])
    m.save('optical_flow_decoder.h5')

    print(fit.history)
    # score = m.evaluate(...)
    # print('Test loss:', score[0])
    # print('Test accuracy:', score[1])


if __name__ == '__main__':
    main(training_args_parser().parse_args())
