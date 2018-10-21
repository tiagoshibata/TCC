#!/usr/bin/env python3
import random
from multiprocessing import Process, Pipe
from pathlib import Path
import sys

from keras import backend as K
from keras.callbacks import ModelCheckpoint
from keras.preprocessing.image import ImageDataGenerator
import numpy as np
import tensorflow as tf

from colormotion import dataset
from colormotion.argparse import training_args_parser
from colormotion.nn.generators import VideoFramesGenerator
from colormotion.nn.layers import load_weights
from colormotion.nn.model.filters_optical_flow import interpolate_and_decode, warp_features
from colormotion.nn.model.user_guided import encoder_model
from colormotion.user_guided import ab_and_mask_matrix


def tf_allow_growth(memory_fraction=.4):
    config = tf.ConfigProto()
    # config.gpu_options.allow_growth = True
    config.gpu_options.per_process_gpu_memory_fraction = memory_fraction
    config.gpu_options.visible_device_list = '0'
    K.set_session(tf.Session(config=config))


def random_augmentation():
    return {
        'theta': random.uniform(-15, 15),
        'tx': random.uniform(-4, 4),
        'ty': random.uniform(-4, 4),
        'shear': random.uniform(-20, 20),
        'zx': random.uniform(.7, 1),
        'zy': random.uniform(.7, 1),
        'flip_horizontal': random.choices((False, True)),
    }


def small_flow_transform():
    return {
        'theta': random.uniform(-5, 5),
        'tx': random.uniform(-4, 4),
        'ty': random.uniform(-4, 4),
        'shear': random.uniform(-10, 10),
        'zx': random.uniform(9, 1),
        'zy': random.uniform(9, 1),
    }


def augment_l_ab(image_data_generator, l_ab, transform):
    x = np.dstack(l_ab)
    x = image_data_generator.apply_transform(x, transform)
    return x[:, :, :1], x[:, :, 1:]


def encoder_eval(pipe, weights, target_size):  # pylint: disable=too-many-locals
    tf_allow_growth()
    image_data_generator = ImageDataGenerator()
    # inputs: l_input, ab_and_mask_input
    # outputs: encoded_features, conv1_2norm, conv2_2norm, conv3_3norm
    encoder = encoder_model()
    load_weights(encoder, weights, by_name=True)
    while True:
        try:
            start_frames = pipe.recv()
        except EOFError:
            pipe.close()
            return  # end of data from parent process
        # decoder inputs: warped_features, features, conv1_2norm, conv2_2norm, conv3_3norm
        # decoder outputs: x
        y_batch = []
        l_batch = []
        l_tm1_batch = []
        ab_mask_batch = []
        ab_mask_tm1_batch = []
        for scene, frame in start_frames:
            transform = random_augmentation()
            l_tm1, ab_tm1 = augment_l_ab(image_data_generator,
                                         dataset.read_frame_lab(scene, frame, target_size), transform)
            if isinstance(frame, int):
                l, ab = augment_l_ab(image_data_generator,
                                     dataset.read_frame_lab(scene, frame + 1, target_size), transform)
            else:
                # Augment artificially
                l, ab = augment_l_ab(image_data_generator, (l_tm1, ab_tm1), small_flow_transform())

            l_tm1_batch.append(l_tm1)
            ab_mask_tm1_batch.append(ab_and_mask_matrix(ab_tm1, .00016))
            l_batch.append(l)
            ab_mask_batch.append(ab_and_mask_matrix(ab, .00008))
            y_batch.append(ab)

        features_tm1 = encoder.predict([
            np.array(l_tm1_batch),
            np.array(ab_mask_tm1_batch),
        ])[0]
        warped_features = [warp_features(l_tm1, l, feature_tm1)
                           for l_tm1, l, feature_tm1 in zip(l_tm1_batch, l_batch, features_tm1)]
        features, conv1_2norm, conv2_2norm, conv3_3norm = encoder.predict([
            np.array(l_batch),
            np.array(ab_mask_batch),
        ])
        pipe.send(
            ([np.array(warped_features), features, conv1_2norm, conv2_2norm, conv3_3norm],
             np.array(y_batch))
        )


class Generator(VideoFramesGenerator):
    '''Generate groups of contiguous frames from a dataset.

    The generated data has inputs [warped_features, features, conv1_2norm, conv2_2norm, conv3_3norm].'''
    def __init__(self, encoder_pipe, **kwargs):
        self.encoder_pipe = encoder_pipe
        super().__init__(**kwargs)

    def flow_from_directory(self, root, batch_size=32, target_size=None):
        contiguous_frames = self.get_contiguous_frames(dataset.get_all_scenes(root, names_as_int=False))
        print('Dataset {} has {} contiguous subscenes'.format(root, len(contiguous_frames)))
        while True:
            yield self.load_batch(random.choices(contiguous_frames, k=batch_size),
                                  target_size=target_size)

    def load_batch(self, start_frames, target_size):
        assert self.contiguous_count == 1
        # decoder inputs: warped_features, features, conv1_2norm, conv2_2norm, conv3_3norm
        # decoder outputs: x
        self.encoder_pipe.send(start_frames)
        return self.encoder_pipe.recv()

    def load_sample(self, scene, start_frame, target_size):
        pass  # unused


def data_generators(dataset_folder, encoder_pipe):
    flow_params = {
        'batch_size': 6,
        'target_size': (256, 256),
    }
    # TODO Split train and test datasets
    train = Generator(encoder_pipe).flow_from_directory(dataset_folder / 'train', **flow_params)
    # test = Generator(encoder_pipe).flow_from_directory(dataset_folder / 'validation', **flow_params)
    return train, None


def train_decoder(parent_pipe, args):
    tf_allow_growth(.5)
    train_generator, _ = data_generators(args.dataset, parent_pipe)
    checkpoint = ModelCheckpoint('epoch-{epoch:03d}-{loss:.3f}.h5', verbose=1, period=5)
    decoder = interpolate_and_decode()
    if args.weights:
        load_weights(decoder, args.weights, by_name=True)
    fit = decoder.fit_generator(
        train_generator,
        steps_per_epoch=args.steps_per_epoch,
        epochs=args.epochs,
        # validation_data=test_generator,
        # validation_steps=args.validation_steps,
        callbacks=[checkpoint])
    decoder.save('optical_flow_decoder.h5')
    print(fit.history)
    # score = m.evaluate(...)
    # print('Test loss:', score[0])
    # print('Test accuracy:', score[1])
    parent_pipe.close()


def main(args):
    child_pipe, parent_pipe = Pipe()
    encoder_process = Process(target=encoder_eval, args=(child_pipe, args.encoder_weights, (256, 256)))
    encoder_process.start()

    train_decoder(parent_pipe, args)
    encoder_process.join()


if __name__ == '__main__':
    parser = training_args_parser()
    parser.add_argument('--encoder_weights', type=Path, required=True, help='encoder weights')
    main(parser.parse_args())
