#!/usr/bin/env python3
import random
from multiprocessing import Process, Pipe
from pathlib import Path
import sys

from keras import backend as K
from keras.callbacks import ModelCheckpoint
import numpy as np
import tensorflow as tf

from colormotion import dataset
from colormotion.argparse import training_args_parser
from colormotion.nn.generators import VideoFramesGenerator
from colormotion.nn.layers import load_weights
from colormotion.nn.model.filters_optical_flow import interpolate_and_decode, warp_features
from colormotion.nn.model.user_guided import encoder_model
from colormotion.user_guided import ab_and_mask_matrix


def tf_allow_growth():
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.gpu_options.visible_device_list = '0'
    K.set_session(tf.Session(config=config))


def encoder_eval(pipe, weights, target_size):
    while True:
        tf_allow_growth()
        try:
            start_frames = pipe.recv()
        except EOFError:
            break  # end of data from parent process
        import time
        time.sleep(9)
        # inputs: l_input, ab_and_mask_input
        # outputs: encoded_features, conv1_2norm, conv2_2norm, conv3_3norm
        encoder = encoder_model()
        load_weights(encoder, weights, by_name=True)
        # decoder inputs: warped_features, features, conv1_2norm, conv2_2norm, conv3_3norm
        # decoder outputs: x
        x_batch = [[], [], [], [], []]
        y_batch = []
        for scene, frame in start_frames:
            l, ab = dataset.read_frame_lab(scene, frame + 1, target_size)
            l_tm1, ab_tm1 = dataset.read_frame_lab(scene, frame, target_size)
            # TODO implement augmentation
            # TODO generate artificial flow data from ImageNet
            # TODO call predict in batches
            features_tm1 = encoder.predict([
                np.array([l_tm1]),
                np.array([ab_and_mask_matrix(ab_tm1, .00016)]),
            ])[0]
            warped_features = warp_features(l_tm1, l, features_tm1[0])
            features, conv1_2norm, conv2_2norm, conv3_3norm = encoder.predict([
                np.array([l]),
                np.array([ab_and_mask_matrix(ab, .00016)]),
            ])

            x_batch[0].append(warped_features)
            x_batch[1].append(features[0])
            x_batch[2].append(conv1_2norm[0])
            x_batch[3].append(conv2_2norm[0])
            x_batch[4].append(conv3_3norm[0])
            y_batch.append(ab)
        K.clear_session()
        del encoder
        pipe.send(
            ([np.array(model_input) for model_input in x_batch], np.array(y_batch))
        )
    pipe.close()


class Generator(VideoFramesGenerator):
    '''Generate groups of contiguous frames from a dataset.

    The generated data has inputs [warped_features, features, conv1_2norm, conv2_2norm, conv3_3norm].'''
    def __init__(self, encoder_pipe, **kwargs):
        self.encoder_pipe = encoder_pipe
        super().__init__(**kwargs)

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
        'batch_size': 4,
        'target_size': (256, 256),
        'seed': random.randrange(sys.maxsize),
    }
    # TODO Split train and test datasets
    train = Generator(encoder_pipe).flow_from_directory(dataset_folder, **flow_params)
    # test = Generator(encoder_pipe).flow_from_directory(dataset_folder, **flow_params)
    return train, None


def main(args):
    child_pipe, parent_pipe = Pipe()
    p = Process(target=encoder_eval, args=(child_pipe, args.encoder_weights, (256, 256)))
    p.start()

    tf_allow_growth()
    train_generator, _ = data_generators(args.dataset, parent_pipe)

    checkpoint = ModelCheckpoint('epoch-{epoch:03d}-{loss:.3f}.h5', verbose=1, period=1)
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

    parent_pipe.close()
    p.join()
    print(fit.history)
    # score = m.evaluate(...)
    # print('Test loss:', score[0])
    # print('Test accuracy:', score[1])


if __name__ == '__main__':
    parser = training_args_parser()
    parser.add_argument('--encoder_weights', type=Path, required=True, help='encoder weights')
    main(parser.parse_args())
