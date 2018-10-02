#!/usr/bin/env python3
import random
from multiprocessing import Process, Pipe
import sys

from keras import backend as K
from keras.callbacks import ModelCheckpoint
import numpy as np
import tensorflow as tf

from colormotion import dataset
from colormotion.argparse import directory_path, training_args_parser
from colormotion.nn.generators import VideoFramesGenerator
from colormotion.nn.layers import load_weights
from colormotion.nn.model.filters_optical_flow import interpolate_and_decode, warp_features
from colormotion.nn.model.user_guided import encoder_head_model


def tf_allow_growth():
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.gpu_options.visible_device_list = "0"
    K.set_session(tf.Session(config=config))


def encoded_feature_path(root, scene, frame):
    return dataset.get_frame_path(root,
                                  scene.relative_to(scene.parents[1]),
                                  frame)


def skip_connections_eval_cpu(pipe, weights, target_size):
    tf_allow_growth()

    encoder = encoder_head_model()
    load_weights(encoder, weights, by_name=True)

    while True:
        try:
            start_frames, encoded_features_path = pipe.recv()
        except EOFError:
            break  # end of data from parent process
        l_batch, ab_and_mask_matrix_t_batch = [], []
        for scene, frame in start_frames:
            l, _ = dataset.read_frame_lab(scene, frame + 1, target_size)
            l_batch.append(l)

            ab_and_mask = np.load('{}_encoded_mask.npz'.format(
                encoded_feature_path(encoded_features_path, scene, frame + 1)))['arr_0']
            ab_and_mask_matrix_t_batch.append(ab_and_mask)
        pipe.send(
            encoder.predict([np.array(x) for x in (l_batch, ab_and_mask_matrix_t_batch)])
        )
    pipe.close()


class Generator(VideoFramesGenerator):
    '''Generate groups of contiguous frames from a dataset.

    The generated data has inputs [warped_features, features, conv1_2norm, conv2_2norm, conv3_3norm].'''
    def __init__(self, encoded_features_path, skip_connections_pipe, **kwargs):
        self.encoded_features_path = encoded_features_path
        self.skip_connections_pipe = skip_connections_pipe
        super().__init__(**kwargs)

    def load_batch(self, start_frames, target_size):  # pylint: disable=too-many-locals
        assert self.contiguous_count == 1
        x_batch = [[], []]
        y_batch = []

        self.skip_connections_pipe.send((start_frames, self.encoded_features_path))

        for scene, frame in start_frames:
            features_tm1 = np.load('{}_encoded.npz'.format(
                encoded_feature_path(self.encoded_features_path, scene, frame)))['arr_0']
            features = np.load('{}_encoded.npz'.format(
                encoded_feature_path(self.encoded_features_path, scene, frame + self.contiguous_count)))['arr_0']

            l, ab = dataset.read_frame_lab(scene, frame + self.contiguous_count, target_size)
            l_tm1, _ = dataset.read_frame_lab(scene, frame, target_size)
            warped_features = warp_features(l_tm1, l, features_tm1)

            x_batch[0].append(warped_features)
            x_batch[1].append(features)
            y_batch.append(ab)
        for skip_connection in self.skip_connections_pipe.recv():
            x_batch.append(skip_connection)
        return [np.array(model_input) for model_input in x_batch], np.array(y_batch)

    def load_sample(self, scene, start_frame, target_size):
        pass  # unused


def data_generators(dataset_folder, encoded_features_path, skip_connections_pipe):
    flow_params = {
        'batch_size': 2,
        'target_size': (256, 256),
        'seed': random.randrange(sys.maxsize),
    }
    # TODO Split train and test datasets
    train = Generator(encoded_features_path, skip_connections_pipe).flow_from_directory(dataset_folder, **flow_params)
    # test = Generator(encoded_features_path, skip_connections_pipe).flow_from_directory(dataset_folder, **flow_params)
    return train, None


def main(args):
    assert args.weights, 'This training requires a pre-trained, frozen encoder'

    child_pipe, parent_pipe = Pipe()
    p = Process(target=skip_connections_eval_cpu, args=(child_pipe, args.weights, (256, 256)))
    p.start()

    tf_allow_growth()
    train_generator, _ = data_generators(args.dataset, args.encoded_features_path, parent_pipe)

    checkpoint = ModelCheckpoint('epoch-{epoch:03d}.h5', verbose=1, period=5)
    decoder = interpolate_and_decode()
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
    parser.add_argument('encoded_features_path', type=directory_path, help='directory with encoded features')
    main(parser.parse_args())
