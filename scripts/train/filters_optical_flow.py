#!/usr/bin/env python3
import random
import sys

from keras.callbacks import ModelCheckpoint
import numpy as np

from colormotion import dataset
from colormotion.argparse import training_args_parser
from colormotion.nn.generators import VideoFramesGenerator
from colormotion.nn.graph import new_model_session
from colormotion.nn.layers import load_weights
from colormotion.nn.model.filters_optical_flow import interpolate_and_decode, warp_features
from colormotion.nn.model.user_guided import encoder_model
from colormotion.user_guided import ab_and_mask_matrix


class Generator(VideoFramesGenerator):
    '''Generate groups of contiguous frames from a dataset.

    The generated data has inputs [warped_features, features, conv1_2norm, conv2_2norm, conv3_3norm].'''
    def __init__(self, encoder, session, **kwargs):
        self.session = session
        self.encoder = encoder
        super().__init__(**kwargs)

    def load_batch(self, start_frames, target_size):  # pylint: disable=too-many-locals
        assert self.contiguous_count == 1
        x_batch = [[], [], [], [], []]
        y_batch = []
        for scene, frame in start_frames:
            l, ab = dataset.read_frame_lab(scene, frame + self.contiguous_count, target_size)
            l_tm1, _ = dataset.read_frame_lab(scene, frame, target_size)

            ab_and_mask_matrix_t = ab_and_mask_matrix(ab, .00008)
            ab_and_mask_matrix_tm1 = ab_and_mask_matrix(ab, .00004)

            from keras import backend as K  # FIXME
            K.clear_session()  # FIXME
            with self.session:
                features_tm1, _, _, _ = self.encoder.predict([np.array([x]) for x in (l_tm1, ab_and_mask_matrix_tm1)])
            features, conv1_2norm, conv2_2norm, conv3_3norm = self.encoder.predict(
                [np.array([x]) for x in (ab_and_mask_matrix_t, l)])
            warped_features = warp_features(l_tm1, l, features_tm1)

            x_batch[0].append(warped_features)
            x_batch[1].append(features)
            x_batch[2].append(conv1_2norm)
            x_batch[3].append(conv2_2norm)
            x_batch[4].append(conv3_3norm)
            y_batch.append(ab)
        return [np.array(model_input) for model_input in x_batch], np.array(y_batch)

    def load_sample(self, scene, start_frame, target_size):
        pass  # unused


def data_generators(dataset_folder, nn_model, session):
    flow_params = {
        'batch_size': 8,
        'target_size': (256, 256),
        'seed': random.randrange(sys.maxsize),
    }
    # TODO Split train and test datasets
    train = Generator(nn_model, session).flow_from_directory(dataset_folder, **flow_params)
    test = Generator(nn_model, session).flow_from_directory(dataset_folder, **flow_params)
    return train, test


def main(args):
    assert args.weights, 'This training requires a pre-trained, frozen encoder'
    with new_model_session() as encoder_session:
        encoder = encoder_model()
        load_weights(encoder, args.weights, by_name=True)
    train_generator, test_generator = data_generators(args.dataset, encoder, encoder_session)

    checkpoint = ModelCheckpoint('epoch-{epoch:03d}-{val_loss:.3f}.h5', verbose=1, period=5)
    with new_model_session():
        decoder = interpolate_and_decode()
        load_weights(decoder, args.weights, by_name=True)
        fit = decoder.fit_generator(
            train_generator,
            steps_per_epoch=args.steps_per_epoch,
            epochs=args.epochs,
            validation_data=test_generator,
            validation_steps=args.validation_steps,
            callbacks=[checkpoint])
        decoder.save('optical_flow_decoder.h5')
    print(fit.history)
    # score = m.evaluate(...)
    # print('Test loss:', score[0])
    # print('Test accuracy:', score[1])


if __name__ == '__main__':
    main(training_args_parser().parse_args())
