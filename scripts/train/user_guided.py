#!/usr/bin/env python3
import random
import sys

from keras.callbacks import ModelCheckpoint
from keras.preprocessing.image import ImageDataGenerator
import numpy as np

from colormotion import dataset
from colormotion.argparse import training_args_parser
from colormotion.nn.generators import VideoFramesGenerator
from colormotion.nn.layers import load_weights
from colormotion.nn.model.user_guided import model
from colormotion.user_guided import ab_and_mask_matrix


class Generator(VideoFramesGenerator):
    '''Generate groups of contiguous frames from a dataset.

    The generated data has inputs [l_input, ab_and_mask_input].'''
    def __init__(self, **kwargs):
        augment = kwargs.pop('augment', False)
        super().__init__(**kwargs)
        self.augmentation = ImageDataGenerator() if augment else None

    def augment(self, x):
        return self.augmentation.apply_transform(x, {
            'theta': random.uniform(-15, 15),
            'tx': random.uniform(-4, 4),
            'ty': random.uniform(-4, 4),
            'shear': random.uniform(-20, 20),
            'zx': random.uniform(.7, 1),
            'zy': random.uniform(.7, 1),
            'flip_horizontal': random.choices((False, True)),
        })

    def load_batch(self, start_frames, target_size):
        assert self.contiguous_count == 1
        x_batch = [[], []]
        y_batch = []
        for scene, frame in start_frames:
            l, ab = dataset.read_frame_lab(scene, frame + self.contiguous_count, target_size)
            if self.augmentation:
                x = np.dstack((l, ab))
                x = self.augment(x)
                l, ab = x[:, :, :1], x[:, :, 1:]
            x_batch[0].append(ab_and_mask_matrix(ab, .00016))
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
    train = Generator(augment=True).flow_from_directory(dataset_folder / 'train', **flow_params)
    test = Generator().flow_from_directory(dataset_folder / 'validation', **flow_params)
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
    main(training_args_parser().parse_args())
