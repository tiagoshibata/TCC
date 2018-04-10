#!/usr/bin/env python3
from random import Random

import numpy as np

import colormotion.dataset as dataset


class VideoFramesDataGenerator():  # pylint: disable=too-few-public-methods
    def __init__(self, data_format='channels_last', rescale=1 / 255, contiguous_count=1):
        '''Generate groups of contiguous frames from a dataset.

        data_format: see keras.preprocessing.image.ImageDataGenerator
        rescale: see keras.preprocessing.image.ImageDataGenerator
        contiguous_count: number of previous frames to yield at each call (default 1)
        '''
        # TODO Support data augmentation, similar to https://keras.io/preprocessing/image/
        if data_format != 'channels_last':
            raise NotImplementedError()
        self.rescale = rescale
        self.contiguous_count = contiguous_count

    def flow_from_directory(self, root, batch_size=32, target_size=None, seed=None):
        contiguous_frames = self._get_contiguous_frames(dataset.get_frames(root))
        random = Random(seed)
        while True:
            yield self._load_batch(random.choices(contiguous_frames, k=batch_size),
                                   target_size=target_size)

    def _load_batch(self, start_frames, target_size):
        samples = [self._load_sample(scene, frame, target_size) for scene, frame in start_frames]
        state = np.array([i[0:-2] for i in samples]).transpose((1, 0, 2, 3, 4))
        grayscale = np.array([i[-2] for i in samples])
        y = np.array([i[-1] for i in samples])
        return list(state) + [grayscale], y

    def _load_sample(self, scene, start_frame, target_size):
        '''Load a sample to build a batch.'''
        def read_image(scene, frame_number):
            return dataset.read_image(str(dataset.get_frame_path(scene, frame_number)), resolution=target_size)

        # y = expected colorization in last frame
        # x = previous frames colorized and current frame in grayscale
        last_frame = read_image(scene, start_frame + self.contiguous_count)
        grayscale, y = dataset.to_l_ab(last_frame)
        grayscale, y = self.rescale * grayscale, self.rescale * y
        state = [
            self.rescale * read_image(scene, start_frame + i)  # FIXME call to_l_ab on the image
            for i in range(self.contiguous_count)
        ]
        return state + [grayscale, y]

    def _get_contiguous_frames(self, frames):
        # Remove frames at the end of a scene and discard too short scenes
        frames = [(directory, frame)
                  for directory, images in frames.items() if len(images) > self.contiguous_count
                  for frame in images[:-self.contiguous_count]]
        return frames
