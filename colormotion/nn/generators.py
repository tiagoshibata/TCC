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
        x = [[]] * (self.contiguous_count + 1)
        y = []
        for scene, frame in start_frames:
            sample = self._load_sample(scene, frame, target_size)
            x = [state + [sample_frame] for sample_frame, state in zip(sample[:-1], x)]
            y.append(sample[-1])
        return [np.array(i) for i in x], np.array(y)

    def _load_sample(self, scene, start_frame, target_size):
        '''Load a sample to build a batch.'''
        def read_image(scene, frame_number):
            return dataset.read_image(str(dataset.get_frame_path(scene, frame_number)), resolution=target_size)

        # y = expected colorization in final frame
        y = read_image(scene, start_frame + self.contiguous_count)
        # network input (previous frames colorized and current frame in grayscale)
        state = [
            self.rescale * read_image(scene, start_frame + i)
            for i in range(self.contiguous_count)
        ]
        grayscale = self.rescale * dataset.convert_to_grayscale(y)
        y = self.rescale * y
        return state + [grayscale, y]

    def _get_contiguous_frames(self, frames):
        # Remove frames at the end of a scene and discard too short scenes
        frames = [(directory, frame)
                  for directory, images in frames.items() if len(images) > self.contiguous_count
                  for frame in images[:-self.contiguous_count]]
        return frames
