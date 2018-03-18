#!/usr/bin/env python3
import random

import colormotion.dataset as dataset


class VideoFramesDataGenerator():
    def __init__(self, data_format='channels_last', rescale=1 / 255, contiguous_frames=2):
        '''Generate groups of contiguous frames from a dataset.

        data_format: see keras.preprocessing.image.ImageDataGenerator
        rescale: see keras.preprocessing.image.ImageDataGenerator
        contiguous_frames: number of frames to yield at each call (default 2)
        '''
        if data_format != 'channels_last':
            raise NotImplementedError()
        self.rescale = rescale
        self.contiguous_frames = contiguous_frames

    def flow_from_directory(self, root, target_size=None, seed=None):
        if seed is None:
            seed = random.random()
        frames = dataset.get_frames(root)
        contiguous_frames = self._get_contiguous_frames(frames)
        while True:
            random.choices(contiguous_frames, k=self.batch_size)
            yield random.choices(contiguous_frames, k=self.batch_size)

    def _get_contiguous_frames(self, frames):
        # Remove too short scenes
        frames = [(directory, frame)
                  for directory, images in frames.items() if len(images) >= self.contiguous_frames
                  for frame in images[:-self.contiguous_frames + 1]]
        return frames
