#!/usr/bin/env python3
from pathlib import Path
import random


class VideoFramesDataGenerator():
    def __init__(self, data_format='channels_last', rescale=1 / 255, frame_count=2):
        '''Generate groups of contiguous frames from a dataset.

        data_format: see keras.preprocessing.image.ImageDataGenerator
        rescale: see keras.preprocessing.image.ImageDataGenerator
        frame_count: number of frames to yield at each call (default 2)
        '''
        if data_format != 'channels_last':
            raise NotImplementedError()
        self.rescale = rescale
        self.frame_count = frame_count

    def flow_from_directory(self, directory, color_mode='rgb', target_size=None, seed=None):
        if seed is None:
            seed = random.random()
        directory = Path(directory)
        frames = {x: list(x.iterdir()) for x in directory.iterdir()}
        contiguous_frame_count = self._get_contiguous_frames_count(frames)

    def _get_contiguous_frames_count(self, frames):
        raise NotImplementedError()

    def _get_contiguous_at(self, position):
        raise NotImplementedError()
