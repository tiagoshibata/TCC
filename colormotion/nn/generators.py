#!/usr/bin/env python3
from random import Random
from types import SimpleNamespace

import colormotion.dataset as dataset


class VideoFramesDataGenerator():  # pylint: disable=too-few-public-methods
    def __init__(self, data_format='channels_last', rescale=1 / 255, contiguous_count=2):
        '''Generate groups of contiguous frames from a dataset.

        data_format: see keras.preprocessing.image.ImageDataGenerator
        rescale: see keras.preprocessing.image.ImageDataGenerator
        contiguous_count: number of frames to yield at each call (default 2)
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
        batch = [
            self._load_sequence(scene, frame, target_size)
            for scene, frame in start_frames
        ]
        return [x[0] for x in batch], [x[1] for x in batch]

    def _load_sequence(self, scene, start_frame, target_size):
        def read_image(scene, frame_number):
            return dataset.read_image(str(dataset.get_frame_path(scene, frame_number)), resolution=target_size)

        # y = expected colorization in final frame
        y = read_image(scene, start_frame + self.contiguous_count - 1)
        # network input (previous frames colorized and current frame in grayscale)
        state = [
            read_image(scene, start_frame + i)
            for i in range(self.contiguous_count - 1)
        ]
        grayscale = dataset.convert_to_grayscale(y)
        # Rescale data
        state = [self.rescale * i for i in state]
        grayscale *= self.rescale
        y *= self.rescale
        return {'state_input': state, 'grayscale_input': grayscale}, y

    def _get_contiguous_frames(self, frames):
        # Remove frames at the end of a scene and discard too short scenes
        frames = [(directory, frame)
                  for directory, images in frames.items() if len(images) >= self.contiguous_count
                  for frame in images[:-self.contiguous_count + 1]]
        return frames
