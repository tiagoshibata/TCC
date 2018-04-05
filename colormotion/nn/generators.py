#!/usr/bin/env python3
import random

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
        random_generator = random.Random(seed)
        frames = dataset.get_frames(root)
        contiguous_frames = self._get_contiguous_frames(frames)
        while True:
            yield self._load_batch(random_generator.choices(contiguous_frames, k=batch_size),
                                   target_size=target_size)

    def _load_batch(self, start_frames, target_size):
        if len(start_frames) != 1:
            raise NotImplementedError()
        batch = [
            self._load_sequence(scene, frame, target_size)
            for scene, frame in start_frames
        ]
        return [batch[0, 0]], [batch[0, 1]]

    def _load_sequence(self, scene, start_frame, target_size):
        def load_image(scene, frame_number):
            return dataset.load_image(str(dataset.get_frame_path(scene, frame_number)), resolution=target_size)

        # y = expected colorization in final frame
        y = load_image(scene, start_frame + self.contiguous_count - 1)
        # x = network input (previous frames colorized + current frame in grayscale)
        x = [
            load_image(scene, start_frame + i)
            for i in range(self.contiguous_count - 1)
        ]
        x.append(dataset.convert_to_grayscale(y))
        return x, y

    def _get_contiguous_frames(self, frames):
        # Remove frames at the end of a scene and discard too short scenes
        frames = [(directory, frame)
                  for directory, images in frames.items() if len(images) >= self.contiguous_count
                  for frame in images[:-self.contiguous_count + 1]]
        return frames
