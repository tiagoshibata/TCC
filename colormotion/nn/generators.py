from random import Random

import numpy as np

from colormotion import dataset


def read_image_lab(scene, frame_number, target_size):
    bgr_image = dataset.read_image(dataset.get_frame_path(scene, frame_number), resolution=target_size)
    return dataset.bgr_to_lab(bgr_image)


class VideoFramesGenerator():
    def __init__(self, data_format='channels_last', contiguous_count=1):
        '''Generate groups of contiguous frames from a dataset.

        data_format: see keras.preprocessing.image.ImageDataGenerator
        contiguous_count: number of previous frames to yield at each call to be used as the model state (default 1)
        '''
        # TODO Support data augmentation, similar to https://keras.io/preprocessing/image/
        if data_format != 'channels_last':
            raise NotImplementedError()
        self.contiguous_count = contiguous_count

    def flow_from_directory(self, root, batch_size=32, target_size=None, seed=None):
        contiguous_frames = self._get_contiguous_frames(dataset.get_all_scenes(root))
        random = Random(seed)
        while True:
            yield self.load_batch(random.choices(contiguous_frames, k=batch_size),
                                  target_size=target_size)

    def load_batch(self, start_frames, target_size):
        samples = [self.load_sample(scene, frame, target_size) for scene, frame in start_frames]
        grayscale = np.array([i[-2] for i in samples])
        y = np.array([i[-1] for i in samples])
        if self.contiguous_count:
            state = np.array([i[0:-2] for i in samples])
            state = state.transpose((1, 0, *range(2, len(state.shape))))  # swap first two channels
            return list(state) + [grayscale], y
        return [grayscale], y

    def load_sample(self, scene, start_frame, target_size):
        raise NotImplementedError()

    def get_contiguous_frames(self, frames):
        # Remove frames at the end of a scene and discard too short scenes
        frames = [(directory, frame)
                  for directory, images in frames.items() if len(images) > self.contiguous_count
                  # Use None (end of list) if contiguous_count == 0
                  for frame in images[:-self.contiguous_count or None]]
        return frames


class VideoFramesWithLabStateGenerator(VideoFramesGenerator):  # pylint: disable=too-few-public-methods
    '''Generate groups of contiguous frames from a dataset, with the L*a*b* channels of previous frames as the state.'''
    def load_sample(self, scene, start_frame, target_size):
        '''Load a sample to build a batch.'''
        # y = expected colorization in last frame
        # state = previous frames colorized and current frame in grayscale
        grayscale, y = read_image_lab(scene, start_frame + self.contiguous_count, target_size)
        state = [
            np.dstack(read_image_lab(scene, start_frame + i, target_size))
            for i in range(self.contiguous_count)
        ]
        return state + [grayscale, y]


class VideoFramesWithLStateGenerator(VideoFramesGenerator):  # pylint: disable=too-few-public-methods
    '''Generate groups of contiguous frames from a dataset, with the L* channel of previous frames as the state.'''
    def load_sample(self, scene, start_frame, target_size):
        '''Load a sample to build a batch.'''
        # y = expected colorization in last frame
        # state = previous frames colorized and current frame in grayscale
        grayscale, y = read_image_lab(scene, start_frame + self.contiguous_count, target_size)
        state = [
            read_image_lab(scene, start_frame + i, target_size)[0]
            for i in range(self.contiguous_count)
        ]
        return state + [grayscale, y]
