from collections import OrderedDict
import hashlib
from pathlib import Path

import cv2
import numpy as np
import skimage.color

from colormotion.environment import fail


def hash_file(filename):
    '''Hash of the first 32KB of a file.'''
    with open(filename, 'rb') as f:
        return hashlib.blake2b(f.read(32 * 1024), digest_size=20).hexdigest()


def create_video_destination_folder(video_filename, root):
    root = Path(root)
    if not root.exists():
        fail('Folder {} does not exist'.format(root))
    video_destination = root / hash_file(video_filename)
    if video_destination.exists():
        fail('Video has already been processed (folder {} exists)'.format(video_destination))
    video_destination.mkdir()
    return video_destination


def get_scene_directory(root, scene, mkdir=True):
    # TODO Split train and validation datasets
    if isinstance(scene, int):
        scene = '{:06d}'.format(scene)
    directory = Path(root, scene)
    if mkdir:
        directory.mkdir(exist_ok=True)
    return directory


def get_frame_path(*args):
    scene_directory, frame_number = Path(*args[:-1]), args[-1]
    return scene_directory / '{:06d}.png'.format(frame_number)


def read_image(filename, color=True, resolution=None):
    image = cv2.imread(str(filename), color and cv2.IMREAD_COLOR or cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise RuntimeError('Cannot read image {}'.format(filename))
    if resolution:
        image = cv2.resize(image, (resolution[0], resolution[1]), interpolation=cv2.INTER_AREA)
    return (image / 255).astype(np.float32)


def to_lab(image):
    '''Convert BGR image to L*a*b* colorspace.

    Return a tuple (l, ab), where l was mean centered.
    '''
    lab = skimage.color.rgb2lab(image[:, :, ::-1]).astype(np.float32)
    l, ab = lab[:, :, 0], lab[:, :, 1:]
    l -= 50  # mean centering
    # Reshape to enforce three dimensions, even if last one has a single element (required by Keras)
    l = l.reshape(*l.shape, 1)
    return l, ab


def lab_to_bgr(l, ab):
    # Undo mean centering in L channel
    l = l.copy() + 50
    # Check if L*a*b* value is valid, and adjust b channel if not
    b = ab[:, :, 1:]
    # Check in CIEXYZ colorspace (based on scikit's lab2xyz)
    y = (l + 16.) / 116.
    z = y - (b / 200.)
    invalid = z < 0
    if invalid.any():
        print('Some L*a*b* values are invalid')
        invalid = np.nonzero(invalid)
        # TODO Adjust b[invalid]
    # Optionally, OpenCV's implementation (in C) could be used. It might be faster:
    # lab = np.dstack((l, ab)).astype(np.float32)
    # return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    lab = np.dstack((l, ab)).astype(np.float64)
    return skimage.color.lab2rgb(lab)[:, :, ::-1]


def get_scenes(movie_directory):
    scenes = {scene: sorted(int(frame.stem) for frame in scene.iterdir())
              for scene in Path(movie_directory).iterdir()}
    return OrderedDict(sorted(scenes.items()))


def get_all_scenes(dataset_directory):
    scenes = {scene: frames
              for movie in Path(dataset_directory).iterdir()
              for scene, frames in get_scenes(movie).items()}
    return OrderedDict(sorted(scenes.items()))
