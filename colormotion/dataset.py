#!/usr/bin/env python3
import hashlib
from pathlib import Path

import cv2
import numpy as np
import skimage.color

from colormotion.environment import fail


def hash_file(filename):
    digest = hashlib.blake2b(digest_size=20)
    with open(filename, 'rb') as f:
        for chunk in iter(lambda: f.read(8192), b''):
            digest.update(chunk)
    return digest.hexdigest()


def create_video_destination_folder(video_filename, root):
    root = Path(root)
    if not root.exists():
        fail('Folder {} does not exist'.format(root))
    video_destination = root / hash_file(video_filename)
    if video_destination.exists():
        fail('Video has already been processed (folder {} exists)'.format(video_destination))
    video_destination.mkdir()
    return video_destination


def get_scene_directory(root, scene_number):
    # TODO Split train and validation datasets
    directory = Path(root) / '{:06d}'.format(scene_number)
    directory.mkdir(exist_ok=True)
    return directory


def get_frame_path(*args):
    scene_directory, frame_number = Path(*args[:-1]), args[-1]
    return scene_directory / '{:06d}.png'.format(frame_number)


def read_image(filename, color=True, resolution=None):
    image = cv2.imread(filename, color and cv2.IMREAD_COLOR or cv2.IMREAD_GRAYSCALE)
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
    l.reshape(*l.shape, 1)
    return l, ab


def lab_to_bgr(l, ab):
    l += 50  # mean centering
    lab = np.dstack((l, ab)).astype(np.float64)
    return skimage.color.lab2rgb(lab)[:, :, ::-1]


def get_frames(root):
    return {scene: sorted(int(frame.stem) for frame in scene.iterdir())
            for movie in Path(root).iterdir()
            for scene in movie.iterdir()}
