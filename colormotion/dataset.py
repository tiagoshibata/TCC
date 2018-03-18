#!/usr/bin/env python3
import hashlib
from pathlib import Path

import cv2

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
    directory = Path(root) / '{:06d}'.format(scene_number)
    directory.mkdir(exist_ok=True)
    return directory


def get_frame_path(scene_directory, frame_number):
    return Path(scene_directory) / '{:06d}.png'.format(frame_number)


def load_image(filename, color=True, resolution=None):
    image = cv2.imread(filename, color and cv2.IMREAD_COLOR or cv2.IMREAD_GRAYSCALE)
    if resolution:
        image = cv2.resize(image, (resolution[0], resolution[1]), interpolation=cv2.INTER_AREA)
    return image


def get_frames(root):
    return {scene: sorted(int(frame.stem) for frame in scene.iterdir())
            for scene in Path(root).iterdir()}
