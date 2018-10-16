from collections import OrderedDict
import hashlib
from pathlib import Path

import cv2
import numpy as np

from colormotion.environment import fail


def hash_file(filename):
    '''Hash of the first 32KB of a file.'''
    with open(filename, 'rb') as f:
        return hashlib.blake2b(f.read(32 * 1024), digest_size=20).hexdigest()  # pylint: disable=unexpected-keyword-arg


def create_video_destination_folder(video_filename, root, exist_ok=False):
    root = Path(root)
    if not root.exists():
        fail('Folder {} does not exist'.format(root))
    video_destination = root / hash_file(video_filename)
    if video_destination.exists():
        message = 'Video has already been processed (folder {} exists)'.format(video_destination)
        if not exist_ok:
            fail(message)
        print(message)
    video_destination.mkdir(exist_ok=exist_ok)
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
    scene_directory, frame = Path(*args[:-1]), args[-1]
    if isinstance(frame, int):
        frame = '{:06d}.png'.format(frame)
    return scene_directory / frame


def read_image(filename, color=True, resolution=None):
    image = cv2.imread(str(filename), color and cv2.IMREAD_COLOR or cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise RuntimeError('Cannot read image {}'.format(filename))
    if resolution:
        image = cv2.resize(image, (resolution[0], resolution[1]), interpolation=cv2.INTER_AREA)
    return (image / 255).astype(np.float32)


def bgr_to_lab(image):
    '''Convert BGR image to L*a*b* colorspace.

    Return a tuple (l, ab), where l was mean centered.
    '''
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, ab = lab[:, :, 0:1], lab[:, :, 1:]
    l -= 50  # mean centering
    return l, ab

def read_frame_lab(scene, frame_number, target_size):
    bgr_image = read_image(get_frame_path(scene, frame_number), resolution=target_size)
    return bgr_to_lab(bgr_image)

def _lab_to_bgr(l, ab):
    # Undo mean centering in L channel
    l = l.copy() + 50
    lab = np.dstack((l, ab))
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


def lab_to_bgr(l, ab):
    # Hack to adjust a*b* channels so L*a*b* is in BGR gamut
    for _ in range(3):
        bgr = np.clip(_lab_to_bgr(l, ab), 0, 1)
        new_l, new_ab = bgr_to_lab(bgr)
        error = np.sum(np.abs(new_l - l)) + np.sum(np.abs(new_ab - ab))
        if error < 1:
            break
        ab = new_ab
    return bgr


def get_scenes(movie_directory, names_as_int=True):
    if names_as_int:
        def parse_filename(path):
            assert path.suffix == '.png'
            assert path.stem.isnumeric() and len(path.stem) == 6
            return int(path.stem)
    else:
        def parse_filename(path):
            return path
    scenes = {scene: sorted(parse_filename(frame) for frame in scene.iterdir())
              for scene in Path(movie_directory).iterdir()}
    return OrderedDict(sorted(scenes.items()))


def get_all_scenes(dataset_directory, names_as_int=True):
    scenes = {scene: frames
              for movie in Path(dataset_directory).iterdir()
              for scene, frames in get_scenes(movie, names_as_int=names_as_int).items()}
    return OrderedDict(sorted(scenes.items()))
