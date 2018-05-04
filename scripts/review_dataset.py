#!/usr/bin/env python3
import argparse
from pathlib import Path
import subprocess
from tempfile import TemporaryDirectory

import colormotion.dataset as dataset


def parse_args():
    parser = argparse.ArgumentParser(description='Review existing dataset.')
    parser.add_argument('movie', help='movie directory')
    parser.add_argument('--skip', type=int, default=0, help='number of frames to skip from review')
    parser.add_argument('--viewer', default='xdg-open', help='directory viewer')
    return parser.parse_args()


def rmdir_if_empty(directory):
    try:
        directory.rmdir()
        return True
    except OSError as e:
        if e.errno == 39:
            return False
        raise


def merge_prompt(*directories, viewer='xdg-open'):
    directories = [str(x) for x in directories]  # convert from pathlib.Path
    subprocess.check_call([viewer, *directories])
    while True:
        answer = input('Merge frames? ').lower()
        if not answer or answer[0] == 'n':
            return False
        if answer[0] == 'y':
            return True


def merge_short_scenes(movie_directory, scene_frames, viewer):
    # Fast camera movement will generate false negatives in SSIM, which splits the same scene into multiple small
    # scenes. Here, the user will be prompted whether small scenes should be merged.
    if len(scene_frames) < 2:
        return
    with TemporaryDirectory() as temporary_directory:
        temporary_directory = Path(temporary_directory)
        scene = scene_frames[0]
        for next_scene in scene_frames[1:]:
            if len(scene[1]) < 3 or len(next_scene[1]) < 3:
                for frame in [*scene[0].iterdir(), *next_scene[0].iterdir()]:
                    preview = temporary_directory / frame.name
                    if not preview.exists():
                        preview.symlink_to(frame.resolve())
            else:
                scene = next_scene
                frames = sorted(temporary_directory.iterdir())
                if not frames:
                    continue
                if merge_prompt(temporary_directory, viewer=viewer):
                    merge_to_scene = dataset.get_scene_directory(movie_directory, frames[0].stem)
                    for frame in frames:
                        if (merge_to_scene / frame.name).exists():
                            continue
                        new_scene = dataset.get_scene_directory(movie_directory, frame.stem, mkdir=False)
                        if new_scene.exists():
                            merge_from_scene = new_scene
                        (merge_from_scene / frame.name).rename(merge_to_scene / frame.name)
                for frame in frames:
                    frame.unlink()


def review_dataset(movie_directory, viewer, frames_to_skip=0):
    scene_frames = [
        (scene, frames)
        for scene, frames in dataset.get_scenes(movie_directory).items()
        if not rmdir_if_empty(scene) and int(scene.name) >= frames_to_skip
    ]

    merge_short_scenes(movie_directory, scene_frames, viewer)

    scenes = [x[0] for x in scene_frames]
    if len(scenes) >= 2:
        scene = scenes[0]
        for next_scene in scenes[1:]:
            if merge_prompt(scene, next_scene, viewer=viewer):
                for frame in next_scene.iterdir():
                    frame.rename(scene / frame.name)
                next_scene.rmdir()
            else:
                scene = next_scene


def main(args):
    review_dataset(Path(args.movie), args.viewer, frames_to_skip=args.skip)


if __name__ == '__main__':
    main(parse_args())
