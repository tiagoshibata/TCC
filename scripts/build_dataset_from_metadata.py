#!/usr/bin/env python3
import argparse
from itertools import count
import json
import random
import sys

import cv2

from colormotion.argparse import directory_path
from colormotion import dataset
from colormotion.threading import ConsumerPool


def parse_args():
    parser = argparse.ArgumentParser(description='Build dataset from metadata files.')
    parser.add_argument('--resolution', type=int, nargs=2, default=(256, 256), help='output resolution')
    parser.add_argument('movies', type=directory_path, help='movies directory')
    parser.add_argument('metadata', type=directory_path, help='metadata directory')
    parser.add_argument('dataset', type=directory_path, help='dataset output directory')
    return parser.parse_args()


def write_video_scenes(video, scene_boundaries, dataset_path, resolution):
    if not scene_boundaries:
        return
    scene_boundaries = iter(scene_boundaries)
    destination = dataset.create_video_destination_folder(video, dataset_path, exist_ok=True)
    video = cv2.VideoCapture(str(video))
    scene_lower, scene_upper = next(scene_boundaries)

    def write(path, frame):
        if path.exists():
            return
        frame = cv2.resize(frame, (resolution[0], resolution[1]), interpolation=cv2.INTER_AREA)
        cv2.imwrite(str(path), frame, [cv2.IMWRITE_PNG_COMPRESSION, 9])

    with ConsumerPool(lambda args: write(*args)) as write_consumer_pool:
        for frame_count in count():
            has_frame, frame = video.read()
            if not has_frame:
                return
            if scene_lower <= frame_count <= scene_upper:
                scene = dataset.get_scene_directory(destination, scene_lower)
                write_consumer_pool.put((
                    dataset.get_frame_path(scene, frame_count),
                    frame,
                ))
            elif frame_count > scene_upper:
                scene_lower, scene_upper = next(scene_boundaries, (None, None))
                if scene_lower is None:
                    return


def build_dataset_from_metadata(movies_path, metadata_path, dataset_path, resolution):
    video_hashes = {dataset.hash_file(movie): movie for movie in movies_path.iterdir()}
    with ConsumerPool(lambda args: write_video_scenes(*args)) as scene_consumer_pool:
        for metadata_file in metadata_path.iterdir():
            print('Processing metadata file {}'.format(metadata_file))
            with open(metadata_file) as f:
                metadata = json.load(f)
            for video_hash, scene_boundaries in metadata.items():
                video_path = video_hashes.get(video_hash)
                if not video_path:
                    print('ERROR: Video with hash {} not found'.format(video_hash), file=sys.stderr)
                    continue
                print('Writing scenes {} from {}'.format(scene_boundaries, video_path))
                scene_consumer_pool.put((video_path, scene_boundaries, dataset_path, resolution))


def train_validation_split(dataset_path):
    train_path = dataset_path / 'train'
    validation_path = dataset_path / 'validation'

    movie_scenes = {movie: list(movie.iterdir())
                    for movie in train_path.iterdir()}
    scene_list = []
    for movie, scenes in movie_scenes.items():
        scene_list.extend(((movie, scene) for scene in scenes))
    print('{} scenes in dataset, splitting into train/validation'.format(len(scene_list)))

    random.shuffle(scene_list)
    validation = scene_list[:len(scene_list) // 5]
    for movie, scene in validation:
        movie_path = validation_path / movie.stem
        movie_path.mkdir(exist_ok=True)
        (scene).rename(movie_path / scene.stem)


def main(args):
    build_dataset_from_metadata(args.movies, args.metadata, args.dataset, args.resolution)
    (args.dataset / 'train').mkdir(exist_ok=True)
    (args.dataset / 'validation').mkdir(exist_ok=True)
    train_validation_split(args.dataset)


if __name__ == '__main__':
    main(parse_args())
