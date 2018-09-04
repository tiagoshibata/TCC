#!/usr/bin/env python3
import argparse
from itertools import count
import json

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
    destination = dataset.create_video_destination_folder(video, dataset_path)
    video = cv2.VideoCapture(str(video))
    scene_lower, scene_upper = next(scene_boundaries)

    def write(path, frame):
        frame = cv2.resize(frame, (resolution[0], resolution[1]), interpolation=cv2.INTER_AREA)
        cv2.imwrite(path, frame, [cv2.IMWRITE_PNG_COMPRESSION, 9])

    with ConsumerPool(lambda args: write(*args)) as write_consumer_pool:
        for frame_count in count():
            has_frame, frame = video.read()
            if not has_frame:
                return
            if scene_lower <= frame_count <= scene_upper:
                scene = dataset.get_scene_directory(destination, scene_lower)
                write_consumer_pool.put((
                    str(dataset.get_frame_path(scene, frame_count)),
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
                    raise RuntimeError('Video with hash {} not found'.format(video_hash))
                print('Writing scenes {} from {}'.format(scene_boundaries, video_path))
                scene_consumer_pool.put((video_path, scene_boundaries, dataset_path, resolution))


def main(args):
    build_dataset_from_metadata(args.movies, args.metadata, args.dataset, args.resolution)


if __name__ == '__main__':
    main(parse_args())
