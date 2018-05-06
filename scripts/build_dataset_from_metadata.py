#!/usr/bin/env python3.6
import argparse
from itertools import count
import json
from pathlib import Path

import cv2

import colormotion.dataset as dataset


def parse_args():
    parser = argparse.ArgumentParser(description='Build dataset from metadata files.')
    parser.add_argument('--resolution', type=int, nargs=2, default=(256, 256), help='output resolution')
    parser.add_argument('movies', help='movies directory')
    parser.add_argument('metadata', help='metadata directory')
    parser.add_argument('dataset', help='dataset output directory')
    return parser.parse_args()


def write_video_scenes(video, scene_boundaries, dataset_path, resolution):
    if not scene_boundaries:
        return
    scene_boundaries = iter(scene_boundaries)
    destination = dataset.create_video_destination_folder(video, dataset_path)
    video = cv2.VideoCapture(str(video))
    scene_lower, scene_upper = next(scene_boundaries)
    for frame_count in count():
        has_frame, frame = video.read()
        if not has_frame:
            return
        if scene_lower <= frame_count <= scene_upper:
            frame = cv2.resize(frame, (resolution[0], resolution[1]), interpolation=cv2.INTER_AREA)
            scene = dataset.get_scene_directory(destination, scene_lower)
            cv2.imwrite(str(dataset.get_frame_path(scene, frame_count)), frame, [cv2.IMWRITE_PNG_COMPRESSION, 9])
        elif frame_count > scene_upper:
            scene_lower, scene_upper = next(scene_boundaries, (None, None))
            if scene_lower is None:
                return


def build_dataset_from_metadata(movies_path, metadata_path, dataset_path, resolution):
    video_hashes = {dataset.hash_file(movie): movie for movie in movies_path.iterdir()}
    for metadata_file in metadata_path.iterdir():
        with open(metadata_file) as f:
            metadata = json.load(f)
        for video_hash, scene_boundaries in metadata.items():
            if video_hash not in video_hashes:
                raise RuntimeError('Video with hash {} not found'.format(video_hash))
            write_video_scenes(video_hashes[video_hash], scene_boundaries, dataset_path, resolution)


def main(args):
    build_dataset_from_metadata(Path(args.movies), Path(args.metadata), Path(args.dataset), args.resolution)


if __name__ == '__main__':
    main(parse_args())
