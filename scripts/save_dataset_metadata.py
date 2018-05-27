#!/usr/bin/env python3
import argparse
from collections import defaultdict
import json

from colormotion.argparse import directory_path
from colormotion.dataset import get_all_scenes


def parse_args():
    parser = argparse.ArgumentParser(description='Save metadata containing scene start/end frames.')
    parser.add_argument('dataset', type=directory_path, help='dataset folder')
    parser.add_argument('metadata', help='output file')
    return parser.parse_args()


def build_metadata(dataset):
    '''Generate metadata from a dataset.

    Return a dictionary containing video IDs and a list of start/end frames of all of its scenes.
    '''
    metadata = defaultdict(list)
    for scene_path, frames in dataset.items():
        video_id = scene_path.parent.name
        metadata[video_id].append((frames[0], frames[-1]))
    metadata = {k: sorted(v) for k, v in metadata.items()}
    return metadata


def main(args):
    dataset = get_all_scenes(args.dataset)
    metadata = build_metadata(dataset)
    with open(args.metadata, 'w') as f:
        json.dump(metadata, f)


if __name__ == '__main__':
    main(parse_args())
