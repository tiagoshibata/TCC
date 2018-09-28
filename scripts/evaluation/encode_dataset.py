#!/usr/bin/env python3
import argparse
from pathlib import Path
import time

import numpy as np

from colormotion import dataset
from colormotion.argparse import directory_path
from colormotion.nn.layers import load_weights
from colormotion.nn.model.user_guided import encoder_model
from colormotion.threading import ConsumerPool
from colormotion.user_guided import ab_and_mask_matrix



def parse_args():
    parser = argparse.ArgumentParser(description='Encode dataset.')
    parser.add_argument('--resolution', type=int, nargs=2, default=(256, 256), help='output resolution')
    parser.add_argument('--weights', type=Path, required=True, help='weights file')
    parser.add_argument('dataset', type=directory_path, help='dataset directory')
    parser.add_argument('encoded', type=directory_path, help='output directory for encoded features')
    return parser.parse_args()


class InferenceConsumerPool(ConsumerPool):
    def __init__(self, weights, batch_size=64):
        self.weights = weights
        self.batch_size = batch_size
        super().__init__(None, num_workers=1, queue_size=8)

    def thread(self):
        encoder = encoder_model()
        load_weights(encoder, self.weights, by_name=True)
        with ConsumerPool(lambda args: np.savez_compressed(*args), queue_size=16) as save_consumer_pool:
            while True:
                l, ab_and_mask, filenames = [], [], []
                for _ in range(self.batch_size):
                    job = self.queue.get()
                    if job is None:
                        break
                    lab, encoded_filename = job
                    l.append(lab[0])
                    ab_and_mask.append(ab_and_mask_matrix(lab[1], .00008))
                    filenames.append(encoded_filename)
                    self.queue.task_done()
                if l:
                    print('Encoding batch of size {}'.format(len(l)))
                    start = time.time()
                    encoded_batch, _, _, _ = encoder.predict([np.array(x) for x in (l, ab_and_mask)])
                    print('Encoded in {}'.format(time.time() - start))
                    for filename, encoded_features in zip(filenames, encoded_batch):
                        save_consumer_pool.put((filename, encoded_features))


def main(args):
    scenes = dataset.get_all_scenes(args.dataset)
    encoded_total = 0
    with InferenceConsumerPool(args.weights) as consume_pool:
        try:
            for scene, frames in scenes.items():
                for frame in frames:
                    frame_path = dataset.get_frame_path(scene, frame)
                    encoded_path = args.encoded / '{}_encoded'.format(frame_path.relative_to(args.dataset))
                    encoded_path.parents[1].mkdir(exist_ok=True)
                    encoded_path.parent.mkdir(exist_ok=True)
                    consume_pool.put((
                        dataset.read_frame_lab(scene, frame, args.resolution),
                        encoded_path,
                    ))
                encoded_total += len(frames)
                print('Total encoded: {}'.format(encoded_total))
        finally:
            consume_pool.put(None)

if __name__ == '__main__':
    main(parse_args())
