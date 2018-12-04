#!/usr/bin/env python3
import argparse
from keras.models import load_model


def parse_args():
    parser = argparse.ArgumentParser(description='Print a model summary.')
    parser.add_argument('--plot', action='store_true', help='plot model')
    parser.add_argument('model')
    return parser.parse_args()


def summary(args):
    m = load_model(args.path)
    m.summary()
    if args.plot:
        from keras.utils import plot_model
        plot_model(m, to_file='model.png')

if __name__ == '__main__':
    summary(parse_args())
