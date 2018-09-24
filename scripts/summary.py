#!/usr/bin/env python3
import argparse
from keras.models import load_model


def parse_args():
    parser = argparse.ArgumentParser(description='Print a model summary.')
    parser.add_argument('model')
    return parser.parse_args()


def summary(path):
    m = load_model(path)
    m.summary()

if __name__ == '__main__':
    summary(parse_args().model)
