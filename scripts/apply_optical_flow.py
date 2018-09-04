#!/usr/bin/env python3
import argparse

import cv2
import numpy as np

from colormotion.optical_flow import optical_flow, warp


def parse_args():
    parser = argparse.ArgumentParser(description="Apply optical flow to a video's frames.")
    parser.add_argument('source', help='source video')
    return parser.parse_args()


def apply_optical_flow(capture):
    def get_frame():
        valid, frame = capture.read()
        return frame if valid else None
    frame = get_frame()
    if frame is None:
        return
    previous = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    flow = np.empty((*frame.shape[:2], 2), dtype=np.float32)
    frame = get_frame()
    while frame is not None:
        current = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        optical_flow(previous, current, destination=flow)
        warp(previous, flow, destination=previous)
        cv2.imshow('Warped', previous)
        cv2.waitKey(1)
        frame = get_frame()


def main(args):
    capture = cv2.VideoCapture(args.source)
    apply_optical_flow(capture)
    capture.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main(parse_args())
