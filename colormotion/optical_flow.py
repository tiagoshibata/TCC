import cv2
import numpy as np


def optical_flow(previous, current):
    return cv2.calcOpticalFlowFarneback(previous, current, None, 0.5, 3, 15, 3, 5, 1.2, 0)


def warp(features, flow):
    # Based on: https://stackoverflow.com/questions/16235955/create-a-multichannel-zeros-mat-in-python-with-cv2
    # and https://github.com/opencv/opencv/issues/11068
    h, w = flow.shape[:2]
    flow = -flow
    flow[:, :, 0] += np.arange(w)
    flow[:, :, 1] += np.arange(h)[:, np.newaxis]
    return cv2.remap(features, flow, None, cv2.INTER_LINEAR)
