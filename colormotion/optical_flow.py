import cv2
import numpy as np


def _adjust_range(image):
    # To quickly decide whether we should adjust the range, we assume integer data types are in 0-255 range, and
    # floating point data types are normalized. Our optical flow method uses images in the 0-255 range, so normalized
    # images must be multiplied by 255.
    if np.issubdtype(image.dtype, np.integer):
        return image
    return image * 255


def numerical_optical_flow(previous, current, destination=None):
    # TODO OPTFLOW_USE_INITIAL_FLOW could enhance the results
    # This method is slow and single threaded. It should be run in parallel on multiple frames to speed it up.
    return cv2.calcOpticalFlowFarneback(_adjust_range(previous), _adjust_range(current),
                                        destination, 0.5, 3, 15, 3, 5, 1.2, 0)


def warp(features, flow, destination=None):
    '''Warp features using a flow matrix.

    Return a warped feature.
    '''
    # Based on: https://stackoverflow.com/questions/17459584/opencv-warping-image-based-on-calcopticalflowfarneback
    # and https://github.com/opencv/opencv/issues/11068
    h, w = flow.shape[:2]
    flow = -flow
    flow[:, :, 0] += np.arange(w)
    flow[:, :, 1] += np.arange(h)[:, np.newaxis]
    if np.issubdtype(features.dtype, np.integer):
        # Don't pass integer features, else the output will be rounded to the nearest int
        features = features.astype(np.float32)
    # TODO Test with border modes other than cv2.BORDER_REPLICATE
    return cv2.remap(features, flow, None, cv2.INTER_LINEAR, destination, cv2.BORDER_REPLICATE)
