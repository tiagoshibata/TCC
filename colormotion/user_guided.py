import numpy as np


def binary_mask(size, ones_ratio):
    return np.random.choice((0, 1), size=size, p=[1 - ones_ratio, ones_ratio])


def apply_mask(data, ones_ratio):
    '''Return a (mask, masked_data) tuple.

    data must be in HWC format.'''
    assert len(data.shape) == 3 and data.shape[2] == 2  # HWC
    mask = binary_mask(data.shape[:2], ones_ratio)[..., np.newaxis]
    return mask * data, mask


def ab_and_mask_matrix(data, ones_ratio):
    return np.dstack(apply_mask(data, ones_ratio))
