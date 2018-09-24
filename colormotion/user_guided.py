import numpy as np


def binary_mask(size, ones_ratio):
    return np.random.choice((0, 1), size=size, p=[1 - ones_ratio, ones_ratio])


def apply_mask(data, ones_ratio):
    mask = binary_mask(data.shape[-2:], ones_ratio)
    return mask, mask * data


def ab_and_mask_matrix(data, ones_ratio):
    assert len(data.shape) == 3  # CHW
    mask, data = apply_mask(data, ones_ratio)
    # Convert to HWC
    mask, data = mask[..., np.newaxis], np.moveaxis(data, 0, -1)
    return np.dstack((mask, data))
