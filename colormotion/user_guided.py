import numpy as np


def binary_mask(size, ones_ratio):
    return np.random.choice((0, 1), size=size, p=[1 - ones_ratio, ones_ratio])


def apply_mask(data, ones_ratio):
    mask = binary_mask(data.shape, ones_ratio)
    return mask, mask * data
