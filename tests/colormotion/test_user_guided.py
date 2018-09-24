import numpy as np

from colormotion.user_guided import apply_mask, binary_mask


def test_binary_mask():
    assert (binary_mask((128, 128), 0) == 0).all()
    assert (binary_mask((128, 128), 1) == 1).all()
    mask = binary_mask((128, 128), .5)
    assert .3 < np.count_nonzero(mask) / (128 * 128) < .7
    assert np.logical_or(mask == 0, mask == 1).all()


def test_apply_mask():
    data = np.arange(32).reshape((8, 4))
    mask, masked_data = apply_mask(data, .5)
    assert (masked_data[mask == 0] == 0).all()
    assert (masked_data[mask == 1] == data[mask == 1]).all()
