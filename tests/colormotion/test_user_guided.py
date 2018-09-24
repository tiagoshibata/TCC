from unittest.mock import MagicMock, patch

import numpy as np

from colormotion.user_guided import ab_and_mask_matrix, apply_mask, binary_mask


def test_binary_mask():
    assert (binary_mask((128, 128), 0) == 0).all()
    assert (binary_mask((128, 128), 1) == 1).all()
    mask = binary_mask((128, 128), .5)
    assert .3 < np.count_nonzero(mask) / (128 * 128) < .7
    assert np.logical_or(mask == 0, mask == 1).all()


def test_apply_mask():
    data = np.arange(32).reshape((4, 8))
    mask, masked_data = apply_mask(data, .5)
    assert (masked_data[mask == 0] == 0).all()
    assert (masked_data[mask == 1] == data[mask == 1]).all()

    # Should work with multidimensional arrays
    data = np.arange(64).reshape((2, 4, 8))
    mask, masked_data = apply_mask(data, .5)
    assert mask.shape == (4, 8)
    assert masked_data.shape == data.shape
    tiled_mask = np.tile(mask, (2, 1, 1))
    assert (masked_data[tiled_mask == 0] == 0).all()
    assert (masked_data[tiled_mask == 1] == data[tiled_mask == 1]).all()


@patch('colormotion.user_guided.apply_mask')
def test_ab_and_mask_matrix(mock_apply_mask):
    mask, masked_data = np.empty((64, 128)), np.empty((2, 64, 128))
    mock_apply_mask.return_value = mask, masked_data
    data = np.empty((2, 64, 128))
    m = ab_and_mask_matrix(data, .5)

    mock_apply_mask.assert_called_once_with(data, .5)
    assert m.shape == (64, 128, 3)
    assert (m[:, :, :2] == np.moveaxis(masked_data, 0, -1)).all()
    assert (m[:, :, -1] == mask).all()
