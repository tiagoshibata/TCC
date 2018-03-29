from keras import backend as backend
from keras.layers import Input
import pytest

from colormotion.nn.layers import Crop


@pytest.mark.parametrize('crop_args,expected_shape', [
    (
        (1, 0, 16),
        (None, 16, 32, 4),
    ), (
        (2, 16, 33),
        (None, 32, 16, 4),
    ), (
        (3, 0, 2),
        (None, 32, 32, 2),
    ),
])
def test_crop(crop_args, expected_shape):
    x = Input(shape=(32, 32, 4), name="input")
    assert backend.int_shape(Crop(*crop_args)(x)) == expected_shape
