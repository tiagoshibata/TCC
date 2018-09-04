from keras import backend
from keras.layers import Input
from keras.models import Sequential
import numpy as np
import pytest

from colormotion.nn.layers import Crop, Scale


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
def test_Crop(crop_args, expected_shape):
    x = Input(shape=(32, 32, 4))
    assert backend.int_shape(Crop(*crop_args)(x)) == expected_shape


def test_Scale():
    m = Sequential()
    m.add(Scale(10, input_shape=(2, 2, 3)))
    data = np.array(np.arange(12).reshape((2, 2, 3)), ndmin=4)
    assert (m.predict(data, batch_size=1) == np.array([
        [
            [[0, 10, 20], [30, 40, 50]],
            [[60, 70, 80], [90, 100, 110]],
        ]
    ])).all()
