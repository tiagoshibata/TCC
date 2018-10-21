from keras import backend as K
import numpy as np

from colormotion.nn.model.filters_optical_flow import interpolate, interpolate_and_decode


def test_compiles():
    interpolate_and_decode()


def test_interpolate_multiply_elementwise_depthwise():
    interpolation = interpolate(K.ones((8, 8, 16)), K.zeros((8, 8, 16)), K.expand_dims(K.eye(8)))
    assert (K.eval(interpolation) == np.repeat(np.eye(8)[..., np.newaxis], 16, axis=-1)).all()
