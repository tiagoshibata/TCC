'''Recurrent warp unit based on optical flow.

Some concepts are taken from https://arxiv.org/abs/1703.09211.
'''
import keras.backend as K
from keras.layers import Add, Conv2D, Input, Lambda, Multiply, Subtract
from keras.models import Model

from colormotion.optical_flow import numerical_optical_flow, warp
from colormotion.nn.model import user_guided


def Conv2D_default(filters, **kwargs):  # pylint: disable=invalid-name
    '''Conv2D block with most commonly used options.'''
    return Conv2D(filters, 3, padding='same', activation='relu', **kwargs)


def mask_network(difference):
    # TODO Scale so that range is approx. 0-1
    x = Conv2D_default(16, name='mask_conv1')(difference)
    x = Conv2D_default(32, name='mask_conv2')(x)
    return Conv2D_default(1, name='mask_conv3')(x)


def warp_features(l_input_tm1, l_input, features_tm1):
    # TODO Replace with a optical flow network and optimize globally with the rest of the model
    # TODO Reuse destination for performance
    flow = numerical_optical_flow(l_input_tm1, l_input)
    return warp(features_tm1, flow)


def interpolate(previous, new, mask):
    previous = Multiply(name='interpolate_mul_previous')([previous, mask])
    ones_minus_mask = Subtract(name='interpolate_one_minus_mask')([Lambda(K.ones_like)(mask), mask])
    new = Multiply(name='interpolate_mul_new')([new, ones_minus_mask])
    return Add(name='interpolate_add')([previous, new])


def feature_interpolation_network():
    raise NotImplementedError()


def interpolate_and_decode():
    warped_features = Input(shape=(32, 32, 512), name='warped_features')
    features = Input(shape=(32, 32, 512), name='features')
    conv1_2norm = Input(shape=(256, 256, 64), name='conv1_2norm_input')
    conv2_2norm = Input(shape=(128, 128, 128), name='conv2_2norm_input')
    conv3_3norm = Input(shape=(64, 64, 256), name='conv3_3norm_input')

    difference = Subtract()([features, warped_features])
    # Composition mask
    mask = mask_network(difference)
    # Interpolate warped features and encoded features
    interpolated_features = interpolate(warped_features, features, mask)
    x = user_guided.decoder(interpolated_features, conv1_2norm, conv2_2norm, conv3_3norm)

    m = Model(inputs=[warped_features, features, conv1_2norm, conv2_2norm, conv3_3norm],
              outputs=x)
    m.compile(loss='mean_squared_error', optimizer='adam')
    return m
