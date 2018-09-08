'''Recurrent warp unit based on optical flow.

Some concepts are taken from https://arxiv.org/abs/1703.09211.
'''
import keras.backend as K
from keras.initializers import RandomNormal
from keras.layers import (Activation, Add, AveragePooling2D, BatchNormalization, Conv2D, Conv2DTranspose, Input, Lambda,
                           Multiply, Subtract)
from keras.layers.advanced_activations import LeakyReLU
from keras.losses import mean_squared_error
from keras.models import Model

from colormotion.optical_flow import numerical_optical_flow, warp
from colormotion.nn.layers import numpy_layer, Scale


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


def warp_features_placeholder(l_input_tm1, l_input, features_tm1):
    input_shape = K.int_shape(l_input)
    assert K.int_shape(l_input_tm1) == input_shape
    assert len(input_shape) == 4 and input_shape[0] is None

    features_shape = K.int_shape(features_tm1)
    assert len(features_shape) == 4 and features_shape[0] is None
    return K.placeholder(shape=features_shape)


def model():  # pylint: disable=too-many-statements,too-many-locals
    l_input = Input(shape=(256, 256, 1), name='grayscale_input')
    l_input_tm1 = Input(shape=(256, 256, 1), name='grayscale_input_tm1')


    def Downscale():  # pylint: disable=invalid-name
        return AveragePooling2D(pool_size=1, strides=2)

    # conv1
    # conv1_1
    x = Conv2D_default(64, name='conv1_1')(l_input)
    # conv1_2
    x = Conv2D_default(64, name='conv1_2')(x)
    conv1_2norm = BatchNormalization(name='conv1_2norm')(x)
    x = Downscale()(x)  # TODO Test downscaling the normalized values

    # conv2
    # conv2_1
    x = Conv2D_default(128, name='conv2_1')(x)
    # conv2_2
    x = Conv2D_default(128, name='conv2_2')(x)
    conv2_2norm = BatchNormalization(name='conv2_2norm')(x)
    x = Downscale()(x)

    # conv3
    # conv3_1
    x = Conv2D_default(256, name='conv3_1')(x)
    # conv3_2
    x = Conv2D_default(256, name='conv3_2')(x)
    # conv3_3
    x = Conv2D_default(256, name='conv3_3')(x)
    conv3_3_norm = BatchNormalization(name='conv3_3norm')(x)
    x = Downscale()(x)

    # conv4
    # conv4_1
    x = Conv2D_default(512, name='conv4_1')(x)
    # conv4_2
    x = Conv2D_default(512, name='conv4_2')(x)
    # conv4_3
    x = Conv2D_default(512, name='conv4_3')(x)
    x = BatchNormalization(name='conv4_3norm')(x)

    # conv5
    # conv5_1
    x = Conv2D_default(512, dilation_rate=2, name='conv5_1')(x)
    # conv5_2
    x = Conv2D_default(512, dilation_rate=2, name='conv5_2')(x)
    # conv5_3
    x = Conv2D_default(512, dilation_rate=2, name='conv5_3')(x)
    x = BatchNormalization(name='conv5_3norm')(x)

    # conv6
    # conv6_1
    x = Conv2D_default(512, dilation_rate=2, name='conv6_1')(x)
    # conv6_2
    x = Conv2D_default(512, dilation_rate=2, name='conv6_2')(x)
    # conv6_3
    x = Conv2D_default(512, dilation_rate=2, name='conv6_3')(x)
    features = BatchNormalization(name='conv6_3norm')(x)
    features_shape = K.int_shape(features)[1:]
    print('Encoded features have shape {}'.format(features_shape))
    features_tm1 = Input(shape=features_shape, name='features_tm1')

    warped_features = numpy_layer(warp_features, warp_features_placeholder)([l_input_tm1, l_input, features_tm1])
    assert K.int_shape(features) == K.int_shape(warped_features)
    difference = Subtract()([features, warped_features])
    # Composition mask
    mask = mask_network(difference)
    # Interpolate warped features and encoded features
    warped_features = Multiply()([warped_features, mask])
    ones_minus_mask = Subtract()([Lambda(K.ones_like)(mask), mask])
    features = Multiply()([features, ones_minus_mask])
    features = Add()([warped_features, features])

    # Shortcuts, transpose convolutions and some convolutions use a custom initializer
    custom_initializer = {
        'kernel_initializer': RandomNormal(stddev=.01),
        'bias_initializer': 'ones',
    }

    # conv7
    # conv7_1
    x = Conv2DTranspose(256, 4, padding='same', strides=2, name='conv7_1')(x)
    # Shortcut
    shortcut = Conv2D(256, 3, padding='same', **custom_initializer, name='conv3_3_short')(conv3_3_norm)
    x = Add()([x, shortcut])
    x = Activation('relu')(x)
    # conv7_2
    x = Conv2D_default(256, name='conv7_2')(x)
    # conv7_3
    x = Conv2D_default(256, name='conv7_3')(x)
    x = BatchNormalization(name='conv7_3norm')(x)

    # conv8
    # conv8_1
    x = Conv2DTranspose(128, 4, padding='same', strides=2, **custom_initializer, name='conv8_1')(x)
    # Shortcut
    shortcut = Conv2D(128, 3, padding='same', **custom_initializer, name='conv2_2_short')(conv2_2norm)
    x = Add()([x, shortcut])
    x = Activation('relu')(x)
    # conv8_2
    x = Conv2D_default(128, **custom_initializer, name='conv8_2')(x)
    x = BatchNormalization(name='conv8_2norm')(x)

    # conv9
    # conv9_1
    x = Conv2DTranspose(128, 4, padding='same', strides=2, **custom_initializer, name='conv9_1')(x)
    # Shortcut
    shortcut = Conv2D(128, 3, padding='same', **custom_initializer, name='conv1_2_short')(conv1_2norm)
    x = Add()([x, shortcut])
    x = Activation('relu')(x)
    # conv9_2
    x = Conv2D(128, 3, padding='same', **custom_initializer, name='conv9_2')(x)
    x = LeakyReLU(alpha=.2)(x)
    # conv9_ab
    x = Conv2D(2, 1, activation='tanh', name='conv9_ab')(x)
    x = Scale(100)(x)  # FIXME Scale uses a Lambda layer, preprocessing the output during training is probably faster

    m = Model(inputs=[l_input, l_input_tm1, features_tm1], outputs=[x, features])
    m.compile(loss=lambda y_true, y_pred: mean_squared_error(y_true, y_pred[0]),
              optimizer='adam')
    return m
