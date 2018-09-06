'''Recurrent warp unit based on optical flow.

Some concepts are taken from https://arxiv.org/abs/1703.09211.
'''
import keras.backend as K
from keras.layers import Add, AveragePooling2D, BatchNormalization, Conv2D, Input, Lambda, Multiply, Subtract
from keras.models import Model

from colormotion.optical_flow import numerical_optical_flow, warp


def Conv2D_default(filters, **kwargs):  # pylint: disable=invalid-name
    '''Conv2D block with most commonly used options.'''
    return Conv2D(filters, 3, padding='same', activation='relu', **kwargs)


def to_layer(f):
    return Lambda(lambda args: f(*(K.eval(x) for x in args)))


def mask_network(difference):
    x = Conv2D_default(16, name='mask_conv1')(difference)
    x = Conv2D_default(32, name='mask_conv2')(x)
    return Conv2D_default(1, name='mask_conv3')(x)


def warp_features(l_input_tm1, l_input, features_tm1):
    # TODO Replace with a optical flow network and optimize globally with the rest of the model
    # TODO Reuse destination for performance
    print(l_input_tm1, l_input, features_tm1)
    flow = numerical_optical_flow(l_input_tm1, l_input)
    return warp(features_tm1, flow)


def model():  # pylint: disable=too-many-statements,too-many-locals
    l_input = Input(shape=(256, 256, 1), name='grayscale_input')
    l_input_tm1 = Input(shape=(256, 256, 1), name='grayscale_input_tm1')
    features_tm1 = Input(shape=(256, 256, 1), name='features_tm1')


    def Downscale():  # pylint: disable=invalid-name
        return AveragePooling2D(pool_size=1, strides=2)

    # conv1
    # conv1_1
    x = Conv2D_default(64, name='conv1_1')(l_input)
    # conv1_2
    x = Conv2D_default(64, name='conv1_2')(x)
    # conv1_2norm = BatchNormalization(name='conv1_2norm')(x)
    x = Downscale()(x)

    # conv2
    # conv2_1
    x = Conv2D_default(128, name='conv2_1')(x)
    # conv2_2
    x = Conv2D_default(128, name='conv2_2')(x)
    # conv2_2norm = BatchNormalization(name='conv2_2norm')(x)
    x = Downscale()(x)

    # conv3
    # conv3_1
    x = Conv2D_default(256, name='conv3_1')(x)
    # conv3_2
    x = Conv2D_default(256, name='conv3_2')(x)
    # conv3_3
    x = Conv2D_default(256, name='conv3_3')(x)
    # conv3_3_norm = BatchNormalization(name='conv3_3norm')(x)
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

    warped_features = to_layer(warp_features)([l_input_tm1, l_input, features_tm1])
    assert K.int_shape(features) == K.int_shape(warped_features)
    difference = Subtract()([features, warped_features])
    # Composition mask
    mask = mask_network(difference)
    # Interpolate warped features and encoded features
    warped_features = Multiply()([warped_features, mask])
    ones_minus_mask = Subtract()([K.ones(K.shape(mask)), mask])
    features = Multiply()([features, ones_minus_mask])
    features = Add()([warped_features, features])

    assert K.int_shape(x) == K.int_shape(features)
    # TODO Decoder

    m = Model(inputs=[l_input, l_input_tm1, features_tm1], outputs=[x, features])
    # TODO Another loss function might be more appropriate
    m.compile(loss='mean_squared_error',
              optimizer='adam')
    return m
