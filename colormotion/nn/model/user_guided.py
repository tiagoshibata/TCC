import keras.backend as K
from keras.initializers import RandomNormal
from keras.layers import Activation, Add, AveragePooling2D, BatchNormalization, Conv2D, Conv2DTranspose, Input
from keras.layers.advanced_activations import LeakyReLU
from keras.models import Model

from colormotion.nn.layers import Scale


def Conv2D_default(filters, **kwargs):  # pylint: disable=invalid-name
    '''Conv2D block with most commonly used options.'''
    return Conv2D(filters, 3, padding='same', activation='relu', **kwargs)


def Downscale():  # pylint: disable=invalid-name
    return AveragePooling2D(pool_size=1, strides=2)


def encoder(l_input, ab_and_mask_input):
    # conv1
    # conv1_1
    ab_and_mask = Conv2D(64, 3, padding='same', name='ab_conv1_1')(ab_and_mask_input)
    x = Conv2D(64, 3, padding='same', name='bw_conv1_1')(l_input)
    x = Add()([ab_and_mask, x])
    x = Activation('relu')(x)
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
    x = BatchNormalization(name='conv6_3norm')(x)

    # conv7
    # conv7_1
    x = Conv2D_default(512, name='conv7_1')(x)
    # conv7_2
    x = Conv2D_default(512, name='conv7_2')(x)
    # conv7_3
    x = Conv2D_default(512, name='conv7_3')(x)
    return (BatchNormalization(name='conv7_3norm')(x), conv1_2norm, conv2_2norm, conv3_3_norm)


def decoder(features, conv1_2norm, conv2_2norm, conv3_3_norm):
    # Shortcuts, transpose convolutions and some convolutions use a custom initializer
    custom_initializer = {
        'kernel_initializer': RandomNormal(stddev=.01),
        'bias_initializer': 'ones',
    }

    # conv8
    # conv8_1
    x = Conv2DTranspose(256, 4, padding='same', strides=2, name='conv8_1')(features)
    # Shortcut
    shortcut = Conv2D(256, 3, padding='same', **custom_initializer, name='conv3_3_short')(conv3_3_norm)
    x = Add()([x, shortcut])
    x = Activation('relu')(x)
    # conv8_2
    x = Conv2D_default(256, name='conv8_2')(x)
    # conv8_3
    x = Conv2D_default(256, name='conv8_3')(x)
    x = BatchNormalization(name='conv8_3norm')(x)

    # conv9
    # conv9_1
    x = Conv2DTranspose(128, 4, padding='same', strides=2, **custom_initializer, name='conv9_1')(x)
    # Shortcut
    shortcut = Conv2D(128, 3, padding='same', **custom_initializer, name='conv2_2_short')(conv2_2norm)
    x = Add()([x, shortcut])
    x = Activation('relu')(x)
    # conv9_2
    x = Conv2D_default(128, **custom_initializer, name='conv9_2')(x)
    x = BatchNormalization(name='conv9_2norm')(x)

    # conv10
    # conv10_1
    x = Conv2DTranspose(128, 4, padding='same', strides=2, **custom_initializer, name='conv10_1')(x)
    # Shortcut
    shortcut = Conv2D(128, 3, padding='same', **custom_initializer, name='conv1_2_short')(conv1_2norm)
    x = Add()([x, shortcut])
    x = Activation('relu')(x)
    # conv10_2
    x = Conv2D(128, 3, padding='same', **custom_initializer, name='conv10_2')(x)
    x = LeakyReLU(alpha=.2)(x)
    # conv10_ab
    x = Conv2D(2, 1, activation='tanh', name='conv10_ab')(x)
    return Scale(100)(x)  # FIXME Scale uses a Lambda layer, preprocessing the output during training is probably faster


def model():
    l_input = Input(shape=(256, 256, 1), name='grayscale_input')
    ab_and_mask_input = Input(shape=(256, 256, 3), name='ab_and_mask_input')
    encoded_features, conv1_2norm, conv2_2norm, conv3_3_norm = encoder(l_input, ab_and_mask_input)
    print('Encoded features have shape {}'.format(K.int_shape(encoded_features)))
    x = decoder(encoded_features, conv1_2norm, conv2_2norm, conv3_3_norm)
    m = Model(inputs=[ab_and_mask_input, l_input],
              outputs=x)
    m.compile(loss='mean_squared_error', optimizer='adam')
    return m
