from keras.layers import Activation, BatchNormalization, Conv2D
from keras.models import Sequential

from colormotion.nn.layers import Scale


def model():
    '''Colorful implementation of ab inference in L*a*b* colorspace'''
    m = Sequential()

    def conv_block(filter_count, convolution_parameters):
        # Series of Conv2D blocks with ReLU activation, followed by batch normalization
        for parameters in convolution_parameters:
            conv2d_params = {
                'kernel_size': 3,
                'padding': 'same',
                'activation': 'relu',
                **parameters,
            }
            m.add(Conv2D(filter_count, **conv2d_params))
        m.add(BatchNormalization())

    # conv1
    # TODO resize input or make input_shape configurable
    conv_block(64, [
        {'input_shape': (224, 224, 1)},
        {'strides': 2},
    ])

    # conv2
    conv_block(128, [
        {},
        {'strides': 2},
    ])

    # conv3
    conv_block(256, [
        {},
        {},
        {'strides': 2},
    ])

    # conv4 (dilation = 1)
    conv_block(512, [{}] * 3)

    # conv5 (dilation = 2)
    conv_block(512, [{'dilation_rate': 2}] * 3)

    # conv6 (dilation = 2)
    conv_block(512, [{'dilation_rate': 2}] * 3)

    # conv7
    conv_block(512, [{}] * 3)

    # conv8
    conv_block(256, [
        {'kernel_size': 4, 'strides': 2},
        {},
        {},
    ])

    # Softmax
    m.add(Conv2D(313, 1, padding='same'))
    m.add(Scale(2.606))
    m.add(Activation('softmax'))

    # Decoding
    # TODO Implement class rebalancing, implement annealed-mean interpolation
    m.add(Conv2D(2, 1, padding='same'))  # FIXME Placeholder, should be an annealed-mean interpolation
    # TODO Implement loss function (cross entropy with soft-encoded ground truth)
    m.compile(loss='mean_squared_error',
              optimizer='adam',
              metrics=['accuracy'])
    return m
