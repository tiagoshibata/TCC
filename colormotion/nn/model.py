#!/usr/bin/env python3
from keras.layers import Activation, Add, BatchNormalization, Conv2D, Conv2DTranspose, Input, Lambda
from keras.models import Model, Sequential

from colormotion.nn.layers import Scale


def previous_frame_input():
    '''Model receiving the previous frame and current L value as input.

    Based on Real-Time User-Guided Image Colorization with Learned Deep Priors (R. Zhang et al).
    '''
    state_input = Input(shape=(256, 256, 3), name='state_input')  # state from previous frame
    l_input = Input(shape=(256, 256, 1), name='l_input')

    # conv1
    # conv1_1
    state = Conv2D(64, 3, padding='same')(state_input)
    x = Conv2D(64, 3, padding='same')(l_input)
    x = Add()([state, x])
    x = Activation('relu')(x)
    # conv1_2
    x = Conv2D(64, 3, padding='same', activation='relu')(x)
    conv1_2norm = BatchNormalization()(x)
    x = Conv2D(64, 1, strides=2, use_bias=False, kernel_initializer='ones')(conv1_2norm)
    # FIXME Apply following parameters (from Caffe protobuf description):
    # layer {
    #   param {lr_mult: 0 decay_mult: 0} # UNARY_BRANCH_PROPAGATE
    #   convolution_param {
    #     group: 64
    #   }
    # }

    # conv2
    # conv2_1
    x = Conv2D(128, 3, padding='same', activation='relu')(x)
    # conv2_2
    x = Conv2D(128, 3, padding='same', activation='relu')(x)
    conv2_2norm = BatchNormalization()(x)  # TODO param {lr_mult: 0 decay_mult: 0}
    x = Conv2D(128, 1, strides=2)(conv2_2norm)
    # FIXME Apply following parameters (from Caffe protobuf description):
    # layer {
    #   param {lr_mult: 0 decay_mult: 0} # UNARY_BRANCH_PROPAGATE
    #   convolution_param {
    #     bias_term: false
    #     weight_filler { type: 'constant' value: 1 }
    #     group: 128
    #   }
    # }

    # conv3
    # conv3_1
    x = Conv2D(256, 3, padding='same', activation='relu')(x)
    # conv3_2
    x = Conv2D(256, 3, padding='same', activation='relu')(x)
    # conv3_3
    x = Conv2D(256, 3, padding='same', activation='relu')(x)
    conv3_3_norm = BatchNormalization()(x)  # TODO param {lr_mult: 0 decay_mult: 0}
    x = Conv2D(256, 1, strides=2)(conv3_3_norm)
    # FIXME Apply following parameters (from Caffe protobuf description):
    # layer {
    #   param {lr_mult: 0 decay_mult: 0} # UNARY_BRANCH_PROPAGATE
    #   convolution_param {
    #     bias_term: false
    #     weight_filler { type: 'constant' value: 1 }
    #     group: 256
    #   }
    # }

    # conv4
    # conv4_1
    x = Conv2D(512, 3, padding='same', activation='relu')(x)
    # conv4_2
    x = Conv2D(512, 3, padding='same', activation='relu')(x)
    # conv4_3
    x = Conv2D(512, 3, padding='same', activation='relu')(x)
    x = BatchNormalization()(x)  # TODO param {lr_mult: 0 decay_mult: 0}

    # conv5
    # conv5_1
    x = Conv2D(512, 3, padding='same', dilation_rate=2, activation='relu')(x)
    # conv5_2
    x = Conv2D(512, 3, padding='same', dilation_rate=2, activation='relu')(x)
    # conv5_3
    x = Conv2D(512, 3, padding='same', dilation_rate=2, activation='relu')(x)
    x = BatchNormalization()(x)  # TODO param {lr_mult: 0 decay_mult: 0}

    # conv6
    # conv6_1
    x = Conv2D(512, 3, padding='same', dilation_rate=2, activation='relu')(x)
    # conv6_2
    x = Conv2D(512, 3, padding='same', dilation_rate=2, activation='relu')(x)
    # conv6_3
    x = Conv2D(512, 3, padding='same', dilation_rate=2, activation='relu')(x)
    x = BatchNormalization()(x)  # TODO param {lr_mult: 0 decay_mult: 0}

    # conv7
    # conv7_1
    x = Conv2D(512, 3, padding='same', activation='relu')(x)
    # conv7_2
    x = Conv2D(512, 3, padding='same', activation='relu')(x)
    # conv7_3
    x = Conv2D(512, 3, padding='same', activation='relu')(x)
    x = BatchNormalization()(x)  # TODO param {lr_mult: 0 decay_mult: 0}

    # conv8
    # conv8_1
    x = Conv2DTranspose(256, 4, padding='same', strides=2, activation='relu')(x)
    # Shortcut
    # TODO weight_filler { type: "gaussian" std: .01 }
    # bias_filler { type: "constant" value: 1 }
    shortcut = Conv2D(256, 3, padding='same')(conv3_3_norm)
    x = Add()([x, shortcut])
    x = Activation('relu')(x)
    # conv8_2
    x = Conv2D(256, 3, padding='same', activation='relu')(x)
    # conv8_3
    x = Conv2D(256, 3, padding='same', activation='relu')(x)
    x = BatchNormalization()(x)  # TODO param {lr_mult: 0 decay_mult: 0}

    # conv9
    # conv9_1
    # TODO weight_filler { type: "gaussian" std: .01 }
    # bias_filler { type: "constant" value: 1 }
    x = Conv2DTranspose(128, 4, padding='same', strides=2, activation='relu')(x)
    # Shortcut
    # TODO weight_filler { type: "gaussian" std: .01 }
    # bias_filler { type: "constant" value: 1 }
    shortcut = Conv2D(128, 3, padding='same')(conv2_2norm)
    x = Add()([x, shortcut])
    x = Activation('relu')(x)
    # conv9_2
    # TODO weight_filler { type: "gaussian" std: .01 }
    # bias_filler { type: "constant" value: 1 }
    x = Conv2D(128, 3, padding='same', activation='relu')(x)
    # conv9_3
    x = Conv2D(128, 3, padding='same', activation='relu')(x)
    x = BatchNormalization()(x)  # TODO param {lr_mult: 0 decay_mult: 0}

    # conv10
    # conv10_1
    # TODO weight_filler { type: "gaussian" std: .01 }
    # bias_filler { type: "constant" value: 1 }
    x = Conv2DTranspose(128, 4, padding='same', strides=2, activation='relu')(x)
    # Shortcut
    # TODO weight_filler { type: "gaussian" std: .01 }
    # bias_filler { type: "constant" value: 1 }
    shortcut = Conv2D(128, 3, padding='same')(conv1_2norm)
    x = Add()([x, shortcut])
    x = Activation('relu')(x)
    # conv10_2
    # TODO weight_filler { type: "gaussian" std: .01 }
    # bias_filler { type: "constant" value: 1 }
    # relu_param { negative_slope: 0.2 }
    x = Conv2D(128, 3, padding='same', activation='relu')(x)
    # conv10_ab
    x = Conv2D(2, 1, activation='tanh')(x)
    x = Scale(100)(x)

    m = Model(inputs=[state_input, l_input], outputs=x)
    # TODO Another loss function might be more appropriate
    m.compile(loss='mean_squared_error',
              optimizer='adam',
              metrics=['accuracy'])
    return m


def l_to_ab():
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
    m.add(Lambda(lambda x: 2.606 * x))
    m.add(Activation('softmax'))

    # Decoding
    # TODO Implement class rebalancing, implement annealed-mean interpolation
    m.add(Conv2D(2, 1, padding='same'))  # FIXME Placeholder, should be an annealed-mean interpolation
    # TODO Implement loss function (cross entropy with soft-encoded ground truth)
    m.compile(loss='mean_squared_error',
              optimizer='adam',
              metrics=['accuracy'])
    return m
