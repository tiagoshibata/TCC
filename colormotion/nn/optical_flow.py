'''Recurrent warp unit based on optical flow.

Some concepts are taken from https://arxiv.org/abs/1703.09211.
'''
import keras.backend as K
from keras.layers import Add, Layer, Multiply, Subtract


def optical_flow(previous, current):  # pylint: disable=unused-argument
    raise NotImplementedError()


def warp(features, flow):  # pylint: disable=unused-argument
    raise NotImplementedError()


class FlowMaskRNNCell(Layer):
    def __init__(self, units, **kwargs):
        # batch_size, timesteps, input_dim = units
        self.units = units
        self.state_size = units
        super().__init__(**kwargs)

    def build(self, input_shape):
        # pylint: disable=attribute-defined-outside-init
        self.kernel = None  # TODO
        self.recurrent_kernel = None  # TODO
        super().build(input_shape)

    def call(self, inputs, states):  # pylint: disable=arguments-differ
        assert len(inputs) == 3
        assert len(states) == 1
        previous_features = states[0]
        previous_image, current_image, current_features = inputs
        flow = optical_flow(previous_image, current_image)
        warped_features = warp(previous_features, flow)
        difference = Subtract()([previous_features, warped_features])
        # Composition mask
        raise NotImplementedError()
        mask = difference  # pylint: disable=unreachable
        # Interpolate warped features and encoded features
        warped_features = Multiply()([warped_features, mask])
        ones = K.constant(1, shape=K.shape(current_features))
        ones_minus_mask = Subtract()([ones, mask])
        current_features = Multiply()([current_features, ones_minus_mask])
        features = Add()([warped_features, current_features])
        return features, [features]
