import keras.backend as K
from keras.engine.topology import Layer
from keras.layers import Lambda


def Crop(axis, start, end, **kwargs):  # pylint: disable=invalid-name
    def f(x):
        if axis == 0:
            return x[start:end]
        if axis == 1:
            return x[:, start:end]
        if axis == 2:
            return x[:, :, start:end]
        if axis == 3:
            return x[:, :, :, start:end]
        if axis == 4:
            return x[:, :, :, :, start:end]
        raise NotImplementedError()
    return Lambda(f, **kwargs)


def Scale(amount, **kwargs):  # pylint: disable=invalid-name
    return Lambda(lambda x: x * amount, **kwargs)


class NumpyLayer(Layer):
    def __init__(self, f, compute_shape, **kwargs):
        super().__init__(**kwargs)
        self.f = f
        self.compute_shape = compute_shape

    def call(self, inputs, **kwargs):
        return K.variable(self.f(*(K.eval(x) for x in inputs)))

    def compute_output_shape(self, input_shapes):  # pylint: disable=arguments-differ
        return self.compute_shape(*input_shapes)

if K.backend() == 'tensorflow':
    import tensorflow as tf
    depthwise_conv_2d = tf.keras.layers.DepthwiseConv2D  # pylint: disable=invalid-name
else:
    # Grouped convolutions can also be done with Slice/Concat operations, but with a performance hit.
    # See e.g.:
    # https://gist.github.com/mjdietzx/0cb95922aac14d446a6530f87b3a04ce#file-residual_network-py-L39
    def depthwise_conv_2d(*_, **__):
        raise NotImplementedError('DepthwiseConv2D is only supported on tensorflow')

DepthwiseConv2D = depthwise_conv_2d


def _load_weights_numpy(model, weights_path):
    import numpy as np
    weights_data = np.load(str(weights_path)).item()
    for layer in model.layers:
        weights = weights_data.pop(layer.name, None)
        if weights:
            keys = set(weights.keys())
            if keys == {'weights', 'bias'}:
                layer.set_weights((weights['weights'], weights['bias']))
            elif keys == {'mean', 'var'}:
                zeros = np.zeros_like(weights['mean'])
                layer.set_weights((zeros, zeros, weights['mean'], weights['var']))
            else:
                raise NotImplementedError("Can't load layer {} with params {}".format(layer.name, keys))
        else:
            print('Layer {} has no pretrained weights'.format(layer.name))
    if weights_data:
        print('The following layers are in the weights file, but have no corresponding '
              'layer in the model: {}'.format(', '.join(weights_data.keys())))


def load_weights(model, weights_path, by_name=False):
    suffix = weights_path.suffix
    if suffix == '.npy':
        _load_weights_numpy(model, weights_path)
    elif suffix in ('.h5', '.hdf5'):
        model.load_weights(str(weights_path), by_name=by_name)
    else:
        raise NotImplementedError()
