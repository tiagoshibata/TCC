import keras.backend as K
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
