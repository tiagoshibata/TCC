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
