#!/usr/bin/env python3
from keras.models import Sequential
from keras.layers import Conv2D, Conv2DTranspose


def model(input_shape):
    m = Sequential()
    m.add(Conv2D(16, kernel_size=(3, 3), activation='relu', input_shape=input_shape))
    # m.add(Conv2D(32, (3, 3), activation='relu'))
    # m.add(Conv2D(64, (3, 3), activation='relu'))
    # SeparableConv2D
    # m.add(Conv1D(32, 3))
    # Dropout
    # m.add(Conv2DTranspose(32, (3, 3)))
    # m.add(Conv2DTranspose(16, (3, 3)))
    m.add(Conv2DTranspose(3, (3, 3)))
    m.compile(loss='mean_squared_error',
              optimizer='adam',
              metrics=['accuracy'])
    return m
