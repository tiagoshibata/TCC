import numpy as np

from colormotion.nn.model.user_guided import encoder_model, model


def test_encoder_predicts():
    m = encoder_model()
    l = np.empty((1, 256, 256, 1))
    ab_and_mask = np.empty((1, 256, 256, 3))
    m.predict(x=[l, ab_and_mask])


def test_compiles():
    model()
