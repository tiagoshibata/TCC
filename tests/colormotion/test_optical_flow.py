import numpy as np

from colormotion.optical_flow import optical_flow, warp


def test_warp():
    previous, current = (np.random.random((256, 256, 1)) for _ in range(2))
    flow = optical_flow(previous, current)
    assert flow.shape == (256, 256, 2)
    assert warp(previous, flow).shape == (256, 256)
