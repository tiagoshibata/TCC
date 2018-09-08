import cv2
import numpy as np

from colormotion.optical_flow import numerical_optical_flow, warp


def test_warp():
    previous, current = (np.random.random((128, 128, 1)) for _ in range(2))
    flow = numerical_optical_flow(previous, current)
    assert flow.shape == (128, 128, 2)
    assert warp(previous, flow).shape == (128, 128)


def test_warp_multiple_filters():
    previous, current = (np.random.random((128, 128, 1)) for _ in range(2))
    flow = numerical_optical_flow(previous, current)
    assert warp(np.random.random((32, 32, 16)), flow).shape == (32, 32, 16)


def test_warp_nonmatching_resolutions():
    previous, current = (np.random.random((128, 128, 1)) for _ in range(2))
    flow = numerical_optical_flow(previous, current)
    features = np.random.random((32, 32, 16))
    warped = warp(features, flow)
    scaled_before_warp = cv2.resize(warp(cv2.resize(features, (128, 128)), flow), (32, 32))
    relative_error = np.abs((warped - scaled_before_warp) / warped)
    assert np.mean(relative_error) < .06


def test_optical_flow_normalized():
    # Should return the same results for normalized and [0, 255] images
    previous, current = (np.random.randint(256, size=(128, 128, 1), dtype=np.uint8) for _ in range(2))
    flow = numerical_optical_flow(previous, current)
    assert (flow == numerical_optical_flow(previous / 255, current / 255)).all()
    assert np.allclose(warp(previous, flow) / 255, warp(previous / 255, flow))


def test_optical_flow_reuse_destination():
    resolution = (128, 128)
    previous, current = (np.random.randint(256, size=(*resolution, 1), dtype=np.uint8) for _ in range(2))
    flow = np.empty((*resolution, 2), dtype=np.float32)
    numerical_optical_flow(previous, current, destination=flow)
    assert (flow == numerical_optical_flow(previous, current)).all()
    warped = np.empty(resolution, dtype=np.float32)
    warp(previous, flow, destination=warped)
    assert (warped == warp(previous, flow)).all()
