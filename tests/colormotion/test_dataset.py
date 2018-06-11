#!/usr/bin/env python3
from collections import OrderedDict
import copy
from pathlib import Path
from unittest.mock import ANY, mock_open, patch

import cv2
import numpy as np
import pytest
import skimage

import colormotion.dataset as dataset

base_dir = Path(__file__).resolve().parent


def test_hash_file():
    data = b'stubdata' * 2 * 1024  # 16 KB of data
    with patch("builtins.open", mock_open(read_data=data)) as mock_file:
        small_hash = dataset.hash_file('stub_filename.mp4')
        mock_file.assert_called_once_with('stub_filename.mp4', 'rb')
        assert len(small_hash) == 20 * 2
    data = 2 * data  # 32 KB of data
    with patch("builtins.open", mock_open(read_data=data)) as mock_file:
        big_hash = dataset.hash_file('stub_filename.mp4')
        assert len(big_hash) == 20 * 2
        assert big_hash != small_hash


@pytest.mark.parametrize('dataset_directory,scene_number,expected', [
    ('dataset', 1, 'dataset/000001'),
    (Path('dataset'), 2, 'dataset/000002'),
])
@patch('pathlib.Path.mkdir')
def test_get_scene_directory(mock_mkdir, dataset_directory, scene_number, expected):
    assert dataset.get_scene_directory(dataset_directory, scene_number) == Path(expected)
    mock_mkdir.called_once_with(ANY, exist_ok=True)


@pytest.mark.parametrize('scene_directory,frame_number,expected', [
    ('dataset/scene', 1, 'dataset/scene/000001.png'),
    (Path('dataset/scene'), 2, 'dataset/scene/000002.png'),
])
def test_get_frame_path(scene_directory, frame_number, expected):
    assert dataset.get_frame_path(scene_directory, frame_number) == Path(expected)


@patch('cv2.imread')
def test_read_image(mock_imread):
    mock_image = np.zeros((200, 200, 3), dtype='uint8')
    mock_imread.return_value = mock_image
    assert (dataset.read_image('image.png') == mock_image).all()
    assert dataset.read_image('image.png', resolution=(100, 50)).shape == (50, 100, 3)

    mock_image = np.zeros((200, 200), dtype='uint8')
    mock_imread.return_value = mock_image
    assert dataset.read_image('image.png', color=False, resolution=(100, 50)).shape == (50, 100)


def test_bgr_to_lab():
    bgr = np.random.random((256, 256, 3)).astype('float32')
    l, ab = dataset.bgr_to_lab(bgr)
    assert np.logical_and(-50 < l, l < 50).all()  # should be mean centered
    assert l.shape == (256, 256, 1)
    assert ab.shape == (256, 256, 2)


def test_lab_to_bgr():
    bgr_image = np.random.random((256, 256, 3)).astype(np.float32)
    bgr_original_image = bgr_image.copy()
    lab_image = dataset.bgr_to_lab(bgr_image)
    # No conversions should be done in place and the bgr_image parameter should remain the same
    assert (bgr_original_image == bgr_image).all()
    lab_original_image = copy.deepcopy(lab_image)
    assert np.allclose(dataset.lab_to_bgr(*lab_image), bgr_image, atol=1e-2)
    assert all((original_channel == channel).all() for original_channel, channel in zip(lab_original_image, lab_image))


def test_lab_to_bgr_implementations():
    # We want OpenCV in L*a*b->RGB conversion (since it's ~20 times faster than scikit), but it
    # has to behave like scikit.
    # Here we test whether both implementations return the same results within a small margin.
    rgb = np.random.random((256, 256, 3))
    lab = skimage.color.rgb2lab(rgb)
    scikit_result = skimage.color.lab2rgb(lab)
    opencv_result = cv2.cvtColor(lab.astype(np.float32), cv2.COLOR_LAB2RGB)
    assert np.allclose(scikit_result, opencv_result, atol=5e-4)


def test_get_all_scenes():
    dataset_directory = base_dir / 'test_dataset'
    frames = dataset.get_all_scenes(dataset_directory)
    assert frames == OrderedDict([
        (dataset_directory / 'movie/000000', [0, 1]),
        (dataset_directory / 'movie/000002', [2]),
    ])
