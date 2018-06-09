#!/usr/bin/env python3
from collections import OrderedDict
import copy
from pathlib import Path
from unittest.mock import ANY, mock_open, patch

import cv2
import numpy as np
import pytest

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


def test_to_lab():
    image = np.random.random((2, 2, 3)).astype(np.float32)
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2Lab)
    expected_l, expected_ab = lab[:, :, 0], lab[:, :, 1:]
    expected_l -= 50  # mean centering
    expected_l = expected_l.reshape(2, 2, 1)
    l, ab = dataset.to_lab(image)
    assert l.shape == (2, 2, 1)
    assert ab.shape == (2, 2, 2)
    assert np.allclose(l, expected_l, rtol=.3)
    assert np.allclose(ab, expected_ab, rtol=.3)


def test_lab_to_bgr():
    rgb_image = np.random.random((2, 2, 3)).astype(np.float32)
    rgb_original_image = rgb_image.copy()
    lab_image = dataset.to_lab(rgb_image)
    assert (rgb_original_image == rgb_image).all()  # no conversions should be done in place
    lab_original_image = copy.deepcopy(lab_image)
    assert np.allclose(dataset.lab_to_bgr(*lab_image), rgb_image, rtol=1e-3)
    assert all((original_channel == channel).all() for original_channel, channel in zip(lab_original_image, lab_image))


def test_get_all_scenes():
    dataset_directory = base_dir / 'test_dataset'
    frames = dataset.get_all_scenes(dataset_directory)
    assert frames == OrderedDict([
        (dataset_directory / 'movie/000000', [0, 1]),
        (dataset_directory / 'movie/000002', [2]),
    ])
