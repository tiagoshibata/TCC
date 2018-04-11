#!/usr/bin/env python3
from pathlib import Path
from unittest.mock import ANY, patch

import cv2
import numpy as np
import pytest

import colormotion.dataset as dataset

base_dir = Path(__file__).resolve().parent


@pytest.mark.parametrize("dataset_directory,scene_number,expected", [
    ('dataset', 1, 'dataset/000001'),
    (Path('dataset'), 2, 'dataset/000002'),
])
@patch('pathlib.Path.mkdir')
def test_get_scene_directory(mock_mkdir, dataset_directory, scene_number, expected):
    assert dataset.get_scene_directory(dataset_directory, scene_number) == Path(expected)
    mock_mkdir.called_once_with(ANY, exist_ok=True)


@pytest.mark.parametrize("scene_directory,frame_number,expected", [
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
    assert np.allclose(l, expected_l, rtol=.15)
    assert np.allclose(ab, expected_ab, rtol=.15)


def test_lab_to_bgr():
    image = np.random.random((2, 2, 3)).astype(np.float32)
    assert np.allclose(dataset.lab_to_bgr(*dataset.to_lab(image)), image)


def test_get_frames():
    dataset_directory = base_dir / 'test_dataset'
    assert dataset.get_frames(dataset_directory) == {
        dataset_directory / 'movie/000000': [0, 1],
        dataset_directory / 'movie/000002': [2],
    }
