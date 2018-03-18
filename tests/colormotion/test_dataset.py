#!/usr/bin/env python3
from pathlib import Path
from unittest.mock import ANY, patch

import numpy as np
import pytest

import colormotion.dataset as dataset


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
def test_load_image(mock_imread):
    mock_image = np.zeros((200, 200, 3), dtype='uint8')
    mock_imread.return_value = mock_image
    assert (dataset.load_image('image.png') == mock_image).all()
    assert dataset.load_image('image.png', resolution=(100, 50)).shape == (50, 100, 3)

    mock_image = np.zeros((200, 200), dtype='uint8')
    mock_imread.return_value = mock_image
    assert dataset.load_image('image.png', color=False, resolution=(100, 50)).shape == (50, 100)
