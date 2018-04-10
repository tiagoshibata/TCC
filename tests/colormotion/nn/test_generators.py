from unittest.mock import call, patch

import numpy as np

import colormotion.nn.generators as generators


def test__get_contiguous_frames():
    generator = generators.VideoFramesDataGenerator()
    frames = {
        '0001': ['0001.png'],
        '0002': ['0002.png', '0003.png', '0004.png'],
        '0005': ['0005.png', '0006.png'],
    }
    assert generator._get_contiguous_frames(frames) == [
        ('0002', '0002.png'),
        ('0002', '0003.png'),
        ('0005', '0005.png'),
    ]


@patch('colormotion.nn.generators.VideoFramesDataGenerator._load_sample')
def test__load_batch(mock__load_sample):
    state = [np.random.random((32, 32, 3)) for _ in range(3)]
    grayscale = [np.random.random((32, 32, 1)) for _ in range(3)]
    y = [np.random.random((32, 32, 2)) for _ in range(3)]
    mock__load_sample.side_effect = lambda _, start_frame, __: (state[start_frame], grayscale[start_frame], y[start_frame])

    generator = generators.VideoFramesDataGenerator()
    x_batch, y_batch = generator._load_batch([('scene', 0), ('scene', 1), ('other_scene', 2)], (32, 32))

    mock__load_sample.assert_has_calls([
        call('scene', 0, (32, 32)),
        call('scene', 1, (32, 32)),
        call('other_scene', 2, (32, 32)),
    ])
    # Should have two inputs (state and grayscale) and one output, each with 3 elements, matching the list given to _load_batch
    assert (x_batch[0] == np.array(state)).all()
    assert (x_batch[1] == np.array(grayscale)).all()
    assert (y_batch == np.array(y)).all()
