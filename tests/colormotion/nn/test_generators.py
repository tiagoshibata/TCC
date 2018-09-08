from unittest.mock import call, patch

import numpy as np

from colormotion.nn import generators


def test_get_contiguous_frames():
    generator = generators.VideoFramesGenerator()
    frames = {
        '0001': ['0001.png'],
        '0002': ['0002.png', '0003.png', '0004.png'],
        '0005': ['0005.png', '0006.png'],
    }
    assert generator.get_contiguous_frames(frames) == [
        ('0002', '0002.png'),
        ('0002', '0003.png'),
        ('0005', '0005.png'),
    ]


@patch('colormotion.nn.generators.VideoFramesWithLabStateGenerator.load_sample')
def test_lab_load_batch(mock_load_sample):
    state = [np.random.random((32, 32, 3)) for _ in range(3)]
    grayscale = [np.random.random((32, 32, 1)) for _ in range(3)]
    y = [np.random.random((32, 32, 2)) for _ in range(3)]
    mock_load_sample.side_effect = lambda _, frame, __: (state[frame], grayscale[frame], y[frame])

    generator = generators.VideoFramesWithLabStateGenerator()
    x_batch, y_batch = generator.load_batch([('scene', 0), ('scene', 1), ('other_scene', 2)], (32, 32))

    mock_load_sample.assert_has_calls([
        call('scene', 0, (32, 32)),
        call('scene', 1, (32, 32)),
        call('other_scene', 2, (32, 32)),
    ])
    # Should have two inputs (state, grayscale) and one output, each with 3 elements, matching the list given to
    # load_batch
    assert (x_batch[0] == np.array(state)).all()
    assert (x_batch[1] == np.array(grayscale)).all()
    assert (y_batch == np.array(y)).all()


@patch('colormotion.nn.generators.VideoFramesWithLStateGenerator.load_sample')
def test_l_load_batch(mock_load_sample):
    state = [np.random.random((32, 32)) for _ in range(3)]
    grayscale = [np.random.random((32, 32, 1)) for _ in range(3)]
    y = [np.random.random((32, 32, 2)) for _ in range(3)]
    mock_load_sample.side_effect = lambda _, frame, __: (state[frame], grayscale[frame], y[frame])

    generator = generators.VideoFramesWithLStateGenerator()
    x_batch, y_batch = generator.load_batch([('scene', 0), ('scene', 1), ('other_scene', 2)], (32, 32))

    mock_load_sample.assert_has_calls([
        call('scene', 0, (32, 32)),
        call('scene', 1, (32, 32)),
        call('other_scene', 2, (32, 32)),
    ])
    # Should have two inputs (state, grayscale) and one output, each with 3 elements, matching the list given to
    # load_batch
    assert (x_batch[0] == np.array(state)).all()
    assert (x_batch[1] == np.array(grayscale)).all()
    assert (y_batch == np.array(y)).all()


@patch('colormotion.nn.generators.read_image_lab')
def test_l_load_sample(mock_read_image_lab):
    l = [np.random.random((32, 32, 3)) for _ in range(4)]
    ab = [np.random.random((32, 32, 3)) for _ in range(4)]
    mock_read_image_lab.side_effect = lambda _, frame, __: (l[frame], ab[frame])

    generator = generators.VideoFramesWithLStateGenerator()
    x_batch, y_batch = generator.load_batch([('scene', 0), ('scene', 1), ('other_scene', 2)], (32, 32))

    mock_read_image_lab.assert_has_calls([
        call('scene', 1, (32, 32)),
        call('scene', 0, (32, 32)),
        call('scene', 2, (32, 32)),
        call('scene', 1, (32, 32)),
        call('other_scene', 3, (32, 32)),
        call('other_scene', 2, (32, 32)),
    ])
    # Should have two inputs (state (the L input of the previous frame) and grayscale) and one output, each with 3
    # elements, matching the list given to load_batch
    assert (x_batch[0] == np.array([l[0], l[1], l[2]])).all()
    assert (x_batch[1] == np.array([l[1], l[2], l[3]])).all()
    assert (y_batch == np.array([ab[1], ab[2], ab[3]])).all()
