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
