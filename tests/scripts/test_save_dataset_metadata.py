from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[2] / 'scripts'))


def test_build_metadata():
    from save_dataset_metadata import build_metadata
    assert build_metadata({}) == {}
    root = Path('path/to/dataset/root')
    assert build_metadata({
        root / 'video_0_id/0': [0, 1, 2],
        root / 'video_0_id/3': [3, 4, 5],
        root / 'video_1_id/0': [0, 1],
    }) == {
        'video_0_id': [
            (0, 2),
            (3, 5),
        ],
        'video_1_id': [
            (0, 1),
        ],
    }
