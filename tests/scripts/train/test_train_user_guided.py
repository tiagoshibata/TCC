from collections import namedtuple
from pathlib import Path
import sys

import pytest

file_path = Path(__file__).resolve()
root = file_path.parents[3]
tests = file_path.parents[2]
sys.path.append(str(root / 'scripts'))


@pytest.mark.skip('WIP')
def test_train():
    from train.user_guided import main  # pylint: disable=import-error
    Args = namedtuple('Args', [
        'weights',
        'steps_per_epoch',
        'epochs',
        'validation_steps',
        'dataset',
        'dry',
    ])
    main(Args(
        None,
        5,
        2,
        1,
        tests / 'dataset',
        True,
    ))
