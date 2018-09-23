from collections import namedtuple
from pathlib import Path
import sys

import pytest

file_path = Path(__file__).resolve()
root = file_path.parents[3]
tests = file_path.parents[2]
sys.path.append(str(root / 'scripts'))


@pytest.mark.skip('WIP')
def test_build_metadata():
    from train.filters_optical_flow import main  # pylint: disable=import-error
    Args = namedtuple('Args', [
        'weights',
        'steps_per_epoch',
        'epochs',
        'validation_steps',
        'dataset',
    ])
    main(Args(
        None,
        5,
        2,
        1,
        tests / 'dataset',
    ))


@pytest.mark.skip('WIP')
def test_data_generators():
    from train.filters_optical_flow import data_generators, model  # pylint: disable=import-error
    m = model()
    m.summary()
    train, test = data_generators(tests / 'dataset', m)
    for _ in range(5):
        next(train)
        next(test)
