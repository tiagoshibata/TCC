from collections import namedtuple
from pathlib import Path
import sys

file_path = Path(__file__).resolve()
root = file_path.parents[3]
tests = file_path.parents[2]
sys.path.append(str(root / 'scripts'))


def test_build_metadata():
    from train.filters_optical_flow import main
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


def test_data_generators():
    from train.filters_optical_flow import data_generators, model
    m = model()
    m.summary()
    print(next(data_generators(tests / 'dataset', m)[0]))
