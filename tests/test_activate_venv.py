import pytest

from colormotion.environment import parse_environment


@pytest.mark.parametrize('environment,expected_dict', [
    (
        'a=0\n'
        'b=1',
        {'a': '0', 'b': '1'},
    ), (
        'a=0=\n'
        '\n'
        'b==1\n',
        {'a': '0=', 'b': '=1'},
    ),
])
def test_parse_environment(environment, expected_dict):
    assert parse_environment(environment) == expected_dict
