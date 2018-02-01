import TODO.activate_venv as activate_venv
import pytest


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
    assert activate_venv.parse_environment(environment) == expected_dict
