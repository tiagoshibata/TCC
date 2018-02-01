#!/usr/bin/env python3
import os
from pathlib import Path
import sys
import subprocess
import venv


def parse_environment(environment):
    key_values = [x.partition('=') for x in environment.splitlines()]
    return {key: value for key, _, value in key_values}


def main(target_command):
    base_dir = Path(__file__).resolve().parent
    venv.create(base_dir / 'venv', with_pip=True)

    environment = subprocess.check_output([
        'bash', '-c', 'source {} && env'.format(base_dir / 'venv/bin/activate')
    ], universal_newlines=True)
    os.environ.update(parse_environment(environment))

    subprocess.check_call(['pip', 'install', '-r', base_dir / 'requirements.txt'])
    subprocess.check_call(['pip', 'install', '-e', base_dir])

    os.execvp(target_command[0], target_command)


if __name__ == '__main__':
    assert sys.version_info.major == 3 and sys.version_info.minor >= 6, 'Use python3 >= 3.6'
    assert len(sys.argv) > 1, 'Pass target command as an argument'
    main(sys.argv[1:])
