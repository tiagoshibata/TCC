#!/usr/bin/env python3
import os
from pathlib import Path
import sys
import subprocess
import venv

from colormotion.environment import bash_source


def main(target_command):
    base_dir = Path(__file__).resolve().parent
    venv.create(base_dir / 'venv', with_pip=True)

    bash_source(base_dir / 'venv/bin/activate')

    requirements = base_dir / 'requirements.txt'
    if requirements.exists():
        subprocess.check_call(['pip', 'install', '-r', requirements])
    else:
        print('No requirements.txt file found, installing latest versions')
    subprocess.check_call(['pip', 'install', '-e', base_dir])

    os.execvp(target_command[0], target_command)


if __name__ == '__main__':
    assert sys.version_info.major == 3 and sys.version_info.minor >= 6, 'Use python3 >= 3.6'
    assert len(sys.argv) > 1, 'Pass target command as an argument'
    main(sys.argv[1:])
