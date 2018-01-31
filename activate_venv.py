#!/usr/bin/env python3
import os
from pathlib import Path
import sys
import subprocess
import venv

assert sys.version_info.major == 3 and sys.version_info.minor >= 6, 'Use python3 >= 3.6'
assert len(sys.argv) > 1, 'Pass target executable as an argument'
base_dir = Path(__file__).resolve().parent

venv.create(base_dir / 'venv', with_pip=True)


def parse_environment(environment):
    key_values = [x.partition('=') for x in environment.splitlines()]
    return {key: value for key, _, value in key_values}


environment = subprocess.check_output([
    'bash', '-c', 'source {} && env'.format(base_dir / 'venv/bin/activate')
], universal_newlines=True)
os.environ.update(parse_environment(environment))

subprocess.check_call(['pip', 'install', '-r', base_dir / 'requirements.txt'])

os.execvp(sys.argv[1], sys.argv[1:])
