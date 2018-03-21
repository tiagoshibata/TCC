#!/usr/bin/env python3
import argparse
import os
from pathlib import Path
import sys
import subprocess
import venv

from colormotion.environment import bash_source


def main(args):
    base_dir = Path(__file__).resolve().parent
    venv.create(base_dir / 'venv', system_site_packages=True, with_pip=True)

    bash_source(base_dir / 'venv/bin/activate')

    requirements = base_dir / 'requirements.txt'
    if requirements.exists():
        subprocess.check_call(['pip', 'install', '-Ir', requirements])
    else:
        print("No requirements.txt file found, versions aren't frozen!")
    subprocess.check_call(['pip', 'install', '-e', str(base_dir)])

    os.execvp(args.command[0], args.command)


def parse_args():
    parser = argparse.ArgumentParser(description='Activates a venv for the project.')
    parser.add_argument('command', nargs='*', default=['/bin/sh'],
                        help='command to execute in the venv, with optional arguments')
    return parser.parse_args()


if __name__ == '__main__':
    assert sys.version_info.major == 3 and sys.version_info.minor >= 6, 'Use python3 >= 3.6'
    main(parse_args())
