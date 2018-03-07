# ColorMotion

[![Build Status](https://travis-ci.org/ColorMotion/ColorMotion.svg?branch=master)](https://travis-ci.org/ColorMotion/ColorMotion)

## Running in a venv

The `activate_venv.py` script can be used to run commands inside a virtual environment. Run `./activate_venv.py bash` to start a `bash` shell inside a venv, or `./activate_venv.py python` to start `python`.

## Freezing dependencies

To regenerate `requirements.txt`, execute `pip freeze --local | grep -v colormotion > requirements.txt`.
