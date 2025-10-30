#!/usr/bin/env bash

source venv/bin/activate

python setup.py clean --all
mkdir -p dist
python -m pip wheel . -w dist/ --no-deps --no-build-isolation
