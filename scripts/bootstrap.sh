#!/usr/bin/env bash
set -e

PY=${PYTHON:-python3}

if [ ! -d ".venv" ]; then
    $PY -m venv .venv
fi

source .venv/bin/activate
python -m pip install --upgrade pip wheel setuptools
python -m pip install -e ".[gui]"

echo "ADE Insight installed."
