#!/usr/bin/env bash
set -e

source .venv/bin/activate
python -m pip install --upgrade pyinstaller
pyinstaller --noconfirm --clean pyinstaller/adeinsight.spec

echo "Build complete. See dist/"
