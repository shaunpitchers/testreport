# Installing ADE Insight

## Requirements
- Python 3.10+
- pip

## Install (All Platforms)

Linux/macOS:
    python3 -m venv .venv
    source .venv/bin/activate
    python -m pip install --upgrade pip
    python -m pip install -e .[gui]

Windows:
    py -m venv .venv
    .\.venv\Scripts\Activate.ps1
    python -m pip install --upgrade pip
    python -m pip install -e .[gui]

Run:
    adeinsight-gui
