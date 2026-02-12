$ErrorActionPreference = "Stop"

if (!(Test-Path ".venv")) {
    py -m venv .venv
}

.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip wheel setuptools
python -m pip install -e ".[gui]"

Write-Host "ADE Insight installed."
