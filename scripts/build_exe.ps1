$ErrorActionPreference = "Stop"

.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pyinstaller
pyinstaller --noconfirm --clean pyinstaller\adeinsight.spec

Write-Host "Build complete. See dist\"
