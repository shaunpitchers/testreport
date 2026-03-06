$ErrorActionPreference = "Stop"

function Die($m) { Write-Error $m; exit 1 }

$RepoRoot = (Resolve-Path "$PSScriptRoot\..").Path
Set-Location $RepoRoot

$VenvDir = Join-Path $RepoRoot ".venv"
$Spec = Join-Path $RepoRoot "build\pyinstaller\adeinsight.spec"

if (-not (Get-Command python -ErrorAction SilentlyContinue)) {
  Die "Python not found on PATH. Install Python 3.11+ and retry."
}

if (-not (Test-Path $VenvDir)) {
  python -m venv $VenvDir
}

. (Join-Path $VenvDir "Scripts\Activate.ps1")

python -m pip install --upgrade pip
python -m pip install -e ".[dev,gui]"
python -m pip install --upgrade pyinstaller

if (-not (Test-Path $Spec)) { Die "PyInstaller spec not found: $Spec" }

python -m PyInstaller --noconfirm --clean --workpath .pyi_build --distpath dist $Spec
if ($LASTEXITCODE -ne 0) {
  Die "PyInstaller build failed."
}

Write-Host "Build complete. See dist/"
