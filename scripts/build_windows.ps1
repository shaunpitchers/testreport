$ErrorActionPreference = "Stop"

function Msg($m) { Write-Host "`n==> $m" }
function Die($m) { Write-Error $m; exit 1 }

# --- config ---
$RepoRoot = (Resolve-Path "$PSScriptRoot\..").Path
$VenvDir = Join-Path $RepoRoot ".venv"
$ArtifactsDir = Join-Path $RepoRoot "artifacts"
$PyInstallerSpec = Join-Path $RepoRoot "build\pyinstaller\adeinsight.spec"
$WixBuildScript = Join-Path $RepoRoot "build\installer\wix3\build_msi.ps1"
$DistDir = Join-Path $RepoRoot "dist"
$DistApp = Join-Path $RepoRoot "dist\ade-insight"

# Ensure artifacts directory exists early (before any Copy-Item)
New-Item -ItemType Directory -Force -Path $ArtifactsDir | Out-Null

# --- checks ---
Msg "Checking Python"
$py = Get-Command python -ErrorAction SilentlyContinue
if (-not $py) { Die "Python not found on PATH. Install Python 3.10+ and retry." }

$pyver = python -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')"
Msg "Python version: $pyver"
python -c "import sys; assert sys.version_info >= (3,10)" | Out-Null

Msg "Creating venv: $VenvDir"
if (-not (Test-Path $VenvDir)) {
  python -m venv $VenvDir
}

. (Join-Path $VenvDir "Scripts\Activate.ps1")

Msg "Upgrading pip"
python -m pip install --upgrade pip

Msg "Installing project + build deps"
pip install -e ".[dev,gui]"
pip install pyinstaller build

Msg "Building wheel/sdist (sanity check)"
python -m build | Out-Null

# --- PyInstaller build ---
if (-not (Test-Path $PyInstallerSpec)) { Die "PyInstaller spec not found: $PyInstallerSpec" }

Msg "Running PyInstaller"
pyinstaller --noconfirm --clean $PyInstallerSpec

if (-not (Test-Path $DistDir)) { Die "dist/ not found after PyInstaller run." }

# Clean previous copied app to avoid mixing old/new
$CopiedAppDir = Join-Path $ArtifactsDir "ade-insight"
if (Test-Path $CopiedAppDir) { Remove-Item -Recurse -Force $CopiedAppDir }

Msg "Copying EXE folder to artifacts"
if (Test-Path $DistApp) {
  Copy-Item -Recurse -Force $DistApp $CopiedAppDir
} else {
  # fallback: copy whole dist once
  $CopiedDistDir = Join-Path $ArtifactsDir "dist"
  if (Test-Path $CopiedDistDir) { Remove-Item -Recurse -Force $CopiedDistDir }
  Copy-Item -Recurse -Force $DistDir $CopiedDistDir
}

# --- WiX MSI build (optional) ---
$BuiltMsi = $null
if (Test-Path $WixBuildScript) {
  Msg "Running WiX MSI build script"
  powershell -ExecutionPolicy Bypass -File $WixBuildScript

  $msis = Get-ChildItem -Path $RepoRoot -Recurse -Filter "*.msi" -ErrorAction SilentlyContinue |
    Sort-Object LastWriteTime -Descending
  if ($msis.Count -gt 0) { $BuiltMsi = $msis[0].FullName }
} else {
  Msg "WiX build script not found (skipping MSI build): $WixBuildScript"
}

if ($BuiltMsi) {
  Msg "Copying MSI to artifacts"
  Copy-Item -Force $BuiltMsi $ArtifactsDir
}

# --- checksums ---
Msg "Generating SHA256 checksums"
Get-ChildItem -Path $ArtifactsDir -Recurse -File | ForEach-Object {
  $hash = Get-FileHash $_.FullName -Algorithm SHA256
  "$($hash.Hash)  $($_.Name)" | Out-File -Encoding ascii -FilePath "$($_.FullName).sha256"
}

Msg "Done. Artifacts are in: $ArtifactsDir"
Write-Host "Next: test the EXE/MSI on a clean Windows machine."
