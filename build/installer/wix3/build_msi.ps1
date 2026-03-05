$ErrorActionPreference = "Stop"

function Msg($m) { Write-Host "`n==> $m" }
function Die($m) { Write-Error $m; exit 1 }

# Repo root: build/installer/wix3 -> build/installer -> build -> repo root
$Root = Resolve-Path (Join-Path $PSScriptRoot "..\..\..")
Set-Location $Root

# Canonical wix folder (under build/)
$Wix = Join-Path $Root "build\installer\wix3"

# Output directory for MSI/EXE
$OutDir = Join-Path $Root "dist-installer"
New-Item -ItemType Directory -Force -Path $OutDir | Out-Null

# Check WiX tools exist on PATH
foreach ($tool in @("heat", "candle", "light")) {
  if (-not (Get-Command $tool -ErrorAction SilentlyContinue)) {
    Die "WiX tool '$tool' not found on PATH. Install WiX Toolset (v3) and ensure tools are available in this shell."
  }
}

Msg "[1/5] Build app folder via PyInstaller..."
& "$Root\scripts\build_exe.ps1"

# Expect PyInstaller one-folder output (match your spec name)
$DistApp = Join-Path $Root "dist\ade-insight"
if (!(Test-Path $DistApp)) { Die "Missing PyInstaller output folder: $DistApp" }

Msg "[2/5] Harvest dist folder into WiX components..."
$HarvestWxs = Join-Path $Wix "harvest.wxs"
if (Test-Path $HarvestWxs) { Remove-Item -Force $HarvestWxs }

heat dir "$DistApp" `
  -cg ADEInsightFiles `
  -dr INSTALLFOLDER `
  -gg -sreg -srd -sfrag `
  -var var.SourceDir `
  -out "$HarvestWxs"

Msg "[3/5] Build MSI..."
candle -nologo `
  -dSourceDir="$DistApp" `
  -out "$OutDir\\" `
  "$Wix\Product.wxs" `
  "$HarvestWxs"

$MsiPath = Join-Path $OutDir "ADE_Insight.msi"
light -nologo -ext WixUIExtension `
  -out "$MsiPath" `
  "$OutDir\Product.wixobj" `
  "$OutDir\harvest.wixobj"

Write-Host "MSI created: $MsiPath"

Msg "[4/5] Ensure vc_redist.x64.exe present..."
$VcRedist = Join-Path $Wix "vc_redist.x64.exe"
if (!(Test-Path $VcRedist)) {
  $url = "https://aka.ms/vs/17/release/vc_redist.x64.exe"
  Write-Host "Downloading VC++ redist from: $url"
  Invoke-WebRequest -Uri $url -OutFile $VcRedist
}

Msg "[5/5] Build Bootstrapper EXE..."
candle -nologo `
  -out "$OutDir\\" `
  -ext WixBalExtension `
  -ext WixUtilExtension `
  "$Wix\Bundle.wxs"

$SetupExe = Join-Path $OutDir "ADEInsightSetup.exe"
light -nologo `
  -out "$SetupExe" `
  -ext WixBalExtension `
  -ext WixUtilExtension `
  "$OutDir\Bundle.wixobj"

Write-Host ""
Write-Host "DONE."
Write-Host "MSI:  $MsiPath"
Write-Host "EXE:  $SetupExe"
