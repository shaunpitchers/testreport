$ErrorActionPreference = "Stop"

$Root = Resolve-Path (Join-Path $PSScriptRoot "..\..")
Set-Location $Root

$Wix = Join-Path $Root "installer\wix3"
$OutDir = Join-Path $Root "dist-installer"
New-Item -ItemType Directory -Force -Path $OutDir | Out-Null

Write-Host "[1/4] Build app folder via PyInstaller..."
& "$Root\scripts\build_exe.ps1"

$DistApp = Join-Path $Root "dist\ADE Insight"
if (!(Test-Path $DistApp)) { throw "Missing PyInstaller output folder: $DistApp" }

Write-Host "[2/4] Harvest dist folder into WiX components..."
$HarvestWxs = Join-Path $Wix "harvest.wxs"
heat dir "$DistApp" `
  -cg ADEInsightFiles `
  -dr INSTALLFOLDER `
  -gg -sreg -srd -sfrag `
  -var var.SourceDir `
  -out "$HarvestWxs"

Write-Host "[3/4] Build MSI..."
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

Write-Host "[4/4] Ensure vc_redist.x64.exe present..."
$VcRedist = Join-Path $Wix "vc_redist.x64.exe"
if (!(Test-Path $VcRedist)) {
  # Official aka.ms link often redirects; this works in PowerShell with -L behavior
  $url = "https://aka.ms/vs/17/release/vc_redist.x64.exe"
  Write-Host "Downloading VC++ redist from: $url"
  Invoke-WebRequest -Uri $url -OutFile $VcRedist
}

Write-Host "[5/5] Build Bootstrapper EXE..."
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

