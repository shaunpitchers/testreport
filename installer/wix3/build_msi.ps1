$ErrorActionPreference = "Stop"

# Paths
$Root = Resolve-Path (Join-Path $PSScriptRoot "..\..")
$DistDir = Join-Path $Root "dist\ADE Insight"
$OutDir  = Join-Path $Root "dist-installer"
$WixDir  = $PSScriptRoot

if (!(Test-Path $DistDir)) {
  throw "Missing PyInstaller output: $DistDir  (run scripts/build_exe.ps1 first)"
}

New-Item -ItemType Directory -Force -Path $OutDir | Out-Null

# 1) Harvest the app folder into a component group
# -cg: component group name
# -dr: directory ref id in Product.wxs (INSTALLFOLDER)
# -gg: generate GUIDs
# -sreg/-srd/-sfrag reduce noise
# -var: makes source path variable so build works anywhere
$HarvestWxs = Join-Path $WixDir "harvest.wxs"
heat dir "$DistDir" `
  -cg ADEInsightFiles `
  -dr INSTALLFOLDER `
  -gg -sreg -srd -sfrag `
  -var var.SourceDir `
  -out "$HarvestWxs"

# 2) Compile
candle -nologo `
  -dSourceDir="$DistDir" `
  -out "$OutDir\\" `
  "$WixDir\Product.wxs" `
  "$HarvestWxs"

# 3) Link -> MSI
$MsiPath = Join-Path $OutDir "ADE_Insight.msi"
light -nologo -ext WixUIExtension `
  -out "$MsiPath" `
  "$OutDir\Product.wixobj" `
  "$OutDir\harvest.wixobj"

Write-Host "MSI created: $MsiPath"

