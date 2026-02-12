# ADE Insight — Windows Installer (MSI + Bootstrapper EXE)

This guide builds a proper Windows installer for deployment:

- `ADE_Insight.msi` (installs the app into Program Files, adds Start Menu shortcut, uninstaller entry)
- `ADEInsightSetup.exe` (bootstrapper EXE that installs prerequisites like VC++ Runtime, then installs the MSI)

Target: **Windows x64 only**

---

## 1) Prerequisites (Windows build machine)

Install the following:

### A) Python
- Python 3.10+ recommended (3.11 ideal)

### B) WiX Toolset v3.11.x
Install WiX 3.11 and ensure these are on PATH:
- `heat.exe`
- `candle.exe`
- `light.exe`

To check:
```powershell
heat -?
candle -?
light -?
