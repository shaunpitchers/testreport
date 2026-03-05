# Installation (ADE Insight)

This guide covers: - Using ADE Insight on **Linux** (pip + venv) -
Installing ADE Insight on **Windows** (recommended: MSI/Setup EXE) -
Building Windows binaries (developer workflow)

------------------------------------------------------------------------

## Linux (pip + venv) --- Recommended for Linux users

### Prerequisites

-   Python 3.10+
-   `python3 -m venv` available

### Install using the provided script

From the repo root:

``` bash
chmod +x scripts/install_linux.sh
./scripts/install_linux.sh
```

This will: - create a venv under `~/.local/ade-insight/venv` - install
ADE Insight - create launchers in `~/.local/bin`: - `adeinsight` -
`adeinsight-gui`

If `~/.local/bin` is not on your PATH, add this to your shell profile:

``` bash
export PATH="$HOME/.local/bin:$PATH"
```

### Verify

``` bash
adeinsight --help
```

------------------------------------------------------------------------

## Windows (end users) --- Recommended install method

### Install from released artifacts

Use one of the provided release artifacts: - `ADEInsightSetup.exe`
(bootstrapper) - or `ADE_Insight.msi`

Install by double-clicking the installer.

### Verify

After install: - launch via Start Menu shortcut (if provided) - or run
the installed EXE if your installer places it in a known location.

------------------------------------------------------------------------

## Windows (developers) --- Build EXE/MSI locally

### Prerequisites

-   Windows 10/11
-   Python 3.10+ on PATH
-   (For MSI builds) WiX Toolset v3 installed and tools available on
    PATH:
    -   `heat`
    -   `candle`
    -   `light`

### Build everything (EXE + MSI + checksums)

From PowerShell at repo root:

``` powershell
Set-ExecutionPolicy -Scope Process Bypass
.\scripts\build_windows.ps1
```

Outputs: - `artifacts\ade-insight\...` (PyInstaller one-folder build) or
`artifacts\dist\...` - MSI (if WiX available) - `.sha256` files for
integrity verification

### Build EXE only (quick build)

``` powershell
Set-ExecutionPolicy -Scope Process Bypass
.\scripts\build_exe.ps1
```

------------------------------------------------------------------------

## Notes

### Where outputs go

-   PyInstaller output: `dist\`
-   Installer output: `dist-installer\` (if building MSI)
-   Packaged artifacts: `artifacts\`

### Integrity checks (SHA256)

Artifacts produced by the Windows build script include `.sha256` files.

To verify on Windows:

``` powershell
Get-FileHash .\path\to\file -Algorithm SHA256
```

Compare the hash with the published `.sha256` file.

------------------------------------------------------------------------

## Uninstall

### Linux

Remove the install directory and launchers:

``` bash
rm -rf ~/.local/ade-insight
rm -f ~/.local/bin/adeinsight ~/.local/bin/adeinsight-gui
```

### Windows

Uninstall via **Add or Remove Programs**.
