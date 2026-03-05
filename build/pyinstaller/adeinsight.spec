# build/pyinstaller/adeinsight.spec
from PyInstaller.utils.hooks import (
    collect_submodules,
    collect_data_files,
    collect_dynamic_libs,
    collect_data_files,
)

block_cipher = None

# Collect Qt (PySide6) runtime bits explicitly (key reliability boost)
pyside6_datas = collect_data_files("PySide6")
pyside6_bins = collect_dynamic_libs("PySide6")

# Optional: matplotlib runtime data (fonts, mpl-data) if you see missing-resource issues
mpl_datas = collect_data_files("matplotlib")

hiddenimports = (
    collect_submodules("PySide6")
    + collect_submodules("matplotlib")
    + ["matplotlib.backends.backend_qtagg"]
)

a = Analysis(
    ["build/pyinstaller/run_gui.py"],
    pathex=["."],
    binaries=[
        *pyside6_bins,
    ],
    datas=[
        ("src/ade_insight/gui/style.qss", "ade_insight/gui"),
        ("src/ade_insight/gui/assets", "ade_insight/gui/assets"),
        *pyside6_datas,
        *mpl_datas,   # optional; remove if you want smaller builds
    ],
    hiddenimports=hiddenimports,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="ade-insight",
    console=False,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    name="ade-insight",
)
