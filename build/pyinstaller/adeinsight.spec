block_cipher = None

a = Analysis(
    ['pyinstaller/run_gui.py'],
    pathex=['.'],
    binaries=[],
    datas=[],
    hiddenimports=['matplotlib.backends.backend_qtagg', 'PySide6'],
)

pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='ADE Insight',
    console=False,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    name='ADE Insight',
)
