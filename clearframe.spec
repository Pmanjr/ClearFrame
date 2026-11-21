# PyInstaller spec: one-folder build in dist/ClearFrame/.
# Build with build.bat (or: pyinstaller --noconfirm clearframe.spec).
from PyInstaller.utils.hooks import collect_all, collect_dynamic_libs

datas, binaries, hiddenimports = [], [], []
for pkg in ("mediapipe", "pyvirtualcam", "pygrabber", "comtypes"):
    d, b, h = collect_all(pkg)
    datas += d
    binaries += b
    hiddenimports += h
binaries += collect_dynamic_libs("onnxruntime")          # onnxruntime.dll + DirectML.dll
hiddenimports += [
    "clearframe.ui.app", "clearframe.ui.panels_extra", "clearframe.ui.panels_bonus", "clearframe.lut", "clearframe.presenter",
    "clearframe.calibrate", "pynput.keyboard._win32", "pynput.mouse._win32", "mss.windows",
]

a = Analysis(
    ["run_app.py"],
    pathex=["."],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    excludes=["tkinter", "PySide6.QtQml", "PySide6.QtQuick", "PySide6.QtSql",
              "PySide6.QtOpenGL", "PySide6.QtPdf", "IPython", "pytest", "sounddevice"],
    noarchive=False,
)
pyz = PYZ(a.pure)
exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="ClearFrame",
    console=False,                # windowed app; use `python -m clearframe cli` for the CLI
    upx=False,
)
coll = COLLECT(exe, a.binaries, a.datas, name="ClearFrame", upx=False)
