# -*- mode: python ; coding: utf-8 -*-
#
# PyInstaller build spec for Voice Flow.
#
#   pip install pyinstaller
#   pyinstaller VoiceFlow.spec
#
# Produces dist/VoiceFlow.exe — a single, windowed (no-console) executable.
# The speech model is NOT bundled; it is downloaded on first run into
# %LOCALAPPDATA%\VoiceFlow\models. The build is CPU-only: CUDA/cuDNN DLLs that
# the ctranslate2 wheel may carry are stripped out below to keep the exe small
# and to avoid shipping gigabytes that a CPU build never touches.

from PyInstaller.utils.hooks import collect_all

datas, binaries, hiddenimports = [], [], []

# Packages with native libraries / data files PyInstaller can't infer alone.
for pkg in (
    "faster_whisper",
    "ctranslate2",
    "av",
    "sounddevice",
    "huggingface_hub",
    "tokenizers",
):
    d, b, h = collect_all(pkg)
    datas += d
    binaries += b
    hiddenimports += h

# Strip CUDA runtime libraries — this is a CPU-only build. Dropping them is safe
# because int8/CPU inference never loads them; it just saves ~1-2 GB.
_CUDA_MARKERS = ("cudnn", "cublas", "cudart", "cufft", "cuda", "nvrtc", "nvinfer")


def _is_cuda(entry):
    name = entry[0].lower().replace("\\", "/").split("/")[-1]
    return any(marker in name for marker in _CUDA_MARKERS)


binaries = [b for b in binaries if not _is_cuda(b)]

hiddenimports += [
    "pynput.keyboard._win32",
    "pynput.mouse._win32",
]

a = Analysis(
    ["app.py"],
    pathex=[],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        "tkinter",
        "matplotlib",
        "PySide6.QtQml",
        "PySide6.QtQuick",
        "PySide6.QtWebEngineCore",
        "PySide6.Qt3DCore",
        "PySide6.QtNetwork",
        "PySide6.QtMultimedia",
    ],
    noarchive=False,
)

pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name="VoiceFlow",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,          # windowed app — no terminal window
    disable_windowed_traceback=False,
    icon=None,              # drop a path to an .ico here to brand the exe
)
