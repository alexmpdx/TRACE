# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller spec for EZcheezeMeasure Windows executable."""

import os

block_cipher = None

spec_dir = os.path.dirname(os.path.abspath(SPEC))
landmark_locator_pkg = os.path.join(spec_dir, "..", "LandmarkLocator", "landmark_locator")
model_path = os.path.join(spec_dir, "trained_model", "landmark_model_grace.5.pt")

a = Analysis(
    [os.path.join(spec_dir, "run_pipeline.py")],
    pathex=[
        spec_dir,
        os.path.join(spec_dir, "..", "LandmarkLocator"),
    ],
    binaries=[],
    datas=[
        (model_path, "trained_model"),
        (os.path.join(spec_dir, "measure_landmarks.py"), "."),
        # Bundle the landmark_locator package source directly
        (landmark_locator_pkg, "landmark_locator"),
    ],
    hiddenimports=[
        "yaml",
        "cv2",
        "numpy",
        "torch",
        "torchvision",
        "torchvision.models",
        "torchvision.models.resnet",
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        "matplotlib",
        "PyQt5",
        "scipy",
        "pandas",
        "albumentations",
        "sklearn",
        "IPython",
        "jupyter",
        "notebook",
    ],
    noarchive=False,
    optimize=0,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="EZcheezeMeasure",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=True,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name="EZcheezeMeasure",
)
