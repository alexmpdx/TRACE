# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller spec for EZcheezeMeasure Windows executable."""

import os
import sys
from pathlib import Path

block_cipher = None

# Paths
spec_dir = os.path.dirname(os.path.abspath(SPEC))
landmark_locator_dir = os.path.join(spec_dir, "..", "LandmarkLocator", "landmark_locator")
model_path = os.path.join(spec_dir, "trained_model", "landmark_model_grace.5.pt")

a = Analysis(
    [os.path.join(spec_dir, "run_pipeline.py")],
    pathex=[spec_dir],
    binaries=[],
    datas=[
        # Bundle the trained model checkpoint
        (model_path, "trained_model"),
        # Bundle measure_landmarks.py so run_pipeline can import it
        (os.path.join(spec_dir, "measure_landmarks.py"), "."),
    ],
    hiddenimports=[
        "landmark_locator",
        "landmark_locator.inference",
        "landmark_locator.inference.predict",
        "landmark_locator.models",
        "landmark_locator.models.unet",
        "landmark_locator.training",
        "landmark_locator.training.train",
        "landmark_locator.data",
        "landmark_locator.data.dataset",
        "measure_landmarks",
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
        # Exclude unnecessary large packages to reduce size
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
    console=True,  # Keep console for progress output
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
