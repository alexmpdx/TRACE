#!/usr/bin/env python3
"""Entry point for the preprocessing pipeline GUI."""

# OpenMP duplicate-library guard. See TRACE/run_gui.py for the full
# rationale — bundled torch + onnxruntime each pull in their own OpenMP
# runtime, and the second to load aborts on Windows.
import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import sys
from pathlib import Path

# Add sibling package directories to sys.path
_project_root = str(Path(__file__).resolve().parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

_hinge_dir = str(Path(__file__).resolve().parent.parent / "HingeChopper")
if _hinge_dir not in sys.path:
    sys.path.insert(0, _hinge_dir)

_mtj_dir = str(Path(__file__).resolve().parent.parent / "modelTOjson")
if _mtj_dir not in sys.path:
    sys.path.insert(0, _mtj_dir)

_rot_dir = str(Path(__file__).resolve().parent.parent / "wingRotator")
if _rot_dir not in sys.path:
    sys.path.insert(0, _rot_dir)

_ll_dir = str(Path(__file__).resolve().parent.parent / "LandmarkLocator")
if _ll_dir not in sys.path:
    sys.path.insert(0, _ll_dir)

from preprocessing.gui import main

if __name__ == "__main__":
    main()
