#!/usr/bin/env python3
"""Entry point for the TRACE combined pipeline GUI."""

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

_idf_dir = str(Path(__file__).resolve().parent.parent / "identifyFeatures")
if _idf_dir not in sys.path:
    sys.path.insert(0, _idf_dir)

_rot_dir = str(Path(__file__).resolve().parent.parent / "wingRotator")
if _rot_dir not in sys.path:
    sys.path.insert(0, _rot_dir)

_mm_dir = str(Path(__file__).resolve().parent.parent / "measurementMaker")
if _mm_dir not in sys.path:
    sys.path.insert(0, _mm_dir)

_se_dir = str(Path(__file__).resolve().parent.parent / "scaleEstimator")
if _se_dir not in sys.path:
    sys.path.insert(0, _se_dir)

from TRACE.gui import main

if __name__ == "__main__":
    main()
