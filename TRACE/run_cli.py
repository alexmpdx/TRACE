#!/usr/bin/env python3
"""Entry point for the TRACE combined pipeline CLI."""

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

from TRACE.cli import main

if __name__ == "__main__":
    main()
