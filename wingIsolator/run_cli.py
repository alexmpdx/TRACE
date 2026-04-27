#!/usr/bin/env python3
"""Entry point for the wingIsolator CLI. Sets up sys.path for sibling packages."""

import sys
from pathlib import Path

_project_root = str(Path(__file__).resolve().parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

_mtj_dir = str(Path(__file__).resolve().parent.parent / "modelTOjson")
if _mtj_dir not in sys.path:
    sys.path.insert(0, _mtj_dir)

from wingIsolator.cli import main

if __name__ == "__main__":
    sys.exit(main())
