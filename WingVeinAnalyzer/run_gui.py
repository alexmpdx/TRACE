#!/usr/bin/env python3
"""Entry point for the WingVeinAnalyzer step-by-step GUI."""

import sys
from pathlib import Path

# Ensure the parent directory is on sys.path so that
# `import WingVeinAnalyzer.*` works when running from inside
# the WingVeinAnalyzer/ directory.
_parent = str(Path(__file__).resolve().parent.parent)
if _parent not in sys.path:
    sys.path.insert(0, _parent)

from WingVeinAnalyzer.gui import main

if __name__ == "__main__":
    main()
