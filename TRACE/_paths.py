"""Where TRACE keeps the files it writes at runtime (models, startup log).

Stdlib-only on purpose: ``startup_log`` and ``fetch_assets`` import this
before anything heavy is loaded, and it must never be the thing that fails.

Three layouts:

- **Dev checkout** — everything lives in the ``TRACE/`` source folder.
- **Frozen Windows** — next to ``TRACE.exe``. The per-user Inno Setup
  install dir (``%LocalAppData%\\Programs\\TRACE``) is writable, and
  existing installs already have their models there.
- **Frozen macOS** — a ``.app`` bundle must not be written into: it is
  code-signed (any change invalidates the signature), it may be run from a
  read-only disk image or a Gatekeeper-translocated path, and replacing the
  app on update would throw away the 1.6 GB model download. Use the
  standard per-user locations instead.
"""

from __future__ import annotations

import sys
from pathlib import Path

_TRACE_DIR = Path(__file__).resolve().parent


def is_frozen() -> bool:
    """True inside a PyInstaller bundle."""
    return bool(getattr(sys, "frozen", False)) and hasattr(sys, "_MEIPASS")


def _is_frozen_mac() -> bool:
    return is_frozen() and sys.platform == "darwin"


def models_dir() -> Path:
    """Folder the downloaded model weights live in."""
    if _is_frozen_mac():
        return Path.home() / "Library" / "Application Support" / "TRACE" / "models"
    if is_frozen():
        return Path(sys.executable).resolve().parent / "TRACE" / "models"
    return _TRACE_DIR / "models"


def log_dir() -> Path:
    """Folder ``trace_startup.log`` is written to."""
    if _is_frozen_mac():
        return Path.home() / "Library" / "Logs" / "TRACE"
    if is_frozen():
        return Path(sys.executable).resolve().parent
    return _TRACE_DIR
