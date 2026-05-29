"""TRACE app icon loader.

Two reasons this is its own module instead of inline in gui.py / run_gui.py:

1. **napari clobbers the app icon.** ``_QtMainWindow.__init__`` in napari
   calls ``app.setWindowIcon(_svg_path_to_icon(...))`` unconditionally, so
   the moment LandmarkPickerWidget constructs its embedded viewer our
   ``QApplication.setWindowIcon()`` from earlier is overwritten with the
   napari logo. The workaround is (a) re-set the app icon after the main
   window is built, and (b) call ``setWindowIcon`` on each top-level
   window we own — window-level icons override the app-wide one and are
   immune to later ``QApplication.setWindowIcon`` changes.

2. **Frozen-build SVG plugin reliability.** PyInstaller bundles PyQt5's
   ``QtSvg`` import for us (pinned in trace.spec) but the ``qsvgicon``
   Qt *image plugin* — which is what ``QIcon("foo.svg")`` calls into
   under the hood — sometimes doesn't make it into the bundle. Rendering
   the SVG ourselves via ``QSvgRenderer`` → ``QPixmap`` → ``QIcon.addPixmap``
   sidesteps that plugin entirely; we get a valid icon every time.

The OS-theme detection picks the white-circle LogoThick_dark.svg for
dark-mode and the black-circle LogoThick_light.svg for light mode —
the chrome behind the icon (title bar, taskbar, alt-tab) is rendered
by the OS, not Qt, so it follows the OS theme regardless of our
palette. macOS, Windows, and Linux are handled via TRACE.theme's
shared os_is_dark() detector.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional


def app_logo_path() -> Path:
    """Resolve the OS-theme-appropriate logo SVG path (frozen + dev safe)."""
    if getattr(sys, "frozen", False) and hasattr(sys, "_MEIPASS"):
        base = Path(sys._MEIPASS) / "TRACE" / "GUI_images"
    else:
        base = Path(__file__).resolve().parent / "GUI_images"
    # Defer to the shared cross-platform detector — it's authoritative
    # on macOS / Windows and falls back to None on Linux (we treat None
    # as light, matching the previous behavior on Windows when the
    # registry read failed).
    from TRACE.theme import os_is_dark

    variant = "LogoThick_dark.svg" if os_is_dark() is True else "LogoThick_light.svg"
    return base / "logo" / variant


def make_app_icon() -> Optional["object"]:
    """Build a multi-size QIcon by rendering the logo SVG via QSvgRenderer.

    Returns a fully-populated ``QIcon`` (Windows shell icon sizes 16-256
    rendered into individual pixmaps) or ``None`` if the SVG is missing or
    fails to parse. Callers should ``setWindowIcon(icon)`` on each
    top-level widget they own — the app-wide icon gets clobbered by
    napari, but per-window icons stick.
    """
    from PyQt5.QtCore import Qt
    from PyQt5.QtGui import QIcon, QPainter, QPixmap
    from PyQt5.QtSvg import QSvgRenderer

    svg_path = app_logo_path()
    if not svg_path.is_file():
        return None
    renderer = QSvgRenderer(str(svg_path))
    if not renderer.isValid():
        return None
    icon = QIcon()
    for size in (16, 24, 32, 48, 64, 128, 256):
        pixmap = QPixmap(size, size)
        pixmap.fill(Qt.transparent)
        painter = QPainter(pixmap)
        renderer.render(painter)
        painter.end()
        icon.addPixmap(pixmap)
    return icon
