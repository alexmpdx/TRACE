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
from enum import Enum
from pathlib import Path
from typing import Optional


class IconPreference(str, Enum):
    """User-facing preference for the app icon variant.

    Stored in QSettings under ``app_icon/preference``. SYSTEM resolves
    via ``TRACE.theme.os_is_dark`` (same detector the theme uses) so
    by default the icon follows the OS color scheme. LIGHT and DARK
    are explicit overrides — useful for users whose OS chrome theme
    doesn't match what they want behind the TRACE icon.
    """

    SYSTEM = "system"
    LIGHT = "light"
    DARK = "dark"


def _read_preference() -> IconPreference:
    """Read the saved icon preference from QSettings (defaults to SYSTEM)."""
    from PyQt5.QtCore import QSettings

    raw = str(QSettings("TRACE", "WingAnalysisPipeline").value("app_icon/preference", IconPreference.SYSTEM.value) or "")
    try:
        return IconPreference(raw)
    except ValueError:
        return IconPreference.SYSTEM


def _resolve_variant(pref: IconPreference) -> str:
    """Return the SVG filename for an IconPreference value."""
    if pref is IconPreference.LIGHT:
        return "LogoThick_light.svg"
    if pref is IconPreference.DARK:
        return "LogoThick_dark.svg"
    # SYSTEM — defer to the shared cross-platform OS-theme detector.
    # Authoritative on macOS / Windows, None on Linux (we treat None
    # as light, matching the prior behavior).
    from TRACE.theme import os_is_dark

    return "LogoThick_dark.svg" if os_is_dark() is True else "LogoThick_light.svg"


def app_logo_path() -> Path:
    """Resolve the logo SVG path for the active icon preference (frozen + dev safe)."""
    if getattr(sys, "frozen", False) and hasattr(sys, "_MEIPASS"):
        base = Path(sys._MEIPASS) / "TRACE" / "GUI_images"
    else:
        base = Path(__file__).resolve().parent / "GUI_images"
    return base / "logo" / _resolve_variant(_read_preference())


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
