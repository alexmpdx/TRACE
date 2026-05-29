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


def ensure_app_icon_ico(pref: Optional[IconPreference] = None) -> Optional[Path]:
    """Render the active LogoThick SVG to a multi-size .ico in the user cache.

    Used by the Add-desktop-shortcut button so the shortcut's icon
    follows the user's IconPreference rather than the icon embedded
    in TRACE.exe by PyInstaller (which is baked in at build time and
    doesn't change with preference).

    The ICO is written to a stable cache path so that existing
    shortcuts keep working across preference changes: when the user
    picks a different variant we just overwrite the same file in
    place, and Windows re-reads it the next time it draws the icon.

    Returns the cached path on success, ``None`` if the SVG was
    missing or Pillow / Qt couldn't render. Callers should fall back
    to ``sys.executable`` as the IconLocation in the None case.
    """
    if pref is None:
        pref = _read_preference()
    svg_name = _resolve_variant(pref)
    if getattr(sys, "frozen", False) and hasattr(sys, "_MEIPASS"):
        svg_path = Path(sys._MEIPASS) / "TRACE" / "GUI_images" / "logo" / svg_name
    else:
        svg_path = Path(__file__).resolve().parent / "GUI_images" / "logo" / svg_name
    if not svg_path.is_file():
        return None

    try:
        from PIL import Image
        from PyQt5.QtCore import QStandardPaths, Qt
        from PyQt5.QtGui import QImage, QPainter
        from PyQt5.QtSvg import QSvgRenderer
    except Exception:
        return None

    renderer = QSvgRenderer(str(svg_path))
    if not renderer.isValid():
        return None

    base = QStandardPaths.writableLocation(QStandardPaths.AppLocalDataLocation)
    cache_dir = Path(base) if base else Path.home() / ".local" / "share" / "TRACE"
    cache_dir = cache_dir / "icons"
    try:
        cache_dir.mkdir(parents=True, exist_ok=True)
    except OSError:
        return None
    # Same filename regardless of variant — overwritten on preference
    # change so a previously-created desktop shortcut auto-updates
    # without the user having to recreate it.
    ico_path = cache_dir / "TRACE_app_icon.ico"

    # Render the SVG separately at every Windows shell size. Doing it
    # this way (vector → bitmap once per size) keeps each frame crisp
    # at its native resolution; the previous approach (render at 256,
    # let Pillow BICUBIC-downsample) blurred the thin logo strokes at
    # 16/24/32 px, which is what Windows actually shows on the desktop
    # and taskbar.
    sizes = [(16, 16), (24, 24), (32, 32), (48, 48), (64, 64), (128, 128), (256, 256)]

    def _render_pil(size_px: int) -> "Image.Image":
        # QImage with Format_ARGB32 lets us extract straight RGBA bytes
        # without going through a PNG round-trip.
        qimg = QImage(size_px, size_px, QImage.Format_ARGB32)
        qimg.fill(Qt.transparent)
        painter = QPainter(qimg)
        try:
            painter.setRenderHint(QPainter.Antialiasing, True)
            painter.setRenderHint(QPainter.SmoothPixmapTransform, True)
            renderer.render(painter)
        finally:
            painter.end()
        # QImage byte order is BGRA on little-endian systems for
        # Format_ARGB32; convert to RGBA for Pillow. constBits() needs
        # a fixed-size buffer for Pillow's frombuffer to be safe.
        ptr = qimg.constBits()
        ptr.setsize(qimg.byteCount())
        rgba_bytes = bytes(memoryview(ptr).cast("B"))
        bgra = Image.frombuffer("RGBA", (size_px, size_px), rgba_bytes, "raw", "BGRA", 0, 1)
        return bgra.copy()  # detach from the Qt-owned buffer

    try:
        # Render each size from the vector source. Order matters: the
        # base image passed to .save() must be the LARGEST, because
        # Pillow's ICO writer silently skips any requested ``sizes``
        # entry whose dimensions exceed the base image's. That bug is
        # what produced the previous 546-byte single-frame ICO — the
        # base was 16×16, so 24/32/48/64/128/256 were all dropped.
        frames = [_render_pil(w) for (w, _h) in sizes]
        frames_largest_first = list(reversed(frames))
        base_frame = frames_largest_first[0]  # 256×256
        base_frame.save(
            str(ico_path),
            format="ICO",
            sizes=sizes,
            append_images=frames_largest_first[1:],
        )
    except Exception:
        return None
    return ico_path


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
