"""Per-output checkbox tooltip helpers.

Qt tooltips support HTML; when an example image is bundled for a given
output key (in ``TRACE/GUI_images/``), render the tooltip as ``<img>``
markup pointing at a cached PNG. TIFF sources are converted to PNG once
into a tempdir cache so subsequent loads are instant. Keys without a
bundled image fall back to the caller-supplied text tooltip.

Used by both the main window (final outputs checkboxes) and the Settings
tab (intermediate outputs checkboxes).
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path
from typing import Optional

# Source folder for bundled example overlays.
#
# In a PyInstaller onedir bundle, __file__ for TRACE.output_tooltips can
# resolve to a path inside an archive that QImageReader can't actually
# read from (Qt's tooltip <img> handler needs a real filesystem path).
# Use sys._MEIPASS — the temp extraction dir PyInstaller exposes — so
# we point at the unpacked GUI_images directory on disk.
if getattr(sys, "frozen", False) and hasattr(sys, "_MEIPASS"):
    _GUI_IMAGES_DIR = Path(sys._MEIPASS) / "TRACE" / "GUI_images"
else:
    _GUI_IMAGES_DIR = Path(__file__).resolve().parent / "GUI_images"
# Cache folder for TIFF→PNG conversions. Lives in tempdir so the source
# tree stays clean. Re-converted at startup if the source TIFF is newer.
_CACHE_DIR = Path(tempfile.gettempdir()) / "trace_tooltip_cache"

# Mapping from OUTPUT_TYPES key → example image filename in GUI_images/.
# Keys without an entry (csv, geojson, …) fall back to the text tooltip.
_OUTPUT_IMAGES: dict[str, str] = {
    "vein_overlay": "vein_overlay.png",
    "intervein_overlay": "intervein_overlay.png",
    "ap_overlay": "ap_overlay.png",
    "cv_ratio_overlay": "cv_ratio_overlay.png",
    "landmarks_overlay": "landmarks_overlay.png",
    "segmentation_overlay": "segmentation_overlay.png",
    "chopped_image": "chopped.png",
    "wing_isolated_image": "isolated.png",
}

# Width (px) for the rendered <img> in the tooltip. Qt scales the image to
# fit. 400 px is roughly half a typical screen — wide enough to read fine
# vein detail, narrow enough to fit beside the source checkbox.
_TOOLTIP_IMAGE_WIDTH = 400


def _ensure_png(src: Path) -> Optional[Path]:
    """Return a PNG path for ``src``. PNGs are passed through; TIFFs (and
    anything else cv2 can decode) are converted into ``_CACHE_DIR``.

    Returns None when the source is missing or conversion fails.
    """
    if not src.is_file():
        return None
    if src.suffix.lower() == ".png":
        return src
    _CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cached = _CACHE_DIR / (src.stem + ".png")
    if cached.is_file() and cached.stat().st_mtime >= src.stat().st_mtime:
        return cached
    try:
        import cv2

        from TRACE.psd_loader import imread_any

        img = imread_any(src)
        if img is None:
            return None
        cv2.imwrite(str(cached), img)
        return cached
    except Exception:
        return None


def output_tooltip_html(key: str, fallback_text: str = "") -> str:
    """Return tooltip markup for an output checkbox.

    If ``key`` has a bundled example image, returns ``<img src="..." width="...">``
    pointing at the cached PNG. Otherwise returns ``fallback_text`` unchanged.
    """
    filename = _OUTPUT_IMAGES.get(key)
    if not filename:
        return fallback_text
    png = _ensure_png(_GUI_IMAGES_DIR / filename)
    if png is None:
        return fallback_text
    # file:// URL is the most portable way to embed a local image in a Qt
    # tooltip — QTextDocument resolves it via QImageReader.
    return f'<img src="file://{png.as_posix()}" width="{_TOOLTIP_IMAGE_WIDTH}">'
