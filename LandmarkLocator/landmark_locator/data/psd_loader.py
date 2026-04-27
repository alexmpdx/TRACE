"""cv2.imread-compatible loader with extended-format support.

Handles, in addition to formats cv2 reads natively:
  - .psd / .psb  (psd-tools, flattened composite)
  - .heic / .heif (pillow-heif)
  - .raw / camera RAW (.dng .nef .cr2 .cr3 .arw .raf .orf .pef .rw2 .srw) via rawpy
  - .svg (cairosvg, requires the system Cairo library)

Returns a BGR / BGRA / Gray ndarray matching the cv2 flag. 16-bit / CMYK / Lab /
indexed modes are coerced to 8-bit on load.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Union

import cv2
import numpy as np

_PSD_EXTS = {".psd", ".psb"}
_HEIC_EXTS = {".heic", ".heif"}
_SVG_EXTS = {".svg"}
_RAW_EXTS = {
    ".raw",
    ".dng",
    ".nef",
    ".cr2",
    ".cr3",
    ".arw",
    ".raf",
    ".orf",
    ".pef",
    ".rw2",
    ".srw",
}


def imread_any(path: Union[str, Path], flag: int = cv2.IMREAD_COLOR) -> Optional[np.ndarray]:
    p = str(path)
    ext = Path(p).suffix.lower()
    if ext in _PSD_EXTS:
        return _imread_psd(p, flag)
    if ext in _HEIC_EXTS:
        return _imread_heic(p, flag)
    if ext in _SVG_EXTS:
        return _imread_svg(p, flag)
    if ext in _RAW_EXTS:
        return _imread_raw(p, flag)
    return cv2.imread(p, flag)


def _pil_to_cv(pil, flag: int) -> np.ndarray:
    """Convert a PIL Image into a cv2-shaped ndarray for the requested flag."""
    if flag == cv2.IMREAD_GRAYSCALE:
        return np.array(pil.convert("L"))
    if flag == cv2.IMREAD_UNCHANGED:
        mode = pil.mode
        if mode in ("RGBA", "LA") or (mode == "P" and "transparency" in pil.info):
            arr = np.array(pil.convert("RGBA"))
            return cv2.cvtColor(arr, cv2.COLOR_RGBA2BGRA)
        if mode == "L":
            return np.array(pil)
        arr = np.array(pil.convert("RGB"))
        return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
    arr = np.array(pil.convert("RGB"))
    return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)


def _imread_psd(path: str, flag: int) -> Optional[np.ndarray]:
    try:
        from psd_tools import PSDImage
    except ImportError as exc:  # pragma: no cover
        raise ImportError("Reading .psd files requires psd-tools. Install with: pip install psd-tools") from exc

    try:
        psd = PSDImage.open(path)
    except Exception:
        return None
    pil = psd.composite()
    if pil is None:
        return None
    return _pil_to_cv(pil, flag)


def _imread_heic(path: str, flag: int) -> Optional[np.ndarray]:
    try:
        import pillow_heif
        from PIL import Image
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "Reading .heic/.heif files requires pillow-heif. Install with: pip install pillow-heif"
        ) from exc

    pillow_heif.register_heif_opener()
    try:
        pil = Image.open(path)
    except Exception:
        return None
    return _pil_to_cv(pil, flag)


def _imread_svg(path: str, flag: int) -> Optional[np.ndarray]:
    """Rasterize an SVG to a BGR ndarray. Default DPI 300, no resizing."""
    try:
        import cairosvg
    except (ImportError, OSError) as exc:  # pragma: no cover
        raise ImportError(
            "Reading .svg files requires cairosvg + the system Cairo library. "
            "On macOS: `brew install cairo && pip install cairosvg`. "
            f"(Underlying error: {exc})"
        ) from exc

    from io import BytesIO

    from PIL import Image

    try:
        png_bytes = cairosvg.svg2png(url=path, dpi=300)
    except Exception:
        return None
    pil = Image.open(BytesIO(png_bytes))
    return _pil_to_cv(pil, flag)


def _imread_raw(path: str, flag: int) -> Optional[np.ndarray]:
    """Decode a camera RAW via rawpy.postprocess() (auto white balance, sRGB)."""
    try:
        import rawpy
    except ImportError as exc:  # pragma: no cover
        raise ImportError("Reading camera RAW files requires rawpy. Install with: pip install rawpy") from exc

    try:
        with rawpy.imread(path) as raw:
            rgb = raw.postprocess(use_camera_wb=True, no_auto_bright=False, output_bps=8)
    except Exception:
        return None
    if rgb is None:
        return None
    if flag == cv2.IMREAD_GRAYSCALE:
        return cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    if flag == cv2.IMREAD_UNCHANGED and rgb.ndim == 2:
        return rgb
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
