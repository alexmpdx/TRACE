"""Auto-detect µm/px from a folder of training images.

Reads the standard TIFF resolution tags (XResolution + ResolutionUnit) and the
OME-XML PhysicalSizeX field when present. Averages over the images that carry
parseable metadata; returns (avg or None, n_with_metadata, n_total) so the
caller can surface "5/12 images had metadata" in the GUI.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

_TIFF_EXTS = {".tif", ".tiff"}

# Unit string -> conversion factor to micrometers.
_UNIT_TO_UM = {
    "µm": 1.0,
    "um": 1.0,
    "micron": 1.0,
    "microns": 1.0,
    "micrometer": 1.0,
    "micrometers": 1.0,
    "mm": 1000.0,
    "millimeter": 1000.0,
    "nm": 1.0 / 1000.0,
    "nanometer": 1.0 / 1000.0,
}

# Plausibility band for microscopy µm/px values on a fly wing. A 100X
# super-resolution lens caps out around 0.04 µm/px; a whole-slide macro
# lens is around 10 µm/px. The 0.02 – 50 band is generous either way.
#
# The specific failure mode this guards is a very common bad-metadata
# pattern: capture / conversion software writes the generic screen-DPI
# tag ``XResolution=96, ResolutionUnit=inch`` when the operator didn't
# calibrate, which converts to 25400/96 ≈ 264 µm/px. That value is
# ~500× coarser than real wing microscopy, so resolutionAdjust would
# try to upscale each image ~500× per axis (250 000× area) and blow
# OpenCV's allocator past a terabyte. The same nonsense happens with
# 72 dpi → 353 µm/px. Anything outside this band is rejected as bad
# metadata regardless of source (TIFF tags OR OME-XML PhysicalSizeX).
_MIN_PLAUSIBLE_UM_PER_PX = 0.02
_MAX_PLAUSIBLE_UM_PER_PX = 50.0


def _is_plausible_um_per_px(v: Optional[float]) -> bool:
    """True when ``v`` is inside the microscopy-plausibility band."""
    return v is not None and _MIN_PLAUSIBLE_UM_PER_PX <= v <= _MAX_PLAUSIBLE_UM_PER_PX


def _resolution_to_um_per_px(res_value: float, unit: int) -> Optional[float]:
    """Convert TIFF XResolution + ResolutionUnit to µm/px.

    ResolutionUnit: 1=no unit, 2=inch, 3=centimeter.
    XResolution stores pixels-per-unit, so µm/px = (µm-per-unit) / res_value.
    """
    if res_value <= 0:
        return None
    if unit == 3:  # centimeter
        return 10000.0 / res_value
    if unit == 2:  # inch
        return 25400.0 / res_value
    return None


def _read_um_per_px_from_tiff(path: Path, *, allow_implausible: bool = False) -> Optional[float]:
    """Best-effort read of µm/px from a single TIFF.

    Tries (in order): OME-XML PhysicalSizeX in the ImageDescription tag, then
    XResolution + ResolutionUnit. Returns None when neither yields a usable
    value.

    Plausibility filter: by default, values outside
    [``_MIN_PLAUSIBLE_UM_PER_PX``, ``_MAX_PLAUSIBLE_UM_PER_PX``] are treated
    as bad metadata (see the module-level constants for the specific failure
    mode this guards) and return None with a WARNING log. Callers that need
    to see the raw reading — the pre-flight check in TRACE, which
    distinguishes "no metadata" from "implausible metadata" for the user
    dialog — can pass ``allow_implausible=True``.
    """
    try:
        import tifffile
    except ImportError:
        logger.warning("tifffile not installed; auto-detect cannot read TIFF metadata")
        return None

    reading: Optional[float] = None
    try:
        with tifffile.TiffFile(str(path)) as tif:
            page = tif.pages[0]
            tags = page.tags

            desc = tags.get("ImageDescription")
            if desc is not None:
                val = desc.value if hasattr(desc, "value") else None
                if isinstance(val, str) and "PhysicalSizeX" in val:
                    m_size = re.search(r'PhysicalSizeX="([0-9.eE+\-]+)"', val)
                    m_unit = re.search(r'PhysicalSizeXUnit="([^"]+)"', val)
                    if m_size:
                        try:
                            size = float(m_size.group(1))
                            unit_str = m_unit.group(1).strip().lower() if m_unit else "µm"
                            factor = _UNIT_TO_UM.get(unit_str, _UNIT_TO_UM.get(unit_str.replace("μ", "µ")))
                            if factor is not None and size > 0:
                                reading = size * factor
                        except ValueError:
                            pass

            if reading is None:
                xres = tags.get("XResolution")
                unit = tags.get("ResolutionUnit")
                if xres is not None and unit is not None:
                    val = xres.value
                    if isinstance(val, tuple) and len(val) == 2:
                        num, den = val
                        if den:
                            res = num / den
                            reading = _resolution_to_um_per_px(float(res), int(unit.value))
                    elif isinstance(val, (int, float)) and val > 0:
                        reading = _resolution_to_um_per_px(float(val), int(unit.value))
    except Exception as exc:
        logger.debug("auto-detect: failed to read %s: %s", path.name, exc)
        return None

    if reading is None or reading <= 0:
        return None
    if allow_implausible:
        return reading
    if not _is_plausible_um_per_px(reading):
        logger.warning(
            "auto-detect: %s reported µm/px=%.4f, outside plausibility band "
            "[%.2f, %.2f] — rejecting as bad metadata (common cause: capture "
            "software wrote a screen-DPI default like 72 or 96 dpi instead of "
            "a real physical calibration). Falling back to manual scale.",
            path.name,
            reading,
            _MIN_PLAUSIBLE_UM_PER_PX,
            _MAX_PLAUSIBLE_UM_PER_PX,
        )
        return None
    return reading


def autodetect_um_per_px_from_folder(folder: Path) -> tuple[Optional[float], int, int]:
    """Average µm/px over the TIFFs in `folder` that carry parseable metadata.

    Discovery order — matches the calibration and main-window "Include
    subfolders" behavior so users can point at either a leaf folder full
    of TIFFs or a parent whose TIFFs sit in subfolders:
      1. Direct children (fast path; historical behavior).
      2. Recursive walk via ``rglob`` — engaged only when the direct-child
         scan returned zero TIFFs. This is what TRACE's main window does
         when "Include subfolders" is checked, and it's what a user who
         hands us a batch parent like
         ``<batch>/40X_magnification_Leica_ome-tiff`` (whose TIFFs live
         in ``part1/partA/…``) expects.

    Returns (average_um_per_px or None, n_with_metadata, n_total). When
    zero images yield a value the caller should treat the result as a
    soft failure.
    """
    folder = Path(folder)
    if not folder.is_dir():
        raise NotADirectoryError(f"Not a directory: {folder}")

    candidates = sorted(p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in _TIFF_EXTS)
    if not candidates:
        # Nothing at top level → recurse. Fast to fall through on folders
        # that DO have top-level TIFFs (rglob is skipped) and lets nested
        # batches average their metadata correctly.
        candidates = sorted(
            p for p in folder.rglob("*") if p.is_file() and p.suffix.lower() in _TIFF_EXTS
        )
    n_total = len(candidates)

    values: list[float] = []
    for p in candidates:
        v = _read_um_per_px_from_tiff(p)
        if v is not None and v > 0:
            values.append(v)

    if not values:
        return (None, 0, n_total)
    avg = sum(values) / len(values)
    return (avg, len(values), n_total)
