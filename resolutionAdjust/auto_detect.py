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


def _read_um_per_px_from_tiff(path: Path) -> Optional[float]:
    """Best-effort read of µm/px from a single TIFF.

    Tries (in order): OME-XML PhysicalSizeX in the ImageDescription tag, then
    XResolution + ResolutionUnit. Returns None when neither yields a usable
    value.
    """
    try:
        import tifffile
    except ImportError:
        logger.warning("tifffile not installed; auto-detect cannot read TIFF metadata")
        return None

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
                            if factor is None:
                                # Fall through to TIFF resolution tags.
                                pass
                            elif size > 0:
                                return size * factor
                        except ValueError:
                            pass

            xres = tags.get("XResolution")
            unit = tags.get("ResolutionUnit")
            if xres is not None and unit is not None:
                val = xres.value
                if isinstance(val, tuple) and len(val) == 2:
                    num, den = val
                    if den:
                        res = num / den
                        return _resolution_to_um_per_px(float(res), int(unit.value))
                elif isinstance(val, (int, float)) and val > 0:
                    return _resolution_to_um_per_px(float(val), int(unit.value))
    except Exception as exc:
        logger.debug("auto-detect: failed to read %s: %s", path.name, exc)
        return None
    return None


def autodetect_um_per_px_from_folder(folder: Path) -> tuple[Optional[float], int, int]:
    """Average µm/px over the TIFFs in `folder` that carry parseable metadata.

    Walks the folder one level deep (no recursion) and only considers .tif /
    .tiff files — other formats rarely carry physical-size metadata. Returns
    (average_um_per_px or None, n_with_metadata, n_total). When zero images
    yield a value the caller should treat the result as a soft failure.
    """
    folder = Path(folder)
    if not folder.is_dir():
        raise NotADirectoryError(f"Not a directory: {folder}")

    candidates = sorted(p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in _TIFF_EXTS)
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
