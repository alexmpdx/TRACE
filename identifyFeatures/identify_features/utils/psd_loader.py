"""cv2.imread-compatible loader with Photoshop .psd support.

Flattens PSD composite via psd-tools and returns a BGR/BGRA/Gray ndarray
matching the cv2 flag. 16-bit / CMYK / Lab / indexed modes are coerced to
8-bit RGB on load.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Union

import cv2
import numpy as np


def imread_any(path: Union[str, Path], flag: int = cv2.IMREAD_COLOR) -> Optional[np.ndarray]:
    p = str(path)
    if p.lower().endswith(".psd") or p.lower().endswith(".psb"):
        return _imread_psd(p, flag)
    return cv2.imread(p, flag)


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
