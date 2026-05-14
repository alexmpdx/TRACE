"""resolutionAdjust — rescale wing images to a target µm/px before DL inference.

Designed to slot into preprocessing as Stage -1, ahead of wing isolation and
landmark detection. Rescales each image so the DL models see something close to
the resolution they were trained on, then preserves the scale factor so callers
can transform any geometry produced downstream back to the original pixel grid.

Pass-through when the input's µm/px is already inside a user-defined tolerance
band around the target — avoids wasted resampling on near-match images.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class ResolutionAdjustResult:
    image_path: Path
    scale_factor: float
    original_shape: tuple[int, int]
    rescaled: bool
    target_um_per_px: float
    effective_um_per_px: float
    input_um_per_px: float
    ratio: float


def _read_image(path: Path) -> np.ndarray:
    """Read with cv2; fall back to preprocessing.psd_loader for exotic formats."""
    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if img is None:
        try:
            from preprocessing.psd_loader import imread_any

            img = imread_any(str(path), cv2.IMREAD_UNCHANGED)
        except Exception:
            img = None
    if img is None:
        raise IOError(f"Failed to read image: {path}")
    return img


def _write_image(path: Path, image: np.ndarray) -> Path:
    """Write image; coerce TIFFs to OME-TIFF (mirrors wing_rotator / HingeChopper)."""
    name_low = path.name.lower()
    is_tiff = path.suffix.lower() in (".tif", ".tiff") or name_low.endswith((".ome.tif", ".ome.tiff"))
    if is_tiff:
        try:
            import tifffile

            if image.ndim == 3 and image.shape[-1] == 3:
                rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                tifffile.imwrite(str(path), rgb, ome=True, photometric="rgb")
                return path
            if image.ndim == 3 and image.shape[-1] == 4:
                rgba = cv2.cvtColor(image, cv2.COLOR_BGRA2RGBA)
                tifffile.imwrite(str(path), rgba, ome=True, photometric="rgb")
                return path
            if image.ndim == 2:
                tifffile.imwrite(str(path), image, ome=True, photometric="minisblack")
                return path
            tifffile.imwrite(str(path), image, ome=True)
            return path
        except Exception:
            pass
    cv2.imwrite(str(path), image)
    return path


def _clean_stem(image_path: Path) -> str:
    """Strip .ome.tif / .ome.tiff compound suffixes."""
    name = image_path.name
    low = name.lower()
    if low.endswith(".ome.tif"):
        return name[: -len(".ome.tif")]
    if low.endswith(".ome.tiff"):
        return name[: -len(".ome.tiff")]
    return image_path.stem


def _resample(image: np.ndarray, scale_factor: float) -> np.ndarray:
    """Resize by scale_factor; INTER_AREA when shrinking, INTER_CUBIC when growing."""
    h, w = image.shape[:2]
    new_w = max(int(round(w * scale_factor)), 2)
    new_h = max(int(round(h * scale_factor)), 2)
    interp = cv2.INTER_AREA if scale_factor < 1.0 else cv2.INTER_CUBIC
    return cv2.resize(image, (new_w, new_h), interpolation=interp)


def adjust_resolution(
    image_path: Path,
    input_um_per_px: float,
    target_um_per_px: float,
    output_dir: Path,
    tolerance_low: float = 0.85,
    tolerance_high: float = 1.15,
) -> ResolutionAdjustResult:
    """Rescale `image_path` toward `target_um_per_px` when the ratio falls outside
    [tolerance_low, tolerance_high].

    Ratio is defined as `input_um_per_px / target_um_per_px`:
      - ratio > 1  → input is coarser than target → upscale (scale_factor = ratio)
      - ratio < 1  → input is finer than target   → downscale (scale_factor = ratio)
      - inside band → pass through, scale_factor = 1.0

    Returns:
        ResolutionAdjustResult. `image_path` is the rescaled file when a rescale
        actually happened, or the original input path otherwise. `scale_factor`
        is `new_pixels / old_pixels` (so multiplying coordinates by 1/scale_factor
        returns them to the original pixel grid).
    """
    image_path = Path(image_path)
    output_dir = Path(output_dir)
    if input_um_per_px <= 0:
        raise ValueError(f"input_um_per_px must be > 0 (got {input_um_per_px})")
    if target_um_per_px <= 0:
        raise ValueError(f"target_um_per_px must be > 0 (got {target_um_per_px})")
    if not (tolerance_low > 0 and tolerance_high >= tolerance_low):
        raise ValueError(
            f"tolerance band invalid: low={tolerance_low}, high={tolerance_high} " "(need 0 < low <= high)"
        )

    ratio = input_um_per_px / target_um_per_px

    img = _read_image(image_path)
    h, w = img.shape[:2]

    if tolerance_low <= ratio <= tolerance_high:
        logger.info(
            "resolutionAdjust: %s ratio=%.3f inside band [%.3f, %.3f]; pass-through",
            image_path.name,
            ratio,
            tolerance_low,
            tolerance_high,
        )
        return ResolutionAdjustResult(
            image_path=image_path,
            scale_factor=1.0,
            original_shape=(h, w),
            rescaled=False,
            target_um_per_px=target_um_per_px,
            effective_um_per_px=input_um_per_px,
            input_um_per_px=input_um_per_px,
            ratio=ratio,
        )

    scale_factor = ratio
    output_dir.mkdir(parents=True, exist_ok=True)
    rescaled = _resample(img, scale_factor)

    stem = _clean_stem(image_path)
    suffix = image_path.suffix
    if suffix.lower() in (".psd", ".psb"):
        out_path = output_dir / f"{stem}_resampled.ome.tif"
    else:
        out_path = output_dir / f"{stem}_resampled{suffix}"
    _write_image(out_path, rescaled)

    logger.info(
        "resolutionAdjust: %s ratio=%.3f scale_factor=%.4f %dx%d -> %dx%d (-> %s)",
        image_path.name,
        ratio,
        scale_factor,
        w,
        h,
        rescaled.shape[1],
        rescaled.shape[0],
        out_path.name,
    )

    return ResolutionAdjustResult(
        image_path=out_path,
        scale_factor=scale_factor,
        original_shape=(h, w),
        rescaled=True,
        target_um_per_px=target_um_per_px,
        effective_um_per_px=target_um_per_px,
        input_um_per_px=input_um_per_px,
        ratio=ratio,
    )


# ---------------------------------------------------------------------------
# Inverse transforms — undo the rescale on geometry / images at the end
# ---------------------------------------------------------------------------
def inverse_transform_coords(xy, scale_factor: float):
    """Multiply (x, y) or an iterable of them by 1/scale_factor.

    Returns the same shape the caller passed in — single tuple → tuple, list
    of tuples → list of tuples — to make this drop-in usable from both per-point
    and per-polyline contexts.
    """
    if scale_factor == 0:
        raise ValueError("scale_factor must be non-zero")
    inv = 1.0 / scale_factor
    if xy is None:
        return xy
    if isinstance(xy, (list, tuple)) and len(xy) >= 2 and isinstance(xy[0], (int, float)):
        if len(xy) == 2:
            return (xy[0] * inv, xy[1] * inv)
        return (xy[0] * inv, xy[1] * inv, *xy[2:])
    return [inverse_transform_coords(item, scale_factor) for item in xy]


def _inverse_transform_geojson_coords(coords, scale_factor: float):
    if not coords:
        return coords
    first = coords[0]
    if isinstance(first, (int, float)):
        inv = 1.0 / scale_factor
        x = float(coords[0]) * inv
        y = float(coords[1]) * inv
        if len(coords) > 2:
            return [x, y, coords[2]]
        return [x, y]
    return [_inverse_transform_geojson_coords(c, scale_factor) for c in coords]


def inverse_transform_geojson(data: dict, scale_factor: float) -> dict:
    """Multiply every geometry coordinate in a GeoJSON dict by 1/scale_factor."""
    if scale_factor == 0:
        raise ValueError("scale_factor must be non-zero")
    if scale_factor == 1.0:
        return data
    out = dict(data)
    new_features = []
    for feat in data.get("features", []):
        new_feat = dict(feat)
        geom = feat.get("geometry")
        if geom and "coordinates" in geom:
            new_geom = dict(geom)
            new_geom["coordinates"] = _inverse_transform_geojson_coords(geom["coordinates"], scale_factor)
            new_feat["geometry"] = new_geom
        new_features.append(new_feat)
    out["features"] = new_features
    return out


def inverse_rescale_wing_result(wing_result, scale_factor: float, um_per_px: Optional[float] = None) -> None:
    """In-place inverse rescale of an identifyFeatures `WingResult`.

    Maps every shapely geometry (vein centerlines + tissue polygons, intervein
    polygons, landmark points, wing outline) from rescaled-pixel space back to
    original-pixel space via `shapely.affinity.scale(g, 1/sf, 1/sf, origin=(0, 0))`.

    Also recomputes cached pixel/µm fields so downstream CSV / GeoJSON exporters
    see consistent values:
      - `vein.length_px = centerline.length`
      - `vein.length_um = length_px × um_per_px`  (when um_per_px is set)
      - `region.area_px2 = polygon.area`
      - `region.area_um2 = area_px2 × um_per_px²`
    """
    if scale_factor == 0:
        raise ValueError("scale_factor must be non-zero")
    if scale_factor == 1.0:
        return

    from shapely.affinity import scale as _sscale

    inv = 1.0 / scale_factor

    def _rescale(g):
        if g is None:
            return None
        try:
            if g.is_empty:
                return g
        except AttributeError:
            return g
        return _sscale(g, xfact=inv, yfact=inv, origin=(0, 0))

    for v in getattr(wing_result, "veins", []):
        v.centerline = _rescale(v.centerline)
        v.tissue_polygon = _rescale(v.tissue_polygon)
        if v.centerline is not None and not v.centerline.is_empty:
            v.length_px = float(v.centerline.length)
            if um_per_px is not None and um_per_px > 0:
                v.length_um = v.length_px * um_per_px

    for r in getattr(wing_result, "intervein_regions", []):
        r.polygon = _rescale(r.polygon)
        if r.polygon is not None and not r.polygon.is_empty:
            r.area_px2 = float(r.polygon.area)
            if um_per_px is not None and um_per_px > 0:
                r.area_um2 = r.area_px2 * (um_per_px**2)

    wing_result.wing_outline = _rescale(getattr(wing_result, "wing_outline", None))

    landmarks = getattr(wing_result, "landmarks", None) or {}
    for lm in landmarks.values():
        if getattr(lm, "point", None) is not None and not lm.point.is_empty:
            lm.point = _rescale(lm.point)


def inverse_resize_image(
    image: np.ndarray, scale_factor: float, target_shape: Optional[tuple[int, int]] = None
) -> np.ndarray:
    """Resize an image back to its pre-rescale resolution.

    When `target_shape` is given (h, w), resize exactly to that — avoids a 1-px
    drift from successive round-trips. Otherwise resize by 1/scale_factor.
    """
    if scale_factor == 0:
        raise ValueError("scale_factor must be non-zero")
    if scale_factor == 1.0:
        return image
    if target_shape is not None:
        h_t, w_t = target_shape
        new_w = max(int(w_t), 2)
        new_h = max(int(h_t), 2)
    else:
        inv = 1.0 / scale_factor
        h, w = image.shape[:2]
        new_w = max(int(round(w * inv)), 2)
        new_h = max(int(round(h * inv)), 2)
    interp = cv2.INTER_AREA if (new_w * new_h) < (image.shape[1] * image.shape[0]) else cv2.INTER_CUBIC
    return cv2.resize(image, (new_w, new_h), interpolation=interp)
