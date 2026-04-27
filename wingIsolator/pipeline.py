"""
wingIsolator pipeline — isolate the centered wing from a multi-wing detection.

Library API used by TRACE / preprocessing / external callers. Given an image
and its segmentation GeoJSON (e.g. from modelTOjson), pick the polygon that
covers the image center (with a largest-polygon fallback), watershed-split it
if it is a merged collision of multiple wings, dilate the result by a small
fraction of its characteristic size, and emit a single-wing GeoJSON + masked
image.

Two entry points:

* :func:`isolate_main_wing` — file-in / files-out.  Returns an
  :class:`IsolationResult` dataclass.  Use this from CLI/GUI/TRACE when output
  files are wanted.
* :func:`isolate_in_memory` — in-memory variant. Takes a numpy image and a
  list of shapely polygons, returns the dilated main-wing polygon + binary
  mask + diagnostic counts.  Use this from TRACE to chain stages without
  round-tripping through disk.

Plus :func:`isolate_folder` which loops :func:`isolate_main_wing` over
matching images + geojsons in two directories.
"""

import json
import os
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional

import geojson
import numpy as np
import rasterio.features
from PIL import Image
from rasterio.transform import from_bounds
from scipy import ndimage as ndi
from shapely.geometry import MultiPolygon, Point, Polygon, mapping, shape
from skimage.feature import peak_local_max
from skimage.segmentation import watershed

# Reuse modelTOjson's reader so PSD / multi-band TIFFs are handled the same
# way as the rest of the pipeline. The sibling-package path is added by
# run_cli.py / TRACE's run_cli.py; we still try a relative fallback so that
# `python -m wingIsolator.pipeline` works in development.
try:
    from modeltojson import read_image as _modeltojson_read_image
except Exception:
    _mtj_dir = Path(__file__).resolve().parent.parent / "modelTOjson"
    if _mtj_dir.exists() and str(_mtj_dir) not in sys.path:
        sys.path.insert(0, str(_mtj_dir))
    try:
        from modeltojson import read_image as _modeltojson_read_image  # noqa: E402
    except Exception:
        _modeltojson_read_image = None


WING_CLASS_NAMES = ("wing",)
DEFAULT_OUTPUT_SUFFIX = "_main_wing"
SUPPORTED_IMAGE_EXTS = {
    ".tif",
    ".tiff",
    ".bmp",
    ".png",
    ".jpg",
    ".jpeg",
    ".psd",
    ".psb",
}


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------
@dataclass
class IsolationResult:
    """Outcome of isolating one image / geojson pair.

    `status` is "ok" on success or one of:
      - "no_wing_polygons" — geojson had no polygons matching the class filter
      - "empty_after_split" — watershed produced no nonzero label at center
      - "vectorize_failed"  — chosen mask could not be re-vectorized
      - "no_geojson"        — batch mode: no matching geojson found
      - "error"             — uncaught exception (see `error`)

    `masked_image_path` may differ from the requested path when the input is
    PSD/PSB (rewritten as PNG, since there's no good open-source PSD writer).
    """

    image_path: str
    geojson_path: Optional[str] = None
    masked_image_path: Optional[str] = None
    main_geojson_path: Optional[str] = None
    num_input_polygons: int = 0
    num_subwings: int = 0
    status: str = "error"
    error: Optional[str] = None

    def to_dict(self) -> dict:
        return asdict(self)


# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------
def load_image(path) -> np.ndarray:
    """Read an image as RGB uint8 (H, W, 3). Falls back to PIL if needed."""
    if _modeltojson_read_image is not None:
        return _modeltojson_read_image(str(path))
    return np.array(Image.open(str(path)).convert("RGB"))


def load_wing_polygons(geojson_path, class_names=WING_CLASS_NAMES) -> list[Polygon]:
    """Load wing polygons from a GeoJSON file as a list of shapely Polygons.

    Filters by `properties.class` (or `properties.classification.name`).
    Falls back to every polygon-like feature if no features match the filter.
    """
    with open(geojson_path) as f:
        fc = json.load(f)
    features = fc.get("features", [])

    def _matches(feature):
        props = feature.get("properties") or {}
        cls = props.get("class")
        if cls is None:
            cls = (props.get("classification") or {}).get("name")
        if cls is None:
            return False
        return any(name.lower() in str(cls).lower() for name in class_names)

    matched = [f for f in features if _matches(f)] or features

    polygons: list[Polygon] = []
    for feature in matched:
        geom = feature.get("geometry")
        if geom is None:
            continue
        try:
            shp = shape(geom)
        except Exception:
            continue
        if isinstance(shp, Polygon) and not shp.is_empty:
            polygons.append(shp)
        elif isinstance(shp, MultiPolygon):
            polygons.extend(p for p in shp.geoms if not p.is_empty)
    return polygons


# ---------------------------------------------------------------------------
# Main-wing selection
# ---------------------------------------------------------------------------
def select_main_polygon(polygons, center) -> Optional[Polygon]:
    """Polygon containing center, else largest by area."""
    if not polygons:
        return None
    pt = Point(center[0], center[1])
    containing = [p for p in polygons if p.contains(pt) or p.intersects(pt)]
    if containing:
        return max(containing, key=lambda p: p.area)
    return max(polygons, key=lambda p: p.area)


# ---------------------------------------------------------------------------
# Rasterize / vectorize helpers
# ---------------------------------------------------------------------------
def _identity_transform(image_shape):
    h, w = image_shape[:2]
    # Pixel (col, row) → world (col, row), matching modeltojson.mask_to_geojson.
    return from_bounds(0, h, w, 0, w, h)


def rasterize_polygon(polygon, image_shape) -> np.ndarray:
    """Rasterize a shapely polygon to a uint8 binary mask matching image_shape."""
    h, w = image_shape[:2]
    return rasterio.features.rasterize(
        [(mapping(polygon), 1)],
        out_shape=(h, w),
        transform=_identity_transform(image_shape),
        fill=0,
        dtype=np.uint8,
    )


def vectorize_mask(mask, simplify_tolerance=1.0) -> Optional[Polygon]:
    """Convert a binary mask to a single shapely Polygon (largest component)."""
    binary = (mask > 0).astype(np.uint8)
    if binary.sum() == 0:
        return None
    h, w = binary.shape
    transform = from_bounds(0, h, w, 0, w, h)
    polys: list[Polygon] = []
    for geom, value in rasterio.features.shapes(binary, mask=binary > 0, transform=transform):
        if value != 1:
            continue
        try:
            p = shape(geom)
        except Exception:
            continue
        if p.is_empty:
            continue
        if isinstance(p, MultiPolygon):
            polys.extend(p.geoms)
        else:
            polys.append(p)
    if not polys:
        return None
    poly = max(polys, key=lambda x: x.area)
    if simplify_tolerance and simplify_tolerance > 0:
        poly = poly.simplify(simplify_tolerance, preserve_topology=True)
    return poly


# ---------------------------------------------------------------------------
# Watershed-based split + label selection
# ---------------------------------------------------------------------------
def split_merged_wing(
    polygon,
    image_shape,
    smoothing_sigma=2.0,
    min_seed_distance=None,
    threshold_rel=0.2,
    debug=False,
):
    """Watershed-split a polygon if it contains multiple wing peaks.

    Returns (labels, mask) where labels is (H, W) int32 with 0 = background
    and 1..N for separated wings; mask is the (H, W) uint8 binary input mask.
    """
    mask = rasterize_polygon(polygon, image_shape)
    if mask.sum() == 0:
        return np.zeros(image_shape[:2], dtype=np.int32), mask

    distance = ndi.distance_transform_edt(mask)
    peak = float(distance.max())
    if peak <= 0:
        return mask.astype(np.int32), mask

    seed_image = (
        ndi.gaussian_filter(distance, sigma=smoothing_sigma) if smoothing_sigma and smoothing_sigma > 0 else distance
    )

    # Each wing's distance-transform peak ≈ inscribed-circle radius. For two
    # touching wings, their peaks are separated by ~r1+r2 ≥ peak. Using
    # min_distance ≈ peak admits both peaks while rejecting sub-peaks within
    # the same wing.
    if min_seed_distance is None:
        min_seed_distance = max(20, int(round(peak)))

    coords = peak_local_max(
        seed_image,
        min_distance=int(min_seed_distance),
        threshold_rel=threshold_rel,
        labels=mask,
        exclude_border=False,
    )

    if coords.size == 0:
        return mask.astype(np.int32), mask

    seeds = np.zeros(distance.shape, dtype=bool)
    seeds[tuple(coords.T)] = True
    markers, num_markers = ndi.label(seeds)

    if debug:
        print(f"  split: peak_dist={peak:.1f}, " f"min_seed_distance={int(min_seed_distance)}, " f"seeds={num_markers}")

    if num_markers <= 1:
        return mask.astype(np.int32), mask

    labels = watershed(-distance, markers, mask=mask.astype(bool))
    return labels.astype(np.int32), mask


def select_main_label(labels, center) -> int:
    """Watershed label at the center pixel, else nearest centroid."""
    cx, cy = int(round(center[0])), int(round(center[1]))
    h, w = labels.shape
    if 0 <= cy < h and 0 <= cx < w:
        center_label = int(labels[cy, cx])
        if center_label != 0:
            return center_label

    unique = [int(v) for v in np.unique(labels) if v != 0]
    if not unique:
        return 0

    best_label, best_dist = unique[0], float("inf")
    for lbl in unique:
        ys, xs = np.where(labels == lbl)
        if xs.size == 0:
            continue
        d = (xs.mean() - cx) ** 2 + (ys.mean() - cy) ** 2
        if d < best_dist:
            best_dist = d
            best_label = lbl
    return best_label


def dilate_polygon_to_image(polygon, image_shape, expand_fraction):
    """Uniform Minkowski dilation, clipped to image bounds.

    Buffer distance = expand_fraction * sqrt(polygon.area) so every edge
    moves outward by the same number of pixels regardless of shape.
    """
    if not (expand_fraction and expand_fraction > 0):
        return polygon
    h, w = image_shape[:2]
    buffer_dist = expand_fraction * float(np.sqrt(polygon.area))
    dilated = polygon.buffer(buffer_dist)
    if dilated.is_empty:
        dilated = polygon
    return dilated.intersection(Polygon([(0, 0), (w, 0), (w, h), (0, h)]))


# ---------------------------------------------------------------------------
# Image masking + writing
# ---------------------------------------------------------------------------
def apply_mask_to_image(image, mask, bg_value=0) -> np.ndarray:
    """Set all pixels outside `mask` to `bg_value`."""
    out = image.copy()
    out[mask == 0] = bg_value
    return out


def write_masked_image(image, output_path) -> str:
    """Write an image array to disk; falls back from PSD → PNG.

    Returns the path actually written (may differ from `output_path` for PSD).
    """
    output_path = str(output_path)
    ext = Path(output_path).suffix.lower()
    parent = os.path.dirname(os.path.abspath(output_path))
    if parent:
        os.makedirs(parent, exist_ok=True)

    if ext in (".tif", ".tiff"):
        try:
            import tifffile

            tifffile.imwrite(output_path, image)
            return output_path
        except Exception:
            pass

    if ext in (".psd", ".psb"):
        png_path = str(Path(output_path).with_suffix(".png"))
        Image.fromarray(image).save(png_path)
        return png_path

    Image.fromarray(image).save(output_path)
    return output_path


def build_geojson(polygon, source_image_path, class_name="wing") -> dict:
    """Build a single-feature GeoJSON FeatureCollection for the main wing."""
    feature = geojson.Feature(
        geometry=mapping(polygon),
        properties={
            "class": class_name,
            "source_image": os.path.basename(str(source_image_path)),
            "isolated_by": "wingIsolator",
        },
    )
    return geojson.FeatureCollection([feature])


# ---------------------------------------------------------------------------
# In-memory pipeline (for TRACE chaining)
# ---------------------------------------------------------------------------
def isolate_in_memory(
    image: np.ndarray,
    polygons: list,
    *,
    simplify_tolerance: float = 1.0,
    smoothing_sigma: float = 2.0,
    min_seed_distance: Optional[int] = None,
    threshold_rel: float = 0.2,
    expand_fraction: float = 0.05,
    debug: bool = False,
):
    """Run the full isolate-main-wing pipeline on already-loaded data.

    Args:
        image: RGB uint8 array of shape (H, W, 3).
        polygons: list of shapely Polygons (typically from
            :func:`load_wing_polygons`). MultiPolygons should already be
            flattened.

    Returns:
        Dict with keys:
          - ``polygon``: dilated main-wing shapely Polygon (or None if the
            pipeline could not produce one).
          - ``mask``: (H, W) uint8 binary mask matching the dilated polygon.
          - ``num_input_polygons``: count of polygons supplied.
          - ``num_subwings``: number of watershed labels produced (1 if no
            split was needed).
          - ``status``: "ok", "no_wing_polygons", "empty_after_split", or
            "vectorize_failed".
    """
    h, w = image.shape[:2]
    center = (w / 2.0, h / 2.0)

    if not polygons:
        return {
            "polygon": None,
            "mask": np.zeros((h, w), dtype=np.uint8),
            "num_input_polygons": 0,
            "num_subwings": 0,
            "status": "no_wing_polygons",
        }

    main_poly = select_main_polygon(polygons, center)
    labels, _ = split_merged_wing(
        main_poly,
        image.shape,
        smoothing_sigma=smoothing_sigma,
        min_seed_distance=min_seed_distance,
        threshold_rel=threshold_rel,
        debug=debug,
    )
    main_label = select_main_label(labels, center)
    if main_label == 0:
        return {
            "polygon": None,
            "mask": np.zeros((h, w), dtype=np.uint8),
            "num_input_polygons": len(polygons),
            "num_subwings": int(labels.max()),
            "status": "empty_after_split",
        }

    keep_mask = (labels == main_label).astype(np.uint8)
    main_polygon = vectorize_mask(keep_mask, simplify_tolerance=simplify_tolerance)
    if main_polygon is None:
        return {
            "polygon": None,
            "mask": keep_mask,
            "num_input_polygons": len(polygons),
            "num_subwings": int(labels.max()),
            "status": "vectorize_failed",
        }

    main_polygon = dilate_polygon_to_image(main_polygon, image.shape, expand_fraction)
    keep_mask = rasterize_polygon(main_polygon, image.shape)

    return {
        "polygon": main_polygon,
        "mask": keep_mask,
        "num_input_polygons": len(polygons),
        "num_subwings": int(labels.max()),
        "status": "ok",
    }


# ---------------------------------------------------------------------------
# File-based orchestrator
# ---------------------------------------------------------------------------
def isolate_main_wing(
    image_path,
    geojson_path,
    output_dir,
    *,
    class_names=WING_CLASS_NAMES,
    output_suffix=DEFAULT_OUTPUT_SUFFIX,
    bg_value=0,
    simplify_tolerance=1.0,
    smoothing_sigma=2.0,
    min_seed_distance=None,
    threshold_rel=0.2,
    expand_fraction=0.05,
    debug=False,
) -> IsolationResult:
    """Isolate the main wing for one image / geojson pair, writing outputs.

    Returns an :class:`IsolationResult`. On error (uncaught exception) the
    result has ``status="error"`` and the exception message in ``error``.
    """
    try:
        os.makedirs(output_dir, exist_ok=True)
        image = load_image(image_path)
        polygons = load_wing_polygons(geojson_path, class_names=class_names)

        outcome = isolate_in_memory(
            image,
            polygons,
            simplify_tolerance=simplify_tolerance,
            smoothing_sigma=smoothing_sigma,
            min_seed_distance=min_seed_distance,
            threshold_rel=threshold_rel,
            expand_fraction=expand_fraction,
            debug=debug,
        )

        if outcome["status"] != "ok":
            return IsolationResult(
                image_path=str(image_path),
                geojson_path=str(geojson_path),
                num_input_polygons=outcome["num_input_polygons"],
                num_subwings=outcome["num_subwings"],
                status=outcome["status"],
            )

        masked_image = apply_mask_to_image(image, outcome["mask"], bg_value=bg_value)

        stem = Path(image_path).stem
        img_ext = Path(image_path).suffix.lower() or ".png"
        masked_image_path = os.path.join(output_dir, f"{stem}{output_suffix}{img_ext}")
        main_geojson_path = os.path.join(output_dir, f"{stem}{output_suffix}.geojson")

        actual_image_path = write_masked_image(masked_image, masked_image_path)
        fc = build_geojson(outcome["polygon"], image_path)
        with open(main_geojson_path, "w") as f:
            geojson.dump(fc, f, indent=2)

        return IsolationResult(
            image_path=str(image_path),
            geojson_path=str(geojson_path),
            masked_image_path=actual_image_path,
            main_geojson_path=main_geojson_path,
            num_input_polygons=outcome["num_input_polygons"],
            num_subwings=outcome["num_subwings"],
            status="ok",
        )
    except Exception as exc:
        return IsolationResult(
            image_path=str(image_path),
            geojson_path=str(geojson_path),
            status="error",
            error=f"{type(exc).__name__}: {exc}",
        )


# ---------------------------------------------------------------------------
# Folder discovery + batch
# ---------------------------------------------------------------------------
def discover_images(folder) -> list[Path]:
    """Find supported image files in a folder, skipping hidden/resource files."""
    folder = Path(folder)
    paths: list[Path] = []
    for f in sorted(folder.iterdir()):
        if f.name.startswith(".") or f.name.startswith("._"):
            continue
        if f.suffix.lower() in SUPPORTED_IMAGE_EXTS:
            paths.append(f)
    return paths


def find_geojson_for_image(image_path, geojson_dir) -> Optional[str]:
    """Find a matching geojson for an image: <stem>_detections.geojson, then
    <stem>.geojson, then any <stem>*.geojson."""
    stem = Path(image_path).stem
    candidates = [
        Path(geojson_dir) / f"{stem}_detections.geojson",
        Path(geojson_dir) / f"{stem}.geojson",
    ]
    for c in candidates:
        if c.exists():
            return str(c)
    matches = sorted(Path(geojson_dir).glob(f"{stem}*.geojson"))
    return str(matches[0]) if matches else None


def isolate_folder(
    image_dir,
    geojson_dir,
    output_dir,
    *,
    progress_callback=None,
    write_summary: bool = True,
    **isolate_kwargs,
) -> list[IsolationResult]:
    """Run :func:`isolate_main_wing` across paired images + geojsons.

    `progress_callback`, if given, is called as
    ``progress_callback(index, total, image_path, result)`` after each image.

    If ``write_summary`` is True, a ``wing_isolator_summary.json`` file is
    written to ``output_dir`` containing every result as a dict.
    """
    image_dir = Path(image_dir)
    geojson_dir = Path(geojson_dir)
    images = discover_images(image_dir)

    results: list[IsolationResult] = []
    for i, img in enumerate(images, 1):
        gj = find_geojson_for_image(img, geojson_dir)
        if gj is None:
            res = IsolationResult(image_path=str(img), status="no_geojson")
        else:
            res = isolate_main_wing(str(img), gj, str(output_dir), **isolate_kwargs)
        results.append(res)
        if progress_callback:
            progress_callback(i, len(images), str(img), res)

    if write_summary and results:
        summary_path = Path(output_dir) / "wing_isolator_summary.json"
        os.makedirs(output_dir, exist_ok=True)
        with open(summary_path, "w") as f:
            json.dump([r.to_dict() for r in results], f, indent=2)
    return results
