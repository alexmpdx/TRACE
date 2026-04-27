# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Purpose

Pre-processing utility for the mapThemVeins pipeline. Takes a wing-vs-background segmentation GeoJSON (from `modelTOjson`) plus the source image, and outputs a single-wing GeoJSON + masked image isolating the **main** (image-centered) wing. Used when multiple wings appear in one image — sometimes as separate polygons, sometimes accidentally merged by the segmenter.

## Layout

`wingIsolator/` is a Python package (matches preprocessing/, TRACE/ conventions):

- `pipeline.py` — library API. Contains `IsolationResult` dataclass, `isolate_main_wing` (file-based), `isolate_folder` (batch), and `isolate_in_memory` (numpy + shapely in, mask + polygon out).
- `cli.py` — argparse CLI; calls into `pipeline`.
- `run_cli.py` — entry script that adds the project root + `modelTOjson` to `sys.path`, then runs `cli.main`.
- `__init__.py` — re-exports the public API.
- `wing_isolator.py` — backward-compat shim re-exporting the same names.

## Library usage

From TRACE or any sibling package (after the project root is on `sys.path`):

```python
from wingIsolator import isolate_main_wing, isolate_in_memory, IsolationResult

# File-in / file-out — returns IsolationResult dataclass
result = isolate_main_wing(image_path, geojson_path, output_dir)

# In-memory — for chaining without disk I/O. Returns dict with
# 'polygon', 'mask', 'num_input_polygons', 'num_subwings', 'status'.
out = isolate_in_memory(image_array, list_of_shapely_polygons)
```

## Running the CLI

```bash
# Single image
python run_cli.py -i wing.tif -g wing_detections.geojson -o out/

# Batch
python run_cli.py --batch <image_dir> <geojson_dir> -o <out_dir>

# QA overlays after a batch run (hardcoded to testpics/ + testpics_geojsons/ + testpics_isolated/)
python make_overlays.py
```

The CLI's `--batch` mode pairs each image with `<stem>_detections.geojson` (or `<stem>.geojson`) in the geojson dir. It writes `<stem>_main_wing.{ext,geojson}` and a `wing_isolator_summary.json` per run.

To regenerate detection GeoJSONs from images, call `modeltojson.process_folder(model_dir, image_dir, geojson_dir)` from `../modelTOjson` with a wing-vs-background segmentation model (e.g. `../workingModels/alextinynet_4x_*`).

## Pipeline

`wing_isolator.isolate_main_wing()` runs six stages:

1. **Load polygons** — filter GeoJSON features by `properties.class` (default `"wing"`); falls back to all polygon features if nothing matches. MultiPolygon features are flattened.
2. **Pick main polygon** — first polygon containing the image-center point; falls back to largest by area.
3. **Watershed split** — rasterize → `scipy.ndimage.distance_transform_edt` → smooth → `skimage.feature.peak_local_max` → `skimage.segmentation.watershed`. Seed `min_distance` defaults to ~max DT value, which admits one peak per touching wing while rejecting sub-peaks inside a single wing. Most inputs produce a single seed and skip the split.
4. **Pick main label** — watershed label at the center pixel; fallback to nearest centroid.
5. **Vectorize + dilate** — `rasterio.features.shapes` → simplify → uniform Minkowski dilation via `polygon.buffer(expand_fraction * sqrt(area))`, then clipped to image bounds. Default `expand_fraction=0.05`. The dilated polygon is re-rasterized so the masked image and GeoJSON output stay in sync.
6. **Write** — single-feature GeoJSON + masked image (`bg_value` outside the dilated mask).

Tunables exposed on the CLI: `--threshold-rel` (lower catches small sliver wings), `--min-seed-distance` (raise to prevent over-splitting one wing), `--smoothing-sigma`, `--simplify`, `--bg-value`, `--class-name` (repeatable), `--expand` (dilation fraction; 0 disables).

## Image I/O

`load_image()` imports `read_image` from `../modelTOjson/modeltojson.py` to share PSD / multi-band TIFF handling with the rest of the pipeline (sys.path is patched at module load). Falls back to PIL when `modelTOjson` is unavailable.

`write_masked_image()` writes the masked output in the input's format. PSD/PSB has no good open-source writer, so it falls back to PNG. TIFF goes through `tifffile`; everything else through PIL.

## Coordinate convention

Rasterization uses `rasterio.transform.from_bounds(0, h, w, 0, w, h)` so pixel `(col, row)` maps to world `(col, row)` — same convention as `modeltojson.mask_to_geojson`. Don't change this without auditing the round-trip in `rasterize_polygon` ↔ `vectorize_mask`.

## Test data

`testpics/` holds 8 multi-wing images. `testpics_geojsons/` are the segmenter outputs, `testpics_isolated/` is the wing_isolator output, `testpics_overlays/` has the QA visualizations. On the current test set, 2/8 images required watershed splitting (0012, 0013); the other 6 had separate wing polygons that center-selection handled directly.
