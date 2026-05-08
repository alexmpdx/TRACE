# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What This Is

WingRotator rotates a Drosophila wing image (and its associated GeoJSONs) into a canonical right-side-up, distal-right orientation using a reliability-weighted Procrustes fit against a hardcoded landmark template. It's invoked as Stage 1.5 of `preprocessing` (between LandmarkLocator and HingeChopper) so every downstream stage operates on a normalized wing.

**Rotation only — no reflection.** Per design, the rotator never mirrors; left-vs-right wing chirality is preserved. Opposite-chirality inputs are detected and given an extra 180° rotation on top of the proper-rotation fit so anterior stays at the top of the image (the cost is that PD ends up distal-left instead of distal-right for those wings — a consistent AP split matters more for downstream stages than which way the wing points along the long axis).

## Running

```bash
# Standalone CLI — single pair
python wingRotator/run_cli.py \
    --image <wing.tif> --landmarks <wing_landmarks.geojson> -o <out_dir>

# Batch mode (image dir + landmarks dir; pairs by stem with `_landmarks` suffix)
python wingRotator/run_cli.py --image <img_dir> --landmarks <lm_dir> -o <out_dir>

# With extra GeoJSONs (segmentation, ground-truth overlays, wing-isolation polygons)
# applied with the same affine — useful for re-orienting existing test data
python wingRotator/run_cli.py --image <img> --landmarks <lm> -o <out> \
    --extra-geojson <other.geojson> --extra-geojson <yet_another.geojson>

# Soft-weight unreliable landmarks instead of dropping them
python wingRotator/run_cli.py ... --soft-reliability
```

## Dependencies

`opencv-python`, `numpy` — plus `tifffile` only when writing TIFFs (optional, falls back to `cv2.imwrite`). No shapely, no torch. The image-reader path falls back to `preprocessing.psd_loader.imread_any` for exotic formats; that import is lazy and only touched if `cv2.imread` returns None.

## Architecture

Single file: `wing_rotator.py`. Pipeline flow inside `rotate_from_landmarks()`:

1. `_load_landmarks_for_fit()` — parse landmarks GeoJSON, look up each known landmark in `CANONICAL_LANDMARKS`, compute per-landmark weight, drop zero-weight entries.
2. `_weighted_kabsch_2d()` — closed-form 2D Procrustes (no reflection) returning `(theta, rms_residual)`. Residual is normalized by the canonical-side scale so it's comparable across calls.
3. `_detect_mirror()` — refits against a y-flipped canonical; flags wings whose mirror residual is <70% of the proper-rotation residual.
4. `_build_affine()` — forward affine `M_forward` (src→dst, applied to coordinates) plus inverted `M_warp` (passed to `cv2.warpAffine`). Canvas is expanded to fit rotated content.
5. Apply: `cv2.warpAffine` for the image, `transform_geojson()` recursively for landmark/polygon coordinates.

## Canonical template

`CANONICAL_LANDMARKS` is hardcoded in `wing_rotator.py` — derived from one well-oriented wing in `testdata/testwings/-CTRL_PknRNAi_108870_0007*` with alula notch translated to the origin and DTip placed on the +X axis. Only relative geometry matters because the Procrustes fit absorbs scale and translation. To refine, average positions over a handful of well-oriented training wings.

## Weight formula

```
weight = confidence × sharpness / (1 + second_peak_ratio)
```

`reliable=False` landmarks are hard-gated (weight 0) when `soft_reliability=False` (default). When `soft_reliability=True` they contribute at `_UNRELIABLE_WEIGHT_FACTOR = 0.25` of nominal weight.

In TRACE/preprocessing this flag is wired to the existing `--include-unreliable-landmarks` setting: turning that on enables soft-weighting in the rotator (in addition to its original effect of writing low-confidence landmarks to the output GeoJSON).

## Robustness

- **Min landmarks**: 2 to fit, otherwise return `None` (caller passes the image through unchanged).
- **Mirror detection** requires ≥3 landmarks; with only 2, chirality is ambiguous and we don't probe.
- **Residual sanity**: warns when RMS exceeds 25% of the canonical PD span; doesn't abort.
- **Skipped-stage protocol**: returning `None` is the contract. `preprocessing/pipeline.py` handles it by leaving `image_path` unchanged and not appending `"rotation"` to `stages_completed`.

## Integration with preprocessing

`preprocessing.pipeline.run_rotation()` is the bridge. It:

1. Calls `rotate_from_landmarks()` and gets back the rotated image path + result.
2. Applies the same affine to the in-memory `landmarks: dict[name, (x, y)]` so the hinge stage sees rotated coordinates without reloading.
3. Returns `(rotated_image_path, rotated_landmarks_geojson_path, rotated_landmarks_dict, RotationResult)`.

`process_single_image()` then rebinds `image_path`, recomputes `stem`/`ext`/`raster_ext`, and updates `PipelineResult.rotated_*` fields. Every downstream stage (hinge chop, segmentation) sees the rotated image and inherits a `_rotated` infix in its filenames.

When Stage 0 (wing isolation) ran, the wing-isolation GeoJSON is passed in `extra_geojsons` so it stays aligned with the rotated image; `result.wing_geojson_path` is updated to the rotated version.

## Output naming

- Image: `<stem>_rotated.<ext>` (e.g. `wing_rotated.tif`)
- Landmarks GeoJSON: `<stem>_rotated_landmarks.geojson`
- Extra GeoJSONs: `<extra_stem>_rotated.geojson`
- Downstream artifacts inherit the rotated stem: `<stem>_rotated_chopped.<ext>`, `<stem>_rotated.geojson` (segmentation), etc.

## Toggle

Default-on. Disable via:
- `--no-rotation` on `preprocessing/run_cli.py` and `TRACE/run_cli.py`
- The "Rotate to canonical orientation (Stage 1.5)" checkbox in TRACE Settings → Landmarks tab (persisted to QSettings)
- `do_rotation=False` to `process_folder()` / `process_single_image()` / `trace_folder()`

## Code style

Matches the rest of the project: Black + isort + flake8 at 120-char line length. Pre-commit hooks (isort → black → flake8) apply.
