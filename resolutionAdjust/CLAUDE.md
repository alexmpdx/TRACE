# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What This Is

resolutionAdjust rescales wing images toward a per-model "training µm/px" target before any DL inference, and provides the inverse transforms that map geometry produced downstream back to the original pixel grid. It runs as **Stage -1** of `preprocessing` — ahead of Stage 0 (wing isolation), Stage 1 (landmarks), Stage 2 (hinge), Stage 3 (segmentation), and Stage 3.5 (rotation) — so every DL model sees pixels close to the resolution it was trained on.

## Running standalone

```bash
# Single image or folder; --input-um-per-px is the µm/px the input was captured at.
python resolutionAdjust/run_cli.py <image-or-folder> -o <out_dir> \
    --input-um-per-px 0.25 --target-um-per-px 0.483 \
    --tolerance-low 0.85 --tolerance-high 1.15
```

The CLI is for inspecting what Stage -1 would do — it does not run the inverse step.

## Dependencies

`opencv-python`, `numpy`, `shapely`, `tifffile` (lazy — only the auto-detect path needs it, and only the `.ome.tif` writer in `_write_image`). The image reader falls back to `preprocessing.psd_loader.imread_any` for exotic formats; that import is lazy.

## Architecture

Three files, no class hierarchy:

- `resolution_adjust.py` — `adjust_resolution()` is the entry point. Returns a `ResolutionAdjustResult` with `scale_factor = new_pixels / original_pixels`. The inverse helpers all live here.
- `auto_detect.py` — `autodetect_um_per_px_from_folder(folder)` reads `XResolution + ResolutionUnit` and OME-XML `PhysicalSizeX` from TIFFs and averages over images that yield a value. Returns `(avg_or_None, n_with_metadata, n_total)` so the GUI can surface "5/12 had metadata".
- `run_cli.py` — standalone CLI; not used by the pipeline.

## Pass-through band semantics

`adjust_resolution` skips rescaling when `input_um_per_px / target_um_per_px ∈ [tolerance_low, tolerance_high]`. The ratio is **input ÷ target**, so:

- ratio > 1  → input is coarser → upscale, `scale_factor = ratio`
- ratio < 1  → input is finer   → downscale, `scale_factor = ratio`
- ratio == 1 → exact match, would be skipped by any sane band

When the input is inside the band, `scale_factor = 1.0`, no temp file is written, and `image_path` in the result points back at the original.

## Inverse round-trip semantics

The rescale and the inverse are intentionally symmetric so the rest of the pipeline can stay ignorant of Stage -1:

1. `adjust_resolution` rewrites the image at `scale_factor × original_size` and stores the factor.
2. Every DL stage and identifyFeatures run in **rescaled-pixel space**.
3. `inverse_rescale_wing_result(wing_result, scale_factor, um_per_px)` maps all shapely geometry (vein centerlines, tissue polygons, intervein polygons, landmarks, wing outline) back to original pixels via `shapely.affinity.scale(g, 1/sf, 1/sf, origin=(0, 0))`, then recomputes `length_px / area_px2` from the new geometry and `length_um / area_um2` from `length_px × um_per_px`.
4. `inverse_resize_image(image, scale_factor)` resizes the overlay base back to original-resolution (rotation stays baked in — only resolution is undone).
5. `inverse_transform_geojson(data, scale_factor)` is for raw GeoJSON dicts (landmarks/segmentation overlays) that still live in rescaled space when their overlays are rendered.

After step 3, downstream consumers (CSV export, GeoJSON export, overlay rendering) can be passed `um_per_px = input_um_per_px` and produce correct physical-unit measurements — `length_px × input_um_per_px` is the same number as the un-rescaled `length_px_rescaled × target_um_per_px` by construction.

## TIFF output coercion

`_write_image` follows the same rule as `wing_rotator` / `HingeChopper`: `.tif` / `.tiff` / `.ome.tif` go through `tifffile` with the right photometric tag (`rgb` for 3/4-channel, `minisblack` for 2D); everything else hits `cv2.imwrite`. PSD inputs are written as `<stem>_resampled.ome.tif`.

## Integration points (where to look in other modules)

- **Trigger**: `preprocessing/pipeline.py` `process_single_image()` — Stage -1 runs between format coercion and Stage 0 (wing isolation). New params: `input_um_per_px`, `target_um_per_px`, `rescale_tolerance_low/high`. `PipelineResult.rescale_factor` and `PipelineResult.original_shape` carry the info downstream. The landmark batch prefetch in `process_folder` is disabled when Stage -1 is active (path-key mismatch between original and rescaled file).
- **Inverse**: `TRACE/pipeline.py` `_run()._analyze_one()` — calls `inverse_rescale_wing_result` after `identify_wing`, then `inverse_resize_image` on the overlay base. `_render_landmarks_overlay` and `_render_segmentation_overlay` take an `inverse_scale` arg that multiplies coordinates from their geojson before drawing.
- **GUI**: `TRACE/settings_dialog.py` Models tab — per-model `Training µm/px` field + Auto-detect button (`_make_target_row`, `_autodetect_target_um_per_px`); "Resolution adjustment" group with the active-model radio (default = Wing features) and tolerance low/high spinboxes. Each tolerance spinbox has a live µm/px label updated by `_update_tolerance_um_labels`.
- **Persistence**: QSettings keys under `models/{landmark,segmentation,wing_isolation}_target_um_per_px`, `models/active_rescale_target`, `resolution/tolerance_low|high`. Wing features defaults to **0.483 µm/px** on first launch (the resolution the bundled segmentation model was trained at); the default is only applied when the QSettings key has never been written, so a user's deliberate clear-to-blank is preserved.
- **CLI**: `TRACE/cli.py` flags `--target-um-per-px`, `--rescale-tolerance-low`, `--rescale-tolerance-high`. There is no per-model selector on the CLI — pass the active model's target directly.

## Skipped-stage protocol

`adjust_resolution` is called from preprocessing only when `input_um_per_px > 0 AND target_um_per_px > 0`. When the active model has no target set, the target arg arrives as None and preprocessing skips Stage -1 entirely — `rescale_factor` stays at 1.0 and the inverse helpers all no-op. Failures inside `adjust_resolution` are soft: preprocessing logs a warning and falls through with the original image rather than aborting the run.

## Code style

Matches the rest of the project: Black + isort + flake8 at 120-char line length. Pre-commit hooks (isort → black → flake8) apply.
