# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Drosophila wing morphology analysis suite. Takes brightfield wing images through deep-learning preprocessing (landmark detection, hinge removal, semantic segmentation) and landmark-anchored vein analysis (vein identification, measurement, region naming). Outputs overlay images and CSV measurements.

## Module Dependency Chain

```
LandmarkLocator (landmark detection)
HingeChopper (hinge masking)        ──→ preprocessing (stages 1-3) ──→ TRACE (end-to-end)
modelTOjson (segmentation to GeoJSON)                                      ↑
                                                                   identifyFeatures
                                                                   (vein ID + measurement)
```

- **TRACE** is the top-level pipeline: runs preprocessing then identifyFeatures, outputs consolidated CSV + overlays
- **preprocessing** orchestrates LandmarkLocator → HingeChopper → modelTOjson
- **identifyFeatures** takes image + GeoJSON + landmarks, identifies veins/regions, measures, generates overlays

## Running

Entry points use `run_cli.py` / `run_gui.py` per module. Each sets up `sys.path` for sibling imports.

```bash
# TRACE (full pipeline, GUI)
python TRACE/run_gui.py

# TRACE (CLI)
python TRACE/run_cli.py -i <images> -o <output> --landmark-model <path.pt> --segmentation-model <model_dir>

# Preprocessing only
python preprocessing/run_cli.py -i <images> -o <output> --landmark-model <path.pt> --segmentation-model <model_dir>

# identifyFeatures only (needs pre-existing detection + landmarks GeoJSONs)
identify-features --batch <det_dir> <lm_dir> [image_dir] -o <output>

# LandmarkLocator training
cd LandmarkLocator && pip install -e . && landmark-train --config <yaml>

# LandmarkLocator batch predict (folder → per-image *_landmarks.geojson)
python LandmarkLocator/landmark_locator/scripts/predict.py <folder> --batch \
    --output-dir <out_dir> --checkpoint <path.pt>
```

## Code Style

- **Black** + **isort** + **flake8**, all at 120-char line length
- Pre-commit hooks run automatically: isort → black → flake8
- Run manually: `pre-commit run --all-files`

## Architecture Notes

**Import plumbing**: LandmarkLocator, HingeChopper, and modelTOjson export bare module names (`landmark_locator`, `hinge_chopper`, `modeltojson`). Entry scripts add their parent dirs to `sys.path`. Always include `<project_root>`, `<project_root>/HingeChopper`, `<project_root>/modelTOjson`, and `<project_root>/identifyFeatures` when importing across modules.

**GeoJSON data flow**: Preprocessing outputs `properties.class` (e.g. "vein", "intervein", "wing"). identifyFeatures' parser falls back from `properties.classification.name` to `properties.class`, so both formats work.

**TRACE measurement groups**: `MEASUREMENT_GROUPS` in `pipeline.py` defines which CSV column groups are available. `filter_csv_columns()` prunes the consolidated CSV to only selected groups. The GUI exposes these as checkboxes.

**TRACE stage skipping**: `_required_stages()` in `TRACE/pipeline.py` computes the minimal set of preprocessing stages needed for the user-selected outputs (landmarks / hinge / segmentation). Picking only `chopped_image`, for example, skips segmentation and Stage 2 entirely.

**Manual inspector + overrides**: `TRACE/landmark_inspector_dialog.py` (launched from the Main-tab image-list right-click menu or the post-run "Review failed images" button) is a tabbed napari editor. The **Landmarks** tab lets users drag/add/delete landmarks; the **Veins / Interveins** tab lets them reclassify / delete / draw vein & intervein polygons. Each tab writes a sidecar next to the *source* image — `<stem>_landmarks_override.geojson` and `<stem>_segmentation_override.geojson` respectively — and `preprocessing/pipeline.py` short-circuits the matching stage when the sidecar exists: Stage 3 via `load_landmarks_override()`, Stage 5 via `load_segmentation_override()`. Both lookups key on `original_input_path` (captured before any stage rebinds `image_path`). Landmark overrides are marked `reliable: true` / `confidence: 1.0` to pass downstream gates. **On-demand generation**: opening either tab on an un-run image generates predictions with the confidence gate disabled (`_generate_landmarks_for_image(disable_gates=True)`), because an image you open to hand-correct is exactly the one likely to fail the gate. The **Veins tab** can't segment the raw image — the model needs the fully-preprocessed image — so it calls `TraceWindow.run_single_image_preprocessing_for_segmentation()` to run the configured chain (wing isolation + hinge chop, rescale if set) with **rotation OFF** and gates off. That preprocessing flushes any unsaved landmark edits first (`persist_landmark_edits_for_pipeline`) so Stage 3 picks them up and the hinge chop uses the corrected landmarks. Because a vein traces a thin network whose holes are the enclosed interveins (and napari Shapes can't render polygon holes), the segmentation is shown as a per-class **label mask** over the **original** image and edited with paint/fill/erase (`napari` Labels); on save the mask is re-vectorized via modelTOjson's `mask_to_geojson` — the same converter the pipeline uses. The override is written in **original-input pixel space** (wing isolation / hinge chop only mask pixels — no coord change — so only rescale moves coords; generated polygons are divided by the rescale factor, and the Stage-5 short-circuit multiplies the override back by `result.rescale_factor` via `_scale_geojson_coords`). A saved override is already in original space, so reopening it skips preprocessing entirely. The editors are read-write siblings of `measurementMaker`'s read-only `LandmarkPickerWidget`; mirror that widget's napari API (`border_color`/`border_width`, `features=`, `size=90`) rather than older `edge_color`/`properties=` calls.

## Key Data Structures

- `identify_features.WingResult` — veins, intervein regions, measurements
- `preprocessing.PipelineResult` — image_path, landmarks, geojson paths, error, stages_completed
- `TRACE.TraceResult` — image_path, overlay paths, error, error_stage

## Test Data

- `preprocessing/testinput_images/` — raw wing TIFFs
- `preprocessing/testinput_DLmodels/` — landmark (.pt) and segmentation model checkpoints
- `testdata/` — additional sample wing images
- `identifyFeatures/geojsons/`, `identifyFeatures/GT_naming/` — detection GeoJSONs and ground-truth naming overlays
