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

**Manual landmark overrides**: `TRACE/landmark_inspector_dialog.py` (launched from the Main-tab image-list right-click menu or the post-run "Review failed images" button) lets users drag/add/delete landmarks in an embedded napari viewer and save a sidecar `<image_dir>/<stem>_landmarks_override.geojson` next to the *source* image. Stage 3 of `preprocessing/pipeline.py` checks for this sidecar (keyed on the original input path, captured as `original_input_path` before any stage rebinds `image_path`) and, if present, short-circuits LandmarkLocator entirely via `load_landmarks_override()` — overrides are marked `reliable: true` / `confidence: 1.0` so they sail through downstream gates. The inspector is a read-write sibling of `measurementMaker`'s read-only `LandmarkPickerWidget`; mirror that widget's napari API (`border_color`/`border_width`, `features=`, `size=90`) rather than older `edge_color`/`properties=` calls.

## Key Data Structures

- `identify_features.WingResult` — veins, intervein regions, measurements
- `preprocessing.PipelineResult` — image_path, landmarks, geojson paths, error, stages_completed
- `TRACE.TraceResult` — image_path, overlay paths, error, error_stage

## Test Data

- `preprocessing/testinput_images/` — raw wing TIFFs
- `preprocessing/testinput_DLmodels/` — landmark (.pt) and segmentation model checkpoints
- `testdata/` — additional sample wing images
- `identifyFeatures/geojsons/`, `identifyFeatures/GT_naming/` — detection GeoJSONs and ground-truth naming overlays
