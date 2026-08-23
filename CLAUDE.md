# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Drosophila wing morphology analysis suite. Takes brightfield wing images through deep-learning preprocessing (landmark detection, hinge removal, semantic segmentation) and landmark-anchored vein analysis (vein identification, measurement, region naming). Outputs overlay images and CSV measurements.

## Module Dependency Chain

TRACE is a 2-stage wrapper. **TRACE Stage 1 = preprocessing** (itself 6 numbered sub-stages); **TRACE Stage 2 = identifyFeatures**. Be careful with stage numbering — `Stage 3` inside `preprocessing/pipeline.py` means landmarks, but in `TRACE/pipeline.py` there is no "Stage 3" at all.

```
TRACE/pipeline.py
├── Stage 1: preprocessing.process_folder()        (6 sub-stages below)
│       │
│       ├── Sub-stage 1  resolution_adjust   ← resolutionAdjust    (rescale to target µm/px, if outside tolerance)
│       ├── Sub-stage 2  wing_isolation      ← modelTOjson + wingIsolator  (optional — gated on wing-isolation model dir)
│       ├── Sub-stage 3  landmarks           ← LandmarkLocator     (anatomical landmark detection; GPU-batched)
│       ├── Sub-stage 4  hinge               ← HingeChopper        (black out proximal hinge region)
│       ├── Sub-stage 5  segmentation        ← modelTOjson         (vein / intervein semantic seg → GeoJSON)
│       └── Sub-stage 6  rotation            ← wingRotator         (align to canonical distal-right; optional)
│
└── Stage 2: identify_features.identify_wing()     (landmark-anchored vein ID + measurement + overlay rendering)
```

- **TRACE** is the top-level pipeline: runs preprocessing then identifyFeatures, outputs consolidated CSV + overlays.
- **preprocessing** orchestrates the 6 sub-stages above. `_required_stages()` in `TRACE/pipeline.py` returns a `(landmarks, hinge, seg)` triple that lets the orchestrator skip sub-stages the user-selected outputs don't depend on (e.g. picking only `chopped_image` skips Sub-stages 5–6).
- **identifyFeatures** takes image + segmentation GeoJSON + landmarks, identifies veins/regions, measures, generates overlays. Stage 2 entry is `identify_wing(detection_geojson, landmarks_geojson, image_path, config, specimen_id)`.

### Sibling helpers used by TRACE (not in either pipeline stage)

These live alongside the pipeline modules and are imported by the TRACE GUI, **not** by the pipeline orchestration code:

- **scaleEstimator** — interactive landmark-distance → µm/px calculator, surfaced in `inline_panels.py` and the settings dialog.
- **measurementMaker** — user-defined landmark-pair distance CSV columns. Has both a GUI editor and a fast post-CSV augmentation path called from `TRACE/pipeline.py`.
- **liveSettings** — exports the `live_tune` package; `TRACE/settings_dialog.py` attaches its live-preview widgets via runtime `sys.path` injection (see `settings_dialog.py:343`).
- **manualInspector/** — *spec doc only* (`landmark_inspector_spec.md`). The actual landmark/segmentation override inspector lives in `TRACE/landmark_inspector_dialog.py`.

`resolutionAdjust` also has a few **post-hoc** uses inside TRACE (scaling overlay GeoJSON coords back to original resolution in `_render_segmentation_overlay()` and inverse-rescaling `WingResult` before export) — separate from its Sub-stage 1 role above.

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

**Import plumbing**: Sibling modules export bare package names that don't match their containing directory — `LandmarkLocator/` exports `landmark_locator`, `HingeChopper/` exports `hinge_chopper`, `modelTOjson/` exports `modeltojson`, `wingRotator/` exports `wing_rotator`, `wingIsolator/` exports `wing_isolator`, `resolutionAdjust/` exports `resolution_adjust`, `liveSettings/` exports `live_tune`, `measurementMaker/` exports `measurement_maker`, `scaleEstimator/` exports `scale_estimator`. Entry scripts (`TRACE/run_cli.py`, `TRACE/run_gui.py`, `preprocessing/run_cli.py`) add each parent dir to `sys.path`. The canonical TRACE set is: `<project_root>`, `<project_root>/HingeChopper`, `<project_root>/modelTOjson`, `<project_root>/identifyFeatures`, `<project_root>/wingRotator`, `<project_root>/measurementMaker`, `<project_root>/scaleEstimator`, `<project_root>/LandmarkLocator`. `liveSettings/` is added on-demand from inside `TRACE/settings_dialog.py` when the live-preview widget loads.

**GeoJSON data flow**: Preprocessing outputs `properties.class` (e.g. "vein", "intervein", "wing"). identifyFeatures' parser falls back from `properties.classification.name` to `properties.class`, so both formats work.

**TRACE measurement groups**: `MEASUREMENT_GROUPS` in `pipeline.py` defines which CSV column groups are available. `filter_csv_columns()` prunes the consolidated CSV to only selected groups. The GUI exposes these as checkboxes.

**TRACE stage skipping**: `_required_stages()` in `TRACE/pipeline.py` computes the minimal set of preprocessing stages needed for the user-selected outputs (landmarks / hinge / segmentation). Picking only `chopped_image`, for example, skips segmentation and Stage 2 entirely.

**Manual inspector + overrides**: `TRACE/landmark_inspector_dialog.py` (launched from the Main-tab image-list right-click menu or the post-run "Review failed images" button) is a tabbed napari editor. The **Landmarks** tab lets users drag/add/delete landmarks; the **Veins / Interveins** tab lets them reclassify / delete / draw vein & intervein polygons. Each tab writes a sidecar into a dedicated `manual_overrides/` subfolder next to the *source* image — `<image_dir>/manual_overrides/<stem>_landmarks_override.geojson` and `…/<stem>_segmentation_override.geojson` respectively (kept in their own folder so they stay organized and aren't accidentally deleted when tidying the image folder) — and `preprocessing/pipeline.py` short-circuits the matching stage when the sidecar exists: Stage 3 via `load_landmarks_override()`, Stage 5 via `load_segmentation_override()`. The single source of truth for these paths is `preprocessing.pipeline.{landmarks,segmentation}_override_path()` (write side) / `find_{landmarks,segmentation}_override()` (read side); the inspector imports them so the two sides never drift. The `find_*` helpers also auto-migrate: an override still in the older loose location next to the image is moved into `manual_overrides/` on first access (best-effort via `_relocate_legacy_override`, falling back to the loose path if the move fails), so older overrides both keep working and get tidied away. Both lookups key on `original_input_path` (captured before any stage rebinds `image_path`). Landmark overrides are marked `reliable: true` / `confidence: 1.0` to pass downstream gates. **On-demand generation**: opening either tab on an un-run image generates predictions with the confidence gate disabled (`_generate_landmarks_for_image(disable_gates=True)`), because an image you open to hand-correct is exactly the one likely to fail the gate. The **Veins tab** can't segment the raw image — the model needs the fully-preprocessed image — so it calls `TraceWindow.run_single_image_preprocessing_for_segmentation()` to run the configured chain (wing isolation + hinge chop, rescale if set) with **rotation OFF** and gates off. That preprocessing flushes any unsaved landmark edits first (`persist_landmark_edits_for_pipeline`) so Stage 3 picks them up and the hinge chop uses the corrected landmarks. Because a vein traces a thin network whose holes are the enclosed interveins (and napari Shapes can't render polygon holes), the segmentation is shown as a per-class **label mask** over the **preprocessed** image (rescaled + isolated + hinge-chopped, pre-rotation) — the exact image the model saw — and edited with paint/fill/erase (`napari` Labels); on save the mask is re-vectorized via modelTOjson's `mask_to_geojson` — the same converter the pipeline uses. **Displaying over the preprocessed image (not the original) matters when Stage 1 rescales heavily**: the model produces its mask at the rescaled resolution, so painting over the original diverged from real inference. The mask is edited in rescaled pixel space; on save the polygons are **divided by `rescale_factor`** back to **original-input pixel space** (the on-disk sidecar contract — resolution-independent), and the Stage-5 short-circuit multiplies the override back by `result.rescale_factor` via `_scale_geojson_coords`. Because the inspector always needs the preprocessed image, **reopening a saved override re-runs preprocessing (segmentation forward pass skipped) to regenerate that image**, then scales the sidecar back *into* rescaled space to rasterize. The editors are read-write siblings of `measurementMaker`'s read-only `LandmarkPickerWidget`; mirror that widget's napari API (`border_color`/`border_width`, `features=`, `size=90`) rather than older `edge_color`/`properties=` calls.

## Key Data Structures

- `identify_features.WingResult` — veins, intervein regions, measurements
- `preprocessing.PipelineResult` — image_path, landmarks, geojson paths, error, stages_completed
- `TRACE.TraceResult` — image_path, overlay paths, error, error_stage

## Test Data

- `preprocessing/testinput_images/` — raw wing TIFFs
- `preprocessing/testinput_DLmodels/` — landmark (.pt) and segmentation model checkpoints
- `testdata/` — additional sample wing images
- `identifyFeatures/geojsons/`, `identifyFeatures/GT_naming/` — detection GeoJSONs and ground-truth naming overlays
