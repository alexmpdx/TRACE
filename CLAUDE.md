# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Drosophila wing morphology analysis suite. Takes brightfield wing images through deep-learning preprocessing (landmark detection, hinge removal, semantic segmentation) and computer vision analysis (vein identification, measurement, region naming). Outputs overlay images and CSV measurements.

## Module Dependency Chain

```
LandmarkLocator (landmark detection)
HingeChopper (hinge masking)        ──→ preprocessing (stages 1-4) ──→ TRACE (end-to-end)
modelTOjson (segmentation to GeoJSON)                                      ↑
                                                                   WingVeinAnalyzer
                                                                   (vein ID + measurement)
```

- **TRACE** is the top-level pipeline: runs preprocessing then WingVeinAnalyzer, outputs consolidated CSV + overlays
- **preprocessing** orchestrates LandmarkLocator → HingeChopper → modelTOjson → add_wing
- **WingVeinAnalyzer** takes image + GeoJSON pairs, identifies veins/regions, measures, generates overlays
- **EZcheezeMeasure** is a standalone landmark distance tool (legacy, partially absorbed into TRACE)

## Running

Entry points use `run_cli.py` / `run_gui.py` per module. Each sets up `sys.path` for sibling imports.

```bash
# TRACE (full pipeline, GUI)
python TRACE/run_gui.py

# TRACE (CLI)
python TRACE/run_cli.py -i <images> -o <output> --landmark-model <path.pt> --segmentation-model <model_dir>

# Preprocessing only
python preprocessing/run_cli.py -i <images> -o <output> --landmark-model <path.pt> --segmentation-model <model_dir>

# WingVeinAnalyzer only (needs pre-existing GeoJSON)
python WingVeinAnalyzer/run_batch.py <folder_with_tif_and_geojson> -o <output>

# LandmarkLocator training
cd LandmarkLocator && pip install -e . && landmark-train --config <yaml>
```

## Code Style

- **Black** + **isort** + **flake8**, all at 120-char line length
- Pre-commit hooks run automatically: isort → black → flake8
- Run manually: `pre-commit run --all-files`

## Architecture Notes

**WingVeinAnalyzer** follows strict MVC:
- `models/` — data structures, algorithms, no I/O (vein_identifier, vein_labeler, vein_skeleton, vein_graph, vein_map, wing_geometry, geojson_parser)
- `controllers/` — pipeline orchestration (analysis_controller.run_pipeline, measurement_controller)
- `views/` — overlays (overlay_view) and CSV export (results_view)
- `gui/` — step-by-step PyQt5 debugger interface

**Import plumbing**: LandmarkLocator, HingeChopper, and modelTOjson export bare module names (`landmark_locator`, `hinge_chopper`, `modeltojson`). Entry scripts add their parent dirs to `sys.path`. Always include `<project_root>`, `<project_root>/HingeChopper`, and `<project_root>/modelTOjson` when importing across modules.

**GeoJSON data flow**: Preprocessing outputs `properties.class` (e.g. "vein", "intervein", "wing"). WingVeinAnalyzer's parser (`geojson_parser.py:73`) falls back from `properties.classification.name` to `properties.class`, so both formats work.

**WingVeinAnalyzer file discovery** (`gui/file_selector.py`): `discover_file_pairs()` only finds `.tif`/`.tiff` images. TRACE bypasses this by calling `run_pipeline()` directly with explicit paths, which uses `cv2.imread()` (supports all formats).

**TRACE measurement groups**: `MEASUREMENT_GROUPS` in `pipeline.py` defines which CSV column groups are available. `filter_csv_columns()` prunes the consolidated CSV to only selected groups. The GUI exposes these as checkboxes.

## Key Data Structures

- `WingVeinAnalyzer.PipelineResult` — assignments, measurements, poly_names, overlay paths
- `WingVeinAnalyzer.VeinAssignment` — vein_id, status, confidence, length_px, LineString geometry
- `WingVeinAnalyzer.WingMeasurements` — per-vein lengths, wing dims, compartment areas, intervein areas
- `preprocessing.PipelineResult` — image_path, landmarks, geojson paths, error, stages_completed
- `TRACE.TraceResult` — image_path, overlay paths, error, error_stage

## Test Data

- `WingVeinAnalyzer/test_data/testwing{1-5}/` — TIFF + GeoJSON + expected overlay GeoJSON pairs
- `preprocessing/testinput_images/` — raw wing TIFFs
- `preprocessing/testinput_DLmodels/` — landmark (.pt) and segmentation model checkpoints
