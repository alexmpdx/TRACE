# TRACE

End-to-end *Drosophila* wing morphology analysis. Takes a folder of brightfield wing images and produces named vein/intervein overlays, a per-batch measurements CSV (areas, lengths, optional custom landmark distances), and intermediate GeoJSON artifacts. The orchestration spans six preprocessing stages and six analysis steps; this README is the entry point.

For a visual map of the full pipeline (nodes, edges, artifacts), open the interactive viewer:

```bash
TRACE/.venv-pipeline-map/bin/python TRACE/pipeline_map.py
```

A static reference is rendered at `TRACE/pipeline_map_reference.png`. The map is intentionally kept up to date — see [`PIPELINE_MAP_GUIDE.md`](PIPELINE_MAP_GUIDE.md) before adding new stages.

---

## What the pipeline does

```
User input ──┐
              ├─→ Stage 1: Resolution adjust       (resolutionAdjust)
              ├─→ Stage 2: Wing isolation          (wingIsolator, optional)
              ├─→ Stage 3: Landmark detection      (LandmarkLocator)
              ├─→ Stage 4: Hinge chop              (HingeChopper)
              ├─→ Stage 5: Segmentation            (modelTOjson)
              └─→ Stage 6: Wing rotation           (wingRotator, optional)
                              │
                              ▼
                  Stage 2 (identifyFeatures):
                  Step 1: Parse inputs
                  Step 2: Build skeleton graph
                  Step 3: Anchor landmarks
                  Step 4: Compute wing axis
                  Step 5: Call veins
                  Step 6: Call intervein regions
                              │
                              ▼
            measurementMaker (optional): user-defined landmark-pair distances
                              │
                              ▼
            WingResult ──→ overlays + CSV + per-wing GeoJSON
```

The only thing **the user must provide** is the wing image. Models (landmark, segmentation, optional wing-isolation) and `PipelineConfig` ship with the project under `TRACE/models/`.

---

## Quick start

### GUI (recommended)

```bash
python TRACE/run_gui.py
```

First-launch defaults:

- **Models**: `TRACE/models/{landmarks, vein-intervein, wingIsolation}` (auto-discovered if present).
- **Outputs**: full overlay PNGs + batch CSV + per-wing GeoJSON.
- **Scale**: blank — the user has to enter a µm/px conversion factor or `--scale` on the CLI.

Settings → General is the main configuration panel. Restore Defaults reapplies the bundled models and clears any persisted landmark-gate override. "wipe my memories" on the main window goes further — clears every QSettings value.

### CLI

```bash
python TRACE/run_cli.py \
  -i <input_folder> \
  -o <output_folder> \
  --landmark-model TRACE/models/landmarks \
  --segmentation-model TRACE/models/vein-intervein \
  --scale 0.483
```

Flags worth knowing:

| Flag | Purpose |
| --- | --- |
| `--input` / `-i` | Folder of wing images. Recursive search via `--recursive`. |
| `--output` / `-o` | Output folder (created if missing). |
| `--landmark-model` | Model folder with `best_fold*.pt` + `gate_config.yaml`. |
| `--segmentation-model` | modelTOjson folder with `weights` + `metadata.json`. |
| `--wing-isolation-model` | Optional Stage 2 wing-vs-background model. |
| `--scale` | Microns per pixel. Required unless `--config` provides one. |
| `--outputs` | Comma-separated subset of `OUTPUT_TYPES` keys. |
| `--config` | JSON file written by the GUI's Export button. Includes the landmark gate override when one was set. |
| `--gate-override-yaml` | Per-landmark threshold YAML. Wins over the override in `--config`. |
| `--workers` | Stage-2 parallelism (default 1). The Calibrate widget in the GUI estimates a safe value. |

`python TRACE/run_cli.py --help` for the full list.

---

## Supported wing-image formats

Standard: `tif` / `tiff` / `bmp` / `png` / `jpg` / `jpeg`
Adobe: `psd` / `psb`
Modern: `heic` / `heif` / `svg`
Camera RAW: `dng` / `nef` / `cr2` / `cr3` / `arw` / `raf` / `orf` / `pef` / `rw2` / `srw` / `raw`
Microscopy: `czi` / `nd2` / `lif` / `lsm` (auto-converted to OME-TIFF in Stage 0)

JPGs trigger a "be careful" dialog in the GUI — JPEG compression can shift vein widths by a pixel or two. Convert to TIFF if possible.

---

## Outputs

All written into `<output_folder>` with a per-image stem prefix. Toggle each on/off in Settings → General → Output options, or via `--outputs` on the CLI.

| Output | Filename | Notes |
| --- | --- | --- |
| Vein + intervein overlay | `<stem>_overlay.png` | Combined view. Per-vein/per-region colors + opacities are configurable. |
| Landmarks overlay | `<stem>_landmarks_overlay.png` | Predicted landmarks drawn on the image. |
| Segmentation overlay | `<stem>_segmentation_overlay.png` | Raw vein/intervein semantic-segmentation classes. |
| AP compartment overlay | `<stem>_ap_overlay.png` | Anterior/posterior split with percentage labels. |
| CV ratio overlay | `<stem>_cv_ratio_overlay.png` | Anterior vs posterior crossvein position. |
| Per-wing GeoJSON | `<stem>_output.geojson` | Named veins + intervein polygons. QuPath / napari compatible. |
| Batch measurements CSV | `measurements.csv` | One row per image. Areas, lengths, custom distances. |
| Isolated wing image | `<stem>_isolated.tif` | Masked single-wing image (Stage 2 artifact). |
| Wing after hinge removal | `<stem>_chopped.tif` | Hinge-blanked image (Stage 4 artifact). |

The `measurements.csv` columns are content-gated by what other outputs were requested — see `MEASUREMENT_GROUPS` in `pipeline.py` for the breakdown.

---

## Models

The pipeline expects three model folders. The bundled defaults under `TRACE/models/` are auto-loaded on first launch (or after Restore Defaults / wipe-my-memories):

| Key | Folder | What's inside |
| --- | --- | --- |
| Landmark points | `TRACE/models/landmarks/` | `best_fold0.pt` … `best_fold4.pt` (5-fold ensemble), `gate_config.yaml`, `training_chart.png`. |
| Wing features | `TRACE/models/vein-intervein/` | modelTOjson semantic-segmentation weights + `metadata.json`. |
| Wing isolation (optional) | `TRACE/models/wingIsolation/` | modelTOjson wing-vs-background weights + `metadata.json`. |

The landmark model's `gate_config.yaml` is the only authoritative source of per-landmark confidence-gate thresholds (the previous "embed in checkpoint" path was retired). The Landmarks tab in Settings reads this YAML directly. Filename can be `gate_config.yaml`, `gate_config_*.yaml`, anything containing `gate_config`, or just any `.yaml` in the folder — tiered fallback.

To use a different model, point the Settings → Models picker (or CLI `--*-model` flag) at any folder following the layout above. The `Browse...` button is folder-only — single-checkpoint inference is no longer supported.

---

## Configuration

### In-session

Settings → General is the canonical place. Sub-tabs cover Custom Distances (measurementMaker pairs), Landmarks (per-landmark gate thresholds), Models (paths + per-model training µm/px), Skeletonization & Pruning, Bridging, Tracing, and Intervein.

### Persistence

Everything in Settings is persisted to QSettings between launches. The landmark gate override (from the Landmarks tab) persists too — clear it via Restore Defaults.

### Export / Import

Settings → Export writes the full `PipelineConfig` plus any gate override to a JSON file. Settings → Import reads it back. The same JSON can be passed to the CLI via `--config`. Old config files (no gate-override key) still load — the in-session override is left alone.

---

## Module layout

TRACE depends on several sibling packages in the parent project:

```
mapThemVeins/
  TRACE/                 ← this folder (top-level orchestrator)
  preprocessing/         ← Stages 1-5 wrapper
  LandmarkLocator/       ← Stage 3 (landmark detection)
  HingeChopper/          ← Stage 4 (hinge removal)
  modelTOjson/           ← Stage 5 (segmentation → GeoJSON)
  wingIsolator/          ← Stage 2 (wing isolation)
  wingRotator/           ← Stage 6 (canonical rotation)
  resolutionAdjust/      ← Stage 1 (rescale to model's training µm/px)
  identifyFeatures/      ← Stage 2 of TRACE (vein ID, naming, measurement)
  measurementMaker/      ← Custom landmark-distance pairs (post-CSV augmentation)
  scaleEstimator/        ← Optional µm/px estimation utility
```

`run_cli.py` and `run_gui.py` set up `sys.path` so the imports just work. If you import TRACE from outside `run_*.py`, you'll need to mirror that path-setup yourself.

---

## Inside this folder

| File | Purpose |
| --- | --- |
| `cli.py` | Argparse + main entrypoint for the CLI. |
| `gui.py` | PyQt5 main-window logic, settings persistence, run orchestration. |
| `settings_dialog.py` | The Settings dialog (tabs, model picker, gate config). |
| `pipeline.py` | Glue between preprocessing and identifyFeatures + result tracking. |
| `pipeline_map.py` | Interactive vispy diagram of the pipeline; reads `pipeline_layout.plain`. |
| `pipeline_layout.plain` | Cached graphviz output. Regenerate with `pipeline_map.py --regenerate-layout`. |
| `config_io.py` | `save_settings` / `load_settings` for the pipeline-config JSON format. |
| `calibrate_widget.py` | Estimates a safe worker count from a sample wing run. |
| `presets/` | JSON presets that the Settings dialog exposes via the preset dropdown. |
| `models/` | Bundled DL models (see *Models* section). |
| `PIPELINE_MAP_GUIDE.md` | Layout principles for `pipeline_map.py`. Read before editing the diagram. |

---

## Development

### Code style

`black` + `isort` + `flake8`, all at 120-char line length. Run manually:

```bash
pre-commit run --all-files
```

### Re-running graphviz for the pipeline map

```bash
python TRACE/pipeline_map.py --regenerate-layout
```

Requires the `dot` binary (`brew install graphviz`). The cached layout file (`pipeline_layout.plain`) is checked into the repo so a fresh clone can render the map without graphviz.

### Pipeline-map venv

The map uses vispy + numpy and lives in its own venv to avoid bloating the main TRACE install:

```bash
# Already exists at TRACE/.venv-pipeline-map/
.venv-pipeline-map/bin/python pipeline_map.py
```
