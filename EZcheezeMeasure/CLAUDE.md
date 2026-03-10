# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Purpose

Measures distances between ML-detected landmark points on Drosophila (fruit fly) wing images. An upstream ML model outputs paired files per wing: a GeoJSON with landmark coordinates and a JPG with the landmarks overlaid. This tool computes inter-landmark distances and ratios, exports a CSV, and generates annotated overlay images.

## Running

```bash
# Full pipeline: ML landmark detection → measurements (opens folder picker)
python3 run_pipeline.py                              # uses default checkpoint
python3 run_pipeline.py /path/to/checkpoint.pt       # custom checkpoint

# Standalone measurement from existing GeoJSON files
python3 measure_landmarks.py                         # defaults to landmarkPoints/
python3 measure_landmarks.py /path/to/folder         # custom input folder
```

Output goes to `output/` (sibling of input folder): `landmark_measurements.csv` and `overlays/` with annotated JPGs.

## Dependencies

Python 3.10+, opencv-python, numpy, torch, torchvision, pyyaml, landmark_locator (local package from `../LandmarkLocator`). Standard library: json, csv, math, pathlib, tkinter.

## Windows Executable

Built via PyInstaller on GitHub Actions CI. The workflow (`.github/workflows/build-windows.yml`) produces a `EZcheezeMeasure-windows.zip` artifact.

### One-time setup: upload the model checkpoint

```bash
gh release create model --title "Model Checkpoint" --notes "Trained model for landmark detection"
gh release upload model /path/to/landmark_model_grace.5.pt
```

### Triggering a build

- **Manual**: Actions → "Build Windows Executable" → Run workflow
- **On release**: creating a GitHub release auto-triggers the build and attaches the zip

### How it works

- `EZcheezeMeasure.spec` — PyInstaller config, bundles the model checkpoint + all deps into a `dist/EZcheezeMeasure/` folder with `EZcheezeMeasure.exe`
- Uses CPU-only PyTorch to keep the build size down
- `_get_base_dir()` in `run_pipeline.py` handles the frozen exe path (`sys._MEIPASS`) vs normal Python

## Input Data Format

Each wing has two files in the input folder with matching basenames:
- `*_landmarks.geojson` — GeoJSON FeatureCollection of Point features. Each feature has `properties.classification.name` identifying the landmark.
- `*_landmarks.jpg` — wing image with ML overlay dots.

### Landmark Set (7 points per wing)

`ACV.p` (posterior ACV end), `alula notch`, `DTip` (distal wing tip), `L1-Rs` (L1/radial-sector junction), `L4-L5` (L4/L5 junction), `PCV.a` (anterior PCV end), `subcostal break`

### Current Measurements

| Measurement | Endpoints | Column |
|---|---|---|
| Wing length | L1-Rs → DTip | `wing_length_px` |
| CV distance | ACV.p → PCV.a | `cv_distance_px` |
| CV/WL ratio | cv_distance / wing_length | `cv_wl_ratio` |

## Architecture

Two-tier design:

- `run_pipeline.py` — full pipeline: loads LandmarkLocator model, predicts landmarks on input images, computes measurements, writes CSV + overlay JPGs. Entry point for the Windows exe.
- `measure_landmarks.py` — standalone measurement from pre-existing GeoJSON files. Also provides `draw_overlay()` and `euclidean()` imported by the pipeline.

`LANDMARK_TO_GEOJSON` dict in `run_pipeline.py` maps internal model names (`acv_p`, `dtip`, etc.) to GeoJSON display names (`ACV.p`, `DTip`, etc.).

## Filename Convention

Image IDs encode genotype, date, sex, and specimen number:
`{genotype}_{date}_{sex}_{number}` — e.g., `BDSC31488Fx25752M_021126_female_0001`
