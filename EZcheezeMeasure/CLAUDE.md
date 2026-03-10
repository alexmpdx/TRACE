# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Purpose

Measures distances between ML-detected landmark points on Drosophila (fruit fly) wing images. An upstream ML model outputs paired files per wing: a GeoJSON with landmark coordinates and a JPG with the landmarks overlaid. This tool computes inter-landmark distances and ratios, exports a CSV, and generates annotated overlay images.

## Running

```bash
python3 measure_landmarks.py                    # defaults to landmarkPoints/ folder
python3 measure_landmarks.py /path/to/folder    # custom input folder
```

Output goes to `output/` (sibling of input folder): `landmark_measurements.csv` and `overlays/` with annotated JPGs.

## Dependencies

Python 3.10+, opencv-python, numpy. Standard library: json, csv, math, pathlib.

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

Single-script tool (`measure_landmarks.py`). Key functions:
- `load_landmarks()` — parses GeoJSON into `{name: (x, y)}` dict
- `euclidean()` — point-to-point distance
- `draw_overlay()` — renders measurement lines (cyan=wing length, magenta=CV distance) and landmark labels onto wing JPG using OpenCV
- `main()` — batch processes all `*_landmarks.geojson` files, writes CSV and overlays

## Filename Convention

Image IDs encode genotype, date, sex, and specimen number:
`{genotype}_{date}_{sex}_{number}` — e.g., `BDSC31488Fx25752M_021126_female_0001`
