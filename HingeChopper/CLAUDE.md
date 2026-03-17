# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What This Is

HingeChopper is a standalone CLI tool that blacks out the hinge (proximal) region of Drosophila wing images. It uses landmark points from GeoJSON files to construct a hinge line, then masks the proximal side to black. Part of the broader mapThemVeins project but has no dependency on WingVeinAnalyzer.

## Running

```bash
# Single image
python hinge_chopper.py image.tif landmarks.geojson -o output.tif

# Batch mode (pairs by stem: foo.tif + foo_landmarks.geojson)
python hinge_chopper.py --batch pics/ landmarks/ -o output/
```

## Dependencies

`opencv-python`, `numpy` — no shapely, no other project packages.

## Architecture

Single file: `hinge_chopper.py`. Pipeline flow:

1. `load_landmarks()` — parse GeoJSON → dict of name → (x, y)
2. `build_hinge_line()` — ordered points: subcostal break → [L1-Rs] → [L4-L5] → alula notch
3. `extend_to_image_edges()` — ray-cast endpoints to image boundary
4. `make_proximal_mask()` — cross-product test with DTip to determine distal side; fill proximal side polygon via cv2.fillPoly
5. `chop_hinge()` — orchestrator: load image, apply mask, save

## Landmark Requirements

Required: `DTip`, `subcostal break`, `alula notch`. Optional (improve line shape): `L1-Rs`, `L4-L5`.

GeoJSON format: features with `properties.classification.name` and `geometry.coordinates` (Point type).

## Test Data

- `pics/` — wing images (.tif, .bmp), 30 files across 4 genotypes
- `landmarks/` — corresponding `*_landmarks.geojson` files
- Batch mode pairs them by matching stem (strip `_landmarks` suffix)
