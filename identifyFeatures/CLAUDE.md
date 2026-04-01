# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Landmark-anchored Drosophila melanogaster wing vein identification tool. Given a pixel-classifier segmentation (vein/intervein GeoJSON) and deep-learning landmark points, identifies and names individual veins (L1-L6, Rs, ACV, PCV, costa) and intervein regions. Designed to handle mutant wings with missing, partial, or ectopic veins.

## Architecture

MVC layout under `identify_features/`:
- **models/**: Pure logic and data structures. `topology.py` is the single source of truth for wing vein biology (vein ordering, junction topology, region boundaries). `skeleton.py` provides 3 user-selectable skeletonization methods. `vein_tracer.py` is the core algorithm — traces veins outward from landmark-anchored junction nodes.
- **controllers/**: `pipeline.py` orchestrates the 10-step pipeline.
- **views/**: Output rendering (overlays, GeoJSON export, CSV).
- **utils/**: Shared helpers for NetworkX graphs, Shapely geometry, image I/O.

## Key Design Principle

**Landmarks are primary, not supplementary.** Vein identity flows outward from 6 reliable landmark junctions (subcostal break, alula notch, L1-Rs, L2-L3, L4-L5, DTip). This inverts the old WingVeinAnalyzer approach of spatial-prior guessing. Do NOT use unreliable landmarks (ACV.a, ACV.p, PCV.a, PCV.p).

## Build and Run

```bash
pip install -e .
identify-features <detection_geojson> <landmarks_geojson> [image]
identify-features --batch geojsons/ LandmarkLocator_output/ OGpics/ --output-dir output/
```

## Input Data Formats

- **Detection GeoJSON** (`geojsons/*_detections.geojson`): `properties.class` = `"vein"` | `"intervein"` | `"hinge junk"` (discard hinge junk)
- **Landmarks GeoJSON** (`LandmarkLocator_output/*_landmarks.geojson`): `properties.classification.name` = landmark name
- **GT naming** (`GT_naming/*.geojson`): `properties.classification.name` = feature name, with `objectType: "annotation"`. Dev-only ground truth — not a pipeline input.
- File stem matching across folders: `OGpics/{stem}.tif`, `geojsons/{stem}_detections.geojson`, `LandmarkLocator_output/{stem}_landmarks.geojson`. Note: stems can start with `-` and one file has a space (`-CTRL_PknRNAi_108870_0004.tif .geojson`).

## Dependencies

shapely, networkx, scikit-image, numpy, opencv-python, scipy
