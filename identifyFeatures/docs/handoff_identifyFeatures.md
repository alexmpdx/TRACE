# Handoff Document: identifyFeatures → Full Pipeline Integration

## What This Document Is For

You are integrating `identifyFeatures` — a landmark-anchored Drosophila wing vein identification tool — into a broader pipeline that includes preprocessing (landmark detection, hinge masking, semantic segmentation) and a GUI. This document gives you everything you need to understand the module, its inputs/outputs, the surrounding ecosystem, and what remains to be built.

---

## 1. The Big Picture

The mapThemVeins project analyzes Drosophila (fruit fly) wing images. The full pipeline takes raw brightfield microscopy images of dissected wings and produces identified vein structures, named intervein regions, overlay visualizations, and measurement CSVs.

### Pipeline stages

```
RAW WING IMAGES (.tif/.bmp)
       ↓
[Stage 1] LandmarkLocator — deep learning heatmap model → {stem}_landmarks.geojson (13 anatomical points)
       ↓
[Stage 2] HingeChopper — blacks out proximal hinge region using landmark coords → {stem}_chopped.tif
       ↓
[Stage 3] modelTOjson — semantic segmentation (Torch SMP or ONNX) → vein/intervein/hinge polygons → {stem}_detections.geojson
       ↓
[Stage 4] identifyFeatures — landmark-anchored vein ID → named veins, regions, measurements, overlays
```

Stages 1–3 already exist as the `preprocessing` module. Stage 4 is `identifyFeatures`. The `TRACE` module was intended to orchestrate everything end-to-end but currently only runs preprocessing — vein analysis (Stage 4) is stubbed as TODO.

### Existing modules (all under `/Users/alexmurphy/Desktop/claude_scripts/mapThemVeins/`)

| Directory | What it does | GUI? |
|---|---|---|
| `LandmarkLocator/` | Deep learning landmark detection (training + inference) | PyQt5 (training monitor) |
| `HingeChopper/` | Blacks out hinge region using landmark coords | CLI only |
| `modelTOjson/` | Segmentation inference with GeoJSON export | PyQt5 |
| `preprocessing/` | Orchestrates stages 1–3 | PyQt5 + CLI |
| `TRACE/` | Master pipeline (preprocessing + identifyFeatures) | PyQt5 + CLI |
| `identifyFeatures/` | Landmark-anchored vein identification (the analysis stage in TRACE) | CLI |

### What needs to happen

1. Continue refining identifyFeatures as TRACE's analysis stage
2. Build or extend a GUI that covers the full pipeline end-to-end
3. Make it accessible to bench scientists (biology researchers, not programmers)

---

## 2. identifyFeatures: What It Does

Given a pixel-classifier segmentation (vein/intervein GeoJSON polygons) and deep-learning landmark points, it identifies and names individual veins (L1–L6, Rs, ACV, PCV, costa) and intervein regions (marginal, submarginal, 1st basal, 1st posterior, discal, 2nd posterior, 3rd posterior). Designed to handle mutant wings with missing, partial, or ectopic veins.

### Key design principle

**Landmarks are primary, not supplementary.** Vein identity flows outward from 6 reliable landmark junctions (subcostal break, alula notch, L1-Rs, L2-L3, L4-L5, DTip), rather than relying on spatial priors.

### Performance on current test data

- 30/30 specimens: all 10 canonical veins identified, 7/7 regions named
- Mean region IoU: 0.91, mean vein IoU: 0.61, 98.8% detection rate

---

## 3. identifyFeatures API

### Programmatic entry point

```python
from pathlib import Path
from identify_features.controllers.pipeline import identify_wing
from identify_features.config import PipelineConfig

config = PipelineConfig(um_per_px=0.483)
result = identify_wing(
    detection_geojson=Path("specimen_detections.geojson"),
    landmarks_geojson=Path("specimen_landmarks.geojson"),
    image_path=Path("specimen.tif"),       # optional (determines image_shape)
    config=config,                          # optional (sensible defaults)
    specimen_id="specimen_001",             # optional (for CSV labeling)
)
# result is a WingResult dataclass
```

**Location:** `identifyFeatures/identify_features/controllers/pipeline.py:33`

### CLI entry point

```bash
# Single specimen
identify-features <detection.geojson> <landmarks.geojson> [image.tif] \
    --output-dir output/ --um-per-px 0.483 --overlay

# Batch mode
identify-features --batch <det_dir/> <lm_dir/> [img_dir/] \
    --output-dir output/ --workers 4 --overlay
```

**Location:** `identifyFeatures/identify_features/cli.py`

### Batch processing

Uses `ProcessPoolExecutor` with configurable worker count. The `_process_one()` function (cli.py:49) is the per-specimen entry point for parallel execution:

```python
def _process_one(args_tuple):
    # args: stem, det_path, lm_path, img_path, output_dir, um_per_px, verbose, overlay
    # returns: (stem, success_bool, message, WingResult_or_None)
```

WingResult objects are pickled between processes, so all fields must be picklable (they are — Shapely geometries and basic Python types).

---

## 4. Input Formats

### Detection GeoJSON (`*_detections.geojson`)

GeoJSON FeatureCollection. Each Feature has `properties.class` = `"vein"` | `"intervein"` | `"hinge junk"` (discarded). Geometries are Polygons.

**Parser:** `identify_features/models/geojson_io.py:17` → returns `(vein_polygons, intervein_polygons)`

### Landmarks GeoJSON (`*_landmarks.geojson`)

GeoJSON FeatureCollection with Point features. Each has `properties.classification.name` = landmark name.

**13 landmarks total:**
- 5 reliable junctions (required): `subcostal break`, `alula notch`, `L1-Rs`, `L2-L3`, `L4-L5`
- 4 soft endpoints (helpful hints): `DTip`, `L2.d`, `L4.d`, `L5.d`
- 4 unreliable crossvein markers (fallback only): `ACV.a`, `ACV.p`, `PCV.a`, `PCV.p`

**Parser:** `identify_features/models/geojson_io.py:52` → returns `dict[str, Landmark]`

### Image file (optional)

`.tif`, `.bmp`, `.png`, `.jpg`. Used only to determine image dimensions (height, width) for rasterization. If absent, dimensions are estimated from polygon bounding boxes with 100px margin.

### File stem matching

Stems can start with `-` (e.g., `-CTRL_PknRNAi_108870_0007`). At least one file has a spurious space (`-CTRL_PknRNAi_108870_0004.tif .geojson`). The CLI handles both patterns. Convention:

| File | Pattern |
|---|---|
| Image | `{stem}.tif` (or .bmp/.png/.jpg) |
| Detections | `{stem}_detections.geojson` |
| Landmarks | `{stem}_landmarks.geojson` |

**Note:** preprocessing outputs segmentation as `{stem}.geojson` (no `_detections` suffix). identifyFeatures expects `{stem}_detections.geojson`. This naming mismatch will need reconciling when wiring the pipeline.

---

## 5. Output Formats

### WingResult (in-memory)

```python
@dataclass
class WingResult:
    specimen_id: str
    veins: list[VeinIdentification]       # 10 canonical + ectopic EVn
    intervein_regions: list[InterveinRegion]  # 7 named regions
    landmarks: dict[str, Landmark]        # snapped landmarks with graph node IDs
    wing_outline: Optional[Polygon]       # union of all input polygons
    warnings: list[str]
```

**Location:** `identify_features/models/datatypes.py:205`

### VeinIdentification

```python
@dataclass
class VeinIdentification:
    vein_id: str              # "L1", "L2", "ACV", "EV1", etc.
    vein_type: VeinType       # LONGITUDINAL, CROSSVEIN, COSTA, RADIAL_SECTOR
    status: VeinStatus        # IDENTIFIED, INFERRED, PARTIAL, ABSENT, ECTOPIC
    centerline: Optional[LineString]
    tissue_polygon: Optional[Polygon | MultiPolygon]
    edge_ids: list[int]
    length_px: float
    confidence: float
    evidence: list[str]       # human-readable reasoning
    landmark_anchors: list[str]
```

### InterveinRegion

```python
@dataclass
class InterveinRegion:
    name: str                 # "marginal", "discal", "1st posterior", etc.
    polygon: Optional[Polygon | MultiPolygon]
    bounding_veins: set[str]
    area_px2: float
    status: str               # "identified", "inferred", "merged"
```

### File outputs

| Output | Format | When |
|---|---|---|
| `{stem}_output.geojson` | GeoJSON FeatureCollection (GT_naming format) | Always |
| `{stem}_measurements.csv` | Long format (one row per feature) | Single mode |
| `measurements.csv` | Wide format (one row per specimen) | Batch mode |
| `{stem}_overlay.png` | BGR PNG with tinted regions + colored veins + labels | With `--overlay` flag |

### CSV measurements include

- Wing area (px, µm²)
- Wing length: L1-Rs → DTip distance (px, µm)
- Crossvein distance: ACV.p → PCV.a (px, µm)
- CV ratio: crossvein distance / wing length (dimensionless)
- Anterior/posterior compartment areas (split along L4 axis)
- Per-vein tissue area and centerline length
- Per-region polygon area

---

## 6. Pipeline Internals (6 steps)

**Location:** `identify_features/controllers/pipeline.py`

1. **Parse inputs** → vein polygons, intervein polygons, landmarks, wing outline, image shape
2. **Build skeleton graph** → RIDGE skeletonization of vein mask → 17-step cleanup → NetworkX graph
3. **Anchor landmarks** → snap each reliable landmark to nearest skeleton node
4. **Compute wing axis + trace veins** → proximal/distal axis from landmarks → 6-phase edge labeling from landmark junctions outward
5. **Assign vein tissue + split intervein polygons** → assign tissue polygons to veins → h-maxima watershed for merged regions
6. **Name intervein regions** → spatial adjacency to traced vein centerlines → 7 canonical names

---

## 7. Configuration

**Location:** `identify_features/config.py` — `PipelineConfig` dataclass

Key parameters (all have sensible defaults):

| Parameter | Default | Purpose |
|---|---|---|
| `um_per_px` | 0.483 | Scale factor (microns per pixel) |
| `skeleton_methods` | `[RIDGE]` | Skeletonization method |
| `prune_min_length_vein_widths` | 2.0 | Branch pruning threshold |
| `bridge_max_gap_um` | 200 µm | Gap bridging max distance |
| `snap_radius_um` | 100 µm | Landmark snap radius |
| `crossvein_min_length_vw` | 4.0 | Crossvein detection floor |

Most thresholds are resolution-independent (µm or vein-width multiples). The `to_px()` method converts µm values using `um_per_px`.

---

## 8. Biological Domain Reference

### Drosophila wing vein anatomy

The wing has a stereotyped vein pattern: 6 longitudinal veins (L1–L6), 2 crossveins (ACV, PCV), a costa (anterior margin), and Rs (radial sector, a short proximal stem). These veins partition the wing blade into 7 named intervein regions.

### Canonical ordering (anterior → posterior)

**Veins:** costa, L1, Rs, L2, L3, ACV, L4, PCV, L5, L6

**Regions:** marginal, submarginal, 1st basal, 1st posterior, discal, 2nd posterior, 3rd posterior

### Topology constants

All defined in `identify_features/models/topology.py`:
- `VEIN_AP_ORDER` — canonical vein ordering
- `REGION_AP_ORDER` — canonical region ordering  
- `JUNCTION_TOPOLOGY` — which veins meet at each landmark junction
- `REGION_EXPECTED_VEINS` — which veins bound each region
- `VEIN_COLORS` / `REGION_COLORS` — display colors (RGB)

### Mutant wings

The pipeline handles missing veins (status=ABSENT), partial veins (PARTIAL), inferred veins (INFERRED), and ectopic/extra veins (ECTOPIC, labeled EV1, EV2, ...). This is important because many experimental genotypes have altered vein patterns.

---

## 9. Dependencies

**Python ≥ 3.10** required.

```
shapely>=2.0        # polygon/line geometry
networkx>=3.0       # skeleton graph
scikit-image>=0.20  # skeletonization, morphology, watershed
numpy>=1.24
opencv-python>=4.8  # image I/O and drawing
scipy>=1.10         # distance transforms
pandas>=2.0
```

The existing GUIs in sibling modules all use **PyQt5** — that's the established GUI framework for this project.

---

## 10. Existing TRACE Pipeline (what's already built)

**Location:** `TRACE/`

- `pipeline.py` — calls `preprocessing.process_folder()`, then stubs analysis as TODO
- `gui.py` — PyQt5 QMainWindow (414 lines) with dark Fusion theme, QThread worker, progress callbacks
- `cli.py` — argparse CLI with `--input`, `--output`, `--landmark-model`, `--segmentation-model`, `--scale`
- Returns `TraceResult(image_path, error, error_stage)`

The TRACE GUI has: folder selection, model path selectors, progress bar, image list, and stage indicators. It runs preprocessing asynchronously via a QThread worker with a progress callback signature:

```python
progress_callback(idx: int, total: int, name: str, stage: str, detail: str)
```

### What TRACE is missing

- Stage 5 (identifyFeatures) is not wired in
- No results viewing/browsing after processing
- No way to inspect or correct individual specimen results
- No measurement export UI

---

## 11. Key Integration Points

### Wiring identifyFeatures into TRACE

The preprocessing module outputs `{stem}_landmarks.geojson` and `{stem}.geojson` (segmentation) per image. identifyFeatures needs `{stem}_landmarks.geojson` and `{stem}_detections.geojson`. The `_detections` suffix mismatch needs resolving — either rename in preprocessing or make identifyFeatures accept both patterns.

### From preprocessing output to identifyFeatures input

```python
from identify_features.controllers.pipeline import identify_wing
from identify_features.config import PipelineConfig

config = PipelineConfig(um_per_px=scale)
result = identify_wing(
    detection_geojson=output_dir / f"{stem}_detections.geojson",
    landmarks_geojson=output_dir / f"{stem}_landmarks.geojson",
    image_path=input_dir / f"{stem}.tif",
    config=config,
    specimen_id=stem,
)
```

### Batch parallelization

identifyFeatures already handles batch parallel processing with ProcessPoolExecutor. For GUI integration, you'll likely want to run this in a QThread (like TRACE already does for preprocessing) and emit progress signals.

### Output files to generate

Per specimen: `_output.geojson`, `_overlay.png`. Combined: `measurements.csv` (wide format). The export functions are:

```python
from identify_features.views.csv_export import export_csv, export_csv_batch
from identify_features.views.geojson_export import export_geojson
from identify_features.views.overlay import render_overlay_to_file
```

---

## 12. Pre-commit Hooks and CI

The repo has pre-commit hooks: **isort**, **black**, **flake8**. A pre-push hook checks that `docs/pipeline_reference.md` and `docs/project_plan.md` have been updated if any `.py` files changed (compares file modification times). If you modify source code, touch the doc files or update them before pushing.

---

## 13. File Map

```
identifyFeatures/
├── identify_features/
│   ├── cli.py                    # CLI entry point (single + batch)
│   ├── config.py                 # PipelineConfig dataclass (40+ params)
│   ├── controllers/
│   │   └── pipeline.py           # identify_wing() — main orchestrator
│   ├── models/
│   │   ├── datatypes.py          # WingResult, VeinIdentification, InterveinRegion, Landmark, enums
│   │   ├── topology.py           # Vein/region ordering, junction topology, colors
│   │   ├── skeleton.py           # RIDGE skeletonization + 17-step cleanup
│   │   ├── vein_tracer.py        # 6-phase landmark-anchored vein labeling
│   │   ├── costa_detector.py     # Costa margin-band analysis
│   │   ├── junction_resolver.py  # Merge longitudinals through crossveins
│   │   ├── landmark_anchor.py    # Snap landmarks to skeleton nodes
│   │   ├── intervein_splitter.py # h-maxima watershed for merged regions
│   │   ├── intervein_namer.py    # Name regions by vein adjacency
│   │   ├── wing_axis.py          # Proximal/distal axis from landmarks
│   │   └── geojson_io.py         # Parse detection + landmark GeoJSON
│   ├── views/
│   │   ├── csv_export.py         # Long + wide format CSV (measurements, AP areas, CV ratio)
│   │   ├── geojson_export.py     # GT_naming format GeoJSON output
│   │   └── overlay.py            # PNG overlay visualization
│   └── utils/
│       ├── geometry_utils.py     # Shapely helpers
│       ├── graph_utils.py        # NetworkX helpers
│       └── image_utils.py        # Rasterization
├── docs/
│   ├── pipeline_reference.md     # Detailed algorithm documentation
│   └── project_plan.md           # Status and roadmap
├── geojsons/                     # Test detection GeoJSONs (30 specimens)
├── LandmarkLocator_output/       # Test landmark GeoJSONs
├── OGpics/                       # Test wing images
└── pyproject.toml                # Package definition, entry point, dependencies
```

---

## 14. Gotchas and Things to Know

1. **Filenames can start with `-`** (e.g., `-CTRL_PknRNAi_108870_0007`) and one has a spurious space. Handle these in any file-matching logic.

2. **Unreliable landmarks (ACV.a, ACV.p, PCV.a, PCV.p)** should never be used as hard constraints — they're fallback only. The reliable landmarks are the 5 junctions.

3. **The AP compartment split** uses L4's bounding box with a distal-end pivot shift of 0.5× median vein width toward the anterior.

4. **Pre-commit hooks will reformat your code** — isort and black run automatically. Don't fight them.

5. **The pre-push hook** blocks pushes if docs are stale. Touch `docs/pipeline_reference.md` and `docs/project_plan.md` (or update them) after modifying `.py` files.

7. **ProcessPoolExecutor** is used for batch parallelism. WingResult must remain picklable.

8. **All existing GUIs use PyQt5** with a dark Fusion theme. Maintain visual consistency.

9. **Scale default is 0.483 µm/px** — this is the standard for the lab's microscope setup.

10. **TRACE's gui.py** (414 lines) is the best template for the integrated GUI — it already has the QThread worker pattern, progress callbacks, and dark theme.
