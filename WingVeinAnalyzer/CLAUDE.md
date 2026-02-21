# WingVeinAnalyzer — Claude Code Project Instructions

## Project Purpose
A Python tool to identify and measure named veins (L1–L5, ACV, PCV) in *Drosophila* wing brightfield images using GeoJSON annotation files (intervein polygon or vein LineString annotations) produced by QuPath or similar tools.

## Architecture
Strict MVC. No business logic in views. No I/O in models.

```
WingVeinAnalyzer/
├── models/
│   ├── geojson_parser.py      # Parse GeoJSON → typed dataclasses
│   ├── vein_graph.py          # Graph from polygon boundaries or LineString intersections
│   ├── vein_labeler.py        # Topology-based vein identity assignment
│   ├── vein_map.py            # Static Drosophila vein topology rules + colors
│   ├── vein_skeleton.py       # Voronoi-based centerline extraction from vein mask
│   └── wing_geometry.py       # Outline, hinge, intervein partitioning, compartments
├── controllers/
│   ├── analysis_controller.py # Orchestrates full pipeline: TIFF + GeoJSON → overlays + CSV
│   └── measurement_controller.py  # All measurement computations
├── views/
│   ├── overlay_view.py        # Skeleton overlay + rainbow intervein overlay
│   └── results_view.py        # CSV export with all measurement columns
├── utils/
│   └── skeleton_utils.py      # Flip detection helpers
├── tests/
├── CLAUDE.md
└── requirements.txt
```

## Key Dependencies
- `shapely` — vector geometry (LineStrings, Polygons, intersections)
- `networkx` — graph operations
- `numpy`, `pandas`
- `matplotlib` — plotting (optional)
- `opencv-python` — image I/O and overlay rendering

## Input Formats
The pipeline accepts GeoJSON annotation styles:

1. **Intervein polygons + vein mask** (preferred): A `"intervein"` MultiPolygon (8 intervein spaces) plus a `"vein"` MultiPolygon (vein tissue mask). Veins are extracted via Voronoi partition of the vein mask seeded by intervein polygons (`vein_skeleton.py`). Eliminates triple-junction gaps.

2. **Intervein polygons only** (fallback): Just the `"intervein"` MultiPolygon. Veins are extracted as midlines between adjacent polygon boundaries (`vein_graph.py`). May have gaps at triple junctions.

3. **Vein LineStrings** (future): Individual LineString features with `classification.name = "vein"`, plus optional `"posterior outline"` and `"wing outline"` segments.

## Core Data Structures

### ParsedAnnotations (models/geojson_parser.py)
```python
@dataclass
class ParsedAnnotations:
    veins: list[ParsedVein]           # from LineString "vein" features
    posterior_segments: list[ParsedOutline]
    wing_outline_segments: list[ParsedOutline]
    intervein_polygons: list[Polygon]  # from (Multi)Polygon "intervein" features
    vein_polygons: list[Polygon]      # from (Multi)Polygon "vein" features (tissue mask)
```

### VeinAssignment (models/vein_labeler.py)
```python
@dataclass
class VeinAssignment:
    vein_id: str                  # "L1"–"L5", "ACV", "PCV", "costa"
    status: VeinStatus            # COMPLETE, FRAGMENTED, TRUNCATED, ABSENT
    edge_ids: list[int]
    confidence: float             # 0.0–1.0
    evidence: list[str]
    length_px: float
    gap_px: float | None
    length_um: float | None
    line: Optional[LineString]    # merged vein geometry
    endpoints: Optional[list]
```

### WingMeasurements (controllers/measurement_controller.py)
Per-vein lengths, crossvein distance, wing length/width, total area, intervein areas, compartment areas. All in pixels; microns if scale provided.

## Pipeline Order (analysis_controller.py)

### Polygon mode (current):
1. `geojson_parser.parse_geojson()` — extract intervein Polygons + vein mask Polygons
2a. **If vein mask present**: `vein_skeleton.extract_veins_from_mask()` — Voronoi partition → centerlines
2b. **Fallback**: `vein_graph.build_graph_from_polygons()` → `vein_labeler.assign_veins_from_polygons()`
3. `measurement_controller.compile_results()` — apply scale calibration
5. `wing_geometry.build_wing_outline()` — union of buffered polygons
6. `wing_geometry.detect_hinge_landmarks()` — find subcostal break + alula notch
7. `wing_geometry.remove_hinge()` — split along hinge line, keep distal blade
8. `wing_geometry.partition_intervein_spaces()` — clip polygons to wing blade
9. `wing_geometry.compute_compartments()` — split along L4 into anterior/posterior
10. `measurement_controller.compute_measurements()` — all areas and distances
11. `overlay_view.render_skeleton_overlay()` + `render_rainbow_overlay()` — output images
12. `results_view.export_csv()` — one row per wing with all measurements

### LineString mode (future):
1. Parse GeoJSON veins → `build_graph_from_veins()` → `assign_veins()` → overlays + CSV

## Intervein Space Naming (anterior → posterior)
| Region | Bounded by | Color |
|--------|-----------|-------|
| costal_cell | costa – L1 | yellow-green |
| marginal_cell | L1 – L2 | salmon |
| submarginal_cell | L2 – L3 | peach |
| 1st_basal_cell | L3 – L4 (proximal to ACV) | blue |
| discal_cell | L3 – L4 (distal to ACV) | green |
| 2nd_posterior_cell | L4 – L5 | cyan |
| 3rd_posterior_cell | posterior to L5 | purple |

## Vein Identification (polygon mode)
Veins are identified by which intervein regions they separate (defined in `VEIN_BOUNDARIES` in vein_map.py). The costa is extracted as the anterior margin of the marginal cell polygon.

## Scale Calibration
Optional. Passed as `microns_per_pixel: float | None` to `run_pipeline()`.
If None, output columns are pixels only; `_um` columns are NaN.

## CSV Output (results_view.py)
Per-vein: `{vein}_length_px`, `{vein}_status`, optional `{vein}_length_um`, `{vein}_gap_px`.
Wing-level: `crossvein_distance_px`, `wing_length_px`, `wing_width_px`, `total_wing_area_px2`.
Compartments: `anterior_compartment_area_px2`, `posterior_compartment_area_px2`.
Per-region: `{region}_area_px2` for all 7 intervein spaces.

## Conventions
- All image arrays are numpy (H, W, 3) uint8 BGR for OpenCV
- Graph node attributes: `{"x": float, "y": float, "degree": int}`
- Graph edge attributes: `{"edge_id": int, "length_px": float, "line": LineString, "poly_pair": tuple}`
- Do not hardcode file paths; use pathlib.Path throughout
- All public functions have type hints and a one-line docstring
