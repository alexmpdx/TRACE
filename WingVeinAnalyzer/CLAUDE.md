# WingVeinAnalyzer — Claude Code Project Instructions

## Project Purpose
A Python tool to identify and measure named veins (L1–L5, ACV, PCV) in *Drosophila* wing brightfield images using binary mask files produced by an existing pixel classifier (decision tree or ResNet50).

## Architecture
Strict MVC. No business logic in views. No I/O in models.

```
WingVeinAnalyzer/
├── models/
│   ├── vein_graph.py          # Skeletonization + graph construction (uses skan)
│   ├── vein_labeler.py        # Topology-based vein identity assignment
│   └── vein_map.py            # Static Drosophila vein topology rules
├── controllers/
│   ├── analysis_controller.py # Orchestrates full pipeline per image
│   └── measurement_controller.py
├── views/
│   ├── overlay_view.py        # Colored skeleton overlay for review
│   └── results_view.py        # CSV export + summary table
├── utils/
│   └── skeleton_utils.py      # Skeletonize, spur pruning, flip detection
├── tests/
├── CLAUDE.md
└── requirements.txt
```

## Key Dependencies
- `skan` — skeleton-to-graph conversion and branch length computation
- `scikit-image` — skeletonization, morphology
- `networkx` — graph operations
- `numpy`, `pandas`
- `matplotlib` — overlay view only
- `opencv-python` — image I/O and flip operations

## Core Data Structures

### VeinAssignment (models/vein_labeler.py)
```python
class VeinStatus(Enum):
    COMPLETE    = "complete"      # single connected edge, both ends at margin or junction
    FRAGMENTED  = "fragmented"    # 2+ disconnected edges share this vein ID; gap implies classifier error
    TRUNCATED   = "truncated"     # single edge with a free endpoint not on margin; likely biological
    ABSENT      = "absent"        # no edge assigned this vein ID

@dataclass
class VeinAssignment:
    vein_id: str                  # "L1"–"L5", "ACV", "PCV", "costa"
    status: VeinStatus
    edge_ids: list[int]           # skan edge indices; >1 only when FRAGMENTED
    confidence: float             # 0.0–1.0
    evidence: list[str]           # cues used, e.g. ["spatial_position", "acv_anchor"]
    length_px: float              # sum of fragment lengths; for TRUNCATED this is a minimum
    gap_px: float | None          # total gap distance for FRAGMENTED veins; None otherwise
    length_um: float | None       # None if no scale provided
```

### Distinguishing FRAGMENTED vs TRUNCATED (models/vein_labeler.py)
- **FRAGMENTED:** `len(edge_ids) > 1` AND the gap between nearest endpoints of consecutive fragments
  is less than a configurable `max_gap_px` threshold AND the fragments are spatially collinear
  (angle deviation < threshold). Gap distance is recorded in `gap_px`.
- **TRUNCATED:** single edge with a degree-1 endpoint that does not fall within `margin_tolerance_px`
  of the detected wing margin.
- If fragments are present but not collinear, treat as separate unassigned edges rather than
  forcing a fragmented assignment.

## Pipeline Order (analysis_controller.py)
1. Load image + mask
2. `skeleton_utils.detect_and_correct_flip()` — X-flip only, based on anterior vein density
3. `vein_graph.skeletonize_mask()` — skeletonize + prune spurs < threshold
4. `vein_graph.build_graph()` — returns skan Skeleton + networkx graph
5. `vein_labeler.assign_veins()` — returns list[VeinAssignment]
6. `measurement_controller.compile_results()` — aggregate lengths, apply scale
7. `results_view.export_csv()` — one row per wing

## Vein Identity Rules (vein_map.py)
Rules are encoded as a priority-ordered list of cues, not a rigid graph isomorphism.  
Cue types to implement:
- `spatial_anterior_posterior` — normalized X position of edge midpoint
- `margin_connectivity` — does edge terminate at wing margin?
- `crossvein_anchor` — is edge connected to a node also connected to ACV/PCV?
- `relative_length_rank` — rank among longitudinal edges (L3 = longest)
- `entry_angle` — angle at which edge meets margin

Assignments are made with partial evidence — a missing vein should not block labeling of others.

## Flip Detection Logic (skeleton_utils.py)
- Binarize skeleton, split at X midpoint
- Compare edge pixel count anterior half vs. posterior half
- Costa (anterior) consistently denser; if posterior > anterior, flip is detected
- Apply `np.fliplr()` to both image and mask

## Confidence Thresholds
- ≥ 0.75 → high confidence, include in default export
- 0.4–0.75 → low confidence, flagged in CSV, shown in overlay
- < 0.4 → unassigned, reported as null

## CSV Output Flags (results_view.py)
Each vein column gets a paired `_status` column: `complete`, `fragmented`, `truncated`, `absent`.
For `fragmented` veins also export `_gap_px` (and `_gap_um` if scale provided).
Overlay colors: green = complete, yellow = truncated, red = fragmented, grey = absent.

## Scale Calibration
Optional. Passed as `microns_per_pixel: float | None` to `analysis_controller`.  
If None, output length columns are pixels only.

## Testing Priorities
1. Flip detection on known-flipped vs. normal masks
2. Graph construction on synthetic T-junction and crossvein skeletons
3. Vein assignment on a manually labeled ground-truth set (build this early)
4. Fragmented vein handling — mask with artificial gaps

## Conventions
- All image arrays are numpy (H, W) uint8 for masks, (H, W, 3) uint8 for RGB
- Graph node attributes: `{"x": float, "y": float, "degree": int, "on_margin": bool}`
- Graph edge attributes: `{"length_px": float, "branch_id": int}`
- Do not hardcode file paths; use pathlib.Path throughout
- All public functions have type hints and a one-line docstring
