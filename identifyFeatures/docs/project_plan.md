# identifyFeatures — Landmark-Anchored Wing Vein Identification

## Status: Core Pipeline Complete (30/30 specimens, 10/10 veins)

Last updated: 2026-04-08

---

## Context

Landmark-anchored replacement for WingVeinAnalyzer. Vein identity flows outward from 6 reliable anatomical junction landmarks. Handles missing, partial, and ectopic veins in Drosophila melanogaster wings.

**Key design decisions established during implementation:**
- Voronoi partition of intervein polygons fails for vein centerlines — vein mask must be skeletonized directly
- RIDGE (Hessian-based) skeletonization produces the cleanest graphs; PATH_TRACE method was never needed
- Skeleton cleanup requires a 17-step pipeline with vein-width-adaptive thresholds
- Crossvein detection works best inside the vein tracer (not as a separate module) because it leverages the labeled longitudinals
- Costa detection requires its own module with margin-band analysis + SC proximal rejection
- Junction merging requires its own module with landmark protection and perpendicularity guards
- Single-pass stub removal (with local vein width from distance map) prevents cascading graph collapse
- Bridge pass 3 with relaxed facing angle (120°) recovers short stubs near junctions after cleanup
- Landmark snap radius uses 2× median vein width (not fixed 207px) to prevent snapping to unrelated structures

**Inputs per specimen:**
- Original image (TIFF/BMP, typically 5440x3648)
- Detection GeoJSON (`*_detections.geojson`): unnamed polygons with class: `"vein"` | `"intervein"` | `"hinge junk"`
- Landmarks GeoJSON (`*_landmarks.geojson`): 13 points — 5 reliable junctions, 4 soft distal endpoints, 4 unreliable crossvein landmarks

**Outputs (current):**
- Named vein centerlines (LineStrings) with VeinIdentification metadata

**Outputs (planned, not yet implemented):**
- Named vein tissue polygons
- Named intervein region polygons
- GeoJSON matching GT_naming format: `{classification: {name: "...", color: [...]}}`
- Measurements CSV
- Overlay images

**Features identified (current):** costa, L1, L2, L3, L4, L5, L6, Rs, ACV, PCV

**Features planned:** 8 intervein regions, ectopic veins, confidence scores

**Test data:** 30 specimens across 4 genotypes (CTRL ×9, BMP ×6, PknCG736 ×10, en-PknRNAi ×5), with GT_naming ground truth

---

## Project Structure

```
identifyFeatures/
  pyproject.toml
  docs/
    pipeline_reference.md            # Comprehensive pipeline documentation
    project_plan.md                  # This file
  identify_features/
    __init__.py                      # [TODO] Public API: identify_wing()
    cli.py                           # [TODO] CLI entry point
    config.py                        # [DONE] PipelineConfig dataclass + defaults

    models/
      __init__.py
      topology.py                    # [DONE] Static vein topology constants
      datatypes.py                   # [DONE] All dataclasses
      geojson_io.py                  # [DONE] Parse detection + landmark GeoJSON
      skeleton.py                    # [DONE] Vein mask → skeleton → graph (RIDGE method, 17-step pipeline)
      landmark_anchor.py             # [DONE] Snap landmarks to skeleton graph nodes
      junction_resolver.py           # [DONE] Merge longitudinals through crossvein junctions
      costa_detector.py              # [DONE] Costa detection via margin band
      vein_tracer.py                 # [DONE] Trace veins + crossvein detection (Phases 0-5)
      intervein_namer.py             # [TODO] Name intervein regions by adjacency
      trajectory_completer.py        # [TODO] Extrapolate partial veins, split merged regions
      confidence.py                  # [TODO] Multi-factor confidence scoring

    controllers/
      __init__.py
      pipeline.py                    # [TODO] Main orchestrator: identify_wing()

    views/
      __init__.py
      overlay.py                     # [TODO] Render skeleton + rainbow overlays
      geojson_export.py              # [TODO] Export named features as GeoJSON
      csv_export.py                  # [TODO] Export measurements CSV

    utils/
      __init__.py
      graph_utils.py                 # [DONE] NetworkX helpers
      geometry_utils.py              # [DONE] Shapely helpers
      image_utils.py                 # [DONE] Image loading, mask rasterization

  tests/
    test_evaluate.py                 # [TODO] IoU evaluation against GT_naming
```

---

## Pipeline Steps

### Step 1: Parse Inputs — DONE

**File:** `models/geojson_io.py`

- Load detection GeoJSON → separate features by class (vein, intervein, hinge junk)
- Load landmarks GeoJSON → 5 reliable + 4 soft + 4 unreliable landmarks
- Load image for dimensions
- Compute wing outline = `union(all polygons).buffer(20).buffer(-10)`

### Step 2: Build Skeleton Graph — DONE

**File:** `models/skeleton.py` (1,781 lines)

Skeletonizes the vein mask using Hessian-based ridge extraction, then runs a 15-step cleanup pipeline to produce a clean NetworkX graph. All thresholds scale with median vein width for resolution independence.

**Full pipeline (see `docs/pipeline_reference.md` for detailed pseudocode):**

1. Rasterize vein polygons to binary mask
2. Ridge skeletonization (Hessian eigenvalue analysis + NMS + junction gap filling)
3. Compute median vein width from distance map
4. Local-width-aware branch pruning (adaptive threshold: thin veins keep short branches)
5. Build raw graph from skeleton pixels (junction clustering via BFS)
6. Simplify (contract degree-2 nodes)
7. Merge nearby junction nodes (within 2× vein width)
8. Gap bridging pass 1 (5 conditions: combined length, adaptive gap, facing, strict on-axis, relaxed on-axis)
9. Remove redundant overlapping edges (70% overlap threshold)
10. Absorb tiny segments (1× vein width)
11. Merge close nodes (iterative, prefer higher degree)
12. Remove small isolated fragments (< 4× vein width)
13. Gap bridging pass 2 (more permissive: 50% gap fraction, 3.5× vein width combined length floor)
14. Final single-pass stub removal (3× vein width, no cascade)
15. Snap edge endpoints to node positions

### Step 3: Anchor Landmarks to Graph — DONE

**File:** `models/landmark_anchor.py`

- Junction landmarks (L1-Rs, L2-L3, L4-L5): prefer degree-3+ nodes within snap_radius
- Endpoint landmarks (SC, DTip): prefer degree-1 nodes
- Alula notch: margin reference only, skip graph modification
- Fallback: insert new node by splitting nearest edge at projection point

### Step 4: Identify Veins — DONE

**File:** `models/vein_tracer.py` (1,127 lines), `models/costa_detector.py`, `models/junction_resolver.py`

Six phases, each building on previous results:

- **Phase 0:** Merge longitudinals through crossvein junctions (junction_resolver.py)
  - Contracts most-collinear edge pairs at degree-3 nodes
  - Protects landmark nodes and already-labeled edges
  - Perpendicularity guard prevents merging at divergence junctions
- **Phase 1:** Detect costa via margin band (costa_detector.py)
  - Band = pixels within 2× vein width of wing outline
  - SC-AN hinge trim + subcostal break cut
  - ≥50% in-band threshold + proximal departure rejection at SC
- **Phase 2:** Label edges at landmark positions
  - L2-L3: simultaneous matching with L2.d and DTip soft landmarks
  - L4-L5: simultaneous matching with L4.d and L5.d
  - L1-Rs: direction toward SC (< 60° = L1, else Rs)
  - DTip → L3, SC → L1
- **Phase 2b:** Propagate labels through degree-2 nodes (with costa band guard)
- **Phase 2c:** Extend longitudinals to distal landmark points
- **Phase 2d:** Re-propagate after extension
- **Phase 2e:** Connect disconnected vein fragments via shortest unlabeled path
- **Phase 3:** Detect L6 (short posterior branch off L5, 0.5-1.5× Rs length, near L4-L5)
- **Phase 4:** Detect crossveins by endpoint connectivity (ACV: L3↔L4, PCV: L4↔L5)
- **Phase 4a:** Junction-based crossvein detection — BFS through unlabeled paths between degree-3+ junctions on longitudinals
- **Phase 4b:** Fallback crossvein detection using unreliable landmarks (tiered: ACV.p → ACV.a, PCV.a → PCV.p)
- **Phase 4c:** Post-crossvein degree-2 propagation (absorbs unlabeled stubs at crossvein endpoints)
- **Phase 5:** Build VeinIdentification objects with merged centerlines

**Result:** 30/30 specimens correctly identify all 10 canonical veins.

### Step 5: ~~Detect Crossveins~~ — DONE (merged into Step 4)

Originally planned as a separate `crossvein_detector.py`. Implemented as Phases 4 and 4b inside `vein_tracer.py` because crossvein detection depends on the labeled longitudinals produced by earlier phases. No separate module needed.

### Step 6: Name Intervein Regions — TODO

**Planned file:** `models/intervein_namer.py`

For each unnamed intervein polygon from the detection GeoJSON:
- Buffer each identified vein LineString slightly (~25px)
- Determine which veins are adjacent to each polygon boundary (intersection length ≥30px)
- Look up vein-set in `topology.REGION_EXPECTED_VEINS` → region name
- Disambiguate crossvein-split pairs (1st_basal vs 1st_posterior, discal vs 2nd_posterior) using `topology.REGION_DISAMBIGUATION` by proximal/distal position relative to ACV/PCV

**Note:** The previous WingVeinAnalyzer project has a working implementation of this logic that can be adapted. Key functions: `_build_poly_veins_spatial()`, Jaccard scoring, area-based disambiguation.

### Step 7: Complete Partial Veins & Split Merged Regions — TODO

**Planned file:** `models/trajectory_completer.py`

- Detect merged intervein regions: vein-set matches union of two adjacent expected regions
- Extend partial veins via spline extrapolation along terminal tangent direction
- Split merged polygons along extended vein trajectories using `shapely.ops.split`
- Extend partial longitudinals to wing outline for proper region bounding

**Note:** The previous WingVeinAnalyzer had `partition_by_vein_extension()` and `_clip_regions_by_extension()` that implemented vein-extension-based region splitting. This logic can be adapted.

### Step 8: Flag Ectopic Veins — TODO

After all canonical veins identified:
- Collect unassigned skeleton edges, merge connected components
- Filter short fragments (< 50px) as noise
- Long unassigned paths → flag as ectopic veins (EV1, EV2, ...)
- Record position, connectivity, orientation for each

### Step 9: Score Confidence — TODO

**Planned file:** `models/confidence.py`

Multi-factor scoring (0.0–1.0) for every identification:
- **Veins:** landmark anchor quality, trace continuity, expected length/orientation, endpoint destination
- **Crossveins:** connectivity to expected longitudinals, orientation, length, position
- **Regions:** all bounding veins identified, area within range, AP position correct

### Step 10: Assign Vein Tissue Polygons — TODO

Map skeleton-identified vein centerlines back to the original vein tissue polygons:
- For each vein polygon from detection GeoJSON, determine which named centerlines pass through it
- If a single vein polygon contains multiple named veins (likely — vein tissue is often one connected mass), partition using centerlines as guides
- Output named vein polygons alongside centerlines

---

## Remaining Infrastructure

### CLI (`cli.py`) — TODO

```
identify-features <detection_geojson> <landmarks_geojson> [image]
    --output-dir DIR              Output directory [default: ./output]
    --um-per-px FLOAT             Microns per pixel [default: 0.483]
    --skeleton-method METHOD      ridge|medial-axis|voronoi|boundary-smooth
    --snap-radius-um FLOAT        Landmark snap radius in µm [default: 100]
    --smooth-sigma FLOAT          Boundary smoothing sigma [default: 2.0]
    --batch                       Batch mode: args are directories
    --no-overlay                  Skip overlay image generation
    --verbose                     Increase logging
```

Entry point defined in pyproject.toml: `identify-features = "identify_features.cli:main"`

### Pipeline Controller (`controllers/pipeline.py`) — TODO

Orchestrates Steps 1–10 into a single `identify_wing()` function. This becomes the public API.

### Views — TODO

- `views/overlay.py` — Render named veins + regions on wing image
- `views/geojson_export.py` — Export in GT_naming format
- `views/csv_export.py` — Export measurements table

### Tests — TODO

- `tests/test_evaluate.py` — IoU evaluation against GT_naming ground truth
- Automated regression tests for 30/30 specimen suite

---

## File Matching (Batch Mode)

Stems across folders:
- `OGpics/{stem}.tif` or `.bmp`
- `geojsons/{stem}_detections.geojson`
- `LandmarkLocator_output/{stem}_landmarks.geojson`

Handle quirks: stems starting with `-`, spaces in filenames (e.g., `0001 _landmarks.geojson`).

---

## Data Structures (datatypes.py) — DONE

```python
class VeinStatus(Enum):
    IDENTIFIED = "identified"     # Traced from landmark
    INFERRED = "inferred"         # Identified by position/topology
    PARTIAL = "partial"           # Doesn't reach expected endpoint
    ABSENT = "absent"             # Expected but not found
    ECTOPIC = "ectopic"           # Unexpected extra vein

class VeinType(Enum):
    LONGITUDINAL = "longitudinal"
    CROSSVEIN = "crossvein"
    COSTA = "costa"
    RADIAL_SECTOR = "radial_sector"

@dataclass
class Landmark:
    name: str
    point: Point
    reliable: bool
    soft: bool = False            # Helpful hint, may be wrong in mutants
    snapped_node: Optional[int] = None
    snap_distance: float = 0.0

@dataclass
class VeinIdentification:
    vein_id: str                  # "L1", "ACV", "EV1", etc.
    vein_type: VeinType
    status: VeinStatus
    centerline: Optional[LineString]
    tissue_polygon: Optional[Polygon]   # TODO: not yet populated
    edge_ids: list[int]
    length_px: float = 0.0
    length_um: Optional[float] = None
    confidence: float = 0.0       # TODO: not yet scored
    evidence: list[str]

@dataclass
class InterveinRegion:
    name: str
    polygon: Optional[Polygon]
    bounding_veins: set[str]
    area_px2: float = 0.0
    confidence: float = 0.0
    status: str = "identified"

@dataclass
class WingResult:
    specimen_id: str
    veins: list[VeinIdentification]
    intervein_regions: list[InterveinRegion]
    landmarks: dict[str, Landmark]
    wing_outline: Optional[Polygon]
    warnings: list[str]
```

---

## Topology Constants (topology.py) — DONE

Single source of truth for wing vein biology:

| Constant | Content |
|----------|---------|
| `RELIABLE_LANDMARKS` | {subcostal break, alula notch, L1-Rs, L2-L3, L4-L5} |
| `SOFT_LANDMARKS` | {DTip, L2.d, L4.d, L5.d} |
| `UNRELIABLE_LANDMARKS` | {ACV.a, ACV.p, PCV.a, PCV.p} |
| `JUNCTION_TOPOLOGY` | Which veins meet at each landmark junction |
| `TRACE_RULES` | Start landmark + trace direction for each vein |
| `CROSSVEIN_CONNECTIONS` | ACV→(L3,L4), PCV→(L4,L5) |
| `REGION_EXPECTED_VEINS` | Which veins bound each of the 8 intervein regions |
| `REGION_DISAMBIGUATION` | Proximal/distal rules for 1st_basal/1st_posterior and discal/2nd_posterior |
| `VEIN_AP_ORDER` | Anterior-to-posterior ordering of all veins |
| `REGION_AP_ORDER` | Anterior-to-posterior ordering of 8 regions |
| `VEIN_COLORS` / `REGION_COLORS` | Display colors matching GT_naming format |

---

## Implementation Order (remaining work)

The core vein identification pipeline is complete. Remaining work is output-facing:

1. **`controllers/pipeline.py`** — Orchestrate Steps 1–4 into `identify_wing()`. Wire up the existing model functions into a clean public API.

2. **`models/intervein_namer.py`** — Name the 8 intervein regions using vein adjacency. Adapt logic from WingVeinAnalyzer's `_build_poly_veins_spatial()`.

3. **`views/geojson_export.py`** — Export named veins + regions as GeoJSON in GT_naming format.

4. **`views/overlay.py`** — Render named veins + regions on wing image with color coding.

5. **`cli.py`** — Wire up the CLI entry point. Support single-specimen and batch modes.

6. **`tests/test_evaluate.py`** — IoU evaluation against GT_naming ground truth. Automated regression suite.

7. **`models/trajectory_completer.py`** — Extend partial veins, split merged regions. Lower priority since 30/30 already works.

8. **`models/confidence.py`** — Multi-factor confidence scoring. Lower priority.

9. **Ectopic vein flagging** — Flag long unassigned edges as ectopic. Lower priority for wildtype, important for mutants.

10. **Vein tissue polygon assignment** — Partition vein tissue polygons by named centerlines. Lower priority.

---

## Verification

1. Run on all 30 specimens — expect 10/10 veins per specimen (currently passing)
2. Compare output GeoJSON against GT_naming via IoU per feature
3. Visually inspect overlays for correct vein tracing
4. Run on mutant specimens (en-PknRNAi) — verify graceful handling of missing/ectopic veins
