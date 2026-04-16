# identifyFeatures — Landmark-Anchored Wing Vein Identification

## Status: Core Pipeline Complete (30/30 specimens, 10/10 veins)

Last updated: 2026-04-16

**Recent changes:**
- `junction_merge_vein_widths` default → 0.0 (step skipped); merging was collapsing distinct junctions and leaving crossed edges with no shared node (DTip under distance-map).
- `enable_small_fragment_removal` toggle added for skeleton steps 11 / 14; distance-map preset disables it.
- Final skeleton step always keeps only the largest connected component — orphan fragments are unreachable from landmark anchors.

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
- Named vein tissue polygons (buffered centerlines clipped to wing outline)
- Named intervein region polygons (from splitter + namer)
- Ectopic veins (EV1, EV2, ...) with centerlines and tissue polygons
- 7 named intervein regions (30/30 specimens, 7/7 regions)

**Outputs (planned, not yet implemented):**
- (none — all output formats implemented)

**Outputs (done):**
- GeoJSON export in GT_naming format (`views/geojson_export.py`): FeatureCollection with `classification: {name, color}` and area/length measurements

**Features identified (current):** costa, L1, L2, L3, L4, L5, L6, Rs, ACV, PCV, EV1-N, 7 intervein regions

**Features planned:** confidence scores

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
    __init__.py
    cli.py                           # [DONE] CLI entry point (single + batch)
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
      vein_tracer.py                 # [DONE] Trace veins + crossvein + ectopic labeling (Phases 0-5)
      wing_axis.py                   # [DONE] Derive proximal/distal axis from alula notch → DTip
      intervein_splitter.py          # [DONE] Split classifier-merged intervein polygons via h-maxima seed detection + watershed
      intervein_namer.py             # [DONE] Name intervein regions by adjacency (with PD tie-break)
      trajectory_completer.py        # [TODO] Extrapolate partial veins, split merged regions
      confidence.py                  # [TODO] Multi-factor confidence scoring

    controllers/
      __init__.py
      pipeline.py                    # [DONE] Main orchestrator: identify_wing()

    views/
      __init__.py
      overlay.py                     # [DONE] Render veins + regions color overlay
      geojson_export.py              # [DONE] Export named features as GeoJSON
      csv_export.py                  # [DONE] Export measurements CSV

    utils/
      __init__.py
      graph_utils.py                 # [DONE] NetworkX helpers
      geometry_utils.py              # [DONE] Shapely helpers
      image_utils.py                 # [DONE] Image loading, mask rasterization

  tests/
    test_evaluate.py                 # [DONE] IoU evaluation against GT_naming (in project root)
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

### Step 5.5: Split Merged Intervein Polygons — DONE

**File:** `models/intervein_splitter.py`, entry point `split_merged_intervein_polygons()`

Preprocessing pass that sits between vein tracing and intervein naming. Addresses classifier-merged intervein polygons where a crossvein is short, interrupted, or missed. Uses h-maxima peak detection on the Euclidean distance transform to find adaptive seeds, then watershed for competitive dilation. Pure raster implementation — distance transforms are O(H·W) so the operations are cheap even on 5440×3648 images.

- **Build barrier mask**: inset wing outline by `intervein_split_wing_buffer_vw × median_vein_width_px`, then add buffered canonical vein centerlines (`intervein_split_vein_barrier_vw × median_vein_width_px`). L6 and `EV*` ectopic veins are explicitly excluded from the barrier per spec — they are not region dividers.
- **Detect seeds via h-maxima**: for each intervein polygon, compute the EDT (inscribed-radius landscape), smooth with `gaussian_filter(sigma=median_vein_width_px)` to eliminate plateau ripples, then find peaks via `h_maxima(edt_smooth, h)` where `h = median_vein_width_px × intervein_split_h_vw` (default 2.0). Peaks that rise at least `h` above their nearest saddle become seeds. Adaptive: uniform strips get 1 seed, fused polygons with a bottleneck get 2+ seeds at exactly the right locations.
- **Competitive dilation** via `skimage.segmentation.watershed` using the seed labels as markers, with the barrier mask as the constraint. Labels flood outward from their seeds until they meet each other or a barrier.
- **Reseed lost polygons**: any input polygon whose EDT landscape is entirely flat (no h-maxima peaks) but whose original footprint exceeds `intervein_split_reseed_min_area_um2` (default 10,000 µm²) gets a new single-pixel seed at its interior, and the watershed runs again so thin regions like 1st basal aren't silently absorbed by neighbors.
- **Raster → polygons** via `cv2.findContours` for each label. Output is a plain `list[Polygon]` that shadows the original `intervein_polys` for the namer.

The existing merge-detection (`_check_merged` in `intervein_namer.py`) stays in place as a safety net for merges the morphological pass misses.

### Step 6: Name Intervein Regions — DONE

**File:** `models/intervein_namer.py`, entry point `name_intervein_regions()`

For each intervein polygon from the detection GeoJSON (after the splitter has pre-processed them):
- Buffer each identified vein LineString and collect the set of veins adjacent to the polygon
- Match the adjacent set against `topology.REGION_EXPECTED_VEINS`, keeping the most specific candidates
- If multiple candidates tie (currently only discal vs 2nd posterior, which share `{L4, L5, PCV}`), defer the polygon to a second pass
- **Proximal/distal tie resolver**: `_resolve_pd_ties()` groups deferred polygons by their tied set, looks it up in `topology.REGION_PD_PAIRS`, sorts members by `WingAxis.project(centroid)`, and assigns the proximal name to the smallest-PD polygon and the distal name to the largest-PD polygon
- `WingAxis` is computed once per specimen by `models/wing_axis.py:compute_wing_axis()` from the anchored alula notch → DTip landmarks; if either landmark is missing the resolver degrades gracefully and deferred polygons fall through to `_check_merged()`
- `_detect_absorbed_merges()` runs a two-phase post-pass: Phase A splits duplicate-name polygons via forbidden-adjacency rules (e.g. L6-adjacent polygon renamed from "discal" to "3rd posterior"); Phase B falls back to legacy append-style merge naming (`"A + B"`) for anything still missing. `_absorb_ectopic_fragments()` then sweeps stray fragments into their best-matching neighbor
- **N-way merge detection**: `_check_merged()` delegates to `_enumerate_merge_candidates()`, which brute-forces over all connected subsets of the region adjacency graph (derived once from `topology.VEIN_BOUNDARIES` into `_REGION_ADJACENCY` / `_REGION_EDGE_SEPARATOR`). A candidate is a connected subset whose merged expected vein set — the union of per-region expected veins minus the internal separators of the subset — is a subset of the polygon's detected veins. Scoring prefers max `len(merged_expected)` (specificity), then smallest subset size (avoid overclaiming), then AP-ordered region tuple for determinism. This handles 3+ region merges like `marginal + submarginal + 1st posterior` when both L2 and L3 are absent, which the old pair-only detector could not see. `PipelineConfig.max_merge_size` caps the search depth (default `None` = no cap)

### Step 7: Complete Partial Veins & Split Merged Regions — TODO

**Planned file:** `models/trajectory_completer.py`

- Detect merged intervein regions: vein-set matches union of two adjacent expected regions
- Extend partial veins via spline extrapolation along terminal tangent direction
- Split merged polygons along extended vein trajectories using `shapely.ops.split`
- Extend partial longitudinals to wing outline for proper region bounding

**Note:** The previous WingVeinAnalyzer had `partition_by_vein_extension()` and `_clip_regions_by_extension()` that implemented vein-extension-based region splitting. This logic can be adapted.

### Step 8: Flag Ectopic Veins — DONE

**File:** `models/vein_tracer.py`, Phase 4d

After all canonical veins identified, every still-unlabeled edge is promoted to an ectopic vein label. Each connected component of unlabeled edges becomes one EV<N> (`EV1`, `EV2`, ...) materialized as a `VeinIdentification` with `status=ECTOPIC`. Short fragments below `ectopic_min_length_um` (default 25µm, or `ectopic_min_length_vw` × vein width when `um_per_px` is unavailable) are filtered as noise.

### Step 9: Score Confidence — TODO

**Planned file:** `models/confidence.py`

Multi-factor scoring (0.0–1.0) for every identification:
- **Veins:** landmark anchor quality, trace continuity, expected length/orientation, endpoint destination
- **Crossveins:** connectivity to expected longitudinals, orientation, length, position
- **Regions:** all bounding veins identified, area within range, AP position correct

### Step 10: Assign Vein Tissue Polygons — DONE

**File:** `models/intervein_splitter.py`, function `assign_vein_tissue_polygons()`

Buffers each vein's centerline by `median_vein_width_px × intervein_split_vein_barrier_vw` (same radius as the intervein splitter barrier mask), clips to the wing outline, and assigns to `VeinIdentification.tissue_polygon`. All veins with centerlines get tissue polygons, including L6 and ectopic veins.

---

## Remaining Infrastructure

### CLI (`cli.py`) — DONE

```
identify-features <detection_geojson> <landmarks_geojson> [image]
    --output-dir DIR, -o          Output directory [default: ./output]
    --um-per-px FLOAT             Microns per pixel [default: 0.483]
    --batch                       Batch mode: args are directories
    --workers N                   Parallel workers for batch mode
    --verbose, -v                 Increase logging
```

Entry point defined in pyproject.toml: `identify-features = "identify_features.cli:main"`

### Pipeline Controller (`controllers/pipeline.py`) — DONE

`identify_wing()` orchestrates Steps 1–6 (parse → skeleton → anchor → trace → tissue assign → split → name) into a single function returning `WingResult`.

### Views — PARTIAL

- `views/geojson_export.py` — DONE: Export in GT_naming format
- `views/overlay.py` — DONE: Render veins + regions color overlay (used by CLI `--overlay` and test scripts)
- `views/csv_export.py` — DONE: Export measurements CSV. Single mode: long format (one row per feature). Batch mode: wide format (one row per specimen). Wing-level measurements: wing area, wing length (L1-Rs↔DTip), crossvein distance (ACV.p↔PCV.a), anterior/posterior compartment areas (split along L4 axis).

### Tests — PARTIAL

- `test_evaluate.py` — DONE: IoU evaluation against GT_naming ground truth (25 specimens, mean region IoU 0.91, mean vein IoU 0.61, 98.8% detection rate at threshold 0.5)
- Automated regression tests for 30/30 specimen suite — partially done via `test_regions_batch.py`

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
| `CROSSVEIN_CONNECTIONS` | ACV→(L3,L4), PCV→(L4,L5) |
| `REGION_EXPECTED_VEINS` | Which veins bound each of the 8 intervein regions |
| `REGION_PD_PAIRS` | Proximal→distal tie-break pairs (e.g. discal/2nd posterior) |
| `VEIN_AP_ORDER` | Anterior-to-posterior ordering of all veins |
| `REGION_AP_ORDER` | Anterior-to-posterior ordering of 8 regions |
| `VEIN_COLORS` / `REGION_COLORS` | Display colors matching GT_naming format |

---

## Implementation Order (remaining work)

The core pipeline and evaluation are complete. Remaining work:

1. **`models/trajectory_completer.py`** — Extend partial veins, split merged regions. Lower priority since 30/30 works.
2. **`models/confidence.py`** — Multi-factor confidence scoring. Lower priority.

**Done (this round):**
- ~~`controllers/pipeline.py`~~ — `identify_wing()` orchestrator
- ~~`cli.py`~~ — CLI entry point (single + batch mode, parallel processing, `--overlay`)
- ~~`views/geojson_export.py`~~ — GeoJSON export in GT_naming format
- ~~`views/overlay.py`~~ — Render veins + regions color overlay
- ~~`views/csv_export.py`~~ — Measurements CSV (long format single, wide format batch, wing-level measurements)
- ~~`test_evaluate.py`~~ — IoU evaluation (mean region IoU 0.91, mean vein IoU 0.61, 98.8% detection)
- ~~Ectopic vein flagging~~ — Phase 4d in vein_tracer
- ~~Vein tissue polygon assignment~~ — `assign_vein_tissue_polygons()` in intervein_splitter

---

## Verification

1. Run on all 30 specimens — expect 10/10 veins, 7/7 regions per specimen (currently passing)
2. IoU evaluation: `python test_evaluate.py` — 25 specimens, mean region IoU 0.91, 98.8% detection rate
3. Visually inspect overlays for correct vein tracing
4. Run on mutant specimens (en-PknRNAi) — verify graceful handling of missing/ectopic veins
5. Known issue: specimen `-CTRL_PknRNAi_108870_0003` has 2nd/3rd posterior swapped vs GT (cross-IoU 0.96)
