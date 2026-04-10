# identifyFeatures Pipeline Reference

A step-by-step guide to the landmark-anchored Drosophila wing vein identification pipeline. This document describes what each step does, why it exists, and how the logic works.

---

## Table of Contents

1. [Overview](#1-overview)
2. [Input Parsing](#2-input-parsing)
3. [Skeleton Building](#3-skeleton-building) (17 steps)
4. [Landmark Anchoring](#4-landmark-anchoring)
5. [Vein Labeling](#5-vein-labeling) (Phases 0-5, including 4a-4d)
5.5. [Intervein Polygon Splitting](#55-intervein-polygon-splitting-morphological-open-under-constraint)
5.6. [Intervein Region Naming](#56-intervein-region-naming)
6. [Output](#6-output)
7. [Parameter Reference](#7-parameter-reference)

---

## 1. Overview

### What the pipeline does

Given a pixel-classifier segmentation of a Drosophila wing (vein/intervein GeoJSON polygons) and deep-learning landmark points, this pipeline identifies and names individual veins: **costa, L1, L2, L3, L4, L5, L6, Rs, ACV, PCV**.

### Design principle

**Landmarks are primary, not supplementary.** Vein identity flows outward from 6 reliable landmark junctions. This inverts the approach of guessing veins by spatial priors alone.

### High-level flow

```
Detection GeoJSON ─┐
                    ├─→ [Input Parsing] ─→ vein polygons, intervein polygons, wing outline
Landmarks GeoJSON ──┘
                                              │
                                              ▼
                            [Skeleton Building] ─→ NetworkX graph (nodes + edges with LineStrings)
                                              │
                                              ▼
                            [Landmark Anchoring] ─→ landmarks snapped to graph nodes
                                              │
                                              ▼
                              [Vein Labeling] ─→ edge_labels: {(u,v): "L3", ...}
                                              │
                                              ▼
                      [Tissue Polygon Assignment] ─→ buffered centerlines clipped to wing
                                              │
                                              ▼
                      [Intervein Polygon Splitting] ─→ split fused regions via watershed
                                              │
                                              ▼
                      [Intervein Region Naming] ─→ 7 named regions by vein adjacency
                                              │
                                              ▼
                                [Output] ─→ WingResult (veins + regions + GeoJSON)
```

**Entry points:**
- **CLI**: `identify-features <det.geojson> <lm.geojson> [image] -o output/`
- **Batch**: `identify-features --batch <det_dir> <lm_dir> [img_dir] -o output/`
- **Python API**: `from identify_features.controllers.pipeline import identify_wing`

### The 10 canonical veins

| Vein | Type | Description |
|------|------|-------------|
| costa | Costa | Anterior margin vein, runs along wing edge |
| L1 | Longitudinal | Short, between costal cell and marginal cell |
| Rs | Radial sector | Short proximal stem connecting L1-Rs to L2-L3 junction |
| L2 | Longitudinal | Anterior longitudinal, branches from L2-L3 junction |
| L3 | Longitudinal | Central longitudinal, runs from L2-L3 toward wing tip (DTip) |
| ACV | Crossvein | Anterior crossvein, connects L3 to L4 |
| L4 | Longitudinal | Central-posterior longitudinal, from L4-L5 junction |
| PCV | Crossvein | Posterior crossvein, connects L4 to L5 |
| L5 | Longitudinal | Posterior longitudinal, from L4-L5 junction |
| L6 | Longitudinal | Short posterior branch off L5 near L4-L5 |

### The 6 reliable landmarks

| Landmark | Type | Location |
|----------|------|----------|
| subcostal break (SC) | Endpoint | Where L1 meets the costa at the wing margin |
| alula notch (AN) | Margin reference | Posterior notch at the wing base (hinge trim reference) |
| L1-Rs | Junction (deg-3) | Where L1 meets Rs |
| L2-L3 | Junction (deg-3) | Where Rs splits into L2 and L3 |
| L4-L5 | Junction (deg-3) | Where L4 and L5 diverge |
| DTip | Endpoint | Distal tip of L3 at the wing margin |

### The 4 soft landmarks

Soft landmarks are helpful hints but may be wrong in mutants with premature vein termination. They are never required.

| Landmark | Location |
|----------|----------|
| L2.d | Distal endpoint of L2 |
| L4.d | Distal endpoint of L4 |
| L5.d | Distal endpoint of L5 |

---

## 2. Input Parsing

**Source**: `models/geojson_io.py`

### 2.1 Load detection GeoJSON

```
load_detection_geojson(path) → (vein_polygons, intervein_polygons)
```

Reads a GeoJSON file produced by the pixel classifier. Each feature has `properties.class`:
- `"vein"` → added to vein polygon list
- `"intervein"` → added to intervein polygon list
- `"hinge junk"`, `"wing"` → discarded

**Why**: The pixel classifier segments the wing into vein tissue (thick, dark polygons) and intervein tissue (lighter, between veins). We only use the vein polygons for skeletonization; intervein polygons contribute to the wing outline.

### 2.2 Load landmarks GeoJSON

```
load_landmarks_geojson(path) → dict[name, Landmark]
```

Reads landmark points from the deep-learning detector. Each landmark gets a `reliable` flag based on whether its name appears in `RELIABLE_LANDMARKS` or `SOFT_LANDMARKS`. Unreliable landmarks (ACV.a, ACV.p, PCV.a, PCV.p) are loaded but marked unreliable.

### 2.3 Compute wing outline

```
wing_outline = union(all_polygons).buffer(20).buffer(-10)
```

**What**: Union all vein + intervein polygons, then expand by 20px and contract by 10px.

**Why**: The double-buffer smooths gaps between adjacent polygons while keeping a tight outline. The net +10px expansion ensures the outline covers the wing margin where costa runs. If the result is a MultiPolygon, we keep only the largest piece.

---

## 3. Skeleton Building

**Source**: `models/skeleton.py`, function `build_skeleton_graph()`

This is the most complex part of the pipeline: 17 steps that transform vein polygons into a clean NetworkX graph. Every distance threshold in this section scales with the **median vein width**, which is computed after skeletonization. This makes the pipeline resolution-independent.

### 3.1 Rasterize vein polygons to binary mask

```python
vein_mask = rasterize_polygons(vein_polygons, image_shape)
# Result: uint8 array, 255 where vein tissue exists, 0 elsewhere
```

**What**: Converts Shapely polygons to a pixel mask using OpenCV `fillPoly`.

**Why**: All subsequent image processing (skeletonization, distance transform) operates on pixel arrays, not vector geometry.

### 3.2 Skeletonize (RIDGE method)

```python
skel, distance_map = _skeletonize_ridge(vein_mask, sigma=2.0)
```

**What**: Extracts 1-pixel-wide centerlines from the vein mask using Hessian-based ridge detection.

**Why RIDGE?** It produces inherently clean centerlines with fewer spurious branches than morphological thinning. The Hessian captures the "ridge" of the distance field — the center of each vein.

**Algorithm**:
```
1. distance_map = EDT(vein_mask)              # distance from nearest boundary at each pixel
2. Hxx, Hxy, Hyy = hessian(distance_map, sigma)  # second derivatives with Gaussian smoothing
3. lambda1, lambda2 = eigenvalues(H)          # lambda2 is more negative at ridges
4. cross_direction = eigenvector(lambda2)      # perpendicular to the ridge
5. For each pixel:
     if distance[r,c] >= distance[neighbors along cross_direction]  # local maximum
     AND lambda2[r,c] < -0.05:                                      # strong negative curvature
       mark as ridge pixel
6. Fill junction gaps (where Hessian NMS fails at vein intersections)
7. Thin to 1px with morphological skeletonize
```

**Junction gap filling** (`_fill_ridge_junction_gaps`): At vein intersections, the Hessian's non-maximum suppression creates gaps because the ridge is ambiguous. We detect junction centers as local maxima of the distance field, then dilate the ridge mask into these zones to reconnect fragments.

**Parameters**:
- `sigma = 2.0` — Hessian smoothing scale. Higher values capture thicker veins but may merge close parallel veins.
- `ridge_threshold = -0.05` — Lambda2 cutoff. More negative = stricter (fewer ridge pixels).

### 3.3 Compute median vein width

```python
median_vein_width = 2.0 * median(distance_map[skeleton_pixels])
```

**What**: At each skeleton pixel, the distance map gives the half-width (distance to nearest vein boundary). The median of these values, doubled, gives the typical full vein width.

**Why**: This becomes the universal scaling reference for all subsequent cleanup thresholds. A wing imaged at higher resolution has wider veins in pixels, so all thresholds adapt automatically.

**Typical value**: ~23px for the standard test data (0.483 µm/px).

### 3.4 Local-width-aware branch pruning

```python
skel = _prune_branches(skel, min_length=median_vein_width * 2.0, distance_map=distance_map)
```

**What**: Iteratively removes short terminal branches from the skeleton.

**Why**: Ridge extraction produces many small spurious branches at vein edges and junctions. These need to be removed, but using a single global threshold would over-prune thin distal veins while under-pruning thick proximal veins.

**Algorithm**:
```
repeat until no branches removed:
    for each endpoint pixel (degree-1):
        trace branch from endpoint to nearest junction
        local_half_width = distance_map[endpoint]
        adaptive_threshold = min(local_half_width, prune_min_length)
        if branch_length < adaptive_threshold:
            remove all pixels in this branch
```

**Key insight**: The adaptive threshold uses the **local vein width** at the endpoint. At thin veins (distal wing), the threshold shrinks, preserving short but real branches. At thick veins (proximal hinge), the threshold stays at the cap, aggressively removing noise spurs.

**Parameters**:
- `prune_min_length_vein_widths = 2.0` — Cap expressed as multiple of median vein width.

### 3.5 Build raw graph from skeleton

```python
graph = _skeleton_to_graph(skel)
```

**What**: Converts the pixel skeleton into a NetworkX graph.

**Algorithm**:
```
1. Compute 8-connected neighbor count for each skeleton pixel
2. Pixels with neighbors != 2 are junction/endpoint pixels
3. Cluster adjacent junction pixels via BFS → each cluster = one node
   Node position = median of cluster pixel coordinates
4. Trace edges between adjacent node clusters:
   For each junction pixel, follow skeleton path until reaching another node
   Edge geometry = LineString of traced pixel coordinates
5. Add direct edges between 8-connected junction clusters
```

**Output**: Graph with `{x, y}` node attributes and `{edge_id, line, length_px, pixel_count}` edge attributes.

**Why BFS clustering**: At real vein junctions, multiple pixels are detected as "junction" (degree != 2). Clustering them into a single node prevents explosion of tiny edges at junctions.

### 3.6 Simplify graph (contract degree-2 nodes)

```python
graph = _simplify_graph(graph)
```

**What**: Merges each degree-2 node's two edges into one, removing the pass-through node.

**Why**: Degree-2 nodes are not topologically meaningful — the vein simply passes through. Contracting them reduces graph complexity and creates longer, more meaningful edges.

**Algorithm**:
```
repeat until no degree-2 nodes:
    for each degree-2 node n with neighbors n1, n2:
        merged_line = concatenate(line(n1,n), line(n,n2))
        remove node n and both edges
        add edge(n1, n2, line=merged_line)
```

### 3.7 Merge nearby junction nodes

```python
_merge_junction_nodes(graph, min_dist=median_vein_width * 2.0)
graph = _simplify_graph(graph)
```

**What**: Merges degree-2/3 nodes that are closer than `min_dist` to each other.

**Why**: The skeleton often produces clusters of closely-spaced junctions where veins meet at complex intersections. These near-duplicate junctions cause problems for bridging (creates many tiny stubs) and labeling (ambiguous which edge belongs to which vein). Merging them creates clean, well-separated junctions.

**Algorithm**:
```
collect all (node_a, node_b) pairs where both are deg-2 or deg-3
    and distance(node_a, node_b) < min_dist
sort by distance (closest first)
for each pair:
    keep = node with higher degree (or median position if equal)
    drop = other node
    remove direct edge between them (if any)
    transfer all of drop's edges to keep, snapping LineString endpoints
    remove drop from graph
```

**LineString snapping**: When transferring edges, the first/last coordinate of each LineString is replaced with the kept node's position. This prevents geometric discontinuities.

**Parameters**:
- `junction_merge_vein_widths = 2.0` — Merge radius as multiple of median vein width.

### 3.8 Gap bridging (first pass)

```python
graph = _bridge_and_simplify(graph, ...)
```

**What**: Connects nearby endpoints that represent gaps in the skeleton — places where a vein was interrupted by noise, thin tissue, or imaging artifacts.

**Why**: The ridge skeleton may fragment at faint vein regions, creating two endpoints where there should be a continuous path. Bridging reconnects these.

**Algorithm overview**:
```
repeat up to 10 times:
    endpoints = all degree-1 nodes
    for each endpoint, compute direction = _full_edge_direction(endpoint)
    sort endpoints by edge_length descending (bridge longest pairs first)
    for each pair (ep1, ep2):
        check all 5 conditions (see below)
        if all pass: score = dist + strict_angle + relaxed_angle * 0.5
    for each endpoint, pick best-scoring partner
    add bridge LineString between matched pairs
    simplify graph (contract new degree-2 nodes)
    if no bridges added: break
```

**The 5 bridging conditions**:

1. **Minimum combined length**: `ep1.edge_len + ep2.edge_len >= min_combined_length`
   - *Why*: Prevents bridging two tiny stubs together (likely noise, not a real vein gap).
   - *Default*: 207px (~100µm)

2. **Adaptive gap distance**: `distance(ep1, ep2) <= max(min_gap, min(max_gap, fraction * longer_edge))`
   - *Why*: Gap tolerance scales with edge length. A long vein can tolerate a larger gap than a short stub. The adaptive formula prevents bridging across large distances while allowing reasonable gaps.
   - *Default*: `fraction = 0.15`, `max_gap = 414px` (~200µm)

3. **Facing angle**: `angle_between(direction1, direction2) >= 150°`
   - *Why*: Two endpoints that are continuations of the same vein must point toward each other (near 180°). If they point the same direction (0°) or at right angles (90°), they're not the same vein.

4. **Strict on-axis angle** (longer edge): `angle_between(longer_direction, bridge_axis) <= 45°`
   - *Why*: The longer edge (more reliable direction) must point roughly along the line connecting the two endpoints. This prevents bridging edges that point away from each other but happen to be close.

5. **Relaxed on-axis angle** (shorter edge): `angle <= min(45°, 45° * (1 + length_ratio * 0.1))`
   - *Why*: The shorter edge gets a relaxed threshold because short edges have less reliable direction estimates. The threshold scales with the length ratio — a very short edge paired with a very long one gets more tolerance.

**Direction computation** (`_full_edge_direction`):
```
sample_length = min(direction_window, edge_length * 0.8)
pt_a = edge.interpolate(edge.length - sample_length)
pt_b = edge endpoint
direction = normalize(pt_b - pt_a)
```
Samples the last `direction_window` pixels of the edge (up to 80% of total length) to compute the direction the vein was heading when it terminated.

**Parameters (first pass)**:
| Parameter | Default | Description |
|-----------|---------|-------------|
| `bridge_max_gap_um` | 200µm (414px) | Absolute max gap |
| `bridge_gap_fraction` | 0.15 | Gap as fraction of longer edge |
| `bridge_min_combined_length_um` | 100µm (207px) | Min total edge length |
| `bridge_min_facing_angle` | 150° | Must face each other |
| `bridge_on_axis_max_angle` | 45° | Strict angle for longer edge |
| `bridge_on_axis_relaxed_cap` | 45° | Max relaxed angle for shorter edge |

### 3.9 Remove redundant overlapping edges

```python
_remove_redundant_edges(graph)
graph = _simplify_graph(graph)
```

**What**: Removes shorter edges whose geometry overlaps significantly with a longer edge.

**Why**: The skeleton can produce parallel duplicate edges that trace the same vein. These create false junctions and confuse labeling.

**Algorithm**:
```
sort edges by length descending
for each edge (shortest to longest):
    buffer the edge's LineString by 15px
    for each longer (already-kept) edge:
        compute overlap = length of shorter edge within longer's buffer
        if overlap >= 0.70 * shorter_edge_length:
            mark shorter edge as redundant
remove all redundant edges
```

**Additional guard**: Both edges must run roughly parallel (departure directions within 45°). This prevents removing an edge that crosses a longer one at an angle (e.g., a crossvein crossing a longitudinal).

### 3.10 Absorb tiny segments

```python
_absorb_tiny_segments(graph, min_length=median_vein_width)
graph = _simplify_graph(graph)
```

**What**: Removes dead-end stubs shorter than 1x median vein width at junctions.

**Why**: After redundant edge removal, some tiny stubs remain at junctions. These are residual skeleton artifacts, not real vein features. The threshold (1x vein width) is deliberately below the pruning threshold (2x) to avoid cascading collapse.

**Algorithm**:
```
for each edge with length < min_length:
    if one end is degree-1 and other end is degree-3+:
        remove the degree-1 node and this edge
```

### 3.11 Merge close nodes

```python
_merge_close_nodes(graph, min_dist=median_vein_width)
graph = _simplify_graph(graph)
```

**What**: Iteratively merges any pair of nodes closer than `min_dist`.

**Why**: After absorption and simplification, some nodes end up very close together (within one vein width). These represent the same junction point and should be collapsed.

**Algorithm**:
```
repeat until no merges:
    for each node pair within min_dist:
        keep the higher-degree node (more connected = more likely real)
        transfer lower-degree node's edges to keeper
        remove lower-degree node
```

### 3.12 Remove small isolated fragments

```python
_remove_small_fragments(graph, min_length=median_vein_width * 4)
```

**What**: Removes connected components that have no degree-3+ junctions and total edge length less than 4x median vein width.

**Why**: After all cleanup, some tiny disconnected fragments remain — short isolated edges that aren't connected to the main vein network. These are noise, not veins.

**Algorithm**:
```
for each connected component of the graph:
    if component has no degree-3+ node (i.e., it's a simple chain):
        total_length = sum of all edge lengths in component
        if total_length < 4 * median_vein_width:
            remove entire component
also remove any isolated degree-0 nodes
```

### 3.13 Gap bridging (second pass)

```python
graph = _bridge_and_simplify(graph, ...)  # with bridge2_* parameters
```

**What**: Same algorithm as step 3.8, but with **more permissive parameters**.

**Why**: The cleanup steps (3.9–3.12) may have exposed new bridgeable endpoints by removing redundant edges or fragments. The second pass catches these with a larger gap tolerance.

**Key differences from first pass**:

| Parameter | Pass 1 | Pass 2 | Why |
|-----------|--------|--------|-----|
| `gap_fraction` | 0.15 | **0.50** | After cleanup, remaining edges are more likely real. Allow larger gaps. |
| `min_gap` | 0 | **2× vein width** | Don't bridge edges that are already touching (cleanup should have merged those). |
| `min_combined_length` | 207px | **3.5× vein width (~82px)** | Allow shorter edges to bridge after cleanup has removed noise. |

### 3.14 Cleanup before final bridge pass

Same cleanup sequence as step 9-12 (remove redundant edges, absorb tiny segments, merge close nodes, remove small fragments, remove isolated nodes). No stub removal here — stubs are preserved as bridge candidates for step 3.15.

### 3.15 Gap bridging (third pass — relaxed facing for short stubs)

```python
_bridge_and_simplify(graph, ..., min_facing_angle=120°,
    median_vein_width=median_vein_width, short_edge_vw=3.0)
```

**What**: A third bridge pass with a relaxed facing angle (120° vs 150°) that only fires when at least one edge is shorter than 3× median vein width and the gap is within 4× median vein width. Targets short stubs near junctions that failed the stricter facing threshold.

**Why**: After maximum cleanup, some short stubs remain disconnected from nearby edges because their terminal direction doesn't align well enough for the 150° facing threshold. These are typically junction fragments where the vein curves sharply. The relaxed threshold is safe here because the graph has been heavily pruned, minimizing false-positive bridges.

**Parameters**:
- `bridge3_max_gap_vw = 4.0` — Max gap as × median vein width.
- `bridge3_short_edge_vw = 3.0` — At least one edge must be shorter than this × median vein width.
- `bridge3_relaxed_facing_angle = 120.0` — Relaxed facing angle threshold (degrees).

### 3.16 Final single-pass stub removal (local vein width)

```python
_remove_stubs_single_pass(graph, max_length=median_vein_width * 3.0,
    distance_map=distance_map, vein_width_multiplier=3.0)
```

**What**: Removes degree-1 stubs at junctions in a single sweep, using local vein width from the distance map.

**Why**: After all bridging and cleanup, a few short dead-end stubs may remain at junctions. The threshold at each stub is `max(global_threshold, 2 × local_half_width × multiplier)`, so stubs in thick-vein areas (e.g., the hinge) are removed more aggressively than with a flat global threshold.

**Critical design choice — single pass, no cascade**: Earlier versions used iterative stub removal, which caused cascading collapse: removing a stub at a degree-3 junction demoted it to degree-2, then simplification contracted through it, exposing more stubs, and so on until the graph collapsed. The single-pass approach removes only the stubs that exist right now.

**Algorithm**:
```
stubs_to_remove = []
for each degree-1 node n:
    neighbor = only neighbor of n
    if degree(neighbor) >= 3:   # it's a stub at a junction
        threshold = max(global_max_length, 2 * distance_map[junction] * multiplier)
        if edge_length < threshold:
            stubs_to_remove.append(n)
for each stub in stubs_to_remove:
    remove node and its edge
```

**Parameters**:
- `final_stub_vein_widths = 3.0` — Max stub length as multiple of vein width (global floor + local).

### 3.17 Snap edge endpoints to node positions

```python
_snap_edge_endpoints(graph)
```

**What**: Replaces the first/last coordinate of each edge's LineString with the actual node position.

**Why**: After simplification and merging, LineString coordinates may not exactly match node positions (they came from original skeleton pixels, but nodes were moved during merging). This creates tiny visual stubs. Snapping ensures geometric consistency.

---

## 4. Landmark Anchoring

**Source**: `models/landmark_anchor.py`

### What it does

Snaps each reliable landmark to the nearest appropriate graph node, modifying the graph if necessary.

### Algorithm

```
for each reliable landmark:
    if landmark is a JUNCTION type (L1-Rs, L2-L3, L4-L5):
        find nearest node with degree >= 3 within snap_radius
        (prefer high-degree nodes at junction regions)
        if only degree-1 found: reject and try edge insertion
    elif landmark is an ENDPOINT type (subcostal break, DTip):
        find nearest node with degree == 1 within snap_radius
    elif landmark is "alula notch":
        skip (margin reference only, don't modify graph)
    else:
        find nearest node of any degree within snap_radius

    if no node found within snap_radius:
        insert a new node on the nearest edge:
            project landmark point onto closest edge's LineString
            split edge at projection point into two edges
            new node becomes the landmark's snap target
```

**Why different preferences by type?** Junction landmarks should snap to actual junctions (degree-3+ nodes) — snapping them to a degree-1 endpoint would be incorrect. Endpoint landmarks (SC, DTip) should snap to degree-1 nodes where the vein terminates.

**Edge insertion**: If no suitable node exists nearby, we split the nearest edge at the point closest to the landmark. This creates a new node precisely where the landmark says a junction should be. The split preserves all edge attributes and creates two new edges.

**Parameters**:
- `snap_radius` — Maximum distance to snap. When median vein width is available, uses `snap_radius_vw × median_vein_width` (default: 2× vein width). Falls back to `snap_radius_um / um_per_px` (207px) when vein width is unavailable.

---

## 5. Vein Labeling

**Source**: `models/vein_tracer.py`, function `trace_veins_from_landmarks()`

This phase assigns a vein identity label (e.g., "L3", "costa") to each edge in the graph. It operates in 6 phases, each building on the previous one's results.

**Central data structure**: `edge_labels: dict[tuple[int,int], str]` — maps canonical edge keys `(min(u,v), max(u,v))` to vein name strings.

### Phase 0: Merge longitudinals through crossvein junctions

**Source**: `models/junction_resolver.py`, function `merge_through_junctions()`

**What**: At degree-3 junctions, finds the most collinear pair of edges and merges them into a single edge, leaving the third edge as a branch (crossvein).

**Why**: At a crossvein junction, two longitudinal veins pass through while the crossvein branches off. Without merging, the longitudinals would each be split into separate edges at the junction, complicating labeling. After merging, each longitudinal is a single continuous edge.

**Algorithm**:
```
for each degree-3 node:
    for each pair of edges at this node:
        angle = angle_between_departure_directions(edge_a, edge_b)
    best_pair = pair with angle closest to 180° (most collinear)

    GUARD: perpendicularity check
        third_edge_angle = angle between third edge and merged pair
        if third_edge is too collinear (> 150°) with either merged edge:
            skip this junction (it's a divergence, not a crossvein)

    merge best_pair:
        concatenate their LineStrings through the junction node
        remove junction node (contract into the merged edge)
        third edge remains as a branch
```

**Protected nodes**: Landmark nodes are never contracted. This preserves the topology that the labeling phase depends on.

**Protected edges**: Already-labeled edges (e.g., from a previous pass) are not merged across label boundaries.

### Phase 1: Detect costa edges

**Source**: `models/costa_detector.py`

**What**: Identifies graph edges that run along the anterior wing margin as "costa".

**Why**: Costa doesn't pass through any landmark junction — it runs along the wing edge from the subcostal break distally. It's detected by proximity to the wing outline rather than by landmark connectivity.

**Algorithm**:

#### Step 1.1: Build margin band
```
wing_mask = rasterize(wing_outline)
edge_distance = EDT(wing_mask)     # distance inward from wing boundary
margin_band = (wing_mask > 0) AND (edge_distance <= 2 * median_vein_width)
```
This creates a band of pixels along the wing margin, 2x vein width deep.

#### Step 1.2: Trim hinge
```
draw line from SC to AN, extend ±500px
split wing_outline at this line
keep margin_band pixels only in the distal (larger) piece
```
**Why**: The margin band wraps around the entire wing including the hinge. Veins in the hinge are not costa — they're the convergence zone where all veins merge into the body.

#### Step 1.3: Cut at subcostal break
```
for each band pixel:
    if dot(pixel - SC, SC→AN) > 0:        # proximal to SC
    AND cross(SC→AN, SC→pixel) != centroid_side:  # anterior side
        remove this pixel from band
```
**Why**: The band extends past SC into L1 territory on the anterior margin. This targeted cut removes only the proximal-anterior wedge without touching the posterior margin or distal costa.

#### Step 1.4: Score edges and reject proximal departures
```
for each graph edge:
    fraction = count(edge samples in band) / total_samples
    if fraction >= 0.50:
        if edge touches SC node AND departs toward AN (proximal direction):
            REJECT — this is L1, not costa
        else:
            label as "costa"
```
**Why the proximal rejection**: An edge departing SC toward the hinge is L1 (going toward the wing base), not costa (going toward the wing tip). The dot product of the edge direction with the SC→AN vector discriminates proximal vs. distal departure.

**Parameters**:
- `costa_min_in_band_fraction = 0.50` — At least 50% of edge must lie within the margin band.
- `costa_propagation_max_distance_vw = 4.0` — During later propagation (Phase 2b), costa edges must stay within 4× vein width of the band.

### Phase 2: Label edges at landmark positions

**Source**: `vein_tracer.py`, function `_label_landmark_edges()`

**What**: Uses landmark positions and the graph topology to label edges directly connected to landmark nodes.

**Why**: This is the core of the landmark-anchored approach. Each reliable landmark tells us exactly which veins meet at that point. By examining the direction and connectivity of edges at each landmark, we can assign confident labels.

#### DTip → L3
```
edge at DTip node → label as "L3"
```
DTip is a degree-1 endpoint where L3 terminates at the wing tip. Its single edge is L3.

#### Subcostal break → L1
```
edge at SC node → label as "L1"
```
SC is where L1 meets the costa. After costa detection (Phase 1), the remaining edge at SC is L1.

#### L2-L3 junction → L2, L3, Rs

This is the most complex labeling because three veins meet here.

```
unlabeled_edges = edges at L2-L3 node not already labeled (e.g., not costa)

if L2.d AND DTip landmarks both available:
    SIMULTANEOUS MATCHING:
    for each edge:
        dist_to_L2d = edge.LineString.distance(L2.d.point)
        dist_to_DTip = edge.LineString.distance(DTip.point)
    edge nearest to L2.d → "L2"
    edge nearest to DTip → "L3"  (if not already labeled as L3 from DTip phase)
    remaining edge(s) → "Rs"

elif only L2.d available:
    edge nearest to L2.d → "L2"
    remaining → "Rs"

elif only DTip available:
    direction = vector from L2-L3 node toward DTip
    for each edge:
        departure_angle = angle between edge departure and DTip direction
    edge with smallest angle → "L3"
    remaining → "Rs"
```

**Why simultaneous matching?** Sequential matching (assign L2 first, then L3) can produce swaps when one landmark is equidistant from two edges. Scoring all edges against all landmarks simultaneously finds the globally optimal assignment.

#### L1-Rs junction → L1, Rs
```
if SC landmark available:
    direction_to_SC = vector from L1-Rs node toward SC
    for each unlabeled edge:
        departure = edge departure direction (sampled 80px from junction)
        angle = angle_between(departure, direction_to_SC)
        if angle < 60°:
            label as "L1"  (heading toward SC)
        else:
            label as "Rs"  (heading away from SC, toward L2-L3)
```

**Why 60° threshold?** L1 runs from the L1-Rs junction toward SC. An edge heading within 60° of the SC direction is almost certainly L1. Everything else is Rs or another vein.

#### L4-L5 junction → L4, L5

```
unlabeled_edges = edges at L4-L5 node not already labeled

if L4.d AND L5.d both available:
    SIMULTANEOUS MATCHING (same approach as L2-L3):
    for each edge:
        dist_to_L4d = edge.LineString.distance(L4.d.point)
        dist_to_L5d = edge.LineString.distance(L5.d.point)
    best edge for L4.d → "L4"
    best edge for L5.d → "L5"

    CONTESTED EDGE RESOLUTION:
    if both landmarks point to same edge:
        give edge to whichever landmark is closer
        assign other landmark's vein to the remaining edge

elif DTip available (fallback):
    direction = vector from L4-L5 node toward DTip
    edge with smallest angle to this direction → "L4" (anterior)
    edge with largest angle → "L5" (posterior)
```

### Phase 2b: Propagate labels through degree-2 nodes

**Source**: `_propagate_through_degree2()`

**What**: Extends vein labels through simple pass-through nodes (degree-2 nodes where the vein continues straight).

**Why**: After Phase 2, only edges directly at landmarks are labeled. The vein continues through many degree-2 nodes (pass-through points) that need the same label.

**Algorithm**:
```
repeat until no changes:
    for each degree-2 node:
        edge_a, edge_b = the two edges at this node
        if edge_a is labeled and edge_b is not:
            label edge_b with same label as edge_a
        elif edge_b is labeled and edge_a is not:
            label edge_a with same label as edge_b
```

**Costa propagation guard**: If propagating a costa label, the new edge must stay within 4× vein width of the margin band. This prevents costa from "leaking" inland through degree-2 nodes into longitudinal veins.

```
if label == "costa":
    band_distance = precomputed distance-from-band map
    for each coordinate in candidate edge's LineString:
        if band_distance[coord] > costa_max_dist:
            BLOCK propagation — this edge is too far from the wing margin
```

### Phase 2c: Extend to distal landmarks

**Source**: `_extend_to_distal_landmarks()`

**What**: Ensures each longitudinal vein (L2, L3, L4, L5) reaches its distal landmark point.

**Why**: Fragmentation or junction topology may prevent the label from propagating all the way to the distal endpoint. If L4's label stops one edge short of L4.d, this phase bridges the gap.

**Algorithm**:
```
for each (vein, landmark) pair: [(L2, L2.d), (L3, DTip), (L4, L4.d), (L5, L5.d)]:
    search_radius = median_vein_width * distal_landmark_search_vw

    # Check if vein already reaches landmark
    for each edge labeled as this vein:
        if edge.LineString.distance(landmark.point) <= search_radius:
            already_reached = True; break

    if not already_reached:
        # Find nearest unlabeled edge to landmark
        for each unlabeled edge:
            dist = edge.LineString.distance(landmark.point)
        if best_dist <= search_radius:
            label that edge with this vein
```

**Parameters**:
- `distal_landmark_search_vw = 2.0` — Search radius as multiple of median vein width.

### Phase 2d: Re-propagate after extension

Same as Phase 2b. After extending labels to distal landmarks, new degree-2 nodes may have become propagatable.

### Phase 2e: Connect disconnected vein fragments

**Source**: `_connect_vein_fragments()`

**What**: For each longitudinal vein with disconnected labeled segments, finds the shortest path through unlabeled edges to reconnect them.

**Why**: Even after propagation and extension, a vein may exist as two or more disconnected fragments. This phase stitches them together using the shortest path through the unlabeled graph — the most parsimonious connection.

**Algorithm**:
```
for each longitudinal vein (L2, L3, L4, L5):
    components = connected components of this vein's labeled edges
    if len(components) <= 1:
        continue  # vein is contiguous

    # Build unlabeled subgraph (weighted by edge length)
    unlabeled_graph = subgraph of edges not in edge_labels, weight = length_px

    for each pair of components (i, j):
        for each node in component_i, each node in component_j:
            path = shortest_weighted_path(unlabeled_graph, node_i, node_j)
            track best (shortest total length)

        if best path found:
            label all edges on path with this vein's label
```

After connection, another round of degree-2 propagation runs to fill any remaining gaps.

### Phase 3: Detect L6

**Source**: `_detect_l6()`

**What**: Identifies L6 — a short posterior branch off L5 near the L4-L5 junction.

**Why**: L6 is the only longitudinal vein not connected to a landmark junction. It's detected by its morphological characteristics: short, near L4-L5, heading posteriorly.

**Prerequisites**: Rs must be labeled (its length serves as reference), L4-L5 landmark must exist.

**Algorithm**:
```
rs_length = total length of Rs edges
search_radius = rs_length * 1.5

for each unlabeled edge:
    # Length filter: 50-150% of Rs length
    if edge_length < rs_length * 0.5 or edge_length > rs_length * 1.5:
        skip

    # Proximity filter: one endpoint within 1.5× Rs length of L4-L5
    min_dist_to_l4l5 = min distance of either endpoint to L4-L5 node
    if min_dist_to_l4l5 > search_radius:
        skip

    # Direction filter: must head substantially posteriorly.
    # With a WingAxis, project the edge onto the AP unit vector
    # (rotation-invariant). Without one, fall back to abs(dy).
    if wing_axis is not None:
        posterior_component = abs(dot(edge_vec, wing_axis.ap_vector))
    else:
        posterior_component = abs(end_y - start_y)
    if posterior_component < edge_length * 0.3:
        skip  # not heading substantially posterior

    score = min_dist_to_l4l5  # prefer closer to junction

best scoring candidate → label as "L6"
```

**Why the axis projection**: The legacy check assumed the wing was upright in the image (posterior = +Y). When a specimen is rotated, that assumption breaks and L6 can be rejected on valid wings. The `WingAxis.ap_vector` is derived from the alula notch → DTip PD axis rotated 90°, so the posterior direction travels with the wing regardless of image orientation.

### Phase 4: Detect crossveins (primary method)

**Source**: `_detect_crossveins()`

**What**: Identifies ACV and PCV by finding unlabeled edges whose endpoints connect to the correct pair of longitudinal veins.

**Why**: Crossveins bridge two longitudinals:
- **ACV** connects L3 ↔ L4
- **PCV** connects L4 ↔ L5

After junction merging (Phase 0), crossveins are typically short unlabeled branches at junctions. Their endpoints are geometrically close to (or shared with) the longitudinals they connect.

**Algorithm**:
```
for each crossvein (ACV: L3↔L4, PCV: L4↔L5):
    if either required longitudinal is not labeled:
        skip

    for each unlabeled edge (u, v):
        # Try both orientations
        dist_a = distance(u, vein_a) + distance(v, vein_b)
        dist_b = distance(u, vein_b) + distance(v, vein_a)
        score = min(dist_a, dist_b)

    best scoring edge → label as crossvein name
```

**Node-to-vein distance** (`_node_vein_distance`):
- Returns 0.0 if the node is a graph endpoint of the labeled vein
- Returns geometric distance if node is within 50px of the vein's LineString
- Returns None if node is too far (not connected to that vein)

### Phase 4a: Junction-based crossvein detection

**Source**: `_detect_crossveins_via_junctions()`

**What**: Detects crossveins by tracing unlabeled paths between degree-3+ junctions on labeled longitudinal veins. Runs after Phase 4 for any crossveins not yet found.

**Why**: Phase 4 only finds single-edge crossveins where both endpoints touch the longitudinals. Phase 4a handles multi-edge crossveins that pass through degree-2 nodes, or crossveins whose attachment points are embedded in the longitudinal graph rather than being direct endpoint-to-endpoint connections.

**Algorithm**:
```
for each crossvein (ACV: L3↔L4, PCV: L4↔L5):
    if already detected by Phase 4:
        skip

    find degree-3+ nodes on vein_a with unlabeled branches → starts
    find node set for vein_b → targets

    for each (junction, unlabeled_neighbor) in starts:
        BFS through unlabeled edges from unlabeled_neighbor
        if BFS reaches a node in targets:
            record path and total length

    shortest path → label all edges as crossvein name
```

**Analogous to** `_extend_to_distal_landmarks` for longitudinals: instead of landmark points, uses degree-3+ junctions on the longitudinals as anchor points.

### Phase 4b: Fallback crossvein detection using landmarks

**Source**: `_detect_crossveins_fallback()`

**What**: If Phase 4 failed to find a crossvein, uses the (unreliable) crossvein landmarks as a fallback.

**Why**: Some specimens have crossveins that don't neatly connect endpoint-to-endpoint with longitudinals. The crossvein landmarks (ACV.a, ACV.p, PCV.a, PCV.p) provide independent evidence for crossvein location.

**Landmark tiers** (checked in order, stop after first success):
- ACV: Tier 1 = ACV.p, Tier 2 = ACV.a
- PCV: Tier 1 = PCV.a, Tier 2 = PCV.p

**Candidate scoring**:
```
for each unlabeled edge near the tier landmark:
    length_ok = crossvein_min_length_vw * vein_width <= length <= crossvein_max_length_vw * vein_width
    proximity = distance to landmark point
    perpendicularity = angle between edge and adjacent longitudinals (0-90°, higher = more perpendicular)
    length_score = closeness to ideal length

    combined_score = proximity + (1 - perp_score) * 200 + (1 - length_score) * 100
```

The scoring heavily weights perpendicularity (crossveins cross longitudinals at ~90°) and proximity to the landmark.

### Phase 4c: Post-crossvein degree-2 propagation

**Source**: `_propagate_through_degree2()` (same function as Phase 2b)

**What**: Re-runs degree-2 label propagation after all crossvein detection phases. At any degree-2 node where one edge is now labeled (including newly labeled crossveins) and the other is not, the unlabeled edge inherits the label.

**Why**: Crossvein detection (Phases 4/4a/4b) may label edges that create new degree-2 propagation opportunities. For example, a short unlabeled stub at a PCV endpoint becomes propagatable once the adjacent PCV edge is labeled.

### Phase 4d: Label ectopic veins (EV1, EV2, …)

**Source**: `_label_ectopic_edges()`

**What**: Promotes every still-unlabeled edge to an ectopic vein label. Each connected component of unlabeled edges becomes one `EV<N>`; the downstream `_build_vein_identifications()` then materializes them as `VeinIdentification` objects with `status=VeinStatus.ECTOPIC`.

**Why**: Previously, edges that failed every labeling phase were logged and discarded. On mutant wings this silently dropped real vein tissue — genuine ectopic veins and stray fragments — so downstream stages (region naming, overlay, measurement) never saw it. Preserving them as first-class EV veins lets adjacency/forbidden-region logic reason about the extra tissue.

**Algorithm**:
```
noise_floor = max(median_vein_width_px * 2, 50px)

unlabeled = [edges with no label and a valid LineString]
H = subgraph over unlabeled edges
components = connected_components(H)

# Drop noise, preserve significant runs
kept = [c for c in components if total_length(c) >= noise_floor]

# Deterministic: longest first, tie-break on minimum node id
kept.sort(key=(-total_length, min(nodes)))

for idx, component in enumerate(kept, start=1):
    label every edge in component as f"EV{idx}"
```

**Noise floor**: reuses the same heuristic as `intervein_namer.py`'s `frag_buffer` (`max(median_vein_width_px * 2, 50px)`). Sub-threshold components are silently dropped — identical behavior to the pre-change "log and drop".

**Downstream contract**: After Phase 4d, every surviving graph edge has a label. `EV*` labels flow through `_build_vein_identifications()` unchanged except that the per-vein build loop assigns `VeinStatus.ECTOPIC` when `vein_id.startswith("EV")`. `_vein_type()` falls EVs through to `VeinType.LONGITUDINAL` — the status field is the canonical signal.

### Phase 5: Build VeinIdentification objects

**Source**: `_build_vein_identifications()`

**What**: Converts the `edge_labels` dict into a list of `VeinIdentification` objects with merged centerline geometry and metadata.

**Algorithm**:
```
for each unique label in edge_labels:
    edges = all graph edges with this label
    lines = [edge.line for edge in edges]

    if len(lines) == 1:
        centerline = lines[0]
    elif len(lines) > 1:
        centerline = _merge_nearby_lines(lines, max_gap)
        # Greedy nearest-neighbor chaining:
        # Start with first line, repeatedly append/prepend nearest neighbor
        # Stop if nearest neighbor is farther than max_gap

    vein = VeinIdentification(
        vein_id = label,
        vein_type = LONGITUDINAL / CROSSVEIN / COSTA / RADIAL_SECTOR,
        status = IDENTIFIED,
        centerline = centerline,
        edge_ids = [edge.edge_id for edge in edges],
        length_px = sum(line.length for line in lines),
        evidence = [f"{len(edges)} edges"],
    )
```

---

## 5.5. Intervein Polygon Splitting (h-maxima seed detection + watershed)

**Source**: `models/intervein_splitter.py`, function `split_merged_intervein_polygons()`

**What**: Preprocessing pass that runs between vein labeling (Step 5) and intervein region naming (Step 6). The pixel classifier occasionally fuses adjacent intervein regions where a crossvein is short or interrupted. This pass physically re-splits such polygons using h-maxima peak detection on the distance transform, followed by constrained watershed.

**Why**: Downstream region naming (`intervein_namer.py`) can only label the polygons it's given. If the classifier merged discal and 2nd posterior into one blob, the namer can at best report a `"discal + 2nd posterior"` merge label. This stage attempts to produce two separate polygons so each region gets its own entry.

**Pipeline** (all raster at full image resolution):

```
# Step 1: barrier mask
wing_mask = rasterize(wing_outline)
wing_mask = distance_transform_edt(wing_mask) > wing_buffer_px   # inset edge

barrier = rasterize(canonical vein centerlines, excluding L6 and EV*)
vein_barrier = distance_transform_edt(barrier == 0) <= vein_barrier_px

interior_mask = wing_mask & ~vein_barrier

# Step 2: seeds via h-maxima peak detection
h = median_vein_width_px × intervein_split_h_vw
for each intervein_poly:
    poly_mask = rasterize(poly)
    edt = distance_transform_edt(poly_mask)
    edt_smooth = gaussian_filter(edt, sigma=median_vein_width_px)
    peaks = h_maxima(edt_smooth, h)
    if no peaks:
        record poly as lost — candidate for reseed pass
    else:
        for each connected component of peaks:
            assign a new seed label

# Step 3: competitive dilation
surface = -distance_transform_edt(interior_mask)
labels_out = watershed(surface, markers=seeds, mask=interior_mask)

# Step 4: reseed lost polygons
for each lost poly mask:
    if area >= reseed_min_area_px:
        drop a single seed at an interior point of the original mask
re-run watershed if any reseeds happened

# Step 5: raster → polygons
for each label:
    contour = findContours(labels_out == label)
    emit Polygon(contour)
```

**Why h-maxima, not fixed-radius erosion**: The original approach eroded each polygon by a fixed radius and used connected components as seeds. This caused catastrophic over-segmentation on thin polygons — a long strip with minor width variations would fracture into many disconnected pieces, each becoming a phantom seed. h-maxima is adaptive: it finds peaks in the EDT (inscribed-radius landscape) that rise at least `h` pixels above their nearest saddle. A uniform thin strip has no deep saddle and produces exactly one seed regardless of width. A genuinely fused polygon with a fat body and a thin bridge has a deep saddle between two peaks, producing two seeds at exactly the right split location.

**Gaussian smoothing**: The raw EDT can have plateau ripples — floating-point noise on flat ridges where many pixels share nearly identical inscribed radii. Without smoothing, `h_maxima` finds spurious 1-pixel peaks on these plateaus. Smoothing with `sigma = median_vein_width_px` eliminates ripples while preserving real bottlenecks (saddles between genuinely distinct peaks).

**Excluded barriers**: `L6` (a short stub off L5 — not a region divider) and any `EV*` ectopic vein (noisy tissue shouldn't carve territory). Everything else canonical (L1-L5, Rs, ACV, PCV, costa) acts as a buffered barrier.

**Reseed rationale**: Polygons whose EDT landscape is entirely flat (max inscribed radius < h) produce no h-maxima peaks and are "lost" — their territory gets claimed by neighboring labels. If the original footprint is larger than the reseed threshold (default 10,000 µm²), a fresh single-pixel seed is dropped in Step 4 and the polygon reclaims its territory in the second watershed.

**Parameters**:
- `intervein_split_h_vw` (default 2.0) — h-maxima depth threshold as × median vein width
- `intervein_split_reseed_min_area_um2` (default 10,000) — minimum original area (µm²) to trigger a reseed
- `intervein_split_vein_barrier_vw` (default 1.0) — vein centerline buffer radius as × median vein width
- `intervein_split_wing_buffer_vw` (default 1.0) — wing outline inset as × median vein width

**Debug overlay**: passing `debug_out=<path>` to `split_merged_intervein_polygons` writes a diagnostic PNG showing the barrier mask (red tint), seed pixels (green tint), watershed label boundaries (cyan), and unbuffered canonical vein centerlines with endpoint markers (yellow). Useful for diagnosing barrier leaks or seed placement issues.

---

## 5.6. Intervein Region Naming

**Source**: `models/intervein_namer.py`, function `name_intervein_regions()`

**What**: Assigns a region name (marginal, submarginal, 1st basal, 1st posterior, discal, 2nd posterior, 3rd posterior) to each intervein polygon output by Step 5.5 by matching the polygon's adjacent veins against `topology.REGION_EXPECTED_VEINS`.

**Pipeline**:
1. **Adjacency**: buffer each identified vein centerline by `vein_buffer_px` and compute the set of veins whose buffered footprint intersects the polygon boundary by at least `adjacency_min_length_px`.
2. **Primary match**: find every region in `REGION_EXPECTED_VEINS` whose expected vein set is a subset of the detected set. Pick the highest-specificity match (largest expected set).
3. **PD tie resolver**: if the top specificity is tied between multiple regions (currently only `discal` vs `2nd posterior`), defer the polygon. After the main loop, `_resolve_pd_ties()` groups deferred polygons by their tied candidate set, looks up `topology.REGION_PD_PAIRS`, and assigns names by sorting members along the wing PD axis (proximal → distal).
4. **Merge detection** (`_check_merged`): if no single region matches, run N-way merge enumeration. Details below.
5. **Coverage fallback**: if no merge matches either, pick the region with the highest per-region coverage (fraction of its expected veins present); threshold 0.5.
6. **Absorbed merge detection**: `_detect_absorbed_merges()` runs a two-phase post-pass. **Phase A** (forbidden-adjacency split): when a canonical region is missing AND another canonical name is duplicated across 2+ polygons, uses `build_region_forbidden_veins()` to identify which duplicate holds a vein impossible for its claimed identity (e.g. L6-adjacent polygon cannot be discal → must be 3rd posterior) and renames it to the best-matching missing region. **Phase B** (legacy append fallback): for anything still missing after Phase A, walks named regions checking whether any region's bounding veins are a superset of the missing region's expected veins — if so, appends `" + <missing>"` to the name and marks it merged. The adjacency constraint prevents non-adjacent concatenation.
7. **Ectopic fragment absorption**: `_absorb_ectopic_fragments()` sweeps small unnamed or tiny-named polygons into the adjacent named region sharing the most boundary.

### N-way merge detection

`_check_merged()` delegates to `_enumerate_merge_candidates()`, which brute-forces over all connected subsets of a region adjacency graph derived once from `topology.VEIN_BOUNDARIES`:

```python
# Built at module import:
_REGION_ADJACENCY: dict[str, dict[str, str]]       # region -> {neighbor: separator_vein}
_REGION_EDGE_SEPARATOR: dict[frozenset[str], str]  # {r1, r2} -> separator
```

For each subset size from 2 up to `max_merge_size` (default: no cap, so up to all 7 regions):
1. Skip subsets that are not connected in `_REGION_ADJACENCY`.
2. Compute the internal separator veins — every vein whose `VEIN_BOUNDARIES` pair lies entirely inside the subset.
3. Compute `merged_expected = union(effective_expected[r] for r in subset) - internal_separators`.
4. Keep the subset only if `merged_expected ⊆ detected`.

Scoring (descending):
1. `len(merged_expected)` — prefer the merge that "explains" the most bounding veins (most specific match).
2. `-subset_size` — among equal-specificity candidates, prefer the smallest subset (avoid overclaiming).
3. AP-ordered region tuple — final deterministic tie-break.

This correctly handles 3+ region fusions that pair-only detection missed, e.g. `marginal + submarginal + 1st posterior` when both L2 and L3 are absent. The returned name is AP-ordered (`" + ".join(regions)`) for stable output.

**Parameters**:
- `vein_buffer_px` (default 25) — buffer radius around each vein centerline for adjacency testing
- `adjacency_min_length_px` (default 30) — minimum shared boundary length for a vein to count as adjacent
- `max_merge_size` (default `None`) — cap on N-way merge size; `None` = no cap, any connected subset of the 7 regions is a candidate

---

## 5.7. Vein Tissue Polygon Assignment

**Source**: `models/intervein_splitter.py`, function `assign_vein_tissue_polygons()`

**What**: Populates `VeinIdentification.tissue_polygon` on every vein that has a centerline, by buffering the centerline to tissue width and clipping to the wing outline.

**Why**: Downstream GeoJSON export and overlay rendering need filled polygons representing vein tissue, not just centerlines. The buffer radius is the same as the intervein splitter barrier mask (`median_vein_width_px × intervein_split_vein_barrier_vw`), ensuring geometric consistency between vein tissue boundaries and intervein region boundaries.

**Algorithm**:
```
buffer_px = round(median_vein_width_px × config.intervein_split_vein_barrier_vw)
for each vein with a centerline:
    tissue = centerline.buffer(buffer_px)
    tissue = tissue.intersection(wing_outline)  # clip to wing
    if MultiPolygon: keep largest piece
    vein.tissue_polygon = tissue
```

**Scope**: All veins with centerlines get tissue polygons — including L6 and ectopic veins (EV*). Veins with no centerline (e.g., `status=ABSENT`) are skipped.

---

## 6. Output

### 6.1 VeinIdentification objects

Each run produces a `list[VeinIdentification]` — one per canonical vein plus any ectopic veins (EV1, EV2, ...):

| Field | Type | Description |
|-------|------|-------------|
| `vein_id` | str | "L1", "L2", ..., "costa", "ACV", "EV1", etc. |
| `vein_type` | VeinType | LONGITUDINAL, CROSSVEIN, COSTA, RADIAL_SECTOR |
| `status` | VeinStatus | IDENTIFIED, ECTOPIC, ABSENT |
| `centerline` | LineString | Merged centerline geometry |
| `tissue_polygon` | Polygon | Buffered centerline clipped to wing outline |
| `edge_ids` | list[int] | Graph edge IDs comprising this vein |
| `length_px` | float | Total length in pixels |
| `evidence` | list[str] | How the vein was identified (e.g., "3 edges") |

### 6.2 InterveinRegion objects

The naming stage produces a `list[InterveinRegion]` with 7 entries (one per canonical region):

| Field | Type | Description |
|-------|------|-------------|
| `name` | str | "marginal", "submarginal", "1st basal", etc. |
| `polygon` | Polygon | Region boundary |
| `area_px2` | float | Area in pixels² |
| `bounding_veins` | set[str] | Veins adjacent to this region |
| `status` | str | "identified", "merged", "inferred" |

### 6.3 GeoJSON export (GT_naming format)

**Source**: `views/geojson_export.py`, function `export_geojson()`

**What**: Writes veins and regions as a GeoJSON FeatureCollection matching the GT_naming annotation format used by QuPath and ground-truth files.

**Format**: Each feature has:
- `geometry`: Polygon (tissue polygon for veins, region polygon for regions)
- `properties.objectType`: `"annotation"`
- `properties.classification.name`: feature name (e.g., "L3", "discal")
- `properties.classification.color`: RGB array from `VEIN_COLORS` / `REGION_COLORS`
- `properties.measurements`: area in pixels, area in µm² (if `um_per_px` provided), vein length in pixels/µm

### 6.4 CSV export

**Source**: `views/csv_export.py`

**Two formats** depending on mode:

**Single mode** (`export_csv()`): Long format, one row per feature. Columns: `specimen`, `feature`, `category`, `type`, `status`, `area_px`, `area_um2`, `length_px`, `length_um`. Includes wing-level measurements (wing area, wing length, crossvein distance) as rows with `category: "wing"`.

**Batch mode** (`export_csv_batch()`): Wide format, one row per specimen. All measurements as columns in canonical AP order. Wing-level columns first (`wing area_px`, `wing area_um2`, `wing length_px`, `wing length_um`, `crossvein distance_px`, `crossvein distance_um`), then per-vein length columns, then per-region area columns. Ectopic veins are omitted from the wide format.

**Wing-level measurements**:
- **Wing area**: `wing_outline.area` (union of all detection polygons)
- **Wing length**: Euclidean distance between landmarks L1-Rs and DTip
- **Crossvein distance**: Euclidean distance between landmarks ACV.p and PCV.a
- **Anterior/posterior compartment areas**: Wing split along L4 vein axis. Method: minimum rotated bounding box around L4 centerline → anterior long edge (closer to L3) → extended to bisect wing outline. Halves labeled by proximity to L3 (anterior) vs L5 (posterior).

---

## 7. Parameter Reference

All parameters live in `config.py` as fields of `PipelineConfig`. Distance thresholds specified in µm are converted to pixels using `um_per_px = 0.483`.

### Scale
| Parameter | Default | Description |
|-----------|---------|-------------|
| `um_per_px` | 0.483 | Microns per pixel |

### Skeletonization
| Parameter | Default | Description |
|-----------|---------|-------------|
| `skeleton_methods` | [RIDGE] | Skeletonization method |
| `smooth_sigma` | 2.0 | Gaussian sigma for Hessian smoothing |

### Pruning
| Parameter | Default | Description |
|-----------|---------|-------------|
| `prune_min_length_vein_widths` | 2.0 | Cap for adaptive branch pruning (× vein width) |
| `final_stub_vein_widths` | 3.0 | Max stub length for final removal (× vein width) |
| `junction_merge_vein_widths` | 2.0 | Junction merge radius (× vein width) |

### Gap bridging (pass 1)
| Parameter | Default | Description |
|-----------|---------|-------------|
| `bridge_max_gap_um` | 200µm | Absolute max gap |
| `bridge_gap_fraction` | 0.15 | Adaptive gap = fraction × longer edge |
| `bridge_direction_window_um` | 100µm | Window for direction sampling |
| `bridge_min_combined_length_um` | 100µm | Min total edge length |
| `bridge_min_facing_angle` | 150° | Endpoints must face each other |
| `bridge_on_axis_max_angle` | 45° | Strict on-axis for longer edge |
| `bridge_on_axis_relaxed_cap` | 45° | Relaxed cap for shorter edge |

### Gap bridging (pass 2)
| Parameter | Default | Description |
|-----------|---------|-------------|
| `bridge2_gap_fraction` | 0.50 | More permissive gap fraction |
| `bridge2_min_gap_vw` | 2.0 | Minimum gap floor (× vein width) |
| `bridge2_min_combined_length_vw` | 3.5 | Min combined length (× vein width) |

### Gap bridging (pass 3 — relaxed facing for short stubs)
| Parameter | Default | Description |
|-----------|---------|-------------|
| `bridge3_max_gap_vw` | 4.0 | Max gap (× vein width) |
| `bridge3_short_edge_vw` | 3.0 | Short edge threshold (× vein width) |
| `bridge3_relaxed_facing_angle` | 120° | Relaxed facing angle |
| `bridge3_on_axis_max_angle` | 45° | On-axis angle |
| `bridge3_on_axis_relaxed_cap` | 45° | Relaxed cap for shorter edge |

### Landmark anchoring
| Parameter | Default | Description |
|-----------|---------|-------------|
| `snap_radius_vw` | 2.0 | Max snap distance (× vein width, primary) |
| `snap_radius_um` | 100µm | Max snap distance (fallback when vein width unavailable) |

### Costa detection
| Parameter | Default | Description |
|-----------|---------|-------------|
| `costa_min_in_band_fraction` | 0.50 | Min edge fraction in margin band |
| `costa_propagation_max_distance_vw` | 4.0 | Max propagation distance from band (× vein width) |

### Crossvein detection
| Parameter | Default | Description |
|-----------|---------|-------------|
| `crossvein_min_length_vw` | 4.0 | Min crossvein length (× vein width) |
| `crossvein_max_length_vw` | 25.0 | Max crossvein length (× vein width) |

### Vein tracing
| Parameter | Default | Description |
|-----------|---------|-------------|
| `departure_sample_um` | 100µm | Direction sampling window |
| `distal_landmark_search_vw` | 2.0 | Search radius for distal extension (× vein width) |
| `merge_max_gap_um` | 50µm | Max gap for line merging in output |

### Intervein region naming
| Parameter | Default | Description |
|-----------|---------|-------------|
| `vein_buffer_px` | 25 | Buffer radius for adjacency testing |
| `adjacency_min_length_px` | 30 | Min shared boundary length to count as adjacent |
| `max_merge_size` | None | Cap on N-way merge size; None = no cap |
