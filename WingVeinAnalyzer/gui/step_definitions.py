"""Step metadata: names, descriptions, parameter specs for all 18 pipeline steps."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class StepParam:
    """A single read-only parameter displayed in the parameter bar."""
    name: str
    value: str
    tooltip: str = ""


@dataclass
class StepDef:
    """Metadata for one pipeline step."""
    index: int
    name: str
    short_name: str
    description: str
    pseudocode: str
    params: list[StepParam] = field(default_factory=list)
    runs_computation: bool = True  # False = visualization-only (uses cached result)


STEP_DEFS: list[StepDef] = [
    # 0: Load Inputs
    StepDef(
        index=0,
        name="Load Inputs",
        short_name="Load",
        description=(
            "Load the TIFF image and parse the GeoJSON annotation file. "
            "Extracts intervein polygons (8 distinct regions) and vein mask polygons "
            "(defining where vein tissue exists). Computes the wing bounding box from "
            "polygon extents."
        ),
        pseudocode=(
            "image = cv2.imread(tiff_path)\n"
            "annotations = parse_geojson(geojson_path)\n"
            "polygons = annotations.intervein_polygons\n"
            "vein_polygons = annotations.vein_polygons\n"
            "wing_bbox = bounding_box(polygons)"
        ),
        params=[],
        runs_computation=True,
    ),
    # 1: Rasterize & Voronoi
    StepDef(
        index=1,
        name="Rasterize & Voronoi Partition",
        short_name="Voronoi",
        description=(
            "Rasterize intervein polygons to a label map and the vein mask to binary. "
            "Apply morphological closing to bridge small gaps. Use distance_transform_edt "
            "to build a Voronoi partition: every background pixel is assigned to its nearest "
            "intervein polygon. Within the vein mask, boundaries between adjacent Voronoi "
            "regions become the equidistant centerlines."
        ),
        pseudocode=(
            "label_map[polygon_i] = i + 1\n"
            "vein_mask = rasterize(vein_polygons)\n"
            "vein_mask = morphological_close(vein_mask, kernel=11)\n"
            "nearest_labels = voronoi_edt(label_map)\n"
            "# Each vein-mask pixel now knows which polygon it's closest to"
        ),
        params=[
            StepParam("closing_kernel_size", "11", "Morphological closing kernel for vein mask"),
        ],
        runs_computation=True,
    ),
    # 2: Centerline Extraction
    StepDef(
        index=2,
        name="Centerline Extraction",
        short_name="Centerlines",
        description=(
            "Extract centerlines from the Voronoi partition. For each pair of adjacent "
            "polygon labels, the pixels where the Voronoi label changes form a 1-pixel "
            "boundary. These boundary pixels are traced into ordered LineString geometries. "
            "Short fragments below min_line_length are discarded; nearby endpoints within "
            "bridge_threshold are connected."
        ),
        pseudocode=(
            "for each adjacent label pair (i, j):\n"
            "  boundary = vein_mask & (label_changes from i to j)\n"
            "  pixels = scan_median_order(boundary)\n"
            "  if len(pixels) >= min_line_length:\n"
            "    centerlines[(i,j)] = LineString(pixels)"
        ),
        params=[
            StepParam("min_line_length", "10 px", "Minimum centerline segment length"),
            StepParam("bridge_threshold", "30 px", "Max gap to bridge nearby endpoints"),
        ],
        runs_computation=False,  # cached from step 1
    ),
    # 3: Find Junctions
    StepDef(
        index=3,
        name="Find Triple Junctions",
        short_name="Junctions",
        description=(
            "Detect triple (or higher) junctions where 3+ vein centerline segments "
            "converge. All segment endpoints are collected, then clustered: endpoints "
            "within snap_radius pixels of each other are grouped. Clusters with 3+ "
            "endpoints become junction points."
        ),
        pseudocode=(
            "endpoints = [start, end for each centerline segment]\n"
            "for ep in endpoints:\n"
            "  cluster = find_nearby(ep, snap_radius=30px)\n"
            "  if len(cluster) >= 3:\n"
            "    junctions.append(JunctionPoint(mean(cluster)))"
        ),
        params=[
            StepParam("snap_radius", "30 px", "Max distance to cluster endpoints"),
        ],
        runs_computation=True,  # runs identify_veins_and_regions() (full)
    ),
    # 4: Merge Segments
    StepDef(
        index=4,
        name="Merge Segments at Junctions",
        short_name="Merge",
        description=(
            "At each triple junction, determine which pair of incoming segments should be "
            "merged (they're part of the same vein). Uses tangent direction at the junction: "
            "the pair with the most collinear tangents (smallest angle between them) is merged. "
            "An orientation guard prevents merging longitudinals (<25 deg) with crossveins (>55 deg)."
        ),
        pseudocode=(
            "for junction in junctions:\n"
            "  segments = junction.arriving_segments\n"
            "  for pair in combinations(segments, 2):\n"
            "    angle = tangent_angle_between(pair)\n"
            "    if angle < collinearity_threshold:\n"
            "      if not (one_longitudinal AND one_crossvein):\n"
            "        merge(pair) → MergedPath"
        ),
        params=[
            StepParam("collinearity_threshold", "45°", "Max angle between tangents to merge"),
            StepParam("min_gap", "15°", "Min separation from next-best pair"),
            StepParam("orientation_guard", "25°/55°", "Longitudinal/crossvein cutoffs"),
        ],
        runs_computation=False,  # cached from step 3
    ),
    # 5: Split Sharp Turns
    StepDef(
        index=5,
        name="Split at Sharp Turns",
        short_name="Split",
        description=(
            "After merging, some paths contain sharp direction changes where a crossvein "
            "was incorrectly merged with a longitudinal. Walk along each merged path, "
            "computing direction changes over a sliding window. Where the direction changes "
            "by more than angle_threshold, split the path into two. Both halves must exceed "
            "min_split_length to keep the split."
        ),
        pseudocode=(
            "for path in merged_paths:\n"
            "  if path.length < min_path_length: skip\n"
            "  for point along path (step=50px):\n"
            "    angle_change = direction_at(point+step) - direction_at(point-step)\n"
            "    if angle_change > 70°:\n"
            "      split path at point"
        ),
        params=[
            StepParam("angle_threshold", "70°", "Direction change to trigger split"),
            StepParam("step_dist", "50 px", "Window size for direction estimation"),
            StepParam("min_path_length", "500 px", "Only split paths longer than this"),
            StepParam("min_split_length", "200 px", "Both halves must exceed this"),
        ],
        runs_computation=False,  # cached from step 3
    ),
    # 6: Classify Crossveins
    StepDef(
        index=6,
        name="Classify Crossveins",
        short_name="Crossveins",
        description=(
            "Identify ACV and PCV from the merged/split paths. Crossveins are characterized "
            "by steep orientation (>60 deg from horizontal) and short length (<15%% of wing "
            "span). Among crossvein candidates, the more anterior one is ACV and the more "
            "posterior one is PCV. Proximity to longitudinal endpoints helps confirm identity."
        ),
        pseudocode=(
            "candidates = [p for p in paths\n"
            "  if p.orientation > 60° and p.length < 0.15 * wing_span]\n"
            "sort candidates by y_centroid (anterior to posterior)\n"
            "ACV = candidates[0]  # more anterior\n"
            "PCV = candidates[1]  # more posterior"
        ),
        params=[
            StepParam("max_crossvein_len", "15% wing span", "Maximum crossvein length"),
            StepParam("orientation_cutoff", "60°", "Minimum orientation from horizontal"),
            StepParam("proximity_threshold", "100 px", "Max distance to longitudinal endpoint"),
        ],
        runs_computation=False,  # cached from step 3
    ),
    # 7: Classify Longitudinals
    StepDef(
        index=7,
        name="Classify Longitudinals",
        short_name="Longitudinals",
        description=(
            "Assign L1-L5 identities to the remaining paths using combinatorial scoring. "
            "All permutations of candidate-to-vein assignments are evaluated. Each candidate "
            "is scored on: Y-position match (0.40 weight), length match to priors (0.30), "
            "and proximity to known crossvein endpoints (0.30). The permutation with the "
            "highest total score wins. An ACV-based L4/L5 swap check runs post-assignment."
        ),
        pseudocode=(
            "candidates = [p for p in paths if not crossvein]\n"
            "for perm in permutations(candidates, [L1..L5]):\n"
            "  score = sum(\n"
            "    0.40 * y_position_score(c, vein) +\n"
            "    0.30 * length_score(c, vein) +\n"
            "    0.30 * crossvein_proximity(c, vein)\n"
            "  )\n"
            "best_perm = argmax(score)"
        ),
        params=[
            StepParam("y_position_weight", "0.40", "Weight for Y-position scoring"),
            StepParam("length_weight", "0.30", "Weight for length prior match"),
            StepParam("crossvein_weight", "0.30", "Weight for crossvein proximity"),
            StepParam("SPATIAL_PRIORS_Y", "L1:0.02-0.20, L2:0.05-0.32, ...", "Normalized Y ranges"),
        ],
        runs_computation=False,  # cached from step 3
    ),
    # 8: Name Regions
    StepDef(
        index=8,
        name="Name Regions from Veins",
        short_name="Regions",
        description=(
            "Name each intervein polygon based on which veins bound it. Uses the "
            "segment_keys from each MergedPath to determine which polygon indices each "
            "vein borders. For each polygon, compute a Jaccard-like overlap score with each "
            "expected region's vein set. Area-based priors disambiguate basal/posterior and "
            "discal/2nd_posterior pairs (which share the same bounding veins)."
        ),
        pseudocode=(
            "for polygon_idx, polygon in enumerate(polygons):\n"
            "  boundary_veins = {vein for vein in vein_map\n"
            "    if polygon_idx in vein.segment_keys}\n"
            "  for region_name in REGION_EXPECTED_VEINS:\n"
            "    score = jaccard(boundary_veins, expected_veins[region_name])\n"
            "    score *= area_prior_multiplier(polygon.area, region_name)\n"
            "  best_match = argmax(scores)"
        ),
        params=[
            StepParam("REGION_EXPECTED_VEINS", "marginal:{L1,L2}, submarginal:{L2,L3}, ...", ""),
            StepParam("REGION_AREA_PRIORS", "marginal:0.05-0.18, submarginal:0.08-0.25, ...", ""),
        ],
        runs_computation=False,  # cached from step 3
    ),
    # 9: Split Merged Polygons
    StepDef(
        index=9,
        name="Split Merged Polygons",
        short_name="Poly Split",
        description=(
            "Detect polygons that are too large (area > expected_max x 1.5) and likely "
            "represent two merged regions from the pixel classifier. Split them using the "
            "identified vein centerline as a dividing line. This creates a new polygon and "
            "re-assigns region names. Skipped if no oversized polygons found."
        ),
        pseudocode=(
            "for idx, polygon in poly_names.items():\n"
            "  if polygon.area > expected_area * 1.5:\n"
            "    dividing_vein = find_separating_vein(polygon)\n"
            "    upper, lower = split_along_vein(polygon, dividing_vein)\n"
            "    polygons.append(lower)\n"
            "    reassign_names(upper, lower)"
        ),
        params=[
            StepParam("area_threshold", "1.5x expected max", "Threshold for oversized detection"),
        ],
        runs_computation=False,  # cached from step 3
    ),
    # 10: Cross-Validation
    StepDef(
        index=10,
        name="Cross-Validation",
        short_name="Validate",
        description=(
            "Cross-validate vein and region assignments for consistency. Checks include: "
            "vein Y-ordering (L1 should be most anterior, L5 most posterior), boundary "
            "consistency (each vein should border the expected regions), crossvein "
            "connectivity (ACV should connect L3-L4, PCV should connect L4-L5), and area "
            "outlier detection for regions."
        ),
        pseudocode=(
            "# Check vein Y-ordering\n"
            "for i, j in adjacent_pairs(L1..L5):\n"
            "  assert y_centroid[i] < y_centroid[j]\n"
            "# Check boundary consistency\n"
            "for vein, regions in VEIN_BOUNDARIES:\n"
            "  assert vein borders expected regions\n"
            "# Flag warnings (don't modify assignments)"
        ),
        params=[],
        runs_computation=False,  # cached from step 3
    ),
    # 11: L1 Recovery
    StepDef(
        index=11,
        name="L1 Recovery from Anterior Edge",
        short_name="L1 Recovery",
        description=(
            "When no costal cell exists, the Voronoi approach can't find L1 (no "
            "costal-marginal boundary). This step skeletonizes the vein mask region "
            "anterior to L2 in the distal wing to extract L1's centerline directly. "
            "Takes the most distal sufficiently-long skeleton component. Skipped when "
            "costal cell is present."
        ),
        pseudocode=(
            "if costal_cell present: SKIP\n"
            "roi = vein_mask pixels anterior to L2 in distal wing\n"
            "skeleton = skeletonize(roi)\n"
            "components = connected_components(skeleton)\n"
            "l1_line = most_distal_long_component\n"
            "if l1_line.length > existing_l1 + 100px:\n"
            "  update L1 assignment"
        ),
        params=[
            StepParam("min_improvement", "100 px", "New L1 must exceed existing by this much"),
        ],
        runs_computation=True,
    ),
    # 12: Costa Extraction
    StepDef(
        index=12,
        name="Costa Extraction",
        short_name="Costa",
        description=(
            "Extract the costa (leading edge vein) as the anterior margin of the marginal "
            "cell polygon. The costa runs along the wing's anterior edge from the hinge "
            "region to where L1 meets the margin. Skipped when no costal region exists."
        ),
        pseudocode=(
            "if not costal_cell in poly_names: mark costa ABSENT\n"
            "marginal_poly = polygons[marginal_cell_idx]\n"
            "costa_line = anterior_margin(marginal_poly)\n"
            "assignments.append(costa assignment)"
        ),
        params=[],
        runs_computation=True,
    ),
    # 13: Wing Outline
    StepDef(
        index=13,
        name="Build Wing Outline",
        short_name="Outline",
        description=(
            "Build the wing outline from the union of all polygons. Each intervein polygon "
            "is buffered by 20px to bridge the vein gaps, then the union is computed. Vein "
            "polygons are included with a smaller 5px buffer to extend the outline to the "
            "full wing tip. The outer boundary of the union becomes the wing outline."
        ),
        pseudocode=(
            "buffered = [poly.buffer(20px) for poly in polygons]\n"
            "buffered += [vp.buffer(5px) for vp in vein_polygons]\n"
            "union = unary_union(buffered)\n"
            "outline = largest_polygon(union).exterior"
        ),
        params=[
            StepParam("buffer_dist", "20 px", "Buffer for intervein polygons"),
            StepParam("vein_buffer", "5 px", "Buffer for vein polygons"),
        ],
        runs_computation=True,
    ),
    # 14: Hinge Detection & Removal
    StepDef(
        index=14,
        name="Hinge Detection & Removal",
        short_name="Hinge",
        description=(
            "Detect the hinge landmarks (subcostal break + alula notch) and draw a cut "
            "line to separate the wing blade from the hinge/body region. The hinge side "
            "is detected by clustering proximal regions (1st_basal, costal, discal). The "
            "cut line is extended beyond the wing outline to ensure a clean split. The "
            "distal piece is kept as the wing blade."
        ),
        pseudocode=(
            "hinge_side = detect_hinge_side(polygons, poly_names)\n"
            "subcostal = find_subcostal_break(outline)\n"
            "alula = find_alula_notch(outline)\n"
            "cut_line = extend(LineString(subcostal, alula), 100px)\n"
            "wing_blade = split(outline, cut_line).distal_piece"
        ),
        params=[
            StepParam("extend", "100 px", "Cut line extension beyond outline"),
            StepParam("min_fragment", "5%", "Minimum fragment size to keep"),
        ],
        runs_computation=True,
    ),
    # 15: Compartments
    StepDef(
        index=15,
        name="Compute Compartments",
        short_name="Compartments",
        description=(
            "Split the wing blade into anterior and posterior compartments along the L4 "
            "vein. L4's LineString is simplified and extended to reach the wing outline "
            "boundary on both ends. The blade polygon is then split by this line into "
            "two pieces: the anterior compartment (containing L1-L3) and the posterior "
            "compartment (containing L5)."
        ),
        pseudocode=(
            "l4_line = assignments['L4'].line\n"
            "l4_extended = extend_to_boundary(l4_line, wing_blade, 500px)\n"
            "l4_simplified = l4_extended.simplify(10px)\n"
            "anterior, posterior = split(wing_blade, l4_simplified)"
        ),
        params=[
            StepParam("simplify", "10 px", "Simplification tolerance for L4"),
            StepParam("extend", "500 px", "Extension distance to reach outline"),
        ],
        runs_computation=True,
    ),
    # 16: Measurements
    StepDef(
        index=16,
        name="Compute Measurements",
        short_name="Measurements",
        description=(
            "Compute all wing measurements: per-vein lengths, crossvein distance, wing "
            "length and width, total wing area, per-region intervein areas, and anterior/"
            "posterior compartment areas. All measurements in pixels; micron values are NaN "
            "if no scale calibration provided."
        ),
        pseudocode=(
            "for vein in assignments:\n"
            "  vein_lengths[vein.id] = vein.line.length\n"
            "crossvein_dist = distance(ACV.centroid, PCV.centroid)\n"
            "wing_area = wing_blade.area\n"
            "wing_length = max_x - min_x of blade\n"
            "for region in intervein_regions:\n"
            "  region_areas[name] = region.area"
        ),
        params=[],
        runs_computation=True,
    ),
    # 17: Final Overlays
    StepDef(
        index=17,
        name="Final Overlays",
        short_name="Overlays",
        description=(
            "Render the two output overlay images. The skeleton overlay draws each "
            "classified vein in its assigned color on the original image with a legend. "
            "The rainbow overlay fills each intervein region with a semi-transparent color "
            "showing the region boundaries and names."
        ),
        pseudocode=(
            "skeleton = render_skeleton_overlay(image, assignments, outline)\n"
            "rainbow = render_rainbow_overlay(image, regions, opacity=0.75)\n"
            "# Left pane: skeleton, Right pane: rainbow"
        ),
        params=[
            StepParam("line_thickness", "6 px", "Vein line thickness on skeleton overlay"),
            StepParam("opacity", "0.75", "Region fill opacity on rainbow overlay"),
        ],
        runs_computation=True,
    ),
]

NUM_STEPS = len(STEP_DEFS)
