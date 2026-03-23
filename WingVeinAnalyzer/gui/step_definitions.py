"""Step metadata: names, descriptions, parameter specs for all 20 pipeline steps."""

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
            "polygon extents. Calls set_scale(um_per_px) to configure the global "
            "micrometer-to-pixel conversion used by all downstream constants."
        ),
        pseudocode=(
            "set_scale(um_per_px)  # configure µm↔px conversion\n"
            "image = cv2.imread(tiff_path)\n"
            "annotations = parse_geojson(geojson_path)\n"
            "polygons = annotations.intervein_polygons\n"
            "vein_polygons = annotations.vein_polygons\n"
            "wing_bbox = bounding_box(polygons)"
        ),
        params=[
            StepParam("um_per_px", "0.483", "Micrometer-per-pixel scale factor"),
        ],
        runs_computation=True,
    ),
    # 1: Rasterize & Voronoi
    StepDef(
        index=1,
        name="Rasterize & Hull-Seeded Voronoi",
        short_name="Voronoi",
        description=(
            "Rasterize the vein mask to binary. Apply morphological closing to bridge "
            "small gaps. Compute the convex hull of the vein mask, subtract the vein mask "
            "to get non-vein regions (hull seeds). Assign sequential labels 1..M to large "
            "connected components (>=10k px). Phantom seeds with <10%% overlap with the "
            "original intervein polygons are filtered out. Use distance_transform_edt to "
            "build a Voronoi partition from these hull seeds. Boundaries between adjacent "
            "Voronoi regions within the vein mask become vein centerlines. Voronoi regions "
            "are vectorized into polygons that replace the input intervein polygons."
        ),
        pseudocode=(
            "vein_mask = rasterize(vein_polygons)\n"
            "vein_mask = morphological_close(vein_mask, kernel=11)\n"
            "hull = convex_hull(vein_mask_points)\n"
            "seed_mask = hull_raster & ~vein_mask\n"
            "components = connected_components(seed_mask)\n"
            "seed_labels[large_comp_i] = i  # sequential 1..M\n"
            "filter_phantom_seeds(seed_labels, intervein_polygons)\n"
            "nearest_labels = voronoi_edt(seed_labels)\n"
            "voronoi_polygons = vectorize(nearest_labels, hull)"
        ),
        params=[
            StepParam("closing_kernel_size", "11", "Morphological closing kernel for vein mask"),
            StepParam("min_seed_area", "2333 µm²", "Minimum component area to use as seed"),
        ],
        runs_computation=True,
    ),
    # 2: Hull Seeds
    StepDef(
        index=2,
        name="Hull Seed Visualization",
        short_name="Hull Seeds",
        description=(
            "Visualize the hull seeding process. Left: the convex hull outline drawn "
            "over the vein mask. Right: the hull-minus-vein connected components colored "
            "by their sequential label. Small components (<10k px) from vein mask "
            "holes are filtered out. Each large component becomes its own Voronoi seed — "
            "the number of regions is determined by the vein mask topology, not the input "
            "polygon count."
        ),
        pseudocode=(
            "hull_mask = rasterize(convex_hull(vein_mask))\n"
            "seed_mask = hull_mask & ~vein_mask\n"
            "components = label(seed_mask, 8-connectivity)\n"
            "for comp in components:\n"
            "  if comp.area < 10k: skip\n"
            "  seed_labels[comp] = next_label++\n"
            "voronoi_polygons = vectorize(nearest_labels)"
        ),
        params=[],
        runs_computation=False,  # visualization of step 2 data
    ),
    # 3: Centerline Extraction
    StepDef(
        index=3,
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
            StepParam("min_line_length", "4.8 µm", "Minimum centerline segment length"),
            StepParam("bridge_threshold", "14.5 µm", "Max gap to bridge nearby endpoints"),
        ],
        runs_computation=False,  # cached from step 2
    ),
    # 4: Identify Veins & Regions
    StepDef(
        index=4,
        name="Identify Veins & Regions",
        short_name="Identify",
        description=(
            "Run the full identify_veins_and_regions() pipeline: find triple junctions, "
            "merge collinear segments, split sharp turns, assign longitudinals (L1-L5) "
            "first, then classify crossveins (ACV/PCV) by proximity to assigned "
            "longitudinals, name intervein regions from vein boundaries, split oversized "
            "polygons, and cross-validate. Original annotation polygons are named directly "
            "using spatial proximity to classified veins. Steps 6-12 visualize cached sub-results."
        ),
        pseudocode=(
            "junctions = find_triple_junctions(centerlines, snap=14.5µm)\n"
            "merged = merge_segments_at_junctions(centerlines, junctions)\n"
            "split = split_sharp_turns(merged, angle=70°)\n"
            "longitudinals = assign_longitudinals(split, dtip)\n"
            "crossveins = classify_crossveins_from_longitudinals(split)\n"
            "poly_veins = build_poly_veins_spatial(original_polygons, vein_map)\n"
            "poly_names = name_regions_from_veins(original_polygons, vein_map, poly_veins)\n"
            "poly_names = split_merged_polygons(polygons, poly_names)\n"
            "validation = cross_validate(assignments, poly_names, poly_veins)"
        ),
        params=[
            StepParam("snap_radius", "14.5 µm", "Max distance to cluster endpoints"),
        ],
        runs_computation=True,  # runs identify_veins_and_regions() (full)
    ),
    # 5: Merge Segments
    StepDef(
        index=5,
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
        runs_computation=False,  # cached from step 5
    ),
    # 6: Split Sharp Turns
    StepDef(
        index=6,
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
            "  for point along path (step=24.2µm):\n"
            "    angle_change = direction_at(point+step) - direction_at(point-step)\n"
            "    if angle_change > 70°:\n"
            "      split path at point"
        ),
        params=[
            StepParam("angle_threshold", "70°", "Direction change to trigger split"),
            StepParam("step_dist", "24.2 µm", "Window size for direction estimation"),
            StepParam("min_path_length", "241.5 µm", "Only split paths longer than this"),
            StepParam("min_split_length", "96.6 µm", "Both halves must exceed this"),
        ],
        runs_computation=False,  # cached from step 5
    ),
    # 7: Classify Longitudinals
    StepDef(
        index=7,
        name="Classify Longitudinals",
        short_name="Longitudinals",
        description=(
            "Assign L1-L5 identities to the remaining paths after L3/L4 anchoring. "
            "L3 is anchored via the DTip landmark (where L3 meets the distal wing tip). "
            "L4 is the next posterior vein. L2 is the longest anterior vein above L3, "
            "L1 is the most anterior short vein above L2, and L5 is the longest posterior "
            "vein below L4."
        ),
        pseudocode=(
            "# L3 anchored from DTip landmark\n"
            "anterior = [p for p in remaining if p.y < L3.y]\n"
            "posterior = [p for p in remaining if p.y > L4.y]\n"
            "L2 = longest(anterior)\n"
            "L1 = most_anterior_short(anterior - L2)\n"
            "L5 = longest(posterior)"
        ),
        params=[
            StepParam("y_position_weight", "0.30", "Weight for Y-position scoring"),
            StepParam("length_weight", "0.25", "Weight for length prior match"),
            StepParam("anchor", "DTip", "L3 anchored by DTip landmark proximity"),
        ],
        runs_computation=False,  # cached from step 5
    ),
    # 8: Classify Crossveins
    StepDef(
        index=8,
        name="Classify Crossveins",
        short_name="Crossveins",
        description=(
            "Identify ACV and PCV from the crossvein candidates using proximity to "
            "assigned longitudinals. ACV is scored by closeness to L3+L4, PCV by closeness "
            "to L4+L5. When multiple candidates exist, all pairings are evaluated and the "
            "best total score wins. Falls back to Y-sort if insufficient longitudinals."
        ),
        pseudocode=(
            "candidates = [p for p in paths\n"
            "  if p.orientation > 60° and p.length < 0.15 * wing_span]\n"
            "for each candidate:\n"
            "  acv_score = 1 - (dist_to_L3 + dist_to_L4) / (2 * 96.6µm)\n"
            "  pcv_score = 1 - (dist_to_L4 + dist_to_L5) / (2 * 96.6µm)\n"
            "best_pairing = argmax(acv_score[i] + pcv_score[j])"
        ),
        params=[
            StepParam("max_crossvein_len", "15% wing span", "Maximum crossvein length"),
            StepParam("orientation_cutoff", "60°", "Minimum orientation from horizontal"),
            StepParam("norm_dist", "96.6 µm", "Normalization distance for proximity scoring"),
        ],
        runs_computation=False,  # cached from step 5
    ),
    # 9: Name Regions
    StepDef(
        index=9,
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
        runs_computation=False,  # cached from step 5
    ),
    # 10: Vein-Extension Clipping
    StepDef(
        index=10,
        name="Vein-Extension Clipping",
        short_name="Vein Clip",
        description=(
            "Clip existing intervein regions using vein-extension boundaries. Extends vein "
            "lines to the wing outline to create ideal region boundaries, then intersects "
            "each original polygon with its matching vein-extension region. This only makes "
            "regions smaller — trimming areas that extend beyond the vein lines. Polygons "
            "and poly_names are updated for all downstream steps."
        ),
        pseudocode=(
            "ext_polys = partition_by_vein_extension(outline, vein_lines)\n"
            "ext_names = name_regions_from_veins(ext_polys, vein_map)\n"
            "ext_by_name = {name: union(ext_polys) for name}\n"
            "for i, name in poly_names.items():\n"
            "  polygons[i] = polygons[i].intersection(ext_by_name[name])"
        ),
        params=[
            StepParam("method", "vein_extension_clip", "Clip regions to vein boundaries"),
        ],
        runs_computation=True,
    ),
    # 11: Cross-Validation
    StepDef(
        index=11,
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
        runs_computation=False,  # cached from step 5
    ),
    # 12: Wing Outline
    StepDef(
        index=12,
        name="Build Wing Outline",
        short_name="Outline",
        description=(
            "Build the wing outline from the union of all polygons. Each intervein polygon "
            "is buffered by 9.7 µm to bridge the vein gaps, then the union is computed. Vein "
            "polygons are included with a smaller 2.4 µm buffer to extend the outline to the "
            "full wing tip. The outer boundary of the union becomes the wing outline."
        ),
        pseudocode=(
            "buffered = [poly.buffer(9.7µm) for poly in polygons]\n"
            "buffered += [vp.buffer(2.4µm) for vp in vein_polygons]\n"
            "union = unary_union(buffered)\n"
            "outline = largest_polygon(union).exterior"
        ),
        params=[
            StepParam("buffer_dist", "9.7 µm", "Buffer for intervein polygons"),
            StepParam("vein_buffer", "2.4 µm", "Buffer for vein polygons"),
        ],
        runs_computation=True,
    ),
    # 13: Hinge Detection & Removal
    StepDef(
        index=13,
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
            "cut_line = extend(LineString(subcostal, alula), 48.3µm)\n"
            "wing_blade = split(outline, cut_line).distal_piece"
        ),
        params=[
            StepParam("extend", "48.3 µm", "Cut line extension beyond outline"),
            StepParam("min_fragment", "5%", "Minimum fragment size to keep"),
        ],
        runs_computation=True,
    ),
    # 14: Compartments
    StepDef(
        index=14,
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
            "l4_extended = extend_to_boundary(l4_line, wing_blade, 241.5µm)\n"
            "l4_simplified = l4_extended.simplify(4.8µm)\n"
            "anterior, posterior = split(wing_blade, l4_simplified)"
        ),
        params=[
            StepParam("simplify", "4.8 µm", "Simplification tolerance for L4"),
            StepParam("extend", "241.5 µm", "Extension distance to reach outline"),
        ],
        runs_computation=True,
    ),
    # 15: Measurements
    StepDef(
        index=15,
        name="Compute Measurements",
        short_name="Measurements",
        description=(
            "Compute all wing measurements: per-vein lengths, crossvein distance, wing "
            "length and width, total wing area, per-region intervein areas, and anterior/"
            "posterior compartment areas. Scale calibration from the GUI µm/px input "
            "provides both pixel and micrometer columns."
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
    # 16: Final Overlays
    StepDef(
        index=16,
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
