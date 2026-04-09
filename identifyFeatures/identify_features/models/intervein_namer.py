"""Name intervein regions by adjacency to identified veins."""

from __future__ import annotations

import logging
from typing import Optional

from identify_features.config import PipelineConfig
from identify_features.models.datatypes import (
    InterveinRegion,
    Landmark,
    VeinIdentification,
    WingAxis,
)
from identify_features.models.topology import REGION_EXPECTED_VEINS, REGION_PD_PAIRS
from shapely.geometry import MultiPolygon, Polygon
from shapely.ops import unary_union

logger = logging.getLogger(__name__)

# Mergeable adjacent region pairs: (region_a, region_b, separator_vein)
# Derived from topology.VEIN_BOUNDARIES — every vein-separated region pair.
# Costal is omitted because the costal cell is removed in preprocessing.
_MERGEABLE_PAIRS: list[tuple[str, str, str]] = [
    ("1st basal", "1st posterior", "ACV"),
    ("discal", "2nd posterior", "PCV"),
    ("2nd posterior", "3rd posterior", "L5"),
    ("discal", "3rd posterior", "L5"),
    ("1st basal", "discal", "L4"),
    ("1st posterior", "2nd posterior", "L4"),
    ("discal", "1st posterior", "L4"),
    ("marginal", "submarginal", "L2"),
    ("marginal", "1st basal", "Rs"),
    ("submarginal", "1st basal", "L3"),
    ("submarginal", "1st posterior", "L3"),
]


def name_intervein_regions(
    intervein_polys: list[Polygon | MultiPolygon],
    veins: list[VeinIdentification],
    landmarks: dict[str, Landmark],
    config: PipelineConfig,
    median_vein_width_px: float = 0.0,
    wing_outline: Optional[Polygon] = None,
    wing_axis: Optional[WingAxis] = None,
) -> list[InterveinRegion]:
    """Name intervein regions by spatial adjacency to identified veins.

    For each intervein polygon, determines which vein centerlines border it,
    then finds the best-matching region from topology.REGION_EXPECTED_VEINS.

    Matching uses subset logic: a region matches if ALL of its expected veins
    are detected adjacent to the polygon. Extra detected veins (e.g., costa
    bleeding into anterior regions) are tolerated. When multiple regions match,
    the most specific one (largest expected set) wins.

    Args:
        intervein_polys: Unnamed intervein polygons from detection GeoJSON.
        veins: Identified veins with centerlines.
        landmarks: Anchored landmarks. Currently unused; reserved for future
            landmark-aware scoring.
        config: Pipeline configuration.
        median_vein_width_px: Median vein width in pixels (for fragment absorption).
        wing_outline: Wing outline polygon (currently unused).
        wing_axis: Wing proximal/distal axis. When provided, used to break
            ties between regions that share the same expected vein set
            (currently just discal vs 2nd posterior) by relative PD position.

    Returns:
        List of named InterveinRegion objects.
    """
    buffer_px = config.vein_buffer_px
    min_length = config.adjacency_min_length_px

    # Build buffered vein centerlines
    vein_buffers: dict[str, Polygon] = {}
    vein_map: dict[str, VeinIdentification] = {}
    for v in veins:
        if v.centerline is not None:
            vein_buffers[v.vein_id] = v.centerline.buffer(buffer_px)
            vein_map[v.vein_id] = v

    # Build effective expected veins — augment 3rd posterior with L6 when available
    effective_expected: dict[str, set[str]] = {k: set(v) for k, v in REGION_EXPECTED_VEINS.items()}
    if "L6" in vein_map:
        effective_expected["3rd posterior"] = {"L5", "L6"}
        logger.info("L6 detected — using {L5, L6} for 3rd posterior identification")

    results: list[InterveinRegion] = []
    all_polys: list[Polygon] = []  # All processed polygons (for fragment absorption)
    # Polygons whose top-specificity match is a tie; resolved after the
    # main loop using the wing PD axis.
    deferred_ties: list[tuple[int, Polygon, frozenset[str], frozenset[str]]] = []

    for i, poly in enumerate(intervein_polys):
        # Handle MultiPolygon — use largest sub-polygon
        if isinstance(poly, MultiPolygon):
            poly = max(poly.geoms, key=lambda g: g.area)

        if poly.is_empty:
            continue

        all_polys.append(poly)

        # Find adjacent veins by intersection length with polygon boundary
        boundary = poly.boundary
        adjacent_veins: set[str] = set()
        for vein_id, buf in vein_buffers.items():
            try:
                intersection = boundary.intersection(buf)
                if not intersection.is_empty and intersection.length >= min_length:
                    adjacent_veins.add(vein_id)
            except Exception:
                continue

        if not adjacent_veins:
            logger.debug("Polygon %d: no adjacent veins found", i)
            continue

        detected = frozenset(adjacent_veins)

        # Find matching regions: expected veins must be a SUBSET of detected veins
        candidates: list[tuple[str, int]] = []  # (region_name, specificity)
        for region_name, expected_veins in effective_expected.items():
            expected = frozenset(expected_veins)
            if expected <= detected:  # subset check
                candidates.append((region_name, len(expected)))

        name: Optional[str] = None
        status = "identified"

        if candidates:
            # Pick the most specific match (largest expected set)
            candidates.sort(key=lambda c: -c[1])
            max_specificity = candidates[0][1]
            top = [c for c in candidates if c[1] == max_specificity]

            if len(top) == 1:
                name = top[0][0]
            else:
                # Tie at top specificity — defer and resolve later using PD axis.
                tied_names = frozenset(c[0] for c in top)
                deferred_ties.append((i, poly, detected, tied_names))
                continue

        if name is None:
            # Try merged region detection
            name, status = _check_merged(detected, vein_map, effective_expected)

        if name is None:
            # Coverage fallback: what fraction of expected veins are present?
            name, status = _coverage_fallback(detected, effective_expected)

        if name is None:
            logger.debug("Polygon %d: unmatched vein set %s", i, adjacent_veins)
            continue

        region = InterveinRegion(
            name=name,
            polygon=poly,
            bounding_veins=adjacent_veins,
            area_px2=poly.area,
            status=status,
        )
        results.append(region)
        logger.info(
            "Region %r: %.0fpx² (veins: %s, status: %s)",
            name,
            poly.area,
            adjacent_veins,
            status,
        )

    # Resolve deferred ties with the PD axis where possible.
    unresolved_ties = _resolve_pd_ties(deferred_ties, results, wing_axis)

    # Anything the PD resolver couldn't handle (no axis, no matching PD pair,
    # or a lone polygon spanning both regions) falls through to the old
    # merge/coverage fallback path.
    for _, poly, detected, _tied in unresolved_ties:
        name, status = _check_merged(detected, vein_map, effective_expected)
        if name is None:
            name, status = _coverage_fallback(detected, effective_expected)
        if name is None:
            logger.debug("Deferred tie polygon unmatched: veins=%s", set(detected))
            continue
        region = InterveinRegion(
            name=name,
            polygon=poly,
            bounding_veins=set(detected),
            area_px2=poly.area,
            status=status,
        )
        results.append(region)
        logger.info(
            "Region %r: %.0fpx² (veins: %s, status: %s)",
            name,
            poly.area,
            set(detected),
            status,
        )

    # Post-naming: detect merged regions where a higher-specificity match
    # absorbed a neighboring region (e.g. "discal" absorbing 3rd posterior).
    _detect_absorbed_merges(results, effective_expected)

    # Absorb small fragments split off by ectopic (unlabeled) veins.
    # Buffer by 2× median vein width to bridge the ectopic vein gap.
    frag_buffer = median_vein_width_px * 2 if median_vein_width_px > 0 else buffer_px * 2
    _absorb_ectopic_fragments(results, all_polys, frag_buffer)

    return results


def _resolve_pd_ties(
    deferred: list[tuple[int, Polygon, frozenset[str], frozenset[str]]],
    results: list[InterveinRegion],
    wing_axis: Optional[WingAxis],
) -> list[tuple[int, Polygon, frozenset[str], frozenset[str]]]:
    """Resolve tied region candidates using the wing PD axis.

    Groups deferred polygons by their tied candidate set, matches each group
    against topology.REGION_PD_PAIRS (an ordered (proximal, distal) pair per
    tie), and assigns names by sorting the group's polygons along the PD axis.

    Entries returned are those the resolver could not handle:
    - wing_axis is None
    - no matching PD pair for the tied candidate set
    - only a single polygon in the group (the two regions likely fused)
    - middle polygons in a group of 3+
    Callers should send these down the existing merge/coverage fallback path.
    """
    if not deferred or wing_axis is None:
        return list(deferred)

    # Group by tied candidate set
    groups: dict[frozenset[str], list[tuple[int, Polygon, frozenset[str]]]] = {}
    for poly_idx, poly, detected, tied in deferred:
        groups.setdefault(tied, []).append((poly_idx, poly, detected))

    unresolved: list[tuple[int, Polygon, frozenset[str], frozenset[str]]] = []

    for tied, members in groups.items():
        # Find the matching PD pair in topology
        pd_pair: Optional[tuple[str, str]] = None
        for proximal_name, distal_name in REGION_PD_PAIRS:
            if tied == frozenset({proximal_name, distal_name}):
                pd_pair = (proximal_name, distal_name)
                break

        if pd_pair is None or len(members) < 2:
            for poly_idx, poly, detected in members:
                unresolved.append((poly_idx, poly, detected, tied))
            continue

        # Sort group members by PD coordinate of centroid (proximal first)
        with_pd = sorted(
            ((wing_axis.project(poly.centroid), poly_idx, poly, detected) for poly_idx, poly, detected in members),
            key=lambda x: x[0],
        )

        proximal_name, distal_name = pd_pair
        # Smallest PD → proximal, largest PD → distal, middle → unresolved
        assign_names: list[Optional[str]] = [None] * len(with_pd)
        assign_names[0] = proximal_name
        assign_names[-1] = distal_name

        for (pd_coord, poly_idx, poly, detected), name in zip(with_pd, assign_names):
            if name is None:
                unresolved.append((poly_idx, poly, detected, tied))
                continue
            region = InterveinRegion(
                name=name,
                polygon=poly,
                bounding_veins=set(detected),
                area_px2=poly.area,
                status="identified",
            )
            results.append(region)
            logger.info(
                "PD tie resolver: %r (PD=%.2f, veins=%s)",
                name,
                pd_coord,
                set(detected),
            )

    return unresolved


def _absorb_ectopic_fragments(
    results: list[InterveinRegion],
    all_polys: list[Polygon],
    gap_buffer_px: float,
) -> None:
    """Absorb small fragments created by ectopic (unlabeled) veins.

    Ectopic veins split what should be a single region into a large polygon
    and one or more small fragments. For each small fragment, we find which
    named region shares the most boundary with it (i.e., they're on opposite
    sides of the ectopic vein) and absorb the fragment into that region.

    Small fragments may be named (possibly incorrectly) or unnamed (skipped
    during naming). Both cases are handled: named fragments below the size
    threshold are removed from results and absorbed into their neighbor.

    Args:
        gap_buffer_px: Buffer distance to bridge the ectopic vein gap
                       (typically 2× median vein width).
    """
    if not results:
        return

    # Size threshold: fragments below 5% of the median named region area
    areas = sorted(r.area_px2 for r in results)
    median_area = areas[len(areas) // 2]
    fragment_threshold = median_area * 0.05

    # Collect named polygons into a set for fast lookup
    named_polys = {id(r.polygon) for r in results}

    # Find all small polygons — both unnamed (not in results) and named fragments
    unnamed_fragments: list[Polygon] = []
    for poly in all_polys:
        if id(poly) not in named_polys:
            unnamed_fragments.append(poly)

    named_fragments: list[InterveinRegion] = [r for r in results if r.area_px2 < fragment_threshold]

    all_fragments = [(p, None) for p in unnamed_fragments] + [(r.polygon, r) for r in named_fragments]

    if not all_fragments:
        return

    # Remove named fragments from results — they'll be absorbed
    for _, region in all_fragments:
        if region is not None and region in results:
            results.remove(region)

    # For each fragment, find the named region sharing the most boundary
    for frag_poly, frag_region in all_fragments:
        if frag_poly is None or frag_poly.is_empty:
            continue

        buffered_frag = frag_poly.buffer(gap_buffer_px)
        best_result: Optional[InterveinRegion] = None
        best_length = 0.0

        for r in results:
            if r.polygon is None:
                continue
            try:
                # Intersect named polygon's boundary with the buffered fragment
                # (as a filled polygon, not its boundary) to get the shared edge length
                intersection = r.polygon.boundary.intersection(buffered_frag)
                if not intersection.is_empty and intersection.length > best_length:
                    best_length = intersection.length
                    best_result = r
            except Exception:
                continue

        if best_result is not None:
            # Absorb: union the fragment into the neighbor's polygon
            merged = unary_union([best_result.polygon, frag_poly])
            if isinstance(merged, MultiPolygon):
                merged = max(merged.geoms, key=lambda g: g.area)
            best_result.polygon = merged
            best_result.area_px2 = merged.area
            logger.info(
                "Absorbed fragment (%.0fpx²) into %r (shared boundary %.0fpx)",
                frag_poly.area,
                best_result.name,
                best_length,
            )


def _detect_absorbed_merges(
    results: list[InterveinRegion],
    effective_expected: dict[str, set[str]],
) -> None:
    """Detect regions absorbed into a higher-specificity neighbor.

    When a polygon matches a high-specificity region (e.g. discal with 3 veins)
    but also contains a lower-specificity neighbor's veins (e.g. 3rd posterior
    with 2 veins), the neighbor is missing from results. This pass detects
    such absorptions by checking if any named region's bounding veins are a
    superset of a missing region's expected veins, constrained to valid
    adjacent pairs from _MERGEABLE_PAIRS.
    """
    # Collect all region names already found (including parts of merged names)
    found_names: set[str] = set()
    for r in results:
        for part in r.name.split(" + "):
            found_names.add(part)

    missing = set(effective_expected.keys()) - found_names
    if not missing:
        return

    # Build adjacency lookup from _MERGEABLE_PAIRS
    merge_partners: dict[str, set[str]] = {}
    for a, b, _sep in _MERGEABLE_PAIRS:
        merge_partners.setdefault(a, set()).add(b)
        merge_partners.setdefault(b, set()).add(a)

    for missing_name in sorted(missing):
        missing_veins = frozenset(effective_expected.get(missing_name, set()))
        if not missing_veins:
            continue

        valid_partners = merge_partners.get(missing_name, set())

        for r in results:
            # Check if this result is a valid merge partner
            result_parts = set(r.name.split(" + "))
            if not result_parts & valid_partners:
                continue

            # Check if the missing region's expected veins are all adjacent
            if missing_veins <= r.bounding_veins:
                r.name = f"{r.name} + {missing_name}"
                r.status = "merged"
                logger.info(
                    "%s veins %s found in %r → merged",
                    missing_name,
                    missing_veins,
                    r.name,
                )
                # Update found_names so we don't double-merge
                found_names.add(missing_name)
                break


def _check_merged(
    detected: frozenset[str],
    vein_map: dict[str, VeinIdentification],
    effective_expected: dict[str, set[str]],
) -> tuple[Optional[str], str]:
    """Check if a vein set matches a merged region pair.

    A merge occurs when two adjacent regions fuse because the separating
    vein is absent or partial. The detected set should match the union of
    both regions' expected sets minus the separator.

    Uses subset logic: the merged expected set must be a subset of detected.
    """
    best_name: Optional[str] = None
    best_specificity = 0

    for region_a, region_b, separator in _MERGEABLE_PAIRS:
        expected_a = frozenset(effective_expected.get(region_a, set()))
        expected_b = frozenset(effective_expected.get(region_b, set()))

        # Merged expected set = union minus separator
        merged_expected = (expected_a | expected_b) - {separator}

        if merged_expected <= detected:  # subset check
            specificity = len(merged_expected)
            if specificity > best_specificity:
                best_specificity = specificity

                if separator in vein_map:
                    reason = f"partial {separator}"
                else:
                    reason = f"absent {separator}"

                best_name = f"{region_a} + {region_b}"
                logger.info("Merged region detected: %s (%s)", best_name, reason)

    if best_name is not None:
        return best_name, "merged"
    return None, ""


def _coverage_fallback(
    detected: frozenset[str],
    effective_expected: dict[str, set[str]],
) -> tuple[Optional[str], str]:
    """Score by coverage: fraction of expected veins present in detected set.

    Returns (name, "inferred") if best coverage >= 0.5, else (None, "").
    """
    best_name = None
    best_score = 0.0

    for region_name, expected_veins in effective_expected.items():
        expected = frozenset(expected_veins)
        if not expected:
            continue
        coverage = len(detected & expected) / len(expected)
        if coverage > best_score:
            best_score = coverage
            best_name = region_name

    if best_score >= 0.5 and best_name is not None:
        logger.info("Coverage fallback: %s (score=%.2f)", best_name, best_score)
        return best_name, "inferred"

    return None, ""
