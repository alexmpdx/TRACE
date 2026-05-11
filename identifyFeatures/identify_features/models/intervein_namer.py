"""Name intervein regions by adjacency to identified veins."""

from __future__ import annotations

import logging
from itertools import combinations
from typing import Optional

from identify_features.config import PipelineConfig
from identify_features.models.datatypes import (
    InterveinRegion,
    Landmark,
    VeinIdentification,
    WingAxis,
)
from identify_features.models.topology import (
    REGION_AP_ORDER,
    REGION_EXPECTED_VEINS,
    REGION_PD_PAIRS,
    VEIN_BOUNDARIES,
    build_region_forbidden_veins,
)
from shapely.geometry import MultiPolygon, Polygon
from shapely.ops import unary_union

logger = logging.getLogger(__name__)


def _build_region_adjacency() -> tuple[
    dict[str, dict[str, str]],
    dict[frozenset[str], str],
]:
    """Derive the region adjacency graph from topology.VEIN_BOUNDARIES.

    Returns (neighbors, edge_separator):
        - neighbors[r][n] = vein that separates region r from neighbor n
        - edge_separator[frozenset({r, n})] = same, keyed by unordered pair
    """
    neighbors: dict[str, dict[str, str]] = {}
    edge_sep: dict[frozenset[str], str] = {}
    for vein, pairs in VEIN_BOUNDARIES.items():
        for a, b in pairs:
            neighbors.setdefault(a, {})[b] = vein
            neighbors.setdefault(b, {})[a] = vein
            edge_sep[frozenset({a, b})] = vein
    return neighbors, edge_sep


_REGION_ADJACENCY, _REGION_EDGE_SEPARATOR = _build_region_adjacency()


def _claimed_single_names(results: list[InterveinRegion]) -> frozenset[str]:
    """Names already assigned as canonical single-region polygons.

    Includes both ``identified`` (subset-match) and ``inferred``
    (coverage-fallback) statuses — once a name is on a polygon, it
    shouldn't be re-used by a later polygon's merge or coverage path.
    Compound merged names (containing " + ") are excluded since their
    sub-region names remain available.
    """
    return frozenset(r.name for r in results if " + " not in r.name and r.status in ("identified", "inferred"))


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
    buffer_px = median_vein_width_px * config.vein_buffer_vw
    min_length = median_vein_width_px * config.adjacency_min_length_vw

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
    # Polygons that fell through to coverage_fallback in the main loop are
    # deferred until ALL canonical / merged matches have been claimed —
    # this prevents a low-confidence inferred name from stealing a region
    # name that a later high-confidence canonical match would have wanted
    # (BDSC..._021126_0011: a 60k sliver was getting "1st posterior" via
    # coverage and a 940k polygon got "1st basal" via coverage, then the
    # genuine 70k 1st basal also primary-matched "1st basal" for a duplicate).
    deferred_coverage: list[tuple[Polygon, frozenset[str], set[str]]] = []

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
            # Try merged region detection. Exclude regions already assigned
            # as canonical singles in earlier loop iterations — merging into
            # them would double-count those polygons.
            claimed_singles = _claimed_single_names(results)
            name, status = _check_merged(
                detected,
                vein_map,
                effective_expected,
                config.max_merge_size,
                claimed_names=claimed_singles,
            )

        if name is None:
            # Defer coverage-fallback assignment to the last pass so this
            # polygon can't claim a name a later canonical-subset match
            # would want.
            deferred_coverage.append((poly, detected, set(adjacent_veins)))
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
    # merge/coverage fallback path. The polygon's tied initial-match set
    # constrains valid merge candidates: the merge must include every region
    # that matched best in single-region scoring.
    claimed_singles = _claimed_single_names(results)
    for _, poly, detected, _tied in unresolved_ties:
        name, status = _check_merged(
            detected,
            vein_map,
            effective_expected,
            config.max_merge_size,
            claimed_names=claimed_singles,
            required_names=_tied,
        )
        if name is None:
            name, status = _coverage_fallback(
                detected, effective_expected, claimed_names=_claimed_single_names(results)
            )
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

    # Second pass: coverage-fallback for polygons that found no canonical
    # subset match in the main loop. Runs after canonical and merged
    # regions have all been claimed, so an inferred name can't squat on
    # a region a canonical match was going to want. Each fallback
    # excludes the current claimed-singles set, so successive deferrals
    # see the latest claims.
    #
    # Process largest polygons first so anatomically-real regions claim
    # their names before any small distal slivers do. On
    # BDSC..._021126_0011, processing in input order had a 63k distal
    # submarginal sliver claiming "1st posterior" via coverage before
    # the genuine 940k 1st posterior polygon could; with size-DESC order
    # the 940k polygon claims first and the sliver falls through to
    # _absorb_ectopic_fragments, where it merges back into submarginal.
    deferred_coverage.sort(key=lambda entry: entry[0].area, reverse=True)
    for poly, detected, adjacent_veins in deferred_coverage:
        claimed = _claimed_single_names(results)
        # Try merged region detection first — produces a higher-quality
        # compound name than coverage when applicable.
        name, status = _check_merged(
            detected,
            vein_map,
            effective_expected,
            config.max_merge_size,
            claimed_names=claimed,
        )
        if name is None:
            name, status = _coverage_fallback(detected, effective_expected, claimed_names=claimed)
        if name is None:
            logger.debug("Deferred coverage polygon unmatched: veins=%s", adjacent_veins)
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

    # Post-naming: detect merged regions where a higher-specificity match
    # absorbed a neighboring region (e.g. "discal" absorbing 3rd posterior).
    _detect_absorbed_merges(results, effective_expected)

    # Absorb small fragments split off by ectopic (unlabeled) veins.
    # Buffer by 2× median vein width to bridge the ectopic vein gap.
    frag_buffer = median_vein_width_px * 2 if median_vein_width_px > 0 else buffer_px * 2
    _absorb_ectopic_fragments(results, all_polys, frag_buffer)

    # Final pass: any polygon still unnamed at this point is typically a
    # piece of an adjacent region cut off by a slide defect (or a
    # fragment too large to be absorbed by _absorb_ectopic_fragments's
    # size cap). Re-run the naming logic on each leftover with a
    # preference for the name of the adjacent named region with the
    # longest shared boundary, then merge the leftover into that
    # neighbor's polygon via unary_union.
    _name_leftover_regions(
        results,
        all_polys,
        vein_buffers,
        vein_map,
        effective_expected,
        config,
        min_length,
        frag_buffer,
    )

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

    # Find all small polygons — both unnamed (not in results) and named fragments.
    # Apply the same fragment_threshold to unnamed polygons so a large orphan
    # (e.g. a real region the namer couldn't name because all candidate names
    # were claimed by other polygons) isn't silently merged into a neighbor.
    # On BDSC..._021126_0011 the unnamed 940k 1st-posterior polygon was being
    # absorbed into 2nd posterior, which then dropped the genuine 914k
    # 2nd posterior via the unary_union → max(area) collapse.
    unnamed_fragments: list[Polygon] = []
    for poly in all_polys:
        if id(poly) not in named_polys and poly.area < fragment_threshold:
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


def _name_leftover_regions(
    results: list[InterveinRegion],
    all_polys: list[Polygon],
    vein_buffers: dict[str, Polygon],
    vein_map: dict[str, "VeinIdentification"],
    effective_expected: dict[str, set[str]],
    config: PipelineConfig,
    min_length: float,
    gap_buffer_px: float,
) -> None:
    """Final naming pass for polygons still unnamed after all earlier
    passes. Typically these are pieces of an adjacent region split off
    by an external slide defect (or fragments too large to be absorbed
    by ``_absorb_ectopic_fragments``'s size cap).

    Strategy: for each leftover polygon
      1. compute its adjacent_veins via the same boundary-buffer test
         used in the main loop;
      2. find named neighbors by intersecting the leftover (buffered by
         ``gap_buffer_px``) with each named region's boundary, scoring
         by shared-boundary length;
      3. run canonical → merged → coverage naming on the leftover, but
         prefer any candidate whose name matches a named neighbor;
      4. if no candidate matches a neighbor, default to the
         longest-shared-boundary neighbor's name;
      5. merge the leftover into the chosen neighbor's polygon via
         ``unary_union`` (the neighbor's polygon becomes a MultiPolygon
         if the union spans a thin vein-tissue gap).

    No claimed_names exclusion is applied — this is the final naming
    chance and the candidate space is already constrained by the
    "must match a neighbor" preference.
    """
    if not results:
        return

    named_polys = {id(r.polygon) for r in results}
    leftovers = [p for p in all_polys if id(p) not in named_polys and not p.is_empty]
    if not leftovers:
        return

    for poly in leftovers:
        # Adjacent veins (same logic as main loop)
        boundary = poly.boundary
        adjacent_veins: set[str] = set()
        for vein_id, buf in vein_buffers.items():
            try:
                inter = boundary.intersection(buf)
                if not inter.is_empty and inter.length >= min_length:
                    adjacent_veins.add(vein_id)
            except Exception:
                continue
        detected = frozenset(adjacent_veins)

        # Named neighbors by shared-boundary length (buffered to bridge
        # any thin vein-tissue gap from segmentation).
        buffered = poly.buffer(gap_buffer_px)
        neighbor_shared: list[tuple[InterveinRegion, float]] = []
        for r in results:
            if r.polygon is None or r.polygon.is_empty:
                continue
            try:
                shared = r.polygon.boundary.intersection(buffered).length
            except Exception:
                continue
            if shared > 0:
                neighbor_shared.append((r, shared))
        if not neighbor_shared:
            continue  # orphan island; nothing to absorb into
        neighbor_shared.sort(key=lambda nb: -nb[1])
        neighbor_names = {nb[0].name for nb in neighbor_shared}

        # Try canonical subset match, preferring a name that matches a
        # neighbor (in most-specific-first order).
        chosen_name: Optional[str] = None
        canonical_candidates = sorted(
            (
                (region_name, len(expected_veins))
                for region_name, expected_veins in effective_expected.items()
                if frozenset(expected_veins) <= detected
            ),
            key=lambda c: -c[1],
        )
        for cand_name, _ in canonical_candidates:
            if cand_name in neighbor_names:
                chosen_name = cand_name
                break

        if chosen_name is None:
            mname, _ = _check_merged(detected, vein_map, effective_expected, config.max_merge_size)
            if mname in neighbor_names:
                chosen_name = mname

        if chosen_name is None:
            cname, _ = _coverage_fallback(detected, effective_expected)
            if cname in neighbor_names:
                chosen_name = cname

        if chosen_name is None:
            # Last resort: longest-shared-boundary neighbor's name.
            chosen_name = neighbor_shared[0][0].name
            source = "longest-shared-boundary"
        else:
            source = "naming-pipeline (matched neighbor)"

        # Merge leftover into the chosen-name neighbor with the longest
        # shared boundary (when multiple same-name neighbors exist).
        target = next(nb[0] for nb in neighbor_shared if nb[0].name == chosen_name)
        merged = unary_union([target.polygon, poly])
        target.polygon = merged
        target.area_px2 = merged.area
        logger.info(
            "Leftover polygon (%dpx²) absorbed into %r [%s], shared boundary %.0fpx",
            int(poly.area),
            chosen_name,
            source,
            neighbor_shared[0][1],
        )


def _detect_absorbed_merges(
    results: list[InterveinRegion],
    effective_expected: dict[str, set[str]],
) -> None:
    """Split duplicate-name polygons via forbidden-adjacency, falling back
    to append-style merge naming when no split signal is available.

    Runs as a post-pass after all first-tier naming, tie resolution, and
    per-polygon _check_merged() calls. Modifies ``results`` in place.
    """
    found_names: set[str] = set()
    for r in results:
        for part in r.name.split(" + "):
            found_names.add(part)

    missing = set(effective_expected.keys()) - found_names
    if not missing:
        return

    # --- Phase A: forbidden-adjacency split of duplicate names ---
    forbidden = build_region_forbidden_veins(effective_expected)
    changed = _split_duplicates_by_forbidden(results, missing, effective_expected, forbidden)

    # Recompute found/missing after any splits
    if changed:
        found_names = set()
        for r in results:
            for part in r.name.split(" + "):
                found_names.add(part)
        missing = set(effective_expected.keys()) - found_names
        if not missing:
            return

    # --- Phase B: legacy append-style fallback for anything still missing ---
    _append_missing_to_neighbor(results, missing, effective_expected)

    # Log final state
    found_final: set[str] = set()
    for r in results:
        for part in r.name.split(" + "):
            found_final.add(part)
    still_missing = set(effective_expected.keys()) - found_final
    if still_missing:
        logger.info("Unresolved missing regions: %s", sorted(still_missing))


def _split_duplicates_by_forbidden(
    results: list[InterveinRegion],
    missing: set[str],
    effective_expected: dict[str, set[str]],
    forbidden: dict[str, set[str]],
) -> bool:
    """For each duplicated canonical name, try to rename one of the
    duplicates to a missing region via forbidden-adjacency.

    A duplicate polygon is eligible for renaming if its bounding_veins
    contains at least one vein that is forbidden for the name it currently
    holds. The target missing region is chosen by best-matching expected
    set against bounding_veins.

    Returns True if any result was renamed.
    """
    any_change = False

    # Build {canonical_name: [indices of results whose parsed parts contain it]}
    dup_groups: dict[str, list[int]] = {}
    for idx, r in enumerate(results):
        for part in r.name.split(" + "):
            dup_groups.setdefault(part, []).append(idx)

    for dup_name, indices in list(dup_groups.items()):
        if len(indices) < 2:
            continue
        if dup_name not in forbidden:
            continue

        forbidden_here = forbidden[dup_name]

        # Candidate indices: polygons whose bounding_veins trip a forbidden vein
        eligible = [i for i in indices if results[i].bounding_veins & forbidden_here]
        if not eligible:
            continue

        for i in eligible:
            poly_veins = results[i].bounding_veins
            target = _best_missing_match(poly_veins, missing, effective_expected)
            if target is None:
                continue

            old_name = results[i].name
            parts = [p for p in old_name.split(" + ") if p != dup_name]
            parts.append(target)
            results[i].name = " + ".join(parts) if len(parts) > 1 else parts[0]
            results[i].status = "identified" if len(parts) == 1 else "merged"
            missing.discard(target)
            any_change = True
            logger.info(
                "Split duplicate %r → %r (forbidden veins: %s)",
                old_name,
                results[i].name,
                sorted(poly_veins & forbidden_here),
            )
            break

    return any_change


def _best_missing_match(
    poly_veins: set[str],
    missing: set[str],
    effective_expected: dict[str, set[str]],
) -> str | None:
    """Return the missing region whose expected set best matches poly_veins.

    Ranking:
      1. Largest |expected ∩ poly_veins| (most of its veins present)
      2. Largest |expected|              (most specific wins ties)
      3. REGION_AP_ORDER index           (deterministic final tie-break)
    Returns None if no missing region has any overlap with poly_veins.
    """
    best: tuple[int, int, int] | None = None
    best_name: str | None = None
    for name in missing:
        expected = effective_expected.get(name, set())
        if not expected:
            continue
        overlap = len(expected & poly_veins)
        if overlap == 0:
            continue
        specificity = len(expected)
        ap_idx = -REGION_AP_ORDER.index(name) if name in REGION_AP_ORDER else -999
        score = (overlap, specificity, ap_idx)
        if best is None or score > best:
            best = score
            best_name = name
    return best_name


def _append_missing_to_neighbor(
    results: list[InterveinRegion],
    missing: set[str],
    effective_expected: dict[str, set[str]],
) -> None:
    """Legacy fallback: append missing region name to an adjacent neighbor.

    When a region is missing from results and one of the existing named
    regions has all of the missing region's expected veins in its adjacency,
    append the missing name to produce a merged label (e.g. "discal + 3rd
    posterior"). Only valid adjacency-graph neighbors are considered.
    """
    merge_partners: dict[str, set[str]] = {r: set(nbrs.keys()) for r, nbrs in _REGION_ADJACENCY.items()}

    found_names: set[str] = set()
    for r in results:
        for part in r.name.split(" + "):
            found_names.add(part)

    for missing_name in sorted(missing):
        missing_veins = frozenset(effective_expected.get(missing_name, set()))
        if not missing_veins:
            continue

        valid_partners = merge_partners.get(missing_name, set())

        for r in results:
            result_parts = set(r.name.split(" + "))
            if not result_parts & valid_partners:
                continue

            if missing_veins <= r.bounding_veins:
                r.name = f"{r.name} + {missing_name}"
                r.status = "merged"
                logger.info(
                    "%s veins %s found in %r → merged",
                    missing_name,
                    missing_veins,
                    r.name,
                )
                found_names.add(missing_name)
                break


def _is_connected(nodes: set[str], adjacency: dict[str, dict[str, str]]) -> bool:
    """Return True if `nodes` forms a connected subgraph under `adjacency`."""
    if len(nodes) <= 1:
        return True
    start = next(iter(nodes))
    seen = {start}
    stack = [start]
    while stack:
        cur = stack.pop()
        for nb in adjacency.get(cur, {}):
            if nb in nodes and nb not in seen:
                seen.add(nb)
                stack.append(nb)
    return seen == nodes


def _enumerate_merge_candidates(
    detected: frozenset[str],
    effective_expected: dict[str, set[str]],
    max_merge_size: Optional[int] = None,
    claimed_names: frozenset[str] = frozenset(),
    required_names: frozenset[str] = frozenset(),
) -> Optional[tuple[tuple[str, ...], frozenset[str], set[str]]]:
    """Find the best connected subset of regions matching a detected vein set.

    Enumerates all connected subgraphs of the region adjacency graph of size
    2..N (where N = ``max_merge_size`` or the region count), computes each
    candidate's merged expected veins (union of per-region expected minus
    internal separators), and returns the best match whose merged expected
    set is a subset of ``detected``.

    Scoring (descending):
        1. ``len(merged_expected)`` — prefer the most specific match (largest
           set of bounding veins the merge "explains").
        2. ``-subset_size`` — among equal-specificity matches, prefer the
           smallest merge (avoid absorbing extra regions we don't need).
        3. AP-ordered region tuple — deterministic final tie-break.

    ``claimed_names`` excludes regions already assigned as canonical
    single-name polygons elsewhere in ``results`` — a merger that absorbs
    them would double-count those polygons. ``required_names`` constrains
    candidates to those that *contain* the polygon's best initial single-
    region matches (the tied set that fell through to merge resolution);
    the merge interpretation must explain why every tied region matched.

    Returns ``(ap_ordered_regions, merged_expected, internal_separators)`` or
    ``None`` if no connected subset matches.
    """
    all_regions = [r for r in REGION_AP_ORDER if r in effective_expected and r not in claimed_names]
    if not all_regions:
        return None
    upper = max_merge_size if max_merge_size is not None else len(all_regions)
    upper = min(upper, len(all_regions))
    if upper < 2:
        return None

    best: Optional[tuple[tuple[str, ...], frozenset[str], set[str]]] = None
    best_score: Optional[tuple[int, int, tuple[str, ...]]] = None

    for size in range(2, upper + 1):
        for combo in combinations(all_regions, size):
            subset = set(combo)
            if required_names and not required_names <= subset:
                continue
            if not _is_connected(subset, _REGION_ADJACENCY):
                continue
            # Internal separators = veins bounding pairs both inside the subset
            internal_seps: set[str] = set()
            for r1, r2 in combinations(subset, 2):
                sep = _REGION_EDGE_SEPARATOR.get(frozenset({r1, r2}))
                if sep is not None:
                    internal_seps.add(sep)
            merged: set[str] = set()
            for r in subset:
                merged |= effective_expected.get(r, set())
            merged -= internal_seps
            if not merged <= detected:
                continue
            ap_tuple = tuple(r for r in REGION_AP_ORDER if r in subset)
            score = (len(merged), -len(subset), ap_tuple)
            if best_score is None or score > best_score:
                best_score = score
                best = (ap_tuple, frozenset(merged), internal_seps)

    return best


def _check_merged(
    detected: frozenset[str],
    vein_map: dict[str, VeinIdentification],
    effective_expected: dict[str, set[str]],
    max_merge_size: Optional[int] = None,
    claimed_names: frozenset[str] = frozenset(),
    required_names: frozenset[str] = frozenset(),
) -> tuple[Optional[str], str]:
    """Detect an N-way merged region matching the given vein set.

    Delegates enumeration to ``_enumerate_merge_candidates``. A merge occurs
    when two or more adjacent regions fuse because one or more separator
    veins along the chain are absent or partial. Any connected subset of
    the region adjacency graph is a valid candidate; the largest/highest-
    specificity match wins.

    ``claimed_names`` and ``required_names`` are forwarded to the enumerator
    to exclude regions already assigned elsewhere and to constrain
    candidates to those containing the tied initial-match regions.
    """
    result = _enumerate_merge_candidates(
        detected,
        effective_expected,
        max_merge_size=max_merge_size,
        claimed_names=claimed_names,
        required_names=required_names,
    )
    if result is None:
        return None, ""

    regions, _merged_expected, internal_seps = result
    reasons = []
    for sep in sorted(internal_seps):
        tag = "partial" if sep in vein_map else "absent"
        reasons.append(f"{tag} {sep}")
    reason = ", ".join(reasons) if reasons else "no internal separators"

    name = " + ".join(regions)
    logger.info("Merged region detected: %s (%s)", name, reason)
    return name, "merged"


def _coverage_fallback(
    detected: frozenset[str],
    effective_expected: dict[str, set[str]],
    claimed_names: frozenset[str] = frozenset(),
) -> tuple[Optional[str], str]:
    """Score by coverage: fraction of expected veins present in detected set.

    Returns (name, "inferred") if best coverage >= 0.5, else (None, "").

    ``claimed_names`` excludes regions already named as canonical singles
    so the inferred-fallback never duplicates an existing name (e.g. on
    BDSC..._021126_0011 a polygon whose adjacent_veins didn't include
    costa was getting "1st basal" via the fallback even though another
    polygon had already been named "1st basal" — the real intent was
    "1st posterior" once 1st basal was off the table).
    """
    best_name = None
    best_score = 0.0

    for region_name, expected_veins in effective_expected.items():
        if region_name in claimed_names:
            continue
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
