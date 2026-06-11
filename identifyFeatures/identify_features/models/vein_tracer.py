"""Identify veins by assigning labels to graph edges.

The skeleton graph may be fragmented (multiple disconnected components).
Labels are assigned in three phases:

1. Landmark-based: edges containing landmark nodes get labeled by
   departure direction and landmark identity.
2. Spatial assignment: unlabeled edges are matched to the nearest
   compatible labeled vein by proximity and direction.
3. Junction resolution: at degree-3+ nodes, tangent continuity
   determines which vein continues through.
"""

from __future__ import annotations

import logging
import math
from collections import defaultdict
from pathlib import Path
from typing import Optional

import cv2
import networkx as nx
import numpy as np
from identify_features.config import PipelineConfig
from identify_features.models.datatypes import (
    Landmark,
    SkeletonGraph,
    VeinIdentification,
    VeinStatus,
    VeinType,
    WingAxis,
)
from identify_features.models.topology import (
    ALL_CANONICAL_VEINS,
    CROSSVEIN_CONNECTIONS,
    LONGITUDINAL_ENDPOINTS,
    VEIN_COLORS,
)
from identify_features.utils.geometry_utils import (
    angle_between_vectors,
    direction_toward,
)
from identify_features.utils.graph_utils import (
    edge_departure_direction,
    edge_line_from_node,
)
from shapely.geometry import LineString, Point

logger = logging.getLogger(__name__)


class _TracerDumper:
    """Save a labelled PNG of the graph + edge labels after each tracer phase."""

    def __init__(self, out_dir: Path, image_shape: tuple[int, int], landmarks: dict) -> None:
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.image_shape = image_shape
        self.landmarks = landmarks
        self.counter = 0

    def dump(self, G: nx.Graph, edge_labels: dict, name: str) -> None:
        self.counter += 1
        h, w = self.image_shape
        canvas = np.full((h, w, 3), 32, dtype=np.uint8)

        for u, v, data in G.edges(data=True):
            line = data.get("line")
            if line is None:
                continue
            key = (u, v) if (u, v) in edge_labels else (v, u)
            label = edge_labels.get(key)
            if label is None:
                color = (120, 120, 120)
                thickness = 1
            else:
                base = VEIN_COLORS.get(label)
                if base is None:
                    base = [200, 255, 200]
                color = (int(base[2]), int(base[1]), int(base[0]))
                thickness = 3
            pts = np.asarray(line.coords, dtype=np.int32).reshape(-1, 1, 2)
            cv2.polylines(canvas, [pts], False, color, thickness, cv2.LINE_AA)

            if label is not None:
                mid_idx = len(line.coords) // 2
                mx, my = line.coords[mid_idx]
                cv2.putText(
                    canvas, label, (int(mx), int(my)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA
                )

        for node, nd in G.nodes(data=True):
            if isinstance(node, tuple) and len(node) >= 2:
                x, y = int(node[0]), int(node[1])
            elif "x" in nd and "y" in nd:
                x, y = int(nd["x"]), int(nd["y"])
            else:
                continue
            deg = G.degree(node)
            color = (0, 128, 255) if deg <= 2 else (255, 80, 255)
            cv2.circle(canvas, (x, y), 4, color, -1)

        for lm_name, lm in self.landmarks.items():
            if lm.point is None:
                continue
            lx, ly = int(lm.point.x), int(lm.point.y)
            cv2.circle(canvas, (lx, ly), 12, (255, 255, 255), 2)
            cv2.putText(
                canvas, lm_name, (lx + 14, ly + 6), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA
            )

        labeled = sum(1 for _ in edge_labels.values())
        cv2.putText(
            canvas,
            f"{self.counter:02d} {name}  labeled={labeled}/{G.number_of_edges()}",
            (30, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.6,
            (255, 255, 255),
            3,
            cv2.LINE_AA,
        )

        path = self.out_dir / f"{self.counter:02d}_{name}.png"
        cv2.imwrite(str(path), canvas)
        logger.info("Tracer debug dump: %s", path)


class _TracerHook:
    """Lightweight debug sink: forwards (G, edge_labels, name) to a callback
    after each tracer phase instead of writing a PNG. Used for programmatic
    per-phase capture (e.g. thesis figures). Backward-compatible — only active
    when ``debug_hook`` is passed to ``trace_veins_from_landmarks``.
    """

    def __init__(self, fn) -> None:
        self.fn = fn

    def dump(self, G: nx.Graph, edge_labels: dict, name: str) -> None:
        self.fn(G, edge_labels, name)


def trace_veins_from_landmarks(
    skel_graph: SkeletonGraph,
    landmarks: dict[str, Landmark],
    wing_outline: "Polygon | None" = None,
    config: PipelineConfig | None = None,
    wing_axis: Optional[WingAxis] = None,
    debug_dir: Path | None = None,
    debug_hook=None,
) -> list[VeinIdentification]:
    """Identify veins in the skeleton graph using landmarks.

    Args:
        skel_graph: Skeleton graph (after landmark anchoring).
        landmarks: Anchored landmarks dict.
        wing_outline: Wing outline polygon for costa detection.
        config: Pipeline configuration.
        wing_axis: Optional wing proximal/distal axis. When provided, L6
            detection uses the axis's AP vector for rotation-invariant
            "posterior heading" checks instead of assuming positive-Y is
            posterior.
        debug_dir: If set, write a labelled PNG of the graph after each phase.
        debug_hook: If set (and debug_dir is None), call ``debug_hook(G,
            edge_labels, name)`` after each phase for in-memory per-phase
            capture (e.g. figure generation). Backward-compatible no-op when
            None.
    """
    if config is None:
        config = PipelineConfig()

    G = skel_graph.graph
    edge_labels: dict[tuple, str] = {}

    if debug_dir is not None:
        dbg = _TracerDumper(debug_dir, skel_graph.image_shape, landmarks)
    elif debug_hook is not None:
        dbg = _TracerHook(debug_hook)
    else:
        dbg = None
    if dbg:
        dbg.dump(G, edge_labels, "initial")

    # Phase 0: intentional no-op. merge_through_junctions used to run here on
    # the bare graph, which destroyed real PCV/ACV Y-junction topology before
    # the labeler had a chance to use it (0001 PCV, 0010 L3). It now runs
    # further down — after Phase 1c shortest-path has placed primary labels —
    # so it can clean up the graph for the legacy fallback phases (2 → 2e)
    # without disturbing the new labeler.
    if dbg:
        dbg.dump(G, edge_labels, "phase0_skipped")

    # Phase 1: Detect costa edges using margin band (on the merged graph)
    costa_band = None
    costa_band_dist = None
    if wing_outline is not None:
        from identify_features.models.costa_detector import detect_costa_edges

        costa_keys, costa_band = detect_costa_edges(skel_graph, landmarks, wing_outline, config)
        for key in costa_keys:
            edge_labels[key] = "costa"
        if costa_band is not None:
            # _propagate_through_degree2 is called ~4 times below with the same
            # costa_band each time; computing the EDT once here and reusing it
            # avoids ~3 full-image distance transforms. distance_transform_edt
            # is a deterministic pure function of its input, so the reuse is
            # byte-identical to the pre-hoist computation.
            from scipy import ndimage

            costa_band_dist = ndimage.distance_transform_edt(costa_band == 0)
    if dbg:
        dbg.dump(G, edge_labels, "phase1_costa")

    # Phase 1b: Assign temporary chain IDs through degree-2 nodes so that
    # landmark labeling can use full chain geometry for soft-landmark matching.
    # Landmark nodes are chain boundaries even when degree-2.
    landmark_nodes = {lm.snapped_node for lm in landmarks.values() if lm.snapped_node is not None}
    chain_lines = _assign_chain_ids(G, edge_labels, boundary_nodes=landmark_nodes)

    # Phase 1c: Primary longitudinal labeling — pixel-length shortest path
    # between each longitudinal's reliable, snapped landmark endpoints with
    # collinearity-based conflict resolution and an iterative endpoint-
    # connectivity gate. Veins not labeled here (no path / dropped by gate /
    # unreliable landmarks) fall through to the production phases below.
    _label_longitudinals_via_shortest_path(G, landmarks, edge_labels, config, skel_graph.median_vein_width_px)
    if dbg:
        dbg.dump(G, edge_labels, "phase1c_shortest_path")

    # Phase 1d: Merge longitudinal edges through Y-fork junctions to prep the
    # graph for the legacy fallback labeling phases (2 → 2e). Only runs when
    # the fallback will actually need to do work — i.e. at least one of the
    # 6 longitudinals (L1, Rs, L2, L3, L4, L5) wasn't labeled by Phase 1c.
    # When shortest-path covered everything, skipping the merge preserves
    # Y-junction topology that downstream crossvein detection benefits from.
    # The fallback was designed against a pre-merged graph (single edge per
    # longitudinal across crossvein junctions); without this step it leaves
    # veins half-labeled (e.g. L4 missing its proximal half on
    # 20241205_…_0015). Running merge AFTER the shortest-path labeler
    # preserves primary labels (merge_through_junctions never combines edges
    # with different labels and propagates a single existing label across
    # the merged edge).
    from identify_features.models.topology import LONGITUDINAL_ENDPOINTS

    labeled_now = set(edge_labels.values())
    missing_longitudinals = set(LONGITUDINAL_ENDPOINTS.keys()) - labeled_now
    if missing_longitudinals:
        from identify_features.models.junction_resolver import merge_through_junctions

        logger.info(
            "Phase 1d: running merge_through_junctions to prep fallback for missing %s",
            sorted(missing_longitudinals),
        )
        # Only protect longitudinal-anchor landmark nodes from contraction.
        # Crossvein-anchor landmarks (ACV.a/ACV.p/PCV.a/PCV.p) sit on the
        # longitudinal trunk at the exact junctions merge_through_junctions
        # is meant to contract (e.g. PCV.a is where PCV branches off L4 —
        # protecting it leaves L4 split into 3 edges and breaks fallback
        # labeling, plus splits the crossvein chain into two pieces so the
        # second piece gets promoted to EV1 at Phase 4d). The
        # perpendicularity guard inside merge_through_junctions already
        # rejects pairs where the third edge is too collinear with the
        # merged pair (handles the steep-crossvein edge case). Crossvein
        # detection downstream uses landmark positions, not snapped node
        # IDs, so contracting these snaps is safe.
        _CROSSVEIN_ANCHORS = {"ACV.a", "ACV.p", "PCV.a", "PCV.p"}
        protected = {
            lm.snapped_node
            for name, lm in landmarks.items()
            if lm.snapped_node is not None and name not in _CROSSVEIN_ANCHORS
        }
        skel_graph.graph = merge_through_junctions(
            G,
            edge_labels,
            config,
            protected_nodes=protected,
            median_vein_width_px=skel_graph.median_vein_width_px,
        )
        G = skel_graph.graph
        # Chain IDs were assigned on the pre-merge graph; rebuild them on the
        # merged graph so the fallback's _label_with_chain helper sees current
        # chain boundaries.
        chain_lines = _assign_chain_ids(G, edge_labels, boundary_nodes=landmark_nodes)
        if dbg:
            dbg.dump(G, edge_labels, "phase1d_merge_junctions")
    else:
        logger.info("Phase 1d: skipped — Phase 1c labeled all longitudinals")
        if dbg:
            dbg.dump(G, edge_labels, "phase1d_skipped")

    # Phase 2: Label edges at landmark positions (on the merged graph)
    _label_landmark_edges(
        G,
        landmarks,
        edge_labels,
        config,
        skel_graph.median_vein_width_px,
        chain_lines,
        wing_axis=wing_axis,
    )
    if dbg:
        dbg.dump(G, edge_labels, "phase2_landmark_edges")

    # Phase 2b: Propagate labels through degree-2 pass-through nodes
    _propagate_through_degree2(
        G,
        edge_labels,
        costa_band=costa_band,
        costa_max_dist=skel_graph.median_vein_width_px * config.costa_propagation_max_distance_vw,
        band_dist=costa_band_dist,
    )
    if dbg:
        dbg.dump(G, edge_labels, "phase2b_propagate")

    # Phase 2c: Extend longitudinals to distal landmarks if they don't reach
    _extend_to_distal_landmarks(G, edge_labels, landmarks, skel_graph.median_vein_width_px, config)
    if dbg:
        dbg.dump(G, edge_labels, "phase2c_extend_distal")

    # Phase 2d: Re-propagate after extension
    _propagate_through_degree2(
        G,
        edge_labels,
        costa_band=costa_band,
        costa_max_dist=skel_graph.median_vein_width_px * config.costa_propagation_max_distance_vw,
        band_dist=costa_band_dist,
    )
    if dbg:
        dbg.dump(G, edge_labels, "phase2d_repropagate")

    # Phase 2e: Connect disconnected vein fragments via shortest unlabeled path
    _connect_vein_fragments(G, edge_labels)
    _propagate_through_degree2(
        G,
        edge_labels,
        costa_band=costa_band,
        costa_max_dist=skel_graph.median_vein_width_px * config.costa_propagation_max_distance_vw,
        band_dist=costa_band_dist,
    )
    if dbg:
        dbg.dump(G, edge_labels, "phase2e_connect_fragments")

    # Phase 3: Detect L6 (short posterior branch off L5 near L4-L5)
    _detect_l6(G, edge_labels, landmarks, wing_axis)
    if dbg:
        dbg.dump(G, edge_labels, "phase3_l6")

    # Phase 4-pre0: Primary crossvein labeler — pixel-length shortest path
    # between reliable + snapped crossvein-anchor landmarks.
    _label_crossveins_via_landmark_path(G, edge_labels, landmarks)
    if dbg:
        dbg.dump(G, edge_labels, "phase4pre0_landmark_path")

    # Phase 4-pre1: Secondary crossvein labeler — per-position corridor +
    # H-shape / partial-CV pair-merge. Labels any crossvein the shortest-
    # path step missed when the chain topology gives a clean answer.
    _label_crossveins_via_chain_topology(G, edge_labels, landmarks, config, skel_graph.median_vein_width_px)
    if dbg:
        dbg.dump(G, edge_labels, "phase4pre1_chain_topology")

    # Phase 4: Detect crossveins (ACV between L3↔L4, PCV between L4↔L5).
    # Now skips any cv_name already labeled by Phase 4-pre0/4-pre1 — acts as
    # a fallback for crossveins those couldn't trace.
    _detect_crossveins(G, edge_labels, config=config, median_vein_width_px=skel_graph.median_vein_width_px)
    if dbg:
        dbg.dump(G, edge_labels, "phase4_crossveins")

    # Phase 4a: Junction-based crossvein detection (trace unlabeled paths between longitudinals)
    _detect_crossveins_via_junctions(G, edge_labels)
    if dbg:
        dbg.dump(G, edge_labels, "phase4a_crossveins_junction")

    # Phase 4b: Fallback crossvein detection using crossvein landmarks
    _detect_crossveins_fallback(G, edge_labels, landmarks, config, skel_graph.median_vein_width_px)
    if dbg:
        dbg.dump(G, edge_labels, "phase4b_crossveins_fallback")

    # Phase 4b': Extend each labeled crossvein by absorbing adjacent unlabeled
    # edges that share a graph node with it and lie inside its AP corridor.
    # Catches cases where the chain detector / fallback labels only a piece
    # of a crossvein (e.g. fallback labels just the edge touching PCV.a) and
    # the rest stays unlabeled, eventually getting promoted to EV*.
    _extend_crossveins_along_corridor(G, edge_labels, skel_graph.median_vein_width_px, config=config)
    if dbg:
        dbg.dump(G, edge_labels, "phase4b_crossveins_extend")

    # Phase 4c: Re-propagate labels through degree-2 nodes after crossvein labeling
    _propagate_through_degree2(
        G,
        edge_labels,
        costa_band=costa_band,
        costa_max_dist=skel_graph.median_vein_width_px * config.costa_propagation_max_distance_vw,
        band_dist=costa_band_dist,
    )
    if dbg:
        dbg.dump(G, edge_labels, "phase4c_repropagate")

    # Phase 4d: Promote remaining unlabeled edges to ectopic veins (EV1, EV2, …)
    _label_ectopic_edges(G, edge_labels, skel_graph.median_vein_width_px, config)
    if dbg:
        dbg.dump(G, edge_labels, "phase4d_ectopic")

    # Phase 5: Build VeinIdentification objects.
    # Primary: convert µm gap. Fallback: 4× median vein width (last-resort
    # is 0 if both are unavailable, i.e. only collinear edges merge).
    if config.um_per_px:
        merge_gap = config.to_px(config.merge_max_gap_um)
    else:
        merge_gap = 4.0 * skel_graph.median_vein_width_px
    veins = _build_vein_identifications(G, edge_labels, max_merge_gap_px=merge_gap)

    # Phase 5b: Synthesize crossveins from landmarks when graph detection failed.
    # On wings where the pixel classifier fuses L3+ACV+L4 (or L4+PCV+L5) into
    # a single tissue blob, the skeleton has no separate crossvein path and
    # every stub-based detection phase returns nothing. A synthetic centerline
    # drawn between the crossvein landmarks still serves as a barrier for
    # intervein polygon splitting and prevents spurious compound regions
    # like "1st basal + 1st posterior". Toggle off when the merged-region
    # output is preferred (e.g. specimens with genuinely absent crossveins).
    if config.synthesize_missing_crossveins:
        _synthesize_crossveins_from_landmarks(G, edge_labels, landmarks, veins)

    # Phase 5c: Promote present-but-incomplete veins to PARTIAL and append
    # placeholders for canonical veins with no labelled path (ABSENT). Runs
    # last so a Phase-5b synthesised crossvein counts as present, not absent,
    # and so any earlier-phase IDENTIFIED vein has its final edge set fixed
    # before connectedness / endpoint tests fire. The flag exists so legacy
    # output (only IDENTIFIED / INFERRED / ECTOPIC ever appear) is recoverable.
    if config.assign_absent_partial_status:
        _assign_absent_and_partial(
            veins,
            G,
            edge_labels,
            landmarks,
            skel_graph.median_vein_width_px,
            config,
        )

    return veins


def _label_longitudinals_via_shortest_path(
    G: nx.Graph,
    landmarks: dict[str, Landmark],
    edge_labels: dict[tuple, str],
    config: PipelineConfig,
    median_vein_width_px: float,
) -> None:
    """Primary longitudinal labeling. For each longitudinal in
    ``LONGITUDINAL_ENDPOINTS`` whose two endpoint landmarks are reliable
    and snapped, find the pixel-length shortest path between the snapped
    nodes and label every edge on it as that vein.

    When two veins claim the same edge, the per-edge collinearity contest
    decides ownership using a 3-tier rule:
      1. higher local collinearity (path-bend angle at the disputed edge);
      2. tie-break (within 5°) by lower cumulative path twist;
      3. final tie-break by the subset rule — a vein whose path is a strict
         subset of another claimer's wins (otherwise it has no exclusive
         tissue).

    After conflict resolution, an endpoint-connectivity gate iteratively
    drops the worst-twisting vein whose surviving edges no longer connect
    its two snapped landmark endpoints, then re-resolves so collateral
    failures recover.

    Edges already in ``edge_labels`` (e.g. costa from phase 1) are never
    overwritten. Veins that fail the gate or have no path are silently
    skipped — the production landmark-edge / propagate / extend / connect
    phases that follow this function act as a fallback for anything still
    unlabeled.
    """
    from identify_features.models.topology import LONGITUDINAL_ENDPOINTS
    from identify_features.utils.geometry_utils import angle_between_vectors
    from identify_features.utils.graph_utils import edge_departure_direction

    window_px = config.departure_sample_px(median_vein_width_px)

    # Pass 1: shortest paths.
    paths_per_vein: dict[str, list[int]] = {}
    for vein_name, (lm_a_name, lm_b_name) in LONGITUDINAL_ENDPOINTS.items():
        a = landmarks.get(lm_a_name)
        b = landmarks.get(lm_b_name)
        if a is None or b is None or not a.reliable or not b.reliable:
            continue
        if a.snapped_node is None or b.snapped_node is None:
            continue
        if not (G.has_node(a.snapped_node) and G.has_node(b.snapped_node)):
            continue
        try:
            paths_per_vein[vein_name] = nx.shortest_path(G, a.snapped_node, b.snapped_node, weight="length_px")
        except nx.NetworkXNoPath:
            continue

    if not paths_per_vein:
        return

    edge_claims: dict[tuple, list[tuple[str, int]]] = defaultdict(list)
    for vein_name, path in paths_per_vein.items():
        for i in range(len(path) - 1):
            edge_claims[_edge_key(path[i], path[i + 1])].append((vein_name, i))

    edges_per_vein = {v: {_edge_key(p[i], p[i + 1]) for i in range(len(p) - 1)} for v, p in paths_per_vein.items()}

    def _local_collinearity(path: list[int], i: int) -> float:
        u, v = path[i], path[i + 1]
        has_prev = i > 0
        has_next = i + 2 < len(path)
        if not has_prev and not has_next:
            return 180.0
        scores: list[float] = []
        if has_prev:
            prev = path[i - 1]
            dep_uv = edge_departure_direction(G, u, v, window_px)
            dep_up = edge_departure_direction(G, u, prev, window_px)
            if dep_uv is not None and dep_up is not None:
                scores.append(angle_between_vectors(dep_uv, dep_up))
        if has_next:
            nxt = path[i + 2]
            dep_vu = edge_departure_direction(G, v, u, window_px)
            dep_vn = edge_departure_direction(G, v, nxt, window_px)
            if dep_vu is not None and dep_vn is not None:
                scores.append(angle_between_vectors(dep_vu, dep_vn))
        return sum(scores) / len(scores) if scores else 180.0

    def _cumulative_twist(path: list[int]) -> float:
        if len(path) < 3:
            return 0.0
        total = 0.0
        for i in range(1, len(path) - 1):
            prev, mid, nxt = path[i - 1], path[i], path[i + 1]
            dep_to_prev = edge_departure_direction(G, mid, prev, window_px)
            dep_to_next = edge_departure_direction(G, mid, nxt, window_px)
            if dep_to_prev is not None and dep_to_next is not None:
                total += 180.0 - angle_between_vectors(dep_to_prev, dep_to_next)
        return total

    twist_per_vein = {v: _cumulative_twist(p) for v, p in paths_per_vein.items()}

    TIE_TOL = 5.0

    def _resolve_round(active: set[str]) -> dict[tuple, str]:
        winners: dict[tuple, str] = {}
        for key, claimers in edge_claims.items():
            active_claimers = [c for c in claimers if c[0] in active]
            if not active_claimers:
                continue
            if len(active_claimers) == 1:
                winners[key] = active_claimers[0][0]
                continue
            names = [c[0] for c in active_claimers]
            local_scores = {c[0]: _local_collinearity(paths_per_vein[c[0]], c[1]) for c in active_claimers}
            subset_of = {v: any(v != w and edges_per_vein[v] < edges_per_vein[w] for w in names) for v in names}
            ranked = sorted(
                names,
                key=lambda n: (-local_scores[n], twist_per_vein[n], not subset_of[n], n),
            )
            top_local = local_scores[ranked[0]]
            within_tol = [n for n in ranked if top_local - local_scores[n] <= TIE_TOL]
            if len(within_tol) > 1:
                within_tol.sort(key=lambda n: (twist_per_vein[n], not subset_of[n], n))
                winners[key] = within_tol[0]
            else:
                winners[key] = ranked[0]
        return winners

    def _reaches_both_endpoints(vein_name: str, winners: dict[tuple, str]) -> bool:
        a_name, b_name = LONGITUDINAL_ENDPOINTS[vein_name]
        snap_a = landmarks[a_name].snapped_node
        snap_b = landmarks[b_name].snapped_node
        if snap_a is None or snap_b is None:
            return False
        own = [k for k, v in winners.items() if v == vein_name]
        if not own:
            return False
        sub = nx.Graph()
        for u, v in own:
            sub.add_edge(u, v)
        if snap_a not in sub or snap_b not in sub:
            return False
        return nx.has_path(sub, snap_a, snap_b)

    active = set(paths_per_vein.keys())
    edge_winner: dict[tuple, str] = {}
    for _ in range(len(active) + 1):
        edge_winner = _resolve_round(active)
        failing = [v for v in active if not _reaches_both_endpoints(v, edge_winner)]
        if not failing:
            break
        worst = max(failing, key=lambda v: (twist_per_vein[v], v))
        logger.info(
            "shortest-path: dropping %s (twist=%.0f°) — owned edges don't connect both endpoints",
            worst,
            twist_per_vein[worst],
        )
        active.discard(worst)

    n_labeled = 0
    for key, vein_name in edge_winner.items():
        if key not in edge_labels:
            edge_labels[key] = vein_name
            n_labeled += 1
    logger.info(
        "shortest-path: labeled %d edge(s) across %d vein(s)",
        n_labeled,
        len({v for v in edge_winner.values()}),
    )


def _label_landmark_edges(
    G: nx.Graph,
    landmarks: dict[str, Landmark],
    edge_labels: dict[tuple, str],
    config: PipelineConfig,
    median_vein_width: float,
    chain_lines: dict[int, LineString] | None = None,
    wing_axis: Optional[WingAxis] = None,
) -> None:
    """Label edges connected to landmark nodes."""
    if chain_lines is None:
        chain_lines = {}

    # Helper: label an edge AND all siblings in its degree-2 chain
    def _label_with_chain(key: tuple, vein_id: str):
        edge_labels[key] = vein_id
        u, v = key
        if not G.has_edge(u, v):
            return
        cid = G[u][v].get("chain_id")
        if cid is None:
            return
        for eu, ev, data in G.edges(data=True):
            if data.get("chain_id") == cid:
                ek = _edge_key(eu, ev)
                if ek not in edge_labels:
                    edge_labels[ek] = vein_id
                    logger.info("Chain-propagated %s to edge %s (chain %d)", vein_id, ek, cid)

    # Helper: label the edge at a degree-1 landmark node
    def _label_endpoint_edge(landmark_name: str, vein_id: str):
        lm = landmarks.get(landmark_name)
        if lm is None or lm.snapped_node is None:
            return
        node = lm.snapped_node
        if node not in G:
            return
        for neighbor in G.neighbors(node):
            key = _edge_key(node, neighbor)
            if key not in edge_labels:
                _label_with_chain(key, vein_id)
                logger.info("Labeled edge %s as %s (from %s landmark)", key, vein_id, landmark_name)
                break

    # DTip → the edge there is L3's distal end
    _label_endpoint_edge("DTip", "L3")

    # Helper: get non-costa neighbors at a junction node
    def _unlabeled_neighbors(node):
        """Return neighbors whose edges aren't already labeled (e.g. costa)."""
        result = []
        for n in G.neighbors(node):
            key = _edge_key(node, n)
            if key not in edge_labels:
                result.append(n)
        return result

    # Helper: find the edge whose chain geometry passes closest to a landmark point
    def _nearest_edge_to_landmark(node, neighbors, landmark):
        """Among edges from node to neighbors, find which chain passes closest to landmark."""
        best_n = None
        best_dist = float("inf")
        for n in neighbors:
            cid = G[node][n].get("chain_id") if G.has_edge(node, n) else None
            line = chain_lines.get(cid) if cid is not None else None
            if line is None:
                line = G[node][n].get("line")
            if line is None:
                continue
            dist = line.distance(landmark.point)
            if dist < best_dist:
                best_dist = dist
                best_n = n
        return best_n, best_dist

    # Helper: among neighbors of a junction, find which one reaches a target
    # node cheapest, avoiding the junction itself. Used when a soft distal
    # landmark has snapped reliably but lies past one or more branch points,
    # so line-distance to the junction's immediate chain misses it.
    #
    # Default metric is "path_length" (Dijkstra over edge length_px) — this
    # is robust to chains broken into many short edges by crossvein
    # intersections, and to short-hop detours via ectopic crossveins that
    # would mislead pure hop-counting. The "hops" metric (BFS) is retained
    # as an option via config.soft_landmark_reach_metric.
    reach_metric = getattr(config, "soft_landmark_reach_metric", "path_length")

    def _neighbor_reach_costs(junction, neighbors, target_node):
        """Return {neighbor: cost} under the configured metric.

        The junction is masked out so paths cannot loop back through it.
        Neighbors that cannot reach ``target_node`` get ``inf``.
        """
        if target_node is None or target_node == junction:
            return {n: float("inf") for n in neighbors}
        Gm = G.copy()
        if Gm.has_node(junction):
            Gm.remove_node(junction)
        costs: dict[int, float] = {}
        for n in neighbors:
            if not Gm.has_node(n):
                costs[n] = float("inf")
                continue
            if n == target_node:
                costs[n] = 0.0
                continue
            try:
                if reach_metric == "hops":
                    c = nx.shortest_path_length(Gm, n, target_node)
                else:
                    c = nx.shortest_path_length(Gm, n, target_node, weight="length_px")
                costs[n] = float(c)
            except (nx.NetworkXNoPath, nx.NodeNotFound):
                costs[n] = float("inf")
        return costs

    def _neighbor_toward_node(junction, neighbors, target_node):
        """Return (best_neighbor, cost) under the configured reach metric.

        Backwards-compatible wrapper kept in case callers want a single-shot
        query. Tier 2c uses ``_neighbor_reach_costs`` directly so it can
        cross-compare L4.d and L5.d on the same neighbors.
        """
        costs = _neighbor_reach_costs(junction, neighbors, target_node)
        if not costs:
            return None, float("inf")
        best_n = min(costs, key=costs.get)
        return (best_n, costs[best_n]) if costs[best_n] != float("inf") else (None, float("inf"))

    def _reliable_snap(lm):
        """True if the landmark is snapped within the configured snap radius."""
        return (
            lm is not None
            and lm.snapped_node is not None
            and lm.snap_distance is not None
            and lm.snap_distance <= max_lm_dist
        )

    # L2-L3 junction: simultaneous matching with L2.d and DTip
    lm_l2l3 = landmarks.get("L2-L3")
    lm_dtip = landmarks.get("DTip")
    lm_l1rs = landmarks.get("L1-Rs")
    lm_l2d = landmarks.get("L2.d")
    max_lm_dist = config.snap_radius_px(median_vein_width)

    if lm_l2l3 and lm_l2l3.snapped_node is not None:
        node = lm_l2l3.snapped_node
        if node in G:
            neighbors = _unlabeled_neighbors(node)
            sample_px = config.departure_sample_px(median_vein_width)

            if len(neighbors) >= 1:
                # Check if L3 already labeled at this junction (from DTip endpoint)
                l3_already = any(edge_labels.get(_edge_key(node, n)) == "L3" for n in G.neighbors(node))

                # Simultaneous: for each edge, compute distance to L2.d and DTip
                # Assign L2 to edge nearest L2.d, L3 to edge nearest DTip
                l2_assigned = False
                l3_assigned = l3_already

                if not l3_already and lm_l2d and lm_l2d.snapped_node is not None and lm_dtip:
                    # Score all edges against both landmarks
                    scores = []
                    for n in neighbors:
                        _, dist_l2d = _nearest_edge_to_landmark(node, [n], lm_l2d)
                        _, dist_dtip = _nearest_edge_to_landmark(node, [n], lm_dtip)
                        scores.append((n, dist_l2d, dist_dtip))

                    # L2: edge with smallest dist to L2.d (if within snap radius)
                    scores_l2 = sorted(scores, key=lambda s: s[1])
                    if scores_l2[0][1] <= max_lm_dist:
                        best_l2 = scores_l2[0][0]
                        key = _edge_key(node, best_l2)
                        if key not in edge_labels:
                            _label_with_chain(key, "L2")
                            logger.info(
                                "Labeled edge %s as L2 (from L2-L3, nearest to L2.d, dist=%.0f)", key, scores_l2[0][1]
                            )
                            l2_assigned = True

                    # L3: edge with smallest dist to DTip (excluding L2 edge)
                    remaining_scores = [s for s in scores if _edge_key(node, s[0]) not in edge_labels]
                    if remaining_scores:
                        scores_l3 = sorted(remaining_scores, key=lambda s: s[2])
                        best_l3 = scores_l3[0][0]
                        key = _edge_key(node, best_l3)
                        if key not in edge_labels:
                            _label_with_chain(key, "L3")
                            logger.info(
                                "Labeled edge %s as L3 (from L2-L3, nearest to DTip, dist=%.0f)", key, scores_l3[0][2]
                            )
                            l3_assigned = True

                elif lm_l2d and lm_l2d.snapped_node is not None:
                    # Only L2.d available (L3 already labeled or no DTip)
                    best_l2, dist = _nearest_edge_to_landmark(node, neighbors, lm_l2d)
                    if best_l2 is not None and dist <= max_lm_dist:
                        key = _edge_key(node, best_l2)
                        if key not in edge_labels:
                            _label_with_chain(key, "L2")
                            logger.info("Labeled edge %s as L2 (from L2-L3, nearest to L2.d, dist=%.0f)", key, dist)
                            l2_assigned = True

                elif not l3_already and lm_dtip:
                    # Only DTip available (no L2.d)
                    toward_dtip = direction_toward(
                        (G.nodes[node]["x"], G.nodes[node]["y"]),
                        (lm_dtip.x, lm_dtip.y),
                    )
                    scored = []
                    for n in neighbors:
                        dep = edge_departure_direction(G, node, n, sample_px)
                        angle = angle_between_vectors(dep, toward_dtip)
                        scored.append((n, angle))
                    scored.sort(key=lambda s: s[1])
                    key = _edge_key(node, scored[0][0])
                    if key not in edge_labels:
                        _label_with_chain(key, "L3")
                        logger.info("Labeled edge %s as L3 (from L2-L3, toward DTip fallback)", key)

                # Remaining → Rs
                remaining = [n for n in neighbors if _edge_key(node, n) not in edge_labels]
                for n in remaining:
                    key = _edge_key(node, n)
                    if key not in edge_labels:
                        _label_with_chain(key, "Rs")
                        logger.info("Labeled edge %s as Rs (from L2-L3, remaining)", key)

    # L1-Rs junction
    lm_l1rs = landmarks.get("L1-Rs")
    lm_sc = landmarks.get("subcostal break")

    if lm_l1rs and lm_l1rs.snapped_node is not None:
        node = lm_l1rs.snapped_node
        if node in G:
            neighbors = _unlabeled_neighbors(node)
            sample_px = config.departure_sample_px(median_vein_width)

            for n in neighbors:
                key = _edge_key(node, n)
                if key in edge_labels:
                    continue
                if lm_sc:
                    dep = edge_departure_direction(G, node, n, sample_px)
                    toward_sc = direction_toward(
                        (G.nodes[node]["x"], G.nodes[node]["y"]),
                        (lm_sc.x, lm_sc.y),
                    )
                    angle = angle_between_vectors(dep, toward_sc)
                    if angle < 60:
                        _label_with_chain(key, "L1")
                        logger.info("Labeled edge %s as L1 (from L1-Rs, toward SC)", key)
                    else:
                        _label_with_chain(key, "Rs")
                        logger.info("Labeled edge %s as Rs (from L1-Rs, away from SC)", key)
                else:
                    _label_with_chain(key, "Rs")

    # Subcostal break → L1
    _label_endpoint_edge("subcostal break", "L1")

    # L4-L5 junction: simultaneous matching with L4.d and L5.d, with fallbacks
    # mirroring the L2-L3 pattern. L4-L5 is a reliable hard landmark, so tracing
    # should proceed even if soft distal landmarks fail to anchor.
    lm_l4l5 = landmarks.get("L4-L5")
    lm_l4d = landmarks.get("L4.d")
    lm_l5d = landmarks.get("L5.d")

    if lm_l4l5 and lm_l4l5.snapped_node is not None:
        node = lm_l4l5.snapped_node
        if node in G:
            neighbors = _unlabeled_neighbors(node)

            if len(neighbors) >= 1:
                l4d_ok = lm_l4d is not None and lm_l4d.snapped_node is not None
                l5d_ok = lm_l5d is not None and lm_l5d.snapped_node is not None

                def _l4_assigned() -> bool:
                    return any(edge_labels.get(_edge_key(node, n)) == "L4" for n in G.neighbors(node))

                def _l5_assigned() -> bool:
                    return any(edge_labels.get(_edge_key(node, n)) == "L5" for n in G.neighbors(node))

                # Tier 1: Simultaneous matching (both soft landmarks anchored and close)
                if l4d_ok and l5d_ok and len(neighbors) >= 2:
                    scores = []
                    for n in neighbors:
                        _, dist_l4d = _nearest_edge_to_landmark(node, [n], lm_l4d)
                        _, dist_l5d = _nearest_edge_to_landmark(node, [n], lm_l5d)
                        scores.append((n, dist_l4d, dist_l5d))

                    any_close = any(min(s[1], s[2]) <= max_lm_dist for s in scores)
                    if any_close:
                        best_for_l4 = min(scores, key=lambda s: s[1])
                        best_for_l5 = min(scores, key=lambda s: s[2])

                        if best_for_l4[0] != best_for_l5[0]:
                            for n, dist_l4, dist_l5 in scores:
                                key = _edge_key(node, n)
                                if key in edge_labels:
                                    continue
                                if n == best_for_l4[0]:
                                    _label_with_chain(key, "L4")
                                    logger.info(
                                        "Labeled edge %s as L4 (from L4-L5, nearest to L4.d, dist=%.0f)",
                                        key,
                                        dist_l4,
                                    )
                                else:
                                    _label_with_chain(key, "L5")
                                    logger.info(
                                        "Labeled edge %s as L5 (from L4-L5, nearest to L5.d, dist=%.0f)",
                                        key,
                                        dist_l5,
                                    )
                        else:
                            winner = best_for_l4[0]
                            if best_for_l4[1] <= best_for_l5[2]:
                                for n, dist_l4, dist_l5 in scores:
                                    key = _edge_key(node, n)
                                    if key in edge_labels:
                                        continue
                                    if n == winner:
                                        _label_with_chain(key, "L4")
                                        logger.info(
                                            "Labeled edge %s as L4 (from L4-L5, contested, L4.d closer: %.0f)",
                                            key,
                                            dist_l4,
                                        )
                                    else:
                                        _label_with_chain(key, "L5")
                                        logger.info(
                                            "Labeled edge %s as L5 (from L4-L5, contested, assigned remaining)",
                                            key,
                                        )
                            else:
                                for n, dist_l4, dist_l5 in scores:
                                    key = _edge_key(node, n)
                                    if key in edge_labels:
                                        continue
                                    if n == winner:
                                        _label_with_chain(key, "L5")
                                        logger.info(
                                            "Labeled edge %s as L5 (from L4-L5, contested, L5.d closer: %.0f)",
                                            key,
                                            dist_l5,
                                        )
                                    else:
                                        _label_with_chain(key, "L4")
                                        logger.info(
                                            "Labeled edge %s as L4 (from L4-L5, contested, assigned remaining)",
                                            key,
                                        )

                # Tier 2: Single-landmark fallback (L4.d only), if L4 still unassigned
                if l4d_ok and not _l4_assigned():
                    remaining = _unlabeled_neighbors(node)
                    best_l4, dist = _nearest_edge_to_landmark(node, remaining, lm_l4d)
                    if best_l4 is not None and dist <= max_lm_dist:
                        key = _edge_key(node, best_l4)
                        if key not in edge_labels:
                            _label_with_chain(key, "L4")
                            logger.info("Labeled edge %s as L4 (from L4-L5, L4.d fallback, dist=%.0f)", key, dist)

                # Tier 2b: Single-landmark fallback (L5.d only), if L5 still unassigned
                if l5d_ok and not _l5_assigned():
                    remaining = _unlabeled_neighbors(node)
                    best_l5, dist = _nearest_edge_to_landmark(node, remaining, lm_l5d)
                    if best_l5 is not None and dist <= max_lm_dist:
                        key = _edge_key(node, best_l5)
                        if key not in edge_labels:
                            _label_with_chain(key, "L5")
                            logger.info("Labeled edge %s as L5 (from L4-L5, L5.d fallback, dist=%.0f)", key, dist)

                # Tier 2c: Graph-reachability fallback. Fires when a soft
                # landmark is reliably snapped but the junction's immediate
                # chain terminates at another branch point before reaching
                # the landmark's node — line distance misses it, but walking
                # the graph past the branch point does not. Metric defaults
                # to pixel path length (Dijkstra over edge length_px), which
                # is robust to chains fragmented by crossvein intersections
                # and to short-hop detours via ectopic crossveins. Hops is
                # available as an option via config.soft_landmark_reach_metric.
                l4d_costs_by_n: dict[int, float] = {}
                l5d_costs_by_n: dict[int, float] = {}
                remaining = _unlabeled_neighbors(node)
                if remaining and _reliable_snap(lm_l4d) and not _l4_assigned():
                    l4d_costs_by_n = _neighbor_reach_costs(node, remaining, lm_l4d.snapped_node)
                if remaining and _reliable_snap(lm_l5d) and not _l5_assigned():
                    l5d_costs_by_n = _neighbor_reach_costs(node, remaining, lm_l5d.snapped_node)

                if l4d_costs_by_n and not _l4_assigned():
                    best_l4 = min(l4d_costs_by_n, key=l4d_costs_by_n.get)
                    # If L5.d is also reliable and prefers the same neighbor,
                    # give it to whichever landmark has the smaller cost.
                    if l5d_costs_by_n and l5d_costs_by_n.get(best_l4, float("inf")) < l4d_costs_by_n[best_l4]:
                        pass  # Let L5.d claim it via its own block below
                    elif l4d_costs_by_n[best_l4] != float("inf"):
                        key = _edge_key(node, best_l4)
                        if key not in edge_labels:
                            _label_with_chain(key, "L4")
                            logger.info(
                                "Labeled edge %s as L4 (from L4-L5, L4.d graph-reach, metric=%s, cost=%.1f)",
                                key,
                                reach_metric,
                                l4d_costs_by_n[best_l4],
                            )

                if l5d_costs_by_n and not _l5_assigned():
                    candidates = {
                        n: c for n, c in l5d_costs_by_n.items() if edge_labels.get(_edge_key(node, n)) != "L4"
                    }
                    if candidates:
                        best_l5 = min(candidates, key=candidates.get)
                        if candidates[best_l5] != float("inf"):
                            key = _edge_key(node, best_l5)
                            if key not in edge_labels:
                                _label_with_chain(key, "L5")
                                logger.info(
                                    "Labeled edge %s as L5 (from L4-L5, L5.d graph-reach, metric=%s, cost=%.1f)",
                                    key,
                                    reach_metric,
                                    candidates[best_l5],
                                )

                # Tier 3: AP-orientation fallback — fill whatever is still missing.
                # ap_vector points posterior; project each unlabeled neighbor's far
                # chain endpoint onto it. Most anterior (lowest AP) = L4; most posterior = L5.
                if wing_axis is not None and (not _l4_assigned() or not _l5_assigned()):
                    remaining = _unlabeled_neighbors(node)
                    if remaining:
                        ap_x, ap_y = wing_axis.ap_vector
                        node_x = G.nodes[node]["x"]
                        node_y = G.nodes[node]["y"]
                        scored_ap: list[tuple] = []
                        for n in remaining:
                            cid = G[node][n].get("chain_id") if G.has_edge(node, n) else None
                            line = chain_lines.get(cid) if cid is not None else None
                            if line is None:
                                line = G[node][n].get("line")
                            if line is None:
                                continue
                            coords = list(line.coords)
                            start_d = math.hypot(coords[0][0] - node_x, coords[0][1] - node_y)
                            end_d = math.hypot(coords[-1][0] - node_x, coords[-1][1] - node_y)
                            far = coords[-1] if end_d > start_d else coords[0]
                            ap_comp = (far[0] - node_x) * ap_x + (far[1] - node_y) * ap_y
                            scored_ap.append((n, ap_comp))

                        if scored_ap:
                            scored_ap.sort(key=lambda s: s[1])
                            if not _l4_assigned():
                                l4_n, l4_ap = scored_ap[0]
                                key = _edge_key(node, l4_n)
                                if key not in edge_labels:
                                    _label_with_chain(key, "L4")
                                    logger.info("Labeled edge %s as L4 (from L4-L5, AP fallback, ap=%.1f)", key, l4_ap)
                            if not _l5_assigned():
                                l5_n, l5_ap = scored_ap[-1]
                                key = _edge_key(node, l5_n)
                                if key not in edge_labels:
                                    _label_with_chain(key, "L5")
                                    logger.info("Labeled edge %s as L5 (from L4-L5, AP fallback, ap=%.1f)", key, l5_ap)


def _assign_by_proximity(
    G: nx.Graph,
    edge_labels: dict[tuple, str],
    config: PipelineConfig,
) -> None:
    """Assign unlabeled edges to the nearest compatible labeled vein.

    For each unlabeled edge, find the labeled edge whose LineString is
    closest and roughly parallel. Assign the same label.
    """
    # Build labeled line index
    labeled_lines: dict[str, list[LineString]] = defaultdict(list)
    for (u, v), label in edge_labels.items():
        if G.has_edge(u, v):
            labeled_lines[label].append(G[u][v]["line"])

    if not labeled_lines:
        return

    # Assign unlabeled edges
    changed = True
    max_rounds = 5

    for _ in range(max_rounds):
        if not changed:
            break
        changed = False

        for u, v, data in list(G.edges(data=True)):
            key = _edge_key(u, v)
            if key in edge_labels:
                continue

            line = data.get("line")
            if line is None:
                continue

            # Find nearest labeled vein
            best_label = None
            best_dist = float("inf")

            midpoint = line.interpolate(0.5, normalized=True)

            for label, lines in labeled_lines.items():
                for lline in lines:
                    dist = lline.distance(midpoint)
                    if dist < best_dist:
                        best_dist = dist
                        best_label = label

            # Only assign if reasonably close (within ~500px)
            if best_label is not None and best_dist < 500:
                edge_labels[key] = best_label
                labeled_lines[best_label].append(line)
                changed = True
                logger.debug("Proximity-assigned edge %s as %s (dist=%.0fpx)", key, best_label, best_dist)


def _build_vein_identifications(
    G: nx.Graph,
    edge_labels: dict[tuple, str],
    max_merge_gap_px: float = float("inf"),
) -> list[VeinIdentification]:
    """Build VeinIdentification objects from labeled edges."""
    vein_edges: dict[str, list[tuple]] = defaultdict(list)
    for (u, v), label in edge_labels.items():
        if G.has_edge(u, v):
            vein_edges[label].append((u, v))

    veins = []
    for vein_id, edges in sorted(vein_edges.items()):
        # Collect all LineStrings for this vein
        lines = []
        for u, v in edges:
            line = G[u][v].get("line")
            if line is not None:
                lines.append(line)

        # Merge into single LineString if connected, or MultiLineString
        if len(lines) == 1:
            merged = lines[0]
        elif len(lines) > 1:
            merged = _merge_nearby_lines(lines, max_gap_px=max_merge_gap_px)
        else:
            merged = None

        is_ectopic = vein_id.startswith("EV")
        vein = VeinIdentification(
            vein_id=vein_id,
            vein_type=_vein_type(vein_id),
            status=VeinStatus.ECTOPIC if is_ectopic else VeinStatus.IDENTIFIED,
            centerline=merged,
            edge_ids=[G[u][v].get("edge_id", -1) for u, v in edges],
            length_px=sum(l.length for l in lines),
            evidence=[f"{len(edges)} edges"],
        )
        veins.append(vein)
        label = "Ectopic" if is_ectopic else "Identified"
        logger.info("%s %s: %.0fpx (%d edges)", label, vein_id, vein.length_px, len(edges))

    return veins


def _merge_nearby_lines(
    lines: list[LineString],
    max_gap_px: float = float("inf"),
) -> LineString:
    """Merge multiple LineStrings into one, ordering by spatial proximity.

    Lines that are farther apart than max_gap_px are NOT connected —
    only lines within the gap threshold are chained together. Distant
    lines are skipped to avoid drawing long straight connectors.
    """
    if len(lines) <= 1:
        return lines[0] if lines else LineString()

    # Greedy nearest-neighbor chain
    remaining = list(lines)
    result_coords = list(remaining.pop(0).coords)

    while remaining:
        end = Point(result_coords[-1])
        start = Point(result_coords[0])

        best_idx = None
        best_dist = float("inf")
        best_reverse = False
        best_prepend = False

        for i, line in enumerate(remaining):
            coords = list(line.coords)
            for prepend in (False, True):
                ref = start if prepend else end
                for reverse in (False, True):
                    candidate_end = Point(coords[-1] if reverse else coords[0])
                    dist = ref.distance(candidate_end)
                    if dist < best_dist:
                        best_dist = dist
                        best_idx = i
                        best_reverse = reverse
                        best_prepend = prepend

        # Stop if the nearest remaining line is too far away
        if best_idx is None or best_dist > max_gap_px:
            break

        next_line = remaining.pop(best_idx)
        next_coords = list(next_line.coords)

        if best_prepend:
            # Shared point must be at next_coords[-1] — invert reverse logic
            if not best_reverse:
                next_coords = next_coords[::-1]
            result_coords = next_coords[:-1] + result_coords
        else:
            # Shared point must be at next_coords[0]
            if best_reverse:
                next_coords = next_coords[::-1]
            result_coords = result_coords + next_coords[1:]

    return LineString(result_coords)


def _assign_chain_ids(
    G: nx.Graph,
    edge_labels: dict[tuple, str],
    boundary_nodes: set[int] | None = None,
) -> dict[int, LineString]:
    """Assign temporary chain IDs to edges connected through degree-2 nodes.

    Each maximal chain of edges linked by degree-2 nodes gets a unique
    integer ID.  Every edge in the chain is tagged with ``chain_id`` in
    its data dict, and the function returns a mapping from chain ID to
    merged LineString for distance queries.

    Edges that already carry a label (e.g. costa) are excluded so that
    named veins don't merge with unnamed chains.

    *boundary_nodes* (e.g. landmark-snapped nodes) are treated as chain
    terminators even when they have degree 2, preventing chains from
    spanning semantic junction points.
    """
    if boundary_nodes is None:
        boundary_nodes = set()
    chain_id = 0
    visited: set[tuple] = set()
    chain_lines: dict[int, LineString] = {}

    def _is_pass_through(node):
        return G.degree(node) == 2 and node not in boundary_nodes

    for u, v, data in G.edges(data=True):
        key = _edge_key(u, v)
        if key in visited or key in edge_labels:
            continue

        # Walk the chain in both directions from this edge
        chain_edges = [key]
        visited.add(key)

        for start, direction in [(u, v), (v, u)]:
            cur = start
            prev = direction
            # Walk backward from `start` through degree-2 nodes
            while _is_pass_through(cur):
                nxt = [n for n in G.neighbors(cur) if n != prev]
                if not nxt:
                    break
                nxt = nxt[0]
                nkey = _edge_key(cur, nxt)
                if nkey in visited or nkey in edge_labels:
                    break
                chain_edges.append(nkey)
                visited.add(nkey)
                prev = cur
                cur = nxt

        # Tag every edge in this chain
        for ek in chain_edges:
            eu, ev = ek
            if G.has_edge(eu, ev):
                G[eu][ev]["chain_id"] = chain_id

        # Build merged LineString for the chain by walking from one endpoint.
        # A chain endpoint is a node that touches exactly one edge of THIS chain
        # (interior nodes touch two). This is more reliable than using
        # ``G.degree(n) != 2`` because landmark boundary nodes can still have
        # graph degree 2 — they're chain boundaries by virtue of being in
        # ``boundary_nodes``, not because of their graph degree.
        edge_set = set(chain_edges)
        chain_nodes: set[int] = set()
        for eu, ev in chain_edges:
            chain_nodes.add(eu)
            chain_nodes.add(ev)
        touch_count: dict[int, int] = {n: 0 for n in chain_nodes}
        for eu, ev in chain_edges:
            touch_count[eu] += 1
            touch_count[ev] += 1
        endpoints = [n for n, c in touch_count.items() if c == 1]
        if not endpoints:
            # Closed loop or single-edge degenerate chain — start anywhere
            endpoints = [chain_edges[0][0]]

        start_node = endpoints[0]
        coords: list[tuple] = []
        walked: set[tuple] = set()
        node = start_node

        # Walk along chain edges from start_node
        while True:
            moved = False
            for n in G.neighbors(node):
                ek = _edge_key(node, n)
                if ek in edge_set and ek not in walked:
                    walked.add(ek)
                    seg = G[node][n].get("line")
                    if seg is not None:
                        seg_coords = list(seg.coords)
                        node_pt = (G.nodes[node]["x"], G.nodes[node]["y"])
                        d0 = (seg_coords[0][0] - node_pt[0]) ** 2 + (seg_coords[0][1] - node_pt[1]) ** 2
                        d1 = (seg_coords[-1][0] - node_pt[0]) ** 2 + (seg_coords[-1][1] - node_pt[1]) ** 2
                        if d0 > d1:
                            seg_coords = seg_coords[::-1]
                        if not coords:
                            coords.extend(seg_coords)
                        else:
                            coords.extend(seg_coords[1:])
                    node = n
                    moved = True
                    break
            if not moved:
                break

        if len(coords) >= 2:
            chain_lines[chain_id] = LineString(coords)
        chain_id += 1

    return chain_lines


def _propagate_through_degree2(
    G: nx.Graph,
    edge_labels: dict[tuple, str],
    costa_band: "np.ndarray | None" = None,
    costa_max_dist: float = 96.0,
    band_dist: "np.ndarray | None" = None,
) -> None:
    """Propagate vein labels through degree-2 pass-through nodes.

    At any degree-2 node where one edge is labeled and the other is not,
    the unlabeled edge gets the same label. Repeats until stable.

    Costa propagation is restricted: if any part of the new edge runs
    ≥ costa_max_dist pixels from the nearest costa band pixel,
    propagation is blocked (the edge has left the wing margin).

    ``band_dist`` is the distance transform of ``costa_band == 0``. The
    trace function calls this helper ~4 times per wing with the same
    ``costa_band`` each time; computing the EDT once and passing it in
    avoids ~3 full-image distance transforms (deterministic function of
    its input, so reuse is byte-identical). If only ``costa_band`` is
    given and ``band_dist`` is None, the EDT is computed here for callers
    that don't pre-compute it.
    """
    from scipy import ndimage

    # Precompute distance-from-band map for costa checks (skipped if the
    # caller already hoisted it out of a hot loop and passed it in).
    if band_dist is None and costa_band is not None:
        band_dist = ndimage.distance_transform_edt(costa_band == 0)

    def _edge_in_costa_band(u, v):
        """Check if entire edge stays within costa_max_dist of the band."""
        if band_dist is None:
            return True
        line = G[u][v].get("line")
        if line is None:
            return True
        for cx, cy in line.coords:
            row, col = int(round(cy)), int(round(cx))
            if 0 <= row < band_dist.shape[0] and 0 <= col < band_dist.shape[1]:
                if band_dist[row, col] >= costa_max_dist:
                    return False  # this point is too far from the band
        return True

    changed = True
    while changed:
        changed = False
        for node in G.nodes():
            if G.degree(node) != 2:
                continue
            neighbors = list(G.neighbors(node))
            key0 = _edge_key(node, neighbors[0])
            key1 = _edge_key(node, neighbors[1])
            lbl0 = edge_labels.get(key0)
            lbl1 = edge_labels.get(key1)

            if lbl0 is not None and lbl1 is None:
                # Costa check: don't propagate costa outside the band
                if lbl0 == "costa" and not _edge_in_costa_band(node, neighbors[1]):
                    continue
                edge_labels[key1] = lbl0
                changed = True
                logger.debug("Propagated %s through deg-2 node %d", lbl0, node)
            elif lbl1 is not None and lbl0 is None:
                if lbl1 == "costa" and not _edge_in_costa_band(node, neighbors[0]):
                    continue
                edge_labels[key0] = lbl1
                changed = True
                logger.debug("Propagated %s through deg-2 node %d", lbl1, node)


def _extend_to_distal_landmarks(
    G: nx.Graph,
    edge_labels: dict[tuple, str],
    landmarks: dict[str, "Landmark"],
    median_vein_width: float,
    config: "PipelineConfig",
) -> None:
    """Extend longitudinal veins to their distal landmark if they don't reach.

    For each of L2/L3/L4/L5, checks if the labeled edges reach the
    corresponding distal landmark (L2.d, DTip, L4.d, L5.d). If not,
    finds the nearest unlabeled edge within search_radius of the
    landmark and labels it.
    """
    search_radius = median_vein_width * config.distal_landmark_search_vw

    # Vein → distal landmark mapping
    vein_landmarks = {
        "L2": "L2.d",
        "L3": "DTip",
        "L4": "L4.d",
        "L5": "L5.d",
    }

    for vein_id, lm_name in vein_landmarks.items():
        lm = landmarks.get(lm_name)
        if lm is None:
            continue

        # Check if vein already reaches the landmark
        vein_reaches = False
        for (u, v), label in edge_labels.items():
            if label != vein_id or not G.has_edge(u, v):
                continue
            line = G[u][v].get("line")
            if line is not None and line.distance(lm.point) <= search_radius:
                vein_reaches = True
                break

        if vein_reaches:
            continue

        # Vein doesn't reach — find nearest unlabeled edge to the landmark
        best_key = None
        best_dist = float("inf")
        for u, v, data in G.edges(data=True):
            key = _edge_key(u, v)
            if key in edge_labels:
                continue
            line = data.get("line")
            if line is None:
                continue
            dist = line.distance(lm.point)
            if dist < best_dist and dist <= search_radius:
                best_dist = dist
                best_key = key

        if best_key is not None:
            edge_labels[best_key] = vein_id
            u, v = best_key
            length = G[u][v].get("length_px", 0)
            logger.info(
                "Extended %s to %s: edge %d↔%d (%.0fpx, dist=%.0fpx)",
                vein_id,
                lm_name,
                u,
                v,
                length,
                best_dist,
            )


def _connect_vein_fragments(
    G: nx.Graph,
    edge_labels: dict[tuple, str],
) -> None:
    """Connect disconnected fragments of the same vein via shortest unlabeled path.

    For each longitudinal vein with multiple disconnected edge groups,
    finds the shortest path of unlabeled edges between the closest
    endpoints of the fragments and labels it.
    """
    from collections import defaultdict

    for vein_id in ["L2", "L3", "L4", "L5"]:
        # Collect nodes that are endpoints of this vein's labeled edges
        vein_nodes: set[int] = set()
        for (u, v), label in edge_labels.items():
            if label == vein_id and G.has_edge(u, v):
                vein_nodes.add(u)
                vein_nodes.add(v)

        if len(vein_nodes) < 2:
            continue

        # Find connected components of this vein's edges
        vein_subgraph = nx.Graph()
        for (u, v), label in edge_labels.items():
            if label == vein_id and G.has_edge(u, v):
                vein_subgraph.add_edge(u, v)

        components = list(nx.connected_components(vein_subgraph))
        if len(components) < 2:
            continue

        # Build subgraph of unlabeled edges (weighted by length)
        unlabeled = nx.Graph()
        for u, v, data in G.edges(data=True):
            key = _edge_key(u, v)
            if key not in edge_labels:
                unlabeled.add_edge(u, v, weight=data.get("length_px", 1.0))

        # Try to connect each pair of components via shortest unlabeled path
        for i in range(len(components)):
            for j in range(i + 1, len(components)):
                # Find closest endpoint pair between the two components
                best_path = None
                best_length = float("inf")

                for n1 in components[i]:
                    if n1 not in unlabeled:
                        continue
                    for n2 in components[j]:
                        if n2 not in unlabeled:
                            continue
                        try:
                            path = nx.shortest_path(unlabeled, n1, n2, weight="weight")
                            path_length = sum(unlabeled[path[k]][path[k + 1]]["weight"] for k in range(len(path) - 1))
                            if path_length < best_length:
                                best_length = path_length
                                best_path = path
                        except nx.NetworkXNoPath:
                            continue

                if best_path is not None:
                    # Label the path edges
                    for k in range(len(best_path) - 1):
                        key = _edge_key(best_path[k], best_path[k + 1])
                        if key not in edge_labels:
                            edge_labels[key] = vein_id
                    logger.info(
                        "Connected %s fragments: %d edges, %.0fpx path",
                        vein_id,
                        len(best_path) - 1,
                        best_length,
                    )


def _detect_l6(
    G: nx.Graph,
    edge_labels: dict[tuple, str],
    landmarks: dict[str, Landmark],
    wing_axis: Optional[WingAxis] = None,
) -> None:
    """Detect L6: a short posterior branch off L5 near L4-L5.

    L6 branches from L5 near the proximal end (within 0.5-1.5× Rs length
    from L4-L5) and heads posteriorly. It's similar in length to Rs/L1
    and may be absent.

    When ``wing_axis`` is provided, the "heads posteriorly" check uses the
    axis's AP vector (rotation-invariant). Otherwise, falls back to the
    legacy positive-Y assumption.
    """
    # Need Rs total length as reference — sum across all Rs-labeled edges,
    # not just the first one encountered. When Rs ends up as a chain of
    # multiple graph edges (degree-2 propagation splitting across interior
    # nodes), the prior "first edge only" read produced a tiny rs_length
    # and the L6 length filter ``[0.5, 1.5] × rs_length`` excluded real L6
    # candidates.
    rs_length = 0.0
    for key, label in edge_labels.items():
        if label != "Rs":
            continue
        u, v = key
        if G.has_edge(u, v):
            rs_length += G[u][v].get("length_px", 0.0)

    if rs_length < 10:
        return

    # Find L4-L5 landmark position
    l4l5 = landmarks.get("L4-L5")
    if l4l5 is None:
        return

    l4l5_x, l4l5_y = l4l5.x, l4l5.y

    # Look for unlabeled edges that:
    # 1. Have at least one endpoint near the L4-L5 area (within 1.5× Rs)
    # 2. Are short (0.5-1.5× Rs length)
    # 3. Head posteriorly (positive Y direction = toward bottom of wing)
    min_length = rs_length * 0.5
    max_length = rs_length * 1.5
    max_dist_from_l4l5 = rs_length * 1.5

    best_candidate = None
    best_score = float("inf")

    for u, v, data in G.edges(data=True):
        key = _edge_key(u, v)
        if key in edge_labels:
            continue

        length = data.get("length_px", 0)
        if length < min_length or length > max_length:
            continue

        # Check if either endpoint is near L4-L5
        nd_u = G.nodes[u]
        nd_v = G.nodes[v]
        dist_u = math.hypot(nd_u["x"] - l4l5_x, nd_u["y"] - l4l5_y)
        dist_v = math.hypot(nd_v["x"] - l4l5_x, nd_v["y"] - l4l5_y)
        min_dist = min(dist_u, dist_v)

        if min_dist > max_dist_from_l4l5:
            continue

        # Check direction: must head posteriorly.
        # With a wing axis, project onto the AP vector (rotation-invariant).
        # Without one, fall back to positive-Y = posterior.
        line = data.get("line")
        if line is None:
            continue
        start = line.coords[0]
        end = line.coords[-1]
        dx_edge = end[0] - start[0]
        dy_edge = end[1] - start[1]
        if wing_axis is not None:
            ap_x, ap_y = wing_axis.ap_vector
            posterior_component = abs(dx_edge * ap_x + dy_edge * ap_y)
        else:
            posterior_component = abs(dy_edge)
        if posterior_component < length * 0.3:
            continue

        # Score: prefer edges closer to L4-L5 and more posterior
        score = min_dist
        if score < best_score:
            best_score = score
            best_candidate = key

    if best_candidate is not None:
        edge_labels[best_candidate] = "L6"
        u, v = best_candidate
        length = G[u][v].get("length_px", 0)
        logger.info("Detected L6: edge %d↔%d, %.0fpx", u, v, length)


_CROSSVEIN_LANDMARKS = {"ACV": ("ACV.a", "ACV.p"), "PCV": ("PCV.a", "PCV.p")}


def _label_crossveins_via_landmark_path(
    G: nx.Graph,
    edge_labels: dict[tuple, str],
    landmarks: dict[str, Landmark],
) -> None:
    """Phase 4-pre0: pixel-length shortest path between reliable + snapped
    crossvein-anchor landmarks.

    For each crossvein (ACV/PCV), if both anchor landmarks are reliable AND
    snapped to graph nodes, walk the pixel-length shortest path between
    those snap nodes and label every edge on the path as the crossvein —
    PROVIDED every edge on the path is currently unlabeled. If the path
    crosses a longitudinal-labeled edge the anchor is probably mis-snapped
    or the wing's anatomy is too unusual for a direct trace; we leave it
    for the chain-topology / production fallback to handle.

    Mirrors the longitudinal `_label_longitudinals_via_shortest_path`
    primary-labeler. Inserted before the existing `_detect_crossveins`
    chain detector, which now skips cv_names that this step already
    labeled.
    """
    for cv_name, (a_name, b_name) in _CROSSVEIN_LANDMARKS.items():
        a = landmarks.get(a_name)
        b = landmarks.get(b_name)
        if a is None or b is None or not a.reliable or not b.reliable:
            continue
        if a.snapped_node is None or b.snapped_node is None:
            continue
        if not (G.has_node(a.snapped_node) and G.has_node(b.snapped_node)):
            continue
        try:
            path = nx.shortest_path(G, a.snapped_node, b.snapped_node, weight="length_px")
        except nx.NetworkXNoPath:
            continue
        path_edges = [_edge_key(path[i], path[i + 1]) for i in range(len(path) - 1)]
        # Don't carve through a labeled longitudinal — sanity gate.
        if any(k in edge_labels for k in path_edges):
            continue
        for k in path_edges:
            edge_labels[k] = cv_name
        total_len = sum(G[u][v].get("length_px", 0) for u, v in path_edges if G.has_edge(u, v))
        logger.info(
            "Phase 4-pre0: %s labeled via landmark shortest-path (%d edge(s), %.0fpx)",
            cv_name,
            len(path_edges),
            total_len,
        )


def _label_crossveins_via_chain_topology(
    G: nx.Graph,
    edge_labels: dict[tuple, str],
    landmarks: dict[str, Landmark],
    config: "PipelineConfig",
    median_vein_width_px: float,
) -> None:
    """Phase 4-pre1: per-position corridor + H-shape / partial-CV detection.

    For any crossvein not already labeled by Phase 4-pre0 (landmark
    shortest-path), build chains of unlabeled edges, filter to those whose
    centroid lies in the per-position AP corridor between the two bounding
    longitudinals (margin=0), then pick the winning chain via:

      - Round 1 (H-shape): a chain whose two endpoint nodes each anchor on
        a different bounding vein (within 1× median vein width) is the
        crossvein by construction.
      - Round 2 (partial-CV pair-merge): when no chain in the pool passes
        H-shape, look for chains where exactly one endpoint anchors on a
        bounding vein. Merge two partials touching opposite veins (via
        shared node, or via shortest unlabeled bridge through other pool
        chains, with strict per-node corridor confinement). Orphan
        partials are kept as candidates.
      - Final picker: minimum (dist_a + dist_b) sum-distance among all
        candidates from Rounds 1 + 2.

    Mirrors the sandbox's `_evaluate_chain` + `_select_crossvein` (see
    crossvein_sandbox.py).
    """
    from identify_features.models.topology import CROSSVEIN_CONNECTIONS
    from shapely.geometry import MultiLineString
    from shapely.geometry import Point as _Pt
    from shapely.ops import nearest_points

    # Stub-length cap for chain contraction (matches `_detect_crossveins`).
    if config.um_per_px is not None and config.um_per_px > 0:
        stub_len_floor_px = config.to_px(60.0)
    elif median_vein_width_px > 0:
        stub_len_floor_px = 2.5 * median_vein_width_px
    else:
        stub_len_floor_px = 0.0
    stub_len_cap = max(5.0 * median_vein_width_px, stub_len_floor_px)

    def _is_short_stub(neighbor: int, edge_length: float) -> bool:
        return G.degree(neighbor) == 1 and edge_length < stub_len_cap

    # Build labeled-vein lookups (same as `_detect_crossveins`).
    vein_lines: dict[str, list[LineString]] = defaultdict(list)
    vein_nodes: dict[str, set[int]] = defaultdict(set)
    for (u, v), label in edge_labels.items():
        if G.has_edge(u, v):
            line = G[u][v].get("line")
            if line:
                vein_lines[label].append(line)
            vein_nodes[label].add(u)
            vein_nodes[label].add(v)

    # Group unlabeled edges into chains.
    chains: list[dict] = []
    visited: set[tuple] = set()
    for u, v, data in G.edges(data=True):
        key = _edge_key(u, v)
        if key in visited or key in edge_labels:
            continue
        if data.get("line") is None:
            continue
        chain_edges = [key]
        visited.add(key)
        for start, direction in [(u, v), (v, u)]:
            cur = start
            prev = direction
            while True:
                deg = G.degree(cur)
                if deg == 2:
                    candidates = [n for n in G.neighbors(cur) if n != prev]
                elif deg == 3:
                    forward = [n for n in G.neighbors(cur) if n != prev]
                    if len(forward) != 2:
                        break
                    non_stub = [n for n in forward if not _is_short_stub(n, G[cur][n].get("length_px", 0))]
                    if len(non_stub) != 1:
                        break
                    candidates = non_stub
                else:
                    break
                if not candidates:
                    break
                nxt = candidates[0]
                nkey = _edge_key(cur, nxt)
                if nkey in visited or nkey in edge_labels:
                    break
                if G[cur][nxt].get("line") is None:
                    break
                chain_edges.append(nkey)
                visited.add(nkey)
                prev = cur
                cur = nxt
        touch: dict[int, int] = defaultdict(int)
        for eu, ev in chain_edges:
            touch[eu] += 1
            touch[ev] += 1
        eps = [n for n, c in touch.items() if c == 1]
        if len(eps) != 2:
            continue
        total_len = sum(G[a][b].get("length_px", 0) for a, b in chain_edges if G.has_edge(a, b))
        chains.append(
            {
                "edges": chain_edges,
                "endpoints": (eps[0], eps[1]),
                "length": total_len,
            }
        )

    if not chains:
        return

    # Per-position corridor margin = 0 (strict band between vein_a and vein_b).
    corridor_margin = 0.0
    node_vein_max_dist = (
        max(2.0 * median_vein_width_px, config.to_px(60.0) if (config and config.um_per_px) else 0.0) or 50.0
    )
    h_max_dist = median_vein_width_px or 1.0

    def _node_in_corridor(node: int, vein_a_geom, vein_b_geom, margin: float) -> bool:
        x, y = G.nodes[node]["x"], G.nodes[node]["y"]
        try:
            _, near_a = nearest_points(_Pt(x, y), vein_a_geom)
            _, near_b = nearest_points(_Pt(x, y), vein_b_geom)
        except Exception:
            return False
        y_lo = min(near_a.y, near_b.y) - margin
        y_hi = max(near_a.y, near_b.y) + margin
        return y_lo <= y <= y_hi

    def _all_nodes_in_corridor(edges: list[tuple], vein_a_geom, vein_b_geom, margin: float) -> bool:
        seen: set[int] = set()
        for u, v in edges:
            seen.add(u)
            seen.add(v)
        return all(_node_in_corridor(n, vein_a_geom, vein_b_geom, margin) for n in seen)

    for cv_name, (vein_a, vein_b) in CROSSVEIN_CONNECTIONS.items():
        # Skip if Phase 4-pre0 already labeled this crossvein.
        if any(label == cv_name for label in edge_labels.values()):
            continue
        if vein_a not in vein_lines or vein_b not in vein_lines:
            continue
        vein_a_geom = vein_lines[vein_a][0] if len(vein_lines[vein_a]) == 1 else MultiLineString(vein_lines[vein_a])
        vein_b_geom = vein_lines[vein_b][0] if len(vein_lines[vein_b]) == 1 else MultiLineString(vein_lines[vein_b])

        # Step 1: per-position corridor pool.
        pool: list[dict] = []
        for ch in chains:
            pts = []
            for u, v in ch["edges"]:
                line = G[u][v].get("line") if G.has_edge(u, v) else None
                if line is not None:
                    pts.extend(line.coords)
            if not pts:
                continue
            cx = sum(p[0] for p in pts) / len(pts)
            cy = sum(p[1] for p in pts) / len(pts)
            try:
                _, na = nearest_points(_Pt(cx, cy), vein_a_geom)
                _, nb = nearest_points(_Pt(cx, cy), vein_b_geom)
            except Exception:
                continue
            y_lo = min(na.y, nb.y) - corridor_margin
            y_hi = max(na.y, nb.y) + corridor_margin
            if y_lo <= cy <= y_hi:
                pool.append(ch)
        if not pool:
            continue

        # Round 1 — H-shape (1× mvw tolerance). Round 2 — partial CVs.
        h_candidates: list[dict] = []
        partials_a: list[dict] = []
        partials_b: list[dict] = []
        for ch in pool:
            ea, eb = ch["endpoints"]
            da_a = _node_vein_distance(G, ea, vein_a, vein_lines, vein_nodes, max_dist=h_max_dist)
            db_a = _node_vein_distance(G, eb, vein_a, vein_lines, vein_nodes, max_dist=h_max_dist)
            da_b = _node_vein_distance(G, ea, vein_b, vein_lines, vein_nodes, max_dist=h_max_dist)
            db_b = _node_vein_distance(G, eb, vein_b, vein_lines, vein_nodes, max_dist=h_max_dist)
            h_orient_1 = da_a is not None and db_b is not None
            h_orient_2 = db_a is not None and da_b is not None
            if h_orient_1 or h_orient_2:
                h_candidates.append({"edges": ch["edges"], "endpoints": ch["endpoints"], "length": ch["length"]})
                continue
            on_a_count = int(da_a is not None) + int(db_a is not None)
            on_b_count = int(da_b is not None) + int(db_b is not None)
            if on_a_count == 1 and on_b_count == 0:
                anchor = ea if da_a is not None else eb
                free = eb if da_a is not None else ea
                partials_a.append(
                    {
                        "edges": ch["edges"],
                        "endpoints": ch["endpoints"],
                        "length": ch["length"],
                        "anchor": anchor,
                        "free": free,
                    }
                )
            elif on_b_count == 1 and on_a_count == 0:
                anchor = ea if da_b is not None else eb
                free = eb if da_b is not None else ea
                partials_b.append(
                    {
                        "edges": ch["edges"],
                        "endpoints": ch["endpoints"],
                        "length": ch["length"],
                        "anchor": anchor,
                        "free": free,
                    }
                )

        candidates: list[dict] = list(h_candidates)
        if not h_candidates:
            # Round 2 pair-merge through corridor-pool unlabeled bridges.
            pool_edges_set: set[tuple] = set()
            for ev in pool:
                for u, v in ev["edges"]:
                    pool_edges_set.add(_edge_key(u, v))
            pool_sub = nx.Graph()
            for u, v, data in G.edges(data=True):
                if _edge_key(u, v) in pool_edges_set:
                    pool_sub.add_edge(u, v, **data)
            used_partial_keys: set[tuple] = set()
            for pa in partials_a:
                for pb in partials_b:
                    shared = set(pa["endpoints"]) & set(pb["endpoints"])
                    bridge_edges: list[tuple] = []
                    if shared:
                        pass  # adjacent — empty bridge
                    else:
                        bridge_g = pool_sub.copy()
                        for u, v in pa["edges"]:
                            if bridge_g.has_edge(u, v):
                                bridge_g.remove_edge(u, v)
                        for u, v in pb["edges"]:
                            if bridge_g.has_edge(u, v):
                                bridge_g.remove_edge(u, v)
                        if not (bridge_g.has_node(pa["free"]) and bridge_g.has_node(pb["free"])):
                            continue
                        try:
                            path = nx.shortest_path(bridge_g, pa["free"], pb["free"], weight="length_px")
                        except nx.NetworkXNoPath:
                            continue
                        bridge_edges = [_edge_key(path[i], path[i + 1]) for i in range(len(path) - 1)]
                    merged = list(pa["edges"]) + bridge_edges + list(pb["edges"])
                    seen_edges: set[tuple] = set()
                    unique_edges = []
                    for e in merged:
                        if e in seen_edges:
                            continue
                        seen_edges.add(e)
                        unique_edges.append(e)
                    if not _all_nodes_in_corridor(unique_edges, vein_a_geom, vein_b_geom, corridor_margin):
                        continue
                    total_len = sum(G[u][v].get("length_px", 0) for u, v in unique_edges if G.has_edge(u, v))
                    candidates.append(
                        {"edges": unique_edges, "endpoints": (pa["anchor"], pb["anchor"]), "length": total_len}
                    )
                    used_partial_keys.update(tuple(e) for e in pa["edges"])
                    used_partial_keys.update(tuple(e) for e in pb["edges"])
            # Orphans
            for p in partials_a + partials_b:
                if any(tuple(e) in used_partial_keys for e in p["edges"]):
                    continue
                candidates.append({"edges": p["edges"], "endpoints": p["endpoints"], "length": p["length"]})

        if not candidates:
            continue

        # Final picker: lowest sum-distance over both endpoint orientations.
        best = None
        best_score = float("inf")
        for c in candidates:
            ea, eb = c["endpoints"]
            for orient_a, orient_b in [(ea, eb), (eb, ea)]:
                da = _node_vein_distance(G, orient_a, vein_a, vein_lines, vein_nodes, max_dist=node_vein_max_dist)
                db = _node_vein_distance(G, orient_b, vein_b, vein_lines, vein_nodes, max_dist=node_vein_max_dist)
                if da is not None and db is not None and (da + db) < best_score:
                    best_score = da + db
                    best = c
        if best is None:
            continue

        for key in best["edges"]:
            if key not in edge_labels:
                edge_labels[key] = cv_name
        logger.info(
            "Phase 4-pre1: %s labeled via chain-topology (%d edge(s), %.0fpx, score=%.1f)",
            cv_name,
            len(best["edges"]),
            best["length"],
            best_score,
        )


def _detect_crossveins(
    G: nx.Graph,
    edge_labels: dict[tuple, str],
    config: Optional["PipelineConfig"] = None,
    median_vein_width_px: float = 0.0,
) -> None:
    """Detect ACV and PCV crossveins.

    After junction merging (Phase 1), crossvein edges are unlabeled
    branches whose endpoints sit on or near two different longitudinal
    veins.  ACV connects L3↔L4, PCV connects L4↔L5.

    Detection: group unlabeled edges into chains by contracting
    degree-2 pass-through nodes (and degree-3 nodes whose third neighbor
    is a short dead-end stub — same stub guard as `merge_through_junctions`
    in junction_resolver.py), then for each crossvein find the chain
    whose two outer endpoints best match vein_a and vein_b AND whose
    bounding box lies inside the (vein_a, vein_b) corridor — i.e. ACV
    must sit between L3 and L4 longitudinally (in AP order), PCV between
    L4 and L5. The corridor gate prevents the detector from grabbing a
    distant unlabeled chain whose endpoints happen to score well purely
    by coincidence.

    "Near" means the node is an endpoint of a labeled edge (shared
    graph node) or its coordinates lie on/close to a labeled vein
    LineString (typical after junction merging contracts the junction
    node into the longitudinal's line).
    """
    from identify_features.models.topology import CROSSVEIN_CONNECTIONS

    # Stub-length cap: same anatomical floor as junction_resolver.merge_through_junctions
    # so this detector and the junction merger agree on what counts as a stub.
    if config is not None and config.um_per_px is not None and config.um_per_px > 0:
        stub_len_floor_px = config.to_px(60.0)
    elif median_vein_width_px > 0:
        stub_len_floor_px = 2.5 * median_vein_width_px
    else:
        stub_len_floor_px = 0.0
    stub_len_cap = max(5.0 * median_vein_width_px, stub_len_floor_px)

    def _is_short_stub(neighbor: int, edge_length: float) -> bool:
        return G.degree(neighbor) == 1 and edge_length < stub_len_cap

    # Build lookup: labeled vein LineStrings and endpoint node sets
    vein_lines: dict[str, list[LineString]] = defaultdict(list)
    vein_nodes: dict[str, set[int]] = defaultdict(set)
    for (u, v), label in edge_labels.items():
        if G.has_edge(u, v):
            line = G[u][v].get("line")
            if line:
                vein_lines[label].append(line)
            vein_nodes[label].add(u)
            vein_nodes[label].add(v)

    # Group unlabeled edges into chains via degree-2 contraction.
    # A chain is a maximal run of unlabeled edges connected through
    # degree-2 nodes; it terminates at any leaf or degree-3+ junction.
    chains: list[dict] = []
    visited: set[tuple] = set()
    for u, v, data in G.edges(data=True):
        key = _edge_key(u, v)
        if key in visited or key in edge_labels:
            continue
        if data.get("line") is None:
            continue

        chain_edges = [key]
        visited.add(key)
        for start, direction in [(u, v), (v, u)]:
            cur = start
            prev = direction
            while True:
                deg = G.degree(cur)
                if deg == 2:
                    candidates = [n for n in G.neighbors(cur) if n != prev]
                elif deg == 3:
                    # Bridge across a degree-3 node when exactly one of the two
                    # forward neighbors is a short dead-end stub. Mirrors the
                    # stub guard in junction_resolver.merge_through_junctions
                    # (commit b5a8253) so chain contraction here doesn't break
                    # apart over the same kind of ectopic-stub junctions.
                    forward = [n for n in G.neighbors(cur) if n != prev]
                    if len(forward) != 2:
                        break
                    non_stub = [n for n in forward if not _is_short_stub(n, G[cur][n].get("length_px", 0))]
                    if len(non_stub) != 1:
                        break
                    candidates = non_stub
                else:
                    break
                if not candidates:
                    break
                nxt = candidates[0]
                nkey = _edge_key(cur, nxt)
                if nkey in visited or nkey in edge_labels:
                    break
                if G[cur][nxt].get("line") is None:
                    break
                chain_edges.append(nkey)
                visited.add(nkey)
                prev = cur
                cur = nxt

        # A chain endpoint is a node touched by exactly one chain edge.
        touch_count: dict[int, int] = defaultdict(int)
        for eu, ev in chain_edges:
            touch_count[eu] += 1
            touch_count[ev] += 1
        endpoints = [n for n, c in touch_count.items() if c == 1]
        if len(endpoints) != 2:
            # Closed loop or degenerate — cannot match endpoint pair against a crossvein
            continue

        total_len = sum(G[a][b].get("length_px", 0) for a, b in chain_edges if G.has_edge(a, b))
        chains.append(
            {
                "edges": chain_edges,
                "endpoints": (endpoints[0], endpoints[1]),
                "length": total_len,
                "claimed": False,
            }
        )

    # Margin for the corridor gate — a chain whose bounding box pokes a few
    # vein widths outside the [vein_a, vein_b] AP corridor is still plausibly
    # a real crossvein (anchor noise, label-line jitter), but a chain that
    # sits entirely on the wrong side of one of the boundary veins is not.
    # Primary: 2× median vein width. Fallback when vein width is unknown:
    # 60 µm anatomical floor (matches the stub-length floor used elsewhere).
    corridor_margin = max(
        2.0 * median_vein_width_px,
        config.to_px(60.0) if (config and config.um_per_px) else 0.0,
    )
    # "Near a labeled vein" radius for chain endpoint scoring. Primary:
    # 2× median vein width. Fallback: 60 µm. The fallback to a hard 50 px
    # default in `_node_vein_distance` is only hit when this scope itself
    # has neither vein width nor µm scale.
    node_vein_max_dist = (
        max(
            2.0 * median_vein_width_px,
            config.to_px(60.0) if (config and config.um_per_px) else 0.0,
        )
        or 50.0
    )

    for cv_name, (vein_a, vein_b) in CROSSVEIN_CONNECTIONS.items():
        # Skip if Phase 4-pre0 / 4-pre1 already labeled this crossvein.
        if any(label == cv_name for label in edge_labels.values()):
            continue
        if vein_a not in vein_lines or vein_b not in vein_lines:
            logger.info("Cannot detect %s: %s or %s not labeled", cv_name, vein_a, vein_b)
            continue

        # Build the AP-axis corridor from the two longitudinal veins. We use
        # mean Y on each vein since L3/L4/L5 are roughly horizontal in the
        # canonical wing pose; the corridor gate keeps the chain's bbox-Y
        # within the [min, max] of the two veins (± margin).
        a_pts = [pt for ln in vein_lines[vein_a] for pt in ln.coords]
        b_pts = [pt for ln in vein_lines[vein_b] for pt in ln.coords]
        if not a_pts or not b_pts:
            logger.info("Cannot detect %s: missing geometry for %s/%s", cv_name, vein_a, vein_b)
            continue
        a_y_mean = sum(p[1] for p in a_pts) / len(a_pts)
        b_y_mean = sum(p[1] for p in b_pts) / len(b_pts)
        corridor_y_lo = min(a_y_mean, b_y_mean) - corridor_margin
        corridor_y_hi = max(a_y_mean, b_y_mean) + corridor_margin

        best_chain = None
        best_score = float("inf")
        rejected_corridor = 0

        for chain in chains:
            if chain["claimed"]:
                continue

            # Corridor gate: the chain's centroid Y must lie within the
            # AP corridor between vein_a and vein_b. This blocks distant
            # unlabeled chains (e.g. a hinge-area stub or a chain on the
            # opposite side of a longitudinal vein) from winning the
            # endpoint-distance tournament when their endpoints happen to
            # sit close to (vein_a, vein_b) by coincidence.
            chain_pts = []
            for a, b in chain["edges"]:
                line = G[a][b].get("line") if G.has_edge(a, b) else None
                if line is not None:
                    chain_pts.extend(line.coords)
            if not chain_pts:
                continue
            chain_y_mean = sum(p[1] for p in chain_pts) / len(chain_pts)
            if not (corridor_y_lo <= chain_y_mean <= corridor_y_hi):
                rejected_corridor += 1
                continue

            end_a, end_b = chain["endpoints"]
            # Try both orientations
            for ea, eb in [(end_a, end_b), (end_b, end_a)]:
                dist_a = _node_vein_distance(G, ea, vein_a, vein_lines, vein_nodes, max_dist=node_vein_max_dist)
                dist_b = _node_vein_distance(G, eb, vein_b, vein_lines, vein_nodes, max_dist=node_vein_max_dist)
                if dist_a is not None and dist_b is not None:
                    score = dist_a + dist_b
                    if score < best_score:
                        best_score = score
                        best_chain = chain
                    break  # valid orientation found

        if best_chain is not None:
            for key in best_chain["edges"]:
                edge_labels[key] = cv_name
            best_chain["claimed"] = True
            logger.info(
                "Detected %s: %d-edge chain, %.0fpx total (score=%.1f, %d chain(s) rejected by corridor)",
                cv_name,
                len(best_chain["edges"]),
                best_chain["length"],
                best_score,
                rejected_corridor,
            )
        else:
            logger.info(
                "No candidate found for %s (%d chain(s) rejected by corridor)",
                cv_name,
                rejected_corridor,
            )


def _extend_crossveins_along_corridor(
    G: nx.Graph,
    edge_labels: dict[tuple, str],
    median_vein_width_px: float = 0.0,
    config: Optional["PipelineConfig"] = None,
) -> None:
    """Greedily absorb unlabeled edges into a labeled crossvein when the
    edge shares a graph node with the crossvein, sits within a small
    distance of the labeled crossvein's existing centerline, AND points
    in a direction that continues the labeled crossvein (rather than
    branching off it).

    Repeats until no more absorptions qualify. Catches cases where the
    chain detector / fallback labels only a fragment of the real crossvein
    (e.g. the fallback labels just the segment touching PCV.a) and the
    rest of the crossvein stays unlabeled, eventually getting promoted to
    EV*. Common when a degree-3 kink in the skeleton broke the original
    crossvein chain into pieces that don't all anchor to vein_a / vein_b
    by themselves.

    Three guards keep absorption conservative:
      1. Shared graph node — the candidate must share an endpoint with an
         already-labeled crossvein edge.
      2. Distance to labeled centerline — the candidate's centroid must
         be within ~max(4× median vein width, 200 µm) of the existing
         labeled crossvein geometry. Without this, an edge that shares a
         junction node with PCV but extends far from PCV's actual
         centerline gets erroneously pulled in.
      3. Departure-angle gate — the candidate edge's direction at the
         shared node must be roughly opposite to the labeled crossvein's
         direction at that node (i.e., the candidate continues the
         crossvein across the junction). An ectopic branch off PCV
         points in a different direction (T- or Y-junction) and gets
         rejected even when it would otherwise pass the centerline-
         distance gate.

    The AP corridor band that gates `_detect_crossveins` is intentionally
    NOT applied here: when L4/L5 sweep diagonally, the band's mean-Y
    representation collapses to a tight slice (sometimes <30 px) and
    rejects PCV continuation pieces whose Y is outside that slice but
    correctly between L4 and L5 locally.
    """
    from identify_features.models.topology import CROSSVEIN_CONNECTIONS
    from identify_features.utils.geometry_utils import angle_between_vectors
    from identify_features.utils.graph_utils import edge_departure_direction
    from shapely.geometry import MultiLineString
    from shapely.geometry import Point as _Pt

    # Distance from candidate edge centroid to labeled crossvein centerline.
    # Primary: 4× median vein width. Fallback: 200 µm (anatomical scale —
    # roughly twice a typical median vein width on standard wing imagery).
    centerline_distance_cap = max(
        4.0 * median_vein_width_px,
        config.to_px(200.0) if (config and config.um_per_px) else 0.0,
    )
    # Direction sample window for departure-direction computation.
    direction_window = (
        config.departure_sample_px(median_vein_width_px)
        if config is not None
        else max(median_vein_width_px * 4.0, 50.0)
    )
    # Continuation-angle threshold (degrees). Both edges leave the shared
    # node going outward; a candidate is "collinear with the already-
    # labeled crossvein" only if the two outward directions point in
    # roughly opposite directions (close to 180°). Reuses the project's
    # standard collinear threshold (`config.collinear_min_angle`,
    # default 150°), so absorption requires the candidate to be within
    # 30° of a perfectly straight continuation. Rejects every form of
    # branch (T-, Y-, kink) that doesn't actually continue the crossvein.
    min_continuation_angle = config.collinear_min_angle if config is not None else 150.0

    # Reuse the same lookups as _detect_crossveins.
    vein_lines: dict[str, list[LineString]] = defaultdict(list)
    vein_nodes: dict[str, set[int]] = defaultdict(set)
    for (u, v), label in edge_labels.items():
        if G.has_edge(u, v):
            line = G[u][v].get("line")
            if line:
                vein_lines[label].append(line)
            vein_nodes[label].add(u)
            vein_nodes[label].add(v)

    for cv_name, vein_pair in CROSSVEIN_CONNECTIONS.items():
        if cv_name not in vein_lines or not vein_lines[cv_name]:
            continue
        vein_a, vein_b = vein_pair

        # Track a growing geometry of the labeled crossvein for the
        # centerline-distance gate. MultiLineString is fine even if there's
        # only one line.
        cv_geom = vein_lines[cv_name][0] if len(vein_lines[cv_name]) == 1 else MultiLineString(vein_lines[cv_name])

        cv_nodes = set(vein_nodes[cv_name])  # mutable copy
        cv_lines_list = list(vein_lines[cv_name])

        def _opposite_anchor_nodes() -> set[int]:
            """Return graph nodes labeled as the anchor vein this crossvein
            does NOT yet touch, or empty set if both/neither are touched.
            Re-evaluated on each absorption attempt because cv_nodes grows."""
            touched_a = bool(cv_nodes & vein_nodes.get(vein_a, set()))
            touched_b = bool(cv_nodes & vein_nodes.get(vein_b, set()))
            if touched_a and not touched_b:
                return vein_nodes.get(vein_b, set())
            if touched_b and not touched_a:
                return vein_nodes.get(vein_a, set())
            return set()

        def _branch_reaches(start: int, target_nodes: set[int]) -> bool:
            """BFS from start through unlabeled non-cv edges. True iff any
            reachable node is in target_nodes."""
            if not target_nodes:
                return False
            if start in target_nodes:
                return True
            visited: set[int] = {start}
            frontier: list[int] = [start]
            while frontier:
                next_frontier: list[int] = []
                for node in frontier:
                    for nbr in G.neighbors(node):
                        if nbr in visited:
                            continue
                        nkey = _edge_key(node, nbr)
                        if nkey in edge_labels:
                            # Don't traverse already-labeled edges; they
                            # may belong to other veins.
                            continue
                        if nbr in target_nodes:
                            return True
                        visited.add(nbr)
                        next_frontier.append(nbr)
                frontier = next_frontier
            return False

        absorbed = 0
        changed = True
        while changed:
            changed = False
            opposite_nodes = _opposite_anchor_nodes()
            for u, v in list(G.edges()):
                key = _edge_key(u, v)
                if key in edge_labels:
                    continue
                line = G[u][v].get("line")
                if line is None:
                    continue
                if u not in cv_nodes and v not in cv_nodes:
                    continue
                pts = list(line.coords)
                cx = sum(p[0] for p in pts) / len(pts)
                cy = sum(p[1] for p in pts) / len(pts)
                centroid = _Pt(cx, cy)
                # Centerline-distance gate: candidate must hug the existing
                # labeled crossvein.
                if centroid.distance(cv_geom) > centerline_distance_cap:
                    continue

                shared_node = u if u in cv_nodes else v
                other_node = v if shared_node == u else u

                # Reach-opposite-vein gate: if walking from this branch
                # through unlabeled edges eventually arrives at a node on
                # the opposite anchor vein, the branch is the genuine
                # crossvein continuation regardless of departure angle.
                # This rescues cases where an ectopic stub at a Y-junction
                # is more collinear than the real branch (e.g. PCV on
                # std30 0001.bmp where the ectopic is near-perfectly
                # collinear and the L5-reaching branch bends away).
                if _branch_reaches(other_node, opposite_nodes):
                    edge_labels[key] = cv_name
                    cv_nodes.add(u)
                    cv_nodes.add(v)
                    cv_lines_list.append(line)
                    cv_geom = MultiLineString(cv_lines_list) if len(cv_lines_list) > 1 else cv_lines_list[0]
                    absorbed += 1
                    changed = True
                    continue

                # Collinearity gate (fallback when the branch doesn't reach
                # the opposite anchor): identify the shared node's labeled-
                # crossvein neighbor and require the candidate's outward
                # direction to be roughly opposite (collinear) to the
                # labeled crossvein's outward direction.
                cv_neighbor = None
                for nbr in G.neighbors(shared_node):
                    if nbr == other_node:
                        continue
                    if edge_labels.get(_edge_key(shared_node, nbr)) == cv_name:
                        cv_neighbor = nbr
                        break
                if cv_neighbor is None:
                    continue
                cand_dep = edge_departure_direction(G, shared_node, other_node, direction_window)
                cv_dep = edge_departure_direction(G, shared_node, cv_neighbor, direction_window)
                if cand_dep is None or cv_dep is None:
                    continue
                angle = angle_between_vectors(cand_dep, cv_dep)
                if angle < min_continuation_angle:
                    continue

                edge_labels[key] = cv_name
                cv_nodes.add(u)
                cv_nodes.add(v)
                cv_lines_list.append(line)
                cv_geom = MultiLineString(cv_lines_list) if len(cv_lines_list) > 1 else cv_lines_list[0]
                absorbed += 1
                changed = True
        if absorbed:
            logger.info("Extended %s: absorbed %d additional edge(s)", cv_name, absorbed)


def _detect_crossveins_via_junctions(
    G: nx.Graph,
    edge_labels: dict[tuple, str],
) -> None:
    """Detect crossveins by tracing unlabeled paths between longitudinal junctions.

    Analogous to _extend_to_distal_landmarks for longitudinals: instead of
    using landmark points, finds degree-3+ nodes on labeled longitudinals
    that have unlabeled branches, then traces unlabeled paths between them.

    ACV: unlabeled path from a junction on L3 to a junction on L4.
    PCV: unlabeled path from a junction on L4 to a junction on L5.

    Handles multi-edge crossveins that pass through degree-2 nodes.
    """
    from identify_features.models.topology import CROSSVEIN_CONNECTIONS

    # Build node sets for each labeled vein
    vein_nodes: dict[str, set[int]] = defaultdict(set)
    for (u, v), label in edge_labels.items():
        if G.has_edge(u, v):
            vein_nodes[label].add(u)
            vein_nodes[label].add(v)

    for cv_name, (vein_a, vein_b) in CROSSVEIN_CONNECTIONS.items():
        # Skip if already found
        if any(label == cv_name for label in edge_labels.values()):
            continue

        nodes_a = vein_nodes.get(vein_a, set())
        nodes_b = vein_nodes.get(vein_b, set())
        if not nodes_a or not nodes_b:
            continue

        # Find degree-3+ junctions on vein_a with unlabeled branches
        starts: list[tuple[int, int, tuple]] = []  # (junction_node, first_nbr, edge_key)
        for node in nodes_a:
            if G.degree(node) < 3:
                continue
            for nbr in G.neighbors(node):
                key = _edge_key(node, nbr)
                if key not in edge_labels:
                    starts.append((node, nbr, key))

        # BFS from each unlabeled branch to find paths reaching vein_b
        best_path: list[tuple] | None = None
        best_length = float("inf")

        for start_node, first_nbr, first_key in starts:
            visited_nodes = {start_node}
            # Queue: (current_node, list_of_edge_keys, total_length)
            queue = [(first_nbr, [first_key], G[start_node][first_nbr].get("length_px", 0))]

            while queue:
                current, path, total_len = queue.pop(0)
                if current in visited_nodes:
                    continue
                visited_nodes.add(current)

                # Check if we reached vein_b
                if current in nodes_b:
                    if total_len < best_length:
                        best_length = total_len
                        best_path = path
                    continue  # Don't explore further past vein_b

                # Continue BFS through unlabeled edges only
                for nbr in G.neighbors(current):
                    if nbr in visited_nodes:
                        continue
                    key = _edge_key(current, nbr)
                    if key in edge_labels:
                        continue  # Don't cross labeled edges
                    edge_len = G[current][nbr].get("length_px", 0)
                    queue.append((nbr, path + [key], total_len + edge_len))

        if best_path is not None:
            for key in best_path:
                edge_labels[key] = cv_name
            logger.info(
                "Detected %s (via junctions): %d edges, %.0fpx total",
                cv_name,
                len(best_path),
                best_length,
            )


def _node_vein_distance(
    G: nx.Graph,
    node: int,
    vein_label: str,
    vein_lines: dict[str, list[LineString]],
    vein_nodes: dict[str, set[int]],
    max_dist: float = 50.0,
) -> Optional[float]:
    """Distance from a graph node to a labeled vein.

    Returns 0 if the node shares a graph edge endpoint with the vein.
    Returns geometric distance to the nearest vein LineString if within
    *max_dist* (covers the post-merge case where the junction node's
    coordinates are embedded in the merged longitudinal line).
    Returns None if too far — the node is not connected to this vein.
    """
    # Direct graph connectivity: node is an endpoint of a vein edge
    if node in vein_nodes.get(vein_label, set()):
        return 0.0

    # Geometric proximity (post-merge: node coords on the LineString)
    pt = Point(G.nodes[node]["x"], G.nodes[node]["y"])
    min_dist = float("inf")
    for line in vein_lines.get(vein_label, []):
        d = line.distance(pt)
        if d < min_dist:
            min_dist = d

    if min_dist <= max_dist:
        return min_dist

    return None


def _detect_crossveins_fallback(
    G: nx.Graph,
    edge_labels: dict[tuple, str],
    landmarks: dict[str, Landmark],
    config: "PipelineConfig",
    median_vein_width: float,
) -> None:
    """Fallback crossvein detection using crossvein landmark points.

    Runs after primary detection. For each crossvein not yet found:
    - Tier 2: Use the more reliable landmark (ACV.p for ACV, PCV.a for PCV)
    - Tier 3: Use the less reliable landmark (ACV.a for ACV, PCV.p for PCV)

    Candidates are scored by length (prefer crossvein-sized) and
    perpendicularity to nearby labeled longitudinals.
    """
    from identify_features.utils.geometry_utils import (
        angle_between_vectors,
        line_direction,
    )

    min_len = median_vein_width * config.crossvein_min_length_vw
    max_len = median_vein_width * config.crossvein_max_length_vw
    search_radius = config.snap_radius_px(median_vein_width)
    sample_px = config.departure_sample_px(median_vein_width)

    # Build labeled vein lines for perpendicularity checks
    vein_lines: dict[str, list[LineString]] = defaultdict(list)
    for (u, v), label in edge_labels.items():
        if G.has_edge(u, v):
            line = G[u][v].get("line")
            if line:
                vein_lines[label].append(line)

    # Crossvein landmark tiers: (cv_name, [(landmark_name, adjacent_longitudinals), ...])
    cv_tiers = {
        "ACV": [("ACV.p", ["L3", "L4"]), ("ACV.a", ["L3", "L4"])],
        "PCV": [("PCV.a", ["L4", "L5"]), ("PCV.p", ["L4", "L5"])],
    }

    for cv_name, tiers in cv_tiers.items():
        # Skip if already found by primary detection
        if any(label == cv_name for label in edge_labels.values()):
            continue

        for lm_name, adj_veins in tiers:
            lm = landmarks.get(lm_name)
            if lm is None:
                continue

            # Find unlabeled edges near this landmark
            candidates = []
            for u, v, data in G.edges(data=True):
                key = _edge_key(u, v)
                if key in edge_labels:
                    continue
                line = data.get("line")
                if line is None:
                    continue
                length = data.get("length_px", 0)

                # Length filter
                if length < min_len or length > max_len:
                    continue

                # Distance to landmark
                dist = line.distance(lm.point)
                if dist > search_radius:
                    continue

                # Perpendicularity score against adjacent longitudinals
                perp_score = 0.0
                perp_count = 0
                edge_dir = line_direction(line, sample_px=line.length)

                for adj_vein in adj_veins:
                    for adj_line in vein_lines.get(adj_vein, []):
                        # Find direction of longitudinal at nearest point to candidate
                        mid = line.interpolate(0.5, normalized=True)
                        proj_dist = adj_line.project(mid)
                        if proj_dist <= 0 or proj_dist >= adj_line.length:
                            continue
                        # Sample longitudinal direction at the projected point
                        half_win = min(sample_px / 2, proj_dist, adj_line.length - proj_dist)
                        if half_win < 1:
                            continue
                        pt_a = adj_line.interpolate(proj_dist - half_win)
                        pt_b = adj_line.interpolate(proj_dist + half_win)
                        long_dir = (pt_b.x - pt_a.x, pt_b.y - pt_a.y)
                        mag = (long_dir[0] ** 2 + long_dir[1] ** 2) ** 0.5
                        if mag < 1e-6:
                            continue
                        long_dir = (long_dir[0] / mag, long_dir[1] / mag)

                        angle = angle_between_vectors(edge_dir, long_dir)
                        # Normalize to 0-90 (direction doesn't matter)
                        if angle > 90:
                            angle = 180 - angle
                        # Score: 1.0 at 90° (perfect perp), 0.0 at 0° (parallel)
                        perp_score += angle / 90.0
                        perp_count += 1

                if perp_count > 0:
                    perp_score /= perp_count
                else:
                    perp_score = 0.5  # no longitudinal to check — neutral

                # Length score: 1.0 at ideal length, lower at extremes
                ideal_len = (min_len + max_len) / 2
                length_score = 1.0 - abs(length - ideal_len) / ideal_len
                length_score = max(0.0, length_score)

                # Combined: proximity + perpendicularity + length
                score = dist + (1 - perp_score) * 200 + (1 - length_score) * 100
                candidates.append((key, score, length, dist, perp_score))

            if candidates:
                candidates.sort(key=lambda c: c[1])
                best_key, best_score, best_len, best_dist, best_perp = candidates[0]
                edge_labels[best_key] = cv_name
                u, v = best_key
                logger.info(
                    "Detected %s (fallback via %s): edge %d↔%d, %.0fpx, dist=%.0f, perp=%.2f",
                    cv_name,
                    lm_name,
                    u,
                    v,
                    best_len,
                    best_dist,
                    best_perp,
                )
                break  # Found it, don't try next tier


def _label_ectopic_edges(
    G: nx.Graph,
    edge_labels: dict[tuple, str],
    median_vein_width_px: float,
    config: "PipelineConfig",
) -> int:
    """Promote still-unlabeled edges to ectopic veins (EV1, EV2, ...).

    Each connected component of unlabeled edges becomes one EV. Components
    whose total length falls below the config-driven noise floor are
    silently dropped. Ordering is deterministic: longest component first,
    tie-break on minimum node id.
    """
    noise_floor = config.ectopic_min_length_px(median_vein_width_px)

    unlabeled_edges = [
        (u, v) for u, v in G.edges() if _edge_key(u, v) not in edge_labels and G[u][v].get("line") is not None
    ]
    if not unlabeled_edges:
        return 0

    H = nx.Graph()
    H.add_edges_from(unlabeled_edges)
    components = list(nx.connected_components(H))

    def _comp_length(nodes: set[int]) -> float:
        return sum(G[u][v]["line"].length for u, v in H.subgraph(nodes).edges())

    scored = [(c, _comp_length(c)) for c in components]
    kept = [(c, L) for c, L in scored if L >= noise_floor]
    dropped = len(scored) - len(kept)
    if dropped:
        logger.info(
            "Dropped %d sub-threshold unlabeled components (<%.0fpx)",
            dropped,
            noise_floor,
        )

    kept.sort(key=lambda cL: (-cL[1], min(cL[0])))

    for idx, (nodes, length) in enumerate(kept, start=1):
        name = f"EV{idx}"
        sub = H.subgraph(nodes)
        for u, v in sub.edges():
            edge_labels[_edge_key(u, v)] = name
        logger.info("%s: %d edges, %.0fpx total", name, sub.number_of_edges(), length)

    return len(kept)


def _edge_key(u: int, v: int) -> tuple[int, int]:
    return (min(u, v), max(u, v))


def _vein_type(vein_id: str) -> VeinType:
    if vein_id == "Rs":
        return VeinType.RADIAL_SECTOR
    elif vein_id in ("ACV", "PCV"):
        return VeinType.CROSSVEIN
    elif vein_id == "costa":
        return VeinType.COSTA
    else:
        return VeinType.LONGITUDINAL


def _assign_absent_and_partial(
    veins: list[VeinIdentification],
    G: nx.Graph,
    edge_labels: dict[tuple, str],
    landmarks: dict[str, Landmark],
    median_vein_width_px: float,
    config: PipelineConfig,
) -> None:
    """Mark canonical veins entirely missing ABSENT (appended as
    centerline-None placeholders) and present-but-incomplete veins PARTIAL —
    longitudinals AND crossveins. Never touches ECTOPIC / INFERRED veins.

    Two PARTIAL signals are evaluated for each IDENTIFIED vein:

    * **gapped** — its labelled edges form more than one connected component
      in the skeleton graph. Applied to every vein type, including costa and
      L6 (which have no clean endpoint definition).
    * **truncated** — for longitudinals listed in ``LONGITUDINAL_ENDPOINTS``,
      at least one *reliable* anchor landmark is not reached by any of the
      vein's labelled edges (line-to-point distance > ``search_radius``).
      For crossveins listed in ``CROSSVEIN_CONNECTIONS``, the vein's
      centerline doesn't reach a *present* bounding longitudinal (centerline-
      to-centerline distance > ``search_radius``). A bounding vein that is
      itself absent is skipped — that side simply can't be judged.

    ``search_radius`` is ``median_vein_width_px * partial_endpoint_search_vw``,
    a dedicated knob (default 3.0) separate from Phase 2c's
    ``distal_landmark_search_vw`` (default 2.0) so the partial/absent
    sensitivity can be tuned in Advanced Settings without changing tracing
    behaviour upstream.

    ABSENT veins are then appended as explicit placeholders: any name in
    ``ALL_CANONICAL_VEINS`` with no surviving ``VeinIdentification`` gets a
    fresh entry with ``centerline=None`` and ``length_px=0.0``. Downstream
    consumers (overlays, GeoJSON, AP split, intervein tissue assignment, CSV)
    already guard ``centerline is None`` so the new rows pass straight through
    to the long-format CSV's ``status`` column.
    """
    veins_by_id = {v.vein_id: v for v in veins}
    present = set(veins_by_id)
    search_radius = median_vein_width_px * config.partial_endpoint_search_vw

    def _edges_for(vid: str) -> list[tuple]:
        return [
            (u, w) for (u, w), lbl in edge_labels.items() if lbl == vid and G.has_edge(u, w)
        ]

    # ---- PARTIAL: downgrade IDENTIFIED veins that are truncated or gapped ----
    for v in veins:
        if v.status != VeinStatus.IDENTIFIED:
            continue  # leave ECTOPIC / INFERRED alone
        edges = _edges_for(v.vein_id)

        # (a) gapped: labelled edges form >1 connected component (every vein type).
        sub = nx.Graph()
        sub.add_edges_from(edges)
        gapped = sub.number_of_nodes() > 0 and nx.number_connected_components(sub) > 1

        # (b) truncated: doesn't reach its expected endpoints.
        truncated = False
        if v.vein_id in LONGITUDINAL_ENDPOINTS:
            for lm_name in LONGITUDINAL_ENDPOINTS[v.vein_id]:  # (proximal, distal)
                lm = landmarks.get(lm_name)
                if lm is None or not getattr(lm, "reliable", False):
                    continue  # can't judge this endpoint
                reaches = any(
                    G[u][w].get("line") is not None
                    and G[u][w]["line"].distance(lm.point) <= search_radius
                    for u, w in edges
                )
                if not reaches:
                    truncated = True
                    break
        elif v.vein_id in CROSSVEIN_CONNECTIONS and v.centerline is not None:
            for bounding_id in CROSSVEIN_CONNECTIONS[v.vein_id]:  # e.g. ACV -> (L3, L4)
                b = veins_by_id.get(bounding_id)
                if b is None or b.centerline is None:
                    continue  # bounding vein absent / not yet present -> skip this side
                if v.centerline.distance(b.centerline) > search_radius:
                    truncated = True
                    break

        if gapped or truncated:
            v.status = VeinStatus.PARTIAL
            v.evidence.append("gapped" if gapped else "truncated (endpoint not reached)")

    # ---- ABSENT: canonical veins with no VeinIdentification at all ----
    # Always emitted as explicit rows (project decision — makes "is L4 present?"
    # a direct CSV column query rather than an inference from a missing row).
    for vein_id in ALL_CANONICAL_VEINS:
        if vein_id in present:
            continue
        veins.append(
            VeinIdentification(
                vein_id=vein_id,
                vein_type=_vein_type(vein_id),
                status=VeinStatus.ABSENT,
                centerline=None,
                length_px=0.0,
                evidence=["no labelled path"],
            )
        )


def _synthesize_crossveins_from_landmarks(
    G: nx.Graph,
    edge_labels: dict[tuple, str],
    landmarks: dict[str, Landmark],
    veins: list[VeinIdentification],
) -> None:
    """Inject synthetic crossvein centerlines when graph detection failed.

    Runs as the last stage of vein identification. For each unlabeled
    crossvein (ACV, PCV), if both anchor landmarks are present AND both
    adjacent longitudinals were identified, draw a centerline from the
    projection of the anterior landmark onto the anterior longitudinal,
    through both landmarks, to the projection of the posterior landmark
    onto the posterior longitudinal. The result is marked INFERRED so
    downstream consumers can distinguish a measured crossvein from an
    axis-derived stand-in.

    The centerline is used as a watershed barrier by the intervein
    splitter, which is the whole point: it stops two adjacent regions
    (e.g. 1st basal and 1st posterior) from fusing just because the
    pixel classifier failed to resolve the crossvein as separate tissue.
    """
    from identify_features.models.topology import CROSSVEIN_CONNECTIONS

    cv_landmarks = {
        "ACV": ("ACV.a", "ACV.p"),
        "PCV": ("PCV.a", "PCV.p"),
    }

    existing_ids = {v.vein_id for v in veins}

    # Collect labeled vein lines per longitudinal
    vein_lines: dict[str, list[LineString]] = defaultdict(list)
    for (u, v), label in edge_labels.items():
        if G.has_edge(u, v):
            line = G[u][v].get("line")
            if line is not None:
                vein_lines[label].append(line)

    def _project_to_nearest_line(pt: Point, lines: list[LineString]) -> Point | None:
        best_point = None
        best_dist = float("inf")
        for line in lines:
            proj = line.interpolate(line.project(pt))
            d = proj.distance(pt)
            if d < best_dist:
                best_dist = d
                best_point = proj
        return best_point

    for cv_name, (vein_a, vein_b) in CROSSVEIN_CONNECTIONS.items():
        if cv_name in existing_ids:
            continue

        lm_a_name, lm_p_name = cv_landmarks[cv_name]
        lm_a = landmarks.get(lm_a_name)
        lm_p = landmarks.get(lm_p_name)
        if lm_a is None or lm_p is None:
            continue

        lines_a = vein_lines.get(vein_a) or []
        lines_b = vein_lines.get(vein_b) or []
        if not lines_a or not lines_b:
            continue

        pt_a = _project_to_nearest_line(lm_a.point, lines_a)
        pt_b = _project_to_nearest_line(lm_p.point, lines_b)
        if pt_a is None or pt_b is None:
            continue

        coords = [
            (pt_a.x, pt_a.y),
            (lm_a.point.x, lm_a.point.y),
            (lm_p.point.x, lm_p.point.y),
            (pt_b.x, pt_b.y),
        ]
        # Drop duplicate adjacent points to keep LineString valid
        deduped = [coords[0]]
        for c in coords[1:]:
            if c != deduped[-1]:
                deduped.append(c)
        if len(deduped) < 2:
            continue
        line = LineString(deduped)

        veins.append(
            VeinIdentification(
                vein_id=cv_name,
                vein_type=VeinType.CROSSVEIN,
                status=VeinStatus.INFERRED,
                centerline=line,
                length_px=line.length,
                evidence=[f"synthesized from {lm_a_name}+{lm_p_name}"],
                landmark_anchors=[lm_a_name, lm_p_name],
            )
        )
        logger.info(
            "Synthesized %s from landmarks (%s+%s): %.0fpx",
            cv_name,
            lm_a_name,
            lm_p_name,
            line.length,
        )
