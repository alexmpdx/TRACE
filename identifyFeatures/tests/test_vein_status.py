"""Unit tests for ``_assign_absent_and_partial`` in vein_tracer.py.

Spec: documentation/spec_absent_partial_vein_status.md

These exercise the helper directly with hand-built ``nx.Graph`` + landmark
dicts so they don't depend on real wing fixtures or the slow tracer end-to-
end. The call-site gate (``if config.assign_absent_partial_status: ...``) is
a one-line conditional reading the config field exercised here; trust it.
"""

from __future__ import annotations

import networkx as nx
import pytest
from shapely.geometry import LineString, Point

from identify_features.config import PipelineConfig
from identify_features.models.datatypes import (
    Landmark,
    VeinIdentification,
    VeinStatus,
    VeinType,
)
from identify_features.models.topology import ALL_CANONICAL_VEINS
from identify_features.models.vein_tracer import _assign_absent_and_partial


# ----------------------------------------------------------------------------
# Test helpers
# ----------------------------------------------------------------------------


# distal_landmark_search_vw defaults to 2.0 → search_radius = 2 * 5 = 10 px.
# Pick coordinates so "reaches" and "doesn't reach" are unambiguous at that
# scale (well inside or well past the 10-px tolerance).
MEDIAN_VEIN_WIDTH_PX = 5.0


def _lm(name: str, x: float, y: float, reliable: bool = True) -> Landmark:
    return Landmark(name=name, point=Point(x, y), reliable=reliable)


def _add_edge(
    G: nx.Graph,
    edge_labels: dict,
    u: int,
    v: int,
    u_xy: tuple[float, float],
    v_xy: tuple[float, float],
    label: str,
) -> None:
    """Add an edge with a ``line`` LineString attribute and label it in
    ``edge_labels``. Mirrors the on-graph structure the real tracer builds."""
    G.add_node(u, pos=u_xy)
    G.add_node(v, pos=v_xy)
    G.add_edge(u, v, line=LineString([u_xy, v_xy]))
    edge_labels[(u, v)] = label


def _make_vein(
    vein_id: str,
    centerline: LineString | None,
    status: VeinStatus = VeinStatus.IDENTIFIED,
    vein_type: VeinType = VeinType.LONGITUDINAL,
) -> VeinIdentification:
    return VeinIdentification(
        vein_id=vein_id,
        vein_type=vein_type,
        status=status,
        centerline=centerline,
        length_px=centerline.length if centerline is not None else 0.0,
    )


# ----------------------------------------------------------------------------
# ABSENT — missing canonical veins get explicit placeholders
# ----------------------------------------------------------------------------


def test_empty_input_yields_all_canonical_veins_as_absent_placeholders() -> None:
    """No veins identified anywhere → every ALL_CANONICAL_VEINS entry appears
    as an explicit ABSENT row with centerline=None / length 0."""
    config = PipelineConfig()
    veins: list[VeinIdentification] = []
    _assign_absent_and_partial(
        veins, nx.Graph(), {}, {}, MEDIAN_VEIN_WIDTH_PX, config
    )
    by_id = {v.vein_id: v for v in veins}
    assert set(by_id) == set(ALL_CANONICAL_VEINS)
    for vid, v in by_id.items():
        assert v.status is VeinStatus.ABSENT, vid
        assert v.centerline is None, vid
        assert v.length_px == 0.0, vid
        assert v.evidence == ["no labelled path"], vid


def test_absent_acv_appears_when_other_veins_present() -> None:
    """A wing with most veins identified but no ACV → ACV row appears as
    ABSENT alongside everything else (regression check for crossvein-absent
    handling)."""
    config = PipelineConfig()
    l3 = _make_vein("L3", LineString([(0, 100), (100, 100)]))
    l4 = _make_vein("L4", LineString([(0, 200), (100, 200)]))
    veins = [l3, l4]
    _assign_absent_and_partial(veins, nx.Graph(), {}, {}, MEDIAN_VEIN_WIDTH_PX, config)
    by_id = {v.vein_id: v for v in veins}
    assert by_id["ACV"].status is VeinStatus.ABSENT
    assert by_id["ACV"].centerline is None
    # The identified L3/L4 untouched (no edges → not flagged gapped).
    assert by_id["L3"].status is VeinStatus.IDENTIFIED
    assert by_id["L4"].status is VeinStatus.IDENTIFIED


# ----------------------------------------------------------------------------
# PARTIAL — longitudinal truncated
# ----------------------------------------------------------------------------


def test_partial_longitudinal_truncated_when_endpoint_unreached() -> None:
    """An L4 whose labelled edge stops short of L4.d → PARTIAL (truncated).

    Setup: edge from (0,0) → (50,0). L4-L5 landmark at (0,0) (reached, dist 0).
    L4.d landmark at (200,0) — 150 units past edge end, far beyond the
    search radius (10 px). Truncation fires for the L4.d endpoint.
    """
    config = PipelineConfig()
    G = nx.Graph()
    edge_labels: dict = {}
    _add_edge(G, edge_labels, 1, 2, (0.0, 0.0), (50.0, 0.0), "L4")
    landmarks = {
        "L4-L5": _lm("L4-L5", 0.0, 0.0, reliable=True),
        "L4.d": _lm("L4.d", 200.0, 0.0, reliable=True),
    }
    l4 = _make_vein("L4", LineString([(0.0, 0.0), (50.0, 0.0)]))
    veins = [l4]
    _assign_absent_and_partial(veins, G, edge_labels, landmarks, MEDIAN_VEIN_WIDTH_PX, config)
    assert l4.status is VeinStatus.PARTIAL
    assert any("truncated" in e for e in l4.evidence)


def test_unreliable_endpoint_landmark_does_not_trigger_truncated() -> None:
    """An L4 whose distal-end landmark is flagged unreliable can't be judged
    against — the helper skips that endpoint and leaves status IDENTIFIED."""
    config = PipelineConfig()
    G = nx.Graph()
    edge_labels: dict = {}
    _add_edge(G, edge_labels, 1, 2, (0.0, 0.0), (50.0, 0.0), "L4")
    landmarks = {
        "L4-L5": _lm("L4-L5", 0.0, 0.0, reliable=True),
        "L4.d": _lm("L4.d", 200.0, 0.0, reliable=False),  # ← unreliable
    }
    l4 = _make_vein("L4", LineString([(0.0, 0.0), (50.0, 0.0)]))
    veins = [l4]
    _assign_absent_and_partial(veins, G, edge_labels, landmarks, MEDIAN_VEIN_WIDTH_PX, config)
    assert l4.status is VeinStatus.IDENTIFIED, "unreliable endpoint must be unjudgeable"


# ----------------------------------------------------------------------------
# PARTIAL — gapped (works for every vein type, including costa/L6)
# ----------------------------------------------------------------------------


def test_partial_when_labelled_edges_form_two_components() -> None:
    """L5 with two disconnected labelled edges → PARTIAL (gapped)."""
    config = PipelineConfig()
    G = nx.Graph()
    edge_labels: dict = {}
    # First component: 1-2
    _add_edge(G, edge_labels, 1, 2, (0.0, 0.0), (40.0, 0.0), "L5")
    # Second component: 3-4 (disjoint nodes, not connected to 1-2)
    _add_edge(G, edge_labels, 3, 4, (200.0, 0.0), (240.0, 0.0), "L5")
    # Reliable endpoint landmarks at both ends so truncation is satisfied
    # for both components separately — isolate the gapped signal.
    landmarks = {
        "L4-L5": _lm("L4-L5", 0.0, 0.0, reliable=True),
        "L5.d": _lm("L5.d", 240.0, 0.0, reliable=True),
    }
    l5 = _make_vein("L5", LineString([(0.0, 0.0), (240.0, 0.0)]))
    veins = [l5]
    _assign_absent_and_partial(veins, G, edge_labels, landmarks, MEDIAN_VEIN_WIDTH_PX, config)
    assert l5.status is VeinStatus.PARTIAL
    assert "gapped" in l5.evidence


def test_costa_only_subject_to_gapped_test_no_truncation_check() -> None:
    """costa has no LONGITUDINAL_ENDPOINTS entry; only the gapped check
    applies. A connected costa edge set must stay IDENTIFIED even without
    any landmarks defined for endpoint checks."""
    config = PipelineConfig()
    G = nx.Graph()
    edge_labels: dict = {}
    _add_edge(G, edge_labels, 1, 2, (0.0, 0.0), (100.0, 0.0), "costa")
    costa = _make_vein("costa", LineString([(0, 0), (100, 0)]), vein_type=VeinType.COSTA)
    veins = [costa]
    _assign_absent_and_partial(veins, G, edge_labels, {}, MEDIAN_VEIN_WIDTH_PX, config)
    assert costa.status is VeinStatus.IDENTIFIED


# ----------------------------------------------------------------------------
# PARTIAL — crossvein
# ----------------------------------------------------------------------------


def test_partial_pcv_reaches_l4_but_not_l5() -> None:
    """PCV centerline runs from L4 partway toward L5 but stops far short →
    PARTIAL via the bounding-longitudinal distance check (not gapped)."""
    config = PipelineConfig()
    G = nx.Graph()
    edge_labels: dict = {}
    # Connected single-edge PCV so the gapped check passes (1 component).
    _add_edge(G, edge_labels, 10, 11, (50.0, 100.0), (50.0, 120.0), "PCV")
    # Bounding longitudinals: L4 at y=100 (PCV's lower endpoint sits on it),
    # L5 at y=300 (PCV's upper endpoint at y=120 is 180 px shy — well past
    # the 10-px search radius).
    l4 = _make_vein("L4", LineString([(0, 100), (100, 100)]))
    l5 = _make_vein("L5", LineString([(0, 300), (100, 300)]))
    pcv = _make_vein(
        "PCV",
        LineString([(50, 100), (50, 120)]),
        vein_type=VeinType.CROSSVEIN,
    )
    veins = [l4, l5, pcv]
    _assign_absent_and_partial(veins, G, edge_labels, {}, MEDIAN_VEIN_WIDTH_PX, config)
    assert pcv.status is VeinStatus.PARTIAL
    assert any("truncated" in e for e in pcv.evidence)


def test_crossvein_with_absent_bounding_longitudinal_is_not_partial() -> None:
    """If a bounding longitudinal is itself absent, the helper can't judge
    that side of the crossvein and falls through to IDENTIFIED. (PCV
    reaches L4 here; L5 has no VeinIdentification — that side is skipped.)
    """
    config = PipelineConfig()
    G = nx.Graph()
    edge_labels: dict = {}
    _add_edge(G, edge_labels, 10, 11, (50.0, 100.0), (50.0, 105.0), "PCV")
    l4 = _make_vein("L4", LineString([(0, 100), (100, 100)]))
    pcv = _make_vein(
        "PCV",
        LineString([(50, 100), (50, 105)]),
        vein_type=VeinType.CROSSVEIN,
    )
    # NB: L5 is NOT in veins.
    veins = [l4, pcv]
    _assign_absent_and_partial(veins, G, edge_labels, {}, MEDIAN_VEIN_WIDTH_PX, config)
    assert pcv.status is VeinStatus.IDENTIFIED, "absent bounding vein must not falsely flag partial"


# ----------------------------------------------------------------------------
# Regression — fully-OK veins keep IDENTIFIED + non-IDENTIFIED untouched
# ----------------------------------------------------------------------------


def test_complete_identified_longitudinal_stays_identified() -> None:
    """An L4 with a single connected edge reaching BOTH reliable endpoints
    inside the search radius must stay IDENTIFIED."""
    config = PipelineConfig()
    G = nx.Graph()
    edge_labels: dict = {}
    _add_edge(G, edge_labels, 1, 2, (0.0, 0.0), (200.0, 0.0), "L4")
    landmarks = {
        "L4-L5": _lm("L4-L5", 0.0, 0.0, reliable=True),
        "L4.d": _lm("L4.d", 200.0, 0.0, reliable=True),
    }
    l4 = _make_vein("L4", LineString([(0.0, 0.0), (200.0, 0.0)]))
    veins = [l4]
    _assign_absent_and_partial(veins, G, edge_labels, landmarks, MEDIAN_VEIN_WIDTH_PX, config)
    assert l4.status is VeinStatus.IDENTIFIED


def test_ectopic_and_inferred_are_never_downgraded() -> None:
    """ECTOPIC + INFERRED veins must pass through untouched even when they
    look gapped or truncated — only IDENTIFIED is subject to downgrade."""
    config = PipelineConfig()
    G = nx.Graph()
    edge_labels: dict = {}
    # An EV (ectopic) with two disconnected labelled edges — would be
    # "gapped" if IDENTIFIED, but ECTOPIC must skip the check.
    _add_edge(G, edge_labels, 1, 2, (0.0, 0.0), (10.0, 0.0), "EV1")
    _add_edge(G, edge_labels, 3, 4, (50.0, 0.0), (60.0, 0.0), "EV1")
    ev1 = VeinIdentification(
        vein_id="EV1",
        vein_type=VeinType.LONGITUDINAL,
        status=VeinStatus.ECTOPIC,
        centerline=LineString([(0, 0), (60, 0)]),
    )
    inferred_pcv = VeinIdentification(
        vein_id="PCV",
        vein_type=VeinType.CROSSVEIN,
        status=VeinStatus.INFERRED,
        centerline=LineString([(50, 100), (50, 105)]),  # far from any bounding line
    )
    veins = [ev1, inferred_pcv]
    _assign_absent_and_partial(veins, G, edge_labels, {}, MEDIAN_VEIN_WIDTH_PX, config)
    assert ev1.status is VeinStatus.ECTOPIC, "ECTOPIC must never be downgraded"
    assert inferred_pcv.status is VeinStatus.INFERRED, "INFERRED must never be downgraded"


# ----------------------------------------------------------------------------
# Flag — defaults + escape hatch
# ----------------------------------------------------------------------------


def test_pipeline_config_defaults_assign_absent_partial_status_true() -> None:
    """Spec §5: the feature is ON by default; legacy output is opt-out."""
    assert PipelineConfig().assign_absent_partial_status is True


def test_helper_runs_idempotently_when_no_identified_or_canonical() -> None:
    """Calling the helper with non-canonical IDENTIFIED veins only must
    leave them untouched and emit a full ABSENT placeholder set."""
    config = PipelineConfig()
    weird = VeinIdentification(
        vein_id="X1",
        vein_type=VeinType.LONGITUDINAL,
        status=VeinStatus.IDENTIFIED,
        centerline=LineString([(0, 0), (1, 0)]),
    )
    veins = [weird]
    _assign_absent_and_partial(veins, nx.Graph(), {}, {}, MEDIAN_VEIN_WIDTH_PX, config)
    # X1 is neither in LONGITUDINAL_ENDPOINTS nor CROSSVEIN_CONNECTIONS, so
    # only the gapped check fires; with no edges in edge_labels, the subgraph
    # is empty → not gapped → IDENTIFIED.
    assert weird.status is VeinStatus.IDENTIFIED
    absent_ids = {v.vein_id for v in veins if v.status is VeinStatus.ABSENT}
    assert absent_ids == set(ALL_CANONICAL_VEINS)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
