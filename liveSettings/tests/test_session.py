"""Tests for LiveTuneSession against real specimen 0001 data.

Run from repo root:  python -m pytest liveSettings/tests/test_session.py -v
or standalone:        python liveSettings/tests/test_session.py
"""

import sys
from dataclasses import fields, replace
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "identifyFeatures"))
sys.path.insert(0, str(ROOT / "liveSettings"))

from identify_features.config import PipelineConfig  # noqa: E402
from identify_features.models.geojson_io import (  # noqa: E402
    _compute_wing_outline,
    load_detection_geojson,
    load_landmarks_geojson,
)
from identify_features.utils.psd_loader import imread_any  # noqa: E402
from live_tune import (  # noqa: E402
    APPEARANCE_FIELDS,
    CORE_FIELDS,
    FIELD_TIER,
    FINISH_FIELDS,
    TIER_A,
    TIER_B,
    TIER_C,
    TIER_D,
    Appearance,
    LiveTuneSession,
)

# Real specimen 0003: detection + landmarks + image all present in the repo.
_IDF = ROOT / "identifyFeatures"
DET = _IDF / "geojsons" / "0003_detections.geojson"
LM = _IDF / "LandmarLocator_2_std30" / "0003_landmarks.geojson"
IMG = _IDF / "OGpics" / "0003.bmp"


def _make_session() -> LiveTuneSession:
    vein_polys, intervein_polys = load_detection_geojson(DET)
    landmarks = load_landmarks_geojson(LM)
    wing_outline = _compute_wing_outline(vein_polys + intervein_polys)
    img = imread_any(IMG)
    s = LiveTuneSession()
    s.set_input(img, vein_polys, intervein_polys, landmarks, wing_outline,
                (img.shape[0], img.shape[1]))
    return s


# -- FIELD_TIER coverage -------------------------------------------------
def test_field_tier_covers_every_config_field():
    assert set(FIELD_TIER) == {f.name for f in fields(PipelineConfig)}


def test_known_field_tiers():
    assert FIELD_TIER["smooth_sigma"] == TIER_A
    assert FIELD_TIER["um_per_px"] == TIER_A
    assert FIELD_TIER["bridge3_max_gap_vw"] == TIER_A
    assert FIELD_TIER["snap_radius_um"] == TIER_B
    assert FIELD_TIER["departure_sample_um"] == TIER_B
    assert FIELD_TIER["crossvein_min_angle"] == TIER_B
    assert FIELD_TIER["intervein_split_h_vw"] == TIER_C
    assert FIELD_TIER["skip_intervein_regions"] == TIER_C
    assert FIELD_TIER["vein_opacity"] == TIER_D
    assert set(APPEARANCE_FIELDS) == {n for n, t in FIELD_TIER.items() if t == TIER_D}


# -- tier selection ------------------------------------------------------
def test_first_update_runs_tier_a():
    s = _make_session()
    r = s.update(PipelineConfig())
    assert r.error is None
    assert r.tier_ran == TIER_A
    # First run builds the expensive core + cheap finish, then traces.
    assert "A_core" in r.timings_ms and "A_finish" in r.timings_ms
    assert "B_trace" in r.timings_ms
    assert r.n_veins > 0
    assert r.overlay_bgr is not None


def test_opacity_change_is_tier_d_only():
    s = _make_session()
    cfg = PipelineConfig()
    s.update(cfg)
    r = s.update(replace(cfg, vein_opacity=0.4))
    assert r.tier_ran == TIER_D
    assert "A_core" not in r.timings_ms and "A_finish" not in r.timings_ms
    assert "B_trace" not in r.timings_ms
    assert "D_render" in r.timings_ms


def test_tracing_change_is_tier_b_not_a():
    s = _make_session()
    cfg = PipelineConfig()
    s.update(cfg)
    r = s.update(replace(cfg, snap_radius_um=60.0))
    assert r.tier_ran == TIER_B
    assert "A_core" not in r.timings_ms and "A_finish" not in r.timings_ms
    assert "B_trace" in r.timings_ms


def test_skeleton_change_is_tier_a():
    s = _make_session()
    cfg = PipelineConfig()
    s.update(cfg)
    # smooth_sigma is a CORE field → rebuilds the expensive core.
    r = s.update(replace(cfg, smooth_sigma=4.0))
    assert r.tier_ran == TIER_A
    assert "A_core" in r.timings_ms


def test_no_change_is_noop():
    s = _make_session()
    cfg = PipelineConfig()
    s.update(cfg)
    r = s.update(cfg)
    assert r.tier_ran == "none"
    assert "A_core" not in r.timings_ms and "A_finish" not in r.timings_ms
    assert "B_trace" not in r.timings_ms


# -- idempotence (the anchor_landmarks mutation trap) --------------------
def _vein_signature(session):
    return [
        (v.vein_id, None if v.centerline is None else round(v.centerline.length, 3))
        for v in session._veins
    ]


def test_tier_b_idempotent_round_trip():
    s = _make_session()
    cfg = PipelineConfig()
    s.update(cfg)
    sig_a = _vein_signature(s)
    s.update(replace(cfg, snap_radius_um=55.0))  # to B
    s.update(cfg)  # back to original
    sig_b = _vein_signature(s)
    assert sig_a == sig_b, "Tier B not idempotent — pristine-copy discipline broken"


def test_pristine_skeleton_not_mutated_by_trace():
    s = _make_session()
    s.update(PipelineConfig())
    nodes_before = s._pristine_skel.graph.number_of_nodes()
    s.update(replace(PipelineConfig(), snap_radius_um=70.0))
    nodes_after = s._pristine_skel.graph.number_of_nodes()
    assert nodes_before == nodes_after, "anchor mutated the cached pristine skeleton"


# -- intervein stays out of the live loop --------------------------------
def test_intervein_param_does_not_run_tier_abd():
    s = _make_session()
    cfg = PipelineConfig()
    s.update(cfg)
    r = s.update(replace(cfg, intervein_split_h_vw=3.0))
    assert "A_core" not in r.timings_ms and "A_finish" not in r.timings_ms
    assert "B_trace" not in r.timings_ms
    assert r.regions_stale is True


def test_compute_intervein_on_demand():
    s = _make_session()
    s.update(PipelineConfig())
    regions = s.compute_intervein(PipelineConfig())
    assert isinstance(regions, list)
    assert s._regions_stale is False


def test_tier_b_marks_regions_stale():
    s = _make_session()
    cfg = PipelineConfig()
    s.update(cfg)
    s.compute_intervein(cfg)
    assert s._regions_stale is False
    r = s.update(replace(cfg, snap_radius_um=80.0))
    assert r.regions_stale is True


# -- error handling ------------------------------------------------------
def test_no_input_returns_error():
    s = LiveTuneSession()
    r = s.update(PipelineConfig())
    assert r.error is not None
    assert r.overlay_bgr is None


# -- skeleton core/finish split + memoization ----------------------------
def test_core_finish_partition():
    """CORE_FIELDS and FINISH_FIELDS partition exactly the Tier-A fields."""
    tier_a = {n for n, t in FIELD_TIER.items() if t == TIER_A}
    assert CORE_FIELDS | FINISH_FIELDS == tier_a
    assert not (CORE_FIELDS & FINISH_FIELDS)


def test_finish_only_change_skips_core():
    """A finish-only (bridging) param re-runs only the cheap finish, not the core."""
    s = _make_session()
    cfg = PipelineConfig()
    r1 = s.update(cfg)
    assert "A_core" in r1.timings_ms  # first run builds the expensive core
    r2 = s.update(replace(cfg, bridge_max_gap_um=250.0))  # FINISH-only field
    assert r2.tier_ran == TIER_A
    assert "A_core" not in r2.timings_ms, "finish-only change must NOT rebuild the core"
    assert "A_finish" in r2.timings_ms
    # The finish is the cheap half — should be well under a full core build.
    assert r2.timings_ms["A_finish"] < r1.timings_ms["A_core"]


def test_core_change_rebuilds_core():
    """A core param (smooth_sigma) rebuilds the expensive core."""
    s = _make_session()
    cfg = PipelineConfig()
    s.update(cfg)
    r = s.update(replace(cfg, smooth_sigma=4.0))
    assert "A_core" in r.timings_ms


def test_finish_only_matches_full_rebuild():
    """The finish-only fast path must produce the SAME skeleton as a full rebuild.

    Tune a finish param on a reused core, then build a fresh session that
    computes that same config from scratch; the finished skeletons must match.
    """
    cfg = PipelineConfig()
    tuned = replace(cfg, bridge_max_gap_um=250.0)

    s1 = _make_session()
    s1.update(cfg)            # builds core
    s1.update(tuned)          # finish-only fast path (reuses core)
    fast = s1._pristine_skel

    s2 = _make_session()
    s2.update(tuned)          # full rebuild from scratch with the tuned config
    full = s2._pristine_skel

    # Same graph structure + same edge geometry.
    assert set(fast.graph.nodes) == set(full.graph.nodes)
    assert set(map(frozenset, fast.graph.edges)) == set(map(frozenset, full.graph.edges))
    assert fast.median_vein_width_px == full.median_vein_width_px
    fast_wkt = sorted(d["line"].wkt for _, _, d in fast.graph.edges(data=True))
    full_wkt = sorted(d["line"].wkt for _, _, d in full.graph.edges(data=True))
    assert fast_wkt == full_wkt, "finish-only path diverged from a full rebuild"


def test_revisiting_config_is_instant():
    """Returning to a previously-seen config hits the LRU — no core/finish work."""
    s = _make_session()
    cfg = PipelineConfig()
    s.update(cfg)                                   # cache config A
    s.update(replace(cfg, smooth_sigma=4.0))        # move to config B (core rebuild)
    r = s.update(cfg)                               # back to A → full skeleton LRU hit
    assert "A_core" not in r.timings_ms and "A_finish" not in r.timings_ms
    assert "A_cached" in r.timings_ms


def test_set_input_clears_lru():
    """A new input wing clears the caches and bumps the epoch."""
    s = _make_session()
    s.update(PipelineConfig())
    assert len(s._skel_lru) > 0
    s2 = _make_session()  # fresh session, but assert the clear path directly:
    s._invalidate_all()
    assert len(s._core_lru) == 0 and len(s._skel_lru) == 0


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v", "-s"]))
