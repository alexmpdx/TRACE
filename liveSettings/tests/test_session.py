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
    FIELD_TIER,
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
    assert "A_skeleton" in r.timings_ms and "B_trace" in r.timings_ms
    assert r.n_veins > 0
    assert r.overlay_bgr is not None


def test_opacity_change_is_tier_d_only():
    s = _make_session()
    cfg = PipelineConfig()
    s.update(cfg)
    r = s.update(replace(cfg, vein_opacity=0.4))
    assert r.tier_ran == TIER_D
    assert "A_skeleton" not in r.timings_ms
    assert "B_trace" not in r.timings_ms
    assert "D_render" in r.timings_ms


def test_tracing_change_is_tier_b_not_a():
    s = _make_session()
    cfg = PipelineConfig()
    s.update(cfg)
    r = s.update(replace(cfg, snap_radius_um=60.0))
    assert r.tier_ran == TIER_B
    assert "A_skeleton" not in r.timings_ms
    assert "B_trace" in r.timings_ms


def test_skeleton_change_is_tier_a():
    s = _make_session()
    cfg = PipelineConfig()
    s.update(cfg)
    r = s.update(replace(cfg, smooth_sigma=4.0))
    assert r.tier_ran == TIER_A
    assert "A_skeleton" in r.timings_ms


def test_no_change_is_noop():
    s = _make_session()
    cfg = PipelineConfig()
    s.update(cfg)
    r = s.update(cfg)
    assert r.tier_ran == "none"
    assert "A_skeleton" not in r.timings_ms
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
    assert "A_skeleton" not in r.timings_ms
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


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v", "-s"]))
