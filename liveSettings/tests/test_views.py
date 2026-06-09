"""Tests for the live-preview view modes (skeleton / traced / final).

Key behaviors:
- Skeleton view renders from Tier A only — no Tier B trace runs, even when a
  Tier-B param changes (the work is deferred).
- Switching to a tracing view runs the deferred trace lazily.
- Each view produces a valid image; traced/final include veins.
- The renderers draw something (skeleton edges, traced strokes) over the base.

Run from repo root:  python -m pytest liveSettings/tests/test_views.py -v
"""

import sys
from dataclasses import replace
from pathlib import Path

import numpy as np

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
    VIEW_FINAL,
    VIEW_SKELETON,
    VIEW_TRACED,
    LiveTuneSession,
)

_IDF = ROOT / "identifyFeatures"
DET = _IDF / "geojsons" / "0003_detections.geojson"
LM = _IDF / "LandmarLocator_2_std30" / "0003_landmarks.geojson"
IMG = _IDF / "OGpics" / "0003.bmp"


def _make_session() -> LiveTuneSession:
    vp, ip = load_detection_geojson(DET)
    lms = load_landmarks_geojson(LM)
    outline = _compute_wing_outline(vp + ip)
    img = imread_any(IMG)
    s = LiveTuneSession()
    s.set_input(img, vp, ip, lms, outline, (img.shape[0], img.shape[1]))
    return s


def test_skeleton_view_skips_tier_b_on_first_run():
    s = _make_session()
    r = s.update(PipelineConfig(), view=VIEW_SKELETON)
    assert r.error is None
    assert "A_skeleton" in r.timings_ms
    assert "B_trace" not in r.timings_ms, "skeleton view must not run the trace"
    assert r.overlay_bgr is not None
    # No veins traced yet (deferred).
    assert r.n_veins == 0


def test_skeleton_view_ignores_tier_b_param_change():
    s = _make_session()
    cfg = PipelineConfig()
    s.update(cfg, view=VIEW_SKELETON)
    # A tracing param changes; skeleton view should not retrace.
    r = s.update(replace(cfg, snap_radius_um=60.0), view=VIEW_SKELETON)
    assert "B_trace" not in r.timings_ms


def test_switch_to_traced_runs_deferred_trace():
    s = _make_session()
    cfg = PipelineConfig()
    s.update(cfg, view=VIEW_SKELETON)  # tier A only
    r = s.update(cfg, view=VIEW_TRACED)  # must trace now
    assert r.error is None
    assert "B_trace" in r.timings_ms
    assert r.n_veins > 0
    assert r.overlay_bgr is not None


def test_final_view_unchanged_behavior():
    s = _make_session()
    r = s.update(PipelineConfig(), view=VIEW_FINAL)
    assert r.error is None
    assert "A_skeleton" in r.timings_ms and "B_trace" in r.timings_ms
    assert r.n_veins > 0


def test_default_view_is_final():
    s = _make_session()
    r = s.update(PipelineConfig())  # no view arg
    assert r.n_veins > 0  # final traces


def test_traced_view_draws_over_base():
    s = _make_session()
    cfg = PipelineConfig()
    r = s.update(cfg, view=VIEW_TRACED)
    # The traced overlay differs from the plain base image (strokes + landmarks).
    assert not np.array_equal(r.overlay_bgr, s._base_image)


def test_skeleton_then_final_traces_once():
    s = _make_session()
    cfg = PipelineConfig()
    s.update(cfg, view=VIEW_SKELETON)
    r1 = s.update(cfg, view=VIEW_FINAL)
    assert "B_trace" in r1.timings_ms  # deferred trace runs
    # No further change → re-render only, no retrace.
    r2 = s.update(cfg, view=VIEW_FINAL)
    assert "B_trace" not in r2.timings_ms
    assert "A_skeleton" not in r2.timings_ms


if __name__ == "__main__":
    import pytest
    sys.exit(pytest.main([__file__, "-v", "-s"]))
