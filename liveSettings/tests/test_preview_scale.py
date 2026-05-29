"""Tests for the reduced-resolution preview path.

These run fast: quarter-resolution skeleton/trace on specimen 0003 is ~0.3 s vs
~5 s at full res, which is the whole point of the feature.

Run:  python -m pytest liveSettings/tests/test_preview_scale.py -v
"""

import sys
from dataclasses import replace
from pathlib import Path

import pytest
from shapely.ops import unary_union

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "identifyFeatures"))
sys.path.insert(0, str(ROOT / "liveSettings"))

from identify_features.config import PipelineConfig  # noqa: E402
from live_tune import LiveTuneSession  # noqa: E402
from live_tune.input_loader import apply_to_session, load_from_geojsons, scale_bundle  # noqa: E402

_IDF = ROOT / "identifyFeatures"
DET = _IDF / "geojsons" / "0003_detections.geojson"
LM = _IDF / "LandmarLocator_2_std30" / "0003_landmarks.geojson"
IMG = _IDF / "OGpics" / "0003.bmp"


def _full_bundle():
    return load_from_geojsons(DET, LM, IMG, um_per_px=2.0)


# -- scale_bundle geometry -------------------------------------------------
def test_scale_bundle_quarter_shrinks_everything():
    b = _full_bundle()
    fh, fw = b.image_shape
    q = scale_bundle(b, 0.25)
    assert q.preview_scale == 0.25
    qh, qw = q.image_shape
    assert abs(qh - round(fh * 0.25)) <= 1
    assert abs(qw - round(fw * 0.25)) <= 1
    assert q.base_image.shape[0] == qh and q.base_image.shape[1] == qw
    # Polygon coordinate extent should shrink ~4x.
    full_w = unary_union(b.vein_polys).bounds[2]
    quarter_w = unary_union(q.vein_polys).bounds[2]
    assert quarter_w == pytest.approx(full_w * 0.25, rel=0.02)
    # Landmarks scaled too.
    k = next(iter(b.landmarks_raw))
    assert q.landmarks_raw[k].point.x == pytest.approx(b.landmarks_raw[k].point.x * 0.25, rel=0.02)
    # Original bundle untouched (replace returns a copy).
    assert b.image_shape == (fh, fw)


def test_scale_bundle_full_is_noop():
    b = _full_bundle()
    f = scale_bundle(b, 1.0)
    assert f.preview_scale == 1.0
    assert f.image_shape == b.image_shape


# -- _effective um/px adjustment ------------------------------------------
def test_effective_scales_um_per_px():
    s = LiveTuneSession()
    s._preview_scale = 0.5
    eff = s._effective(PipelineConfig(um_per_px=2.0))
    assert eff.um_per_px == pytest.approx(4.0)  # px threshold shrinks by 0.5 to match image


def test_effective_noop_when_full_res():
    s = LiveTuneSession()
    s._preview_scale = 1.0
    cfg = PipelineConfig(um_per_px=2.0)
    assert s._effective(cfg) is cfg


def test_effective_noop_when_no_scale_calibration():
    s = LiveTuneSession()
    s._preview_scale = 0.5
    cfg = PipelineConfig(um_per_px=None)
    assert s._effective(cfg) is cfg  # vw fallbacks scale on their own


# -- end-to-end quarter-res run -------------------------------------------
def test_quarter_res_update_finds_veins_and_matches_overlay_shape():
    b = scale_bundle(_full_bundle(), 0.25)
    s = LiveTuneSession()
    apply_to_session(b, s)
    r = s.update(PipelineConfig(um_per_px=2.0))
    assert r.error is None
    assert r.n_veins > 0
    assert r.overlay_bgr.shape[:2] == b.image_shape


def test_preview_scale_propagates_to_session():
    b = scale_bundle(_full_bundle(), 0.5)
    s = LiveTuneSession()
    apply_to_session(b, s)
    assert s._preview_scale == 0.5


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v", "-s"]))
