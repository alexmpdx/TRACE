"""Regression tests for the CV-ratio overlay's landmarks-only fast path.

Background — the bug these pin (fixed in v0.2.29):

When ``cv_ratio_overlay`` is the ONLY Stage-2 output selected, ``identify_wing``
is skipped entirely and the ``WingResult`` is built straight from the Stage-1
landmarks GeoJSON (the v0.2.16 fast path). That GeoJSON is written in
RESCALED-pixel space, while the overlay base image is inverse-resized back to
ORIGINAL-pixel space before drawing. The fast path did not apply the matching
inverse, so every landmark was drawn at rescaled coordinates on an
original-resolution image and landed off-canvas. The overlay still rendered —
the wing, and the "CV ratio: N" readout, which is drawn at a hardcoded
position — but with no landmark dots and no measurement lines. It failed
silently for thirteen releases: no exception, no log warning, a PNG written
every time.

The invariant worth pinning is the one v0.2.16's commit message asserted
without anything enforcing it: the fast path and the normal path must hand the
renderer the same landmark coordinates. These tests are pure data assertions —
no model inference, no GPU, no full pipeline run.
"""

import json
import sys
from pathlib import Path

import pytest

# Sibling modules export bare package names that don't match their directory,
# so importing TRACE.pipeline needs the same sys.path set run_cli.py builds.
_ROOT = Path(__file__).resolve().parents[2]
for _p in (
    _ROOT,
    _ROOT / "HingeChopper",
    _ROOT / "modelTOjson",
    _ROOT / "identifyFeatures",
    _ROOT / "wingRotator",
    _ROOT / "measurementMaker",
    _ROOT / "scaleEstimator",
    _ROOT / "LandmarkLocator",
):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from TRACE.pipeline import build_cv_ratio_wing_result  # noqa: E402

# The only four landmarks render_cv_ratio_overlay reads.
_LANDMARKS_ORIGINAL_SPACE = {
    "L1-Rs": (753.0, 1285.0),
    "DTip": (5153.0, 1968.0),
    "ACV.p": (1839.0, 1678.0),
    "PCV.a": (2655.0, 1990.0),
}


def _write_landmarks_geojson(path: Path, points: dict, scale: float = 1.0) -> Path:
    """Write a Stage-1-style landmarks GeoJSON, optionally in rescaled space."""
    features = [
        {
            "type": "Feature",
            "properties": {"classification": {"name": name}, "reliable": True},
            "geometry": {"type": "Point", "coordinates": [x * scale, y * scale]},
        }
        for name, (x, y) in points.items()
    ]
    path.write_text(json.dumps({"type": "FeatureCollection", "features": features}))
    return path


@pytest.mark.parametrize("rescale_factor", [2.0704, 0.5176, 3.5])
def test_fast_path_maps_landmarks_back_to_original_space(tmp_path, rescale_factor):
    """The core invariant: rescaled-space GeoJSON in, original-space landmarks out.

    This is the assertion that would have caught the v0.2.29 bug. It is
    deliberately independent of ``inverse_rescale_wing_result`` — the expected
    coordinates are the originals we started from, not a re-derivation through
    the same helper the implementation uses.
    """
    gj = _write_landmarks_geojson(
        tmp_path / "wing_landmarks.geojson", _LANDMARKS_ORIGINAL_SPACE, scale=rescale_factor
    )

    result = build_cv_ratio_wing_result(gj, "wing", rescale_factor=rescale_factor)

    assert result is not None
    for name, (want_x, want_y) in _LANDMARKS_ORIGINAL_SPACE.items():
        lm = result.landmarks[name]
        assert lm.x == pytest.approx(want_x, rel=1e-6), f"{name} x in wrong coordinate space"
        assert lm.y == pytest.approx(want_y, rel=1e-6), f"{name} y in wrong coordinate space"


def test_fast_path_is_a_noop_at_scale_one(tmp_path):
    """No rescale means the GeoJSON is already in original space — leave it alone."""
    gj = _write_landmarks_geojson(tmp_path / "wing_landmarks.geojson", _LANDMARKS_ORIGINAL_SPACE)

    result = build_cv_ratio_wing_result(gj, "wing", rescale_factor=1.0)

    assert result is not None
    for name, (want_x, want_y) in _LANDMARKS_ORIGINAL_SPACE.items():
        assert result.landmarks[name].x == pytest.approx(want_x)
        assert result.landmarks[name].y == pytest.approx(want_y)


def test_fast_path_agrees_with_normal_path(tmp_path):
    """The two paths must hand the renderer identical coordinates.

    The normal path loads the same GeoJSON and runs it through
    ``inverse_rescale_wing_result``. Guards against anyone re-open-coding the
    transform in the fast path and letting the two drift apart again.
    """
    from identify_features.models.datatypes import WingResult
    from identify_features.models.geojson_io import load_landmarks_geojson
    from resolutionAdjust import inverse_rescale_wing_result

    rescale_factor = 2.0704
    gj = _write_landmarks_geojson(
        tmp_path / "wing_landmarks.geojson", _LANDMARKS_ORIGINAL_SPACE, scale=rescale_factor
    )

    fast = build_cv_ratio_wing_result(gj, "wing", rescale_factor=rescale_factor)

    normal = WingResult(specimen_id="wing", landmarks=load_landmarks_geojson(gj))
    inverse_rescale_wing_result(normal, rescale_factor)

    assert fast is not None
    for name in _LANDMARKS_ORIGINAL_SPACE:
        assert fast.landmarks[name].x == pytest.approx(normal.landmarks[name].x)
        assert fast.landmarks[name].y == pytest.approx(normal.landmarks[name].y)


def test_landmarks_land_on_canvas_when_rendered(tmp_path):
    """End-to-end guard: the rendered overlay must actually draw the annotations.

    The failure mode was visual, not exceptional — a PNG was always produced. So
    assert on the picture: every landmark inside the frame, and enough pixels
    changed to account for lines and dots rather than just the ratio readout.
    """
    import numpy as np

    from identify_features.views.overlay import render_cv_ratio_overlay

    rescale_factor = 3.5  # large enough that unmapped points leave the canvas entirely
    base = np.full((2500, 6000, 3), 200, np.uint8)
    gj = _write_landmarks_geojson(
        tmp_path / "wing_landmarks.geojson", _LANDMARKS_ORIGINAL_SPACE, scale=rescale_factor
    )

    result = build_cv_ratio_wing_result(gj, "wing", rescale_factor=rescale_factor)
    assert result is not None

    height, width = base.shape[:2]
    for name in _LANDMARKS_ORIGINAL_SPACE:
        lm = result.landmarks[name]
        assert 0 <= lm.x < width, f"{name} drawn off-canvas horizontally"
        assert 0 <= lm.y < height, f"{name} drawn off-canvas vertically"

    rendered = render_cv_ratio_overlay(base, result)
    assert rendered is not None

    changed = int((rendered != base).any(axis=2).sum())
    # The ratio readout alone is a few thousand pixels; both measurement lines
    # plus four dots is an order of magnitude more.
    assert changed > 50_000, f"overlay drew only {changed} px — lines/dots likely missing"


def test_missing_geojson_returns_none(tmp_path):
    """A missing landmarks file is a skip, not a crash."""
    assert build_cv_ratio_wing_result(tmp_path / "does_not_exist.geojson", "wing") is None
