"""Tests for the boundary-landmark polygon overlay renderer.

The renderer draws the ring the ``boundary_polygon`` quality filter tests: green when
simple, red when self-intersecting, amber when too few boundary landmarks are present.
It takes raw landmarks rather than a ``WingResult`` so it can be produced for wings the
filter aborted. These tests are pure pixel assertions on a synthetic grey canvas.
"""

from __future__ import annotations

import numpy as np
import pytest
from identify_features.models.datatypes import Landmark
from identify_features.views.overlay import (
    _BOUNDARY_FAIL_BGR,
    _BOUNDARY_OK_BGR,
    _BOUNDARY_UNDEFINED_BGR,
    render_boundary_polygon_overlay,
    render_boundary_polygon_overlay_to_file,
)
from shapely.geometry import Point

# Clean teardrop (distal-left), scaled to a 1000 × 700 canvas.
CLEAN_XY: dict[str, tuple[float, float]] = {
    "subcostal break": (770.0, 220.0),
    "L2.d": (195.0, 240.0),
    "L3.d": (55.0, 400.0),
    "L4.d": (90.0, 490.0),
    "L5.d": (460.0, 610.0),
    "alula notch": (935.0, 370.0),
    "L4-L5": (960.0, 300.0),
    "L1-Rs": (920.0, 270.0),
    "ACV.a": (690.0, 320.0),  # non-boundary — must be ignored
}


def _lm(xy):
    return {n: Landmark(name=n, point=Point(x, y), reliable=True) for n, (x, y) in xy.items()}


def _canvas():
    return np.full((700, 1000, 3), 128, dtype=np.uint8)


def _count(img, bgr):
    return int(np.all(img == np.array(bgr, dtype=np.uint8), axis=-1).sum())


def _crossed():
    xy = dict(CLEAN_XY)
    xy["L4.d"], xy["L5.d"] = CLEAN_XY["L5.d"], CLEAN_XY["L4.d"]
    return xy


def test_clean_ring_is_green_and_canvas_untouched():
    base = _canvas()
    img = render_boundary_polygon_overlay(base, _lm(CLEAN_XY))
    assert img is not None and img.shape == base.shape
    assert _count(base, (128, 128, 128)) == base.shape[0] * base.shape[1]  # input not mutated
    assert _count(img, _BOUNDARY_OK_BGR) > 500
    assert _count(img, _BOUNDARY_FAIL_BGR) == 0
    assert _count(img, _BOUNDARY_UNDEFINED_BGR) == 0


def test_crossed_ring_is_red():
    img = render_boundary_polygon_overlay(_canvas(), _lm(_crossed()))
    assert img is not None
    assert _count(img, _BOUNDARY_FAIL_BGR) > 500
    assert _count(img, _BOUNDARY_OK_BGR) == 0


def test_crossing_edges_are_emphasised():
    """The two crossing edges are drawn thicker than the rest of the ring."""
    plain = render_boundary_polygon_overlay(_canvas(), _lm(_crossed()), crossings=[])  # forced "simple"
    hot = render_boundary_polygon_overlay(_canvas(), _lm(_crossed()))
    # Same geometry, so the only difference is the emphasis stroke + colour; compare coverage.
    assert _count(hot, _BOUNDARY_FAIL_BGR) > _count(plain, _BOUNDARY_OK_BGR) * 1.2


def test_explicit_empty_crossings_is_honoured():
    """``crossings=[]`` means the caller already knows the ring is simple — don't recompute."""
    img = render_boundary_polygon_overlay(_canvas(), _lm(_crossed()), crossings=[])
    assert _count(img, _BOUNDARY_OK_BGR) > 500
    assert _count(img, _BOUNDARY_FAIL_BGR) == 0


def test_too_few_landmarks_is_amber():
    xy = {k: CLEAN_XY[k] for k in ("subcostal break", "L2.d", "L3.d")}
    img = render_boundary_polygon_overlay(_canvas(), _lm(xy))
    assert img is not None
    assert _count(img, _BOUNDARY_UNDEFINED_BGR) > 100
    assert _count(img, _BOUNDARY_OK_BGR) == 0
    assert _count(img, _BOUNDARY_FAIL_BGR) == 0


def test_no_boundary_landmarks_returns_none(tmp_path):
    xy = {"ACV.a": (1.0, 1.0), "PCV.p": (2.0, 2.0)}
    assert render_boundary_polygon_overlay(_canvas(), _lm(xy)) is None
    assert render_boundary_polygon_overlay_to_file(_canvas(), _lm(xy), tmp_path / "x.png") is False
    assert not (tmp_path / "x.png").exists()


def test_labels_and_status_text_can_be_suppressed():
    with_text = render_boundary_polygon_overlay(_canvas(), _lm(CLEAN_XY))
    bare = render_boundary_polygon_overlay(_canvas(), _lm(CLEAN_XY), show_landmark_labels=False, show_status_text=False)
    # Status text + labels are drawn with a white halo; the bare render has none of it.
    assert _count(with_text, (255, 255, 255)) > 0
    assert _count(bare, (255, 255, 255)) == 0


@pytest.mark.parametrize("xy, colour", [(CLEAN_XY, _BOUNDARY_OK_BGR), (None, _BOUNDARY_FAIL_BGR)])
def test_to_file_writes_png(tmp_path, xy, colour):
    import cv2

    out = tmp_path / "sub" / "w_boundary_polygon_overlay.png"
    assert render_boundary_polygon_overlay_to_file(_canvas(), _lm(xy or _crossed()), out) is True
    assert _count(cv2.imread(str(out)), colour) > 500
