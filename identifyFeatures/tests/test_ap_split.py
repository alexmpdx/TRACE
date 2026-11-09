"""Regression tests for ``compute_ap_split`` (issue #32, "Not getting all of A/P area").

A straight cut along the L4 axis through a concave outline can leave one
compartment in several pieces. The old implementation kept only the two
largest fragments, so the dropped pieces vanished from both the CSV areas
and the overlay. These tests build small synthetic outlines so they don't
depend on real wing fixtures.
"""

from __future__ import annotations

import numpy as np
import pytest
from shapely.geometry import LineString, Polygon

from identify_features.models.datatypes import VeinIdentification, VeinStatus, VeinType, WingResult
from identify_features.views import overlay as overlay_mod
from identify_features.views.csv_export import compute_ap_split

# Anterior is toward smaller y in these fixtures (L3 sits at y=10, L4 at y≈30).
_L4 = LineString([(10, 30), (50, 30.5), (90, 30)])
_L3 = LineString([(10, 10), (90, 10)])


def _vein(vid: str, line: LineString) -> VeinIdentification:
    return VeinIdentification(
        vein_id=vid,
        vein_type=VeinType.LONGITUDINAL,
        status=VeinStatus.IDENTIFIED,
        centerline=line,
    )


def _wing(outline: Polygon) -> WingResult:
    return WingResult(specimen_id="synthetic", veins=[_vein("L4", _L4), _vein("L3", _L3)], wing_outline=outline)


def _notched_outline() -> Polygon:
    """100x60 rectangle with a notch cut in from the posterior (large-y) margin.

    The notch spans x in [45, 55] and reaches from y=60 down to y=20, so the
    cut at y≈30 leaves ONE anterior piece and TWO posterior pieces.
    """
    return Polygon(
        [
            (0, 0),
            (100, 0),
            (100, 60),
            (55, 60),
            (55, 20),
            (45, 20),
            (45, 60),
            (0, 60),
        ]
    )


def test_split_keeps_every_fragment():
    outline = _notched_outline()
    anterior, posterior = compute_ap_split(_wing(outline))

    assert anterior is not None and posterior is not None
    # Both compartments together must tile the whole outline — nothing dropped.
    assert anterior.area + posterior.area == pytest.approx(outline.area, rel=1e-6)
    # The posterior side is the one cut in two by the notch: 2 x (45 x 30).
    assert posterior.area == pytest.approx(2 * 45 * 30, rel=1e-6)
    assert anterior.area == pytest.approx(100 * 30 - 10 * 10, rel=1e-6)
    # Orientation: anterior is the side L3 lies on.
    assert anterior.centroid.y < posterior.centroid.y


def test_simple_outline_still_two_halves():
    outline = Polygon([(0, 0), (100, 0), (100, 60), (0, 60)])
    anterior, posterior = compute_ap_split(_wing(outline))

    assert anterior is not None and posterior is not None
    assert anterior.geom_type == "Polygon" and posterior.geom_type == "Polygon"
    assert anterior.area == pytest.approx(100 * 30, rel=1e-6)
    assert posterior.area == pytest.approx(100 * 30, rel=1e-6)
    assert anterior.centroid.y < posterior.centroid.y


def test_overlay_paints_multipart_compartment():
    outline = _notched_outline()
    base = np.zeros((70, 110, 3), dtype=np.uint8)
    img = overlay_mod.render_ap_overlay(base, _wing(outline), show_compartment_labels=False)

    assert img is not None
    # Both posterior pieces receive the posterior tint; the notch stays black.
    assert img[50, 20].any(), "left posterior piece not painted"
    assert img[50, 80].any(), "right posterior piece not painted"
    assert not img[50, 50].any(), "notch (outside the wing) should be untouched"
    assert img[10, 50].any(), "anterior piece not painted"
