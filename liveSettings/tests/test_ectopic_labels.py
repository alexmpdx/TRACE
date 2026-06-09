"""Tests for suppressing ectopic vein text labels in the traced preview.

render_overlay's new show_ectopic_labels flag (default True, batch unchanged)
controls whether "EV1"/"EV2"… text is drawn. The traced view passes False so
the labels don't clutter the snapped-landmark view; the EV *centerline* still
draws.

Run:  python -m pytest liveSettings/tests/test_ectopic_labels.py -v
"""

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
for _sub in ("", "identifyFeatures", "liveSettings"):
    _p = str(ROOT / _sub) if _sub else str(ROOT)
    if _p not in sys.path:
        sys.path.insert(0, _p)

from identify_features.models.datatypes import VeinIdentification, VeinStatus, VeinType  # noqa: E402
from identify_features.views.overlay import render_overlay  # noqa: E402
from shapely.geometry import LineString  # noqa: E402


def _img():
    return np.full((2000, 2000, 3), 40, dtype=np.uint8)


def _ev_vein():
    # An ectopic vein whose centerline sits low-right, so its label (drawn at
    # centroid + (20, -20)) lands clear of the upper-left color-key corner.
    return [
        VeinIdentification(
            vein_id="EV1",
            vein_type=VeinType.LONGITUDINAL,
            status=VeinStatus.ECTOPIC,
            centerline=LineString([(900, 900), (1100, 1100)]),
        )
    ]


def _label_region(img):
    # Region around the EV label position (centroid ~ (1000,1000), text at +20/-20).
    return img[930:1010, 1010:1300]


def test_ectopic_label_drawn_by_default():
    base = _img()
    with_label = render_overlay(base, _ev_vein(), [], show_color_key=False)
    # Something (the EV text) is painted in the label region.
    assert not np.array_equal(_label_region(with_label), _label_region(base))


def test_ectopic_label_suppressed_when_flag_false():
    with_label = render_overlay(_img(), _ev_vein(), [], show_color_key=False)
    without = render_overlay(_img(), _ev_vein(), [], show_color_key=False,
                             show_ectopic_labels=False)
    # The label region differs between drawn and suppressed.
    assert not np.array_equal(_label_region(with_label), _label_region(without))


def test_ectopic_centerline_still_drawn_when_labels_off():
    base = _img()
    without = render_overlay(base, _ev_vein(), [], show_color_key=False,
                             show_ectopic_labels=False)
    # The EV centerline (midpoint ~ (1000,1000)) is still stroked.
    assert not np.array_equal(without[1000, 1000], base[1000, 1000])


def test_traced_view_passes_flag_false(monkeypatch):
    import live_tune.preview_render as PR

    captured = {}

    def _fake(*args, **kwargs):
        captured.update(kwargs)
        return np.zeros((10, 10, 3), dtype=np.uint8)

    monkeypatch.setattr(PR, "render_overlay", _fake)
    PR.render_traced(_img(), _ev_vein(), {}, None)
    assert captured.get("show_ectopic_labels") is False


if __name__ == "__main__":
    import pytest
    sys.exit(pytest.main([__file__, "-v"]))
