"""Tests for moving the vein color key out of the live-preview overlay.

- render_overlay's new show_color_key flag: default True draws the key (batch
  unchanged), False suppresses it (the key region differs, the rest is identical).
- LiveTuneSession renders with show_color_key=False.
- The pane builds a static UI-side legend with one row per vein + EV.

Run:  python -m pytest liveSettings/tests/test_color_key.py -v
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


def _one_vein():
    return [
        VeinIdentification(
            vein_id="L3",
            vein_type=VeinType.LONGITUDINAL,
            status=VeinStatus.IDENTIFIED,
            centerline=LineString([(500, 500), (1500, 1500)]),
        )
    ]


def _img():
    return np.full((2000, 2000, 3), 40, dtype=np.uint8)


# -- render_overlay flag --------------------------------------------------
def test_color_key_drawn_by_default():
    with_key = render_overlay(_img(), _one_vein(), [])
    without = render_overlay(_img(), _one_vein(), [], show_color_key=False)
    # The key lives in the upper-left corner (panel at x0,y0 = 30,30). With the
    # key on, that corner must differ from the no-key render.
    corner_with = with_key[30:200, 30:300]
    corner_without = without[30:200, 30:300]
    assert not np.array_equal(corner_with, corner_without), "key not drawn by default"


def test_no_key_when_suppressed_leaves_veins_intact():
    base = _img()
    without = render_overlay(base, _one_vein(), [], show_color_key=False)
    # The vein stroke (away from the corner) is still drawn.
    # Sample a point on the diagonal centerline, far from the legend corner.
    assert not np.array_equal(without[1000, 1000], base[1000, 1000]), "vein stroke missing"
    # And the legend corner equals the plain base (no key painted there).
    # Use a patch clear of the stroke: top-left 100x100 (stroke starts at 500,500).
    assert np.array_equal(without[0:100, 0:100], base[0:100, 0:100]), "something painted in key corner"


def test_key_corner_matches_base_only_without_key():
    base = _img()
    with_key = render_overlay(base, _one_vein(), [])
    # With the key, the top-left corner is painted (panel background + swatches).
    assert not np.array_equal(with_key[0:100, 0:100], base[0:100, 0:100])


# -- session renders without the baked-in key -----------------------------
def test_session_render_suppresses_key(monkeypatch):
    import live_tune.session as S

    captured = {}

    def _fake_render(*args, **kwargs):
        captured.update(kwargs)
        return np.zeros((10, 10, 3), dtype=np.uint8)

    monkeypatch.setattr(S, "render_overlay", _fake_render)

    from identify_features.config import PipelineConfig
    from live_tune.session import Appearance, LiveTuneSession

    from live_tune.session import VIEW_FINAL

    sess = LiveTuneSession()
    sess._base_image = _img()
    sess._veins = _one_vein()
    sess._regions = []
    sess._regions_stale = True
    sess._render(PipelineConfig(), Appearance(), VIEW_FINAL, {})
    assert captured.get("show_color_key") is False


# -- pane builds a static legend ------------------------------------------
def test_pane_builds_static_legend():
    import os
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PyQt5.QtWidgets import QApplication, QGroupBox, QLabel

    from identify_features.config import PipelineConfig
    from identify_features.models.topology import VEIN_AP_ORDER
    from live_tune.preview_pane import LivePreviewPane

    app = QApplication.instance() or QApplication(sys.argv)  # noqa: F841
    pane = LivePreviewPane(get_config=lambda: PipelineConfig())
    legend = pane.legend
    assert isinstance(legend, QGroupBox)
    # One text label per canonical vein + the EV bucket (swatches are also
    # QLabels, so filter to the ones carrying vein names).
    texts = {lbl.text() for lbl in legend.findChildren(QLabel) if lbl.text()}
    for vid in VEIN_AP_ORDER:
        assert vid in texts, f"legend missing {vid}"
    assert any("EV" in t for t in texts), "legend missing ectopic bucket"
    pane.shutdown()


if __name__ == "__main__":
    import pytest
    sys.exit(pytest.main([__file__, "-v"]))
