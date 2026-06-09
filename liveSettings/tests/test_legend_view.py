"""The preview color key must match the active view.

- Skeleton view → a graph key (edge + node-degree colors), not vein names.
- Traced / final views → the vein-color key.
- The skeleton key swatches must match the colors render_skeleton draws.

Single-pane standalone script (one QApplication + one pane), matching the other
offscreen Qt tests — creating many panes under pytest crashes Qt's offscreen
platform on teardown.

Run:  QT_QPA_PLATFORM=offscreen python liveSettings/tests/test_legend_view.py
Exits 0 on success.
"""

import os
import sys
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

ROOT = Path(__file__).resolve().parents[2]
for _sub in ("", "identifyFeatures", "liveSettings"):
    _p = str(ROOT / _sub) if _sub else str(ROOT)
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np  # noqa: E402
from PyQt5.QtWidgets import QApplication, QLabel  # noqa: E402

from identify_features.config import PipelineConfig  # noqa: E402
from live_tune.preview_pane import LivePreviewPane  # noqa: E402
from live_tune.session import VIEW_FINAL, VIEW_SKELETON, VIEW_TRACED  # noqa: E402


def _labels(stack):
    return {lbl.text() for lbl in stack.currentWidget().findChildren(QLabel) if lbl.text()}


def _skeleton_key_colors_match_renderer():
    """Legend skeleton swatch RGBs must equal what render_skeleton draws."""
    import networkx as nx
    from live_tune import preview_render as PR
    from shapely.geometry import LineString

    g = nx.Graph()
    g.add_node(1, x=10, y=10)
    g.add_node(2, x=50, y=50)  # junction (deg 3)
    g.add_node(3, x=90, y=10)
    g.add_node(4, x=50, y=90)
    g.add_edge(1, 2, line=LineString([(10, 10), (50, 50)]))
    g.add_edge(2, 3, line=LineString([(50, 50), (90, 10)]))
    g.add_edge(2, 4, line=LineString([(50, 50), (50, 90)]))
    skel = type("S", (), {"graph": g})()
    img = PR.render_skeleton(np.full((100, 100, 3), 40, dtype=np.uint8), skel)

    def rgb_at(y, x):
        b, gg, r = (int(c) for c in img[y, x])
        return [r, gg, b]

    legend = dict(LivePreviewPane._skeleton_legend_entries())
    assert rgb_at(50, 50) == legend["junction (deg ≥ 3)"], "junction color mismatch"
    assert rgb_at(10, 10) == legend["node (path, deg ≤ 2)"], "path-node color mismatch"


def main() -> int:
    app = QApplication.instance() or QApplication(sys.argv)  # noqa: F841
    pane = LivePreviewPane(get_config=lambda: PipelineConfig())

    # Default (final) → vein key.
    assert pane._legend_stack.currentIndex() == 0
    labels = _labels(pane._legend_stack)
    assert "L3" in labels and "vein edge" not in labels
    print("[default] vein key shown ok")

    # Skeleton → graph key, no vein names.
    pane.cmb_view.setCurrentIndex(pane.cmb_view.findData(VIEW_SKELETON))
    sk = _labels(pane._legend_stack)
    assert "vein edge" in sk
    assert any("junction" in t for t in sk) and any("node" in t for t in sk)
    assert "L3" not in sk
    print(f"[skeleton] graph key shown: {sorted(sk)}")

    # Traced → back to vein key.
    pane.cmb_view.setCurrentIndex(pane.cmb_view.findData(VIEW_TRACED))
    assert pane._legend_stack.currentIndex() == 0 and "L3" in _labels(pane._legend_stack)
    print("[traced] vein key restored ok")

    # Final → vein key.
    pane.cmb_view.setCurrentIndex(pane.cmb_view.findData(VIEW_FINAL))
    assert pane._legend_stack.currentIndex() == 0
    print("[final] vein key ok")

    _skeleton_key_colors_match_renderer()
    print("[colors] skeleton key matches renderer ok")

    pane.shutdown()
    print("ALL LEGEND-VIEW CHECKS PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
