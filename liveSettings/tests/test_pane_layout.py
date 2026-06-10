"""Regression: the preview pane must scroll, not overlap, when it's too short.

The zoomable view is a QGraphicsView (clips its content), so the only way the
wing image can paint over the Display box (as reported) is if the pane's
QVBoxLayout overflows when given less height than its content needs — the view
widget then extends down over the controls. Hosting the body in a QScrollArea
fixes it: a short pane scrolls instead of overflowing.

Run:  QT_QPA_PLATFORM=offscreen python liveSettings/tests/test_pane_layout.py
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

from PyQt5.QtWidgets import QApplication, QGroupBox, QScrollArea  # noqa: E402

from identify_features.config import PipelineConfig  # noqa: E402
from live_tune.preview_pane import LivePreviewPane  # noqa: E402


def main() -> int:
    app = QApplication.instance() or QApplication(sys.argv)  # noqa: F841
    pane = LivePreviewPane(get_config=lambda: PipelineConfig())

    sa = pane.findChild(QScrollArea)
    assert sa is not None, "pane body must be hosted in a QScrollArea"
    assert sa.widgetResizable(), "scroll area must resize its widget"
    print("[scrollarea] present + resizable ok")

    # The pane must be able to shrink well below its content's natural height
    # (without the scroll area, Qt clamps minimumSizeHint to ~750 and the
    # layout overflows when the dialog forces it shorter).
    assert pane.minimumSizeHint().height() < 300, (
        f"pane min height too tall: {pane.minimumSizeHint().height()} — would overflow when squeezed"
    )
    print(f"[minheight] {pane.minimumSizeHint().height()}px — pane can shrink ok")

    # Force a short pane (as a short dialog would) and confirm it scrolls.
    pane.resize(1050, 430)
    pane.show()
    app.processEvents()
    pane.layout().activate()
    app.processEvents()
    assert pane.height() == 430, f"pane did not honor short height: {pane.height()}"
    content = sa.widget()
    assert content.sizeHint().height() > sa.viewport().height(), "expected content taller than viewport (scroll)"
    print(f"[scroll] content {content.sizeHint().height()} > viewport {sa.viewport().height()} ok")

    # Inside the content, the view must not overlap the Display box.
    disp = next(g for g in pane.findChildren(QGroupBox) if g.title() == "Display")
    v = pane.view.geometry()
    d = disp.geometry()
    assert v.y() + v.height() <= d.y(), f"view overlaps Display box: view.bottom={v.y()+v.height()} display.top={d.y()}"
    print(f"[overlap] view.bottom={v.y()+v.height()} <= display.top={d.y()} ok")

    pane.shutdown()
    print("ALL PANE-LAYOUT CHECKS PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
