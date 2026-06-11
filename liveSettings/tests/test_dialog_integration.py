"""Offscreen test: attach the live preview to the REAL PipelineConfigDialog.

Proves the layout surgery and widget wiring work against the actual TRACE
dialog (not a mock), and that a real param-widget edit drives a recompute.

Run:  QT_QPA_PLATFORM=offscreen python liveSettings/tests/test_dialog_integration.py
"""

import os
import sys
import time
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

ROOT = Path(__file__).resolve().parents[2]
for sub in ("", "identifyFeatures", "HingeChopper", "modelTOjson", "preprocessing",
            "measurementMaker", "wingIsolator", "resolutionAdjust", "scaleEstimator",
            "wingRotator", "LandmarkLocator", "TRACE", "liveSettings"):
    p = str(ROOT / sub) if sub else str(ROOT)
    if p not in sys.path:
        sys.path.insert(0, p)

from PyQt5.QtWidgets import QApplication  # noqa: E402

from identify_features.config import PipelineConfig  # noqa: E402

_IDF = ROOT / "identifyFeatures"
_DET = _IDF / "geojsons" / "0003_detections.geojson"
_LM = _IDF / "LandmarLocator_2_std30" / "0003_landmarks.geojson"
_IMG = _IDF / "OGpics" / "0003.bmp"


def _pump(app, predicate, timeout=40.0):
    t0 = time.time()
    while time.time() - t0 < timeout:
        app.processEvents()
        if predicate():
            return True
        time.sleep(0.02)
    return False


def main() -> int:
    app = QApplication.instance() or QApplication(sys.argv)

    from settings_dialog import PipelineConfigDialog
    from live_tune.dialog_integration import attach_live_preview

    dialog = PipelineConfigDialog(PipelineConfig())
    pane = attach_live_preview(dialog)
    assert pane is not None, "attach_live_preview returned None"
    print("[attach] pane mounted, dialog still has layout:", dialog.layout() is not None)

    # The dialog must still be able to produce a config (unchanged behavior).
    cfg = dialog.get_config()
    assert isinstance(cfg, PipelineConfig)
    print("[regression] dialog.get_config() works")

    # Capture results from the pane's worker.
    results = []
    pane._worker.result_ready.connect(results.append)

    # The pane is raw-image-only now; inject a GeoJSON loader into the worker so
    # we can test the dialog-widget wiring without running the DL models.
    from live_tune.input_loader import load_from_geojsons

    def _loader():
        return load_from_geojsons(_DET, _LM, _IMG, um_per_px=2.0)

    pane._worker.request_load(_loader, pane._preview_scale, dialog.get_config(), pane._appearance)
    ok = _pump(app, lambda: pane._loaded and len(results) >= 1)
    assert ok, "sample did not load through the pane"
    print(f"[load] veins={results[-1].n_veins} tier={results[-1].tier_ran}")

    # Now flip a REAL dialog widget and confirm it drives a recompute.
    n0 = len(results)
    name, (kind, widget, extra) = next(
        (n, v) for n, v in dialog._widgets.items()
        if n == "snap_radius_um"
    )
    widget.setValue(widget.value() + 5.0)  # emits valueChanged -> pane.on_config_changed
    ok = _pump(app, lambda: len(results) > n0)
    assert ok, "editing a real dialog widget did not trigger recompute"
    rb = results[-1]
    assert "B_trace" in rb.timings_ms and "A_core" not in rb.timings_ms, rb.timings_ms
    print(f"[wire] dialog snap_radius_um edit -> tier {rb.tier_ran} {rb.timings_ms}")

    # And a skeleton-tier (core) widget should rebuild the expensive core.
    n1 = len(results)
    _, (_, sw, _) = next((n, v) for n, v in dialog._widgets.items() if n == "smooth_sigma")
    sw.setValue(sw.value() + 1.0)
    ok = _pump(app, lambda: len(results) > n1)
    assert ok, "skeleton-tier edit produced no recompute"
    assert "A_core" in results[-1].timings_ms
    print(f"[wire] dialog smooth_sigma edit -> tier {results[-1].tier_ran}")

    # Close the dialog -> worker stops cleanly.
    dialog.done(0)
    assert not pane._worker.isRunning(), "worker thread still running after dialog close"
    print("[lifecycle] worker stopped on dialog close")

    print("ALL DIALOG INTEGRATION CHECKS PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
