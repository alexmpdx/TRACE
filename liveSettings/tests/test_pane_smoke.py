"""Offscreen Qt smoke test for LivePreviewPane + LiveTuneWorker.

Run:  QT_QPA_PLATFORM=offscreen python liveSettings/tests/test_pane_smoke.py
Exits 0 on success, prints a short report.
"""

import os
import sys
import time
from dataclasses import replace
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "identifyFeatures"))
sys.path.insert(0, str(ROOT / "liveSettings"))

from PyQt5.QtWidgets import QApplication  # noqa: E402

from identify_features.config import PipelineConfig  # noqa: E402
from live_tune.input_loader import load_from_geojsons  # noqa: E402
from live_tune.preview_pane import LivePreviewPane, _bgr_to_qpixmap  # noqa: E402

_IDF = ROOT / "identifyFeatures"
_DET = _IDF / "geojsons" / "0003_detections.geojson"
_LM = _IDF / "LandmarLocator_2_std30" / "0003_landmarks.geojson"
_IMG = _IDF / "OGpics" / "0003.bmp"


def _pump(app, predicate, timeout=30.0):
    t0 = time.time()
    while time.time() - t0 < timeout:
        app.processEvents()
        if predicate():
            return True
        time.sleep(0.02)
    return False


def main() -> int:
    app = QApplication.instance() or QApplication(sys.argv)
    cfg = {"v": PipelineConfig()}
    results = []

    pane = LivePreviewPane(get_config=lambda: cfg["v"])
    pane._on_result_orig = pane._on_result

    def spy(result):
        results.append(result)
        pane._on_result_orig(result)

    pane._on_result = spy
    pane._worker.result_ready.connect(spy)

    # The pane is now raw-image-only (preprocessing needs the DL models, which we
    # can't run headlessly). Inject a GeoJSON-based loader straight into the
    # worker to exercise the same load→tier-cascade→render round-trip the UI
    # would drive, without the models.
    def _loader():
        return load_from_geojsons(_DET, _LM, _IMG, um_per_px=2.0)

    pane._worker.request_load(_loader, pane._preview_scale, cfg["v"], pane._appearance)
    # request_load runs the loader off-thread and emits load_done → mark loaded.
    ok = _pump(app, lambda: pane._loaded and len(results) >= 1)
    assert ok, "load + first overlay did not arrive"
    first = results[-1]
    assert first.error is None, f"first result error: {first.error}"
    assert first.overlay_bgr is not None and first.n_veins > 0
    print(f"[load] tier={first.tier_ran} veins={first.n_veins} ok")

    # Tier B change.
    n0 = len(results)
    cfg["v"] = replace(cfg["v"], snap_radius_um=60.0)
    pane.on_config_changed("snap_radius_um")
    ok = _pump(app, lambda: len(results) > n0)
    assert ok, "tier B update did not arrive"
    rb = results[-1]
    assert "B_trace" in rb.timings_ms and "A_core" not in rb.timings_ms
    print(f"[tierB] {rb.timings_ms} ok")

    # Tier D appearance change (toggle veins off).
    n1 = len(results)
    pane.cb_veins.setChecked(False)
    ok = _pump(app, lambda: len(results) > n1)
    assert ok, "tier D update did not arrive"
    print(f"[tierD] tier={results[-1].tier_ran} ok")

    # Intervein refresh (slow).
    n2 = len(results)
    pane._on_intervein_clicked()
    ok = _pump(app, lambda: len(results) > n2, timeout=60.0)
    assert ok, "intervein refresh did not arrive"
    ri = results[-1]
    assert ri.tier_ran in ("C", "error")
    print(f"[intervein] tier={ri.tier_ran} stale={ri.regions_stale} err={ri.error}")

    # View switch: select the skeleton view via the combo (index 0). The pane
    # must push the view to the worker and re-render without a trace.
    from live_tune.session import VIEW_SKELETON, VIEW_TRACED
    n3 = len(results)
    idx_skel = pane.cmb_view.findData(VIEW_SKELETON)
    pane.cmb_view.setCurrentIndex(idx_skel)
    ok = _pump(app, lambda: len(results) > n3)
    assert ok, "skeleton view switch produced no render"
    rs = results[-1]
    assert "B_trace" not in rs.timings_ms, "skeleton view should not trace"
    assert pane._worker._view == VIEW_SKELETON
    # Display controls disabled in skeleton view.
    assert not pane.cb_veins.isEnabled()
    print("[view:skeleton] no trace, controls gated ok")

    # Switch to traced view → deferred trace runs.
    n4 = len(results)
    pane.cmb_view.setCurrentIndex(pane.cmb_view.findData(VIEW_TRACED))
    ok = _pump(app, lambda: len(results) > n4)
    assert ok, "traced view switch produced no render"
    print(f"[view:traced] veins={results[-1].n_veins} ok")

    # pixmap conversion sanity
    pm = _bgr_to_qpixmap(first.overlay_bgr)
    assert not pm.isNull() and pm.width() > 0
    print(f"[pixmap] {pm.width()}x{pm.height()} ok")

    pane.shutdown()
    print("ALL PANE SMOKE CHECKS PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
