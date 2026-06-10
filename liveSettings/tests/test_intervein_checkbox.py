"""Regression: checking the "Intervein regions" box must show the regions.

Bug: intervein regions render only after the slow Tier-C step has run, but the
checkbox only toggled a display flag — so checking it while regions were stale
showed nothing. The checkbox now triggers the Tier-C compute when regions are
stale (like the Refresh button); unchecking / re-checking-when-fresh is just a
cheap re-render with no recompute.

Single-pane standalone offscreen script (matches the other Qt tests).
Run:  QT_QPA_PLATFORM=offscreen python liveSettings/tests/test_intervein_checkbox.py
"""

import os
import sys
import time
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

ROOT = Path(__file__).resolve().parents[2]
for _sub in ("", "identifyFeatures", "liveSettings"):
    _p = str(ROOT / _sub) if _sub else str(ROOT)
    if _p not in sys.path:
        sys.path.insert(0, _p)

from PyQt5.QtWidgets import QApplication  # noqa: E402

from identify_features.config import PipelineConfig  # noqa: E402
from live_tune.input_loader import load_from_geojsons  # noqa: E402
from live_tune.preview_pane import LivePreviewPane  # noqa: E402

_IDF = ROOT / "identifyFeatures"
_DET = _IDF / "geojsons" / "0003_detections.geojson"
_LM = _IDF / "LandmarLocator_2_std30" / "0003_landmarks.geojson"
_IMG = _IDF / "OGpics" / "0003.bmp"


def main() -> int:
    app = QApplication.instance() or QApplication(sys.argv)
    cfg = {"v": PipelineConfig()}
    pane = LivePreviewPane(get_config=lambda: cfg["v"])
    results = []
    pane._worker.result_ready.connect(results.append)

    def pump(pred, t=90.0):
        t0 = time.time()
        while time.time() - t0 < t:
            app.processEvents()
            if pred():
                return True
            time.sleep(0.02)
        return False

    pane._worker.request_load(
        lambda: load_from_geojsons(_DET, _LM, _IMG, um_per_px=2.0),
        pane._preview_scale, cfg["v"], pane._appearance,
    )
    assert pump(lambda: pane._loaded and results), "sample did not load"
    assert pane._regions_stale, "regions should start stale"
    print("[load] regions stale ok")

    # THE BUG: check the box → regions must get computed and shown.
    n = len(results)
    pane.cb_regions.setChecked(True)
    assert pump(lambda: len(results) > n), "checking intervein box produced no result"
    r = results[-1]
    assert r.error is None, f"intervein compute errored: {r.error}"
    assert r.tier_ran == "C", f"expected Tier-C compute, got {r.tier_ran}"
    assert len(pane._session._regions) > 0, "no intervein regions produced"
    assert pane._regions_stale is False, "stale flag not cleared after compute"
    print(f"[check] computed {len(pane._session._regions)} regions ok")

    # Uncheck → fast re-render, no recompute.
    n = len(results)
    t0 = time.time()
    pane.cb_regions.setChecked(False)
    assert pump(lambda: len(results) > n), "uncheck produced no result"
    assert results[-1].tier_ran == "D", "uncheck should be a cheap re-render"
    print(f"[uncheck] re-render only ({(time.time()-t0)*1000:.0f}ms) ok")

    # Re-check while fresh → also cheap, regions retained.
    n = len(results)
    pane.cb_regions.setChecked(True)
    assert pump(lambda: len(results) > n), "recheck produced no result"
    assert results[-1].tier_ran == "D", "recheck-when-fresh should not recompute"
    assert len(pane._session._regions) > 0
    print("[recheck-fresh] re-render only, regions retained ok")

    pane.shutdown()
    print("ALL INTERVEIN-CHECKBOX CHECKS PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
