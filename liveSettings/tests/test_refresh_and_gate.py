"""Tests for the Force-refresh button and the confidence-gate warning.

- Force refresh re-runs the loader from scratch (request_load), which feeds
  set_input → clears every cache (the stale-cache escape hatch).
- The gate warning shows when the loaded wing has any gate-failing landmark,
  and hides when all are reliable. Refreshed on load + force-refresh.

Standalone offscreen script (building a pane under pytest crashes Qt teardown).
Run:  QT_QPA_PLATFORM=offscreen python liveSettings/tests/test_refresh_and_gate.py
"""

import os
import sys
import time
from dataclasses import replace
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

ROOT = Path(__file__).resolve().parents[2]
for _sub in ("", "identifyFeatures", "liveSettings"):
    _p = str(ROOT / _sub) if _sub else str(ROOT)
    if _p not in sys.path:
        sys.path.insert(0, _p)

from PyQt5.QtWidgets import QApplication  # noqa: E402
from shapely.geometry import Point  # noqa: E402

from identify_features.config import PipelineConfig  # noqa: E402
from identify_features.models.datatypes import Landmark  # noqa: E402
from live_tune.input_loader import load_from_geojsons  # noqa: E402
from live_tune.preview_pane import LivePreviewPane  # noqa: E402

_IDF = ROOT / "identifyFeatures"
_DET = _IDF / "geojsons" / "0003_detections.geojson"
_LM = _IDF / "LandmarLocator_2_std30" / "0003_landmarks.geojson"
_IMG = _IDF / "OGpics" / "0003.bmp"


def _pump(app, predicate, t=90.0):
    t0 = time.time()
    while time.time() - t0 < t:
        app.processEvents()
        if predicate():
            return True
        time.sleep(0.02)
    return False


def main() -> int:
    app = QApplication.instance() or QApplication(sys.argv)
    cfg = {"v": PipelineConfig()}
    pane = LivePreviewPane(get_config=lambda: cfg["v"])
    results = []
    pane._worker.result_ready.connect(results.append)

    # Refresh button disabled until a sample is loaded.
    assert not pane.btn_refresh.isEnabled(), "refresh should be disabled before load"
    print("[init] refresh disabled pre-load ok")

    # Load via the worker (GeoJSON loader, no DL models needed headlessly).
    pane._worker.request_load(
        lambda: load_from_geojsons(_DET, _LM, _IMG, um_per_px=2.0),
        pane._preview_scale, cfg["v"], pane._appearance,
    )
    assert _pump(app, lambda: pane._loaded and results), "load failed"
    assert pane.btn_refresh.isEnabled(), "refresh should enable after load"
    print(f"[load] {results[-1].n_veins} veins, refresh enabled ok")

    # 0003 is all-reliable → gate warning empty. (Assert on text, not
    # isVisible(): offscreen Qt reports isVisible()==False for a never-shown
    # top-level's children even after .show().)
    assert not pane.gate_warning.text()
    assert not pane.gate_warning.isVisibleTo(pane)  # logical visibility within the pane
    print("[gate] all-reliable wing → no warning ok")

    # Inject a gate-failing landmark and re-evaluate → warning shows with reason.
    lms = pane._session._landmarks_raw
    a_name = next(iter(lms))
    lms[a_name] = replace(lms[a_name], reliable=False, gate_reason="sharpness=1.28<1.30",
                          point=Point(0, 0))
    pane._update_gate_warning()
    assert pane.gate_warning.text(), "warning text not set for failing landmark"
    assert pane.gate_warning.isVisibleTo(pane), "warning not shown for failing landmark"
    assert a_name in pane.gate_warning.text() and "1.28" in pane.gate_warning.text()
    print(f"[gate] failing landmark → warning shown: …{pane.gate_warning.text()[-60:]}")

    # Force refresh: re-runs from scratch (request_load → set_input, which calls
    # _invalidate_all → clears every cache and bumps the input epoch). Verify via
    # the epoch (set_input bumps it on each reload) rather than poking the session
    # from this thread — the worker thread owns it.
    # The real _build_loader needs DL models (absent headlessly); stub it with a
    # GeoJSON loader so we exercise the force-refresh dispatch, like the other
    # pane tests do.
    pane._build_loader = lambda: (lambda: load_from_geojsons(_DET, _LM, _IMG, um_per_px=2.0))
    epoch_before = pane._session._input_epoch
    n = len(results)
    pane._on_force_refresh()
    assert _pump(app, lambda: len(results) > n), "force refresh produced no result"
    assert pane._session._input_epoch > epoch_before, "force refresh did not re-run set_input"
    print(f"[refresh] re-ran from scratch, epoch {epoch_before} -> {pane._session._input_epoch} ok")
    # The reloaded wing is all-reliable again (fresh GeoJSON), so the warning clears.
    assert not pane.gate_warning.text(), "gate warning should reset on fresh load"
    print("[gate] warning reset after fresh load ok")

    pane.shutdown()
    print("ALL REFRESH+GATE CHECKS PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
