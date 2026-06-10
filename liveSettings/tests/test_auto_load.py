"""Auto-load the first main-window image into the preview on first reveal.

- The pane seeds its image field from ``initial_image``.
- ``auto_load_if_seeded`` loads exactly once, only when an image is seeded AND
  the DL models are configured (image mode needs them), and never re-loads.
- ``_first_main_window_image`` reads the main window's image list defensively.

Standalone offscreen script (building a pane under pytest crashes Qt teardown).
Run:  QT_QPA_PLATFORM=offscreen python liveSettings/tests/test_auto_load.py
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
from live_tune.dialog_integration import _first_main_window_image  # noqa: E402
from live_tune.input_loader import load_from_geojsons  # noqa: E402
from live_tune.preview_pane import LivePreviewPane  # noqa: E402

_IDF = ROOT / "identifyFeatures"
_DET = _IDF / "geojsons" / "0003_detections.geojson"
_LM = _IDF / "LandmarLocator_2_std30" / "0003_landmarks.geojson"
_IMG = _IDF / "OGpics" / "0003.bmp"


def _test_first_main_window_image():
    class WithList:
        _image_paths = [Path("/a/0001.tif"), Path("/a/0002.tif")]

    class Empty:
        _image_paths = []

    class Bare:
        pass

    assert _first_main_window_image(WithList()) == "/a/0001.tif"
    assert _first_main_window_image(Empty()) == ""
    assert _first_main_window_image(Bare()) == ""  # no attr → no seed, no raise
    print("[helper] _first_main_window_image reads list defensively ok")


def main() -> int:
    app = QApplication.instance() or QApplication(sys.argv)
    _test_first_main_window_image()

    cfg = {"v": PipelineConfig()}
    # Seed the image field + provide models so image-mode auto-load is allowed.
    # Swap the loader for a GeoJSON one so we don't run the real DL models.
    pane = LivePreviewPane(
        get_config=lambda: cfg["v"],
        model_paths={"landmark_checkpoint": "x.pt", "segmentation_model_dir": "seg"},
        initial_image=str(_IMG),
    )
    assert pane.ed_image.text() == str(_IMG), "image field not seeded"
    print("[seed] image field pre-filled ok")

    results = []
    pane._worker.result_ready.connect(results.append)

    # Replace the model-backed loader with a GeoJSON loader for the test.
    pane._build_loader = lambda: (lambda: load_from_geojsons(_DET, _LM, _IMG, um_per_px=2.0))

    pane.auto_load_if_seeded()  # first reveal

    def pump(pred, t=90.0):
        t0 = time.time()
        while time.time() - t0 < t:
            app.processEvents()
            if pred():
                return True
            time.sleep(0.02)
        return False

    assert pump(lambda: pane._loaded and results), "auto-load did not load the seeded image"
    print(f"[autoload] loaded on first reveal: {results[-1].n_veins} veins ok")

    # Second reveal must NOT reload.
    n = len(results)
    pane.auto_load_if_seeded()
    app.processEvents()
    time.sleep(0.3)
    app.processEvents()
    assert len(results) == n, "auto-load fired again on second reveal"
    print("[idempotent] second reveal does not reload ok")

    # No models → no auto-load (image mode needs them), no crash.
    pane2 = LivePreviewPane(get_config=lambda: cfg["v"], model_paths={}, initial_image=str(_IMG))
    pane2.auto_load_if_seeded()
    app.processEvents()
    assert not pane2._loaded, "auto-load ran without models configured"
    print("[no-models] auto-load skipped without models ok")

    pane.shutdown()
    pane2.shutdown()
    print("ALL AUTO-LOAD CHECKS PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
