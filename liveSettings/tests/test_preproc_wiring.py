"""Tests for the raw-image preprocessing wiring (TODO #14 rework).

Guards the bug class that bit us before — wrong kwarg names passed to
``process_single_image`` — by binding the exact call against the REAL
signature, and by recording forwarded kwargs through a stubbed pipeline.
Also covers the preproc-getter dict built from dialog + main-window state.

Run:  python -m pytest liveSettings/tests/test_preproc_wiring.py -v
"""

import inspect
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
for _sub in ("", "identifyFeatures", "HingeChopper", "modelTOjson", "preprocessing",
             "measurementMaker", "wingIsolator", "resolutionAdjust", "scaleEstimator",
             "wingRotator", "LandmarkLocator", "liveSettings"):
    _p = str(ROOT / _sub) if _sub else str(ROOT)
    if _p not in sys.path:
        sys.path.insert(0, _p)

from live_tune import input_loader as il  # noqa: E402


# -- the kwargs load_from_raw_image forwards must bind to the real signature --
def test_forwarded_kwargs_bind_to_real_signature():
    from preprocessing.pipeline import process_single_image

    sig = inspect.signature(process_single_image)
    # Exactly the kwargs load_from_raw_image passes (names must match the API).
    kwargs = dict(
        image_path="x", output_dir="y",
        landmark_checkpoint=None, segmentation_model_dir=None,
        stages=(True, True, True), predictor_cache={}, model_cache={},
        input_um_per_px=2.0, device=None, progress_callback=None,
        wing_model_dir=None, wing_expand_fraction=0.05,
        do_rotation=False, rotation_mirror_correct=False, target_um_per_px=None,
    )
    sig.bind(**kwargs)  # raises TypeError if any kwarg name is wrong


# -- forwarding through a stubbed pipeline -------------------------------------
class _FakeResult:
    error = None
    error_stage = None
    rescale_factor = 1.0
    rotated_image_path = None
    processed_image_path = None
    segmentation_geojson_path = None
    landmarks_geojson_path = None


def test_preproc_kwargs_forwarded(monkeypatch, tmp_path):
    recorded = {}
    real_det = ROOT / "identifyFeatures" / "geojsons" / "0003_detections.geojson"
    real_lm = ROOT / "identifyFeatures" / "LandmarLocator_2_std30" / "0003_landmarks.geojson"

    def _stub(**kwargs):
        recorded.update(kwargs)
        r = _FakeResult()
        r.segmentation_geojson_path = real_det
        r.landmarks_geojson_path = real_lm
        r.processed_image_path = None
        return r

    import preprocessing.pipeline as pp
    monkeypatch.setattr(pp, "process_single_image", _stub)

    bundle = il.load_from_raw_image(
        image_path=ROOT / "identifyFeatures" / "OGpics" / "0003.bmp",
        output_dir=tmp_path,
        landmark_checkpoint=None,
        segmentation_model_dir=None,
        um_per_px=4.0,
        wing_model_dir="/some/wing/model",
        wing_expand_fraction=0.12,
        do_rotation=True,
        rotation_mirror_correct=True,
        target_um_per_px=0.5,
    )
    assert recorded["wing_expand_fraction"] == 0.12
    assert recorded["do_rotation"] is True
    assert recorded["rotation_mirror_correct"] is True
    assert recorded["target_um_per_px"] == 0.5
    assert str(recorded["wing_model_dir"]) == "/some/wing/model"
    assert recorded["input_um_per_px"] == 4.0
    assert bundle.um_per_px == 4.0  # rescale_factor 1.0 → unchanged


def test_rescale_factor_adjusts_um_per_px(monkeypatch, tmp_path):
    real_det = ROOT / "identifyFeatures" / "geojsons" / "0003_detections.geojson"
    real_lm = ROOT / "identifyFeatures" / "LandmarLocator_2_std30" / "0003_landmarks.geojson"

    def _stub(**kwargs):
        r = _FakeResult()
        r.rescale_factor = 0.5  # output is half-size → µm/px doubles
        r.segmentation_geojson_path = real_det
        r.landmarks_geojson_path = real_lm
        return r

    import preprocessing.pipeline as pp
    monkeypatch.setattr(pp, "process_single_image", _stub)

    bundle = il.load_from_raw_image(
        image_path=ROOT / "identifyFeatures" / "OGpics" / "0003.bmp", output_dir=tmp_path,
        landmark_checkpoint=None, segmentation_model_dir=None, um_per_px=2.0,
    )
    assert bundle.um_per_px == pytest.approx(4.0)  # 2.0 / 0.5


# -- preproc getter built from dialog + main window ---------------------------
class _FakeDialog:
    def get_wing_isolation_model_path(self):
        return "/models/wing"

    def get_wing_expand_fraction(self):
        return 0.07


class _FakeWindow:
    _wing_isolation_enabled = True
    _do_rotation = True
    _rotation_mirror_correct = False
    _wing_expand_fraction = 0.05

    def _resolve_active_target_um_per_px(self):
        return 0.483


def test_preproc_getter_reads_dialog_and_window():
    from live_tune.dialog_integration import _build_preproc_getter

    getter = _build_preproc_getter(_FakeDialog(), _FakeWindow())
    pp = getter()
    assert pp["wing_model_dir"] == "/models/wing"  # isolation enabled
    assert pp["wing_expand_fraction"] == 0.07
    assert pp["do_rotation"] is True
    assert pp["rotation_mirror_correct"] is False
    assert pp["target_um_per_px"] == 0.483


def test_preproc_getter_omits_wing_model_when_isolation_disabled():
    from live_tune.dialog_integration import _build_preproc_getter

    win = _FakeWindow()
    win._wing_isolation_enabled = False
    getter = _build_preproc_getter(_FakeDialog(), win)
    assert "wing_model_dir" not in getter()  # disabled → no model dir passed


def test_preproc_getter_defensive_on_missing_attrs():
    from live_tune.dialog_integration import _build_preproc_getter

    class Bare:
        pass

    # No attributes, no methods — must not raise, must yield safe defaults.
    getter = _build_preproc_getter(Bare(), Bare())
    pp = getter()
    assert pp["do_rotation"] is False
    assert pp["rotation_mirror_correct"] is False
    assert "wing_model_dir" not in pp


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
