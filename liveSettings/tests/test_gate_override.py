"""Regression test for issue #17: live preview dead-ends on a core landmark
that fails the confidence gate (`LowConfidenceLandmarkError`).

The fix empties the core-landmark set via gate_override for the preview only.
This test exercises the REAL gate path (`_assemble_gate_result`) with synthetic
heatmaps that reproduce the bug's failure mode (a core landmark below the
sharpness threshold), and proves:
  - WITHOUT the override → raises (the bug)
  - WITH the override     → no raise, landmark still returned (the fix)

It also verifies input_loader builds the override only when include_unreliable.

Run:  python -m pytest liveSettings/tests/test_gate_override.py -v
"""

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
for _sub in ("", "LandmarkLocator", "preprocessing", "identifyFeatures", "liveSettings"):
    _p = str(ROOT / _sub) if _sub else str(ROOT)
    if _p not in sys.path:
        sys.path.insert(0, _p)

from landmark_locator.inference import predict as P  # noqa: E402


def _heatmaps_one_blunt_peak():
    """Two channels: a sharp peak and a blunt (low-sharpness) peak.

    The blunt channel mimics the bug's l1_rs (sharpness just under threshold):
    a broad bump whose peak/patch-mean ratio is low.
    """
    h = w = 64
    hm = np.zeros((2, h, w), dtype=np.float32)
    # Channel 0: sharp, isolated spike → high sharpness, passes.
    hm[0, 32, 32] = 1.0
    # Channel 1: broad plateau → peak barely above its neighborhood mean → low sharpness.
    hm[1, 28:37, 28:37] = 0.9
    hm[1, 32, 32] = 1.0
    return hm


def _gate_cfg(core):
    return {
        "peak": {"global": 0.0, "per_landmark": {}},
        # Require high sharpness so the blunt channel fails.
        "sharpness": {"global": 5.0, "per_landmark": {}},
        "second_peak_ratio": {"global": 1.0, "per_landmark": {}},
        "second_peak_suppression_radius_px": 30,
        "core_landmarks": list(core),
        "min_metric_failures_to_reject": 1,
    }


ORDER = ["sharp_pt", "l1_rs"]  # l1_rs is the blunt one, matching the bug


def test_core_landmark_failure_raises_without_override():
    hm = _heatmaps_one_blunt_peak()
    with pytest.raises(P.LowConfidenceLandmarkError) as exc:
        P._assemble_gate_result(
            hm, 1.0, 1.0,
            landmark_order=ORDER,
            gate_config=_gate_cfg(core=["l1_rs"]),  # l1_rs IS core → aborts
            include_unreliable=True,  # does NOT prevent the core abort
        )
    assert "l1_rs" in str(exc.value)


def test_empty_core_override_prevents_raise_and_keeps_landmark():
    hm = _heatmaps_one_blunt_peak()
    # The fix: core_landmarks emptied → no core failures → no raise.
    result = P._assemble_gate_result(
        hm, 1.0, 1.0,
        landmark_order=ORDER,
        gate_config=_gate_cfg(core=[]),
        include_unreliable=True,
    )
    # Blunt landmark is still marked unreliable...
    assert result["reliable"]["l1_rs"] is False
    # ...but include_unreliable keeps it in the returned dict for tracing.
    assert "l1_rs" in result["landmarks"]
    assert "sharp_pt" in result["landmarks"]


def test_deep_merge_replaces_core_list_wholesale():
    # The override path relies on _deep_merge replacing (not appending) lists.
    base = {"core_landmarks": ["l1_rs", "l2", "l3"], "peak": {"global": 0.1}}
    merged = P._deep_merge(base, {"core_landmarks": []})
    assert merged["core_landmarks"] == []
    assert merged["peak"]["global"] == 0.1  # untouched siblings preserved


def test_input_loader_builds_override_only_when_unreliable(monkeypatch, tmp_path):
    """load_from_raw_image passes gate_override={'core_landmarks': []} iff
    include_unreliable_landmarks is True; None otherwise (production strictness)."""
    from live_tune import input_loader as il

    captured = {}
    real_det = ROOT / "identifyFeatures" / "geojsons" / "0003_detections.geojson"
    real_lm = ROOT / "identifyFeatures" / "LandmarLocator_2_std30" / "0003_landmarks.geojson"
    real_img = ROOT / "identifyFeatures" / "OGpics" / "0003.bmp"

    class _R:
        error = None
        error_stage = None
        rescale_factor = 1.0
        rotated_image_path = None
        processed_image_path = None
        segmentation_geojson_path = real_det
        landmarks_geojson_path = real_lm

    def _stub(**kwargs):
        captured.update(kwargs)
        return _R()

    import preprocessing.pipeline as pp
    monkeypatch.setattr(pp, "process_single_image", _stub)

    il.load_from_raw_image(
        image_path=real_img, output_dir=tmp_path,
        landmark_checkpoint=None, segmentation_model_dir=None,
        include_unreliable_landmarks=True,
    )
    assert captured["gate_override"] == {"core_landmarks": []}

    captured.clear()
    il.load_from_raw_image(
        image_path=real_img, output_dir=tmp_path,
        landmark_checkpoint=None, segmentation_model_dir=None,
        include_unreliable_landmarks=False,
    )
    assert captured["gate_override"] is None


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
