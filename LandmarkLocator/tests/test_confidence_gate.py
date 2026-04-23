"""Unit tests for the LandmarkPredictor confidence gate."""

from __future__ import annotations

import numpy as np
import pytest

from landmark_locator.inference.predict import (
    DEFAULT_GATE_CONFIG,
    LowConfidenceLandmarkError,
    _assemble_gate_result,
    _deep_merge,
    _extract_coords_and_peaks,
    compute_heatmap_metrics,
)


def _gaussian(h: int, w: int, cy: int, cx: int, sigma: float, amplitude: float) -> np.ndarray:
    ys = np.arange(h)[:, None]
    xs = np.arange(w)[None, :]
    return amplitude * np.exp(-((ys - cy) ** 2 + (xs - cx) ** 2) / (2 * sigma * sigma))


def _make_heatmap_stack() -> tuple[np.ndarray, list[str]]:
    """Four synthetic channels: sharp, near-zero, broad plateau, bimodal."""
    h, w = 64, 96
    channels = []
    # 0: sharp Gaussian — passes all gates
    channels.append(_gaussian(h, w, 32, 48, sigma=2.5, amplitude=1.0))
    # 1: near-zero — fails peak
    channels.append(np.full((h, w), 0.02, dtype=np.float32))
    # 2: broad plateau — high peak but low sharpness (mean of local patch ~= peak)
    channels.append(np.full((h, w), 0.9, dtype=np.float32))
    # 3: bimodal — sharp enough per-peak but second peak comparable
    g1 = _gaussian(h, w, 32, 24, sigma=2.0, amplitude=1.0)
    g2 = _gaussian(h, w, 32, 72, sigma=2.0, amplitude=0.95)
    channels.append(g1 + g2)
    return np.stack(channels).astype(np.float32), ["sharp", "zero", "plateau", "bimodal"]


def _gate_cfg(core: list[str] | None = None) -> dict:
    cfg = _deep_merge(
        DEFAULT_GATE_CONFIG,
        {
            "peak": {"global": 0.3, "per_landmark": {}},
            "sharpness": {"global": 1.5, "per_landmark": {}},
            "second_peak_ratio": {"global": 0.5, "per_landmark": {}},
            "second_peak_suppression_radius_px": 10,
            "core_landmarks": list(core or []),
        },
    )
    return cfg


def test_sharp_gaussian_passes():
    hm, names = _make_heatmap_stack()
    result = _assemble_gate_result(
        hm, scale_x=1.0, scale_y=1.0, landmark_order=names, gate_config=_gate_cfg(), include_unreliable=True
    )
    assert result["reliable"]["sharp"] is True
    assert result["gate_reason"]["sharp"] == ""
    assert "sharp" in result["landmarks"]


def test_near_zero_fails_peak():
    hm, names = _make_heatmap_stack()
    result = _assemble_gate_result(
        hm, scale_x=1.0, scale_y=1.0, landmark_order=names, gate_config=_gate_cfg(), include_unreliable=True
    )
    assert result["reliable"]["zero"] is False
    assert result["gate_reason"]["zero"].startswith("peak=")


def test_plateau_fails_sharpness():
    hm, names = _make_heatmap_stack()
    result = _assemble_gate_result(
        hm, scale_x=1.0, scale_y=1.0, landmark_order=names, gate_config=_gate_cfg(), include_unreliable=True
    )
    assert result["reliable"]["plateau"] is False
    # Peak is ~0.9 so peak gate passes; the failing gate should be sharpness
    assert result["gate_reason"]["plateau"].startswith("sharpness=")


def test_bimodal_fails_second_peak_ratio():
    hm, names = _make_heatmap_stack()
    result = _assemble_gate_result(
        hm, scale_x=1.0, scale_y=1.0, landmark_order=names, gate_config=_gate_cfg(), include_unreliable=True
    )
    assert result["reliable"]["bimodal"] is False
    assert result["gate_reason"]["bimodal"].startswith("second_peak=")


def test_include_unreliable_controls_landmarks_dict():
    hm, names = _make_heatmap_stack()
    cfg = _gate_cfg()
    res_excl = _assemble_gate_result(
        hm, scale_x=1.0, scale_y=1.0, landmark_order=names, gate_config=cfg, include_unreliable=False
    )
    res_incl = _assemble_gate_result(
        hm, scale_x=1.0, scale_y=1.0, landmark_order=names, gate_config=cfg, include_unreliable=True
    )
    # Diagnostics always populated for all landmarks
    for name in names:
        assert name in res_excl["reliable"]
        assert name in res_incl["reliable"]
        assert name in res_excl["confidences"]
    # Only reliable ones survive in landmarks dict with include_unreliable=False
    assert set(res_excl["landmarks"]) == {n for n in names if res_excl["reliable"][n]}
    assert set(res_incl["landmarks"]) == set(names)


def test_core_failure_raises():
    hm, names = _make_heatmap_stack()
    cfg = _gate_cfg(core=["zero"])
    with pytest.raises(LowConfidenceLandmarkError) as exc_info:
        _assemble_gate_result(
            hm, scale_x=1.0, scale_y=1.0, landmark_order=names, gate_config=cfg, include_unreliable=True
        )
    assert "zero" in exc_info.value.failures


def test_core_failure_not_raised_when_non_core_fails():
    hm, names = _make_heatmap_stack()
    # Make "sharp" core — it should pass.
    cfg = _gate_cfg(core=["sharp"])
    result = _assemble_gate_result(
        hm, scale_x=1.0, scale_y=1.0, landmark_order=names, gate_config=cfg, include_unreliable=True
    )
    # Non-core failures don't raise
    assert result["reliable"]["zero"] is False
    assert result["reliable"]["sharp"] is True


def test_scale_is_applied_to_coords():
    hm, names = _make_heatmap_stack()
    cfg = _gate_cfg()
    result = _assemble_gate_result(
        hm, scale_x=4.0, scale_y=2.0, landmark_order=names, gate_config=cfg, include_unreliable=True
    )
    # Sharp peak at (cx=48, cy=32) → scaled (192, 64)
    x, y = result["landmarks"]["sharp"]
    assert abs(x - 192.0) < 1.0
    assert abs(y - 64.0) < 1.0


def test_core_missing_from_model_is_failure():
    hm, names = _make_heatmap_stack()
    cfg = _gate_cfg(core=["sharp", "not_predicted_by_model"])
    with pytest.raises(LowConfidenceLandmarkError) as exc_info:
        _assemble_gate_result(
            hm, scale_x=1.0, scale_y=1.0, landmark_order=names, gate_config=cfg, include_unreliable=True
        )
    assert "not_predicted_by_model" in exc_info.value.failures
    assert "not predicted" in exc_info.value.failures["not_predicted_by_model"]


def test_compute_heatmap_metrics_matches_expected_sharpness():
    """Peak / mean of 11x11 window around a sharp Gaussian should be >> 1."""
    hm = _gaussian(64, 64, 32, 32, sigma=2.0, amplitude=1.0)[None, :, :]
    _, peaks = _extract_coords_and_peaks(hm)
    metrics = compute_heatmap_metrics(hm, peaks, suppression_radius_px=10)
    assert metrics[0]["peak"] == pytest.approx(1.0, abs=1e-5)
    assert metrics[0]["sharpness"] > 1.5
    # Only one peak → second-peak ratio should be very small after suppression
    assert metrics[0]["second_peak_ratio"] < 0.2
