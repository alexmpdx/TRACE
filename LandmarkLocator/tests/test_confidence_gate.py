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
    _gate_landmark,
    compute_heatmap_metrics,
    gate_metric_passes,
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


def test_min_failures_default_is_one_regression():
    """N=1 (default) must reproduce the original single-failure-rejects behavior."""
    hm, names = _make_heatmap_stack()
    cfg = _gate_cfg()
    assert cfg["min_metric_failures_to_reject"] == 1
    result = _assemble_gate_result(
        hm, scale_x=1.0, scale_y=1.0, landmark_order=names, gate_config=cfg, include_unreliable=True
    )
    # bimodal fails only sp_ratio (1 failure) — must still be rejected at N=1
    assert result["reliable"]["bimodal"] is False
    assert result["gate_reason"]["bimodal"].startswith("second_peak=")


def test_n2_single_failure_passes():
    """With N=2, a landmark failing only one gate (bimodal: sp_ratio) should be reliable."""
    hm, names = _make_heatmap_stack()
    cfg = _gate_cfg()
    cfg["min_metric_failures_to_reject"] = 2
    result = _assemble_gate_result(
        hm, scale_x=1.0, scale_y=1.0, landmark_order=names, gate_config=cfg, include_unreliable=True
    )
    assert result["reliable"]["bimodal"] is True
    assert result["gate_reason"]["bimodal"] == ""
    # Sharp channel still passes (zero failures)
    assert result["reliable"]["sharp"] is True


def test_n2_two_failures_rejects_with_joined_reason():
    """Plateau fails sharpness + sp_ratio (2 failures) → unreliable at N=2; reason joins both."""
    hm, names = _make_heatmap_stack()
    cfg = _gate_cfg()
    cfg["min_metric_failures_to_reject"] = 2
    result = _assemble_gate_result(
        hm, scale_x=1.0, scale_y=1.0, landmark_order=names, gate_config=cfg, include_unreliable=True
    )
    assert result["reliable"]["plateau"] is False
    reason = result["gate_reason"]["plateau"]
    assert "sharpness=" in reason
    assert "second_peak=" in reason
    assert "; " in reason


def test_n3_requires_all_three_failures():
    """N=3 only rejects when peak + sharpness + sp_ratio all fail (zero channel)."""
    hm, names = _make_heatmap_stack()
    cfg = _gate_cfg()
    cfg["min_metric_failures_to_reject"] = 3
    result = _assemble_gate_result(
        hm, scale_x=1.0, scale_y=1.0, landmark_order=names, gate_config=cfg, include_unreliable=True
    )
    # zero fails all three — unreliable
    assert result["reliable"]["zero"] is False
    # plateau fails two (sharpness + sp_ratio) — N=3 lets it through
    assert result["reliable"]["plateau"] is True
    # bimodal fails one — N=3 lets it through
    assert result["reliable"]["bimodal"] is True


def test_n2_with_disabled_gate_counts_only_enabled():
    """Disabling sp_ratio leaves plateau with only the sharpness failure (1) → passes at N=2."""
    hm, names = _make_heatmap_stack()
    cfg = _gate_cfg()
    cfg["min_metric_failures_to_reject"] = 2
    cfg["second_peak_ratio"]["enabled"] = False
    result = _assemble_gate_result(
        hm, scale_x=1.0, scale_y=1.0, landmark_order=names, gate_config=cfg, include_unreliable=True
    )
    # Plateau now has only the sharpness failure → 1 < 2 → reliable
    assert result["reliable"]["plateau"] is True
    # bimodal had only the sp_ratio failure; with sp_ratio disabled it's now 0 failures
    assert result["reliable"]["bimodal"] is True


def test_n2_core_landmark_failure_raises():
    """Core-landmark rejection path still triggers when N-failure threshold is met."""
    hm, names = _make_heatmap_stack()
    cfg = _gate_cfg(core=["plateau"])
    cfg["min_metric_failures_to_reject"] = 2
    with pytest.raises(LowConfidenceLandmarkError) as exc_info:
        _assemble_gate_result(
            hm, scale_x=1.0, scale_y=1.0, landmark_order=names, gate_config=cfg, include_unreliable=True
        )
    # plateau crosses N=2 (sharpness + sp_ratio) → it's the core failure
    assert "plateau" in exc_info.value.failures


def test_n2_core_landmark_with_single_failure_does_not_raise():
    """A core landmark with only 1 failure must NOT raise when N=2."""
    hm, names = _make_heatmap_stack()
    cfg = _gate_cfg(core=["bimodal"])
    cfg["min_metric_failures_to_reject"] = 2
    # bimodal has 1 failure → with N=2 it's reliable → no raise
    result = _assemble_gate_result(
        hm, scale_x=1.0, scale_y=1.0, landmark_order=names, gate_config=cfg, include_unreliable=True
    )
    assert result["reliable"]["bimodal"] is True


def test_compute_heatmap_metrics_matches_expected_sharpness():
    """Peak / mean of 11x11 window around a sharp Gaussian should be >> 1."""
    hm = _gaussian(64, 64, 32, 32, sigma=2.0, amplitude=1.0)[None, :, :]
    _, peaks = _extract_coords_and_peaks(hm)
    metrics = compute_heatmap_metrics(hm, peaks, suppression_radius_px=10)
    assert metrics[0]["peak"] == pytest.approx(1.0, abs=1e-5)
    assert metrics[0]["sharpness"] > 1.5
    # Only one peak → second-peak ratio should be very small after suppression
    assert metrics[0]["second_peak_ratio"] < 0.2


# ---- Threshold precision (regression: "sharpness=1.18<1.18") ----


def test_metric_at_threshold_precision_passes():
    """A metric below the threshold by less than its stored precision still passes.

    Regression for the report `sharpness=1.18<1.18`: the observed sharpness was
    1.17903 against a threshold of 1.18 (itself `round(<calibrated percentile>, 2)`),
    so the run failed on a difference finer than the threshold is even specified to,
    and the message rendered both sides identically.
    """
    assert gate_metric_passes("sharpness", 1.1790256484259554, 1.18) is True
    assert gate_metric_passes("sharpness", 1.174, 1.18) is False
    assert gate_metric_passes("peak", 0.02051, 0.021) is True
    assert gate_metric_passes("peak", 0.0203, 0.021) is False
    # second_peak_ratio is a ceiling, so tolerance applies on the other side
    assert gate_metric_passes("second_peak_ratio", 0.9549, 0.95) is True
    assert gate_metric_passes("second_peak_ratio", 0.96, 0.95) is False


def test_exact_equality_passes():
    for metric, value in (("peak", 0.021), ("sharpness", 1.18), ("second_peak_ratio", 0.95)):
        assert gate_metric_passes(metric, value, value) is True


def test_gate_reason_never_shows_equal_sides():
    """When a gate does fail, the two formatted sides must differ."""
    cfg = _gate_cfg()
    stack, names = _make_heatmap_stack()
    coords, peaks = _extract_coords_and_peaks(stack)
    metrics = compute_heatmap_metrics(stack, peaks, suppression_radius_px=10)
    for name, metric in zip(names, metrics):
        passed, reason = _gate_landmark(name, metric, cfg)
        if passed:
            continue
        for part in reason.split("; "):
            body = part.partition("=")[2]
            op = "<" if "<" in body else ">"
            lhs, _, rhs = body.partition(op)
            assert lhs != rhs, f"{name}: gate reason compares identical strings: {part}"


def test_sub_precision_failure_does_not_reach_the_pipeline():
    """A landmark 0.0009 under threshold is reliable, so a core landmark won't abort."""
    cfg = _deep_merge(_gate_cfg(core=["sharp"]), {"sharpness": {"global": 1.5, "per_landmark": {}}})
    stack, names = _make_heatmap_stack()
    metrics = compute_heatmap_metrics(stack, _extract_coords_and_peaks(stack)[1], suppression_radius_px=10)
    sharp_metric = dict(metrics[0])
    # Push sharpness to just under the threshold, inside the 2-dp quantum
    sharp_metric["sharpness"] = 1.4990
    passed, reason = _gate_landmark("sharp", sharp_metric, cfg)
    assert passed is True, reason
