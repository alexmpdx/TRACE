"""Tests for batch-specific landmark gate thresholds (``inference/batch_gate.py``)."""

from __future__ import annotations

import math
import statistics

import pytest

from landmark_locator.inference.batch_gate import (
    DEFAULT_BATCH_BLOCK,
    MAD_TO_SIGMA,
    batch_block,
    batch_gate_active,
    collect_metrics,
    describe_resolution,
    gate_off_override,
    metrics_from_metadata,
    metrics_from_raw,
    regate_raw,
    regate_shaped,
    resolve_batch_gate_config,
    robust_fence,
)
from landmark_locator.inference.predict import (
    DEFAULT_GATE_CONFIG,
    LowConfidenceLandmarkError,
    _deep_merge,
    _gate_landmark,
)

NAMES = ["a", "b", "core"]


def _cfg(**batch) -> dict:
    return _deep_merge(
        DEFAULT_GATE_CONFIG,
        {
            "peak": {"global": 0.3, "per_landmark": {"a": 0.4}},
            "sharpness": {"global": 1.5, "per_landmark": {}},
            "second_peak_ratio": {"global": 0.5, "per_landmark": {}},
            "core_landmarks": ["core"],
            "batch": {"mode": "batch_mad", "k": 3.0, "min_batch_size": 5, **batch},
        },
    )


def _image(peak: float, sharp: float = 3.0, spr: float = 0.2) -> dict:
    """A gates-off raw predictor result where every landmark carries the same metrics."""
    return {
        "landmarks": {n: (10.0, 20.0) for n in NAMES},
        "confidences": {n: peak for n in NAMES},
        "sharpness": {n: sharp for n in NAMES},
        "second_peak_ratio": {n: spr for n in NAMES},
        "reliable": {n: True for n in NAMES},
        "gate_reason": {n: "" for n in NAMES},
    }


def _batch(peaks: list[float]) -> list[dict]:
    return [_image(p) for p in peaks]


# ---------------------------------------------------------------------------
# config helpers
# ---------------------------------------------------------------------------


def test_defaults_are_fixed_mode():
    assert DEFAULT_GATE_CONFIG["batch"] == DEFAULT_BATCH_BLOCK
    assert batch_block({}) == DEFAULT_BATCH_BLOCK
    assert not batch_gate_active(DEFAULT_GATE_CONFIG)


def test_batch_mode_requires_an_enabled_gate():
    cfg = _cfg()
    assert batch_gate_active(cfg)
    for m in ("peak", "sharpness", "second_peak_ratio"):
        cfg[m]["enabled"] = False
    assert not batch_gate_active(cfg)


def test_gate_off_override_disables_everything_but_keeps_thresholds():
    cfg = _deep_merge(_cfg(), gate_off_override())
    assert cfg["core_landmarks"] == []
    assert cfg["peak"]["per_landmark"] == {"a": 0.4}  # thresholds survive, only disabled
    metrics = {"peak": 0.0, "sharpness": 0.0, "second_peak_ratio": 1.0}
    assert _gate_landmark("a", metrics, cfg) == (True, "")


# ---------------------------------------------------------------------------
# metric collection
# ---------------------------------------------------------------------------


def test_metrics_from_raw_and_metadata_agree():
    raw = _image(0.5, 2.0, 0.3)
    shaped = {
        "A": {"confidence": 0.5, "sharpness": 2.0, "second_peak_ratio": 0.3, "reliable": True, "gate_reason": ""},
        "Core": {"confidence": 0.5, "sharpness": 2.0, "second_peak_ratio": 0.3},
        "Manual": {"reliable": True},  # no metrics → ignored
    }
    assert metrics_from_raw(raw)["a"] == {"peak": 0.5, "sharpness": 2.0, "second_peak_ratio": 0.3}
    got = metrics_from_metadata(shaped, {"A": "a", "Core": "core"})
    assert set(got) == {"a", "core"}
    assert got["a"] == metrics_from_raw(raw)["a"]


def test_collect_metrics_pools_per_landmark_and_drops_non_finite():
    pooled = collect_metrics(
        [{"a": {"peak": 0.1, "sharpness": 1.0, "second_peak_ratio": math.nan}}, {"a": {"peak": 0.2}}]
    )
    assert pooled["a"]["peak"] == [0.1, 0.2]
    assert pooled["a"]["sharpness"] == [1.0]
    assert pooled["a"]["second_peak_ratio"] == []


# ---------------------------------------------------------------------------
# fence maths + resolution
# ---------------------------------------------------------------------------


def test_robust_fence_matches_hand_computation():
    vals = [0.80, 0.82, 0.84, 0.85, 0.86, 0.87, 0.90]
    med = statistics.median(vals)
    mad = statistics.median([abs(v - med) for v in vals])
    assert robust_fence(vals, 3.0, lower=True) == pytest.approx(med - 3.0 * MAD_TO_SIGMA * mad)
    assert robust_fence(vals, 3.0, lower=False) == pytest.approx(med + 3.0 * MAD_TO_SIGMA * mad)
    assert robust_fence([0.5] * 10, 3.0, lower=True) is None  # zero MAD
    assert robust_fence([], 3.0, lower=True) is None


def test_fixed_mode_returns_config_unchanged():
    cfg = _cfg(mode="fixed")
    resolved, notes = resolve_batch_gate_config(cfg, collect_metrics(metrics_from_raw(r) for r in _batch([0.5] * 10)))
    assert resolved == cfg and notes == {}


def test_batch_fences_replace_per_landmark_thresholds_and_are_rounded():
    peaks = [0.80, 0.82, 0.84, 0.85, 0.86, 0.87, 0.90, 0.83, 0.85, 0.84]
    collected = collect_metrics(metrics_from_raw(r) for r in _batch(peaks))
    resolved, notes = resolve_batch_gate_config(_cfg(), collected)
    med = statistics.median(peaks)
    mad = statistics.median([abs(v - med) for v in peaks])
    expected = round(med - 3.0 * MAD_TO_SIGMA * mad, 3)
    for n in NAMES:
        assert resolved["peak"]["per_landmark"][n] == expected
        assert notes["peak"][n] == "batch MAD k=3 (n=10)"
    # sharpness / spr were constant across the batch → zero MAD → base thresholds kept
    assert resolved["sharpness"]["per_landmark"] == {}
    assert "MAD is zero" in notes["sharpness"]["a"]
    assert resolved["second_peak_ratio"]["per_landmark"] == {}
    # untouched bits survive
    assert resolved["core_landmarks"] == ["core"]
    assert resolved["batch"]["mode"] == "batch_mad"


def test_small_batch_keeps_fixed_thresholds():
    collected = collect_metrics(metrics_from_raw(r) for r in _batch([0.5, 0.6, 0.7]))
    resolved, notes = resolve_batch_gate_config(_cfg(min_batch_size=25), collected)
    assert resolved["peak"]["per_landmark"] == {"a": 0.4}
    assert notes["peak"]["b"] == "fixed: batch of 3 < 25"


def test_disabled_metric_is_not_resolved():
    cfg = _cfg()
    cfg["peak"]["enabled"] = False
    peaks = [0.80, 0.82, 0.84, 0.85, 0.86, 0.87, 0.90, 0.83, 0.85, 0.84]
    resolved, notes = resolve_batch_gate_config(cfg, collect_metrics(metrics_from_raw(r) for r in _batch(peaks)))
    assert resolved["peak"]["per_landmark"] == {"a": 0.4}
    assert "peak" not in notes


def test_describe_resolution_lists_every_note():
    peaks = [0.80, 0.82, 0.84, 0.85, 0.86, 0.87, 0.90, 0.83, 0.85, 0.84]
    resolved, notes = resolve_batch_gate_config(_cfg(), collect_metrics(metrics_from_raw(r) for r in _batch(peaks)))
    lines = describe_resolution(resolved, notes)
    assert len(lines) == 9  # 3 metrics × 3 landmarks
    assert any(line.startswith("peak a: ≥ ") and "batch MAD k=3" in line for line in lines)
    assert any(line.startswith("second_peak_ratio a: ≤ ") and "MAD is zero" in line for line in lines)


# ---------------------------------------------------------------------------
# re-gating
# ---------------------------------------------------------------------------


def _resolved_with_outlier():
    good = [0.80, 0.82, 0.84, 0.85, 0.86, 0.87, 0.90, 0.83, 0.85, 0.84]
    batch = _batch(good) + [_image(0.10)]  # last image: every landmark's peak is an outlier
    collected = collect_metrics(metrics_from_raw(r) for r in batch)
    resolved, _ = resolve_batch_gate_config(_cfg(), collected)
    return batch, resolved


def test_regate_raw_rejects_outlier_core_and_drops_non_core():
    batch, resolved = _resolved_with_outlier()
    ok = regate_raw(batch[0], resolved, NAMES, include_unreliable=False)
    assert ok["error"] is None and set(ok["landmarks"]) == set(NAMES)
    bad = regate_raw(batch[-1], resolved, NAMES, include_unreliable=False)
    assert isinstance(bad["error"], LowConfidenceLandmarkError)
    assert "core" in bad["error"].failures and "peak=0.100<" in bad["error"].failures["core"]
    assert bad["landmarks"] == {}  # every landmark failed and include_unreliable is off
    assert bad["reliable"] == {n: False for n in NAMES}
    kept = regate_raw(batch[-1], resolved, NAMES, include_unreliable=True)
    assert set(kept["landmarks"]) == set(NAMES)


def test_regate_raw_reports_missing_core_landmark():
    batch, resolved = _resolved_with_outlier()
    resolved["core_landmarks"] = ["core", "ghost"]
    out = regate_raw(batch[0], resolved, NAMES, include_unreliable=False)
    assert out["error"].failures == {"ghost": "not predicted by model"}


def test_regate_shaped_mirrors_raw_through_name_map():
    batch, resolved = _resolved_with_outlier()
    name_map = {"A": "a", "B": "b", "Core": "core"}

    def shape(raw):
        inv = {v: k for k, v in name_map.items()}
        lms = {inv[n]: xy for n, xy in raw["landmarks"].items()}
        md = {
            inv[n]: {
                "reliable": True,
                "gate_reason": "",
                "confidence": raw["confidences"][n],
                "sharpness": raw["sharpness"][n],
                "second_peak_ratio": raw["second_peak_ratio"][n],
            }
            for n in NAMES
        }
        return lms, md

    lms, md = shape(batch[0])
    new_lms, new_md, err = regate_shaped(lms, md, resolved, name_map=name_map, include_unreliable=False)
    assert err is None and set(new_lms) == {"A", "B", "Core"} and all(m["reliable"] for m in new_md.values())

    lms, md = shape(batch[-1])
    new_lms, new_md, err = regate_shaped(lms, md, resolved, name_map=name_map, include_unreliable=False)
    assert isinstance(err, LowConfidenceLandmarkError) and set(err.failures) == {"core"}
    assert new_lms == {}
    assert new_md["A"]["reliable"] is False and new_md["A"]["gate_reason"].startswith("peak=0.100<")
    # original inputs untouched
    assert md["A"]["reliable"] is True


def test_regate_shaped_leaves_manual_override_entries_alone():
    _batch_, resolved = _resolved_with_outlier()
    resolved["core_landmarks"] = []  # a core landmark absent from the metadata would (rightly) abort
    lms = {"A": (1.0, 2.0)}
    md = {"A": {"reliable": True, "gate_reason": "", "confidence": 1.0}}  # override-style: no sharpness/spr
    # confidence present but sharpness missing → metrics_from_metadata still records it (0.0 defaults)
    new_lms, new_md, err = regate_shaped(lms, md, resolved, name_map={"A": "a"}, include_unreliable=True)
    assert err is None and "A" in new_lms
    md2 = {"A": {"reliable": True, "gate_reason": ""}}  # no metrics at all → passed through untouched
    new_lms, new_md, err = regate_shaped(lms, md2, resolved, name_map={"A": "a"}, include_unreliable=False)
    assert err is None and new_lms == lms and new_md["A"]["reliable"] is True
