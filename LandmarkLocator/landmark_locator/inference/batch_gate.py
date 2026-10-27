"""Batch-specific thresholds for the three landmark confidence gates.

The sidecar ``gate_config.yaml`` stores one fixed threshold per landmark for each of the
three heatmap metrics (``peak`` and ``sharpness`` floors, ``second_peak_ratio`` ceiling),
calibrated once on the model's validation set. A batch imaged under different conditions
can sit systematically above or below that calibration. ``batch_mad`` mode instead derives
each landmark's threshold from the batch being predicted:

    floor   = median − k·1.4826·MAD      (peak, sharpness)
    ceiling = median + k·1.4826·MAD      (second_peak_ratio)

computed per landmark over every image in the batch, rounded to the precision each gate
is stored at (``GATE_METRIC_DECIMALS``). Robust statistics are used because the garbage
being detected would otherwise drag a mean/stdev fence toward itself. The fixed threshold
is kept (and the resolution notes say so) when fewer than ``min_batch_size`` images carry
the landmark or when the batch MAD is zero.

The knobs live in the gate config itself, next to the thresholds they replace::

    confidence:
      batch:
        mode: fixed | batch_mad
        k: 5.0
        min_batch_size: 25

so they flow through every existing path (sidecar YAML, runtime override, TRACE's gate
panel, ``--gate-override-yaml``).

How a batch gate is applied
---------------------------
A batch threshold needs every image's metrics before any image can be gated, so an
orchestrator runs two passes:

1. predict every image with the gates switched off (``gate_off_override``) so nothing
   aborts and every landmark's metrics are recorded;
2. resolve the batch gate (``resolve_batch_gate_config``) and re-gate the recorded
   metrics (``regate_raw`` / ``regate_shaped``) — pure dict work, no second forward pass.

The re-gate reproduces exactly what ``_assemble_gate_result`` would have done with the
resolved config: unreliable non-core landmarks are dropped unless ``include_unreliable``,
and a failed core landmark yields a ``LowConfidenceLandmarkError``.
"""

from __future__ import annotations

import copy
import math
import statistics
from pathlib import Path
from typing import Iterable, Mapping, Optional

from landmark_locator.inference.predict import (
    DEFAULT_GATE_CONFIG,
    GATE_METRIC_DECIMALS,
    LowConfidenceLandmarkError,
    _deep_merge,
    _gate_landmark,
    canonicalize_gate_config,
)

METRIC_NAMES: tuple[str, ...] = ("peak", "sharpness", "second_peak_ratio")
# Floors (reject below) vs the one ceiling (reject above).
LOWER_FENCE_METRICS: frozenset[str] = frozenset({"peak", "sharpness"})
# Shaped-metadata key -> gate metric name (``_shape_predict_result`` renames peak).
_METADATA_KEYS: dict[str, str] = {
    "peak": "confidence",
    "sharpness": "sharpness",
    "second_peak_ratio": "second_peak_ratio",
}

MAD_TO_SIGMA = 1.4826
DEFAULT_BATCH_BLOCK: dict = {"mode": "fixed", "k": 5.0, "min_batch_size": 25}
FIXED_NOTE = "fixed"


# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------


def batch_block(gate_cfg: Mapping) -> dict:
    """The ``batch`` block of a gate config, with defaults filled in."""
    block = dict(DEFAULT_BATCH_BLOCK)
    block.update(dict(gate_cfg.get("batch") or {}))
    return block


def batch_gate_active(gate_cfg: Mapping) -> bool:
    """True when the config asks for batch thresholds and at least one gate could use them."""
    if batch_block(gate_cfg).get("mode") != "batch_mad":
        return False
    return any((gate_cfg.get(m) or {}).get("enabled", True) for m in METRIC_NAMES)


def gate_off_override() -> dict:
    """Runtime override that switches every gate off for the measuring pass.

    Metric thresholds are untouched (they are only disabled) and the core set is emptied
    so no image aborts before its metrics are recorded.
    """
    return {**{m: {"enabled": False} for m in METRIC_NAMES}, "core_landmarks": []}


def effective_gate_config(checkpoint_path: Path, confidence_override: Optional[Mapping] = None) -> dict:
    """The gate config a predictor built from ``checkpoint_path`` + ``override`` would use.

    Same precedence as ``LandmarkPredictor.__init__`` (defaults ← sidecar ← override) but
    without loading the model, so an orchestrator can decide on batch mode up front.
    """
    from landmark_locator.inference.predict import _load_sidecar_gate

    base = dict(DEFAULT_GATE_CONFIG)
    _path, sidecar = _load_sidecar_gate(Path(checkpoint_path))
    if sidecar:
        base = _deep_merge(base, sidecar)
    return canonicalize_gate_config(_deep_merge(base, dict(confidence_override or {})))


# ---------------------------------------------------------------------------
# Metric collection
# ---------------------------------------------------------------------------


def metrics_from_raw(result: Mapping) -> dict[str, dict[str, float]]:
    """``{internal_name: {peak, sharpness, second_peak_ratio}}`` from a raw predictor result."""
    out: dict[str, dict[str, float]] = {}
    for name, peak in (result.get("confidences") or {}).items():
        out[name] = {
            "peak": float(peak),
            "sharpness": float((result.get("sharpness") or {}).get(name, 0.0)),
            "second_peak_ratio": float((result.get("second_peak_ratio") or {}).get(name, 0.0)),
        }
    return out


def metrics_from_metadata(
    metadata: Mapping, name_map: Optional[Mapping[str, str]] = None
) -> dict[str, dict[str, float]]:
    """Same as ``metrics_from_raw`` but from shaped per-landmark metadata.

    ``metadata`` is ``{geojson_name: {confidence, sharpness, second_peak_ratio, ...}}`` as
    written by ``preprocessing``; ``name_map`` translates GeoJSON names to the internal
    names the gate config is keyed by (identity when omitted).
    """
    name_map = name_map or {}
    out: dict[str, dict[str, float]] = {}
    for gj_name, md in (metadata or {}).items():
        if not isinstance(md, Mapping) or "confidence" not in md:
            continue
        internal = name_map.get(gj_name, gj_name)
        out[internal] = {m: float(md.get(key, 0.0)) for m, key in _METADATA_KEYS.items()}
    return out


def collect_metrics(per_image: Iterable[Mapping[str, Mapping[str, float]]]) -> dict[str, dict[str, list[float]]]:
    """Pool per-image metric dicts into ``{internal_name: {metric: [values...]}}``."""
    pooled: dict[str, dict[str, list[float]]] = {}
    for image_metrics in per_image:
        for name, metrics in image_metrics.items():
            bucket = pooled.setdefault(name, {m: [] for m in METRIC_NAMES})
            for m in METRIC_NAMES:
                v = metrics.get(m)
                if v is not None and math.isfinite(v):
                    bucket[m].append(float(v))
    return pooled


# ---------------------------------------------------------------------------
# Fence maths + resolution
# ---------------------------------------------------------------------------


def robust_fence(values: list[float], k: float, *, lower: bool) -> Optional[float]:
    """``median ∓ k·1.4826·MAD`` (floor when ``lower``, ceiling otherwise); None if degenerate."""
    if not values:
        return None
    median = statistics.median(values)
    mad = statistics.median([abs(v - median) for v in values])
    sigma = MAD_TO_SIGMA * mad
    if not sigma > 0:
        return None
    return median - k * sigma if lower else median + k * sigma


def resolve_batch_gate_config(
    gate_cfg: Mapping,
    collected: Mapping[str, Mapping[str, list[float]]],
) -> tuple[dict, dict[str, dict[str, str]]]:
    """Return ``(resolved_gate_config, notes)`` with batch fences in ``per_landmark``.

    Only enabled metrics are touched. ``notes[metric][landmark]`` says where each threshold
    came from: ``"batch MAD k=5 (n=80)"``, or ``"fixed: batch of 7 < 25"`` /
    ``"fixed: batch MAD is zero (n=80)"`` when the base threshold was kept. In ``fixed``
    mode the config is returned unchanged with empty notes.
    """
    resolved = copy.deepcopy(dict(gate_cfg))
    notes: dict[str, dict[str, str]] = {}
    if batch_block(gate_cfg).get("mode") != "batch_mad":
        return resolved, notes
    block = batch_block(gate_cfg)
    k = float(block["k"])
    min_size = int(block["min_batch_size"])
    for metric in METRIC_NAMES:
        mcfg = resolved.setdefault(metric, {"global": 0.0, "per_landmark": {}})
        if not mcfg.get("enabled", True):
            continue
        per_landmark = mcfg.setdefault("per_landmark", {})
        decimals = GATE_METRIC_DECIMALS[metric]
        metric_notes = notes.setdefault(metric, {})
        for name, metrics in collected.items():
            values = list(metrics.get(metric, []))
            n = len(values)
            if n < min_size:
                metric_notes[name] = f"{FIXED_NOTE}: batch of {n} < {min_size}"
                continue
            fence = robust_fence(values, k, lower=metric in LOWER_FENCE_METRICS)
            if fence is None:
                metric_notes[name] = f"{FIXED_NOTE}: batch MAD is zero (n={n})"
                continue
            per_landmark[name] = round(fence, decimals)
            metric_notes[name] = f"batch MAD k={k:g} (n={n})"
    return resolved, notes


def describe_resolution(resolved: Mapping, notes: Mapping[str, Mapping[str, str]]) -> list[str]:
    """One human-readable line per (metric, landmark) for logs."""
    lines: list[str] = []
    for metric in METRIC_NAMES:
        mnotes = notes.get(metric) or {}
        if not mnotes:
            continue
        op = "≤" if metric not in LOWER_FENCE_METRICS else "≥"
        d = GATE_METRIC_DECIMALS[metric]
        mcfg = resolved.get(metric) or {}
        for name, note in mnotes.items():
            thr = (mcfg.get("per_landmark") or {}).get(name, mcfg.get("global", 0.0))
            lines.append(f"{metric} {name}: {op} {thr:.{d}f} ({note})")
    return lines


# ---------------------------------------------------------------------------
# Re-gating recorded metrics
# ---------------------------------------------------------------------------


def _core_failures(
    reliable: Mapping[str, bool], gate_reason: Mapping[str, str], predicted: Iterable[str], gate_cfg: Mapping
) -> dict[str, str]:
    core = set(gate_cfg.get("core_landmarks", []) or [])
    predicted = list(predicted)
    failures = {n: gate_reason.get(n, "") for n in predicted if n in core and not reliable.get(n, True)}
    for missing in core - set(predicted):
        failures[missing] = "not predicted by model"
    return failures


def regate_raw(result: Mapping, gate_cfg: Mapping, landmark_order: list[str], *, include_unreliable: bool) -> dict:
    """Apply ``gate_cfg`` to a raw (gates-off) predictor result; returns a new result dict.

    The returned dict has the ``predict()`` shape plus ``error`` (None, or the
    ``LowConfidenceLandmarkError`` a gated prediction would have raised).
    """
    metrics = metrics_from_raw(result)
    all_coords = result.get("landmarks") or {}
    landmarks: dict = {}
    reliable: dict[str, bool] = {}
    gate_reason: dict[str, str] = {}
    for name in landmark_order:
        if name not in metrics:
            continue
        passed, reason = _gate_landmark(name, metrics[name], gate_cfg)
        reliable[name] = passed
        gate_reason[name] = reason
        if name in all_coords and (passed or include_unreliable):
            landmarks[name] = all_coords[name]
    failures = _core_failures(reliable, gate_reason, [n for n in landmark_order if n in metrics], gate_cfg)
    out = dict(result)
    out.update(
        {
            "landmarks": landmarks,
            "reliable": reliable,
            "gate_reason": gate_reason,
            "error": LowConfidenceLandmarkError(failures) if failures else None,
        }
    )
    return out


def regate_shaped(
    landmarks: Mapping,
    metadata: Mapping,
    gate_cfg: Mapping,
    *,
    name_map: Optional[Mapping[str, str]] = None,
    include_unreliable: bool,
) -> tuple[dict, dict, Optional[LowConfidenceLandmarkError]]:
    """Apply ``gate_cfg`` to shaped ``(landmarks, metadata)`` keyed by GeoJSON names.

    Returns ``(landmarks, metadata, error)`` in the same shape ``preprocessing`` stores on
    a ``PipelineResult``: unreliable non-core landmarks are dropped unless
    ``include_unreliable``, metadata carries the new ``reliable`` / ``gate_reason``, and
    ``error`` is the ``LowConfidenceLandmarkError`` for failed core landmarks (or None).
    ``name_map`` translates GeoJSON names to the internal names the config is keyed by.
    """
    name_map = dict(name_map or {})
    metrics_by_internal = metrics_from_metadata(metadata, name_map)
    new_landmarks: dict = {}
    new_metadata: dict = {}
    reliable: dict[str, bool] = {}
    gate_reason: dict[str, str] = {}
    for gj_name, md in (metadata or {}).items():
        internal = name_map.get(gj_name, gj_name)
        new_md = dict(md)
        if internal in metrics_by_internal:
            passed, reason = _gate_landmark(internal, metrics_by_internal[internal], gate_cfg)
        else:  # no metrics recorded (e.g. a manual override) — leave as is
            passed, reason = bool(md.get("reliable", True)), str(md.get("gate_reason", ""))
        reliable[internal] = passed
        gate_reason[internal] = reason
        new_md["reliable"] = passed
        new_md["gate_reason"] = reason
        new_metadata[gj_name] = new_md
        if gj_name in landmarks and (passed or include_unreliable):
            new_landmarks[gj_name] = landmarks[gj_name]
    failures = _core_failures(reliable, gate_reason, reliable.keys(), gate_cfg)
    return new_landmarks, new_metadata, (LowConfidenceLandmarkError(failures) if failures else None)
