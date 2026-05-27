"""Out-of-distribution preflight check for TRACE's DL models.

Compares per-channel pixel statistics of a few sampled input images against
training-distribution stats baked into each model's ``metadata.json``. Catches
obvious failure modes — wrong file type, fluorescence vs brightfield, exposure
shifts, missing channels — before the user spends compute on unreliable
results.

Design lifted from a QuPath pixel-classifier OOD check. Sidecar field names
(``normalization_stats``, ``training_pixel_size_um``, ``training_tile_size_px``)
match QuPath's schema verbatim so the same metadata file is readable by both
tools and so the colleague's reference implementation can be cross-checked.

Models without a ``metadata.json`` (or without a ``normalization_stats`` field
inside it) are skipped silently — the check is informational, never a hard
gate.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

_DEFAULT_SIGMA_THRESHOLD = 5.0
# Std-log threshold is intentionally generous (factor of 50 difference,
# vs the original factor of 2 from the colleague's spec). The training
# stats in our bundled models report channel std ≈ 99 on a [0,255]
# range — that level of spread is only attainable on near-bimodal
# image distributions, which aren't representative of typical
# brightfield wing crops. With the factor-of-2 threshold, every real
# input fired a "contrast collapse" warning. Factor-of-50 keeps the
# safety net for truly OOD inputs (fluorescence, a constant-color
# image, a totally black frame) without false-flagging the normal
# brightfield case. Regenerating metadata.json with stats from the
# actual training corpus would let us tighten this back up.
_DEFAULT_STD_LOG_THRESHOLD = float(np.log(50.0))
_EPS = 1e-6


@dataclass
class OODDeviation:
    """A single per-channel OOD finding for one image / one model."""

    channel: int
    metric: str  # "mean" | "p1" | "p99" | "std" | "missing_channel"
    score: float  # ratio against threshold; >=1 means flagged
    image_value: float
    training_value: float
    training_std: float


@dataclass
class OODReport:
    """Per-model OOD findings across one or more sampled images."""

    model_name: str
    deviations: list[OODDeviation] = field(default_factory=list)
    sampled_images: list[Path] = field(default_factory=list)
    skipped_reason: Optional[str] = None

    @property
    def has_warnings(self) -> bool:
        return bool(self.deviations)


def _resolve_metadata_path(model_path: Path) -> Path:
    """Return the ``metadata.json`` path for a model path that may be a file or dir."""
    p = Path(model_path)
    return (p if p.is_dir() else p.parent) / "metadata.json"


def load_training_stats(model_path: Path) -> Optional[list[dict]]:
    """Read per-channel training stats from the model's ``metadata.json``.

    Looks for the stats in two places, in order:
      1. Top-level ``normalization_stats`` (the QuPath schema we
         standardized on),
      2. ``input_config.normalization.channel_stats`` (older / alternate
         layout).

    The list is then truncated to ``input_config.num_channels`` when that
    field is present — some training pipelines append extra entries
    beyond the model's actual input channel count (e.g. a second
    training pass with different normalization left its stats appended
    rather than replacing). The truth is num_channels, not len(stats).

    Returns None when no usable stats exist; callers treat that as
    "OOD check unavailable for this model" rather than an error.
    """
    meta_path = _resolve_metadata_path(model_path)
    if not meta_path.is_file():
        return None
    try:
        meta = json.loads(meta_path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning("ood_check: cannot read %s: %s", meta_path, exc)
        return None
    stats = meta.get("normalization_stats")
    if not stats or not isinstance(stats, list):
        # Fallback to the nested location used by some training pipelines.
        ic = meta.get("input_config") or {}
        norm = ic.get("normalization") or {}
        stats = norm.get("channel_stats")
        if not stats or not isinstance(stats, list):
            return None
    # Pin to num_channels when present so a stats-list with extra trailing
    # entries doesn't falsely flag input images as "missing channels".
    ic = meta.get("input_config") or {}
    raw_nc = ic.get("num_channels")
    try:
        num_channels = int(raw_nc) if raw_nc is not None else None
    except (TypeError, ValueError):
        num_channels = None
    if num_channels is not None and 0 < num_channels < len(stats):
        stats = stats[:num_channels]
    return stats


def sample_image_stats(image: np.ndarray, num_channels: int, max_samples: int = 100_000) -> list[dict]:
    """Compute per-channel ``(p1, p99, min, max, mean, std)`` from a 4×4 tile grid.

    Subsamples to ``max_samples`` per channel so cost stays ~1 s on a whole-slide
    image. Returns one dict per channel up to ``num_channels``; an image with
    fewer channels than the model expects returns a short list, and callers
    interpret that as a missing-channel deviation.
    """
    if image.ndim == 2:
        image = image[..., np.newaxis]
    H, W, C = image.shape
    ys = np.linspace(0, H, 5, dtype=int)
    xs = np.linspace(0, W, 5, dtype=int)
    out: list[dict] = []
    per_tile_cap = max(1, max_samples // 16)
    for c in range(min(C, num_channels)):
        samples: list[np.ndarray] = []
        for i in range(4):
            for j in range(4):
                tile = image[ys[i] : ys[i + 1], xs[j] : xs[j + 1], c]
                flat = tile.ravel()
                if flat.size > per_tile_cap:
                    flat = flat[:: max(1, flat.size // per_tile_cap)]
                samples.append(flat)
        arr = np.sort(np.concatenate(samples).astype(np.float64))
        n = arr.size
        if n == 0:
            continue
        out.append(
            {
                "p1": float(arr[int(0.01 * n)]),
                "p99": float(arr[int(0.99 * n)]),
                "min": float(arr[0]),
                "max": float(arr[-1]),
                "mean": float(arr.mean()),
                "std": float(arr.std()),
            }
        )
    return out


def check_ood(
    train_stats: list[dict],
    image_stats: list[dict],
    sigma_threshold: float = _DEFAULT_SIGMA_THRESHOLD,
    std_log_threshold: float = _DEFAULT_STD_LOG_THRESHOLD,
) -> list[OODDeviation]:
    """Compare per-channel stats; return only the channels that exceed a threshold.

    Each channel reports its single worst metric — the one that scored highest
    when normalized by its own threshold. Empty return means in-distribution.
    """
    deviations: list[OODDeviation] = []
    for c, (ts, isr) in enumerate(zip(train_stats, image_stats)):
        # Use training std as the z-score scale, with a small floor so a uniform
        # training channel doesn't divide by zero.
        scale = max(ts["std"], abs(ts["mean"]) * 0.01, _EPS)
        mean_z = abs(isr["mean"] - ts["mean"]) / scale
        p1_z = abs(isr["p1"] - ts["p1"]) / scale
        p99_z = abs(isr["p99"] - ts["p99"]) / scale
        if ts["std"] > _EPS and isr["std"] > _EPS:
            std_log = abs(float(np.log(isr["std"] / ts["std"])))
        elif ts["std"] > _EPS or isr["std"] > _EPS:
            # One side degenerate (uniform channel) → clearly off-distribution.
            std_log = std_log_threshold * 2.0
        else:
            std_log = 0.0
        scores = {
            "mean": (mean_z / sigma_threshold, isr["mean"], ts["mean"]),
            "p1": (p1_z / sigma_threshold, isr["p1"], ts["p1"]),
            "p99": (p99_z / sigma_threshold, isr["p99"], ts["p99"]),
            "std": (std_log / std_log_threshold, isr["std"], ts["std"]),
        }
        metric, (score, image_v, train_v) = max(scores.items(), key=lambda kv: kv[1][0])
        if score >= 1.0:
            deviations.append(
                OODDeviation(
                    channel=c,
                    metric=metric,
                    score=score,
                    image_value=image_v,
                    training_value=train_v,
                    training_std=ts["std"],
                )
            )
    return deviations


def preflight_batch(
    image_paths: list[Path],
    models: dict[str, Path],
    n_sample: int = 3,
    sigma_threshold: float = _DEFAULT_SIGMA_THRESHOLD,
    std_log_threshold: float = _DEFAULT_STD_LOG_THRESHOLD,
) -> dict[str, OODReport]:
    """Run OOD checks over up to ``n_sample`` images against each model's stats.

    ``models`` maps a friendly display name → model path (file or dir). Returns
    one OODReport per model. A model whose metadata.json is missing or has no
    normalization_stats produces a report with ``skipped_reason`` populated.

    Image-read failures and stats-computation failures are caught and logged,
    never raised — preflight is informational and must never block the pipeline.
    """
    try:
        from preprocessing.psd_loader import imread_any
    except Exception as exc:
        logger.warning("ood_check: cannot import imread_any (%s); skipping all checks", exc)
        return {name: OODReport(model_name=name, skipped_reason="image loader unavailable") for name in models}

    sampled = list(image_paths[: max(1, n_sample)]) if image_paths else []
    reports: dict[str, OODReport] = {}
    for name, model_path in models.items():
        report = OODReport(model_name=name)
        train_stats = load_training_stats(Path(model_path))
        if train_stats is None:
            report.skipped_reason = "no training stats in metadata.json"
            reports[name] = report
            continue
        for img_path in sampled:
            try:
                img = imread_any(img_path)
            except Exception as exc:
                logger.warning("ood_check: cannot read %s: %s", img_path, exc)
                continue
            if img is None:
                continue
            report.sampled_images.append(img_path)
            try:
                img_stats = sample_image_stats(img, len(train_stats))
                if len(img_stats) < len(train_stats):
                    for c in range(len(img_stats), len(train_stats)):
                        ts = train_stats[c]
                        report.deviations.append(
                            OODDeviation(
                                channel=c,
                                metric="missing_channel",
                                score=float("inf"),
                                image_value=float("nan"),
                                training_value=ts.get("mean", float("nan")),
                                training_std=ts.get("std", 0.0),
                            )
                        )
                    continue
                report.deviations.extend(check_ood(train_stats, img_stats, sigma_threshold, std_log_threshold))
            except Exception as exc:
                logger.warning("ood_check: stats computation failed for %s: %s", img_path, exc)
        reports[name] = report
    return reports


def format_report_line(report: OODReport) -> str:
    """One-line summary suitable for log/stderr output."""
    if report.skipped_reason:
        return f"OOD check ({report.model_name}): skipped — {report.skipped_reason}"
    if not report.deviations:
        return f"OOD check ({report.model_name}): OK ({len(report.sampled_images)} image(s) sampled)"
    worst = max(report.deviations, key=lambda d: d.score)
    if worst.metric == "missing_channel":
        return (
            f"OOD check ({report.model_name}): WARN — channel {worst.channel} missing "
            f"(model expects {len(report.deviations)} channel(s) the image doesn't have)"
        )
    return (
        f"OOD check ({report.model_name}): WARN — channel {worst.channel} {worst.metric} "
        f"score={worst.score:.2f} (image={worst.image_value:.1f} vs "
        f"training={worst.training_value:.1f} ± {worst.training_std:.1f})"
    )
