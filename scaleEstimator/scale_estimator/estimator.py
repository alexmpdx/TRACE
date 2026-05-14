"""Estimate µm/px from the L3-distal-end ↔ L1-Rs-junction landmark distance on wing images."""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional, Sequence

logger = logging.getLogger(__name__)

# Internal LandmarkLocator names (snake_case, as used in the predictor's
# `landmarks` dict). The user-facing names are "L3 distal end" and "L1-Rs junction".
_LANDMARK_A = "dtip"
_LANDMARK_B = "l1_rs"
_LANDMARK_A_DISPLAY = "L3 distal end"
_LANDMARK_B_DISPLAY = "L1-Rs junction"

# Default assumed real-world distance between the L3 distal end and the L1-Rs
# junction on a Drosophila wing. The user can override per call.
DEFAULT_REFERENCE_DISTANCE_UM = 2200.0


@dataclass
class ScaleEstimate:
    """Result of a single-image scale estimation."""

    um_per_px: float
    distance_px: float
    reference_distance_um: float
    dtip_xy: tuple[float, float]
    l1_rs_xy: tuple[float, float]
    dtip_reliable: bool
    l1_rs_reliable: bool


@dataclass
class FolderScaleEstimate:
    """Result of aggregating per-image estimates across a folder."""

    um_per_px: float  # median across successful images
    reference_distance_um: float
    n_used: int  # images that produced a usable per-image estimate
    n_tried: int  # images attempted (== len(image_paths) minus any skipped before inference)
    per_image: list[tuple[Path, Optional[ScaleEstimate], Optional[str]]] = field(default_factory=list)
    cancelled: bool = False


class ScaleEstimationError(RuntimeError):
    """Raised when the estimate cannot be produced (missing landmark, bad image, etc.)."""


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------
def _result_to_estimate(
    result: dict,
    reference_distance_um: float,
) -> ScaleEstimate:
    """Convert a predictor result dict into a ScaleEstimate.

    Raises ScaleEstimationError when the required landmarks are missing or
    coincide.
    """
    landmarks = result.get("landmarks", {}) or {}
    reliable = result.get("reliable", {}) or {}

    pa = landmarks.get(_LANDMARK_A)
    pb = landmarks.get(_LANDMARK_B)
    missing = [name for name, p in ((_LANDMARK_A_DISPLAY, pa), (_LANDMARK_B_DISPLAY, pb)) if p is None]
    if missing:
        raise ScaleEstimationError("Cannot estimate scale — landmark(s) not detected: " + ", ".join(missing))

    dx = float(pb[0]) - float(pa[0])
    dy = float(pb[1]) - float(pa[1])
    distance_px = math.hypot(dx, dy)
    if distance_px <= 0:
        raise ScaleEstimationError(
            f"{_LANDMARK_A_DISPLAY} and {_LANDMARK_B_DISPLAY} predictions coincide; cannot compute scale."
        )

    return ScaleEstimate(
        um_per_px=reference_distance_um / distance_px,
        distance_px=distance_px,
        reference_distance_um=reference_distance_um,
        dtip_xy=(float(pa[0]), float(pa[1])),
        l1_rs_xy=(float(pb[0]), float(pb[1])),
        dtip_reliable=bool(reliable.get(_LANDMARK_A, True)),
        l1_rs_reliable=bool(reliable.get(_LANDMARK_B, True)),
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def estimate_um_per_px(
    image_path: Path,
    landmark_checkpoint: Path,
    *,
    reference_distance_um: float = DEFAULT_REFERENCE_DISTANCE_UM,
) -> ScaleEstimate:
    """Run LandmarkLocator on a single `image_path` and derive µm/px from the
    L3-distal-end ↔ L1-Rs-junction pixel distance.

    Args:
        image_path: wing image to analyse (any format `imread_any` accepts).
        landmark_checkpoint: .pt file or fold folder accepted by
            `landmark_locator.make_predictor`.
        reference_distance_um: assumed real-world L3-distal-end ↔ L1-Rs-junction
            distance, in µm.

    Returns:
        ScaleEstimate with `um_per_px = reference_distance_um / distance_px`.

    Raises:
        ValueError:           reference distance non-positive.
        IOError:              image fails to load.
        ScaleEstimationError: predictor produced no L3 distal end or no L1-Rs
                              junction, or the two points coincide.
    """
    if reference_distance_um <= 0:
        raise ValueError(f"reference_distance_um must be positive, got {reference_distance_um!r}")

    from landmark_locator import make_predictor
    from landmark_locator.data.psd_loader import imread_any

    image_path = Path(image_path)
    landmark_checkpoint = Path(landmark_checkpoint)

    image = imread_any(image_path)
    if image is None:
        raise IOError(f"Failed to load image: {image_path}")

    predictor = make_predictor(landmark_checkpoint)
    # `include_unreliable=True` keeps the points in the result even if the
    # confidence gate would have flagged them — caller still gets a usable
    # number and can inspect the `*_reliable` flags to decide whether to trust it.
    result = predictor.predict(image, include_unreliable=True)
    estimate = _result_to_estimate(result, reference_distance_um)
    logger.info(
        "scaleEstimator: %s → %.4f µm/px (L3 distal end ↔ L1-Rs junction = %.1f px, reference %.1f µm)",
        image_path.name,
        estimate.um_per_px,
        estimate.distance_px,
        estimate.reference_distance_um,
    )
    return estimate


def estimate_um_per_px_from_paths(
    image_paths: Sequence[Path],
    landmark_checkpoint: Path,
    *,
    reference_distance_um: float = DEFAULT_REFERENCE_DISTANCE_UM,
    progress_callback: Optional[Callable[[int, int], bool]] = None,
) -> FolderScaleEstimate:
    """Run LandmarkLocator on many images and aggregate per-image µm/px via the median.

    Per-image failures (unreadable image, missing landmark, coincident points)
    are skipped — only successful estimates contribute to the median.

    Args:
        image_paths: ordered list of images to estimate against.
        landmark_checkpoint: .pt file or fold folder.
        reference_distance_um: assumed real-world L3-distal-end ↔ L1-Rs-junction
            distance, in µm.
        progress_callback: optional `callback(done, total)` invoked after each
            image is processed. Return True to cancel; whatever estimates have
            been gathered so far are aggregated and returned with
            `cancelled=True`.

    Returns:
        FolderScaleEstimate. `um_per_px` is the median across successful images.

    Raises:
        ValueError:           empty path list or non-positive reference distance.
        ScaleEstimationError: zero images produced a usable estimate.
    """
    if reference_distance_um <= 0:
        raise ValueError(f"reference_distance_um must be positive, got {reference_distance_um!r}")
    if not image_paths:
        raise ValueError("image_paths is empty")

    from landmark_locator import auto_batch_size, make_predictor
    from landmark_locator.data.psd_loader import imread_any

    paths: list[Path] = [Path(p) for p in image_paths]
    predictor = make_predictor(Path(landmark_checkpoint))
    batch_size = auto_batch_size(len(paths))

    per_image: list[tuple[Path, Optional[ScaleEstimate], Optional[str]]] = []
    cancelled = False
    done = 0
    total = len(paths)

    for chunk_start in range(0, total, batch_size):
        if cancelled:
            break
        chunk_paths = paths[chunk_start : chunk_start + batch_size]
        chunk_images = []
        chunk_valid_paths = []
        # Per-chunk load step — record load failures up front so per_image stays
        # aligned with the input order.
        load_failures: list[tuple[Path, str]] = []
        for p in chunk_paths:
            img = imread_any(p)
            if img is None:
                load_failures.append((p, "failed to load image"))
                continue
            chunk_images.append(img)
            chunk_valid_paths.append(p)

        results: list[dict] = []
        if chunk_images:
            results = predictor.predict_batch(chunk_images, include_unreliable=True, raise_on_core_fail=False)

        load_idx = 0
        result_idx = 0
        for p in chunk_paths:
            if load_idx < len(load_failures) and load_failures[load_idx][0] == p:
                per_image.append((p, None, load_failures[load_idx][1]))
                load_idx += 1
            else:
                result = results[result_idx] if result_idx < len(results) else {}
                result_idx += 1
                try:
                    estimate = _result_to_estimate(result, reference_distance_um)
                except ScaleEstimationError as exc:
                    per_image.append((p, None, str(exc)))
                else:
                    per_image.append((p, estimate, None))
            done += 1
            if progress_callback is not None:
                try:
                    if progress_callback(done, total):
                        cancelled = True
                except Exception:  # noqa: BLE001
                    # A misbehaving callback shouldn't kill the estimation.
                    logger.exception("scaleEstimator progress_callback raised; ignoring.")
                if cancelled:
                    break

    successes = [est.um_per_px for _, est, _ in per_image if est is not None]
    if not successes:
        raise ScaleEstimationError(
            f"None of {total} image(s) produced a usable estimate "
            f"(L3 distal end / L1-Rs junction not reliably detected on any wing)."
        )

    successes_sorted = sorted(successes)
    n = len(successes_sorted)
    if n % 2 == 1:
        median = successes_sorted[n // 2]
    else:
        median = 0.5 * (successes_sorted[n // 2 - 1] + successes_sorted[n // 2])

    logger.info(
        "scaleEstimator: median %.4f µm/px from %d/%d image(s) (reference %.1f µm)%s",
        median,
        n,
        total,
        reference_distance_um,
        " [cancelled]" if cancelled else "",
    )
    return FolderScaleEstimate(
        um_per_px=float(median),
        reference_distance_um=reference_distance_um,
        n_used=n,
        n_tried=total,
        per_image=per_image,
        cancelled=cancelled,
    )
