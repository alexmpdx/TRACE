"""Rough µm/px estimator for the TRACE pipeline.

Runs LandmarkLocator on a single wing image, measures the pixel distance
between the L3 distal end and the L1-Rs junction, and divides an assumed
real-world distance by it to derive µm/px.

Public API:
- estimate_um_per_px: full pipeline for one image
- ScaleEstimate: result dataclass
- DEFAULT_REFERENCE_DISTANCE_UM: the assumed L3-distal-end ↔ L1-Rs-junction distance (µm)
"""

from scale_estimator.estimator import (
    DEFAULT_REFERENCE_DISTANCE_UM,
    FolderScaleEstimate,
    ScaleEstimate,
    ScaleEstimationError,
    estimate_um_per_px,
    estimate_um_per_px_from_paths,
)

__all__ = [
    "DEFAULT_REFERENCE_DISTANCE_UM",
    "FolderScaleEstimate",
    "ScaleEstimate",
    "ScaleEstimationError",
    "estimate_um_per_px",
    "estimate_um_per_px_from_paths",
]
