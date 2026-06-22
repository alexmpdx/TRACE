"""Garbage detector — unified home for all identifyFeatures data-quality filters.

Each filter is registered against a pipeline hook (``FilterStage``) and, on failure,
aborts the wing via :class:`GarbageRejection` with a clean one-line reason that surfaces
in the CLI output and the TRACE GUI / running log.

Importing this package registers every built-in filter (see ``filters/__init__.py``).
"""

from identify_features.garbage_detector.base import (
    FilterContext,
    FilterStage,
    GarbageFilter,
    GarbageRejection,
    GarbageVerdict,
    run_stage_filters,
)
from identify_features.garbage_detector.filters.fragmentation import compute_secondary_fraction
from identify_features.garbage_detector.filters.solidity import (
    compute_solidity,
    precompute_solidities,
    resolve_solidity_range,
)
from identify_features.garbage_detector.filters.vein_association import compute_unassigned_vein_fraction
from identify_features.garbage_detector.registry import REGISTRY, register_filter

# Importing the filters package fires each filter's @register_filter side effect.
from identify_features.garbage_detector import filters  # noqa: F401  isort:skip

__all__ = [
    "FilterContext",
    "FilterStage",
    "GarbageFilter",
    "GarbageRejection",
    "GarbageVerdict",
    "run_stage_filters",
    "REGISTRY",
    "register_filter",
    "compute_solidity",
    "precompute_solidities",
    "resolve_solidity_range",
    "compute_secondary_fraction",
    "compute_unassigned_vein_fraction",
]
