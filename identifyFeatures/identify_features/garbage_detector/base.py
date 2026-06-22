"""Core types for the garbage detector — the unified data-quality filter framework.

A *filter* inspects intermediate pipeline state at a fixed ``FilterStage`` hook and
returns a :class:`GarbageVerdict`. When a filter fails a wing, ``run_stage_filters``
raises :class:`GarbageRejection` carrying that verdict — aborting the wing as early as
possible with a clean, one-line reason that flows straight into the CLI output and the
TRACE GUI / running log.

This module owns the *contracts*; concrete filters live under ``filters/`` and register
themselves into the registry (see ``registry.py``).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any, Optional, Protocol, runtime_checkable

if TYPE_CHECKING:  # avoid import cycles / heavy imports at module load
    from identify_features.config import PipelineConfig

logger = logging.getLogger(__name__)


class FilterStage(Enum):
    """Hook points in ``identify_features.controllers.pipeline.identify_wing``.

    One value per natural point where intermediate state first becomes available.
    A filter declares the earliest stage at which it has enough state to run. Only
    ``WING_OUTLINE`` carries a registered filter today; the rest are reserved so
    future filters slot in without changing this enum.
    """

    WING_OUTLINE = "wing_outline"  # after _compute_wing_outline (earliest)
    SKELETON = "skeleton"  # after build_skeleton_graph
    LANDMARKS = "landmarks"  # after anchor_landmarks
    WING_AXIS = "wing_axis"  # after compute_wing_axis
    VEINS = "veins"  # after trace_veins_from_landmarks
    TISSUE = "tissue"  # after assign_vein_tissue_polygons
    INTERVEIN_SPLIT = "intervein_split"  # after split_merged_intervein_polygons
    REGIONS = "regions"  # after name_intervein_regions


@dataclass
class GarbageVerdict:
    """The outcome of running one filter on one wing.

    ``reason`` is the single human-readable line shown to the user (CLI + TRACE row +
    log), so keep it short and self-contained. The remaining fields are structured
    metadata for logging / future aggregation.
    """

    filter_name: str
    stage: FilterStage
    passed: bool
    reason: str = ""
    metric_name: Optional[str] = None
    metric_value: Optional[float] = None
    threshold: Optional[tuple[float, float]] = None
    method: Optional[str] = None

    @classmethod
    def ok(cls, filter_name: str, stage: FilterStage, **kw: Any) -> "GarbageVerdict":
        return cls(filter_name=filter_name, stage=stage, passed=True, **kw)

    @classmethod
    def reject(cls, filter_name: str, stage: FilterStage, reason: str, **kw: Any) -> "GarbageVerdict":
        return cls(filter_name=filter_name, stage=stage, passed=False, reason=reason, **kw)


class GarbageRejection(Exception):
    """Raised to abort a wing flagged as bad data.

    ``str(self)`` is the clean one-line reason (no traceback) so it renders well in the
    TRACE list row (via ``_shorten_error``) and the CLI ``ABORTED — …`` line.
    """

    def __init__(self, verdict: GarbageVerdict):
        self.verdict = verdict
        super().__init__(verdict.reason)


@dataclass
class FilterContext:
    """Read-only snapshot of pipeline state passed to filters.

    A filter reads only the fields meaningful at its stage; everything else stays
    ``None``. ``config`` and ``specimen_id`` are always present. ``scratch`` lets a
    filter hand a computed value back to the pipeline (e.g. solidity → WingResult)
    without recomputing.
    """

    config: "PipelineConfig"
    specimen_id: str
    stage: Optional[FilterStage] = None

    # Stage state (populated as the pipeline progresses).
    wing_outline: Any = None  # cleaned outline (largest component, holes dropped)
    all_polys: Any = None  # raw vein+intervein polygons (pre-cleanup; needed by fragmentation)
    image_shape: Optional[tuple[int, int]] = None
    skel: Any = None
    landmarks: Any = None
    wing_axis: Any = None
    veins: Any = None
    regions: Any = None

    # Resolved batch parameters (e.g. solidity range from the batch pre-pass).
    batch: Any = None

    # Outbound scratch — filters write computed metrics here for the pipeline to reuse.
    scratch: dict[str, Any] = field(default_factory=dict)


@runtime_checkable
class GarbageFilter(Protocol):
    """A data-quality filter. Implementations register themselves via ``register_filter``."""

    name: str
    stage: FilterStage

    def enabled(self, config: "PipelineConfig") -> bool:
        """Whether this filter should run for the given config."""
        ...

    def check(self, ctx: FilterContext) -> GarbageVerdict:
        """Inspect ``ctx`` and return a verdict. Must not mutate pipeline state."""
        ...


def run_stage_filters(stage: FilterStage, ctx: FilterContext) -> list[GarbageVerdict]:
    """Run every enabled filter registered for ``stage``.

    Aborts on the first failing verdict by raising :class:`GarbageRejection` (fail
    fast — bad data should stop as early as possible). Returns the list of passing
    verdicts when all filters pass (useful for future QC logging).
    """
    from identify_features.garbage_detector.registry import REGISTRY

    ctx.stage = stage
    passed: list[GarbageVerdict] = []
    for flt in REGISTRY.get(stage, []):
        try:
            if not flt.enabled(ctx.config):
                continue
        except Exception:  # a misbehaving enabled() must never crash the pipeline
            logger.exception("garbage filter %s.enabled() raised; skipping", flt.name)
            continue
        verdict = flt.check(ctx)
        if not verdict.passed:
            logger.info(
                "garbage filter %s rejected %s: %s",
                flt.name,
                ctx.specimen_id,
                verdict.reason,
            )
            raise GarbageRejection(verdict)
        passed.append(verdict)
    return passed
