"""Live vein-tuning preview for TRACE Advanced Settings.

A headless tiered-cache orchestrator (`LiveTuneSession`) plus Qt glue
(`worker`, `preview_pane`) that re-runs only the identifyFeatures stages a
changed parameter actually invalidates, so the vein overlay updates in
near-real-time as the user tunes Wing Graph / Tracing parameters.

See ../IMPLEMENTATION_SPEC.md for the design and the cache-tier model.
"""

from .session import (
    APPEARANCE_FIELDS,
    FIELD_TIER,
    TIER_A,
    TIER_B,
    TIER_C,
    TIER_D,
    VIEW_FINAL,
    VIEW_SKELETON,
    VIEW_TRACED,
    Appearance,
    LiveTuneSession,
    RenderResult,
)

__all__ = [
    "LiveTuneSession",
    "RenderResult",
    "Appearance",
    "FIELD_TIER",
    "APPEARANCE_FIELDS",
    "TIER_A",
    "TIER_B",
    "TIER_C",
    "TIER_D",
    "VIEW_SKELETON",
    "VIEW_TRACED",
    "VIEW_FINAL",
]
