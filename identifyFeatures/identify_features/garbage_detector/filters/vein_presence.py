"""Vein-presence filter — abort when a user-required vein is missing.

Some studies can't use a wing that lacks a particular vein. This filter lets the user
name the veins that *must* be present; if any are missing, the wing is aborted. By
default no vein is required, so the filter never fires unless explicitly configured.

Runs at the VEINS hook (after tracing, where ABSENT veins are already assigned). A
canonical vein counts as "present" when some traced vein with that id has a real
centerline; an ABSENT row (centerline ``None``) or an id absent from the list both
count as missing.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from identify_features.garbage_detector.base import FilterContext, FilterStage, GarbageVerdict
from identify_features.garbage_detector.registry import register_filter

if TYPE_CHECKING:
    from identify_features.config import PipelineConfig

_FILTER_NAME = "vein_presence"


def present_vein_ids(veins) -> set[str]:
    """Set of vein ids that were actually traced (have a non-None centerline)."""
    return {v.vein_id for v in (veins or []) if getattr(v, "centerline", None) is not None}


def missing_required_veins(veins, required) -> list[str]:
    """Required veins (in the given order) that are not present."""
    present = present_vein_ids(veins)
    return [v for v in (required or []) if v not in present]


@register_filter
class VeinPresenceFilter:
    """Reject wings missing any vein the user marked as required."""

    name = _FILTER_NAME
    stage = FilterStage.VEINS

    def enabled(self, config: "PipelineConfig") -> bool:
        # No-op unless the user named at least one required vein.
        return bool(getattr(config, "required_veins", None))

    def check(self, ctx: FilterContext) -> GarbageVerdict:
        required = list(ctx.config.required_veins or [])
        missing = missing_required_veins(ctx.veins, required)
        ctx.scratch["missing_required_veins"] = missing

        if missing:
            label = "vein" if len(missing) == 1 else "veins"
            return GarbageVerdict.reject(
                self.name,
                self.stage,
                f"missing required {label}: {', '.join(missing)}",
                metric_name="missing_required_veins",
            )

        return GarbageVerdict.ok(self.name, self.stage, metric_name="missing_required_veins")
