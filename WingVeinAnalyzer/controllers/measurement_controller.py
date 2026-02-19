"""Aggregate vein measurements and apply scale calibration."""

from __future__ import annotations

from WingVeinAnalyzer.models.vein_labeler import VeinAssignment


def compile_results(
    assignments: list[VeinAssignment],
    microns_per_pixel: float | None = None,
) -> list[VeinAssignment]:
    """Apply scale calibration to assignments and compute derived metrics."""
    if microns_per_pixel is not None:
        for a in assignments:
            a.length_um = a.length_px * microns_per_pixel
    return assignments
