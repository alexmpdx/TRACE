"""Static Drosophila vein topology rules and identity cues."""

from __future__ import annotations

LONGITUDINAL_VEINS = ["L1", "L2", "L3", "L4", "L5"]
CROSSVEINS = ["ACV", "PCV"]
ALL_VEINS = ["costa"] + LONGITUDINAL_VEINS + CROSSVEINS

# Priority-ordered cue types for vein identification
CUE_TYPES = [
    "spatial_anterior_posterior",
    "margin_connectivity",
    "crossvein_anchor",
    "relative_length_rank",
    "entry_angle",
]

# Approximate normalized X positions (anterior=0, posterior=1) for each longitudinal vein
SPATIAL_PRIORS: dict[str, tuple[float, float]] = {
    "costa": (0.0, 0.05),
    "L1": (0.05, 0.15),
    "L2": (0.15, 0.30),
    "L3": (0.30, 0.50),
    "L4": (0.50, 0.70),
    "L5": (0.70, 0.90),
}
