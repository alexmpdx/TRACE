"""CSV export and summary table generation."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd

from WingVeinAnalyzer.controllers.measurement_controller import WingMeasurements
from WingVeinAnalyzer.models.vein_labeler import VeinAssignment
from WingVeinAnalyzer.models.vein_map import ALL_VEINS, INTERVEIN_SPACE_NAMES


def _build_row(
    assignments: list[VeinAssignment],
    image_name: str = "",
    measurements: Optional[WingMeasurements] = None,
) -> dict[str, object]:
    """Build a single measurement row dict for one wing."""
    row: dict[str, object] = {"image": image_name}

    # Per-vein columns
    for vein_id in ALL_VEINS:
        match = next((a for a in assignments if a.vein_id == vein_id), None)
        if match:
            row[vein_id + "_length_px"] = match.length_px
            row[vein_id + "_status"] = match.status.value
            if match.length_um is not None:
                row[vein_id + "_length_um"] = match.length_um
            if match.gap_px is not None:
                row[vein_id + "_gap_px"] = match.gap_px
        else:
            row[vein_id + "_length_px"] = None
            row[vein_id + "_status"] = "absent"

    # Wing-level measurements
    if measurements is not None:
        row["crossvein_distance_px"] = measurements.crossvein_distance_px
        row["crossvein_distance_um"] = measurements.crossvein_distance_um
        row["wing_length_px"] = measurements.wing_length_px
        row["wing_length_um"] = measurements.wing_length_um
        row["wing_width_px"] = measurements.wing_width_px
        row["wing_width_um"] = measurements.wing_width_um
        row["total_wing_area_px2"] = measurements.total_wing_area_px2
        row["total_wing_area_um2"] = measurements.total_wing_area_um2
        row["anterior_compartment_area_px2"] = measurements.anterior_compartment_area_px2
        row["anterior_compartment_area_um2"] = measurements.anterior_compartment_area_um2
        row["posterior_compartment_area_px2"] = measurements.posterior_compartment_area_px2
        row["posterior_compartment_area_um2"] = measurements.posterior_compartment_area_um2

        # Intervein areas (known regions + any extra ER regions)
        for name in INTERVEIN_SPACE_NAMES:
            row[name + "_area_px2"] = measurements.intervein_areas_px2.get(name)
            row[name + "_area_um2"] = measurements.intervein_areas_um2.get(name)
        for name in sorted(measurements.intervein_areas_px2):
            if name.startswith("ER"):
                row[name + "_area_px2"] = measurements.intervein_areas_px2[name]
                row[name + "_area_um2"] = measurements.intervein_areas_um2.get(name)

    return row


def export_csv(
    assignments: list[VeinAssignment],
    output_path: Path,
    image_name: str = "",
    measurements: Optional[WingMeasurements] = None,
) -> None:
    """Export vein measurements to CSV with per-vein status columns."""
    row = _build_row(assignments, image_name, measurements)
    df = pd.DataFrame([row])
    df.to_csv(output_path, index=False)


def consolidate_csv(
    results: list[tuple[str, list[VeinAssignment], Optional[WingMeasurements]]],
    output_path: Path,
) -> Path:
    """Consolidate multiple wings into a single CSV (one row per wing)."""
    rows = [_build_row(assignments, stem, measurements) for stem, assignments, measurements in results]
    df = pd.DataFrame(rows)
    df.to_csv(output_path, index=False)
    return output_path
