"""CSV export and summary table generation."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from WingVeinAnalyzer.models.vein_labeler import VeinAssignment
from WingVeinAnalyzer.models.vein_map import ALL_VEINS


def export_csv(
    assignments: list[VeinAssignment],
    output_path: Path,
    image_name: str = "",
) -> None:
    """Export vein measurements to CSV with per-vein status columns."""
    row: dict[str, object] = {"image": image_name}

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

    df = pd.DataFrame([row])
    df.to_csv(output_path, index=False)
