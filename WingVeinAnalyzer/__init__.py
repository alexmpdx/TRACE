"""WingVeinAnalyzer — Drosophila wing vein analysis pipeline.

Usage::

    from WingVeinAnalyzer import analyze_wing, analyze_folder

    # Single wing
    result = analyze_wing("wing.tif", "wing.geojson")
    print(result.measurements.wing_length_px)

    # Batch folder
    results, csv_path = analyze_folder("path/to/wings/", scale=0.483)
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from WingVeinAnalyzer.controllers.analysis_controller import (
    PipelineResult,
    run_pipeline,
)
from WingVeinAnalyzer.controllers.measurement_controller import WingMeasurements
from WingVeinAnalyzer.gui.file_selector import FilePair, discover_file_pairs
from WingVeinAnalyzer.models.vein_labeler import VeinAssignment, VeinStatus
from WingVeinAnalyzer.views.results_view import consolidate_csv

__all__ = [
    "analyze_wing",
    "analyze_folder",
    "PipelineResult",
    "WingMeasurements",
    "VeinAssignment",
    "VeinStatus",
    "FilePair",
    "discover_file_pairs",
]


def analyze_wing(
    image_path: str | Path,
    geojson_path: str | Path,
    output_dir: str | Path | None = None,
    scale: float | None = None,
    smooth_sigma: float = 3.0,
) -> PipelineResult:
    """Analyze a single Drosophila wing image.

    Args:
        image_path: Path to the wing TIFF image.
        geojson_path: Path to the GeoJSON annotation file.
        output_dir: Where to write overlays, CSV, and diagnostics.
            Defaults to ``<image_dir>/output``.
        scale: Microns per pixel. If None, measurements are pixels only.
        smooth_sigma: Gaussian smoothing sigma for centerline extraction.

    Returns:
        PipelineResult with vein assignments, measurements, region names,
        and paths to output files.
    """
    return run_pipeline(
        image_path=Path(image_path),
        geojson_path=Path(geojson_path),
        output_dir=Path(output_dir) if output_dir else None,
        microns_per_pixel=scale,
        smooth_sigma=smooth_sigma,
    )


def analyze_folder(
    input_folder: str | Path,
    output_dir: str | Path | None = None,
    scale: float | None = None,
    smooth_sigma: float = 3.0,
) -> tuple[list[tuple[str, PipelineResult]], Optional[Path]]:
    """Batch-analyze all wings in a folder.

    Discovers TIFF+GeoJSON pairs by filename stem matching, processes
    each wing, and writes a consolidated CSV.

    Args:
        input_folder: Folder containing ``.tif`` and ``.geojson`` files.
        output_dir: Root output directory. Defaults to ``<input_folder>/output``.
            Each wing gets a subdirectory named by its stem.
        scale: Microns per pixel. If None, measurements are pixels only.
        smooth_sigma: Gaussian smoothing sigma for centerline extraction.

    Returns:
        Tuple of (results, csv_path) where results is a list of
        ``(stem, PipelineResult)`` for successful wings, and csv_path
        is the path to the consolidated CSV (or None if all failed).
    """
    import logging

    logger = logging.getLogger("WingVeinAnalyzer")

    folder = Path(input_folder).resolve()
    out = (Path(output_dir) if output_dir else folder / "output").resolve()
    out.mkdir(parents=True, exist_ok=True)

    pairs = discover_file_pairs(folder)
    if not pairs:
        logger.warning("No TIFF+GeoJSON pairs found in %s", folder)
        return [], None

    results: list[tuple[str, PipelineResult]] = []
    for pair in pairs:
        stem = pair.display_name
        try:
            result = run_pipeline(
                image_path=pair.image_path,
                geojson_path=pair.geojson_path,
                output_dir=out / stem,
                microns_per_pixel=scale,
                smooth_sigma=smooth_sigma,
            )
            results.append((stem, result))
        except Exception:
            logger.exception("Failed to process %s", stem)

    csv_path = None
    if results:
        csv_path = out / "consolidated_measurements.csv"
        consolidate_csv(
            [(stem, r.assignments, r.measurements) for stem, r in results],
            csv_path,
        )

    return results, csv_path
