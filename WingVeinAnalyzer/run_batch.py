#!/usr/bin/env python3
"""CLI batch processing: run the vein analysis pipeline on a folder of wings."""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

from WingVeinAnalyzer.controllers.analysis_controller import run_pipeline
from WingVeinAnalyzer.gui.file_selector import discover_file_pairs
from WingVeinAnalyzer.views.results_view import consolidate_csv

logger = logging.getLogger("WingVeinAnalyzer.batch")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Batch-process Drosophila wing images through the vein analysis pipeline.",
    )
    parser.add_argument(
        "input_folder",
        type=Path,
        help="Folder containing .tif images and matching .geojson annotations",
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory (default: <input_folder>/output)",
    )
    parser.add_argument(
        "-s",
        "--scale",
        type=float,
        default=None,
        help="Microns per pixel (optional; omit for pixel-only measurements)",
    )
    parser.add_argument(
        "--smooth",
        type=float,
        default=3.0,
        help="Smoothing sigma for centerline extraction (default: 3.0)",
    )
    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )

    input_folder = args.input_folder.resolve()
    if not input_folder.is_dir():
        logger.error("Input folder does not exist: %s", input_folder)
        sys.exit(1)

    output_dir = (args.output_dir or input_folder / "output").resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    # Discover file pairs
    pairs = discover_file_pairs(input_folder)
    if not pairs:
        logger.error("No TIFF+GeoJSON pairs found in %s", input_folder)
        sys.exit(1)

    logger.info("Found %d file pair(s) in %s", len(pairs), input_folder)
    for p in pairs:
        logger.info("  %s", p.display_name)

    # Process each wing
    results = []
    failed = []
    for i, pair in enumerate(pairs, 1):
        stem = pair.display_name
        wing_output = output_dir / stem
        logger.info("[%d/%d] Processing %s ...", i, len(pairs), stem)
        t0 = time.time()
        try:
            result = run_pipeline(
                image_path=pair.image_path,
                geojson_path=pair.geojson_path,
                output_dir=wing_output,
                microns_per_pixel=args.scale,
                smooth_sigma=args.smooth,
            )
            elapsed = time.time() - t0
            logger.info("[%d/%d] %s done (%.1fs)", i, len(pairs), stem, elapsed)
            results.append((stem, result.assignments, result.measurements))
        except Exception:
            elapsed = time.time() - t0
            logger.exception("[%d/%d] %s FAILED (%.1fs)", i, len(pairs), stem, elapsed)
            failed.append(stem)

    # Consolidate CSV
    if results:
        csv_path = output_dir / "consolidated_measurements.csv"
        consolidate_csv(results, csv_path)
        logger.info("Consolidated CSV: %s (%d wings)", csv_path, len(results))

    # Summary
    logger.info(
        "Batch complete: %d succeeded, %d failed out of %d",
        len(results),
        len(failed),
        len(pairs),
    )
    if failed:
        logger.warning("Failed wings: %s", ", ".join(failed))


if __name__ == "__main__":
    main()
