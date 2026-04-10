"""CLI entry point for identify-features.

Usage:
    identify-features <detection_geojson> <landmarks_geojson> [image]
    identify-features --batch <det_dir> <lm_dir> [image_dir]
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from identify_features.config import PipelineConfig
from identify_features.controllers.pipeline import identify_wing
from identify_features.views.geojson_export import export_geojson


def _find_batch_specimens(
    det_dir: Path, lm_dir: Path, img_dir: Path | None
) -> list[tuple[str, Path, Path, Path | None]]:
    """Find matched specimen files across directories."""
    specimens = []
    for det in sorted(det_dir.glob("*_detections.geojson")):
        stem = det.name.replace("_detections.geojson", "")
        lm_path = lm_dir / f"{stem}_landmarks.geojson"
        if not lm_path.exists():
            lm_path = lm_dir / f"{stem} _landmarks.geojson"
        if not lm_path.exists():
            continue
        img_path = None
        if img_dir is not None:
            for ext in (".tif", ".bmp", ".png", ".jpg"):
                p = img_dir / f"{stem}{ext}"
                if p.exists():
                    img_path = p
                    break
        specimens.append((stem, det, lm_path, img_path))
    return specimens


def _process_one(args_tuple):
    """Process a single specimen (for parallel execution)."""
    stem, det_path, lm_path, img_path, output_dir, um_per_px, verbose = args_tuple
    try:
        config = PipelineConfig()
        if um_per_px is not None:
            config.um_per_px = um_per_px
        if verbose:
            logging.basicConfig(level=logging.INFO)

        result = identify_wing(det_path, lm_path, img_path, config=config, specimen_id=stem)

        out_path = output_dir / f"{stem}_output.geojson"
        export_geojson(result.veins, result.intervein_regions, out_path, um_per_px=config.um_per_px)

        n_veins = sum(1 for v in result.veins if v.centerline is not None)
        n_regions = len(result.intervein_regions)
        return stem, True, f"{stem}: {n_veins} veins, {n_regions}/7 regions"
    except Exception as e:
        return stem, False, f"{stem}: ERROR — {e}"


def main():
    parser = argparse.ArgumentParser(
        prog="identify-features",
        description="Landmark-anchored Drosophila wing vein identification",
    )
    parser.add_argument(
        "--batch",
        action="store_true",
        help="Batch mode: positional args are directories",
    )
    parser.add_argument(
        "detection",
        type=Path,
        help="Detection GeoJSON file (or directory in --batch mode)",
    )
    parser.add_argument(
        "landmarks",
        type=Path,
        help="Landmarks GeoJSON file (or directory in --batch mode)",
    )
    parser.add_argument(
        "image",
        type=Path,
        nargs="?",
        default=None,
        help="Original image file (or directory in --batch mode)",
    )
    parser.add_argument(
        "--output-dir",
        "-o",
        type=Path,
        default=Path("output"),
        help="Output directory (default: ./output)",
    )
    parser.add_argument(
        "--um-per-px",
        type=float,
        default=None,
        help="Microns per pixel (default: 0.483)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Parallel workers for batch mode (default: half of CPU count)",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Verbose logging",
    )
    args = parser.parse_args()

    if args.verbose:
        logging.basicConfig(level=logging.INFO)
    else:
        logging.basicConfig(level=logging.WARNING)

    args.output_dir.mkdir(parents=True, exist_ok=True)

    if args.batch:
        _run_batch(args)
    else:
        _run_single(args)


def _run_single(args):
    """Process a single specimen."""
    config = PipelineConfig()
    if args.um_per_px is not None:
        config.um_per_px = args.um_per_px

    result = identify_wing(
        args.detection,
        args.landmarks,
        args.image,
        config=config,
    )

    out_path = args.output_dir / f"{result.specimen_id}_output.geojson"
    export_geojson(result.veins, result.intervein_regions, out_path, um_per_px=config.um_per_px)

    n_veins = sum(1 for v in result.veins if v.centerline is not None)
    print(f"{result.specimen_id}: {n_veins} veins, {len(result.intervein_regions)}/7 regions")
    print(f"Output: {out_path}")

    if result.warnings:
        for w in result.warnings:
            print(f"  Warning: {w}")


def _run_batch(args):
    """Process all specimens in batch directories."""
    specimens = _find_batch_specimens(args.detection, args.landmarks, args.image)
    if not specimens:
        print("No matched specimens found.", file=sys.stderr)
        sys.exit(1)

    print(f"Found {len(specimens)} specimens")
    max_workers = args.workers or min(os.cpu_count() // 2, 8)
    print(f"Processing with {max_workers} workers...")

    t0 = time.time()
    work_items = [
        (stem, det, lm, img, args.output_dir, args.um_per_px, args.verbose) for stem, det, lm, img in specimens
    ]

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(_process_one, item) for item in work_items]
        results = [f.result() for f in futures]

    elapsed = time.time() - t0
    print(f"Completed in {elapsed:.0f}s ({elapsed / 60:.1f} min)\n")

    successes = 0
    for stem, ok, line in results:
        print(line)
        if ok:
            successes += 1

    print(f"\n{successes}/{len(results)} succeeded")
    print(f"Output directory: {args.output_dir}")


if __name__ == "__main__":
    main()
