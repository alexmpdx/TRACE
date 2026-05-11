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

import cv2
from identify_features.config import PIPELINE_PRESETS, PipelineConfig, apply_preset
from identify_features.controllers.pipeline import identify_wing
from identify_features.utils.psd_loader import imread_any
from identify_features.views.csv_export import export_csv, export_csv_batch
from identify_features.views.geojson_export import export_geojson
from identify_features.views.overlay import (
    render_ap_overlay_to_file,
    render_cv_ratio_overlay_to_file,
    render_overlay_to_file,
)


def _find_batch_specimens(
    det_dir: Path, lm_dir: Path, img_dir: Path | None
) -> list[tuple[str, Path, Path, Path | None]]:
    """Find matched specimen files across directories."""
    specimens = []
    for det in sorted(det_dir.glob("*_detections.geojson")):
        stem = det.name.replace("_detections.geojson", "")
        # Try canonical, space-padded, and wing-isolation suffix variants.
        for cand in (
            lm_dir / f"{stem}_landmarks.geojson",
            lm_dir / f"{stem} _landmarks.geojson",
            lm_dir / f"{stem}_main_wing_landmarks.geojson",
        ):
            if cand.exists():
                lm_path = cand
                break
        else:
            continue
        img_path = None
        if img_dir is not None:
            for ext in (".tif", ".tiff", ".bmp", ".png", ".jpg", ".jpeg", ".psd", ".psb"):
                p = img_dir / f"{stem}{ext}"
                if p.exists():
                    img_path = p
                    break
        specimens.append((stem, det, lm_path, img_path))
    return specimens


def _process_one(args_tuple):
    """Process a single specimen (for parallel execution)."""
    (
        stem,
        det_path,
        lm_path,
        img_path,
        output_dir,
        um_per_px,
        verbose,
        overlay,
        all_overlays,
        preset,
        show_vein_tissue,
        synthesize_missing_crossveins,
        skip_intervein_regions,
    ) = args_tuple
    try:
        config = PipelineConfig()
        if preset:
            config = apply_preset(config, preset)
        if um_per_px is not None:
            config.um_per_px = um_per_px
        config.synthesize_missing_crossveins = synthesize_missing_crossveins
        config.skip_intervein_regions = skip_intervein_regions
        if verbose:
            logging.basicConfig(level=logging.INFO)

        result = identify_wing(det_path, lm_path, img_path, config=config, specimen_id=stem)

        export_geojson(
            result.veins, result.intervein_regions, output_dir / f"{stem}_output.geojson", um_per_px=config.um_per_px
        )

        if overlay and img_path is not None:
            base_img = imread_any(img_path)
            if base_img is not None:
                render_overlay_to_file(
                    base_img,
                    result.veins,
                    result.intervein_regions,
                    output_dir / f"{stem}_overlay.png",
                    show_vein_tissue=show_vein_tissue,
                )
                if all_overlays:
                    render_ap_overlay_to_file(base_img, result, output_dir / f"{stem}_ap_overlay.png")
                    render_cv_ratio_overlay_to_file(
                        base_img, result, output_dir / f"{stem}_cv_ratio_overlay.png", um_per_px=config.um_per_px
                    )

        n_veins = sum(1 for v in result.veins if v.centerline is not None)
        if skip_intervein_regions:
            return stem, True, f"{stem}: {n_veins} veins (intervein regions skipped)", result
        n_regions = len(result.intervein_regions)
        return stem, True, f"{stem}: {n_veins} veins, {n_regions}/7 regions", result
    except Exception as e:
        return stem, False, f"{stem}: ERROR — {e}", None


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
        "--overlay",
        action="store_true",
        help="Generate the vein/intervein overlay PNG (requires image input)",
    )
    parser.add_argument(
        "--all-overlays",
        action="store_true",
        help="Also render the AP-compartment and CV-ratio overlays (slow; off by default)",
    )
    parser.add_argument(
        "--show-vein-tissue",
        action="store_true",
        help="In the overlay PNG, fill buffered vein tissue polygons (default: skeleton lines only)",
    )
    parser.add_argument(
        "--preset",
        type=str,
        default=None,
        choices=sorted(PIPELINE_PRESETS.keys()),
        help="Pipeline preset (default: built-in defaults)",
    )
    parser.add_argument(
        "--no-synthesize-crossveins",
        dest="synthesize_missing_crossveins",
        action="store_false",
        default=True,
        help="Disable Phase 5b landmark-based ACV/PCV synthesis; preserves merged intervein regions when no crossvein is detected in the skeleton",
    )
    parser.add_argument(
        "--skip-intervein-regions",
        action="store_true",
        help="Skip §6.1 intervein polygon splitting and §6.2 intervein region naming. Use when only vein outputs are needed; saves the watershed/distance-transform work. §6.3 vein tissue assignment still runs so overlays render filled vein polygons.",
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
    if args.preset:
        config = apply_preset(config, args.preset)
    if args.um_per_px is not None:
        config.um_per_px = args.um_per_px
    config.synthesize_missing_crossveins = args.synthesize_missing_crossveins
    config.skip_intervein_regions = args.skip_intervein_regions

    result = identify_wing(
        args.detection,
        args.landmarks,
        args.image,
        config=config,
    )

    out_path = args.output_dir / f"{result.specimen_id}_output.geojson"
    export_geojson(result.veins, result.intervein_regions, out_path, um_per_px=config.um_per_px)

    csv_path = args.output_dir / f"{result.specimen_id}_measurements.csv"
    export_csv(
        result.veins,
        result.intervein_regions,
        csv_path,
        um_per_px=config.um_per_px,
        specimen_id=result.specimen_id,
        wing_result=result,
    )
    print(f"CSV: {csv_path}")

    if args.overlay and args.image is not None:
        base_img = imread_any(args.image)
        if base_img is not None:
            overlay_path = args.output_dir / f"{result.specimen_id}_overlay.png"
            render_overlay_to_file(
                base_img,
                result.veins,
                result.intervein_regions,
                overlay_path,
                show_vein_tissue=args.show_vein_tissue,
            )
            print(f"Overlay: {overlay_path}")

            if args.all_overlays:
                ap_path = args.output_dir / f"{result.specimen_id}_ap_overlay.png"
                if render_ap_overlay_to_file(base_img, result, ap_path):
                    print(f"AP overlay: {ap_path}")

                cv_path = args.output_dir / f"{result.specimen_id}_cv_ratio_overlay.png"
                if render_cv_ratio_overlay_to_file(base_img, result, cv_path, um_per_px=config.um_per_px):
                    print(f"CV ratio overlay: {cv_path}")

    n_veins = sum(1 for v in result.veins if v.centerline is not None)
    if config.skip_intervein_regions:
        print(f"{result.specimen_id}: {n_veins} veins (intervein regions skipped)")
    else:
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
        (
            stem,
            det,
            lm,
            img,
            args.output_dir,
            args.um_per_px,
            args.verbose,
            args.overlay,
            args.all_overlays,
            args.preset,
            args.show_vein_tissue,
            args.synthesize_missing_crossveins,
            args.skip_intervein_regions,
        )
        for stem, det, lm, img in specimens
    ]

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(_process_one, item) for item in work_items]
        results = [f.result() for f in futures]

    elapsed = time.time() - t0
    print(f"Completed in {elapsed:.0f}s ({elapsed / 60:.1f} min)\n")

    successes = 0
    csv_rows = []
    for stem, ok, line, wing_result in results:
        print(line)
        if ok:
            successes += 1
            csv_rows.append((stem, wing_result))

    # Write combined measurements CSV
    um = args.um_per_px if args.um_per_px is not None else PipelineConfig().um_per_px
    csv_path = args.output_dir / "measurements.csv"
    export_csv_batch(csv_rows, csv_path, um_per_px=um)
    print(f"\nCSV: {csv_path}")

    print(f"{successes}/{len(results)} succeeded")
    print(f"Output directory: {args.output_dir}")


if __name__ == "__main__":
    main()
