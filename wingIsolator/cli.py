"""CLI for wingIsolator. Run via ``python -m wingIsolator.cli`` or run_cli.py."""

import argparse
import json
import sys
from pathlib import Path

from wingIsolator.pipeline import (
    WING_CLASS_NAMES,
    isolate_folder,
    isolate_main_wing,
)


def main():
    parser = argparse.ArgumentParser(description="Isolate the main (centered) wing from multi-wing detections.")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("-i", "--image", help="Single input image.")
    mode.add_argument(
        "--batch",
        nargs=2,
        metavar=("IMAGE_DIR", "GEOJSON_DIR"),
        help="Batch mode: directories of images and geojsons.",
    )
    parser.add_argument("-g", "--geojson", help="GeoJSON for single-image mode.")
    parser.add_argument("-o", "--output", required=True, help="Output directory.")
    parser.add_argument(
        "--class-name",
        action="append",
        default=None,
        help="Class name(s) to treat as wings (repeatable). Default: 'wing'.",
    )
    parser.add_argument("--bg-value", type=int, default=0, help="Background value for masked image (default: 0).")
    parser.add_argument(
        "--simplify", type=float, default=1.0, help="Polygon simplify tolerance in pixels (default: 1.0)."
    )
    parser.add_argument(
        "--smoothing-sigma",
        type=float,
        default=2.0,
        help="Gaussian sigma for distance-transform smoothing (default: 2.0).",
    )
    parser.add_argument(
        "--min-seed-distance",
        type=int,
        default=None,
        help="Minimum pixel distance between watershed seeds. " "Default: ~max distance-transform value.",
    )
    parser.add_argument(
        "--threshold-rel", type=float, default=0.2, help="Relative peak threshold for seed detection (default: 0.2)."
    )
    parser.add_argument(
        "--expand",
        type=float,
        default=0.05,
        dest="expand_fraction",
        help="Uniform outward dilation of the final polygon, as a fraction of "
        "sqrt(area) (default: 0.05 = 5%%; 0 disables).",
    )
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    class_names = tuple(args.class_name) if args.class_name else WING_CLASS_NAMES

    common_kwargs = dict(
        class_names=class_names,
        bg_value=args.bg_value,
        simplify_tolerance=args.simplify,
        smoothing_sigma=args.smoothing_sigma,
        min_seed_distance=args.min_seed_distance,
        threshold_rel=args.threshold_rel,
        expand_fraction=args.expand_fraction,
        debug=args.debug,
    )

    if args.image:
        if not args.geojson:
            parser.error("-g/--geojson is required with -i/--image")
        result = isolate_main_wing(args.image, args.geojson, args.output, **common_kwargs)
        print(json.dumps(result.to_dict(), indent=2))
        return 0 if result.status == "ok" else 1

    image_dir, geojson_dir = args.batch

    def _progress(i, n, img_path, res):
        sub_n = res.num_subwings if res.status == "ok" else "?"
        print(f"[{i}/{n}] {Path(img_path).name}: {res.status} (split into {sub_n} subwing(s))")

    results = isolate_folder(
        image_dir,
        geojson_dir,
        args.output,
        progress_callback=_progress,
        **common_kwargs,
    )
    if not results:
        print(f"No images found in {image_dir}", file=sys.stderr)
        return 1

    summary_path = Path(args.output) / "wing_isolator_summary.json"
    print(f"\nSummary written to {summary_path}")

    n_ok = sum(1 for r in results if r.status == "ok")
    return 0 if n_ok > 0 else 1


if __name__ == "__main__":
    sys.exit(main())
