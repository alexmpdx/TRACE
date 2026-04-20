"""CLI interface for the TRACE combined pipeline."""

import argparse
import logging
import sys
from pathlib import Path

from TRACE.pipeline import DEFAULT_MAX_WORKERS, OUTPUT_TYPES, trace_folder


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="TRACE: Full wing analysis pipeline (preprocessing + vein analysis)",
    )
    parser.add_argument(
        "-i",
        "--input",
        type=Path,
        default=None,
        help="Input folder containing wing images (opens folder picker if omitted)",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="Output directory for results (default: <input>/output)",
    )
    parser.add_argument(
        "--landmark-model",
        required=True,
        type=Path,
        help="Path to landmark model checkpoint (.pt file)",
    )
    parser.add_argument(
        "--segmentation-model",
        required=True,
        type=Path,
        help="Path to segmentation model directory",
    )
    parser.add_argument(
        "-s",
        "--scale",
        type=float,
        default=None,
        help=(
            "Microns per pixel. Overrides config.um_per_px if --config is also given. "
            "Omit for pixel-only measurements."
        ),
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help=(
            "Path to a PipelineConfig JSON file (produced by the GUI 'Export...' "
            "button or by hand). Any fields not present fall back to defaults."
        ),
    )
    parser.add_argument(
        "--device",
        default=None,
        choices=["cpu", "cuda", "mps"],
        help="Torch device (default: auto-detect)",
    )
    parser.add_argument(
        "--keep-intermediates",
        action="store_true",
        help="Keep preprocessing intermediate files in output/intermediates/",
    )
    parser.add_argument(
        "--outputs",
        default=",".join(OUTPUT_TYPES.keys()),
        help=(
            "Comma-separated Stage 2 outputs to produce. "
            f"Valid keys: {','.join(OUTPUT_TYPES.keys())}. "
            "Pass an empty string to skip Stage 2."
        ),
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=DEFAULT_MAX_WORKERS,
        help=(
            f"Number of Stage 2 wings to analyze in parallel (default: {DEFAULT_MAX_WORKERS}). "
            "Stage 1 (GPU preprocessing) always runs sequentially. Pass 1 to disable parallelism."
        ),
    )
    parser.add_argument(
        "--show-vein-tissue",
        action="store_true",
        help="In the per-wing overlay PNG, fill buffered vein tissue polygons (default: skeleton lines only)",
    )
    return parser.parse_args(argv)


def _pick_folder(title):
    """Open a folder picker dialog and return the selected path."""
    import tkinter as tk
    from tkinter import filedialog

    root = tk.Tk()
    root.withdraw()
    root.attributes("-topmost", True)
    folder = filedialog.askdirectory(title=title)
    root.destroy()
    if not folder:
        print("No folder selected, exiting.")
        sys.exit(0)
    return Path(folder)


def _validate(args):
    """Validate CLI arguments, prompting for folders if not provided."""
    if args.input is None:
        args.input = _pick_folder("Select folder containing wing images")
    if args.output is None:
        args.output = _pick_folder("Select output folder")
    if not args.input.is_dir():
        print(f"Error: input folder does not exist: {args.input}", file=sys.stderr)
        sys.exit(1)
    if not args.landmark_model.exists():
        print(f"Error: landmark model not found: {args.landmark_model}", file=sys.stderr)
        sys.exit(1)
    if not args.segmentation_model.is_dir():
        print(f"Error: segmentation model dir not found: {args.segmentation_model}", file=sys.stderr)
        sys.exit(1)


def _progress(image_index, total, image_name, stage, detail):
    print(f"[{image_index + 1}/{total}] {image_name}: {stage} - {detail}")


def main(argv=None):
    args = parse_args(argv)
    _validate(args)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )

    device = None
    if args.device:
        import torch

        device = torch.device(args.device)

    print(f"Input:  {args.input}")
    print(f"Output: {args.output}")
    print()

    outputs = {o.strip() for o in args.outputs.split(",") if o.strip()}
    invalid = outputs - set(OUTPUT_TYPES.keys())
    if invalid:
        print(f"Error: unknown output keys: {sorted(invalid)}", file=sys.stderr)
        sys.exit(1)

    # Build PipelineConfig: load from file if given, then apply --scale override.
    from identify_features.config import PipelineConfig

    from TRACE.config_io import load_config

    if args.config is not None:
        if not args.config.exists():
            print(f"Error: config file not found: {args.config}", file=sys.stderr)
            sys.exit(1)
        try:
            config = load_config(args.config)
        except Exception as e:
            print(f"Error: failed to load config: {e}", file=sys.stderr)
            sys.exit(1)
    else:
        config = PipelineConfig()

    if args.scale is not None:
        config.um_per_px = args.scale if args.scale > 0 else None

    if args.workers < 1:
        print(f"Error: --workers must be >= 1, got {args.workers}", file=sys.stderr)
        sys.exit(1)

    results = trace_folder(
        input_dir=args.input,
        output_dir=args.output,
        landmark_checkpoint=args.landmark_model,
        segmentation_model_dir=args.segmentation_model,
        config=config,
        device=device,
        keep_intermediates=args.keep_intermediates,
        outputs=outputs,
        max_workers=args.workers,
        show_vein_tissue=args.show_vein_tissue,
        progress_callback=_progress,
    )

    # Summary
    succeeded = sum(1 for r in results if r.error is None)
    failed = sum(1 for r in results if r.error is not None)
    print()
    print(f"Done: {succeeded} succeeded, {failed} failed out of {len(results)} images.")

    if failed:
        print("\nFailed images:", file=sys.stderr)
        for r in results:
            if r.error:
                print(f"  {r.image_path.name} ({r.error_stage}): {r.error.splitlines()[0]}", file=sys.stderr)
        sys.exit(1)
