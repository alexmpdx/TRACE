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
        help=(
            "Path to landmark model. Accepts either a single .pt checkpoint "
            "or a folder containing best_fold*.pt files (runs 5-fold ensemble)."
        ),
    )
    parser.add_argument(
        "--segmentation-model",
        required=True,
        type=Path,
        help="Path to segmentation model directory",
    )
    parser.add_argument(
        "--wing-isolation-model",
        type=Path,
        default=None,
        help=(
            "Optional Stage 0 model directory. When set, every image is masked through "
            "wingIsolator (using a wing/background segmentation from this model) before "
            "LandmarkLocator sees it. Omit to disable Stage 0."
        ),
    )
    parser.add_argument(
        "--wing-expand-fraction",
        type=float,
        default=0.05,
        help=(
            "Stage 0 mask buffer, as a fraction of sqrt(wing area). Default 0.05. "
            "Ignored when --wing-isolation-model is not set."
        ),
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
    parser.add_argument(
        "--calibrate-workers",
        type=Path,
        metavar="PATH",
        default=None,
        help=(
            "Calibrate Stage 2 memory against PATH (folder of wing images or a "
            "single image), print a recommended --workers value, then exit. Runs "
            "Stage 1 once on the chosen image to produce inputs for Stage 2 "
            "calibration. Requires --landmark-model and --segmentation-model."
        ),
    )
    parser.add_argument(
        "--include-unreliable-landmarks",
        action="store_true",
        help=(
            "Pass low-confidence landmarks to downstream stages (marked reliable=false in GeoJSON). "
            "Core-landmark failures still abort the image regardless of this flag."
        ),
    )
    parser.add_argument(
        "--landmark-batch-size",
        type=int,
        default=0,
        help=(
            "Batch size for the landmark forward pass. 0 (default) tracks --workers. "
            "1 disables batching, larger values trade memory for throughput."
        ),
    )
    parser.add_argument(
        "--gate-override-yaml",
        type=Path,
        default=None,
        help=(
            "YAML file with the same shape as the `confidence:` block in "
            "configs/default.yaml. Applied as a confidence-gate override at predictor "
            "construction time. Produced by the GUI's Landmarks tab Export button."
        ),
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
    if args.landmark_model.is_dir():
        fold_ckpts = sorted(args.landmark_model.glob("best_fold*.pt"))
        if not fold_ckpts:
            print(
                f"Error: --landmark-model points at a directory with no best_fold*.pt: {args.landmark_model}",
                file=sys.stderr,
            )
            sys.exit(1)
    if not args.segmentation_model.is_dir():
        print(f"Error: segmentation model dir not found: {args.segmentation_model}", file=sys.stderr)
        sys.exit(1)
    if args.wing_isolation_model is not None and not args.wing_isolation_model.is_dir():
        print(
            f"Error: --wing-isolation-model dir not found: {args.wing_isolation_model}",
            file=sys.stderr,
        )
        sys.exit(1)


def _progress(image_index, total, image_name, stage, detail):
    print(f"[{image_index + 1}/{total}] {image_name}: {stage} - {detail}")


def _run_calibration(args) -> int:
    """Handle --calibrate-workers PATH and exit. Returns process exit code."""
    if not args.landmark_model.exists():
        print(f"Error: landmark model not found: {args.landmark_model}", file=sys.stderr)
        return 1
    if not args.segmentation_model.is_dir():
        print(f"Error: segmentation model dir not found: {args.segmentation_model}", file=sys.stderr)
        return 1

    logging.basicConfig(level=logging.WARNING)

    device = None
    if args.device:
        import torch

        device = torch.device(args.device)

    from TRACE.calibrate_workers import calibrate_for_trace, format_report

    def _cb(stage, detail):
        print(f"[{stage}] {detail}")

    try:
        result = calibrate_for_trace(
            image_or_folder=args.calibrate_workers,
            landmark_checkpoint=args.landmark_model,
            segmentation_model_dir=args.segmentation_model,
            device=device,
            progress_callback=_cb,
        )
    except Exception as e:
        print(f"Calibration failed: {e}", file=sys.stderr)
        return 1

    print()
    print(format_report(result))
    return 0


def main(argv=None):
    args = parse_args(argv)

    if args.calibrate_workers is not None:
        sys.exit(_run_calibration(args))

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

    gate_override = None
    if args.gate_override_yaml is not None:
        if not args.gate_override_yaml.exists():
            print(f"Error: --gate-override-yaml not found: {args.gate_override_yaml}", file=sys.stderr)
            sys.exit(1)
        try:
            import yaml as _yaml

            gate_doc = _yaml.safe_load(args.gate_override_yaml.read_text()) or {}
        except Exception as e:
            print(f"Error: failed to parse {args.gate_override_yaml}: {e}", file=sys.stderr)
            sys.exit(1)
        gate_override = gate_doc.get("confidence", gate_doc)

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
        include_unreliable_landmarks=args.include_unreliable_landmarks,
        landmark_batch_size=(args.landmark_batch_size or None),
        gate_override=gate_override,
        wing_isolation_model_dir=args.wing_isolation_model,
        wing_expand_fraction=args.wing_expand_fraction,
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
