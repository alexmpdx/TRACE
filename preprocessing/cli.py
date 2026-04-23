"""CLI interface for the preprocessing pipeline."""

import argparse
import sys
from pathlib import Path

from preprocessing.pipeline import process_folder


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Wing Preprocessing Pipeline — LandmarkLocator + HingeChopper + modelTOjson"
    )
    parser.add_argument(
        "-i",
        "--input",
        required=True,
        type=Path,
        help="Input folder containing wing images",
    )
    parser.add_argument(
        "-o",
        "--output",
        required=True,
        type=Path,
        help="Output folder (flat — all outputs go here)",
    )
    parser.add_argument(
        "--landmark-model",
        type=Path,
        default=None,
        help="Path to landmark model checkpoint (.pt file)",
    )
    parser.add_argument(
        "--segmentation-model",
        type=Path,
        default=None,
        help="Path to segmentation model directory (contains metadata.json + weights)",
    )
    parser.add_argument(
        "--stages",
        nargs="+",
        default=["all"],
        choices=["all", "landmarks", "hinge", "segmentation"],
        help="Which stages to run (default: all)",
    )
    parser.add_argument(
        "--device",
        default=None,
        choices=["cpu", "cuda", "mps"],
        help="Torch device (default: auto-detect)",
    )
    parser.add_argument(
        "--keep-chopped",
        action="store_true",
        help="Keep intermediate chopped images in output folder",
    )
    parser.add_argument(
        "--include-unreliable-landmarks",
        action="store_true",
        help=(
            "Include landmarks that failed the confidence gate in the output GeoJSON "
            "(marked reliable=false). Core-landmark failures still abort the image."
        ),
    )
    return parser.parse_args(argv)


def _resolve_stages(stage_names: list[str]) -> tuple[bool, bool, bool]:
    """Convert stage name list to (landmarks, hinge, segmentation) booleans."""
    if "all" in stage_names:
        return (True, True, True)
    return (
        "landmarks" in stage_names,
        "hinge" in stage_names,
        "segmentation" in stage_names,
    )


def _validate(args, stages: tuple[bool, bool, bool]):
    """Validate that required models are provided for selected stages."""
    do_lm, do_hinge, do_seg = stages

    if (do_lm or do_hinge) and args.landmark_model is None:
        print("Error: --landmark-model is required for landmarks/hinge stages.", file=sys.stderr)
        sys.exit(1)
    if do_seg and args.segmentation_model is None:
        print("Error: --segmentation-model is required for segmentation stage.", file=sys.stderr)
        sys.exit(1)
    if not args.input.is_dir():
        print(f"Error: input folder does not exist: {args.input}", file=sys.stderr)
        sys.exit(1)
    if args.landmark_model and not args.landmark_model.exists():
        print(f"Error: landmark model not found: {args.landmark_model}", file=sys.stderr)
        sys.exit(1)
    if args.segmentation_model and not args.segmentation_model.is_dir():
        print(f"Error: segmentation model dir not found: {args.segmentation_model}", file=sys.stderr)
        sys.exit(1)


def _progress(image_index, total, image_name, status):
    print(f"[{image_index + 1}/{total}] {image_name}: {status}")


def main(argv=None):
    args = parse_args(argv)
    stages = _resolve_stages(args.stages)
    _validate(args, stages)

    device = None
    if args.device:
        import torch

        device = torch.device(args.device)

    print(f"Input:  {args.input}")
    print(f"Output: {args.output}")
    stage_names = []
    if stages[0]:
        stage_names.append("landmarks")
    if stages[1]:
        stage_names.append("hinge")
    if stages[2]:
        stage_names.append("segmentation")
    print(f"Stages: {', '.join(stage_names)}")
    print()

    results = process_folder(
        input_dir=args.input,
        output_dir=args.output,
        landmark_checkpoint=args.landmark_model,
        segmentation_model_dir=args.segmentation_model,
        stages=stages,
        device=device,
        keep_chopped=args.keep_chopped,
        progress_callback=_progress,
        include_unreliable_landmarks=args.include_unreliable_landmarks,
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
                stage = f" [stage={r.error_stage}]" if r.error_stage else ""
                print(f"  {r.image_path.name}{stage}: {r.error.splitlines()[0]}", file=sys.stderr)
        sys.exit(1)
