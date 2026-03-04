"""CLI entry point for inference."""

import argparse
import json
import sys
from pathlib import Path

_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from data.dataset import LANDMARK_ORDER
from inference.predict import LandmarkPredictor, predict_ensemble


def main() -> None:
    """Parse args and run prediction."""
    parser = argparse.ArgumentParser(description="Predict landmarks on wing images")
    parser.add_argument("images", type=Path, nargs="+", help="Image path(s)")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=False,
        help="Single checkpoint path",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        required=False,
        help="Directory with fold checkpoints for ensemble prediction",
    )
    parser.add_argument(
        "--device", type=str, default=None, help="Device: mps, cuda, cpu"
    )
    parser.add_argument(
        "--output", type=Path, default=None, help="Output JSON path"
    )
    args = parser.parse_args()

    if not args.checkpoint and not args.checkpoint_dir:
        parser.error("Provide --checkpoint or --checkpoint-dir")

    results = {}

    if args.checkpoint_dir:
        checkpoint_paths = sorted(args.checkpoint_dir.glob("best_fold*.pt"))
        if not checkpoint_paths:
            parser.error(f"No checkpoints found in {args.checkpoint_dir}")

        import cv2
        for img_path in args.images:
            image = cv2.imread(str(img_path))
            if image is None:
                print(f"Warning: could not load {img_path}, skipping")
                continue
            result = predict_ensemble(image, checkpoint_paths, args.device)
            results[str(img_path)] = {
                "landmarks": {k: list(v) for k, v in result["landmarks"].items()},
                "confidences": result["confidences"],
            }
    else:
        predictor = LandmarkPredictor(args.checkpoint, args.device)
        for img_path in args.images:
            result = predictor.predict_from_path(img_path)
            results[str(img_path)] = {
                "landmarks": {k: list(v) for k, v in result["landmarks"].items()},
                "confidences": result["confidences"],
            }

    # Output
    output_json = json.dumps(results, indent=2)
    if args.output:
        args.output.write_text(output_json)
        print(f"Results written to {args.output}")
    else:
        print(output_json)


if __name__ == "__main__":
    main()
