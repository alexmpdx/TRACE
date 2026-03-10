"""CLI entry point for training."""

import argparse
from pathlib import Path

from landmark_locator.training.train import run_training

# Project root (LandmarkLocator/) for locating configs and data
_project_root = Path(__file__).resolve().parent.parent.parent


def main() -> None:
    """Parse args and launch training."""
    parser = argparse.ArgumentParser(description="Train LandmarkLocator model")
    parser.add_argument(
        "--config",
        type=Path,
        default=_project_root / "configs" / "default.yaml",
        help="Path to config YAML",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=_project_root / "trained_models",
        help="Output directory for checkpoints and logs",
    )
    parser.add_argument(
        "--fold",
        type=int,
        default=None,
        help="Train specific fold only (0-indexed). Default: all folds.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device: mps, cuda, cpu. Default: auto-detect.",
    )
    args = parser.parse_args()

    run_training(args.config, args.output_dir, args.device, args.fold)


if __name__ == "__main__":
    main()
