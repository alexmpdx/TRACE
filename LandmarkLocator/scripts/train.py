"""CLI entry point for training."""

import argparse
import sys
from pathlib import Path

# Add project root to path
_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from training.train import run_training


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
        default=_project_root / "output",
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
