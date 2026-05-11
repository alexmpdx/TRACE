"""CLI entry point for training."""

import argparse
import sys
from datetime import datetime
from pathlib import Path

from landmark_locator.training.train import run_training

# Project root (LandmarkLocator/) for locating configs and data
_project_root = Path(__file__).resolve().parent.parent.parent


class _Tee:
    """File-like that mirrors writes to multiple streams (stdout + log file)."""

    def __init__(self, *streams):
        self._streams = streams

    def write(self, data):
        for s in self._streams:
            try:
                s.write(data)
            except Exception:
                pass
        return len(data)

    def flush(self):
        for s in self._streams:
            try:
                s.flush()
            except Exception:
                pass


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
    parser.add_argument(
        "--name",
        type=str,
        default=None,
        help=(
            "Model name. When set, all fold checkpoints go into <output-dir>/<name>/checkpoints/ "
            "so every run is self-contained. Log labels also use this name."
        ),
    )
    parser.add_argument(
        "--no-log-file",
        action="store_true",
        help="Disable saving stdout/stderr to <run_dir>/training.log (default: enabled).",
    )
    args = parser.parse_args()

    # Tee stdout/stderr into a log file under the named run folder so CLI training
    # leaves the same diagnostic trail that the GUI/sweep paths produce.
    run_dir = args.output_dir / args.name if args.name else args.output_dir
    log_file = None
    if not args.no_log_file:
        run_dir.mkdir(parents=True, exist_ok=True)
        log_path = run_dir / "training.log"
        log_file = open(log_path, "w", buffering=1)
        log_file.write(f"# started: {datetime.now().isoformat()}\n# argv: {' '.join(sys.argv)}\n\n")
        sys.stdout = _Tee(sys.__stdout__, log_file)
        sys.stderr = _Tee(sys.__stderr__, log_file)
        print(f"[train] saving log to {log_path}")

    try:
        run_training(args.config, args.output_dir, args.device, args.fold, args.name)
    finally:
        if log_file is not None:
            try:
                sys.stdout.flush()
                sys.stderr.flush()
            except Exception:
                pass
            sys.stdout = sys.__stdout__
            sys.stderr = sys.__stderr__
            log_file.write(f"\n# ended: {datetime.now().isoformat()}\n")
            log_file.close()


if __name__ == "__main__":
    main()
