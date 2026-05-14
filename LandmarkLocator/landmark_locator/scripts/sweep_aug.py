"""Sweep training over augmentation parameters and report the best combo.

Trains the model with each combination of rotation_limit × vertical_flip_p, captures
each run's stdout to a log file under the run folder, and prints a comparison table
of mean validation pixel error at the end.

Defaults to fold 0 only (~30 min/run on MPS, 6 runs ≈ 3h). Use --all-folds for the
full 5-fold CV per combo (~15h).

Usage:
    python -m landmark_locator.scripts.sweep_aug
    python -m landmark_locator.scripts.sweep_aug --all-folds
    python -m landmark_locator.scripts.sweep_aug --rotations 30 60 --vflips 0.0 0.25
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

import yaml

_project_root = Path(__file__).resolve().parent.parent.parent

DEFAULT_ROTATIONS = [30, 45, 60]
DEFAULT_VFLIPS = [0.0, 0.25]


def _combo_name(rotation: int, vflip: float) -> str:
    return f"aug_r{rotation:02d}_v{int(round(vflip * 100)):03d}"


def _write_combo_config(base_cfg: dict, rotation: int, vflip: float, out_path: Path) -> None:
    cfg = json.loads(json.dumps(base_cfg))
    cfg.setdefault("augmentation", {})
    cfg["augmentation"]["rotation_limit"] = rotation
    cfg["augmentation"]["vertical_flip_p"] = float(vflip)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(yaml.safe_dump(cfg, sort_keys=False))


def _expected_checkpoints(run_dir: Path, fold: Optional[int]) -> list[Path]:
    ckpt_dir = run_dir / "checkpoints"
    if fold is not None:
        return [ckpt_dir / f"best_fold{fold}.pt"]
    return [ckpt_dir / f"best_fold{i}.pt" for i in range(5)]


def _has_all_checkpoints(run_dir: Path, fold: Optional[int]) -> bool:
    return all(p.exists() for p in _expected_checkpoints(run_dir, fold))


def _run_one(
    config_path: Path,
    name: str,
    output_dir: Path,
    fold: Optional[int],
    device: Optional[str],
) -> tuple[int, Path]:
    """Run landmark-train as a subprocess; tee stdout to <run_dir>/training.log.

    Returns (returncode, log_path).
    """
    run_dir = output_dir / name
    run_dir.mkdir(parents=True, exist_ok=True)
    log_path = run_dir / "training.log"

    cmd = [
        sys.executable,
        "-u",
        "-m",
        "landmark_locator.scripts.train",
        "--config",
        str(config_path),
        "--output-dir",
        str(output_dir),
        "--name",
        name,
        # Sweep owns the log file; tell the CLI not to compete.
        "--no-log-file",
    ]
    if fold is not None:
        cmd.extend(["--fold", str(fold)])
    if device:
        cmd.extend(["--device", device])

    print(f"\n[sweep] === {name} ===", flush=True)
    print(f"[sweep] cmd: {' '.join(cmd)}", flush=True)
    print(f"[sweep] log: {log_path}", flush=True)

    started = datetime.now()
    with open(log_path, "w") as logf:
        logf.write(f"# sweep: {name}\n# started: {started.isoformat()}\n# cmd: {' '.join(cmd)}\n\n")
        logf.flush()
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            sys.stdout.write(line)
            sys.stdout.flush()
            logf.write(line)
            logf.flush()
        rc = proc.wait()
        ended = datetime.now()
        logf.write(f"\n# ended: {ended.isoformat()} (rc={rc}, duration={ended - started})\n")
    return rc, log_path


def _read_metrics(run_dir: Path, fold: Optional[int]) -> dict[int, dict]:
    """Return {fold_idx: {epoch, val_loss, mean_pixel_error}} for available checkpoints."""
    import torch

    folds = [fold] if fold is not None else list(range(5))
    out: dict[int, dict] = {}
    for f in folds:
        p = run_dir / "checkpoints" / f"best_fold{f}.pt"
        if not p.exists():
            continue
        try:
            ckpt = torch.load(p, map_location="cpu", weights_only=False)
            out[f] = {
                "epoch": ckpt.get("epoch"),
                "val_loss": ckpt.get("val_loss"),
                "mean_pixel_error": ckpt.get("mean_pixel_error"),
            }
        except Exception as e:
            out[f] = {"error": str(e)}
    return out


def _print_summary(results: list[dict], fold: Optional[int]) -> None:
    """Print a sortable comparison table of all combos."""
    print()
    print("=" * 92)
    print(f"SWEEP SUMMARY  ({'fold ' + str(fold) if fold is not None else 'all folds'})")
    print("=" * 92)

    rows = []
    for r in results:
        if not r["metrics"]:
            rows.append((r["name"], r["rotation"], r["vflip"], None, None, None, "no checkpoints"))
            continue
        # Aggregate across available folds.
        errs = [m["mean_pixel_error"] for m in r["metrics"].values() if m.get("mean_pixel_error") is not None]
        if not errs:
            rows.append((r["name"], r["rotation"], r["vflip"], None, None, None, "no metrics"))
            continue
        mean_err = sum(errs) / len(errs)
        max_err = max(errs)
        min_err = min(errs)
        rows.append((r["name"], r["rotation"], r["vflip"], mean_err, min_err, max_err, f"{len(errs)} fold(s)"))

    # Sort by mean error ascending; non-converging runs at bottom.
    rows.sort(key=lambda r: (r[3] is None, r[3] if r[3] is not None else float("inf")))

    print(f"  {'name':<20} {'rot':>4} {'vflp':>6} {'mean_err':>10} {'min_err':>9} {'max_err':>9}  notes")
    print("  " + "-" * 88)
    for name, rot, vflip, mean_err, min_err, max_err, note in rows:
        if mean_err is None:
            print(f"  {name:<20} {rot:>4} {vflip:>6.2f} {'   —':>10} {'   —':>9} {'   —':>9}  {note}")
        else:
            print(f"  {name:<20} {rot:>4} {vflip:>6.2f} {mean_err:>9.1f}px {min_err:>8.1f}px {max_err:>8.1f}px  {note}")
    print("=" * 92)
    if rows and rows[0][3] is not None:
        print(f"\nBest combo: {rows[0][0]} (mean pixel error = {rows[0][3]:.1f}px)")
        print(
            "Pick this combo, re-run with --all-folds to validate across all 5 folds, then promote it to "
            "configs/default.yaml as your default training config."
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--config",
        type=Path,
        default=_project_root / "configs" / "default.yaml",
        help="Base config to derive sweep configs from.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=_project_root / "trained_models",
        help="Where each combo's run folder is written.",
    )
    parser.add_argument(
        "--rotations",
        type=int,
        nargs="+",
        default=DEFAULT_ROTATIONS,
        help=f"rotation_limit values to sweep (default: {DEFAULT_ROTATIONS}).",
    )
    parser.add_argument(
        "--vflips",
        type=float,
        nargs="+",
        default=DEFAULT_VFLIPS,
        help=f"vertical_flip_p values to sweep (default: {DEFAULT_VFLIPS}).",
    )
    parser.add_argument(
        "--all-folds",
        action="store_true",
        help="Train all 5 folds per combo (default: fold 0 only for fast triage).",
    )
    parser.add_argument("--device", type=str, default=None, help="Device override for landmark-train.")
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Re-train combos that already have completed checkpoints (default: skip them).",
    )
    parser.add_argument(
        "--summary-only",
        action="store_true",
        help="Skip training; just read existing checkpoints and print the summary table.",
    )
    args = parser.parse_args()

    base_cfg = yaml.safe_load(args.config.read_text()) or {}
    fold = None if args.all_folds else 0

    sweep_dir = _project_root / "configs" / "sweep"
    combos = [(r, v) for r in args.rotations for v in args.vflips]

    print(f"[sweep] base config: {args.config}")
    print(f"[sweep] {len(combos)} combos × {'5 folds' if fold is None else '1 fold'}")
    for r, v in combos:
        print(f"          - {_combo_name(r, v)}")
    print()

    results: list[dict] = []
    for rotation, vflip in combos:
        name = _combo_name(rotation, vflip)
        run_dir = args.output_dir / name
        cfg_path = sweep_dir / f"{name}.yaml"

        if args.summary_only:
            metrics = _read_metrics(run_dir, fold)
            results.append({"name": name, "rotation": rotation, "vflip": vflip, "metrics": metrics})
            continue

        _write_combo_config(base_cfg, rotation, vflip, cfg_path)

        if not args.no_resume and _has_all_checkpoints(run_dir, fold):
            print(f"[sweep] skip {name} (all expected checkpoints already exist; --no-resume to re-train)")
            metrics = _read_metrics(run_dir, fold)
            results.append({"name": name, "rotation": rotation, "vflip": vflip, "metrics": metrics})
            continue

        rc, log_path = _run_one(cfg_path, name, args.output_dir, fold, args.device)
        if rc != 0:
            print(f"[sweep] {name} FAILED with rc={rc}; see {log_path}")
        metrics = _read_metrics(run_dir, fold)
        results.append({"name": name, "rotation": rotation, "vflip": vflip, "metrics": metrics})

    _print_summary(results, fold)

    # Also persist the summary as JSON for downstream tools.
    out_json = args.output_dir / f"sweep_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    out_json.write_text(json.dumps(results, indent=2, default=str))
    print(f"\nSummary written to {out_json}")


if __name__ == "__main__":
    main()
