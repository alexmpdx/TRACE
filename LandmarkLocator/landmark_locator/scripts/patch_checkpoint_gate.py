"""Inject the confidence-gate block from configs/default.yaml into an existing checkpoint.

Usage:
    python -m landmark_locator.scripts.patch_checkpoint_gate \
        --checkpoint trained_models/checkpoints/evenMOREpoints.pt \
        --config configs/default.yaml
"""

import argparse
import shutil
from pathlib import Path

import torch
import yaml


def patch_checkpoint(checkpoint_path: Path, config_path: Path, output_path: Path | None = None) -> Path:
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    if "confidence" not in cfg:
        raise SystemExit(f"{config_path} has no 'confidence:' block")

    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if "config" not in ckpt:
        raise SystemExit(f"{checkpoint_path} has no 'config' key to patch")

    ckpt["config"]["confidence"] = cfg["confidence"]

    if output_path is None:
        backup = checkpoint_path.with_suffix(checkpoint_path.suffix + ".pre-gate.bak")
        if not backup.exists():
            shutil.copy2(checkpoint_path, backup)
            print(f"Backed up original → {backup}")
        output_path = checkpoint_path

    torch.save(ckpt, output_path)
    print(f"Wrote patched checkpoint → {output_path}")
    print("Injected confidence block:")
    print(yaml.safe_dump(cfg["confidence"], sort_keys=False))
    return output_path


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--checkpoint", type=Path, help="Single checkpoint to patch in-place.")
    g.add_argument(
        "--checkpoint-dir",
        type=Path,
        help="Folder containing best_fold*.pt — patches every fold checkpoint in-place.",
    )
    p.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).resolve().parents[2] / "configs" / "default.yaml",
    )
    p.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Write to a new path instead of overwriting. Only valid with --checkpoint.",
    )
    args = p.parse_args()

    if args.checkpoint:
        patch_checkpoint(args.checkpoint, args.config, args.output)
        return

    # --checkpoint-dir: patch every best_fold*.pt in-place
    if args.output is not None:
        raise SystemExit("--output cannot be combined with --checkpoint-dir")
    fold_ckpts = sorted(args.checkpoint_dir.glob("best_fold*.pt"))
    if not fold_ckpts:
        raise SystemExit(f"No best_fold*.pt in {args.checkpoint_dir}")
    for ckpt in fold_ckpts:
        print(f"\n=== {ckpt.name} ===")
        patch_checkpoint(ckpt, args.config, None)


if __name__ == "__main__":
    main()
