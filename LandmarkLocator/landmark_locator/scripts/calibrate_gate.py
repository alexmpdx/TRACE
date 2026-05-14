"""Calibrate per-landmark confidence-gate thresholds from the peak/sharpness/second-peak
distributions observed on predictions that matched ground truth.

If per-fold checkpoints exist (`best_fold*.pt`) in --checkpoint-dir, each training image
is scored by the fold that held it out, producing a true held-out distribution.
If only a single checkpoint is provided, calibration is in-distribution and the script
warns accordingly.

Only predictions that land within --tolerance-px of the GT coordinate are counted —
wrong predictions would poison the "what does a correct prediction look like?" distribution.

Output:
  - Prints a summary table per landmark (N, min, 5th/50th/95th pct, max).
  - Writes a YAML to --output that slots straight into the `confidence:` block
    in configs/default.yaml (or a --confidence-override for landmark-predict).

Usage:
    python -m landmark_locator.scripts.calibrate_gate \
        --checkpoint-dir trained_models/checkpoints \
        --annotation-dir training_data_new2 \
        --image-dir training_data_pics \
        --output configs/gate_calibrated.yaml
"""

from __future__ import annotations

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import torch
import yaml

from landmark_locator.data.dataset import _normalize_name
from landmark_locator.inference.predict import (
    DEFAULT_GATE_CONFIG,
    LandmarkPredictor,
    _assemble_gate_result,
    _deep_merge,
    make_predictor,
)

_IMAGE_EXTS = {".tif", ".tiff", ".png", ".jpg", ".jpeg", ".bmp", ".psd", ".psb", ".czi", ".nd2", ".lif", ".lsm"}


def _permissive_override() -> dict:
    """Gate override that passes everything — we want raw metrics."""
    return {
        "peak": {"global": 0.0, "per_landmark": {}},
        "sharpness": {"global": 0.0, "per_landmark": {}},
        "second_peak_ratio": {"global": 1.0, "per_landmark": {}},
        "core_landmarks": [],
    }


def _load_gt(path: Path) -> dict[str, tuple[float, float]]:
    """Parse a GeoJSON annotation into internal-name → (x, y)."""
    data = json.loads(path.read_text())
    out: dict[str, tuple[float, float]] = {}
    for feat in data.get("features", []):
        geom = feat.get("geometry", {})
        raw_name = feat.get("properties", {}).get("classification", {}).get("name")
        if not raw_name or geom.get("type") not in {"Point", "MultiPoint"}:
            continue
        coords = geom["coordinates"]
        if geom["type"] == "MultiPoint":
            coords = coords[0]
        out[_normalize_name(raw_name)] = (float(coords[0]), float(coords[1]))
    return out


def _find_fold_checkpoints(ckpt_dir: Path) -> list[Path]:
    candidates = sorted(ckpt_dir.glob("best_fold*.pt"))
    # Skip variants like best_fold0_1.pt (retries / alternates) — only one per fold
    by_fold: dict[int, Path] = {}
    for p in candidates:
        stem = p.stem  # best_fold<N> possibly with _suffix
        rest = stem.replace("best_fold", "")
        try:
            fold = int(rest.split("_")[0])
        except ValueError:
            continue
        if fold not in by_fold:
            by_fold[fold] = p
    return [by_fold[f] for f in sorted(by_fold)]


def _cv_val_split(annotation_dir: Path, n_folds: int) -> list[list[str]]:
    """Return a list of validation-image stems per fold using the same stratification as training."""
    from landmark_locator.training.train import create_cv_splits

    geojson_files = sorted(annotation_dir.glob("*.geojson"))
    image_stems = [g.stem.replace(".tif", "").replace(".tiff", "") for g in geojson_files]
    splits = create_cv_splits(annotation_dir, n_folds)
    val_stems_per_fold = [[image_stems[i] for i in val_idx] for _, val_idx in splits]
    return val_stems_per_fold


def _match_image(image_dir: Path, stem: str) -> Path | None:
    for ext in (".tif", ".tiff", ".png", ".jpg", ".jpeg", ".bmp", ".psd", ".psb"):
        candidate = image_dir / f"{stem}{ext}"
        if candidate.exists():
            return candidate
    return None


def _collect_metrics(
    predictor: LandmarkPredictor,
    pairs: list[tuple[Path, Path | None]],
    tolerance_px: float,
) -> dict[str, dict[str, list[float]]]:
    """Collect per-landmark metrics from each (image, geojson|None) pair.

    When geojson is provided, only predictions within `tolerance_px` of GT count.
    When geojson is None (real-world calibration), every prediction counts — the
    assumption is that the model is generally correct on held-out images and the
    threshold percentile naturally excludes the worst predictions.
    """
    collected: dict[str, dict[str, list[float]]] = defaultdict(
        lambda: {"peak": [], "sharpness": [], "second_peak_ratio": []}
    )
    from landmark_locator.data.psd_loader import imread_any

    for img_path, gj_path in pairs:
        img = imread_any(img_path)
        if img is None:
            print(f"WARN could not load {img_path}")
            continue
        gt = _load_gt(gj_path) if gj_path is not None else None
        result = predictor.predict(img, include_unreliable=True)
        for name, (px, py) in result["landmarks"].items():
            if gt is not None:
                if name not in gt:
                    continue
                gx, gy = gt[name]
                if np.hypot(px - gx, py - gy) > tolerance_px:
                    continue
            collected[name]["peak"].append(result["confidences"][name])
            collected[name]["sharpness"].append(result["sharpness"][name])
            collected[name]["second_peak_ratio"].append(result["second_peak_ratio"][name])
    return collected


def _gather_images(dirs: list[Path]) -> list[Path]:
    """Recursively collect image files (one level deep) from the given directories."""
    out: list[Path] = []
    seen: set[str] = set()
    for d in dirs:
        d = d.resolve()
        if not d.exists():
            print(f"WARN: --calibration-image-dir does not exist: {d}")
            continue
        for p in sorted(d.iterdir()):
            if p.is_file() and p.suffix.lower() in _IMAGE_EXTS and not p.name.startswith("."):
                key = str(p)
                if key in seen:
                    continue
                seen.add(key)
                out.append(p)
    return out


def _training_stems(annotation_dir: Path) -> set[str]:
    """Return the set of training-image stems (whitespace-stripped, image-extension stripped)."""
    stems: set[str] = set()
    for gj in annotation_dir.glob("*.geojson"):
        stem = gj.stem  # foo.tif (from foo.tif.geojson) or foo
        for ext in (".tif", ".tiff", ".png", ".jpg", ".jpeg", ".bmp", ".psd", ".psb", ".czi"):
            if stem.lower().endswith(ext):
                stem = stem[: -len(ext)]
                break
        stems.add(stem.strip())
    return stems


def _percentile(values: list[float], pct: float) -> float:
    if not values:
        return float("nan")
    return float(np.percentile(values, pct))


def _build_yaml(
    collected: dict[str, dict[str, list[float]]],
    *,
    peak_pct: float,
    sharp_pct: float,
    spr_pct: float,
    core_landmarks: list[str],
    suppression_radius: int,
) -> dict:
    peak_pl: dict[str, float] = {}
    sharp_pl: dict[str, float] = {}
    spr_pl: dict[str, float] = {}
    for name, metrics in collected.items():
        if not metrics["peak"]:
            continue
        peak_pl[name] = round(_percentile(metrics["peak"], peak_pct), 3)
        sharp_pl[name] = round(_percentile(metrics["sharpness"], sharp_pct), 2)
        spr_pl[name] = round(_percentile(metrics["second_peak_ratio"], spr_pct), 2)
    # Globals are the floor/ceiling across all landmarks
    global_peak = round(min(peak_pl.values()), 3) if peak_pl else 0.1
    global_sharp = round(min(sharp_pl.values()), 2) if sharp_pl else 1.15
    global_spr = round(max(spr_pl.values()), 2) if spr_pl else 0.8
    return {
        "confidence": {
            "peak": {"global": global_peak, "per_landmark": peak_pl},
            "sharpness": {"global": global_sharp, "per_landmark": sharp_pl},
            "second_peak_ratio": {"global": global_spr, "per_landmark": spr_pl},
            "second_peak_suppression_radius_px": suppression_radius,
            "core_landmarks": sorted(core_landmarks),
        }
    }


def _print_summary(
    collected: dict[str, dict[str, list[float]]],
    peak_pct: float,
    sharp_pct: float,
    spr_pct: float,
) -> None:
    print(
        f"{'landmark':20s} {'N':>3s} "
        f"{'peak_p' + str(int(peak_pct)):>9s} {'peak_med':>9s} "
        f"{'sharp_p' + str(int(sharp_pct)):>10s} {'sharp_med':>10s} "
        f"{'spr_p' + str(int(spr_pct)):>9s} {'spr_med':>8s}"
    )
    for name in sorted(collected):
        m = collected[name]
        n = len(m["peak"])
        if n == 0:
            continue
        peak_thr = _percentile(m["peak"], peak_pct)
        sharp_thr = _percentile(m["sharpness"], sharp_pct)
        spr_thr = _percentile(m["second_peak_ratio"], spr_pct)
        peak_med = statistics.median(m["peak"])
        sharp_med = statistics.median(m["sharpness"])
        spr_med = statistics.median(m["second_peak_ratio"])
        print(
            f"{name:20s} {n:>3d} "
            f"{peak_thr:>9.3f} {peak_med:>9.3f} "
            f"{sharp_thr:>10.2f} {sharp_med:>10.2f} "
            f"{spr_thr:>9.2f} {spr_med:>8.2f}"
        )


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    group = p.add_mutually_exclusive_group(required=True)
    group.add_argument("--checkpoint", type=Path, help="Single checkpoint (in-distribution calibration).")
    group.add_argument("--checkpoint-dir", type=Path, help="Directory with best_fold*.pt (held-out calibration).")
    p.add_argument(
        "--annotation-dir",
        type=Path,
        default=None,
        help="Folder of GeoJSON annotations (required for GT-filtered modes).",
    )
    p.add_argument(
        "--image-dir",
        type=Path,
        default=None,
        help="Folder of training images (required for GT-filtered modes).",
    )
    p.add_argument(
        "--calibration-image-dir",
        type=Path,
        action="append",
        default=None,
        help=(
            "Folder of unannotated real-world images for calibration. Repeatable. "
            "Switches to no-GT mode: every prediction counts. Bad predictions still tend "
            "to cluster at low peak/sharpness so the percentile threshold naturally excludes them."
        ),
    )
    p.add_argument(
        "--skip-stems-from",
        type=Path,
        default=None,
        help=(
            "When set with --calibration-image-dir, skip any calibration image whose stem "
            "matches a .geojson in this directory (typically your training annotations)."
        ),
    )
    p.add_argument("--output", type=Path, required=True, help="Where to write the calibrated YAML.")
    p.add_argument(
        "--tolerance-px",
        type=float,
        default=100.0,
        help="Max pixel distance between prediction and GT to count as 'correct' (default 100).",
    )
    p.add_argument("--peak-pct", type=float, default=5.0, help="Percentile for peak threshold (lower = looser).")
    p.add_argument("--sharpness-pct", type=float, default=5.0, help="Percentile for sharpness threshold.")
    p.add_argument("--sp-ratio-pct", type=float, default=95.0, help="Percentile for second-peak-ratio ceiling.")
    p.add_argument(
        "--strict-output",
        type=Path,
        default=None,
        help=(
            "If set, also write a second YAML with stricter per-landmark thresholds derived from "
            "--strict-peak-pct / --strict-sharpness-pct / --strict-sp-ratio-pct."
        ),
    )
    p.add_argument("--strict-peak-pct", type=float, default=25.0, help="Strict peak threshold percentile (default 25).")
    p.add_argument("--strict-sharpness-pct", type=float, default=25.0, help="Strict sharpness percentile (default 25).")
    p.add_argument(
        "--strict-sp-ratio-pct", type=float, default=75.0, help="Strict second-peak-ratio percentile (default 75)."
    )
    p.add_argument(
        "--core-landmark",
        action="append",
        default=[
            "subcostal_break",
            "alula_notch",
            "l1_rs",
            "l2_l3",
            "l4_l5",
            "dtip",
        ],
        help="Landmark name(s) that abort on failure (repeatable). Default: core hinge+tip set.",
    )
    p.add_argument(
        "--suppression-radius-px",
        type=int,
        default=30,
        help="Second-peak suppression radius at model resolution (default 30).",
    )
    p.add_argument("--device", type=str, default=None)
    args = p.parse_args()

    # Validate input combinations
    if args.calibration_image_dir:
        # Real-world mode: no GT needed
        pass
    elif args.annotation_dir is None or args.image_dir is None:
        raise SystemExit("Must provide either --calibration-image-dir, or both --annotation-dir and --image-dir.")

    override = _permissive_override()

    collected: dict[str, dict[str, list[float]]] = defaultdict(
        lambda: {"peak": [], "sharpness": [], "second_peak_ratio": []}
    )

    if args.calibration_image_dir:
        # ---- Real-world calibration mode (no GT) ----
        skip_stems: set[str] = set()
        if args.skip_stems_from:
            skip_dir = args.skip_stems_from.resolve()
            skip_stems = _training_stems(skip_dir)
            print(f"Skipping {len(skip_stems)} training stems from {skip_dir}")

        all_images = _gather_images([Path(d).resolve() for d in args.calibration_image_dir])
        pairs: list[tuple[Path, Path | None]] = []
        skipped = 0
        for img_path in all_images:
            if img_path.stem.strip() in skip_stems:
                skipped += 1
                continue
            pairs.append((img_path, None))
        print(f"Real-world calibration set: {len(pairs)} images ({skipped} skipped as training set)")
        if not pairs:
            raise SystemExit("No calibration images remaining after filtering.")

        if args.checkpoint_dir:
            predictor = make_predictor(args.checkpoint_dir, device=args.device, confidence_override=override)
            print(f"Using ensemble from {args.checkpoint_dir}")
        else:
            predictor = LandmarkPredictor(args.checkpoint, device=args.device, confidence_override=override)
            print(f"Using single checkpoint {args.checkpoint}")

        collected = _collect_metrics(predictor, pairs, args.tolerance_px)
    elif args.checkpoint_dir:
        annotation_dir = args.annotation_dir.resolve()
        image_dir = args.image_dir.resolve()
        fold_ckpts = _find_fold_checkpoints(args.checkpoint_dir)
        if not fold_ckpts:
            raise SystemExit(f"No best_fold*.pt in {args.checkpoint_dir}")
        val_stems_per_fold = _cv_val_split(annotation_dir, n_folds=len(fold_ckpts))
        for fold_idx, ckpt in enumerate(fold_ckpts):
            val_stems = val_stems_per_fold[fold_idx]
            print(f"Fold {fold_idx}: {ckpt.name} on {len(val_stems)} val images")
            predictor = LandmarkPredictor(ckpt, device=args.device, confidence_override=override)
            pairs: list[tuple[Path, Path]] = []
            for stem in val_stems:
                gj = annotation_dir / f"{stem}.geojson"
                if not gj.exists():
                    # training annotations may append .tif before .geojson
                    alt = list(annotation_dir.glob(f"{stem}.tif*.geojson"))
                    gj = alt[0] if alt else gj
                if not gj.exists():
                    continue
                img = _match_image(image_dir, stem)
                if img is None:
                    # try stem without the .tif stripping
                    img = _match_image(image_dir, stem + ".tif") or _match_image(image_dir, stem)
                if img is None:
                    continue
                pairs.append((img, gj))
            fold_metrics = _collect_metrics(predictor, pairs, args.tolerance_px)
            for name, m in fold_metrics.items():
                collected[name]["peak"].extend(m["peak"])
                collected[name]["sharpness"].extend(m["sharpness"])
                collected[name]["second_peak_ratio"].extend(m["second_peak_ratio"])
    else:
        annotation_dir = args.annotation_dir.resolve()
        image_dir = args.image_dir.resolve()
        print(f"WARN Single-checkpoint calibration is in-distribution, not held-out: {args.checkpoint}")
        predictor = LandmarkPredictor(args.checkpoint, device=args.device, confidence_override=override)
        # Build pairs from every annotation with a matching image
        pairs_all: list[tuple[Path, Path]] = []
        for gj in sorted(annotation_dir.glob("*.geojson")):
            stem = gj.stem
            for suffix in (".tif", ".tiff"):
                if stem.endswith(suffix):
                    stem = stem[: -len(suffix)]
                    break
            img = _match_image(image_dir, stem)
            if img is not None:
                pairs_all.append((img, gj))
        print(f"Scoring {len(pairs_all)} images")
        collected = _collect_metrics(predictor, pairs_all, args.tolerance_px)

    if not collected:
        raise SystemExit("No metrics collected — check annotation/image pairing or tolerance.")

    print()
    _print_summary(collected, args.peak_pct, args.sharpness_pct, args.sp_ratio_pct)

    doc = _build_yaml(
        collected,
        peak_pct=args.peak_pct,
        sharp_pct=args.sharpness_pct,
        spr_pct=args.sp_ratio_pct,
        core_landmarks=args.core_landmark,
        suppression_radius=args.suppression_radius_px,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(yaml.safe_dump(doc, sort_keys=False))
    print()
    print(
        f"Wrote {args.output} (PERMISSIVE: peak_p{int(args.peak_pct)}, "
        f"sharp_p{int(args.sharpness_pct)}, spr_p{int(args.sp_ratio_pct)})"
    )

    if args.strict_output:
        strict_doc = _build_yaml(
            collected,
            peak_pct=args.strict_peak_pct,
            sharp_pct=args.strict_sharpness_pct,
            spr_pct=args.strict_sp_ratio_pct,
            core_landmarks=args.core_landmark,
            suppression_radius=args.suppression_radius_px,
        )
        args.strict_output.parent.mkdir(parents=True, exist_ok=True)
        args.strict_output.write_text(yaml.safe_dump(strict_doc, sort_keys=False))
        print(
            f"Wrote {args.strict_output} (STRICT:    peak_p{int(args.strict_peak_pct)}, "
            f"sharp_p{int(args.strict_sharpness_pct)}, spr_p{int(args.strict_sp_ratio_pct)})"
        )

        # Per-landmark strict vs permissive comparison table
        print()
        print("PERMISSIVE vs STRICT thresholds (per-landmark):")
        print(
            f"  {'landmark':<20} {'perm peak':>10} {'strict peak':>12}  "
            f"{'perm sharp':>11} {'strict sharp':>13}  "
            f"{'perm spr':>9} {'strict spr':>11}"
        )
        perm_peak = doc["confidence"]["peak"]["per_landmark"]
        strict_peak = strict_doc["confidence"]["peak"]["per_landmark"]
        perm_sharp = doc["confidence"]["sharpness"]["per_landmark"]
        strict_sharp = strict_doc["confidence"]["sharpness"]["per_landmark"]
        perm_spr = doc["confidence"]["second_peak_ratio"]["per_landmark"]
        strict_spr = strict_doc["confidence"]["second_peak_ratio"]["per_landmark"]
        for name in sorted(perm_peak):
            print(
                f"  {name:<20} {perm_peak[name]:>10.3f} {strict_peak.get(name, float('nan')):>12.3f}  "
                f"{perm_sharp[name]:>11.2f} {strict_sharp.get(name, float('nan')):>13.2f}  "
                f"{perm_spr[name]:>9.2f} {strict_spr.get(name, float('nan')):>11.2f}"
            )

    print()
    print("Apply with:")
    print(f"  cp {args.output} configs/default.yaml   # or merge into existing config")
    print("Then re-run the patch script to inject into a checkpoint:")
    print(f"  python -m landmark_locator.scripts.patch_checkpoint_gate --checkpoint <ckpt> --config {args.output}")


if __name__ == "__main__":
    main()
