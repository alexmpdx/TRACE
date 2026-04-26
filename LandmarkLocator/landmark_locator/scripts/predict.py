"""CLI entry point for inference."""

import argparse
import json
import sys
from pathlib import Path

import yaml

from landmark_locator.inference.predict import (
    DEFAULT_GATE_CONFIG,
    LandmarkPredictor,
    LowConfidenceLandmarkError,
    _deep_merge,
    predict_ensemble,
)

_IMAGE_EXTS = {".tif", ".tiff", ".bmp", ".png", ".jpg", ".jpeg", ".psd", ".psb"}


def _discover_images(folder: Path) -> list[Path]:
    """List supported images in a folder, skipping hidden / resource-fork files."""
    out = []
    for f in sorted(folder.iterdir()):
        if f.name.startswith(".") or f.name.startswith("._"):
            continue
        if f.suffix.lower() in _IMAGE_EXTS:
            out.append(f)
    return out


def _result_to_geojson(result: dict, internal_to_geojson: dict[str, str]) -> dict:
    """Build a GeoJSON FeatureCollection from a predictor result.

    Matches the schema produced by preprocessing.pipeline.landmarks_to_geojson:
    each Point feature carries its classification name plus gate metadata.
    """
    features = []
    landmarks = result.get("landmarks", {}) or {}
    for internal_name, (x, y) in landmarks.items():
        gj_name = internal_to_geojson.get(internal_name, internal_name)
        props = {
            "classification": {"name": gj_name},
            "reliable": bool(result.get("reliable", {}).get(internal_name, True)),
            "gate_reason": result.get("gate_reason", {}).get(internal_name, ""),
            "confidence": float(result.get("confidences", {}).get(internal_name, 0.0)),
            "sharpness": float(result.get("sharpness", {}).get(internal_name, 0.0)),
            "second_peak_ratio": float(result.get("second_peak_ratio", {}).get(internal_name, 0.0)),
        }
        features.append(
            {
                "type": "Feature",
                "geometry": {"type": "Point", "coordinates": [float(x), float(y)]},
                "properties": props,
            }
        )
    return {"type": "FeatureCollection", "features": features}


def _parse_tier(tier_args: list[str] | None) -> dict:
    """--tier name=strict / name=permissive → per-landmark threshold entries.

    "strict" copies the crossvein defaults from DEFAULT_GATE_CONFIG; "permissive"
    clears any per-landmark entry so the global threshold applies.
    """
    override: dict = {
        "peak": {"per_landmark": {}},
        "sharpness": {"per_landmark": {}},
        "second_peak_ratio": {"per_landmark": {}},
    }
    strict_peak = 0.20
    strict_sharp = 1.25
    strict_spr = 0.65
    for spec in tier_args or []:
        if "=" not in spec:
            raise SystemExit(f"--tier expects NAME=strict|permissive, got {spec!r}")
        name, tier = spec.split("=", 1)
        name = name.strip()
        tier = tier.strip().lower()
        if tier == "strict":
            override["peak"]["per_landmark"][name] = strict_peak
            override["sharpness"]["per_landmark"][name] = strict_sharp
            override["second_peak_ratio"]["per_landmark"][name] = strict_spr
        elif tier == "permissive":
            override["peak"]["per_landmark"][name] = None
            override["sharpness"]["per_landmark"][name] = None
            override["second_peak_ratio"]["per_landmark"][name] = None
        else:
            raise SystemExit(f"--tier: unknown tier {tier!r}; use 'strict' or 'permissive'")
    return override


def _apply_permissive_removals(base: dict, override: dict) -> dict:
    """Handle the None sentinels produced by --tier name=permissive by deleting entries."""
    merged = _deep_merge(base, override)
    for section in ("peak", "sharpness", "second_peak_ratio"):
        pl = merged.get(section, {}).get("per_landmark", {})
        for name, val in list(pl.items()):
            if val is None:
                del pl[name]
    return merged


def _build_override(args: argparse.Namespace, base_gate: dict) -> dict:
    override: dict = {}
    if args.confidence_override:
        with open(args.confidence_override) as f:
            data = yaml.safe_load(f) or {}
        override = data.get("confidence", data)
    tier_override = _parse_tier(args.tier)
    override = _apply_permissive_removals(override, tier_override)

    core = set(base_gate.get("core_landmarks", []) or [])
    core |= set(args.core_landmark or [])
    core -= set(args.no_core_landmark or [])
    override["core_landmarks"] = sorted(core)
    return override


def _format_gate_row(name: str, result: dict, gate_cfg: dict) -> str:
    peak = result["confidences"][name]
    sharp = result["sharpness"][name]
    spr = result["second_peak_ratio"][name]
    rel = "PASS" if result["reliable"][name] else "FAIL"
    reason = result["gate_reason"][name] or "—"
    peak_thr = gate_cfg["peak"]["per_landmark"].get(name, gate_cfg["peak"]["global"])
    sharp_thr = gate_cfg["sharpness"]["per_landmark"].get(name, gate_cfg["sharpness"]["global"])
    spr_thr = gate_cfg["second_peak_ratio"]["per_landmark"].get(name, gate_cfg["second_peak_ratio"]["global"])
    return (
        f"  {rel}  {name:20s} "
        f"peak={peak:.3f} (≥{peak_thr:.3f})  "
        f"sharp={sharp:.2f} (≥{sharp_thr:.2f})  "
        f"sp_ratio={spr:.2f} (≤{spr_thr:.2f})  "
        f"{reason}"
    )


def main() -> None:
    """Parse args and run prediction."""
    parser = argparse.ArgumentParser(description="Predict landmarks on wing images")
    parser.add_argument("images", type=Path, nargs="+", help="Image path(s)")
    parser.add_argument("--checkpoint", type=Path, required=False, help="Single checkpoint path")
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        required=False,
        help="Directory with fold checkpoints for ensemble prediction",
    )
    parser.add_argument("--device", type=str, default=None, help="Device: mps, cuda, cpu")
    parser.add_argument("--output", type=Path, default=None, help="Output JSON path (single-output mode only)")
    parser.add_argument(
        "--batch",
        action="store_true",
        help=(
            "Folder-batch mode. The single positional `images` argument must be a directory; "
            "every supported image in it is processed and a per-image *_landmarks.geojson is "
            "written to --output-dir. Supersedes --output."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory for per-image GeoJSONs in --batch mode (created if missing).",
    )
    parser.add_argument(
        "--include-unreliable",
        action="store_true",
        help="Include landmarks that failed the confidence gate in the output (marked reliable=false).",
    )
    parser.add_argument(
        "--confidence-override",
        type=Path,
        default=None,
        help="YAML file whose top-level shape matches the `confidence:` block in configs/default.yaml.",
    )
    parser.add_argument(
        "--core-landmark",
        action="append",
        default=[],
        help="Add a landmark name to the abort-on-fail core set for this run (repeatable).",
    )
    parser.add_argument(
        "--no-core-landmark",
        action="append",
        default=[],
        help="Remove a landmark name from the core set for this run (repeatable).",
    )
    parser.add_argument(
        "--tier",
        action="append",
        default=[],
        metavar="NAME=strict|permissive",
        help="Set a landmark's threshold tier. 'strict' uses crossvein defaults; "
        "'permissive' clears the per-landmark entry so the global applies.",
    )
    parser.add_argument(
        "--print-gate",
        action="store_true",
        help="Print per-landmark gate result table to stdout for each image.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=0,
        help=(
            "Batch size for the model forward pass. 0 (default) auto-picks based on "
            "available memory and image count. 1 disables batching."
        ),
    )
    args = parser.parse_args()

    if not args.checkpoint and not args.checkpoint_dir:
        parser.error("Provide --checkpoint or --checkpoint-dir")

    # Resolve image inputs: in --batch mode, expand the folder into a list.
    if args.batch:
        if len(args.images) != 1 or not args.images[0].is_dir():
            parser.error("--batch requires exactly one positional argument that is a directory")
        if args.output_dir is None:
            parser.error("--batch requires --output-dir")
        image_paths = _discover_images(args.images[0])
        if not image_paths:
            parser.error(f"No supported images found in {args.images[0]}")
        args.output_dir.mkdir(parents=True, exist_ok=True)
    else:
        image_paths = list(args.images)

    # Load a peek of the checkpoint's existing gate config so we can apply core-set edits on top.
    import torch

    peek_path = args.checkpoint or sorted(args.checkpoint_dir.glob("best_fold*.pt"))[0]
    peek = torch.load(peek_path, map_location="cpu", weights_only=False)
    base_gate = _deep_merge(DEFAULT_GATE_CONFIG, peek.get("config", {}).get("confidence", {}) or {})
    override = _build_override(args, base_gate)

    results: dict = {}

    from landmark_locator.data.psd_loader import imread_any
    from landmark_locator.inference.predict import EnsemblePredictor, auto_batch_size

    # Build the right predictor.
    if args.checkpoint_dir:
        checkpoint_paths = sorted(args.checkpoint_dir.glob("best_fold*.pt"))
        if not checkpoint_paths:
            parser.error(f"No checkpoints found in {args.checkpoint_dir}")
        predictor = EnsemblePredictor(checkpoint_paths, device=args.device, confidence_override=override)
    else:
        predictor = LandmarkPredictor(args.checkpoint, args.device, confidence_override=override)

    # Load all images first, dropping any that fail to read.
    loaded: list[tuple[Path, "np.ndarray"]] = []
    for img_path in image_paths:
        image = imread_any(img_path)
        if image is None:
            print(f"Warning: could not load {img_path}, skipping")
            continue
        loaded.append((img_path, image))

    if not loaded:
        return

    # Pick batch size and run in chunks.
    batch_size = args.batch_size if args.batch_size > 0 else auto_batch_size(len(loaded))
    if args.batch_size == 0:
        print(f"Auto-selected batch size: {batch_size} (for {len(loaded)} image(s))")

    total = len(loaded)
    internal_to_geojson = {v: k for k, v in (predictor.geojson_to_landmark or {}).items()}
    succeeded = 0
    failed = 0
    done = 0

    for chunk_start in range(0, total, batch_size):
        chunk = loaded[chunk_start : chunk_start + batch_size]
        chunk_paths = [p for p, _ in chunk]
        chunk_imgs = [im for _, im in chunk]
        batch_results = predictor.predict_batch(
            chunk_imgs,
            include_unreliable=args.include_unreliable,
            raise_on_core_fail=False,
        )
        for img_path, result in zip(chunk_paths, batch_results):
            done += 1
            err = result.get("error")
            if isinstance(err, LowConfidenceLandmarkError):
                print(f"[{done}/{total}] {img_path.name}: ABORT (low-confidence core landmark) — {err}")
                results[str(img_path)] = {"error": "low_confidence_core", "failures": err.failures}
                failed += 1
                continue
            results[str(img_path)] = _shape_result(result)
            if args.print_gate:
                _print_gate(img_path, result, predictor.gate_config)
            if args.batch:
                gj = _result_to_geojson(result, internal_to_geojson)
                out_path = args.output_dir / f"{img_path.stem}_landmarks.geojson"
                out_path.write_text(json.dumps(gj))
                print(f"[{done}/{total}] {img_path.name}: ok → {out_path.name}")
            succeeded += 1

    if args.batch:
        print(f"\nDone: {succeeded} succeeded, {failed} failed out of {total}.")
        # In batch mode, --output (single JSON dump) is intentionally ignored.
        return

    output_json = json.dumps(results, indent=2)
    if args.output:
        args.output.write_text(output_json)
        print(f"Results written to {args.output}")
    else:
        print(output_json)


def _shape_result(result: dict) -> dict:
    return {
        "landmarks": {k: list(v) for k, v in result["landmarks"].items()},
        "confidences": result["confidences"],
        "sharpness": result["sharpness"],
        "second_peak_ratio": result["second_peak_ratio"],
        "reliable": result["reliable"],
        "gate_reason": result["gate_reason"],
    }


def _print_gate(img_path: Path, result: dict, gate_cfg: dict) -> None:
    print(f"Gate results for {img_path}:")
    for name in result["reliable"]:
        print(_format_gate_row(name, result, gate_cfg))


if __name__ == "__main__":
    main()
