"""Train or fine-tune a landmark model from an explicit list of annotated images.

The classic path (`run_training`) reads one annotation directory and one image
directory named in a YAML config. This one takes ``(geojson_path, image_path)``
pairs that can live anywhere, which is what a host application needs to train on
annotations its users made in place (TRACE's manual landmark overrides).

Two modes:

- **Fine-tune** (`base_model` given): start from an existing model's weights and
  keep its landmark set and channel order. The right choice for a few dozen
  corrected images; each fold reports the starting model's error on its held-out
  images next to the fine-tuned one.
- **From scratch** (`base_model=None`): ImageNet-pretrained encoder, landmark set
  discovered from the annotations. Needs a few hundred images to be competitive.

The output folder has the layout every loader in the suite expects: `best_fold<N>.pt`
directly inside it, plus a `gate_config.yaml` calibrated for the new model (see
`calibrate_gate_config`) and a `training_info.json` record of what went in.
"""

from __future__ import annotations

import copy
import json
import shutil
from datetime import datetime
from pathlib import Path
from typing import Callable, Optional

import numpy as np

from landmark_locator.names import canonical_landmark_name

# Used when training from scratch and no `configs/default.yaml` is on disk (a
# frozen application does not ship it). Mirrors that file's values.
FALLBACK_CONFIG: dict = {
    "input": {"width": 512, "height": 352},
    "heatmap": {"sigma": 5},
    "model": {"encoder": "resnet18", "pretrained": True, "decoder_channels": [256, 128, 64, 32]},
    "training": {
        "epochs": 300,
        "batch_size": 5,
        "num_workers": 0,
        "optimizer": {"type": "AdamW", "lr": 1.0e-3, "weight_decay": 1.0e-4},
        "scheduler": {"type": "CosineAnnealingLR", "T_max": 300, "eta_min": 1.0e-6},
        "encoder_freeze_epochs": 20,
        "encoder_lr_factor": 0.1,
        "early_stopping_patience": 50,
    },
    "augmentation": {
        "rotation_limit": 45,
        "horizontal_flip_p": 0.5,
        "vertical_flip_p": 0.5,
        "scale_limit": 0.15,
        "brightness_limit": 0.2,
        "contrast_limit": 0.2,
        "blur_limit": 5,
        "blur_p": 0.3,
        "coarse_dropout_max_holes": 4,
        "coarse_dropout_max_height": 40,
        "coarse_dropout_max_width": 40,
        "coarse_dropout_p": 0.3,
    },
}

# Fine-tuning defaults: a tenth of the from-scratch learning rate and no frozen
# warm-up (the decoder is already trained), so a small set of corrections nudges
# the model instead of overwriting what it knew.
FINETUNE_EPOCHS = 60
FINETUNE_LR = 1.0e-4
SCRATCH_EPOCHS = 300
MIN_SAMPLES = 2

INFO_FILENAME = "training_info.json"

# Gate calibration. Tier percentiles (peak, sharpness, second-peak ratio) match the
# ones the bundled models' sidecars were calibrated with; the active gate is the
# permissive tier, as it is there.
GATE_TIERS: dict[str, tuple[float, float, float]] = {
    "permissive": (0.5, 0.5, 99.5),
    "standard": (5.0, 5.0, 95.0),
    "strict": (15.0, 15.0, 85.0),
}
ACTIVE_GATE_TIER = "permissive"
# A percentile of fewer correct held-out predictions than this is mostly noise (the
# 0.5th percentile of 8 values is just their minimum), so such a landmark keeps the
# base model's threshold instead — or the model-wide one when there is no base.
MIN_CALIBRATION_SAMPLES = 20
GATE_TOLERANCE_PX = 100.0
_GATE_METRICS = ("peak", "sharpness", "second_peak_ratio")


def _resolve_base_checkpoints(base_model: Path) -> list[Path]:
    """Checkpoints of a base model given as a fold folder or a single `.pt` file."""
    from landmark_locator.inference.predict import _find_fold_checkpoints

    base_model = Path(base_model)
    if base_model.is_dir():
        ckpts = _find_fold_checkpoints(base_model)
        if not ckpts:
            raise FileNotFoundError(f"No best_fold*.pt in {base_model}")
        return ckpts
    if not base_model.is_file():
        raise FileNotFoundError(f"Base model not found: {base_model}")
    return [base_model]


def read_sample_landmarks(geojson_path: Path) -> dict[str, tuple[float, float]]:
    """Named Point landmarks of one annotation file as ``{canonical GeoJSON name: (x, y)}``."""
    data = json.loads(Path(geojson_path).read_text(encoding="utf-8"))
    out: dict[str, tuple[float, float]] = {}
    for feature in data.get("features", []):
        geom = feature.get("geometry") or {}
        coords = geom.get("coordinates") or []
        if geom.get("type") == "MultiPoint" and coords:
            coords = coords[0]
        elif geom.get("type") != "Point":
            continue
        if len(coords) < 2:
            continue
        classification = (feature.get("properties") or {}).get("classification")
        name = classification.get("name") if isinstance(classification, dict) else None
        if not name:
            continue
        out[canonical_landmark_name(str(name))] = (float(coords[0]), float(coords[1]))
    return out


def base_model_landmarks(base_model: Path) -> list[str]:
    """Internal landmark names (channel order) of a base model, without building the network."""
    import torch

    ckpt = torch.load(_resolve_base_checkpoints(base_model)[0], map_location="cpu", weights_only=False)
    return [canonical_landmark_name(n) for n in ckpt["config"]["heatmap"]["landmark_order"]]


def summarize_samples(
    samples: list[tuple[Path, Path]],
    landmark_order: Optional[list[str]] = None,
) -> dict:
    """Count what a sample list can teach a model.

    Returns ``{"usable": [...pairs], "counts": {internal: n}, "ignored": {geojson name: n},
    "skipped": [(geojson_path, reason)], "display": {internal: geojson name}}``. With `landmark_order` (fine-tuning), names the
    model has no channel for are tallied under ``ignored`` and a sample with no known
    landmark is skipped; without it every named landmark counts.
    """
    from landmark_locator.data.dataset import _normalize_name

    known = set(landmark_order) if landmark_order is not None else None
    usable: list[tuple[Path, Path]] = []
    counts: dict[str, int] = {}
    ignored: dict[str, int] = {}
    skipped: list[tuple[Path, str]] = []
    display: dict[str, str] = {}
    for geojson_path, image_path in samples:
        try:
            landmarks = read_sample_landmarks(geojson_path)
        except Exception as exc:  # unreadable / malformed annotation
            skipped.append((Path(geojson_path), f"unreadable annotation ({exc})"))
            continue
        n_known = 0
        for name in landmarks:
            internal = _normalize_name(name)
            if known is not None and internal not in known:
                ignored[name] = ignored.get(name, 0) + 1
                continue
            counts[internal] = counts.get(internal, 0) + 1
            display.setdefault(internal, name)
            n_known += 1
        if n_known == 0:
            skipped.append((Path(geojson_path), "no landmarks the model knows"))
            continue
        usable.append((Path(geojson_path), Path(image_path)))
    return {"usable": usable, "counts": counts, "ignored": ignored, "skipped": skipped, "display": display}


def _gate_block(doc: Optional[dict]) -> dict:
    """The gate block of a sidecar document (which may or may not wrap it in ``confidence:``)."""
    from landmark_locator.inference.predict import canonicalize_gate_config

    if not isinstance(doc, dict):
        return {}
    block = doc.get("confidence", doc)
    return canonicalize_gate_config(block) if isinstance(block, dict) else {}


def collect_held_out_metrics(
    model_dir: Path,
    fold_val_samples: list[list[tuple[Path, Path]]],
    device: Optional[str] = None,
    tolerance_px: Optional[float] = None,
) -> dict[str, dict[str, list[float]]]:
    """Gate metrics of each fold's checkpoint on the images that fold did not train on.

    Returns ``{landmark: {"peak": [...], "sharpness": [...], "second_peak_ratio": [...]}}``
    over predictions that landed within `tolerance_px` (original-image pixels, default
    `GATE_TOLERANCE_PX`) of the
    annotation — a wrong prediction says nothing about what a right one looks like.
    """
    from landmark_locator.inference.predict import LandmarkPredictor, _find_fold_checkpoints
    from landmark_locator.scripts.calibrate_gate import _collect_metrics, _permissive_override

    tolerance_px = GATE_TOLERANCE_PX if tolerance_px is None else tolerance_px
    ckpts = _find_fold_checkpoints(Path(model_dir))
    collected: dict[str, dict[str, list[float]]] = {}
    for ckpt, val_samples in zip(ckpts, fold_val_samples):
        predictor = LandmarkPredictor(ckpt, device=device, confidence_override=_permissive_override())
        pairs = [(Path(image), Path(geojson)) for geojson, image in val_samples]
        for name, metrics in _collect_metrics(predictor, pairs, tolerance_px).items():
            slot = collected.setdefault(name, {m: [] for m in _GATE_METRICS})
            for metric in _GATE_METRICS:
                slot[metric].extend(metrics[metric])
    return collected


def build_gate_config(
    collected: dict[str, dict[str, list[float]]],
    landmark_order: list[str],
    base_doc: Optional[dict] = None,
    min_samples: Optional[int] = None,
) -> tuple[dict, dict]:
    """Turn held-out metrics into a sidecar document with the three tier presets.

    A landmark with at least `min_samples` correct held-out predictions gets its own
    percentile thresholds. One with fewer keeps the base model's value for that tier
    (`base_doc`, the base sidecar) or, with no base, has no per-landmark entry and falls
    under the model-wide threshold. Core landmarks, the suppression radius and any other
    settings come from the base sidecar, restricted to landmarks this model has.

    Returns ``(document, report)``; the report lists which landmarks were calibrated
    (with their sample count) and which were inherited or left to the global value.
    """
    from landmark_locator.inference.predict import DEFAULT_GATE_CONFIG, GATE_METRIC_DECIMALS

    min_samples = MIN_CALIBRATION_SAMPLES if min_samples is None else min_samples
    base = _gate_block(base_doc)
    base_tiers = base.get("tiers") if isinstance(base.get("tiers"), dict) else {}
    counts = {name: len(collected.get(name, {}).get("peak", [])) for name in landmark_order}
    calibrated = [name for name in landmark_order if counts[name] >= min_samples]

    def _base_value(tier: str, metric: str, name: str) -> Optional[float]:
        for block in (base_tiers.get(tier), base):
            value = ((block or {}).get(metric) or {}).get("per_landmark", {}).get(name)
            if value is not None:
                return float(value)
        return None

    def _tier(tier: str) -> dict:
        percentiles = dict(zip(_GATE_METRICS, GATE_TIERS[tier]))
        out: dict = {}
        for metric in _GATE_METRICS:
            per_landmark: dict[str, float] = {}
            for name in landmark_order:
                if name in calibrated:
                    value = float(np.percentile(collected[name][metric], percentiles[metric]))
                else:
                    value = _base_value(tier, metric, name)
                if value is not None:
                    per_landmark[name] = round(value, GATE_METRIC_DECIMALS[metric])
            if per_landmark:
                # The model-wide value is the loosest per-landmark one, as calibrate_gate does.
                loosest = max if metric == "second_peak_ratio" else min
                global_value = loosest(per_landmark.values())
            else:
                base_global = ((base_tiers.get(tier) or base).get(metric) or {}).get("global")
                global_value = DEFAULT_GATE_CONFIG[metric]["global"] if base_global is None else float(base_global)
            out[metric] = {"global": global_value, "per_landmark": per_landmark}
        return out

    known = set(landmark_order)
    block: dict = dict(_tier(ACTIVE_GATE_TIER))
    block["second_peak_suppression_radius_px"] = int(
        base.get("second_peak_suppression_radius_px", DEFAULT_GATE_CONFIG["second_peak_suppression_radius_px"])
    )
    block["core_landmarks"] = sorted(n for n in (base.get("core_landmarks") or []) if n in known)
    for key, value in base.items():  # e.g. batch mode, min_metric_failures_to_reject
        if key not in block and key != "tiers" and key not in _GATE_METRICS:
            block[key] = copy.deepcopy(value)
    block["tiers"] = {tier: _tier(tier) for tier in GATE_TIERS}

    inherited = [
        n for n in landmark_order if n not in calibrated and _base_value(ACTIVE_GATE_TIER, "peak", n) is not None
    ]
    report = {
        "min_samples": min_samples,
        "calibrated": {name: counts[name] for name in calibrated},
        "inherited_from_base": {name: counts[name] for name in inherited},
        "global_only": {n: counts[n] for n in landmark_order if n not in calibrated and n not in inherited},
    }
    return {"confidence": block}, report


def write_gate_config(model_dir: Path, document: dict, report: dict) -> Path:
    """Write a calibrated sidecar next to the model's checkpoints, with a header saying where it came from."""
    import yaml

    from landmark_locator.inference.predict import SIDECAR_GATE_FILENAME

    def _names(group: dict) -> str:
        return ", ".join(f"{name} ({n})" for name, n in group.items()) or "none"

    header = [
        "# Quality-gate thresholds for this landmark model, written when it was trained.",
        f"# Active gate = the {ACTIVE_GATE_TIER} tier; the tiers block holds all three presets.",
        "# Thresholds are percentiles of what this model's correct predictions looked like on images",
        f"# it did not train on. A landmark needs {report['min_samples']} such predictions to get its own;",
        "# the number in brackets is how many it had.",
        f"#   calibrated on this model: {_names(report['calibrated'])}",
        f"#   kept from the base model: {_names(report['inherited_from_base'])}",
        f"#   model-wide threshold only: {_names(report['global_only'])}",
    ]
    path = Path(model_dir) / SIDECAR_GATE_FILENAME
    path.write_text("\n".join(header) + "\n" + yaml.safe_dump(document, sort_keys=False), encoding="utf-8")
    return path


def calibrate_gate_config(
    model_dir: Path,
    fold_val_samples: list[list[tuple[Path, Path]]],
    landmark_order: list[str],
    base_model: Optional[Path] = None,
    device: Optional[str] = None,
) -> dict:
    """Calibrate and write `gate_config.yaml` for a freshly trained model; returns the report.

    `fold_val_samples[i]` are the ``(geojson, image)`` pairs fold *i* held out.
    """
    import yaml

    from landmark_locator.inference.predict import find_sidecar_gate_path

    base_doc = None
    if base_model is not None:
        sidecar = find_sidecar_gate_path(base_model)
        if sidecar is not None:
            base_doc = yaml.safe_load(sidecar.read_text(encoding="utf-8")) or {}
    collected = collect_held_out_metrics(model_dir, fold_val_samples, device=device)
    document, report = build_gate_config(collected, landmark_order, base_doc)
    write_gate_config(model_dir, document, report)
    return report


def make_splits(n_samples: int, n_folds: int, seed: int = 42) -> list[tuple[list[int], list[int]]]:
    """Shuffled train/validation index splits.

    `n_folds >= 2` gives K-fold splits (capped at `n_samples`, so every fold keeps at
    least one validation image); `n_folds == 1` gives a single 80/20 hold-out.
    """
    if n_samples < MIN_SAMPLES:
        raise ValueError(f"Need at least {MIN_SAMPLES} annotated images to train (got {n_samples}).")
    perm = np.random.default_rng(seed).permutation(n_samples)
    n_folds = max(1, min(int(n_folds), n_samples))
    if n_folds == 1:
        n_val = max(1, int(round(0.2 * n_samples)))
        return [(sorted(perm[n_val:].tolist()), sorted(perm[:n_val].tolist()))]
    splits = []
    for val in np.array_split(perm, n_folds):
        val_set = set(val.tolist())
        splits.append((sorted(i for i in range(n_samples) if i not in val_set), sorted(val_set)))
    return splits


def _scratch_template() -> dict:
    """Config for a from-scratch run: the repo's default.yaml when present, else the fallback."""
    yaml_path = Path(__file__).resolve().parents[2] / "configs" / "default.yaml"
    if yaml_path.is_file():
        import yaml

        cfg = yaml.safe_load(yaml_path.read_text()) or {}
        cfg.pop("confidence", None)
        cfg.pop("data", None)
        return cfg
    return copy.deepcopy(FALLBACK_CONFIG)


def build_config(
    samples: list[tuple[Path, Path]],
    base_checkpoint: Optional[Path],
    *,
    epochs: Optional[int] = None,
    lr: Optional[float] = None,
) -> dict:
    """Training config for a sample list: the base checkpoint's own config when fine-tuning."""
    from landmark_locator.data.dataset import _normalize_name

    if base_checkpoint is not None:
        import torch

        cfg = copy.deepcopy(torch.load(base_checkpoint, map_location="cpu", weights_only=False)["config"])
        cfg.pop("confidence", None)  # gate thresholds live in the sidecar YAML, not the checkpoint
        landmark_order = [canonical_landmark_name(n) for n in cfg["heatmap"]["landmark_order"]]
        epochs = int(epochs or FINETUNE_EPOCHS)
        lr = float(lr or FINETUNE_LR)
        freeze_epochs = 0
    else:
        cfg = _scratch_template()
        landmark_order = None
        epochs = int(epochs or SCRATCH_EPOCHS)
        lr = float(lr or cfg["training"]["optimizer"]["lr"])
        freeze_epochs = min(int(cfg["training"].get("encoder_freeze_epochs", 20)), max(epochs // 4, 0))

    # GeoJSON-name → channel map covering every spelling in the samples (and, when
    # fine-tuning, the ones the base model already knew).
    geojson_to_landmark: dict[str, str] = {}
    if base_checkpoint is not None:
        for k, v in cfg["heatmap"].get("geojson_to_landmark", {}).items():
            geojson_to_landmark[canonical_landmark_name(k)] = canonical_landmark_name(v)
    for geojson_path, _ in samples:
        for name in read_sample_landmarks(geojson_path):
            geojson_to_landmark.setdefault(name, _normalize_name(name))
    if landmark_order is None:
        landmark_order = sorted(set(geojson_to_landmark.values()))
    else:
        geojson_to_landmark = {k: v for k, v in geojson_to_landmark.items() if v in set(landmark_order)}
    if not landmark_order:
        raise ValueError("No named landmarks found in the annotations.")

    cfg["heatmap"]["landmark_order"] = landmark_order
    cfg["heatmap"]["geojson_to_landmark"] = geojson_to_landmark
    cfg["heatmap"]["num_landmarks"] = len(landmark_order)

    train_cfg = cfg["training"]
    train_cfg["epochs"] = epochs
    train_cfg["optimizer"]["lr"] = lr
    train_cfg["scheduler"]["T_max"] = max(epochs, 1)
    train_cfg["encoder_freeze_epochs"] = freeze_epochs
    train_cfg["early_stopping_patience"] = max(10, epochs // 4) if base_checkpoint is not None else max(20, epochs // 6)
    train_cfg["num_workers"] = 0  # worker processes cannot be spawned from a frozen GUI app
    return cfg


def train_from_samples(
    samples: list[tuple[Path, Path]],
    output_dir: Path,
    *,
    base_model: Optional[Path] = None,
    n_folds: int = 5,
    epochs: Optional[int] = None,
    lr: Optional[float] = None,
    device: Optional[str] = None,
    name: Optional[str] = None,
    epoch_callback: Optional[Callable[[dict], None]] = None,
    should_stop: Optional[Callable[[], bool]] = None,
) -> dict:
    """Train a fold ensemble on ``(geojson_path, image_path)`` pairs into `output_dir`.

    Args:
        samples: annotation / image pairs; coordinates are pixels of that image.
        output_dir: model folder to create (`best_fold<N>.pt` are written directly in it).
        base_model: fold folder or `.pt` to fine-tune from; None trains from scratch.
        n_folds: ensemble size. Fold *i* starts from the base model's fold *i* (cycling).
        epochs / lr: override the mode's defaults.
        epoch_callback: called once per epoch with
            ``{fold, n_folds, epoch, epochs, mean_error, landmark_errors, train_loss, val_loss}``.
        should_stop: polled every batch; True raises `TrainingCancelled`.

    Returns the dict also written to `training_info.json` (per-fold baseline / best errors).
    """
    from landmark_locator.inference.predict import SIDECAR_GATE_FILENAME, find_sidecar_gate_path
    from landmark_locator.training.train import get_device, log_config_summary, train_fold

    base_ckpts = _resolve_base_checkpoints(base_model) if base_model is not None else []
    order = base_model_landmarks(base_model) if base_model is not None else None
    summary = summarize_samples(samples, order)
    usable = summary["usable"]
    for path, reason in summary["skipped"]:
        print(f"Skipping {path.name}: {reason}")
    for geojson_name, n in sorted(summary["ignored"].items()):
        print(f"Ignoring landmark '{geojson_name}' ({n} image(s)): the base model has no channel for it")

    splits = make_splits(len(usable), n_folds)
    cfg = build_config(usable, base_ckpts[0] if base_ckpts else None, epochs=epochs, lr=lr)
    cfg["user_training"] = {
        "mode": "fine-tune" if base_ckpts else "scratch",
        "base_model": str(base_model) if base_model is not None else None,
        "n_images": len(usable),
    }

    torch_device = get_device(device)
    print(f"Using device: {torch_device}")
    print(f"{len(usable)} annotated images, {len(splits)} model(s) in the ensemble")
    log_config_summary(cfg)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    label = name or output_dir.name

    fold_results = []
    for fold_idx, (train_idx, val_idx) in enumerate(splits):
        print()
        print("=" * 60)
        print(f"{label}_Fold{fold_idx}: {len(train_idx)} train, {len(val_idx)} val")
        print("=" * 60)

        def _on_epoch(epoch, mean_error, landmark_errors, train_loss, val_loss, _fold=fold_idx):
            if epoch_callback is not None:
                epoch_callback(
                    {
                        "fold": _fold,
                        "n_folds": len(splits),
                        "epoch": epoch,
                        "epochs": cfg["training"]["epochs"],
                        "mean_error": float(mean_error),
                        "landmark_errors": dict(landmark_errors),
                        "train_loss": float(train_loss),
                        "val_loss": float(val_loss),
                    }
                )

        metrics = train_fold(
            cfg,
            fold_idx,
            train_idx,
            val_idx,
            output_dir,
            torch_device,
            epoch_callback=_on_epoch,
            interactive=False,
            display_name=label,
            samples=usable,
            init_checkpoint=base_ckpts[fold_idx % len(base_ckpts)] if base_ckpts else None,
            should_stop=should_stop,
        )
        fold_results.append(
            {
                "fold": fold_idx,
                "n_train": len(train_idx),
                "n_val": len(val_idx),
                "best_epoch": int(metrics["epoch"]),
                "baseline_pixel_error": metrics.get("baseline_pixel_error"),
                "best_pixel_error": float(metrics["mean_pixel_error"]),
                "per_landmark_error": {k: float(v) for k, v in metrics["per_landmark_error"].items()},
            }
        )

    # Quality-gate thresholds for the new model, from each fold's predictions on the
    # images it held out. Landmarks with too few of those keep the base model's values.
    # A calibration failure must not cost the user a finished model: fall back to the
    # base sidecar (the pre-calibration behaviour) and say so.
    fold_val_samples = [[usable[i] for i in val_idx] for _, val_idx in splits]
    for fold, val_samples in zip(fold_results, fold_val_samples):
        fold["val_images"] = [str(image) for _, image in val_samples]
        fold["val_annotations"] = [str(geojson) for geojson, _ in val_samples]
    try:
        gate_report = calibrate_gate_config(
            output_dir, fold_val_samples, cfg["heatmap"]["landmark_order"], base_model, device=torch_device.type
        )
        print(
            f"Quality-gate thresholds ({SIDECAR_GATE_FILENAME}): calibrated on this model for "
            f"{len(gate_report['calibrated'])} of {len(cfg['heatmap']['landmark_order'])} landmarks"
            + (
                f"; {len(gate_report['inherited_from_base'])} kept from the base model"
                if gate_report["inherited_from_base"]
                else ""
            )
            + (f"; {len(gate_report['global_only'])} on the model-wide threshold" if gate_report["global_only"] else "")
            + f" (needs {gate_report['min_samples']} correct held-out predictions per landmark)."
        )
    except Exception as exc:
        gate_report = {"error": f"{type(exc).__name__}: {exc}"}
        print(f"WARNING: could not calibrate quality-gate thresholds ({gate_report['error']}).")
        sidecar = find_sidecar_gate_path(base_model) if base_model is not None else None
        if sidecar is not None:
            shutil.copyfile(sidecar, output_dir / SIDECAR_GATE_FILENAME)
            print("Using the base model's thresholds instead.")

    best = [f["best_pixel_error"] for f in fold_results]
    baselines = [f["baseline_pixel_error"] for f in fold_results if f["baseline_pixel_error"] is not None]
    info = {
        "created": datetime.now().isoformat(timespec="seconds"),
        "mode": cfg["user_training"]["mode"],
        "base_model": cfg["user_training"]["base_model"],
        "n_images": len(usable),
        "landmark_order": cfg["heatmap"]["landmark_order"],
        "landmark_counts": summary["counts"],
        "ignored_landmarks": summary["ignored"],
        "epochs": cfg["training"]["epochs"],
        "lr": cfg["training"]["optimizer"]["lr"],
        "mean_pixel_error": float(np.mean(best)),
        "baseline_pixel_error": float(np.mean(baselines)) if baselines else None,
        "gate": gate_report,
        "folds": fold_results,
        "images": [str(img) for _, img in usable],
    }
    (output_dir / INFO_FILENAME).write_text(json.dumps(info, indent=2), encoding="utf-8")

    print()
    print("=" * 60)
    if info["baseline_pixel_error"] is not None:
        print(
            f"Held-out error: {info['baseline_pixel_error']:.1f}px before → {info['mean_pixel_error']:.1f}px after "
            f"(mean over {len(fold_results)} fold(s))"
        )
    else:
        print(f"Held-out error: {info['mean_pixel_error']:.1f}px (mean over {len(fold_results)} fold(s))")
    print(f"Model saved to {output_dir}")
    return info
