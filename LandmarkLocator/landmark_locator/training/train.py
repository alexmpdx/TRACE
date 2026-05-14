"""Training orchestrator: CV splits, training loop, evaluation."""

from pathlib import Path
from typing import Optional

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

from landmark_locator.data.dataset import LandmarkDataset, discover_landmarks, extract_genotype
from landmark_locator.models.unet import LandmarkUNet
from landmark_locator.training.losses import HeatmapMSELoss


def get_device(requested: Optional[str] = None) -> torch.device:
    """Auto-detect best available device: MPS > CUDA > CPU."""
    if requested:
        return torch.device(requested)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def log_config_summary(cfg: dict) -> None:
    """Print the full training config to stdout so the run log captures every knob.

    Called from both the CLI (`run_training`) and the GUI training thread so saved
    `training.log` files always carry the exact augmentation + training + CV settings
    used. Skips the noisy auto-injected `geojson_to_landmark` map for readability.
    """
    log_cfg = {k: v for k, v in cfg.items() if k != "confidence"}
    if "heatmap" in log_cfg:
        log_cfg["heatmap"] = {k: v for k, v in log_cfg["heatmap"].items() if k != "geojson_to_landmark"}
    print("=" * 60)
    print("TRAINING CONFIGURATION")
    print("=" * 60)
    print(yaml.safe_dump(log_cfg, sort_keys=False, default_flow_style=False).rstrip())
    print("=" * 60)


def _populate_landmark_config(cfg: dict, annotation_dir: Path) -> None:
    """Discover landmarks from annotations and inject into config dict."""
    landmark_order, geojson_to_landmark = discover_landmarks(annotation_dir)
    if not landmark_order:
        raise ValueError(f"No landmarks found in {annotation_dir}")
    cfg["heatmap"]["landmark_order"] = landmark_order
    cfg["heatmap"]["geojson_to_landmark"] = geojson_to_landmark
    cfg["heatmap"]["num_landmarks"] = len(landmark_order)
    print(f"Discovered {len(landmark_order)} landmarks: {landmark_order}")


# Re-export for backward compatibility
from landmark_locator.inference.predict import extract_landmarks_from_heatmaps  # noqa: F401, E402


def create_cv_splits(annotation_dir: Path, n_folds: int = 5) -> list[tuple[list[int], list[int]]]:
    """Create stratified K-fold splits by genotype."""
    from sklearn.model_selection import StratifiedKFold

    geojson_files = sorted(annotation_dir.glob("*.geojson"))
    genotypes = [extract_genotype(f.stem) for f in geojson_files]

    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
    splits = []
    dummy_X = np.zeros(len(genotypes))
    for train_idx, val_idx in skf.split(dummy_X, genotypes):
        splits.append((train_idx.tolist(), val_idx.tolist()))

    return splits


def train_fold(
    cfg: dict,
    fold: int,
    train_indices: list[int],
    val_indices: list[int],
    output_dir: Path,
    device: torch.device,
    epoch_callback: Optional[callable] = None,
    checkpoint_name: Optional[str] = None,
    interactive: bool = True,
    display_name: Optional[str] = None,
) -> dict:
    """Train one fold and return best validation metrics."""
    label = f"{display_name}_Fold{fold}" if display_name else f"Fold{fold}"
    project_root = Path(__file__).resolve().parent.parent.parent
    annotation_dir = project_root / cfg["data"]["annotation_dir"]
    image_dir = project_root / cfg["data"]["image_dir"]

    # Create datasets
    train_ds = LandmarkDataset(annotation_dir, image_dir, cfg, train_indices, train=True, interactive=interactive)
    val_ds = LandmarkDataset(annotation_dir, image_dir, cfg, val_indices, train=False, interactive=interactive)

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg["training"]["batch_size"],
        shuffle=True,
        num_workers=cfg["training"]["num_workers"],
        pin_memory=False,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg["training"]["batch_size"],
        shuffle=False,
        num_workers=cfg["training"]["num_workers"],
        pin_memory=False,
    )

    landmark_order = cfg["heatmap"]["landmark_order"]

    # Model
    model = LandmarkUNet(
        num_landmarks=cfg["heatmap"]["num_landmarks"],
        pretrained=cfg["model"]["pretrained"],
    ).to(device)

    # Freeze encoder initially
    freeze_epochs = cfg["training"]["encoder_freeze_epochs"]
    model.freeze_encoder()

    # Optimizer and scheduler (decoder only initially)
    train_cfg = cfg["training"]
    opt_cfg = train_cfg["optimizer"]
    decoder_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        decoder_params,
        lr=opt_cfg["lr"],
        weight_decay=opt_cfg["weight_decay"],
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=train_cfg["scheduler"]["T_max"],
        eta_min=train_cfg["scheduler"]["eta_min"],
    )

    criterion = HeatmapMSELoss()

    # Training state
    best_val_error = float("inf")
    patience_counter = 0
    patience = train_cfg["early_stopping_patience"]
    best_metrics = {}

    # Flat layout: checkpoints + chart + log + gate_config.yaml all live in output_dir
    # (typically `trained_models/<name>/`). No extra `checkpoints/` subfolder.
    checkpoint_dir = output_dir
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(train_cfg["epochs"]):
        # Unfreeze encoder after freeze period
        if epoch == freeze_epochs:
            model.unfreeze_encoder()
            # Rebuild optimizer with differential LR
            param_groups = model.get_param_groups(opt_cfg["lr"], train_cfg["encoder_lr_factor"])
            optimizer = torch.optim.AdamW(
                param_groups,
                weight_decay=opt_cfg["weight_decay"],
            )
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=train_cfg["epochs"] - freeze_epochs,
                eta_min=train_cfg["scheduler"]["eta_min"],
            )
            print(f"  {label} Epoch {epoch}: unfreezing encoder with {train_cfg['encoder_lr_factor']}× LR")

        # --- Train phase ---
        model.train()
        train_loss = 0.0
        for batch in train_loader:
            images = batch["image"].to(device)
            targets = batch["heatmaps"].to(device)
            mask = batch.get("presence")
            if mask is not None:
                mask = mask.to(device)

            pred = model(images)
            loss = criterion(pred, targets, mask=mask)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * images.size(0)

        train_loss /= len(train_ds)
        scheduler.step()

        # --- Validation phase ---
        model.eval()
        val_loss = 0.0
        all_errors = []  # per-landmark pixel errors (only for landmarks present in GT)

        with torch.no_grad():
            for batch in val_loader:
                images = batch["image"].to(device)
                targets = batch["heatmaps"].to(device)
                gt_landmarks = batch["landmarks"]  # (B, N, 2) at model resolution
                presence = batch.get("presence")  # (B, N) or None
                scale_x = batch["scale_x"]  # (B,)
                scale_y = batch["scale_y"]  # (B,)

                pred = model(images)
                mask = presence.to(device) if presence is not None else None
                loss = criterion(pred, targets, mask=mask)
                val_loss += loss.item() * images.size(0)

                # Extract predicted landmarks and compute pixel errors (skip absent GT)
                pred_np = pred.cpu().numpy()
                presence_np = presence.numpy() if presence is not None else None
                for b in range(pred_np.shape[0]):
                    pred_coords = extract_landmarks_from_heatmaps(pred_np[b])
                    gt_coords = gt_landmarks[b].numpy()
                    sx = scale_x[b].item()
                    sy = scale_y[b].item()

                    for i in range(len(landmark_order)):
                        if presence_np is not None and presence_np[b, i] == 0:
                            continue
                        px, py = pred_coords[i]
                        gx, gy = gt_coords[i]
                        err = np.sqrt(((px - gx) * sx) ** 2 + ((py - gy) * sy) ** 2)
                        all_errors.append((landmark_order[i], err))

        val_loss /= len(val_ds)

        # Compute per-landmark mean error
        landmark_errors = {}
        for name in landmark_order:
            errs = [e for n, e in all_errors if n == name]
            landmark_errors[name] = np.mean(errs) if errs else 0.0

        mean_error = np.mean([e for _, e in all_errors])

        # Epoch callback for live monitoring
        if epoch_callback is not None:
            epoch_callback(epoch, mean_error, landmark_errors, train_loss, val_loss)

        # Log progress
        if epoch % 10 == 0 or epoch == train_cfg["epochs"] - 1:
            print(
                f"  {label} Epoch {epoch:3d}: "
                f"train_loss={train_loss:.6f} val_loss={val_loss:.6f} "
                f"mean_px_error={mean_error:.1f}px"
            )
            per_lm = " | ".join(f"{n}: {landmark_errors[n]:.1f}px" for n in landmark_order)
            print(f"    {per_lm}")

        # Checkpointing
        if mean_error < best_val_error:
            best_val_error = mean_error
            patience_counter = 0
            best_metrics = {
                "epoch": epoch,
                "val_loss": val_loss,
                "mean_pixel_error": mean_error,
                "per_landmark_error": landmark_errors.copy(),
            }
            ckpt_filename = checkpoint_name if checkpoint_name else f"best_fold{fold}.pt"
            if not ckpt_filename.endswith(".pt"):
                ckpt_filename += ".pt"
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_loss": val_loss,
                    "mean_pixel_error": mean_error,
                    "config": cfg,
                },
                checkpoint_dir / ckpt_filename,
            )
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"  {label}: early stopping at epoch {epoch}")
                break

    print(f"  {label} best: epoch={best_metrics['epoch']}, " f"mean_error={best_metrics['mean_pixel_error']:.1f}px")
    for name, err in best_metrics["per_landmark_error"].items():
        print(f"    {name}: {err:.1f}px")

    return best_metrics


def run_training(
    config_path: Path,
    output_dir: Path,
    device_str: Optional[str] = None,
    fold: Optional[int] = None,
    name: Optional[str] = None,
    interactive: bool = False,
) -> None:
    """Run full cross-validation training.

    When `name` is provided, checkpoints are written under `output_dir / name /` so
    every training run gets its own self-contained folder (fold checkpoints + chart).
    When `interactive=False` (CLI default), fuzzy image/GeoJSON name matches are auto-accepted.
    """
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    device = get_device(device_str)
    print(f"Using device: {device}")

    project_root = Path(__file__).resolve().parent.parent.parent
    annotation_dir = Path(cfg["data"]["annotation_dir"])
    if not annotation_dir.is_absolute():
        annotation_dir = project_root / annotation_dir

    # Auto-discover landmarks from annotation files
    _populate_landmark_config(cfg, annotation_dir)

    log_config_summary(cfg)

    splits = create_cv_splits(annotation_dir, cfg["cv"]["n_folds"])
    output_dir = Path(output_dir)
    if name:
        output_dir = output_dir / name
    output_dir.mkdir(parents=True, exist_ok=True)

    # Determine which folds to train
    if fold is not None:
        folds_to_train = [fold]
    else:
        folds_to_train = list(range(len(splits)))

    all_metrics = {}
    for f_idx in folds_to_train:
        train_idx, val_idx = splits[f_idx]
        fold_label = f"{name}_Fold{f_idx}" if name else f"Fold{f_idx}"
        print(f"\n{'='*60}")
        print(f"{fold_label}: {len(train_idx)} train, {len(val_idx)} val")
        print(f"{'='*60}")

        metrics = train_fold(
            cfg,
            f_idx,
            train_idx,
            val_idx,
            output_dir,
            device,
            display_name=name,
            interactive=interactive,
        )
        all_metrics[f_idx] = metrics

    # Summary
    if len(all_metrics) > 1:
        mean_errors = [m["mean_pixel_error"] for m in all_metrics.values()]
        print(f"\n{'='*60}")
        print(f"CV Summary: mean_error={np.mean(mean_errors):.1f} ± {np.std(mean_errors):.1f}px")
        for name in cfg["heatmap"]["landmark_order"]:
            errs = [m["per_landmark_error"][name] for m in all_metrics.values()]
            print(f"  {name}: {np.mean(errs):.1f} ± {np.std(errs):.1f}px")
