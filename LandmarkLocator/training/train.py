"""Training orchestrator: CV splits, training loop, evaluation."""

import sys
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import yaml
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader
from tqdm import tqdm

# Add project root to path for imports
_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from data.dataset import LANDMARK_ORDER, LandmarkDataset, extract_genotype
from models.unet import LandmarkUNet
from training.losses import HeatmapMSELoss


def get_device(requested: Optional[str] = None) -> torch.device:
    """Auto-detect best available device: MPS > CUDA > CPU."""
    if requested:
        return torch.device(requested)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def extract_landmarks_from_heatmaps(heatmaps: np.ndarray) -> list[tuple[float, float]]:
    """Extract landmark coordinates as weighted average around peak.

    Args:
        heatmaps: (C, H, W) predicted heatmap array

    Returns:
        List of (x, y) coordinates, one per channel.
    """
    coords = []
    for c in range(heatmaps.shape[0]):
        hm = heatmaps[c]
        # Find peak
        peak_idx = np.unravel_index(np.argmax(hm), hm.shape)
        peak_y, peak_x = peak_idx

        # Weighted average in 11×11 window around peak for sub-pixel accuracy
        radius = 5
        y0 = max(0, peak_y - radius)
        y1 = min(hm.shape[0], peak_y + radius + 1)
        x0 = max(0, peak_x - radius)
        x1 = min(hm.shape[1], peak_x + radius + 1)

        patch = hm[y0:y1, x0:x1]
        patch = np.maximum(patch, 0)  # ReLU to avoid negative weights
        total = patch.sum()

        if total > 1e-8:
            ys = np.arange(y0, y1, dtype=np.float64)
            xs = np.arange(x0, x1, dtype=np.float64)
            xx, yy = np.meshgrid(xs, ys)
            wx = (patch * xx).sum() / total
            wy = (patch * yy).sum() / total
            coords.append((float(wx), float(wy)))
        else:
            # Fallback to argmax if heatmap is near-zero
            coords.append((float(peak_x), float(peak_y)))

    return coords


def create_cv_splits(
    annotation_dir: Path, n_folds: int = 5
) -> list[tuple[list[int], list[int]]]:
    """Create stratified K-fold splits by genotype."""
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
) -> dict:
    """Train one fold and return best validation metrics."""
    project_root = Path(__file__).resolve().parent.parent
    annotation_dir = project_root / cfg["data"]["annotation_dir"]
    image_dir = project_root / cfg["data"]["image_dir"]

    # Create datasets
    train_ds = LandmarkDataset(annotation_dir, image_dir, cfg, train_indices, train=True)
    val_ds = LandmarkDataset(annotation_dir, image_dir, cfg, val_indices, train=False)

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

    checkpoint_dir = output_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(train_cfg["epochs"]):
        # Unfreeze encoder after freeze period
        if epoch == freeze_epochs:
            model.unfreeze_encoder()
            # Rebuild optimizer with differential LR
            param_groups = model.get_param_groups(
                opt_cfg["lr"], train_cfg["encoder_lr_factor"]
            )
            optimizer = torch.optim.AdamW(
                param_groups,
                weight_decay=opt_cfg["weight_decay"],
            )
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=train_cfg["epochs"] - freeze_epochs,
                eta_min=train_cfg["scheduler"]["eta_min"],
            )
            print(f"  Epoch {epoch}: unfreezing encoder with {train_cfg['encoder_lr_factor']}× LR")

        # --- Train phase ---
        model.train()
        train_loss = 0.0
        for batch in train_loader:
            images = batch["image"].to(device)
            targets = batch["heatmaps"].to(device)

            pred = model(images)
            loss = criterion(pred, targets)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * images.size(0)

        train_loss /= len(train_ds)
        scheduler.step()

        # --- Validation phase ---
        model.eval()
        val_loss = 0.0
        all_errors = []  # per-landmark pixel errors in original coords

        with torch.no_grad():
            for batch in val_loader:
                images = batch["image"].to(device)
                targets = batch["heatmaps"].to(device)
                gt_landmarks = batch["landmarks"]  # (B, 5, 2) at model resolution
                scale_x = batch["scale_x"]  # (B,)
                scale_y = batch["scale_y"]  # (B,)

                pred = model(images)
                loss = criterion(pred, targets)
                val_loss += loss.item() * images.size(0)

                # Extract predicted landmarks and compute pixel errors
                pred_np = pred.cpu().numpy()
                for b in range(pred_np.shape[0]):
                    pred_coords = extract_landmarks_from_heatmaps(pred_np[b])
                    gt_coords = gt_landmarks[b].numpy()
                    sx = scale_x[b].item()
                    sy = scale_y[b].item()

                    for i in range(len(LANDMARK_ORDER)):
                        px, py = pred_coords[i]
                        gx, gy = gt_coords[i]
                        # Error in original image pixels
                        err = np.sqrt(((px - gx) * sx) ** 2 + ((py - gy) * sy) ** 2)
                        all_errors.append((LANDMARK_ORDER[i], err))

        val_loss /= len(val_ds)

        # Compute per-landmark mean error
        landmark_errors = {}
        for name in LANDMARK_ORDER:
            errs = [e for n, e in all_errors if n == name]
            landmark_errors[name] = np.mean(errs) if errs else 0.0

        mean_error = np.mean([e for _, e in all_errors])

        # Epoch callback for live monitoring
        if epoch_callback is not None:
            epoch_callback(epoch, mean_error, landmark_errors)

        # Log progress
        if epoch % 10 == 0 or epoch == train_cfg["epochs"] - 1:
            print(
                f"  Fold {fold} Epoch {epoch:3d}: "
                f"train_loss={train_loss:.6f} val_loss={val_loss:.6f} "
                f"mean_px_error={mean_error:.1f}px"
            )

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
                print(f"  Fold {fold}: early stopping at epoch {epoch}")
                break

    print(f"  Fold {fold} best: epoch={best_metrics['epoch']}, "
          f"mean_error={best_metrics['mean_pixel_error']:.1f}px")
    for name, err in best_metrics["per_landmark_error"].items():
        print(f"    {name}: {err:.1f}px")

    return best_metrics


def run_training(
    config_path: Path,
    output_dir: Path,
    device_str: Optional[str] = None,
    fold: Optional[int] = None,
) -> None:
    """Run full cross-validation training."""
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    device = get_device(device_str)
    print(f"Using device: {device}")

    project_root = Path(__file__).resolve().parent.parent
    annotation_dir = project_root / cfg["data"]["annotation_dir"]

    splits = create_cv_splits(annotation_dir, cfg["cv"]["n_folds"])
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Determine which folds to train
    if fold is not None:
        folds_to_train = [fold]
    else:
        folds_to_train = list(range(len(splits)))

    all_metrics = {}
    for f_idx in folds_to_train:
        train_idx, val_idx = splits[f_idx]
        print(f"\n{'='*60}")
        print(f"Fold {f_idx}: {len(train_idx)} train, {len(val_idx)} val")
        print(f"{'='*60}")

        metrics = train_fold(cfg, f_idx, train_idx, val_idx, output_dir, device)
        all_metrics[f_idx] = metrics

    # Summary
    if len(all_metrics) > 1:
        mean_errors = [m["mean_pixel_error"] for m in all_metrics.values()]
        print(f"\n{'='*60}")
        print(f"CV Summary: mean_error={np.mean(mean_errors):.1f} ± {np.std(mean_errors):.1f}px")
        for name in LANDMARK_ORDER:
            errs = [m["per_landmark_error"][name] for m in all_metrics.values()]
            print(f"  {name}: {np.mean(errs):.1f} ± {np.std(errs):.1f}px")
