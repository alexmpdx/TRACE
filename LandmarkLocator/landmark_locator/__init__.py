"""LandmarkLocator — heatmap-based landmark detection for Drosophila wing images.

Public API (available via ``from landmark_locator import ...``):

Inference:
    LandmarkPredictor  — load a checkpoint and predict landmarks on images
    predict_ensemble   — average predictions from multiple fold checkpoints

Model:
    LandmarkUNet       — ResNet18-encoder U-Net for heatmap regression

Data utilities:
    discover_landmarks — scan GeoJSON annotations to find landmark names
    canonical_landmark_name — map a legacy landmark name (e.g. "DTip") to its current one
    LandmarkDataset    — PyTorch Dataset for training

Training (imported from landmark_locator.training):
    run_training, train_fold, create_cv_splits, get_device,
    extract_landmarks_from_heatmaps, HeatmapMSELoss
"""


def __getattr__(name: str):
    """Lazy imports to avoid pulling in heavy deps (sklearn) on every import."""
    # Inference
    if name in (
        "LandmarkPredictor",
        "predict_ensemble",
        "LowConfidenceLandmarkError",
        "EnsemblePredictor",
        "make_predictor",
        "auto_batch_size",
    ):
        from landmark_locator.inference.predict import (
            EnsemblePredictor,
            LandmarkPredictor,
            LowConfidenceLandmarkError,
            auto_batch_size,
            make_predictor,
            predict_ensemble,
        )

        return {
            "LandmarkPredictor": LandmarkPredictor,
            "predict_ensemble": predict_ensemble,
            "LowConfidenceLandmarkError": LowConfidenceLandmarkError,
            "EnsemblePredictor": EnsemblePredictor,
            "make_predictor": make_predictor,
            "auto_batch_size": auto_batch_size,
        }[name]

    # Batch-specific gate thresholds
    if name in (
        "batch_gate_active",
        "effective_gate_config",
        "gate_off_override",
        "resolve_batch_gate_config",
        "regate_raw",
        "regate_shaped",
    ):
        from landmark_locator.inference import batch_gate

        return getattr(batch_gate, name)

    # Model
    if name == "LandmarkUNet":
        from landmark_locator.models.unet import LandmarkUNet

        return LandmarkUNet

    # Data
    # Naming (torch-free)
    if name in ("canonical_landmark_name", "LEGACY_LANDMARK_NAMES"):
        from landmark_locator.names import LEGACY_LANDMARK_NAMES, canonical_landmark_name

        return {
            "canonical_landmark_name": canonical_landmark_name,
            "LEGACY_LANDMARK_NAMES": LEGACY_LANDMARK_NAMES,
        }[name]

    if name in ("LandmarkDataset", "discover_landmarks", "extract_genotype", "IMAGENET_MEAN", "IMAGENET_STD"):
        from landmark_locator.data.dataset import (
            IMAGENET_MEAN,
            IMAGENET_STD,
            LandmarkDataset,
            discover_landmarks,
            extract_genotype,
        )

        return {
            "LandmarkDataset": LandmarkDataset,
            "discover_landmarks": discover_landmarks,
            "extract_genotype": extract_genotype,
            "IMAGENET_MEAN": IMAGENET_MEAN,
            "IMAGENET_STD": IMAGENET_STD,
        }[name]

    # Training
    if name in (
        "run_training",
        "train_fold",
        "create_cv_splits",
        "extract_landmarks_from_heatmaps",
        "get_device",
    ):
        from landmark_locator.training.train import (
            create_cv_splits,
            extract_landmarks_from_heatmaps,
            get_device,
            run_training,
            train_fold,
        )

        return {
            "run_training": run_training,
            "train_fold": train_fold,
            "create_cv_splits": create_cv_splits,
            "extract_landmarks_from_heatmaps": extract_landmarks_from_heatmaps,
            "get_device": get_device,
        }[name]

    if name == "HeatmapMSELoss":
        from landmark_locator.training.losses import HeatmapMSELoss

        return HeatmapMSELoss

    raise AttributeError(f"module 'landmark_locator' has no attribute {name!r}")


__all__ = [
    "LandmarkPredictor",
    "EnsemblePredictor",
    "make_predictor",
    "predict_ensemble",
    "auto_batch_size",
    "LowConfidenceLandmarkError",
    "batch_gate_active",
    "effective_gate_config",
    "gate_off_override",
    "resolve_batch_gate_config",
    "regate_raw",
    "regate_shaped",
    "LandmarkUNet",
    "LandmarkDataset",
    "discover_landmarks",
    "canonical_landmark_name",
    "LEGACY_LANDMARK_NAMES",
    "extract_genotype",
    "IMAGENET_MEAN",
    "IMAGENET_STD",
    "run_training",
    "train_fold",
    "create_cv_splits",
    "extract_landmarks_from_heatmaps",
    "get_device",
    "HeatmapMSELoss",
]
