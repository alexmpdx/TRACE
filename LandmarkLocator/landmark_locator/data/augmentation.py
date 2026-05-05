"""Augmentation pipelines with keypoint co-transforms."""

import albumentations as A
from albumentations.core.composition import Compose


def get_train_transform(cfg: dict) -> Compose:
    """Build training augmentation pipeline with keypoint co-transforms."""
    input_cfg = cfg["input"]
    aug_cfg = cfg["augmentation"]

    return A.Compose(
        [
            A.Resize(input_cfg["height"], input_cfg["width"]),
            A.Rotate(limit=aug_cfg.get("rotation_limit", 90), border_mode=0, p=0.8),
            A.HorizontalFlip(p=aug_cfg["horizontal_flip_p"]),
            A.VerticalFlip(p=aug_cfg.get("vertical_flip_p", 0.5)),
            A.RandomScale(scale_limit=aug_cfg["scale_limit"], p=0.5),
            A.PadIfNeeded(
                min_height=input_cfg["height"],
                min_width=input_cfg["width"],
                border_mode=0,
            ),
            A.CenterCrop(input_cfg["height"], input_cfg["width"]),
            A.ColorJitter(
                brightness=aug_cfg.get("color_jitter_brightness", 0.2),
                contrast=aug_cfg.get("color_jitter_contrast", 0.2),
                saturation=aug_cfg.get("color_jitter_saturation", 0.3),
                hue=aug_cfg.get("color_jitter_hue", 0.05),
                p=aug_cfg.get("color_jitter_p", 0.7),
            ),
            A.GaussianBlur(
                blur_limit=(3, aug_cfg["blur_limit"]),
                p=aug_cfg["blur_p"],
            ),
            A.CoarseDropout(
                num_holes_range=(1, aug_cfg["coarse_dropout_max_holes"]),
                hole_height_range=(aug_cfg["coarse_dropout_max_height"], aug_cfg["coarse_dropout_max_height"]),
                hole_width_range=(aug_cfg["coarse_dropout_max_width"], aug_cfg["coarse_dropout_max_width"]),
                p=aug_cfg["coarse_dropout_p"],
            ),
        ],
        keypoint_params=A.KeypointParams(
            format="xy",
            label_fields=["landmark_names"],
            remove_invisible=False,
        ),
    )


def get_val_transform(cfg: dict) -> Compose:
    """Build validation transform (resize only) with keypoint co-transforms."""
    input_cfg = cfg["input"]

    return A.Compose(
        [
            A.Resize(input_cfg["height"], input_cfg["width"]),
        ],
        keypoint_params=A.KeypointParams(
            format="xy",
            label_fields=["landmark_names"],
            remove_invisible=False,
        ),
    )
