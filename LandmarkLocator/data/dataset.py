"""Core dataset for landmark heatmap regression."""

import json
import re
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from data.augmentation import get_train_transform, get_val_transform

# ImageNet normalization stats
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def _normalize_name(geojson_name: str) -> str:
    """Convert a GeoJSON classification name to a snake_case internal name."""
    name = geojson_name.strip()
    # Replace dots, hyphens, spaces with underscores
    name = re.sub(r"[\s.\-]+", "_", name)
    return name.lower()


def discover_landmarks(annotation_dir: Path) -> tuple[list[str], dict[str, str]]:
    """Scan GeoJSON files to discover all landmark names.

    Returns:
        (landmark_order, geojson_to_landmark) where landmark_order is the
        sorted canonical list of internal names and geojson_to_landmark maps
        GeoJSON classification names to internal names.
    """
    geojson_to_landmark: dict[str, str] = {}
    for path in sorted(annotation_dir.glob("*.geojson")):
        with open(path) as f:
            data = json.load(f)
        for feature in data["features"]:
            props = feature.get("properties", {})
            classification = props.get("classification")
            if not isinstance(classification, dict):
                continue
            geojson_name = classification.get("name")
            if not geojson_name:
                continue
            geom = feature.get("geometry", {})
            if geom.get("type") not in ("Point", "MultiPoint"):
                continue
            if geojson_name not in geojson_to_landmark:
                geojson_to_landmark[geojson_name] = _normalize_name(geojson_name)

    landmark_order = sorted(set(geojson_to_landmark.values()))
    return landmark_order, geojson_to_landmark


def extract_genotype(filename: str) -> str:
    """Extract genotype prefix from image filename."""
    name = filename.lstrip("-")
    if name.startswith("CTRL"):
        return "CTRL"
    elif name.startswith("en-"):
        return "en-PknRNAi"
    elif name.startswith("PknCG736"):
        return "PknCG736"
    return "unknown"


class LandmarkDataset(Dataset):
    """Dataset for wing landmark heatmap regression."""

    def __init__(
        self,
        annotation_dir: Path,
        image_dir: Path,
        cfg: dict,
        indices: Optional[list[int]] = None,
        train: bool = True,
    ) -> None:
        """Load annotation list and configure transforms."""
        self.annotation_dir = Path(annotation_dir)
        self.image_dir = Path(image_dir)
        self.cfg = cfg
        self.train = train
        self.input_h = cfg["input"]["height"]
        self.input_w = cfg["input"]["width"]
        self.sigma = cfg["heatmap"]["sigma"]

        # Load landmark definitions from config (set during training setup)
        self.landmark_order: list[str] = cfg["heatmap"]["landmark_order"]
        self.geojson_to_landmark: dict[str, str] = cfg["heatmap"]["geojson_to_landmark"]

        # Collect all geojson files, sorted for reproducibility
        all_files = sorted(self.annotation_dir.glob("*.geojson"))
        if not all_files:
            raise FileNotFoundError(f"No .geojson files in {self.annotation_dir}")

        # Verify matching images exist
        self.samples = []
        for geojson_path in all_files:
            # filename: foo.tif.geojson → image: foo.tif
            image_name = geojson_path.stem  # foo.tif
            image_path = self.image_dir / image_name
            if not image_path.exists():
                raise FileNotFoundError(
                    f"Image not found: {image_path} for annotation {geojson_path}"
                )
            self.samples.append((geojson_path, image_path))

        # Subset by indices if provided (for CV splits)
        if indices is not None:
            self.samples = [self.samples[i] for i in indices]

        # Build transforms
        if train:
            self.transform = get_train_transform(cfg)
        else:
            self.transform = get_val_transform(cfg)

    def __len__(self) -> int:
        """Return dataset size."""
        return len(self.samples)

    def _parse_geojson(self, path: Path) -> dict[str, tuple[float, float]]:
        """Parse GeoJSON Point annotations, handling MultiPoint anomaly."""
        with open(path) as f:
            data = json.load(f)

        landmarks = {}
        for feature in data["features"]:
            geom = feature.get("geometry", {})
            props = feature.get("properties", {})
            classification = props.get("classification")
            if not isinstance(classification, dict):
                continue
            class_name = classification.get("name")
            if not class_name:
                continue

            internal_name = self.geojson_to_landmark.get(class_name)
            if internal_name is None:
                continue

            if geom["type"] == "Point":
                x, y = geom["coordinates"]
            elif geom["type"] == "MultiPoint":
                # Take first coordinate (anomaly handling)
                x, y = geom["coordinates"][0]
            else:
                continue

            landmarks[internal_name] = (float(x), float(y))

        # Validate all expected landmarks present
        missing = set(self.landmark_order) - set(landmarks.keys())
        if missing:
            raise ValueError(f"Missing landmarks in {path}: {missing}")

        return landmarks

    def _generate_heatmap(
        self, keypoints: list[tuple[float, float]], h: int, w: int
    ) -> np.ndarray:
        """Render Gaussian heatmaps from keypoint coordinates."""
        num_landmarks = len(keypoints)
        heatmaps = np.zeros((num_landmarks, h, w), dtype=np.float32)

        for i, (kx, ky) in enumerate(keypoints):
            # Clamp to image bounds
            kx = np.clip(kx, 0, w - 1)
            ky = np.clip(ky, 0, h - 1)

            # Generate Gaussian around the keypoint
            sigma = self.sigma
            size = int(6 * sigma)
            x0 = max(0, int(kx) - size)
            x1 = min(w, int(kx) + size + 1)
            y0 = max(0, int(ky) - size)
            y1 = min(h, int(ky) + size + 1)

            if x0 >= x1 or y0 >= y1:
                continue

            xs = np.arange(x0, x1, dtype=np.float32)
            ys = np.arange(y0, y1, dtype=np.float32)
            xx, yy = np.meshgrid(xs, ys)

            gaussian = np.exp(-((xx - kx) ** 2 + (yy - ky) ** 2) / (2 * sigma ** 2))
            heatmaps[i, y0:y1, x0:x1] = gaussian

        return heatmaps

    def __getitem__(self, idx: int) -> dict:
        """Load image, apply transforms, generate heatmaps."""
        geojson_path, image_path = self.samples[idx]

        # Load image as BGR uint8
        image = cv2.imread(str(image_path))
        if image is None:
            raise IOError(f"Failed to load image: {image_path}")
        orig_h, orig_w = image.shape[:2]

        # Parse landmarks in original pixel coords
        landmarks_dict = self._parse_geojson(geojson_path)

        # Build keypoint list in canonical order
        keypoints = []
        landmark_names = []
        for name in self.landmark_order:
            x, y = landmarks_dict[name]
            keypoints.append((x, y))
            landmark_names.append(name)

        # Convert BGR → RGB for augmentation
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Apply augmentation with keypoint co-transforms
        transformed = self.transform(
            image=image_rgb,
            keypoints=keypoints,
            landmark_names=landmark_names,
        )
        aug_image = transformed["image"]  # (H, W, 3) uint8 RGB
        aug_keypoints = transformed["keypoints"]
        aug_names = transformed["landmark_names"]

        # Reorder keypoints back to canonical order (albumentations may reorder)
        kp_dict = dict(zip(aug_names, aug_keypoints))
        ordered_kps = [kp_dict[name] for name in self.landmark_order]

        # Generate heatmaps from transformed keypoints
        heatmaps = self._generate_heatmap(ordered_kps, self.input_h, self.input_w)

        # Normalize image: uint8 → float32 [0,1] → ImageNet normalize
        img_float = aug_image.astype(np.float32) / 255.0
        img_float = (img_float - IMAGENET_MEAN) / IMAGENET_STD

        # HWC → CHW
        img_tensor = torch.from_numpy(img_float.transpose(2, 0, 1))
        heatmap_tensor = torch.from_numpy(heatmaps)

        # Scale factors for mapping model coords back to original
        scale_x = orig_w / self.input_w
        scale_y = orig_h / self.input_h

        # Landmark coords at model resolution
        landmark_coords = torch.tensor(ordered_kps, dtype=torch.float32)

        return {
            "image": img_tensor,
            "heatmaps": heatmap_tensor,
            "landmarks": landmark_coords,
            "image_path": str(image_path),
            "scale_x": scale_x,
            "scale_y": scale_y,
        }
