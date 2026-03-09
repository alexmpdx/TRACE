"""Prediction pipeline: load checkpoint, run inference, extract landmarks."""

import sys
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import torch
import yaml

_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from data.dataset import IMAGENET_MEAN, IMAGENET_STD
from models.unet import LandmarkUNet
from training.train import extract_landmarks_from_heatmaps


class LandmarkPredictor:
    """Load a trained model and predict landmark positions on wing images."""

    def __init__(
        self,
        checkpoint_path: Path,
        device: Optional[str] = None,
    ) -> None:
        """Load model from checkpoint."""
        self.device = torch.device(
            device
            if device
            else "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
        )

        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        cfg = checkpoint["config"]
        self.input_w = cfg["input"]["width"]
        self.input_h = cfg["input"]["height"]
        self.landmark_order: list[str] = cfg["heatmap"]["landmark_order"]
        self.geojson_to_landmark: dict[str, str] = cfg["heatmap"].get("geojson_to_landmark", {})

        self.model = LandmarkUNet(
            num_landmarks=cfg["heatmap"]["num_landmarks"],
            pretrained=False,
        )
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.to(self.device)
        self.model.eval()

    def _preprocess(self, image: np.ndarray) -> tuple[torch.Tensor, float, float]:
        """Resize and normalize image for model input."""
        orig_h, orig_w = image.shape[:2]
        scale_x = orig_w / self.input_w
        scale_y = orig_h / self.input_h

        resized = cv2.resize(image, (self.input_w, self.input_h))
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)

        img_float = rgb.astype(np.float32) / 255.0
        img_float = (img_float - IMAGENET_MEAN) / IMAGENET_STD
        tensor = torch.from_numpy(img_float.transpose(2, 0, 1)).unsqueeze(0)

        return tensor, scale_x, scale_y

    def predict(self, image: np.ndarray) -> dict:
        """Predict landmarks on a single image.

        Args:
            image: (H, W, 3) BGR uint8 numpy array

        Returns:
            Dict with landmarks (name→(x,y) in original pixels),
            confidences (name→float), heatmaps (C,H,W array).
        """
        tensor, scale_x, scale_y = self._preprocess(image)
        tensor = tensor.to(self.device)

        with torch.no_grad():
            pred = self.model(tensor)

        heatmaps = pred[0].cpu().numpy()  # (C, H, W)
        model_coords = extract_landmarks_from_heatmaps(heatmaps)

        landmarks = {}
        confidences = {}
        for i, name in enumerate(self.landmark_order):
            mx, my = model_coords[i]
            landmarks[name] = (mx * scale_x, my * scale_y)
            confidences[name] = float(heatmaps[i].max())

        return {
            "landmarks": landmarks,
            "confidences": confidences,
            "heatmaps": heatmaps,
        }

    def predict_from_path(self, image_path: Path) -> dict:
        """Load image from path and predict."""
        image = cv2.imread(str(image_path))
        if image is None:
            raise IOError(f"Failed to load image: {image_path}")
        return self.predict(image)


def predict_ensemble(
    image: np.ndarray,
    checkpoint_paths: list[Path],
    device: Optional[str] = None,
) -> dict:
    """Average heatmaps from multiple fold models before extracting coords."""
    predictors = [LandmarkPredictor(p, device) for p in checkpoint_paths]
    landmark_order = predictors[0].landmark_order

    all_heatmaps = []
    scale_x = scale_y = None

    for pred in predictors:
        tensor, sx, sy = pred._preprocess(image)
        tensor = tensor.to(pred.device)
        with torch.no_grad():
            hm = pred.model(tensor)
        all_heatmaps.append(hm[0].cpu().numpy())
        scale_x, scale_y = sx, sy

    # Average heatmaps
    avg_heatmaps = np.mean(all_heatmaps, axis=0)
    model_coords = extract_landmarks_from_heatmaps(avg_heatmaps)

    landmarks = {}
    confidences = {}
    for i, name in enumerate(landmark_order):
        mx, my = model_coords[i]
        landmarks[name] = (mx * scale_x, my * scale_y)
        confidences[name] = float(avg_heatmaps[i].max())

    return {
        "landmarks": landmarks,
        "confidences": confidences,
        "heatmaps": avg_heatmaps,
    }
