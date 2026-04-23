"""Prediction pipeline: load checkpoint, run inference, extract landmarks."""

from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import torch
from landmark_locator.models.unet import LandmarkUNet

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

DEFAULT_GATE_CONFIG = {
    "peak": {"global": 0.0, "per_landmark": {}},
    "sharpness": {"global": 0.0, "per_landmark": {}},
    "second_peak_ratio": {"global": 1.0, "per_landmark": {}},
    "second_peak_suppression_radius_px": 30,
    "core_landmarks": [],
}


class LowConfidenceLandmarkError(RuntimeError):
    """One or more core landmarks failed the confidence gate."""

    def __init__(self, failures: dict[str, str]):
        self.failures = failures
        super().__init__(
            "Core landmarks failed confidence gate: " + ", ".join(f"{k} ({v})" for k, v in failures.items())
        )


def _deep_merge(base: dict, override: dict) -> dict:
    """Return a new dict with override merged into base (recursive for dicts)."""
    out = {k: (v.copy() if isinstance(v, dict) else list(v) if isinstance(v, list) else v) for k, v in base.items()}
    for key, val in override.items():
        if isinstance(val, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], val)
        else:
            out[key] = val
    return out


def extract_landmarks_from_heatmaps(heatmaps: np.ndarray) -> list[tuple[float, float]]:
    """Extract landmark coordinates as weighted average around peak.

    Args:
        heatmaps: (C, H, W) predicted heatmap array

    Returns:
        List of (x, y) coordinates, one per channel.
    """
    coords, _ = _extract_coords_and_peaks(heatmaps)
    return coords


def _extract_coords_and_peaks(
    heatmaps: np.ndarray,
) -> tuple[list[tuple[float, float]], list[tuple[int, int]]]:
    """Weighted-centroid coords + integer peak locations per channel."""
    coords: list[tuple[float, float]] = []
    peaks: list[tuple[int, int]] = []
    for c in range(heatmaps.shape[0]):
        hm = heatmaps[c]
        peak_idx = np.unravel_index(np.argmax(hm), hm.shape)
        peak_y, peak_x = int(peak_idx[0]), int(peak_idx[1])
        peaks.append((peak_y, peak_x))

        radius = 5
        y0 = max(0, peak_y - radius)
        y1 = min(hm.shape[0], peak_y + radius + 1)
        x0 = max(0, peak_x - radius)
        x1 = min(hm.shape[1], peak_x + radius + 1)

        patch = hm[y0:y1, x0:x1]
        patch = np.maximum(patch, 0)
        total = patch.sum()

        if total > 1e-8:
            ys = np.arange(y0, y1, dtype=np.float64)
            xs = np.arange(x0, x1, dtype=np.float64)
            xx, yy = np.meshgrid(xs, ys)
            wx = (patch * xx).sum() / total
            wy = (patch * yy).sum() / total
            coords.append((float(wx), float(wy)))
        else:
            coords.append((float(peak_x), float(peak_y)))
    return coords, peaks


def compute_heatmap_metrics(
    heatmaps: np.ndarray,
    peaks: list[tuple[int, int]],
    suppression_radius_px: int,
) -> list[dict[str, float]]:
    """Per-channel peak value, sharpness, and second-peak-ratio.

    Args:
        heatmaps: (C, H, W)
        peaks: integer (row, col) of the primary peak per channel (from _extract_coords_and_peaks)
        suppression_radius_px: radius of the disk zeroed around the primary peak when searching for the second

    Returns:
        List of dicts with keys: peak, sharpness, second_peak_ratio.
    """
    metrics: list[dict[str, float]] = []
    h, w = heatmaps.shape[1], heatmaps.shape[2]
    for c in range(heatmaps.shape[0]):
        hm = heatmaps[c]
        peak_y, peak_x = peaks[c]
        peak_value = float(hm.max())

        radius = 5
        y0 = max(0, peak_y - radius)
        y1 = min(h, peak_y + radius + 1)
        x0 = max(0, peak_x - radius)
        x1 = min(w, peak_x + radius + 1)
        patch = hm[y0:y1, x0:x1]
        patch_mean = float(np.maximum(patch, 0).mean())
        sharpness = peak_value / max(patch_mean, 1e-6)

        if peak_value <= 1e-8:
            second_peak_ratio = 0.0
        else:
            ys = np.arange(h)[:, None]
            xs = np.arange(w)[None, :]
            mask = (ys - peak_y) ** 2 + (xs - peak_x) ** 2 <= suppression_radius_px * suppression_radius_px
            residual = np.where(mask, -np.inf, hm)
            second_peak = float(residual.max()) if np.isfinite(residual).any() else 0.0
            second_peak = max(second_peak, 0.0)
            second_peak_ratio = second_peak / peak_value

        metrics.append(
            {
                "peak": peak_value,
                "sharpness": float(sharpness),
                "second_peak_ratio": float(second_peak_ratio),
            }
        )
    return metrics


def _gate_landmark(
    name: str,
    metric: dict[str, float],
    gate_cfg: dict,
) -> tuple[bool, str]:
    """Return (passed, reason). reason is empty when passed."""
    peak_thr = gate_cfg["peak"]["per_landmark"].get(name, gate_cfg["peak"]["global"])
    sharp_thr = gate_cfg["sharpness"]["per_landmark"].get(name, gate_cfg["sharpness"]["global"])
    sp_thr = gate_cfg["second_peak_ratio"]["per_landmark"].get(name, gate_cfg["second_peak_ratio"]["global"])

    if metric["peak"] < peak_thr:
        return False, f"peak={metric['peak']:.3f}<{peak_thr:.3f}"
    if metric["sharpness"] < sharp_thr:
        return False, f"sharpness={metric['sharpness']:.2f}<{sharp_thr:.2f}"
    if metric["second_peak_ratio"] > sp_thr:
        return False, f"second_peak={metric['second_peak_ratio']:.2f}>{sp_thr:.2f}"
    return True, ""


class LandmarkPredictor:
    """Load a trained model and predict landmark positions on wing images."""

    def __init__(
        self,
        checkpoint_path: Path,
        device: Optional[str] = None,
        confidence_override: Optional[dict] = None,
    ) -> None:
        """Load model from checkpoint.

        Args:
            checkpoint_path: path to .pt file.
            device: "mps" | "cuda" | "cpu". Auto-detected when None.
            confidence_override: dict with the same shape as the config `confidence:` block.
                Deep-merged over the checkpoint's embedded gate config (checkpoint file is not mutated).
        """
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

        base_gate = _deep_merge(DEFAULT_GATE_CONFIG, cfg.get("confidence", {}) or {})
        self.gate_config = _deep_merge(base_gate, confidence_override or {})

        self.model = LandmarkUNet(
            num_landmarks=cfg["heatmap"]["num_landmarks"],
            pretrained=False,
        )
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.to(self.device)
        self.model.eval()

    def update_gate_config(self, override: dict) -> None:
        """Deep-merge override into the active gate config (no checkpoint mutation)."""
        self.gate_config = _deep_merge(self.gate_config, override)

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

    def predict(self, image: np.ndarray, *, include_unreliable: bool = False) -> dict:
        """Predict landmarks on a single image, applying the confidence gate.

        Args:
            image: (H, W, 3) BGR uint8 numpy array.
            include_unreliable: when True, landmarks that fail the gate are still
                returned in the `landmarks` dict (downstream should check `reliable`).

        Returns:
            Dict with landmarks, confidences, sharpness, second_peak_ratio, reliable,
            gate_reason, heatmaps.

        Raises:
            LowConfidenceLandmarkError: any core landmark failed the gate.
        """
        tensor, scale_x, scale_y = self._preprocess(image)
        tensor = tensor.to(self.device)

        with torch.no_grad():
            pred = self.model(tensor)

        heatmaps = pred[0].cpu().numpy()
        return self._assemble_result(heatmaps, scale_x, scale_y, include_unreliable=include_unreliable)

    def predict_from_path(self, image_path: Path, *, include_unreliable: bool = False) -> dict:
        """Load image from path and predict."""
        from landmark_locator.data.psd_loader import imread_any

        image = imread_any(image_path)
        if image is None:
            raise IOError(f"Failed to load image: {image_path}")
        return self.predict(image, include_unreliable=include_unreliable)

    def _assemble_result(
        self,
        heatmaps: np.ndarray,
        scale_x: float,
        scale_y: float,
        *,
        include_unreliable: bool,
    ) -> dict:
        return _assemble_gate_result(
            heatmaps,
            scale_x,
            scale_y,
            landmark_order=self.landmark_order,
            gate_config=self.gate_config,
            include_unreliable=include_unreliable,
        )


def _assemble_gate_result(
    heatmaps: np.ndarray,
    scale_x: float,
    scale_y: float,
    *,
    landmark_order: list[str],
    gate_config: dict,
    include_unreliable: bool,
) -> dict:
    coords, peaks = _extract_coords_and_peaks(heatmaps)
    metrics = compute_heatmap_metrics(
        heatmaps,
        peaks,
        suppression_radius_px=int(gate_config.get("second_peak_suppression_radius_px", 30)),
    )

    landmarks: dict[str, tuple[float, float]] = {}
    confidences: dict[str, float] = {}
    sharpness: dict[str, float] = {}
    second_peak_ratio: dict[str, float] = {}
    reliable: dict[str, bool] = {}
    gate_reason: dict[str, str] = {}

    for i, name in enumerate(landmark_order):
        mx, my = coords[i]
        m = metrics[i]
        confidences[name] = m["peak"]
        sharpness[name] = m["sharpness"]
        second_peak_ratio[name] = m["second_peak_ratio"]
        passed, reason = _gate_landmark(name, m, gate_config)
        reliable[name] = passed
        gate_reason[name] = reason
        if passed or include_unreliable:
            landmarks[name] = (mx * scale_x, my * scale_y)

    core = set(gate_config.get("core_landmarks", []) or [])
    core_failures = {n: gate_reason[n] for n in landmark_order if n in core and not reliable[n]}
    for missing in core - set(landmark_order):
        core_failures[missing] = "not predicted by model"
    if core_failures:
        raise LowConfidenceLandmarkError(core_failures)

    return {
        "landmarks": landmarks,
        "confidences": confidences,
        "sharpness": sharpness,
        "second_peak_ratio": second_peak_ratio,
        "reliable": reliable,
        "gate_reason": gate_reason,
        "heatmaps": heatmaps,
    }


def predict_ensemble(
    image: np.ndarray,
    checkpoint_paths: list[Path],
    device: Optional[str] = None,
    confidence_override: Optional[dict] = None,
    *,
    include_unreliable: bool = False,
) -> dict:
    """Average heatmaps from multiple fold models before extracting coords."""
    predictors = [LandmarkPredictor(p, device, confidence_override=confidence_override) for p in checkpoint_paths]
    landmark_order = predictors[0].landmark_order
    gate_config = predictors[0].gate_config

    all_heatmaps = []
    scale_x = scale_y = None

    for pred in predictors:
        tensor, sx, sy = pred._preprocess(image)
        tensor = tensor.to(pred.device)
        with torch.no_grad():
            hm = pred.model(tensor)
        all_heatmaps.append(hm[0].cpu().numpy())
        scale_x, scale_y = sx, sy

    avg_heatmaps = np.mean(all_heatmaps, axis=0)
    return _assemble_gate_result(
        avg_heatmaps,
        scale_x,
        scale_y,
        landmark_order=landmark_order,
        gate_config=gate_config,
        include_unreliable=include_unreliable,
    )
