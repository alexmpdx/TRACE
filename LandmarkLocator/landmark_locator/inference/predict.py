"""Prediction pipeline: load checkpoint, run inference, extract landmarks."""

from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import torch
import yaml

from landmark_locator.models.unet import LandmarkUNet

SIDECAR_GATE_FILENAME = "gate_config.yaml"

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

DEFAULT_GATE_CONFIG = {
    "peak": {"global": 0.0, "per_landmark": {}},
    "sharpness": {"global": 0.0, "per_landmark": {}},
    "second_peak_ratio": {"global": 1.0, "per_landmark": {}},
    "second_peak_suppression_radius_px": 30,
    "core_landmarks": [],
    # Number of metric gates (peak / sharpness / second_peak_ratio) that must
    # fail before a landmark is marked unreliable. 1 = current behavior (any
    # single failure rejects). Disabled gates (`enabled: false`) never count.
    "min_metric_failures_to_reject": 1,
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


def canonical_sidecar_gate_path(checkpoint_path: Path) -> Path:
    """Where the sidecar gate YAML *should* live for a given checkpoint or model dir.

    Flat-layout convention: sidecar lives alongside the .pt files in the model folder.
    Returns a path regardless of whether the file exists yet (use this for writing).
    """
    p = Path(checkpoint_path).resolve()
    return (p if p.is_dir() else p.parent) / SIDECAR_GATE_FILENAME


def find_sidecar_gate_path(checkpoint_path: Path) -> Optional[Path]:
    """Locate the existing `gate_config.yaml` sidecar for a checkpoint, or None.

    Looks alongside the .pt file (flat layout). Also probes one directory up as a
    back-compat for legacy nested layouts where checkpoints lived in a sub-folder
    like `<model>/checkpoints/best_fold0.pt` and the sidecar at `<model>/gate_config.yaml`.
    """
    canonical = canonical_sidecar_gate_path(checkpoint_path)
    if canonical.is_file():
        return canonical
    legacy = canonical.parent.parent / SIDECAR_GATE_FILENAME
    if legacy.is_file():
        return legacy
    return None


def _load_sidecar_gate(checkpoint_path: Path) -> tuple[Optional[Path], Optional[dict]]:
    """Read the sidecar YAML (if present) and return (path, confidence-block)."""
    sidecar_path = find_sidecar_gate_path(checkpoint_path)
    if sidecar_path is None:
        return None, None
    try:
        data = yaml.safe_load(sidecar_path.read_text()) or {}
    except Exception:
        return sidecar_path, None
    # Accept both formats: a full doc with `confidence:` block, or just the block itself.
    return sidecar_path, data.get("confidence", data)


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
    """Return (passed, reason). reason is empty when passed.

    Evaluates all three metric gates (peak / sharpness / second_peak_ratio)
    and rejects the landmark only when the number of failures meets the
    `min_metric_failures_to_reject` threshold (default 1). Disabled gates
    (`enabled: false`) never contribute a failure. When multiple gates fail,
    the reason string joins them with "; " so the log shows the full picture.
    """
    peak_cfg = gate_cfg["peak"]
    sharp_cfg = gate_cfg["sharpness"]
    sp_cfg = gate_cfg["second_peak_ratio"]
    reasons: list[str] = []

    if peak_cfg.get("enabled", True):
        peak_thr = peak_cfg["per_landmark"].get(name, peak_cfg["global"])
        if metric["peak"] < peak_thr:
            reasons.append(f"peak={metric['peak']:.3f}<{peak_thr:.3f}")
    if sharp_cfg.get("enabled", True):
        sharp_thr = sharp_cfg["per_landmark"].get(name, sharp_cfg["global"])
        if metric["sharpness"] < sharp_thr:
            reasons.append(f"sharpness={metric['sharpness']:.2f}<{sharp_thr:.2f}")
    if sp_cfg.get("enabled", True):
        sp_thr = sp_cfg["per_landmark"].get(name, sp_cfg["global"])
        if metric["second_peak_ratio"] > sp_thr:
            reasons.append(f"second_peak={metric['second_peak_ratio']:.2f}>{sp_thr:.2f}")

    min_fails = max(1, int(gate_cfg.get("min_metric_failures_to_reject", 1)))
    if len(reasons) >= min_fails:
        return False, "; ".join(reasons)
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

        self.checkpoint_path = Path(checkpoint_path)
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        cfg = checkpoint["config"]
        self.input_w = cfg["input"]["width"]
        self.input_h = cfg["input"]["height"]
        self.landmark_order: list[str] = cfg["heatmap"]["landmark_order"]
        self.geojson_to_landmark: dict[str, str] = cfg["heatmap"].get("geojson_to_landmark", {})

        # Gate-config precedence (lowest → highest):
        #   DEFAULT_GATE_CONFIG  ←  sidecar gate_config.yaml  ←  runtime override
        # The checkpoint's embedded `config.confidence` block is intentionally
        # ignored — gate thresholds are now sourced exclusively from the
        # sidecar YAML so they can be tuned without retraining.
        base_gate = dict(DEFAULT_GATE_CONFIG)
        self.sidecar_gate_path, sidecar_gate = _load_sidecar_gate(checkpoint_path)
        if sidecar_gate:
            base_gate = _deep_merge(base_gate, sidecar_gate)
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

    def predict_batch(
        self,
        images: list[np.ndarray],
        *,
        include_unreliable: bool = False,
        raise_on_core_fail: bool = False,
    ) -> list[dict]:
        """Run a single forward pass over a batch of images.

        Returns a list of result dicts with the same keys as `predict()`, plus an
        additional `error` field that is None on success or a LowConfidenceLandmarkError
        when the per-image gate would have aborted. Set `raise_on_core_fail=True` to
        get the same raising behavior as `predict()` (first failure aborts the batch).
        """
        if not images:
            return []
        tensors: list[torch.Tensor] = []
        scales: list[tuple[float, float]] = []
        for img in images:
            t, sx, sy = self._preprocess(img)
            tensors.append(t)
            scales.append((sx, sy))
        batch = torch.cat(tensors, dim=0).to(self.device)
        with torch.no_grad():
            out = self.model(batch).cpu().numpy()  # (B, C, H, W)
        return _assemble_batch_results(
            out,
            scales,
            landmark_order=self.landmark_order,
            gate_config=self.gate_config,
            include_unreliable=include_unreliable,
            raise_on_core_fail=raise_on_core_fail,
        )

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


def _find_fold_checkpoints(folder: Path) -> list[Path]:
    """Return `best_fold<N>.pt` checkpoints in `folder`, one per fold (lowest suffix wins)."""
    candidates = sorted(folder.glob("best_fold*.pt"))
    by_fold: dict[int, Path] = {}
    for p in candidates:
        rest = p.stem.replace("best_fold", "")
        try:
            fold = int(rest.split("_")[0])
        except ValueError:
            continue
        if fold not in by_fold:
            by_fold[fold] = p
    return [by_fold[f] for f in sorted(by_fold)]


class EnsemblePredictor:
    """Drop-in replacement for LandmarkPredictor that averages K fold heatmaps.

    Same attributes (landmark_order, geojson_to_landmark, gate_config) and same
    predict()/predict_from_path() contract so callers don't have to branch.
    """

    def __init__(
        self,
        checkpoint_paths: list[Path],
        device: Optional[str] = None,
        confidence_override: Optional[dict] = None,
    ) -> None:
        if not checkpoint_paths:
            raise ValueError("EnsemblePredictor requires at least one checkpoint")
        self._predictors = [
            LandmarkPredictor(p, device, confidence_override=confidence_override) for p in checkpoint_paths
        ]
        first = self._predictors[0]
        self.landmark_order = first.landmark_order
        self.geojson_to_landmark = first.geojson_to_landmark
        self.gate_config = first.gate_config
        self.sidecar_gate_path = first.sidecar_gate_path
        self.checkpoint_paths = list(checkpoint_paths)
        self.device = first.device

    def update_gate_config(self, override: dict) -> None:
        """Deep-merge override into every fold predictor's gate config."""
        self.gate_config = _deep_merge(self.gate_config, override)
        for pred in self._predictors:
            pred.gate_config = self.gate_config

    def predict(self, image: np.ndarray, *, include_unreliable: bool = False) -> dict:
        all_heatmaps = []
        scale_x = scale_y = None
        for pred in self._predictors:
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
            landmark_order=self.landmark_order,
            gate_config=self.gate_config,
            include_unreliable=include_unreliable,
        )

    def predict_from_path(self, image_path: Path, *, include_unreliable: bool = False) -> dict:
        from landmark_locator.data.psd_loader import imread_any

        image = imread_any(image_path)
        if image is None:
            raise IOError(f"Failed to load image: {image_path}")
        return self.predict(image, include_unreliable=include_unreliable)

    def predict_batch(
        self,
        images: list[np.ndarray],
        *,
        include_unreliable: bool = False,
        raise_on_core_fail: bool = False,
    ) -> list[dict]:
        """Batched ensemble prediction: K folds × B images in K forward passes."""
        if not images:
            return []
        first = self._predictors[0]
        tensors: list[torch.Tensor] = []
        scales: list[tuple[float, float]] = []
        for img in images:
            t, sx, sy = first._preprocess(img)
            tensors.append(t)
            scales.append((sx, sy))
        batch_cpu = torch.cat(tensors, dim=0)

        summed: Optional[np.ndarray] = None
        for pred in self._predictors:
            batch = batch_cpu.to(pred.device)
            with torch.no_grad():
                out = pred.model(batch).cpu().numpy()
            summed = out if summed is None else summed + out
        avg = summed / float(len(self._predictors))

        return _assemble_batch_results(
            avg,
            scales,
            landmark_order=self.landmark_order,
            gate_config=self.gate_config,
            include_unreliable=include_unreliable,
            raise_on_core_fail=raise_on_core_fail,
        )


def _assemble_batch_results(
    heatmap_batch: np.ndarray,
    scales: list[tuple[float, float]],
    *,
    landmark_order: list[str],
    gate_config: dict,
    include_unreliable: bool,
    raise_on_core_fail: bool,
) -> list[dict]:
    """Assemble per-image gate results from a (B, C, H, W) heatmap batch.

    Each result has the same shape as `LandmarkPredictor.predict()` plus an `error`
    field that is None on success or a `LowConfidenceLandmarkError` instance when
    `raise_on_core_fail` is False and a core landmark would have aborted the image.
    """
    results: list[dict] = []
    for i, (sx, sy) in enumerate(scales):
        try:
            r = _assemble_gate_result(
                heatmap_batch[i],
                sx,
                sy,
                landmark_order=landmark_order,
                gate_config=gate_config,
                include_unreliable=include_unreliable,
            )
            r["error"] = None
            results.append(r)
        except LowConfidenceLandmarkError as exc:
            if raise_on_core_fail:
                raise
            results.append(
                {
                    "landmarks": {},
                    "confidences": {},
                    "sharpness": {},
                    "second_peak_ratio": {},
                    "reliable": {},
                    "gate_reason": {},
                    "heatmaps": heatmap_batch[i],
                    "error": exc,
                }
            )
    return results


def auto_batch_size(num_images: int, *, max_cap: int = 16) -> int:
    """Heuristic batch size based on available memory and image count.

    Estimates ~12 MB of working memory per image at the model's input resolution
    (input tensor + heatmap output) and uses up to 25% of available RAM. Falls back
    to a fixed cap when psutil isn't installed.
    """
    if num_images <= 1:
        return 1
    bytes_per_image = 12_000_000
    try:
        import psutil

        available = psutil.virtual_memory().available
        max_by_mem = max(1, int(available * 0.25 / bytes_per_image))
    except ImportError:
        max_by_mem = max_cap
    return max(1, min(num_images, max_by_mem, max_cap))


def make_predictor(
    path: Path,
    device: Optional[str] = None,
    confidence_override: Optional[dict] = None,
):
    """Factory: return LandmarkPredictor for a .pt file, EnsemblePredictor for a fold-folder.

    A "fold folder" is any directory containing `best_fold*.pt` files.
    """
    path = Path(path)
    if path.is_dir():
        ckpts = _find_fold_checkpoints(path)
        if not ckpts:
            raise FileNotFoundError(f"No best_fold*.pt in {path}")
        return EnsemblePredictor(ckpts, device=device, confidence_override=confidence_override)
    return LandmarkPredictor(path, device=device, confidence_override=confidence_override)
