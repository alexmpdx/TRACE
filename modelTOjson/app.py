"""
Semantic segmentation inference GUI.

Loads a trained segmentation model (from QuPath extension format) and runs
inference on a folder of images, displaying results with adjustable overlay
and exporting detections as GeoJSON.

Supported model architectures:
  - SMP-based (unet, unet++, deeplabv3, deeplabv3+, fpn, manet, linknet,
    pspnet, pan) with any SMP-compatible backbone (resnet34/50, efficientnet-b2,
    mobilenet_v2, etc.)
  - Full PyTorch models saved as .pt (e.g. custom ViT / MuViT architectures)
  - ONNX models via onnxruntime (custom_onnx)
"""

import json
import os
import sys
import traceback
from abc import ABC, abstractmethod
from pathlib import Path

import geojson
import numpy as np
import rasterio
import shapely.geometry
import tifffile
import torch
import torch.nn as nn
from PIL import Image
from PyQt5.QtCore import QThread, pyqtSignal, Qt, QRectF
from PyQt5.QtGui import QImage, QPixmap, QColor, QPainter
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QFileDialog, QLabel, QSlider, QProgressBar,
    QListWidget, QSplitter, QGroupBox, QGraphicsView, QGraphicsScene,
    QGraphicsPixmapItem, QStatusBar, QMessageBox, QComboBox, QCheckBox,
)


# ---------------------------------------------------------------------------
# BatchRenorm (drop-in replacement for BatchNorm used by QuPath extension)
# ---------------------------------------------------------------------------
class BatchRenorm(nn.Module):
    def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=True):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.affine = affine
        if affine:
            self.weight = nn.Parameter(torch.ones(num_features))
            self.bias = nn.Parameter(torch.zeros(num_features))
        else:
            self.register_parameter("weight", None)
            self.register_parameter("bias", None)
        self.register_buffer("running_mean", torch.zeros(num_features))
        self.register_buffer("running_var", torch.ones(num_features))
        self.register_buffer("num_batches_tracked", torch.tensor(0, dtype=torch.long))

    def forward(self, x):
        # During eval, use running stats (same as BatchNorm)
        if not self.training:
            mean = self.running_mean
            var = self.running_var
        else:
            dims = [0] + list(range(2, x.dim()))
            mean = x.mean(dims)
            var = x.var(dims, unbiased=False)
            with torch.no_grad():
                self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * mean
                self.running_var = (1 - self.momentum) * self.running_var + self.momentum * var
                self.num_batches_tracked += 1

        shape = [1, self.num_features] + [1] * (x.dim() - 2)
        x = (x - mean.view(shape)) / (var.view(shape) + self.eps).sqrt()
        if self.affine:
            x = x * self.weight.view(shape) + self.bias.view(shape)
        return x


def _replace_bn_with_batchrenorm(module):
    """Recursively replace all BatchNorm2d layers with BatchRenorm."""
    for name, child in module.named_children():
        if isinstance(child, nn.BatchNorm2d):
            br = BatchRenorm(
                child.num_features, eps=child.eps,
                momentum=child.momentum, affine=child.affine,
            )
            setattr(module, name, br)
        else:
            _replace_bn_with_batchrenorm(child)


# ---------------------------------------------------------------------------
# Model wrapper abstraction
# ---------------------------------------------------------------------------
class ModelWrapper(ABC):
    """Unified inference interface for all model backends."""

    @abstractmethod
    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        """Run forward pass. Input/output are (N, C, H, W) float tensors."""

    @abstractmethod
    def to(self, device: torch.device) -> "ModelWrapper":
        ...

    def eval(self) -> "ModelWrapper":
        return self


class TorchModelWrapper(ModelWrapper):
    """Wraps any torch.nn.Module."""

    def __init__(self, model: nn.Module):
        self.model = model

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        return self.model(tensor)

    def to(self, device: torch.device) -> "TorchModelWrapper":
        self.model.to(device)
        return self

    def eval(self) -> "TorchModelWrapper":
        self.model.eval()
        return self


class OnnxModelWrapper(ModelWrapper):
    """Wraps an ONNX model via onnxruntime."""

    def __init__(self, onnx_path: str, device: torch.device):
        try:
            import onnxruntime as ort
        except ImportError:
            raise ImportError(
                "onnxruntime is required for ONNX models. "
                "Install it with: pip install onnxruntime  (or onnxruntime-gpu)"
            )
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        if device.type == "cpu":
            providers = ["CPUExecutionProvider"]
        self.session = ort.InferenceSession(str(onnx_path), providers=providers)
        self.input_name = self.session.get_inputs()[0].name
        self.device = device

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        np_input = tensor.cpu().numpy()
        outputs = self.session.run(None, {self.input_name: np_input})
        return torch.from_numpy(outputs[0]).to(self.device)

    def to(self, device: torch.device) -> "OnnxModelWrapper":
        self.device = device
        return self


# ---------------------------------------------------------------------------
# SMP architecture registry
# ---------------------------------------------------------------------------
def _get_smp_model_class(arch_type: str):
    """Map architecture type string to segmentation_models_pytorch class."""
    import segmentation_models_pytorch as smp

    REGISTRY = {
        "unet": smp.Unet,
        "unetplusplus": smp.UnetPlusPlus,
        "unet++": smp.UnetPlusPlus,
        "deeplabv3": smp.DeepLabV3,
        "deeplabv3+": smp.DeepLabV3Plus,
        "deeplabv3plus": smp.DeepLabV3Plus,
        "fpn": smp.FPN,
        "manet": smp.MAnet,
        "linknet": smp.Linknet,
        "pspnet": smp.PSPNet,
        "pan": smp.PAN,
    }
    return REGISTRY.get(arch_type.lower().replace("_", "").replace("-", ""))


def _build_smp_model(arch: dict, num_classes: int) -> nn.Module:
    """Construct an SMP model from metadata architecture dict."""
    import segmentation_models_pytorch as smp

    arch_type = arch.get("type", "unet")
    model_cls = _get_smp_model_class(arch_type)
    if model_cls is None:
        raise ValueError(
            f"Unknown SMP architecture '{arch_type}'. "
            f"Supported: unet, unet++, deeplabv3, deeplabv3+, fpn, manet, "
            f"linknet, pspnet, pan"
        )

    backbone = arch.get("backbone", "resnet34")
    in_channels = int(arch.get("input_channels", 3))

    # Build kwargs — not all SMP architectures accept the same params
    kwargs = dict(
        encoder_name=backbone,
        encoder_weights=None,  # we load our own weights
        in_channels=in_channels,
        classes=num_classes,
    )

    # encoder_depth is supported by most SMP architectures
    if "encoder_depth" in arch:
        kwargs["encoder_depth"] = int(arch["encoder_depth"])

    # decoder_channels is only used by Unet / UnetPlusPlus / MAnet
    if "decoder_channels" in arch and model_cls in (
        smp.Unet, smp.UnetPlusPlus, smp.MAnet
    ):
        kwargs["decoder_channels"] = [int(c) for c in arch["decoder_channels"]]

    return model_cls(**kwargs)


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------
def _find_model_file(model_dir: Path):
    """
    Locate the model weights file in the directory.

    Returns (path, format) where format is 'pt', 'onnx', or 'checkpoint'.
    """
    # Prefer model.pt, then .onnx, then any checkpoint_*.pt
    if (model_dir / "model.pt").exists():
        return model_dir / "model.pt", "pt"
    if (model_dir / "model.onnx").exists():
        return model_dir / "model.onnx", "onnx"

    # Search for .onnx files
    onnx_files = list(model_dir.glob("*.onnx"))
    if onnx_files:
        return onnx_files[0], "onnx"

    # Fall back to checkpoint files
    checkpoints = sorted(model_dir.glob("checkpoint_*.pt"))
    if checkpoints:
        return checkpoints[-1], "pt"

    raise FileNotFoundError(
        f"No model file found in {model_dir}. "
        f"Expected model.pt, model.onnx, or checkpoint_*.pt"
    )


def _load_torch_state(path: Path, device: torch.device):
    """Load a .pt file and return (state_dict_or_model, is_full_model)."""
    loaded = torch.load(path, map_location=device, weights_only=False)

    # Unwrap common checkpoint wrappers
    if isinstance(loaded, dict):
        for key in ("model_state_dict", "state_dict", "model"):
            if key in loaded and isinstance(loaded[key], dict):
                return loaded[key], False
        # Check if it looks like a raw state dict (keys are parameter names)
        sample_keys = list(loaded.keys())[:5]
        if sample_keys and all(
            isinstance(k, str) and ("." in k or "weight" in k or "bias" in k)
            for k in sample_keys
        ):
            return loaded, False
        # Dict but doesn't look like a state dict — could be a full model
        return loaded, False

    # If torch.load returned an nn.Module directly
    if isinstance(loaded, nn.Module):
        return loaded, True

    raise ValueError(
        f"Cannot interpret {path.name}: expected state dict or nn.Module, "
        f"got {type(loaded).__name__}"
    )


def load_model(model_dir: str, device: torch.device):
    """
    Load a trained model from a model directory containing metadata.json.

    Supports:
      - SMP architectures (unet, unet++, deeplabv3, fpn, etc.) with any
        SMP-compatible backbone
      - Full PyTorch models saved with torch.save (custom architectures like
        muvit, vision transformers, etc.)
      - ONNX models via onnxruntime
    """
    model_dir = Path(model_dir)
    with open(model_dir / "metadata.json") as f:
        metadata = json.load(f)

    arch = metadata["architecture"]
    classes = metadata["classes"]
    num_classes = len(classes)
    arch_type = arch.get("type", "unet").lower().strip()

    model_path, model_format = _find_model_file(model_dir)

    # ----- ONNX path -----
    if model_format == "onnx" or arch_type == "custom_onnx":
        onnx_path = model_path
        if model_format != "onnx":
            # arch says custom_onnx but we found a .pt — look harder for .onnx
            onnx_files = list(model_dir.glob("*.onnx"))
            if not onnx_files:
                raise FileNotFoundError(
                    f"Architecture is 'custom_onnx' but no .onnx file found in {model_dir}"
                )
            onnx_path = onnx_files[0]
        wrapper = OnnxModelWrapper(str(onnx_path), device)
        return wrapper, metadata

    # ----- PyTorch path -----
    loaded, is_full_model = _load_torch_state(model_path, device)

    if is_full_model:
        # The .pt contained a complete nn.Module (custom arch like muvit)
        model = loaded
    else:
        # We have a state dict — try to build the architecture
        smp_cls = _get_smp_model_class(arch_type)

        if smp_cls is not None:
            # Known SMP architecture
            model = _build_smp_model(arch, num_classes)
            if arch.get("use_batchrenorm", False):
                _replace_bn_with_batchrenorm(model)
            model.load_state_dict(loaded, strict=False)
        else:
            # Unknown architecture with only a state dict — cannot rebuild
            raise ValueError(
                f"Architecture '{arch_type}' is not a known SMP type and "
                f"the model file contains a state dict (not a full model). "
                f"Cannot reconstruct the model architecture.\n\n"
                f"Options:\n"
                f"  1. Save the model as a full Module: "
                f"torch.save(model, 'model.pt')\n"
                f"  2. Export to ONNX and set architecture.type = 'custom_onnx'\n"
                f"  3. Use a supported SMP architecture type"
            )

    wrapper = TorchModelWrapper(model)
    wrapper.to(device)
    wrapper.eval()
    return wrapper, metadata


# ---------------------------------------------------------------------------
# Image I/O
# ---------------------------------------------------------------------------
SUPPORTED_EXTENSIONS = {".tif", ".tiff", ".bmp", ".png", ".jpg", ".jpeg"}


def read_image(path: str) -> np.ndarray:
    """Read an image file and return as RGB uint8 numpy array (H, W, 3).

    Tries rasterio first (handles GeoTIFF + many GDAL formats), then
    tifffile, then PIL as a universal fallback.
    """
    path = str(path)
    img = None

    # Attempt 1: rasterio (best for GeoTIFF / GDAL-supported formats)
    try:
        with rasterio.open(path) as src:
            bands = src.read()  # (C, H, W)
            if bands.shape[0] >= 3:
                img = np.stack([bands[0], bands[1], bands[2]], axis=-1)
            elif bands.shape[0] == 1:
                img = np.stack([bands[0]] * 3, axis=-1)
            else:
                img = np.stack([bands[0], bands[1], bands[0]], axis=-1)
    except Exception:
        pass

    # Attempt 2: tifffile (handles exotic TIFF variants)
    if img is None:
        try:
            img = tifffile.imread(path)
            if img.ndim == 2:
                img = np.stack([img] * 3, axis=-1)
            elif img.ndim == 3 and img.shape[0] <= 4:
                img = np.moveaxis(img, 0, -1)
            if img.shape[-1] > 3:
                img = img[..., :3]
        except Exception:
            pass

    # Attempt 3: PIL (universal fallback — BMP, PNG, JPEG, etc.)
    if img is None:
        try:
            pil = Image.open(path).convert("RGB")
            img = np.array(pil)
        except Exception as e:
            raise IOError(
                f"Cannot read image: {os.path.basename(path)}\n"
                f"Tried rasterio, tifffile, and PIL — all failed.\n"
                f"Last error: {e}"
            )

    # Convert to uint8 if needed
    if img.dtype != np.uint8:
        if img.max() > 255:
            img = (img / img.max() * 255).astype(np.uint8)
        else:
            img = img.astype(np.uint8)
    return img


def get_geotransform(path: str):
    """Try to read an affine geotransform and CRS from a raster file."""
    try:
        with rasterio.open(path) as src:
            if src.crs is not None:
                return src.transform, src.crs
    except Exception:
        pass
    return None, None


# ---------------------------------------------------------------------------
# Preprocessing & inference
# ---------------------------------------------------------------------------
def preprocess_tile(tile: np.ndarray, norm_stats: list) -> torch.Tensor:
    """Normalize a tile using percentile-99 clipping and convert to tensor."""
    tile = tile.astype(np.float32)
    for c in range(tile.shape[-1]):
        p99 = norm_stats[c]["p99"]
        p1 = norm_stats[c]["p1"]
        if p99 > p1:
            tile[..., c] = (tile[..., c] - p1) / (p99 - p1)
        else:
            tile[..., c] = 0.0
    tile = np.clip(tile, 0.0, 1.0)
    # HWC -> CHW
    return torch.from_numpy(tile.transpose(2, 0, 1))


def _compute_effective_padding(tile_size: int) -> int:
    """Compute padding per side for center-crop tiling.

    Matches QuPath extension: min 25% per side, max 3/8, floor 64px.
    """
    min_pad = tile_size // 4       # 25% per side
    max_pad = 3 * tile_size // 8   # 37.5% per side
    return max(64, min(min_pad, max_pad))


def run_inference(model, image: np.ndarray, metadata: dict, device: torch.device,
                  progress_callback=None, smoothing_sigma: float = 2.0) -> np.ndarray:
    """
    Run tiled inference using expanded reads + center-crop stitching.

    Each tile is read with extra context (real neighboring pixels where
    available, reflection padding at image edges). Only the center portion
    of each tile's prediction is kept — the halo is discarded. This
    eliminates tile boundary artifacts regardless of model normalization
    type (see Buglakova et al., ICCV 2025).

    After stitching, Gaussian smoothing is applied to the probability maps
    before argmax to reduce per-pixel noise.

    Returns predicted class indices as (H, W) int array at original resolution.
    """
    from scipy.ndimage import gaussian_filter

    arch = metadata["architecture"]
    tile_size = int(arch["input_width"])  # 512
    downsample = int(arch["downsample"])  # 2
    norm_stats = metadata["normalization_stats"]
    num_classes = len(metadata["classes"])

    h_orig, w_orig = image.shape[:2]

    # Downsample
    if downsample > 1:
        h_ds = h_orig // downsample
        w_ds = w_orig // downsample
        img_ds = np.array(Image.fromarray(image).resize((w_ds, h_ds), Image.BILINEAR))
    else:
        h_ds, w_ds = h_orig, w_orig
        img_ds = image

    # Center-crop tiling: each tile is tile_size, but we only keep the
    # center (stride x stride) region. The padding on each side provides
    # real context so the model sees actual neighbors, not artifacts.
    input_padding = _compute_effective_padding(tile_size)
    stride = tile_size - 2 * input_padding

    tiles_y = max(1, (h_ds + stride - 1) // stride)
    tiles_x = max(1, (w_ds + stride - 1) // stride)
    total_tiles = tiles_y * tiles_x

    # Output probability map — each pixel written exactly once (no blending)
    prob_map = np.zeros((num_classes, h_ds, w_ds), dtype=np.float32)

    tile_idx = 0
    with torch.no_grad():
        for ty in range(tiles_y):
            for tx in range(tiles_x):
                # Output region for this tile (stride-sized)
                out_y0 = ty * stride
                out_x0 = tx * stride
                out_y1 = min(out_y0 + stride, h_ds)
                out_x1 = min(out_x0 + stride, w_ds)
                out_h = out_y1 - out_y0
                out_w = out_x1 - out_x0

                # Input region: expand by input_padding on each side
                in_y0 = out_y0 - input_padding
                in_x0 = out_x0 - input_padding
                in_y1 = in_y0 + tile_size
                in_x1 = in_x0 + tile_size

                # Clamp to image bounds; compute reflection padding needed
                pad_top = max(0, -in_y0)
                pad_left = max(0, -in_x0)
                pad_bottom = max(0, in_y1 - h_ds)
                pad_right = max(0, in_x1 - w_ds)

                read_y0 = max(0, in_y0)
                read_x0 = max(0, in_x0)
                read_y1 = min(h_ds, in_y1)
                read_x1 = min(w_ds, in_x1)

                tile = img_ds[read_y0:read_y1, read_x0:read_x1]

                # Reflection-pad at image edges (interior tiles need none)
                if pad_top > 0 or pad_left > 0 or pad_bottom > 0 or pad_right > 0:
                    tile = np.pad(
                        tile,
                        ((pad_top, pad_bottom), (pad_left, pad_right), (0, 0)),
                        mode="reflect",
                    )

                # Safety: ensure exactly tile_size x tile_size
                if tile.shape[0] < tile_size or tile.shape[1] < tile_size:
                    extra_h = tile_size - tile.shape[0]
                    extra_w = tile_size - tile.shape[1]
                    tile = np.pad(
                        tile, ((0, extra_h), (0, extra_w), (0, 0)), mode="reflect"
                    )

                tensor = preprocess_tile(tile, norm_stats).unsqueeze(0).to(device)
                output = model(tensor)  # (1, C, H, W)
                if not isinstance(output, torch.Tensor):
                    output = torch.from_numpy(np.asarray(output))
                probs = torch.softmax(output, dim=1).cpu().numpy()[0]

                # Center-crop: discard halo, keep only the reliable center
                center = probs[
                    :,
                    input_padding : input_padding + out_h,
                    input_padding : input_padding + out_w,
                ]
                prob_map[:, out_y0:out_y1, out_x0:out_x1] = center

                tile_idx += 1
                if progress_callback:
                    progress_callback(tile_idx, total_tiles)

    # Gaussian probability smoothing before argmax (reduces per-pixel noise)
    if smoothing_sigma > 0:
        for c in range(num_classes):
            prob_map[c] = gaussian_filter(prob_map[c], sigma=smoothing_sigma)

    pred_classes = prob_map.argmax(axis=0).astype(np.uint8)

    # Upsample back to original resolution
    if downsample > 1:
        pred_classes = np.array(
            Image.fromarray(pred_classes).resize((w_orig, h_orig), Image.NEAREST)
        )

    return pred_classes


# ---------------------------------------------------------------------------
# GeoJSON export
# ---------------------------------------------------------------------------
def mask_to_geojson(mask: np.ndarray, classes: list, image_path: str) -> dict:
    """Convert a segmentation mask to a GeoJSON FeatureCollection."""
    transform, crs = get_geotransform(image_path)

    features = []
    for cls_info in classes:
        idx = cls_info["index"]
        name = cls_info["name"]
        if name.endswith("*"):
            # Skip 'Ignore*' class
            continue

        binary = (mask == idx).astype(np.uint8)
        if binary.sum() == 0:
            continue

        # Find contours using rasterio.features
        try:
            import rasterio.features
            if transform is not None:
                shapes = rasterio.features.shapes(
                    binary, mask=binary > 0, transform=transform
                )
            else:
                from rasterio.transform import from_bounds
                t = from_bounds(0, mask.shape[0], mask.shape[1], 0,
                                mask.shape[1], mask.shape[0])
                shapes = rasterio.features.shapes(binary, mask=binary > 0, transform=t)

            for geom, value in shapes:
                if value == 1:
                    feature = geojson.Feature(
                        geometry=geom,
                        properties={
                            "class": name,
                            "class_index": idx,
                            "color": cls_info["color"],
                        },
                    )
                    features.append(feature)
        except Exception as e:
            print(f"Warning: could not vectorize class {name}: {e}")

    fc = geojson.FeatureCollection(features)
    if crs is not None:
        fc["crs"] = {
            "type": "name",
            "properties": {"name": str(crs)},
        }
    return fc


# ---------------------------------------------------------------------------
# Overlay rendering
# ---------------------------------------------------------------------------
def build_overlay(mask: np.ndarray, classes: list, alpha: int = 128) -> QImage:
    """Render the segmentation mask as a semi-transparent RGBA QImage."""
    h, w = mask.shape
    overlay = np.zeros((h, w, 4), dtype=np.uint8)

    for cls_info in classes:
        idx = cls_info["index"]
        if cls_info["name"].endswith("*"):
            continue
        color_hex = cls_info["color"].lstrip("#")
        r, g, b = int(color_hex[0:2], 16), int(color_hex[2:4], 16), int(color_hex[4:6], 16)
        where = mask == idx
        overlay[where, 0] = r
        overlay[where, 1] = g
        overlay[where, 2] = b
        overlay[where, 3] = alpha

    return QImage(overlay.data.tobytes(), w, h, w * 4, QImage.Format_RGBA8888).copy()


def numpy_to_qimage(img: np.ndarray) -> QImage:
    """Convert RGB numpy array to QImage."""
    h, w, c = img.shape
    bytes_per_line = w * c
    return QImage(img.data.tobytes(), w, h, bytes_per_line, QImage.Format_RGB888).copy()


# ---------------------------------------------------------------------------
# Worker thread
# ---------------------------------------------------------------------------
class InferenceWorker(QThread):
    progress = pyqtSignal(int, int)  # current, total
    image_done = pyqtSignal(str, np.ndarray, np.ndarray)  # path, image, mask
    finished_all = pyqtSignal()
    error = pyqtSignal(str)

    def __init__(self, model, metadata, image_paths, device):
        super().__init__()
        self.model = model
        self.metadata = metadata
        self.image_paths = image_paths
        self.device = device
        self._cancel = False

    def cancel(self):
        self._cancel = True

    def run(self):
        try:
            for path in self.image_paths:
                if self._cancel:
                    break
                img = read_image(path)
                mask = run_inference(
                    self.model, img, self.metadata, self.device,
                    progress_callback=lambda cur, tot: self.progress.emit(cur, tot),
                )
                self.image_done.emit(path, img, mask)
        except Exception as e:
            self.error.emit(f"{e}\n{traceback.format_exc()}")
        self.finished_all.emit()


# ---------------------------------------------------------------------------
# Zoomable image viewer
# ---------------------------------------------------------------------------
class ImageViewer(QGraphicsView):
    def __init__(self):
        super().__init__()
        self._scene = QGraphicsScene(self)
        self.setScene(self._scene)
        self._base_item = None
        self._overlay_item = None
        self.setDragMode(QGraphicsView.ScrollHandDrag)
        self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
        self.setRenderHints(QPainter.Antialiasing | QPainter.SmoothPixmapTransform)

    def set_images(self, base_pixmap: QPixmap, overlay_pixmap: QPixmap):
        self._scene.clear()
        self._base_item = self._scene.addPixmap(base_pixmap)
        self._overlay_item = self._scene.addPixmap(overlay_pixmap)
        self._scene.setSceneRect(QRectF(base_pixmap.rect()))
        self.fitInView(self._scene.sceneRect(), Qt.KeepAspectRatio)

    def update_overlay(self, overlay_pixmap: QPixmap):
        if self._overlay_item:
            self._overlay_item.setPixmap(overlay_pixmap)

    def wheelEvent(self, event):
        factor = 1.15 if event.angleDelta().y() > 0 else 1 / 1.15
        self.scale(factor, factor)

    def resizeEvent(self, event):
        super().resizeEvent(event)


# ---------------------------------------------------------------------------
# Main window
# ---------------------------------------------------------------------------
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Segmentation Inference GUI")
        self.resize(1200, 800)

        self.model = None
        self.metadata = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.results = {}  # path -> (image, mask)
        self.worker = None

        self._build_ui()

    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QHBoxLayout(central)

        # --- Left panel ---
        left = QWidget()
        left_layout = QVBoxLayout(left)
        left_layout.setContentsMargins(4, 4, 4, 4)

        # Model selection
        model_group = QGroupBox("Model")
        mg_layout = QVBoxLayout(model_group)
        self.model_label = QLabel("No model loaded")
        self.model_label.setWordWrap(True)
        btn_model = QPushButton("Select Model Folder...")
        btn_model.clicked.connect(self._select_model)
        mg_layout.addWidget(self.model_label)
        mg_layout.addWidget(btn_model)
        left_layout.addWidget(model_group)

        # Image selection
        img_group = QGroupBox("Images")
        ig_layout = QVBoxLayout(img_group)
        btn_images = QPushButton("Select Image Folder...")
        btn_images.clicked.connect(self._select_images)
        self.image_list = QListWidget()
        self.image_list.currentRowChanged.connect(self._on_image_selected)
        ig_layout.addWidget(btn_images)
        ig_layout.addWidget(self.image_list)
        left_layout.addWidget(img_group)

        # Run / Export
        run_group = QGroupBox("Actions")
        rg_layout = QVBoxLayout(run_group)
        self.btn_run = QPushButton("Run Inference")
        self.btn_run.clicked.connect(self._run_inference)
        self.btn_run.setEnabled(False)
        self.btn_export = QPushButton("Export All as GeoJSON...")
        self.btn_export.clicked.connect(self._export_geojson)
        self.btn_export.setEnabled(False)
        self.btn_export_current = QPushButton("Export Current as GeoJSON...")
        self.btn_export_current.clicked.connect(self._export_current_geojson)
        self.btn_export_current.setEnabled(False)
        self.progress = QProgressBar()
        rg_layout.addWidget(self.btn_run)
        rg_layout.addWidget(self.btn_export)
        rg_layout.addWidget(self.btn_export_current)
        rg_layout.addWidget(self.progress)
        left_layout.addWidget(run_group)

        left.setMaximumWidth(320)

        # --- Right panel ---
        right = QWidget()
        right_layout = QVBoxLayout(right)
        right_layout.setContentsMargins(4, 4, 4, 4)

        self.viewer = ImageViewer()
        right_layout.addWidget(self.viewer, stretch=1)

        # Overlay controls
        overlay_bar = QHBoxLayout()
        overlay_bar.addWidget(QLabel("Overlay opacity:"))
        self.opacity_slider = QSlider(Qt.Horizontal)
        self.opacity_slider.setRange(0, 255)
        self.opacity_slider.setValue(128)
        self.opacity_slider.valueChanged.connect(self._update_overlay)
        overlay_bar.addWidget(self.opacity_slider)
        self.opacity_label = QLabel("50%")
        overlay_bar.addWidget(self.opacity_label)
        right_layout.addLayout(overlay_bar)

        # Legend
        self.legend_layout = QHBoxLayout()
        self.legend_layout.addWidget(QLabel("Classes:"))
        self.legend_layout.addStretch()
        right_layout.addLayout(self.legend_layout)

        splitter = QSplitter(Qt.Horizontal)
        splitter.addWidget(left)
        splitter.addWidget(right)
        splitter.setStretchFactor(1, 1)
        main_layout.addWidget(splitter)

        self.statusBar().showMessage("Ready — select a model folder to begin.")

    # --- Model loading ---
    def _select_model(self):
        folder = QFileDialog.getExistingDirectory(self, "Select Model Folder")
        if not folder:
            return
        meta_path = os.path.join(folder, "metadata.json")
        if not os.path.exists(meta_path):
            QMessageBox.warning(self, "Error", "No metadata.json found in selected folder.")
            return
        try:
            self.statusBar().showMessage("Loading model...")
            QApplication.processEvents()
            self.model, self.metadata = load_model(folder, self.device)
            name = self.metadata.get("name", os.path.basename(folder))
            arch = self.metadata["architecture"]
            arch_desc = arch.get("type", "unknown")
            if "backbone" in arch:
                arch_desc += f" / {arch['backbone']}"
            self.model_label.setText(f"{name}\n{arch_desc} ({self.device})")
            self._update_legend()
            self._check_ready()
            self.statusBar().showMessage(f"Model loaded: {name}")
        except Exception as e:
            QMessageBox.critical(self, "Error loading model", str(e))
            self.statusBar().showMessage("Model loading failed.")

    def _update_legend(self):
        # Clear old legend widgets (keep the "Classes:" label)
        while self.legend_layout.count() > 2:
            item = self.legend_layout.takeAt(self.legend_layout.count() - 1)
            if item.widget():
                item.widget().deleteLater()

        if self.metadata:
            for cls in self.metadata["classes"]:
                if cls["name"].endswith("*"):
                    continue
                lbl = QLabel(f"  {cls['name']}  ")
                lbl.setStyleSheet(
                    f"background-color: {cls['color']}; color: white; "
                    f"padding: 2px 6px; border-radius: 3px; font-weight: bold;"
                )
                self.legend_layout.insertWidget(self.legend_layout.count() - 1, lbl)

    # --- Image selection ---
    def _select_images(self):
        folder = QFileDialog.getExistingDirectory(self, "Select Image Folder")
        if not folder:
            return
        self.image_list.clear()
        self.results.clear()
        self._image_paths = []
        for f in sorted(os.listdir(folder)):
            # Skip macOS resource fork files and hidden files
            if f.startswith("._") or f.startswith("."):
                continue
            ext = os.path.splitext(f)[1].lower()
            if ext in SUPPORTED_EXTENSIONS:
                full = os.path.join(folder, f)
                self._image_paths.append(full)
                self.image_list.addItem(f)
        self._check_ready()
        self.statusBar().showMessage(f"Found {len(self._image_paths)} images.")

    def _check_ready(self):
        ready = self.model is not None and hasattr(self, "_image_paths") and len(self._image_paths) > 0
        self.btn_run.setEnabled(ready)

    # --- Inference ---
    def _run_inference(self):
        self.btn_run.setEnabled(False)
        self.btn_export.setEnabled(False)
        self.btn_export_current.setEnabled(False)
        self.progress.setValue(0)
        self.results.clear()

        self.worker = InferenceWorker(
            self.model, self.metadata, self._image_paths, self.device
        )
        self.worker.progress.connect(self._on_tile_progress)
        self.worker.image_done.connect(self._on_image_done)
        self.worker.finished_all.connect(self._on_all_done)
        self.worker.error.connect(self._on_error)
        self.worker.start()

    def _on_tile_progress(self, current, total):
        self.progress.setMaximum(total)
        self.progress.setValue(current)

    def _on_image_done(self, path, image, mask):
        self.results[path] = (image, mask)
        # Select the just-completed image in the list
        idx = self._image_paths.index(path)
        self.image_list.setCurrentRow(idx)
        self.statusBar().showMessage(f"Done: {os.path.basename(path)}")

    def _on_all_done(self):
        self.btn_run.setEnabled(True)
        self.btn_export.setEnabled(len(self.results) > 0)
        self.btn_export_current.setEnabled(len(self.results) > 0)
        self.statusBar().showMessage(f"Inference complete — {len(self.results)} images processed.")

    def _on_error(self, msg):
        QMessageBox.critical(self, "Inference Error", msg)
        self.btn_run.setEnabled(True)

    # --- Display ---
    def _on_image_selected(self, row):
        if row < 0 or row >= len(self._image_paths):
            return
        path = self._image_paths[row]
        if path not in self.results:
            # Show original image without overlay
            try:
                img = read_image(path)
                qimg = numpy_to_qimage(img)
                pix = QPixmap.fromImage(qimg)
                empty = QPixmap(pix.size())
                empty.fill(QColor(0, 0, 0, 0))
                self.viewer.set_images(pix, empty)
                self.btn_export_current.setEnabled(False)
            except Exception as e:
                self.statusBar().showMessage(f"Error reading image: {e}")
            return

        self.btn_export_current.setEnabled(True)
        image, mask = self.results[path]
        self._current_image = image
        self._current_mask = mask
        self._current_path = path

        qimg = numpy_to_qimage(image)
        base_pix = QPixmap.fromImage(qimg)

        alpha = self.opacity_slider.value()
        overlay_qimg = build_overlay(mask, self.metadata["classes"], alpha)
        overlay_pix = QPixmap.fromImage(overlay_qimg)

        self.viewer.set_images(base_pix, overlay_pix)

    def _update_overlay(self, value):
        pct = int(value / 255 * 100)
        self.opacity_label.setText(f"{pct}%")

        if not hasattr(self, "_current_mask") or self._current_mask is None:
            return
        if self.metadata is None:
            return

        overlay_qimg = build_overlay(self._current_mask, self.metadata["classes"], value)
        overlay_pix = QPixmap.fromImage(overlay_qimg)
        self.viewer.update_overlay(overlay_pix)

    # --- GeoJSON export ---
    def _export_geojson(self):
        folder = QFileDialog.getExistingDirectory(self, "Select Output Folder for GeoJSON files")
        if not folder:
            return
        count = 0
        for path, (image, mask) in self.results.items():
            fc = mask_to_geojson(mask, self.metadata["classes"], path)
            stem = Path(path).stem
            out_path = os.path.join(folder, f"{stem}_detections.geojson")
            with open(out_path, "w") as f:
                geojson.dump(fc, f, indent=2)
            count += 1
        self.statusBar().showMessage(f"Exported {count} GeoJSON files to {folder}")

    def _export_current_geojson(self):
        if not hasattr(self, "_current_path"):
            return
        stem = Path(self._current_path).stem
        path, _ = QFileDialog.getSaveFileName(
            self, "Save GeoJSON", f"{stem}_detections.geojson",
            "GeoJSON (*.geojson);;All Files (*)"
        )
        if not path:
            return
        image, mask = self.results[self._current_path]
        fc = mask_to_geojson(mask, self.metadata["classes"], self._current_path)
        with open(path, "w") as f:
            geojson.dump(fc, f, indent=2)
        self.statusBar().showMessage(f"Exported: {path}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main():
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
