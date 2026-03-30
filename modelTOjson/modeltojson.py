"""
modeltojson — load trained segmentation models and run inference.

Callable library for loading QuPath extension models (SMP, full PyTorch, ONNX),
running tiled inference on images, and exporting detections as GeoJSON.

Supported model architectures:
  - SMP-based (unet, unet++, deeplabv3, deeplabv3+, fpn, manet, linknet,
    pspnet, pan) with any SMP-compatible backbone
  - Full PyTorch models saved as .pt (e.g. custom ViT / MuViT architectures)
  - ONNX models via onnxruntime (custom_onnx)

Example usage::

    from modeltojson import load_model, read_image, run_inference, mask_to_geojson

    model, metadata = load_model("path/to/model_dir")
    image = read_image("path/to/image.tif")
    mask = run_inference(model, image, metadata)
    geojson = mask_to_geojson(mask, metadata["classes"], "path/to/image.tif")
"""

import json
import os
from abc import ABC, abstractmethod
from pathlib import Path

import geojson
import numpy as np
import rasterio
import tifffile
import torch
import torch.nn as nn
from PIL import Image


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
                child.num_features,
                eps=child.eps,
                momentum=child.momentum,
                affine=child.affine,
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
    def to(self, device: torch.device) -> "ModelWrapper": ...

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
        np_input = np.from_dlpack(tensor.detach().cpu().contiguous())
        outputs = self.session.run(None, {self.input_name: np_input})
        return torch.tensor(np.array(outputs[0]), dtype=torch.float32).to(self.device)

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

    # When context_scale > 1, model has 2x channels (detail + context)
    context_scale = int(arch.get("context_scale", 1))
    if context_scale > 1:
        in_channels = in_channels * 2

    kwargs = dict(
        encoder_name=backbone,
        encoder_weights=None,
        in_channels=in_channels,
        classes=num_classes,
    )

    if "encoder_depth" in arch:
        kwargs["encoder_depth"] = int(arch["encoder_depth"])

    if "decoder_channels" in arch and model_cls in (smp.Unet, smp.UnetPlusPlus, smp.MAnet):
        kwargs["decoder_channels"] = [int(c) for c in arch["decoder_channels"]]

    return model_cls(**kwargs)


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------
def _find_model_file(model_dir: Path):
    """Locate the model weights file in the directory.

    Returns (path, format) where format is 'pt' or 'onnx'.
    """
    if (model_dir / "model.pt").exists():
        return model_dir / "model.pt", "pt"
    if (model_dir / "model.onnx").exists():
        return model_dir / "model.onnx", "onnx"

    onnx_files = list(model_dir.glob("*.onnx"))
    if onnx_files:
        return onnx_files[0], "onnx"

    checkpoints = sorted(model_dir.glob("checkpoint_*.pt"))
    if checkpoints:
        return checkpoints[-1], "pt"

    raise FileNotFoundError(
        f"No model file found in {model_dir}. " f"Expected model.pt, model.onnx, or checkpoint_*.pt"
    )


def _load_torch_state(path: Path, device: torch.device):
    """Load a .pt file and return (state_dict_or_model, is_full_model)."""
    loaded = torch.load(path, map_location=device, weights_only=False)

    if isinstance(loaded, dict):
        for key in ("model_state_dict", "state_dict", "model"):
            if key in loaded and isinstance(loaded[key], dict):
                return loaded[key], False
        sample_keys = list(loaded.keys())[:5]
        if sample_keys and all(isinstance(k, str) and ("." in k or "weight" in k or "bias" in k) for k in sample_keys):
            return loaded, False
        return loaded, False

    if isinstance(loaded, nn.Module):
        return loaded, True

    raise ValueError(f"Cannot interpret {path.name}: expected state dict or nn.Module, " f"got {type(loaded).__name__}")


def load_model(model_dir: str, device: torch.device = None):
    """Load a trained model from a model directory containing metadata.json.

    Args:
        model_dir: Path to directory containing metadata.json and model weights.
        device: Torch device. Defaults to CUDA if available, else CPU.

    Returns:
        Tuple of (model_wrapper, metadata_dict).

    Supports:
      - SMP architectures with any SMP-compatible backbone
      - Full PyTorch models saved with torch.save (custom architectures)
      - ONNX models via onnxruntime
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
            onnx_files = list(model_dir.glob("*.onnx"))
            if not onnx_files:
                raise FileNotFoundError(f"Architecture is 'custom_onnx' but no .onnx file found in {model_dir}")
            onnx_path = onnx_files[0]
        wrapper = OnnxModelWrapper(str(onnx_path), device)
        return wrapper, metadata

    # ----- PyTorch path -----
    loaded, is_full_model = _load_torch_state(model_path, device)

    if is_full_model:
        model = loaded
    else:
        smp_cls = _get_smp_model_class(arch_type)

        if smp_cls is not None:
            model = _build_smp_model(arch, num_classes)
            if arch.get("use_batchrenorm", False):
                _replace_bn_with_batchrenorm(model)
            model.load_state_dict(loaded, strict=False)
        else:
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

    try:
        with rasterio.open(path) as src:
            bands = src.read()
            if bands.shape[0] >= 3:
                img = np.stack([bands[0], bands[1], bands[2]], axis=-1)
            elif bands.shape[0] == 1:
                img = np.stack([bands[0]] * 3, axis=-1)
            else:
                img = np.stack([bands[0], bands[1], bands[0]], axis=-1)
    except Exception:
        pass

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


def list_images(folder: str) -> list:
    """List supported image files in a folder, skipping hidden/resource fork files."""
    paths = []
    for f in sorted(os.listdir(folder)):
        if f.startswith("._") or f.startswith("."):
            continue
        ext = os.path.splitext(f)[1].lower()
        if ext in SUPPORTED_EXTENSIONS:
            paths.append(os.path.join(folder, f))
    return paths


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
    return torch.tensor(tile.transpose(2, 0, 1).copy(), dtype=torch.float32)


def _compute_effective_padding(tile_size: int) -> int:
    """Compute padding per side for center-crop tiling.

    Matches QuPath extension: min 25% per side, max 3/8, floor 64px.
    """
    min_pad = tile_size // 4
    max_pad = 3 * tile_size // 8
    return max(64, min(min_pad, max_pad))


def run_inference(
    model,
    image: np.ndarray,
    metadata: dict,
    device: torch.device = None,
    progress_callback=None,
    smoothing_sigma: float = 2.0,
) -> np.ndarray:
    """Run tiled inference using expanded reads + center-crop stitching.

    Each tile is read with extra context (real neighboring pixels where
    available, reflection padding at image edges). Only the center portion
    of each tile's prediction is kept — the halo is discarded.

    Args:
        model: A ModelWrapper (from load_model) or any callable taking (N,C,H,W) tensors.
        image: RGB uint8 numpy array (H, W, 3).
        metadata: The metadata dict returned by load_model.
        device: Torch device. Defaults to CUDA if available, else CPU.
        progress_callback: Optional callable(current_tile, total_tiles).
        smoothing_sigma: Gaussian sigma for probability smoothing (0 to disable).

    Returns:
        Predicted class indices as (H, W) uint8 numpy array at original resolution.
    """
    from scipy.ndimage import gaussian_filter

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    arch = metadata["architecture"]
    tile_size = int(arch["input_width"])
    downsample = int(arch["downsample"])
    context_scale = int(arch.get("context_scale", 1))
    num_classes = len(metadata["classes"])

    h_orig, w_orig = image.shape[:2]

    if downsample > 1:
        h_ds = h_orig // downsample
        w_ds = w_orig // downsample
        img_ds = np.array(Image.fromarray(image).resize((w_ds, h_ds), Image.BILINEAR))
    else:
        h_ds, w_ds = h_orig, w_orig
        img_ds = image

    # Normalization stats: prefer saved per-channel training stats, fall
    # back to estimating per-channel percentiles from the image
    norm_stats = metadata.get("normalization_stats")
    if norm_stats is None:
        input_cfg = metadata.get("input_config", {}).get("normalization", {})
        clip_pct = input_cfg.get("clip_percentile", 99.0)
        num_ch = img_ds.shape[2]
        norm_stats = [
            {
                "p1": float(np.percentile(img_ds[..., c], 100 - clip_pct)),
                "p99": float(np.percentile(img_ds[..., c], clip_pct)),
            }
            for c in range(num_ch)
        ]

    input_padding = _compute_effective_padding(tile_size)
    stride = tile_size - 2 * input_padding

    tiles_y = max(1, (h_ds + stride - 1) // stride)
    tiles_x = max(1, (w_ds + stride - 1) // stride)
    total_tiles = tiles_y * tiles_x

    prob_map = np.zeros((num_classes, h_ds, w_ds), dtype=np.float32)

    tile_idx = 0
    with torch.no_grad():
        for ty in range(tiles_y):
            for tx in range(tiles_x):
                out_y0 = ty * stride
                out_x0 = tx * stride
                out_y1 = min(out_y0 + stride, h_ds)
                out_x1 = min(out_x0 + stride, w_ds)
                out_h = out_y1 - out_y0
                out_w = out_x1 - out_x0

                in_y0 = out_y0 - input_padding
                in_x0 = out_x0 - input_padding
                in_y1 = in_y0 + tile_size
                in_x1 = in_x0 + tile_size

                pad_top = max(0, -in_y0)
                pad_left = max(0, -in_x0)
                pad_bottom = max(0, in_y1 - h_ds)
                pad_right = max(0, in_x1 - w_ds)

                read_y0 = max(0, in_y0)
                read_x0 = max(0, in_x0)
                read_y1 = min(h_ds, in_y1)
                read_x1 = min(w_ds, in_x1)

                tile = img_ds[read_y0:read_y1, read_x0:read_x1]

                if pad_top > 0 or pad_left > 0 or pad_bottom > 0 or pad_right > 0:
                    tile = np.pad(
                        tile,
                        ((pad_top, pad_bottom), (pad_left, pad_right), (0, 0)),
                        mode="reflect",
                    )

                if tile.shape[0] < tile_size or tile.shape[1] < tile_size:
                    extra_h = tile_size - tile.shape[0]
                    extra_w = tile_size - tile.shape[1]
                    tile = np.pad(tile, ((0, extra_h), (0, extra_w), (0, 0)), mode="reflect")

                # Multi-scale context
                if context_scale > 1:
                    ctx_half_w = (tile_size * context_scale) // 2
                    ctx_half_h = (tile_size * context_scale) // 2
                    tile_center_y = in_y0 + tile_size // 2
                    tile_center_x = in_x0 + tile_size // 2

                    ctx_y0 = tile_center_y - ctx_half_h
                    ctx_x0 = tile_center_x - ctx_half_w
                    ctx_y1 = ctx_y0 + tile_size * context_scale
                    ctx_x1 = ctx_x0 + tile_size * context_scale

                    if ctx_y0 < 0:
                        ctx_y0 = 0
                        ctx_y1 = min(tile_size * context_scale, h_ds)
                    if ctx_x0 < 0:
                        ctx_x0 = 0
                        ctx_x1 = min(tile_size * context_scale, w_ds)
                    if ctx_y1 > h_ds:
                        ctx_y1 = h_ds
                        ctx_y0 = max(0, ctx_y1 - tile_size * context_scale)
                    if ctx_x1 > w_ds:
                        ctx_x1 = w_ds
                        ctx_x0 = max(0, ctx_x1 - tile_size * context_scale)

                    ctx_region = img_ds[ctx_y0:ctx_y1, ctx_x0:ctx_x1]
                    ctx_tile = np.array(Image.fromarray(ctx_region).resize((tile_size, tile_size), Image.BILINEAR))
                    tile = np.concatenate([tile, ctx_tile], axis=-1)
                    tile_norm = norm_stats + norm_stats
                else:
                    tile_norm = norm_stats

                tensor = preprocess_tile(tile, tile_norm).unsqueeze(0).to(device)
                output = model(tensor)
                if not isinstance(output, torch.Tensor):
                    output = torch.tensor(np.asarray(output), dtype=torch.float32)
                probs = np.from_dlpack(torch.softmax(output, dim=1).detach().cpu().contiguous())[0]

                center = probs[
                    :,
                    input_padding : input_padding + out_h,
                    input_padding : input_padding + out_w,
                ]
                prob_map[:, out_y0:out_y1, out_x0:out_x1] = center

                tile_idx += 1
                if progress_callback:
                    progress_callback(tile_idx, total_tiles)

    if smoothing_sigma > 0:
        for c in range(num_classes):
            prob_map[c] = gaussian_filter(prob_map[c], sigma=smoothing_sigma)

    pred_classes = prob_map.argmax(axis=0).astype(np.uint8)

    if downsample > 1:
        pred_classes = np.array(Image.fromarray(pred_classes).resize((w_orig, h_orig), Image.NEAREST))

    return pred_classes


# ---------------------------------------------------------------------------
# GeoJSON export
# ---------------------------------------------------------------------------
def mask_to_geojson(mask: np.ndarray, classes: list, image_path: str) -> dict:
    """Convert a segmentation mask to a GeoJSON FeatureCollection.

    Args:
        mask: (H, W) uint8 array of class indices from run_inference.
        classes: The classes list from metadata.
        image_path: Path to the source image (used for georeferencing).

    Returns:
        A GeoJSON FeatureCollection dict.
    """
    transform, crs = get_geotransform(image_path)

    features = []
    for cls_info in classes:
        idx = cls_info["index"]
        name = cls_info["name"]
        if name.endswith("*"):
            continue

        binary = (mask == idx).astype(np.uint8)
        if binary.sum() == 0:
            continue

        try:
            import rasterio.features

            if transform is not None:
                shapes = rasterio.features.shapes(binary, mask=binary > 0, transform=transform)
            else:
                from rasterio.transform import from_bounds

                t = from_bounds(0, mask.shape[0], mask.shape[1], 0, mask.shape[1], mask.shape[0])
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


def save_geojson(fc: dict, path: str):
    """Save a GeoJSON FeatureCollection to a file."""
    with open(path, "w") as f:
        geojson.dump(fc, f, indent=2)


# ---------------------------------------------------------------------------
# Convenience: process a folder of images
# ---------------------------------------------------------------------------
def process_folder(
    model_dir: str,
    image_folder: str,
    output_folder: str,
    device: torch.device = None,
    smoothing_sigma: float = 2.0,
    progress_callback=None,
):
    """Run inference on all images in a folder and save GeoJSON detections.

    Args:
        model_dir: Path to model directory.
        image_folder: Path to folder of images.
        output_folder: Path to save GeoJSON files.
        device: Torch device. Defaults to CUDA if available, else CPU.
        smoothing_sigma: Gaussian sigma for probability smoothing.
        progress_callback: Optional callable(image_index, total_images, image_path).

    Returns:
        List of (image_path, geojson_path) tuples for successfully processed images.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model, metadata = load_model(model_dir, device)
    image_paths = list_images(image_folder)
    os.makedirs(output_folder, exist_ok=True)

    results = []
    for i, img_path in enumerate(image_paths):
        if progress_callback:
            progress_callback(i, len(image_paths), img_path)

        image = read_image(img_path)
        mask = run_inference(model, image, metadata, device, smoothing_sigma=smoothing_sigma)
        fc = mask_to_geojson(mask, metadata["classes"], img_path)

        stem = Path(img_path).stem
        out_path = os.path.join(output_folder, f"{stem}_detections.geojson")
        save_geojson(fc, out_path)
        results.append((img_path, out_path))

    return results
