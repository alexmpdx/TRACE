# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What This Project Does

PyQt5 GUI application that loads trained semantic segmentation models (from QuPath extension format), runs tiled inference on image folders, displays results with adjustable overlay, and exports detections as GeoJSON.

## Setup & Run

```bash
pip install -r requirements.txt
python app.py
```

## Architecture

Single-file application (`app.py`) with these major sections:

- **Model loading** — Three-backend system behind a `ModelWrapper` abstraction:
  - `TorchModelWrapper`: SMP architectures (unet, unet++, deeplabv3, fpn, etc.) rebuilt from `metadata.json`, or full `nn.Module` models loaded directly from `.pt`
  - `OnnxModelWrapper`: ONNX models via `onnxruntime` (architecture type `custom_onnx`)
  - `BatchRenorm`: Drop-in replacement for BatchNorm2d used by QuPath-trained models
  - SMP imports are lazy so ONNX-only users don't need `segmentation_models_pytorch`

- **Inference pipeline** — `run_inference()` handles tiling (configurable size/overlap/downsample from metadata), percentile-99 normalization, softmax averaging of overlapping tiles, and upsampling back to original resolution

- **GeoJSON export** — `mask_to_geojson()` vectorizes segmentation masks using `rasterio.features.shapes`, preserving georeferencing from GeoTIFFs when available

- **GUI** — `MainWindow` with `InferenceWorker` (QThread) for async processing, `ImageViewer` (QGraphicsView with zoom/pan), opacity slider for overlay blending

## Model Directory Format

Models are directories containing:
- `metadata.json` — architecture config, class definitions, normalization stats, training settings
- `model.pt` or `model.onnx` or `checkpoint_*.pt` — weights file

Key metadata fields: `architecture.type` (selects loading path), `architecture.backbone`, `architecture.input_size`, `architecture.downsample`, `normalization_stats` (per-channel p1/p99), `classes` (with color/name/index).
