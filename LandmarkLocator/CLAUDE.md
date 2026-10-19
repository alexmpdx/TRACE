# LandmarkLocator — Claude Code Project Instructions

## Project Purpose
A deep learning sub-project within the mapThemVeins suite. Trains a heatmap-based landmark detection model to locate anatomical keypoints in *Drosophila* wing brightfield images — hinge landmarks (subcostal break, alula notch, L1-Rs, L4-L5), vein junctions, crossvein ends, and the distal ends of L2–L5. These landmarks define the hinge line and wing length, and anchor downstream vein identification in `identifyFeatures`.

## Relationship to other modules
LandmarkLocator produces a trained model that `preprocessing` loads as Sub-stage 3, then passes the resulting landmarks GeoJSON to `HingeChopper` (Sub-stage 4) and `identifyFeatures` (the analysis stage in TRACE). LandmarkLocator imports nothing from sibling modules at runtime.

## Architecture
Single-stage pipeline: full-wing heatmap regression (no canonicalization or cropping). ResNet18-encoder U-Net predicts one Gaussian heatmap per landmark directly on the resized input image.

```
LandmarkLocator/
├── landmark_locator/    # Installable Python package (pip install -e .)
│   ├── __init__.py      # Public API: LandmarkPredictor, predict_ensemble, etc.
│   ├── data/            # Dataset, augmentation, heatmap generation
│   │   ├── dataset.py   # LandmarkDataset, GeoJSON parsing, heatmap rendering
│   │   └── augmentation.py  # Train/val transform pipelines (albumentations)
│   ├── models/          # Neural network definitions
│   │   └── unet.py      # LandmarkUNet (ResNet18 encoder + decoder)
│   ├── training/        # Training loops, loss functions
│   │   ├── train.py     # CV splits, training orchestrator, evaluation
│   │   └── losses.py    # HeatmapMSELoss
│   ├── inference/       # Prediction pipeline
│   │   └── predict.py   # LandmarkPredictor, ensemble prediction
│   └── scripts/         # CLI entry points & GUI
│       ├── train.py     # landmark-train CLI
│       ├── predict.py   # landmark-predict CLI
│       ├── visualize.py # landmark-visualize CLI
│       └── gui.py       # PyQt5 GUI
├── configs/
│   └── default.yaml     # All hyperparameters
├── trainingdata_new3/   # GeoJSON Point annotations (config default: data.annotation_dir)
├── training_data/       # 24-file annotation set, not the config default
├── training_data_pics/  # training images (.tif and .bmp)
├── pyproject.toml       # Package metadata & build config
├── setup.py             # Editable install shim for older pip
├── CLAUDE.md
└── requirements.txt
```

## Landmarks and channel order
The landmark set is not fixed in code: `discover_landmarks()` reads it from the annotations, and the checkpoint stores it as `heatmap.landmark_order` (one output channel per entry). Channel *i* is always `landmark_order[i]`; never re-sort it for a trained model. The deployed `trained_models/360_landmarks_1` model has 13:

| Channel | Internal | GeoJSON | Description |
|---------|----------|---------|-------------|
| 0 | acv_a | ACV.a | ACV–L3 junction |
| 1 | acv_p | ACV.p | ACV–L4 junction |
| 2 | alula_notch | alula notch | Indentation on the posterior margin at the hinge |
| 3 | l3_d | L3.d | Distal end of L3 at the wing margin |
| 4 | l1_rs | L1-Rs | Where L1 meets the radial sector |
| 5 | l2_d | L2.d | Distal end of L2 |
| 6 | l2_l3 | L2-L3 | L2–L3–Rs junction |
| 7 | l4_d | L4.d | Distal end of L4 |
| 8 | l4_l5 | L4-L5 | Where L4 and L5 diverge in the hinge region |
| 9 | l5_d | L5.d | Distal end of L5 |
| 10 | pcv_a | PCV.a | PCV–L4 junction |
| 11 | pcv_p | PCV.p | PCV–L5 junction |
| 12 | subcostal_break | subcostal break | Where the subcosta ends on the anterior margin |

`l3_d` sits at channel 3, out of alphabetical order, because that model was trained when the landmark was called `dtip`.

**Legacy name.** `L3.d` / `l3_d` used to be `DTip` / `dtip`. Annotations, older landmark GeoJSONs, checkpoints and gate sidecars still use the old name and are not rewritten: `canonical_landmark_name()` in `landmark_locator/names.py` renames them on read (training, checkpoint load, gate configs). Write the new name everywhere.

## Annotation Format
GeoJSON FeatureCollection with Point features. One `.geojson` file per image, in the directory the config's `data.annotation_dir` names.
```json
{"type": "Feature", "geometry": {"type": "Point", "coordinates": [x, y]},
 "properties": {"classification": {"name": "subcostal break"}}}
```
GeoJSON names → internal (`_normalize_name`): lower-case, with dots, dashes and spaces turned into underscores, after the legacy alias — `"subcostal break"` → `subcostal_break`, `"L1-Rs"` → `l1_rs`, `"L3.d"` (or legacy `"DTip"`) → `l3_d`.

**Known anomalies** (the parser takes the first coordinate of a MultiPoint and skips features without a name): in `training_data/`, `PknCG736-lacZ_engal4_BL12322_0007.tif.geojson` stores the L3.d point (under the legacy name `DTip`) as a MultiPoint; in `trainingdata_new3/`, `-CTRL_PknRNAi_108870_0007.tif.geojson` has an unnamed MultiPoint feature, which is skipped.

## Model Details
- **Encoder**: Pretrained ResNet18, features at 5 scales (stride 2/4/8/16/32)
- **Decoder**: 4 DecoderBlocks (upsample 2× → concat skip → 2×conv+BN+ReLU), final upsample to input size
- **Output**: (B, N, 352, 512), one channel per landmark — no activation, raw heatmaps
- **Input**: 512×352 (aspect ratio ~1.45:1, divisible by 32), ImageNet-normalized RGB

## Training Details
- **Heatmap targets**: Gaussian blobs (sigma=5px at model resolution) centered on landmark coordinates
- **Augmentation**: rotation ±30°, horizontal flip, scale ±15%, brightness/contrast, Gaussian blur, coarse dropout. Keypoints co-transformed via albumentations, heatmaps rendered after all transforms.
- **Loss**: MSE on heatmaps (HeatmapMSELoss with optional per-channel mask)
- **Optimizer**: AdamW (lr=1e-3, wd=1e-4), cosine LR schedule
- **Encoder freeze**: first 20 epochs, then unfreeze with 0.1× LR
- **CV**: 5-fold stratified by genotype, early stopping patience 50
- **Confidence**: peak value of predicted heatmap
- **Device**: auto-detect MPS (Apple Silicon) > CUDA > CPU

## Data
- Images in `training_data_pics/` (.tif and .bmp), loaded as BGR uint8
- Annotations in the config's `data.annotation_dir` (default `trainingdata_new3/`)
- CV folds are stratified by genotype
- All coordinates in pixel space (origin top-left)

## Key Dependencies
- `torch`, `torchvision` — model training/inference, pretrained backbone
- `numpy`, `opencv-python` — image I/O and processing
- `albumentations` — augmentation pipeline with keypoint co-transforms
- `scikit-learn` — StratifiedKFold for CV splits
- `pyyaml` — config loading

## Conventions
- All image arrays are numpy (H, W, 3) uint8 BGR for OpenCV compatibility
- Model input tensors are (B, C, H, W) float32 normalized with ImageNet stats
- Do not hardcode file paths; use pathlib.Path throughout
- All public functions have type hints and a one-line docstring
- Coordinate convention: (x, y) in pixel space, origin top-left, matching OpenCV
