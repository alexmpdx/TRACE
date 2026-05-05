"""Render N augmented training samples with keypoints overlaid for visual sanity-check."""

import argparse
import json
import random
from pathlib import Path

import albumentations as A
import cv2
import numpy as np
import yaml

from landmark_locator.data.augmentation import get_train_transform
from landmark_locator.data.dataset import discover_landmarks
from landmark_locator.data.psd_loader import imread_any

_project_root = Path(__file__).resolve().parent.parent.parent

_COLORS_BGR = [
    (0, 0, 255),
    (0, 255, 0),
    (255, 0, 0),
    (0, 255, 255),
    (255, 0, 255),
    (255, 255, 0),
    (128, 128, 255),
    (255, 128, 128),
]


def _parse_geojson(path: Path, geojson_to_landmark: dict[str, str]) -> dict[str, tuple[float, float]]:
    data = json.loads(path.read_text())
    landmarks: dict[str, tuple[float, float]] = {}
    for feature in data.get("features", []):
        geom = feature.get("geometry", {})
        props = feature.get("properties", {})
        classification = props.get("classification")
        if not isinstance(classification, dict):
            continue
        class_name = classification.get("name")
        if not class_name:
            continue
        internal = geojson_to_landmark.get(class_name)
        if internal is None:
            continue
        if geom.get("type") == "Point":
            x, y = geom["coordinates"]
        elif geom.get("type") == "MultiPoint":
            x, y = geom["coordinates"][0]
        else:
            continue
        landmarks[internal] = (float(x), float(y))
    return landmarks


def _draw_keypoints(image_rgb, keypoints, names, landmark_order):
    img = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR).copy()
    color_map = {name: _COLORS_BGR[i % len(_COLORS_BGR)] for i, name in enumerate(landmark_order)}
    h, w = img.shape[:2]
    for (x, y), name in zip(keypoints, names):
        xi, yi = int(round(x)), int(round(y))
        if not (0 <= xi < w and 0 <= yi < h):
            continue
        color = color_map.get(name, (255, 255, 255))
        cv2.circle(img, (xi, yi), 6, color, -1)
        cv2.circle(img, (xi, yi), 7, (0, 0, 0), 1)
        cv2.putText(img, name[:6], (xi + 8, yi - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)
    return img


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=_project_root / "configs" / "default.yaml")
    parser.add_argument(
        "--annotation",
        type=Path,
        default=None,
        help="GeoJSON annotation path. Defaults to first *.geojson in cfg.data.annotation_dir.",
    )
    parser.add_argument("--image-dir", type=Path, default=None)
    parser.add_argument("--n", type=int, default=12)
    parser.add_argument("--output-dir", type=Path, default=_project_root / "aug_preview")
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    cfg = yaml.safe_load(args.config.read_text())

    annotation_dir = Path(cfg["data"]["annotation_dir"])
    if not annotation_dir.is_absolute():
        annotation_dir = _project_root / annotation_dir
    image_dir = args.image_dir or Path(cfg["data"]["image_dir"])
    if not image_dir.is_absolute():
        image_dir = _project_root / image_dir

    landmark_order, geojson_to_landmark = discover_landmarks(annotation_dir)
    cfg["heatmap"]["landmark_order"] = landmark_order
    cfg["heatmap"]["geojson_to_landmark"] = geojson_to_landmark
    cfg["heatmap"]["num_landmarks"] = len(landmark_order)

    geojson_path = args.annotation or next(annotation_dir.glob("*.geojson"))
    image_path = image_dir / geojson_path.stem
    if not image_path.exists():
        raise SystemExit(f"Image not found at expected path: {image_path}")

    image = imread_any(image_path)
    if image is None:
        raise SystemExit(f"Failed to read image: {image_path}")
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    landmarks_dict = _parse_geojson(geojson_path, geojson_to_landmark)
    keypoints = [landmarks_dict.get(n, (0.0, 0.0)) for n in landmark_order]
    landmark_names = list(landmark_order)
    present = [n for n in landmark_order if n in landmarks_dict]
    print(f"Image: {image_path.name}  ({image.shape[1]}×{image.shape[0]})")
    print(f"Annotation: {geojson_path.name}  — {len(present)}/{len(landmark_order)} landmarks present: {present}")

    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)

    args.output_dir.mkdir(parents=True, exist_ok=True)

    ref_t = A.Compose(
        [A.Resize(cfg["input"]["height"], cfg["input"]["width"])],
        keypoint_params=A.KeypointParams(format="xy", label_fields=["landmark_names"], remove_invisible=False),
    )
    ref = ref_t(image=image_rgb, keypoints=keypoints, landmark_names=landmark_names)
    ref_vis = _draw_keypoints(ref["image"], ref["keypoints"], ref["landmark_names"], landmark_order)
    ref_path = args.output_dir / f"{image_path.stem}_aug_REF.png"
    cv2.imwrite(str(ref_path), ref_vis)

    transform = get_train_transform(cfg)
    for i in range(args.n):
        out = transform(image=image_rgb, keypoints=keypoints, landmark_names=landmark_names)
        vis = _draw_keypoints(out["image"], out["keypoints"], out["landmark_names"], landmark_order)
        cv2.imwrite(str(args.output_dir / f"{image_path.stem}_aug_{i:02d}.png"), vis)

    print(f"Wrote {args.n + 1} images to {args.output_dir}  (1 reference + {args.n} augmented)")


if __name__ == "__main__":
    main()
