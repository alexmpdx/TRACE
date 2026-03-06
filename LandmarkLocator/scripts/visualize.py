"""Visualization utilities for landmark predictions."""

from __future__ import annotations

import argparse
import colorsys
import json
import sys
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np

_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from data.dataset import _normalize_name
from inference.predict import LandmarkPredictor


def generate_landmark_colors(names: list[str]) -> dict[str, tuple[int, int, int]]:
    """Generate distinct BGR colors for an arbitrary list of landmark names."""
    # Hand-picked hues that maximise perceptual distance; avoids adjacent greens
    _BASE_HUES = [0.0, 0.08, 0.17, 0.35, 0.55, 0.72, 0.85]
    colors = {}
    n = max(len(names), 1)
    for i, name in enumerate(names):
        hue = _BASE_HUES[i] if i < len(_BASE_HUES) else (i / n)
        r, g, b = colorsys.hsv_to_rgb(hue, 0.8, 0.9)
        colors[name] = (int(b * 255), int(g * 255), int(r * 255))  # BGR
    return colors


def load_ground_truth(geojson_path: Path) -> dict[str, tuple[float, float]]:
    """Load ground truth landmarks from GeoJSON."""
    with open(geojson_path) as f:
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
        internal_name = _normalize_name(class_name)
        if geom.get("type") == "Point":
            x, y = geom["coordinates"]
        elif geom.get("type") == "MultiPoint":
            x, y = geom["coordinates"][0]
        else:
            continue
        landmarks[internal_name] = (float(x), float(y))

    return landmarks


# Default colors (generated on first use or overridden per-model)
LANDMARK_COLORS: dict[str, tuple[int, int, int]] = {}


def _ensure_colors(names: list[str]) -> dict[str, tuple[int, int, int]]:
    """Ensure LANDMARK_COLORS has entries for all names."""
    global LANDMARK_COLORS
    missing = [n for n in names if n not in LANDMARK_COLORS]
    if missing:
        new_colors = generate_landmark_colors(names)
        LANDMARK_COLORS.update(new_colors)
    return LANDMARK_COLORS


def draw_landmarks_on_image(
    image: np.ndarray,
    predictions: dict[str, tuple[float, float]],
    ground_truth: dict[str, tuple[float, float]] | None = None,
    radius: int = 15,
    landmark_order: list[str] | None = None,
) -> np.ndarray:
    """Draw predicted landmarks (circles) and ground truth (crosses) on image."""
    vis = image.copy()

    # Determine which landmarks to draw
    if landmark_order is None:
        all_names = sorted(set(list(predictions.keys()) + (list(ground_truth.keys()) if ground_truth else [])))
    else:
        all_names = landmark_order

    colors = _ensure_colors(all_names)

    for name in all_names:
        color = colors.get(name, (255, 255, 255))

        # Ground truth: cross
        if ground_truth and name in ground_truth:
            gx, gy = int(ground_truth[name][0]), int(ground_truth[name][1])
            arm = radius + 5
            cv2.line(vis, (gx - arm, gy), (gx + arm, gy), color, 3)
            cv2.line(vis, (gx, gy - arm), (gx, gy + arm), color, 3)

        # Prediction: filled circle
        if name in predictions:
            px, py = int(predictions[name][0]), int(predictions[name][1])
            cv2.circle(vis, (px, py), radius, color, -1)
            cv2.circle(vis, (px, py), radius, (255, 255, 255), 2)

            # Error vector
            if ground_truth and name in ground_truth:
                gx, gy = int(ground_truth[name][0]), int(ground_truth[name][1])
                cv2.arrowedLine(vis, (px, py), (gx, gy), (0, 0, 255), 2, tipLength=0.3)

    return vis


def visualize_heatmaps(
    heatmaps: np.ndarray,
    image: np.ndarray | None = None,
    landmark_order: list[str] | None = None,
) -> plt.Figure:
    """Plot per-channel heatmaps in a grid."""
    n_channels = heatmaps.shape[0]
    fig, axes = plt.subplots(1, n_channels + 1, figsize=(4 * (n_channels + 1), 4))

    if image is not None:
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(rgb, (heatmaps.shape[2], heatmaps.shape[1]))
        axes[0].imshow(resized)
        axes[0].set_title("Image")
    else:
        axes[0].axis("off")

    for i in range(n_channels):
        axes[i + 1].imshow(heatmaps[i], cmap="hot", vmin=0)
        if landmark_order and i < len(landmark_order):
            axes[i + 1].set_title(landmark_order[i].replace("_", " "))
        else:
            axes[i + 1].set_title(f"Channel {i}")

    for ax in axes:
        ax.axis("off")

    fig.tight_layout()
    return fig


def main() -> None:
    """CLI: visualize predictions on images."""
    parser = argparse.ArgumentParser(description="Visualize landmark predictions")
    parser.add_argument("images", type=Path, nargs="+", help="Image path(s)")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--gt-dir", type=Path, default=None, help="GeoJSON annotation dir")
    parser.add_argument("--output-dir", type=Path, default=Path("vis_output"))
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--heatmaps", action="store_true", help="Also save heatmap plots")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    predictor = LandmarkPredictor(args.checkpoint, args.device)
    landmark_order = predictor.landmark_order
    _ensure_colors(landmark_order)

    for img_path in args.images:
        image = cv2.imread(str(img_path))
        if image is None:
            print(f"Warning: could not load {img_path}")
            continue

        result = predictor.predict(image)

        # Load ground truth if available
        gt = None
        if args.gt_dir:
            gt_path = args.gt_dir / (img_path.name + ".geojson")
            if gt_path.exists():
                gt = load_ground_truth(gt_path)

        # Draw overlay
        vis = draw_landmarks_on_image(image, result["landmarks"], gt, landmark_order=landmark_order)
        out_path = args.output_dir / f"{img_path.stem}_landmarks.jpg"
        cv2.imwrite(str(out_path), vis)
        print(f"Saved: {out_path}")

        # Heatmap visualization
        if args.heatmaps:
            fig = visualize_heatmaps(result["heatmaps"], image, landmark_order=landmark_order)
            hm_path = args.output_dir / f"{img_path.stem}_heatmaps.png"
            fig.savefig(str(hm_path), dpi=100)
            plt.close(fig)
            print(f"Saved: {hm_path}")

        # Print results
        print(f"\n{img_path.name}:")
        for name in landmark_order:
            x, y = result["landmarks"][name]
            conf = result["confidences"][name]
            line = f"  {name:20s}: ({x:7.1f}, {y:7.1f})  conf={conf:.3f}"
            if gt and name in gt:
                gx, gy = gt[name]
                err = np.sqrt((x - gx) ** 2 + (y - gy) ** 2)
                line += f"  error={err:.1f}px"
            print(line)


if __name__ == "__main__":
    main()
