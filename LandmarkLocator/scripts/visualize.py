"""Visualization utilities for landmark predictions."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np

_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from data.dataset import GEOJSON_TO_LANDMARK, LANDMARK_ORDER
from inference.predict import LandmarkPredictor

# Colors per landmark (BGR for OpenCV)
LANDMARK_COLORS = {
    "subcostal_break": (0, 165, 255),   # orange
    "alula_notch": (214, 51, 65),       # blue-ish
    "l1_rs_junction": (147, 160, 229),  # salmon
    "l4_l5_junction": (242, 210, 7),    # cyan
    "wing_tip": (46, 11, 135),          # dark red
}


def load_ground_truth(geojson_path: Path) -> dict[str, tuple[float, float]]:
    """Load ground truth landmarks from GeoJSON."""
    with open(geojson_path) as f:
        data = json.load(f)

    landmarks = {}
    for feature in data["features"]:
        geom = feature["geometry"]
        class_name = feature["properties"]["classification"]["name"]
        internal_name = GEOJSON_TO_LANDMARK.get(class_name)
        if internal_name is None:
            continue
        if geom["type"] == "Point":
            x, y = geom["coordinates"]
        elif geom["type"] == "MultiPoint":
            x, y = geom["coordinates"][0]
        else:
            continue
        landmarks[internal_name] = (float(x), float(y))

    return landmarks


def draw_landmarks_on_image(
    image: np.ndarray,
    predictions: dict[str, tuple[float, float]],
    ground_truth: dict[str, tuple[float, float]] | None = None,
    radius: int = 15,
) -> np.ndarray:
    """Draw predicted landmarks (circles) and ground truth (crosses) on image."""
    vis = image.copy()

    for name in LANDMARK_ORDER:
        color = LANDMARK_COLORS.get(name, (255, 255, 255))

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
        axes[i + 1].set_title(LANDMARK_ORDER[i].replace("_", " "))

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
        vis = draw_landmarks_on_image(image, result["landmarks"], gt)
        out_path = args.output_dir / f"{img_path.stem}_landmarks.jpg"
        cv2.imwrite(str(out_path), vis)
        print(f"Saved: {out_path}")

        # Heatmap visualization
        if args.heatmaps:
            fig = visualize_heatmaps(result["heatmaps"], image)
            hm_path = args.output_dir / f"{img_path.stem}_heatmaps.png"
            fig.savefig(str(hm_path), dpi=100)
            plt.close(fig)
            print(f"Saved: {hm_path}")

        # Print results
        print(f"\n{img_path.name}:")
        for name in LANDMARK_ORDER:
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
