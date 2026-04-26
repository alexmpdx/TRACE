"""Visualization utilities for landmark predictions."""

from __future__ import annotations

import argparse
import colorsys
import json
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
from landmark_locator.data.dataset import _normalize_name
from landmark_locator.inference.predict import LandmarkPredictor

# Hand-picked maximum-contrast BGR palette keyed by canonical internal landmark names.
# Choices favor wide hue separation plus mixed brightness so adjacent landmarks in the
# legend (e.g. acv_a vs acv_p) are never near-duplicates on screen.
_LANDMARK_PALETTE: dict[str, tuple[int, int, int]] = {
    # Core hinge/tip — primaries
    "subcostal_break": (0, 0, 255),  # red
    "alula_notch": (0, 255, 255),  # yellow
    "l1_rs": (255, 0, 0),  # electric blue
    "l2_l3": (0, 255, 0),  # green
    "l4_l5": (255, 0, 255),  # magenta
    "dtip": (255, 255, 0),  # cyan
    # Crossveins — secondaries
    "acv_a": (0, 140, 255),  # orange
    "acv_p": (170, 50, 255),  # hot pink
    "pcv_a": (255, 255, 255),  # white
    "pcv_p": (50, 200, 255),  # warm gold
    # Distal vein tips — tertiaries
    "l2_d": (180, 60, 120),  # deep purple
    "l4_d": (180, 230, 0),  # bright teal/cyan-green
    "l5_d": (70, 30, 160),  # maroon
}

# Distinct fallback palette for any landmark name not in the curated map above.
# Picked to interleave bright and darker shades to avoid adjacent look-alikes.
_FALLBACK_PALETTE: list[tuple[int, int, int]] = [
    (0, 200, 100),  # forest green
    (100, 100, 255),  # salmon
    (255, 150, 50),  # sky blue
    (50, 50, 200),  # brick
    (200, 200, 50),  # turquoise-ish
    (150, 0, 255),  # rose
    (100, 255, 255),  # lemon
    (255, 100, 200),  # orchid
]


def generate_landmark_colors(names: list[str]) -> dict[str, tuple[int, int, int]]:
    """Return a high-contrast BGR color per landmark name.

    Uses the curated palette for known landmark names; falls through to a rotating
    distinct-secondary palette for unknown names.
    """
    colors: dict[str, tuple[int, int, int]] = {}
    used = set()
    fallback_idx = 0
    for name in names:
        if name in _LANDMARK_PALETTE:
            bgr = _LANDMARK_PALETTE[name]
        else:
            bgr = _FALLBACK_PALETTE[fallback_idx % len(_FALLBACK_PALETTE)]
            fallback_idx += 1
        # If a collision somehow slips in (same BGR already used) nudge fallback forward.
        while bgr in used and fallback_idx < 100:
            bgr = _FALLBACK_PALETTE[fallback_idx % len(_FALLBACK_PALETTE)]
            fallback_idx += 1
        used.add(bgr)
        colors[name] = bgr
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
    """Refresh LANDMARK_COLORS for the requested names using the curated palette.

    Always regenerates to avoid stale entries cached from an older palette.
    """
    global LANDMARK_COLORS
    LANDMARK_COLORS.update(generate_landmark_colors(names))
    return LANDMARK_COLORS


def draw_landmarks_on_image(
    image: np.ndarray,
    predictions: dict[str, tuple[float, float]],
    ground_truth: dict[str, tuple[float, float]] | None = None,
    radius: int | None = None,
    landmark_order: list[str] | None = None,
    show_labels: bool = False,
    size_scale: float = 1.0,
) -> np.ndarray:
    """Draw predicted landmarks (circles) and ground truth (crosses) on image.

    When `radius` is None it scales with image size (small — ~0.2% of the min dim).
    `size_scale` (default 1.0) multiplies the auto-radius and font size so callers can
    offer a live slider to grow or shrink every overlay element.
    When `show_labels` is True, each predicted point gets its name painted next to it
    in the landmark's color, with a dark halo for legibility.
    """
    vis = image.copy()

    h, w = vis.shape[:2]
    # Dot size uses a quadratic response on size_scale so the slider produces
    # visible changes at display zoom — a linear scale on this small base reads
    # as almost no change when the image is fit to the viewport.
    dot_scale = size_scale * size_scale
    if radius is None:
        radius = max(2, int(min(h, w) / 500 * dot_scale))
    ring_thick = max(1, radius // 4)
    font_scale = max(0.25, (min(h, w) / 2200.0) * size_scale)
    font_thick = max(1, int(round((min(h, w) / 1800.0) * size_scale)))

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
            arm = radius + max(2, radius // 2)
            cv2.line(vis, (gx - arm, gy), (gx + arm, gy), color, ring_thick)
            cv2.line(vis, (gx, gy - arm), (gx, gy + arm), color, ring_thick)

        # Prediction: filled circle with thin white ring for pop on dark tissue
        if name in predictions:
            px, py = int(predictions[name][0]), int(predictions[name][1])
            cv2.circle(vis, (px, py), radius, color, -1, cv2.LINE_AA)
            cv2.circle(vis, (px, py), radius, (255, 255, 255), ring_thick, cv2.LINE_AA)

            # Optional: landmark name as a labeled chip next to the point.
            if show_labels:
                label = name.replace("_", " ")
                tx = px + radius + max(3, radius)
                ty = py + max(3, radius // 2)
                # Black halo first, then colored text on top — readable on any tissue color.
                cv2.putText(
                    vis,
                    label,
                    (tx, ty),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    font_scale,
                    (0, 0, 0),
                    font_thick + 2,
                    cv2.LINE_AA,
                )
                cv2.putText(
                    vis,
                    label,
                    (tx, ty),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    font_scale,
                    color,
                    font_thick,
                    cv2.LINE_AA,
                )

            # Error vector
            if ground_truth and name in ground_truth:
                gx, gy = int(ground_truth[name][0]), int(ground_truth[name][1])
                cv2.arrowedLine(vis, (px, py), (gx, gy), (0, 0, 255), ring_thick, tipLength=0.3)

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

    from landmark_locator.data.psd_loader import imread_any

    for img_path in args.images:
        image = imread_any(img_path)
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
