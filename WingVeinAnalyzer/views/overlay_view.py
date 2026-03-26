"""Skeleton overlay and rainbow intervein overlay rendering."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from shapely.geometry import LineString, Polygon

from WingVeinAnalyzer.models.vein_labeler import VeinAssignment, VeinStatus
from WingVeinAnalyzer.models.vein_map import (
    INTERVEIN_COLORS,
    INTERVEIN_SPACE_NAMES,
    VEIN_COLORS,
)

VEIN_THICKNESS = 6
OUTLINE_COLOR = (100, 100, 100)
OUTLINE_THICKNESS = 3
# Legend styling — scaled for high-res images (5440x3648)
LEGEND_FONT = cv2.FONT_HERSHEY_SIMPLEX
LEGEND_FONT_SCALE = 1.4
LEGEND_FONT_THICKNESS = 2
LEGEND_SWATCH_SIZE = 36
LEGEND_LINE_HEIGHT = 56
LEGEND_PADDING = 24
LEGEND_BG_COLOR = (255, 255, 255)
LEGEND_BG_ALPHA = 0.85
LEGEND_TEXT_COLOR = (30, 30, 30)

# Human-readable labels for intervein spaces
_REGION_LABELS: dict[str, str] = {
    "marginal_cell": "Marginal cell (L1-L2)",
    "submarginal_cell": "Submarginal cell (L2-L3)",
    "1st_basal_cell": "1st basal cell (L3-L4 prox.)",
    "1st_posterior_cell": "1st posterior cell (L3-L4 dist.)",
    "discal_cell": "Discal cell (L4-L5 prox.)",
    "2nd_posterior_cell": "2nd posterior cell (L4-L5 dist.)",
    "3rd_posterior_cell": "3rd posterior cell (post. L5)",
}


def render_skeleton_overlay(
    image: np.ndarray,
    assignments: list[VeinAssignment],
    outline_polygon: Polygon | None = None,
    output_path: Path | None = None,
) -> np.ndarray:
    """Draw color-coded vein LineStrings on the original image with legend."""
    overlay = image.copy()

    # Draw wing outline
    if outline_polygon is not None and not outline_polygon.is_empty:
        pts = np.array(outline_polygon.exterior.coords, dtype=np.int32)
        cv2.polylines(
            overlay,
            [pts],
            isClosed=True,
            color=OUTLINE_COLOR,
            thickness=OUTLINE_THICKNESS,
            lineType=cv2.LINE_AA,
        )

    # Draw each vein in its assigned color
    legend_entries: list[tuple[tuple[int, int, int], str]] = []
    for a in assignments:
        if a.line is None:
            continue
        pts = np.array(a.line.coords, dtype=np.int32)
        if len(pts) < 2:
            continue
        color = VEIN_COLORS.get(a.vein_id, (40, 40, 40))
        cv2.polylines(
            overlay,
            [pts],
            isClosed=False,
            color=color,
            thickness=VEIN_THICKNESS,
            lineType=cv2.LINE_AA,
        )
        legend_entries.append((color, a.vein_id))

    # Draw legend
    _draw_legend(overlay, legend_entries, position="top_right")

    if output_path is not None:
        cv2.imwrite(str(output_path), overlay)

    return overlay


def render_rainbow_overlay(
    image: np.ndarray,
    wing_regions: dict[str, Polygon],
    output_path: Path | None = None,
    opacity: float = 0.75,
) -> np.ndarray:
    """Render colored intervein space overlay on the original image with legend."""
    result = image.copy()
    color_overlay = np.zeros_like(image)
    mask = np.zeros(image.shape[:2], dtype=np.uint8)

    legend_entries: list[tuple[tuple[int, int, int], str]] = []

    # Render known regions plus any extra regions (ER1, ER2, ...)
    # ER colors cycle through distinct BGR values
    _ER_COLORS = [
        (0, 200, 200),  # yellow
        (180, 105, 255),  # hot pink
        (0, 215, 255),  # gold
        (203, 192, 255),  # pink
        (147, 20, 255),  # deep pink
    ]
    render_order = list(INTERVEIN_SPACE_NAMES) + sorted(k for k in wing_regions if k.startswith("ER"))
    for region_name in render_order:
        poly = wing_regions.get(region_name)
        if poly is None or poly.is_empty:
            continue
        if region_name.startswith("ER"):
            er_idx = int(region_name[2:]) - 1
            color = _ER_COLORS[er_idx % len(_ER_COLORS)]
        else:
            color = INTERVEIN_COLORS.get(region_name, (180, 180, 180))
        pts = np.array(poly.exterior.coords, dtype=np.int32)

        cv2.fillPoly(color_overlay, [pts], color)
        cv2.fillPoly(mask, [pts], 255)

        for interior in poly.interiors:
            hole_pts = np.array(interior.coords, dtype=np.int32)
            cv2.fillPoly(color_overlay, [hole_pts], (0, 0, 0))
            cv2.fillPoly(mask, [hole_pts], 0)

        label = _REGION_LABELS.get(region_name, region_name)
        legend_entries.append((color, label))

    # Blend
    mask_bool = mask > 0
    for c in range(3):
        result[:, :, c] = np.where(
            mask_bool,
            (
                image[:, :, c].astype(np.float32) * (1 - opacity) + color_overlay[:, :, c].astype(np.float32) * opacity
            ).astype(np.uint8),
            image[:, :, c],
        )

    # Draw legend
    _draw_legend(result, legend_entries, position="top_right")

    if output_path is not None:
        cv2.imwrite(str(output_path), result)

    return result


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _draw_legend(
    image: np.ndarray,
    entries: list[tuple[tuple[int, int, int], str]],
    position: str = "top_right",
) -> None:
    """Draw a semi-transparent legend box with color swatches on the image."""
    if not entries:
        return

    # Compute legend dimensions
    max_text_width = 0
    for _, label in entries:
        (tw, _), _ = cv2.getTextSize(label, LEGEND_FONT, LEGEND_FONT_SCALE, LEGEND_FONT_THICKNESS)
        max_text_width = max(max_text_width, tw)

    box_w = LEGEND_PADDING * 3 + LEGEND_SWATCH_SIZE + max_text_width
    box_h = LEGEND_PADDING * 2 + len(entries) * LEGEND_LINE_HEIGHT
    img_h, img_w = image.shape[:2]

    # Position the legend
    if position == "top_right":
        x0 = img_w - box_w - 20
        y0 = 20
    elif position == "top_left":
        x0 = 20
        y0 = 20
    else:
        x0 = img_w - box_w - 20
        y0 = 20

    x1 = x0 + box_w
    y1 = y0 + box_h

    # Semi-transparent background
    roi = image[y0:y1, x0:x1].copy()
    bg = np.full_like(roi, LEGEND_BG_COLOR)
    blended = cv2.addWeighted(roi, 1 - LEGEND_BG_ALPHA, bg, LEGEND_BG_ALPHA, 0)
    image[y0:y1, x0:x1] = blended

    # Border
    cv2.rectangle(image, (x0, y0), (x1, y1), (150, 150, 150), 1)

    # Draw entries
    for i, (color, label) in enumerate(entries):
        ey = y0 + LEGEND_PADDING + i * LEGEND_LINE_HEIGHT
        sx = x0 + LEGEND_PADDING
        sy = ey + 2

        # Color swatch
        cv2.rectangle(
            image,
            (sx, sy),
            (sx + LEGEND_SWATCH_SIZE, sy + LEGEND_SWATCH_SIZE),
            color,
            -1,
        )
        cv2.rectangle(
            image,
            (sx, sy),
            (sx + LEGEND_SWATCH_SIZE, sy + LEGEND_SWATCH_SIZE),
            (100, 100, 100),
            1,
        )

        # Label text
        tx = sx + LEGEND_SWATCH_SIZE + LEGEND_PADDING
        ty = sy + LEGEND_SWATCH_SIZE - 3
        cv2.putText(
            image,
            label,
            (tx, ty),
            LEGEND_FONT,
            LEGEND_FONT_SCALE,
            LEGEND_TEXT_COLOR,
            LEGEND_FONT_THICKNESS,
            cv2.LINE_AA,
        )
