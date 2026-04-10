"""Render named veins and intervein regions as a color overlay on the wing image."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from identify_features.models.datatypes import InterveinRegion, VeinIdentification
from identify_features.models.topology import REGION_COLORS, VEIN_COLORS

# Fallback color for unknown features
_DEFAULT_COLOR_BGR = (128, 128, 128)
# Ectopic vein color (magenta)
_EV_COLOR_BGR = (255, 0, 255)


def _vein_bgr(vein_id: str) -> tuple[int, int, int]:
    """Return BGR color for a vein."""
    if vein_id.startswith("EV"):
        return _EV_COLOR_BGR
    rgb = VEIN_COLORS.get(vein_id, list(_DEFAULT_COLOR_BGR))
    return (rgb[2], rgb[1], rgb[0])


def _region_bgr(name: str) -> tuple[int, int, int]:
    """Return BGR color for a region (uses first name if merged)."""
    color_key = name.split(" + ")[0]
    rgb = REGION_COLORS.get(color_key, list(_DEFAULT_COLOR_BGR))
    return (rgb[2], rgb[1], rgb[0])


def render_overlay(
    base_image: np.ndarray,
    veins: list[VeinIdentification],
    regions: list[InterveinRegion],
) -> np.ndarray:
    """Render veins and regions as a color overlay on the wing image.

    Layers (bottom to top):
    1. Semi-transparent intervein region fills (40% opacity)
    2. Semi-transparent vein tissue polygon fills (50% opacity)
    3. Vein centerline strokes (3px for canonical, 3px magenta for ectopic)
    4. Ectopic vein ID labels
    5. Region name labels (with [M]/[I] status suffixes)

    Args:
        base_image: BGR base image (e.g. original wing photo).
        veins: Identified veins with centerline and tissue_polygon.
        regions: Named intervein regions with polygon.

    Returns:
        BGR overlay image (same dimensions as base_image).
    """
    img = base_image.copy()

    # Layer 1: region fills
    region_layer = img.copy()
    for r in regions:
        if r.polygon is None:
            continue
        coords = np.array(r.polygon.exterior.coords, dtype=np.int32)
        cv2.fillPoly(region_layer, [coords], _region_bgr(r.name))
    img = cv2.addWeighted(region_layer, 0.4, img, 0.6, 0)

    # Layer 2: vein tissue fills
    vein_layer = img.copy()
    for v in veins:
        if v.tissue_polygon is None:
            continue
        coords = np.array(v.tissue_polygon.exterior.coords, dtype=np.int32)
        cv2.fillPoly(vein_layer, [coords], _vein_bgr(v.vein_id))
    img = cv2.addWeighted(vein_layer, 0.5, img, 0.5, 0)

    # Layer 3: vein centerlines
    for v in veins:
        if v.centerline is None:
            continue
        pts = np.array(v.centerline.coords, dtype=np.int32)
        cv2.polylines(img, [pts], False, _vein_bgr(v.vein_id), 3)

    # Layer 4: ectopic vein labels
    for v in veins:
        if v.centerline is None or not v.vein_id.startswith("EV"):
            continue
        mx, my = int(v.centerline.centroid.x), int(v.centerline.centroid.y)
        cv2.putText(img, v.vein_id, (mx + 20, my - 20), cv2.FONT_HERSHEY_SIMPLEX, 3.0, (255, 255, 255), 12)
        cv2.putText(img, v.vein_id, (mx + 20, my - 20), cv2.FONT_HERSHEY_SIMPLEX, 3.0, _EV_COLOR_BGR, 5)

    # Layer 5: region labels
    for r in regions:
        if r.polygon is None:
            continue
        cx, cy = int(r.polygon.centroid.x), int(r.polygon.centroid.y)
        label = r.name
        if r.status == "merged":
            label += " [M]"
        elif r.status == "inferred":
            label += " [I]"
        font_scale = 2.0
        thickness_bg = 8
        thickness_fg = 3
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness_fg)
        tx = cx - tw // 2
        ty = cy + th // 2
        cv2.putText(img, label, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), thickness_bg)
        cv2.putText(img, label, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), thickness_fg)

    return img


def render_overlay_to_file(
    base_image: np.ndarray,
    veins: list[VeinIdentification],
    regions: list[InterveinRegion],
    out_path: Path,
) -> None:
    """Render overlay and write to a PNG file."""
    img_out = render_overlay(base_image, veins, regions)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), img_out)
