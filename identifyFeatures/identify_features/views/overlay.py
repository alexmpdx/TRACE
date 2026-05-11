"""Render named veins and intervein regions as a color overlay on the wing image."""

from __future__ import annotations

import math
from pathlib import Path
from typing import TYPE_CHECKING, Optional

import cv2
import numpy as np
from identify_features.models.datatypes import InterveinRegion, VeinIdentification
from identify_features.models.topology import REGION_COLORS, VEIN_AP_ORDER, VEIN_COLORS
from identify_features.views.csv_export import compute_ap_split

if TYPE_CHECKING:
    from identify_features.models.datatypes import WingResult

# Fallback color for unknown features
_DEFAULT_COLOR_BGR = (128, 128, 128)
# Ectopic vein color (magenta)
_EV_COLOR_BGR = (255, 0, 255)

# Region rendering: light colored fill (low opacity) so the wing image shows through.
_REGION_FILL_OPACITY = 0.2


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


def _polygon_rings(polygon) -> list[np.ndarray]:
    """Yield int32 ring coordinates for a Polygon or MultiPolygon."""
    rings: list[np.ndarray] = []
    if polygon is None:
        return rings
    geoms = polygon.geoms if polygon.geom_type == "MultiPolygon" else [polygon]
    for geom in geoms:
        rings.append(np.array(geom.exterior.coords, dtype=np.int32))
        for interior in geom.interiors:
            rings.append(np.array(interior.coords, dtype=np.int32))
    return rings


def _draw_color_key(img: np.ndarray, veins: list[VeinIdentification]) -> None:
    """Draw a vein color legend in the upper-left corner of the image (in place)."""
    present: list[tuple[str, tuple[int, int, int]]] = []
    seen: set[str] = set()
    for vid in VEIN_AP_ORDER:
        for v in veins:
            if v.vein_id == vid and v.centerline is not None and vid not in seen:
                present.append((vid, _vein_bgr(vid)))
                seen.add(vid)
                break
    # Append ectopic veins after canonicals
    for v in veins:
        if v.vein_id.startswith("EV") and v.centerline is not None and v.vein_id not in seen:
            present.append((v.vein_id, _vein_bgr(v.vein_id)))
            seen.add(v.vein_id)
    if not present:
        return

    # Scale legend proportionally to image size so it reads at full resolution
    # without dominating the view on smaller crops.
    h, w = img.shape[:2]
    scale = max(1.0, min(h, w) / 1800.0)
    pad = int(20 * scale)
    row_h = int(44 * scale)
    swatch_w = int(60 * scale)
    swatch_h = int(28 * scale)
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1.1 * scale
    font_thickness = max(2, int(round(2 * scale)))

    label_w = 0
    for vid, _ in present:
        (tw, _th), _ = cv2.getTextSize(vid, font, font_scale, font_thickness)
        label_w = max(label_w, tw)

    panel_w = pad * 3 + swatch_w + label_w
    panel_h = pad * 2 + row_h * len(present)
    x0, y0 = 30, 30

    # Semi-transparent black background panel
    sub = img[y0 : y0 + panel_h, x0 : x0 + panel_w]
    if sub.size:
        bg = np.zeros_like(sub)
        img[y0 : y0 + panel_h, x0 : x0 + panel_w] = cv2.addWeighted(sub, 0.35, bg, 0.65, 0)
    cv2.rectangle(img, (x0, y0), (x0 + panel_w, y0 + panel_h), (255, 255, 255), 2)

    for i, (vid, color) in enumerate(present):
        row_y = y0 + pad + i * row_h
        sx = x0 + pad
        sy = row_y + (row_h - swatch_h) // 2
        cv2.rectangle(img, (sx, sy), (sx + swatch_w, sy + swatch_h), color, -1)
        cv2.rectangle(img, (sx, sy), (sx + swatch_w, sy + swatch_h), (255, 255, 255), 1)
        tx = sx + swatch_w + pad
        ty = row_y + row_h // 2 + 10
        cv2.putText(img, vid, (tx, ty), font, font_scale, (255, 255, 255), font_thickness)


def render_overlay(
    base_image: np.ndarray,
    veins: list[VeinIdentification],
    regions: list[InterveinRegion],
    show_vein_tissue: bool = False,
    show_veins: bool = True,
    show_regions: bool = True,
) -> np.ndarray:
    """Render veins and regions as a color overlay on the wing image.

    Layers (bottom to top):
    1. Light-opacity colored fills inside intervein regions  (skipped if not show_regions)
    2. (optional) Vein tissue polygon fills if ``show_vein_tissue``  (skipped if not show_veins)
    3. Vein centerline strokes  (skipped if not show_veins)
    4. Ectopic vein ID labels  (skipped if not show_veins)
    5. Region name labels (with [M]/[I] status suffixes)  (skipped if not show_regions)
    6. Color-key legend in the upper-left corner  (skipped if not show_veins; only veins are keyed)

    Args:
        base_image: BGR base image (e.g. original wing photo).
        veins: Identified veins with centerline and tissue_polygon.
        regions: Named intervein regions with polygon.
        show_vein_tissue: If True (and show_veins), overlay buffered vein tissue polygons.
            Default False: skeleton centerlines only.
        show_veins: If True, draw vein-related layers (tissue fills, centerlines,
            ectopic labels, legend). Default True.
        show_regions: If True, draw intervein region fills and labels. Default True.

    Returns:
        BGR overlay image (same dimensions as base_image).
    """
    img = base_image.copy()

    # Layer 1: low-opacity colored region fills
    if show_regions:
        region_layer = img.copy()
        for r in regions:
            color = _region_bgr(r.name)
            for ring in _polygon_rings(r.polygon):
                cv2.fillPoly(region_layer, [ring], color)
        img = cv2.addWeighted(region_layer, _REGION_FILL_OPACITY, img, 1.0 - _REGION_FILL_OPACITY, 0)

    # Layer 2: optional vein tissue fills
    if show_veins and show_vein_tissue:
        vein_layer = img.copy()
        for v in veins:
            if v.tissue_polygon is None:
                continue
            for ring in _polygon_rings(v.tissue_polygon):
                cv2.fillPoly(vein_layer, [ring], _vein_bgr(v.vein_id))
        img = cv2.addWeighted(vein_layer, 0.5, img, 0.5, 0)

    # Layer 3: vein centerlines — thickness scales with image size
    if show_veins:
        h, w = img.shape[:2]
        stroke_scale = max(1.0, min(h, w) / 1800.0)
        vein_thickness = max(3, int(round(5 * stroke_scale)))
        for v in veins:
            if v.centerline is None:
                continue
            pts = np.array(v.centerline.coords, dtype=np.int32)
            cv2.polylines(img, [pts], False, _vein_bgr(v.vein_id), vein_thickness)

    # Layer 4: ectopic vein labels
    if show_veins:
        for v in veins:
            if v.centerline is None or not v.vein_id.startswith("EV"):
                continue
            mx, my = int(v.centerline.centroid.x), int(v.centerline.centroid.y)
            cv2.putText(img, v.vein_id, (mx + 20, my - 20), cv2.FONT_HERSHEY_SIMPLEX, 3.0, (255, 255, 255), 12)
            cv2.putText(img, v.vein_id, (mx + 20, my - 20), cv2.FONT_HERSHEY_SIMPLEX, 3.0, _EV_COLOR_BGR, 5)

    # Layer 5: region labels
    if show_regions:
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

    # Layer 6: color key (vein swatches; only meaningful when veins are drawn)
    if show_veins:
        _draw_color_key(img, veins)

    return img


def render_overlay_to_file(
    base_image: np.ndarray,
    veins: list[VeinIdentification],
    regions: list[InterveinRegion],
    out_path: Path,
    show_vein_tissue: bool = False,
    show_veins: bool = True,
    show_regions: bool = True,
) -> None:
    """Render overlay and write to a PNG file."""
    img_out = render_overlay(
        base_image,
        veins,
        regions,
        show_vein_tissue=show_vein_tissue,
        show_veins=show_veins,
        show_regions=show_regions,
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), img_out)


# ---------------------------------------------------------------------------
# AP compartment overlay
# ---------------------------------------------------------------------------

_ANT_COLOR_BGR = (150, 80, 0)  # warm blue tint
_POST_COLOR_BGR = (0, 60, 180)  # warm red/orange tint


def render_ap_overlay(
    base_image: np.ndarray,
    wing_result: Optional[WingResult],
) -> Optional[np.ndarray]:
    """Render anterior/posterior compartment overlay with percentage labels.

    Returns BGR image, or None if AP split cannot be computed.
    """
    anterior, posterior = compute_ap_split(wing_result)
    if anterior is None or posterior is None:
        return None

    img = base_image.copy()

    # Anterior tint
    layer = img.copy()
    cv2.fillPoly(layer, [np.array(anterior.exterior.coords, dtype=np.int32)], _ANT_COLOR_BGR)
    img = cv2.addWeighted(layer, 0.35, img, 0.65, 0)

    # Posterior tint
    layer = img.copy()
    cv2.fillPoly(layer, [np.array(posterior.exterior.coords, dtype=np.int32)], _POST_COLOR_BGR)
    img = cv2.addWeighted(layer, 0.35, img, 0.65, 0)

    # Percentage labels
    total = anterior.area + posterior.area
    font = cv2.FONT_HERSHEY_SIMPLEX
    for label, geom, color in [
        (f"ANT {anterior.area / total * 100:.1f}%", anterior, (255, 200, 100)),
        (f"POST {posterior.area / total * 100:.1f}%", posterior, (100, 100, 255)),
    ]:
        cx, cy = int(geom.centroid.x), int(geom.centroid.y)
        (tw, th), _ = cv2.getTextSize(label, font, 2.0, 4)
        tx, ty = cx - tw // 2, cy + th // 2
        cv2.putText(img, label, (tx, ty), font, 2.0, (255, 255, 255), 8)
        cv2.putText(img, label, (tx, ty), font, 2.0, color, 4)

    return img


def render_ap_overlay_to_file(
    base_image: np.ndarray,
    wing_result: Optional[WingResult],
    out_path: Path,
) -> bool:
    """Render AP overlay and write to PNG. Returns True if successful."""
    img = render_ap_overlay(base_image, wing_result)
    if img is None:
        return False
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), img)
    return True


# ---------------------------------------------------------------------------
# CV ratio overlay
# ---------------------------------------------------------------------------

_LANDMARK_RADIUS = 15
_LINE_THICKNESS = 4
_CV_LINE_BGR = (0, 200, 200)  # cyan for crossvein distance
_WL_LINE_BGR = (200, 200, 0)  # yellow-ish for wing length
_LANDMARK_BGR = (255, 255, 255)  # white dot fill
_LANDMARK_BORDER_BGR = (0, 0, 0)  # black dot border


def _draw_labeled_point(img, x, y, name, color):
    """Draw a landmark dot with label."""
    cv2.circle(img, (x, y), _LANDMARK_RADIUS, _LANDMARK_BORDER_BGR, -1)
    cv2.circle(img, (x, y), _LANDMARK_RADIUS - 3, color, -1)
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(img, name, (x + _LANDMARK_RADIUS + 8, y + 8), font, 1.5, (255, 255, 255), 6)
    cv2.putText(img, name, (x + _LANDMARK_RADIUS + 8, y + 8), font, 1.5, (0, 0, 0), 2)


def render_cv_ratio_overlay(
    base_image: np.ndarray,
    wing_result: Optional[WingResult],
    um_per_px: Optional[float] = None,
) -> Optional[np.ndarray]:
    """Render wing length and crossvein distance measurement lines with landmarks.

    Shows:
    - Yellow line: wing length (L1-Rs to DTip)
    - Cyan line: crossvein distance (ACV.p to PCV.a)
    - Labeled landmark dots at each endpoint
    - CV ratio value

    Returns BGR image, or None if required landmarks are missing.
    """
    if wing_result is None:
        return None

    landmarks = wing_result.landmarks
    l1rs = landmarks.get("L1-Rs")
    dtip = landmarks.get("DTip")
    acvp = landmarks.get("ACV.p")
    pcva = landmarks.get("PCV.a")

    if not l1rs or not dtip:
        return None

    img = base_image.copy()

    # Wing length line
    pt_l1rs = (int(l1rs.x), int(l1rs.y))
    pt_dtip = (int(dtip.x), int(dtip.y))
    cv2.line(img, pt_l1rs, pt_dtip, _WL_LINE_BGR, _LINE_THICKNESS)
    _draw_labeled_point(img, *pt_l1rs, "L1-Rs", _WL_LINE_BGR)
    _draw_labeled_point(img, *pt_dtip, "DTip", _WL_LINE_BGR)

    # Wing length label at midpoint
    wing_length_px = math.hypot(dtip.x - l1rs.x, dtip.y - l1rs.y)
    wl_mx = (pt_l1rs[0] + pt_dtip[0]) // 2
    wl_my = (pt_l1rs[1] + pt_dtip[1]) // 2 - 25
    wl_text = f"wing length: {wing_length_px:.0f} px"
    if um_per_px:
        wl_text += f" ({wing_length_px * um_per_px:.0f} um)"
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(img, wl_text, (wl_mx - 200, wl_my), font, 1.2, (255, 255, 255), 6)
    cv2.putText(img, wl_text, (wl_mx - 200, wl_my), font, 1.2, _WL_LINE_BGR, 2)

    # Crossvein distance line (if landmarks available)
    if acvp and pcva:
        pt_acvp = (int(acvp.x), int(acvp.y))
        pt_pcva = (int(pcva.x), int(pcva.y))
        cv2.line(img, pt_acvp, pt_pcva, _CV_LINE_BGR, _LINE_THICKNESS)
        _draw_labeled_point(img, *pt_acvp, "ACV.p", _CV_LINE_BGR)
        _draw_labeled_point(img, *pt_pcva, "PCV.a", _CV_LINE_BGR)

        # CV distance label at midpoint
        cv_dist_px = math.hypot(pcva.x - acvp.x, pcva.y - acvp.y)
        cv_mx = (pt_acvp[0] + pt_pcva[0]) // 2
        cv_my = (pt_acvp[1] + pt_pcva[1]) // 2 + 40
        cv_text = f"CV distance: {cv_dist_px:.0f} px"
        if um_per_px:
            cv_text += f" ({cv_dist_px * um_per_px:.0f} um)"
        cv2.putText(img, cv_text, (cv_mx - 200, cv_my), font, 1.2, (255, 255, 255), 6)
        cv2.putText(img, cv_text, (cv_mx - 200, cv_my), font, 1.2, _CV_LINE_BGR, 2)

        # CV ratio in upper-right area
        cv_ratio = cv_dist_px / wing_length_px
        ratio_text = f"CV ratio: {cv_ratio:.4f}"
        cv2.putText(img, ratio_text, (30, 80), font, 2.0, (255, 255, 255), 8)
        cv2.putText(img, ratio_text, (30, 80), font, 2.0, (0, 255, 0), 3)

    return img


def render_cv_ratio_overlay_to_file(
    base_image: np.ndarray,
    wing_result: Optional[WingResult],
    out_path: Path,
    um_per_px: Optional[float] = None,
) -> bool:
    """Render CV ratio overlay and write to PNG. Returns True if successful."""
    img = render_cv_ratio_overlay(base_image, wing_result, um_per_px)
    if img is None:
        return False
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), img)
    return True
