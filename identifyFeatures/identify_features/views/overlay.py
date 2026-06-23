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


def _vein_bgr(vein_id: str, overrides: Optional[dict[str, list[int]]] = None) -> tuple[int, int, int]:
    """Return BGR color for a vein. ``overrides`` (RGB) wins over topology defaults.

    All ectopic ids (EV1, EV2, ...) share a single bucket keyed as ``EV`` so the
    user only has to pick one color for "ectopic veins".
    """
    lookup_key = "EV" if vein_id.startswith("EV") else vein_id
    if overrides is not None and lookup_key in overrides:
        rgb = overrides[lookup_key]
    else:
        rgb = VEIN_COLORS.get(lookup_key, list(_DEFAULT_COLOR_BGR))
    return (rgb[2], rgb[1], rgb[0])


def _region_bgr(name: str, overrides: Optional[dict[str, list[int]]] = None) -> tuple[int, int, int]:
    """Return BGR color for a region (uses first name if merged). ``overrides`` (RGB) wins."""
    color_key = name.split(" + ")[0]
    if overrides is not None and color_key in overrides:
        rgb = overrides[color_key]
    else:
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


def _draw_color_key(
    img: np.ndarray,
    veins: list[VeinIdentification],
    vein_color_overrides: Optional[dict[str, list[int]]] = None,
) -> None:
    """Draw a vein color legend in the upper-left corner of the image (in place)."""
    present: list[tuple[str, tuple[int, int, int]]] = []
    seen: set[str] = set()
    for vid in VEIN_AP_ORDER:
        for v in veins:
            if v.vein_id == vid and v.centerline is not None and vid not in seen:
                present.append((vid, _vein_bgr(vid, vein_color_overrides)))
                seen.add(vid)
                break
    # Append ectopic veins after canonicals
    for v in veins:
        if v.vein_id.startswith("EV") and v.centerline is not None and v.vein_id not in seen:
            present.append((v.vein_id, _vein_bgr(v.vein_id, vein_color_overrides)))
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
    vein_color_overrides: Optional[dict[str, list[int]]] = None,
    region_color_overrides: Optional[dict[str, list[int]]] = None,
    vein_opacity: float = 1.0,
    intervein_opacity: float = _REGION_FILL_OPACITY,
    show_color_key: bool = True,
    show_ectopic_labels: bool = True,
    show_region_labels: bool = True,
    vein_simplify_tolerance_px: float = 0.0,
    ectopic_label_font_scale: float = 1.0,
) -> np.ndarray:
    """Render veins and regions as a color overlay on the wing image.

    Layers (bottom to top):
    1. Light-opacity colored fills inside intervein regions  (skipped if not show_regions)
    2. (optional) Vein tissue polygon fills if ``show_vein_tissue``  (skipped if not show_veins)
    3. Vein centerline strokes  (skipped if not show_veins)
    4. Ectopic vein ID labels  (skipped if not show_veins or not show_ectopic_labels)
    5. Region name labels (with [M]/[I] status suffixes)  (skipped if not show_regions or not show_region_labels)
    6. Color-key legend in the upper-left corner  (skipped if not show_veins or not show_color_key)

    Args:
        base_image: BGR base image (e.g. original wing photo).
        veins: Identified veins with centerline and tissue_polygon.
        regions: Named intervein regions with polygon.
        show_vein_tissue: If True (and show_veins), overlay buffered vein tissue polygons.
            Default False: skeleton centerlines only.
        show_veins: If True, draw vein-related layers (tissue fills, centerlines,
            ectopic labels, legend). Default True.
        show_regions: If True, draw intervein region fills and labels. Default True.
        show_color_key: If True (default), draw the vein color legend baked into the
            image. Set False when a separate (e.g. UI-side) legend is shown so the
            key doesn't occlude the wing — the batch pipeline keeps it True.
        show_ectopic_labels: If True (default), draw the "EV1"/"EV2"… text labels
            next to ectopic veins. Set False to draw ectopic centerlines without
            their text (e.g. an intermediate preview where the labels add clutter).
        ectopic_label_font_scale: cv2 font scale for the EV1/EV2… labels (default
            3.0 — the historical hardcoded size). Outline / fill thicknesses scale
            proportionally so the label stays readable at any size.
        show_region_labels: If True (default), draw the intervein region name text
            (with [M]/[I] status suffixes) at each region's centroid. Set False to
            keep the colored region fills but suppress the text labels — useful
            for publication-style figures.
        vein_simplify_tolerance_px: Douglas-Peucker simplification tolerance applied
            to vein centerlines before drawing. 0 (default) = draw the raw skeleton
            polyline; higher values smooth out pixel-level zigzag. A few px is
            usually enough to remove staircasing while preserving vein direction
            changes. Affects only the rendered overlay, not the saved geometry.

    Returns:
        BGR overlay image (same dimensions as base_image).
    """
    img = base_image.copy()

    # Clamp opacities to [0, 1] so a stray out-of-range config doesn't break cv2.addWeighted.
    intervein_opacity = max(0.0, min(1.0, float(intervein_opacity)))
    vein_opacity = max(0.0, min(1.0, float(vein_opacity)))

    # Layer 1: colored region fills (intervein_opacity controls how strongly
    # the colored layer washes the base image; 0 = invisible, 1 = fully painted).
    if show_regions and intervein_opacity > 0:
        region_layer = img.copy()
        for r in regions:
            color = _region_bgr(r.name, region_color_overrides)
            for ring in _polygon_rings(r.polygon):
                cv2.fillPoly(region_layer, [ring], color)
        img = cv2.addWeighted(region_layer, intervein_opacity, img, 1.0 - intervein_opacity, 0)

    # Layer 2: optional vein tissue fills — share vein_opacity with centerlines
    # so "veins" reads as a single channel from the user's perspective. Tissue
    # fills historically used 0.5; keep that as the upper bound by scaling
    # vein_opacity by 0.5 so vein_opacity=1.0 yields the previous look.
    if show_veins and show_vein_tissue and vein_opacity > 0:
        vein_layer = img.copy()
        for v in veins:
            if v.tissue_polygon is None:
                continue
            for ring in _polygon_rings(v.tissue_polygon):
                cv2.fillPoly(vein_layer, [ring], _vein_bgr(v.vein_id, vein_color_overrides))
        tissue_alpha = 0.5 * vein_opacity
        img = cv2.addWeighted(vein_layer, tissue_alpha, img, 1.0 - tissue_alpha, 0)

    # Layer 3: vein centerlines — thickness scales with image size. When
    # vein_opacity < 1, draw on a copy and blend so only the painted strokes
    # become semi-transparent (untouched pixels stay identical).
    if show_veins and vein_opacity > 0:
        h, w = img.shape[:2]
        stroke_scale = max(1.0, min(h, w) / 1800.0)
        vein_thickness = max(3, int(round(5 * stroke_scale)))
        stroke_target = img.copy() if vein_opacity < 1.0 else img
        for v in veins:
            if v.centerline is None:
                continue
            line = v.centerline
            if vein_simplify_tolerance_px > 0:
                simplified = line.simplify(vein_simplify_tolerance_px)
                if not simplified.is_empty and len(simplified.coords) >= 2:
                    line = simplified
            pts = np.array(line.coords, dtype=np.int32)
            cv2.polylines(stroke_target, [pts], False, _vein_bgr(v.vein_id, vein_color_overrides), vein_thickness)
        if vein_opacity < 1.0:
            img = cv2.addWeighted(stroke_target, vein_opacity, img, 1.0 - vein_opacity, 0)

    # Layer 4: ectopic vein labels — also gated by vein_opacity so a fully
    # transparent "vein" channel leaves no EV text either. Outline / fill
    # thicknesses scale linearly with font size so the EV labels look right at
    # any size (historical sizes: scale=3.0, bg_thickness=12, fg_thickness=5).
    if show_veins and show_ectopic_labels and vein_opacity > 0 and ectopic_label_font_scale > 0:
        ev_text_color = _vein_bgr("EV", vein_color_overrides)
        text_target = img.copy() if vein_opacity < 1.0 else img
        ev_bg_thickness = max(1, int(round(ectopic_label_font_scale * 4.0)))
        ev_fg_thickness = max(1, int(round(ectopic_label_font_scale * 5.0 / 3.0)))
        for v in veins:
            if v.centerline is None or not v.vein_id.startswith("EV"):
                continue
            mx, my = int(v.centerline.centroid.x), int(v.centerline.centroid.y)
            cv2.putText(text_target, v.vein_id, (mx + 20, my - 20), cv2.FONT_HERSHEY_SIMPLEX,
                        ectopic_label_font_scale, (255, 255, 255), ev_bg_thickness)
            cv2.putText(text_target, v.vein_id, (mx + 20, my - 20), cv2.FONT_HERSHEY_SIMPLEX,
                        ectopic_label_font_scale, ev_text_color, ev_fg_thickness)
        if vein_opacity < 1.0:
            img = cv2.addWeighted(text_target, vein_opacity, img, 1.0 - vein_opacity, 0)

    # Layer 5: region labels
    if show_regions and show_region_labels:
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

    # Layer 6: color key (vein swatches; only meaningful when veins are drawn).
    # Suppressed when show_color_key is False so a UI-side static legend can
    # stand in without the key occluding the wing.
    if show_veins and show_color_key:
        _draw_color_key(img, veins, vein_color_overrides)

    return img


def render_overlay_to_file(
    base_image: np.ndarray,
    veins: list[VeinIdentification],
    regions: list[InterveinRegion],
    out_path: Path,
    show_vein_tissue: bool = False,
    show_veins: bool = True,
    show_regions: bool = True,
    vein_color_overrides: Optional[dict[str, list[int]]] = None,
    region_color_overrides: Optional[dict[str, list[int]]] = None,
    vein_opacity: float = 1.0,
    intervein_opacity: float = _REGION_FILL_OPACITY,
    show_color_key: bool = True,
    show_ectopic_labels: bool = True,
    show_region_labels: bool = True,
    vein_simplify_tolerance_px: float = 0.0,
    ectopic_label_font_scale: float = 1.0,
) -> None:
    """Render overlay and write to a PNG file."""
    img_out = render_overlay(
        base_image,
        veins,
        regions,
        show_vein_tissue=show_vein_tissue,
        show_veins=show_veins,
        show_regions=show_regions,
        vein_color_overrides=vein_color_overrides,
        region_color_overrides=region_color_overrides,
        vein_opacity=vein_opacity,
        intervein_opacity=intervein_opacity,
        show_color_key=show_color_key,
        show_ectopic_labels=show_ectopic_labels,
        show_region_labels=show_region_labels,
        vein_simplify_tolerance_px=vein_simplify_tolerance_px,
        ectopic_label_font_scale=ectopic_label_font_scale,
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
    show_compartment_labels: bool = True,
) -> Optional[np.ndarray]:
    """Render anterior/posterior compartment overlay with percentage labels.

    Args:
        base_image: BGR base image.
        wing_result: WingResult; used to compute the AP split.
        show_compartment_labels: If True (default), draw the "ANT xx.x%" /
            "POST xx.x%" text at each compartment's centroid. Set False to
            keep the tinted fills without the percentage labels.

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
    if show_compartment_labels:
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
    show_compartment_labels: bool = True,
) -> bool:
    """Render AP overlay and write to PNG. Returns True if successful."""
    img = render_ap_overlay(base_image, wing_result, show_compartment_labels=show_compartment_labels)
    if img is None:
        return False
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), img)
    return True


# ---------------------------------------------------------------------------
# CV ratio overlay
# ---------------------------------------------------------------------------

_LANDMARK_RADIUS = 50
_LINE_THICKNESS = 24
_CV_LINE_BGR = (180, 75, 0)  # cobalt for crossvein distance
_WL_LINE_BGR = (0, 130, 255)  # orange for wing length
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
    - Orange line: wing length (L1-Rs to DTip)
    - Cobalt line: crossvein distance (ACV.p to PCV.a)
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
