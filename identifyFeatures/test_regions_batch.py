"""Batch test: run intervein region naming on all 30 specimens."""

import logging
import sys
from pathlib import Path

import cv2
import numpy as np
from identify_features.config import PipelineConfig
from identify_features.models.geojson_io import (
    _compute_wing_outline,
    load_detection_geojson,
    load_landmarks_geojson,
)
from identify_features.models.intervein_namer import name_intervein_regions
from identify_features.models.landmark_anchor import anchor_landmarks
from identify_features.models.skeleton import build_skeleton_graph
from identify_features.models.topology import REGION_COLORS, VEIN_COLORS
from identify_features.models.vein_tracer import trace_veins_from_landmarks
from identify_features.models.wing_axis import compute_wing_axis

logging.basicConfig(level=logging.WARNING)

BASE = Path(__file__).parent
GEOJSONS = BASE / "geojsons"
LANDMARKS = BASE / "LandmarkLocator_output"
OGPICS = BASE / "OGpics"
VIZ_OUT = BASE / "viz_output"
VIZ_OUT.mkdir(exist_ok=True)

config = PipelineConfig()


def find_specimens():
    """Find all specimen stems by matching detection geojsons."""
    stems = []
    for det in sorted(GEOJSONS.glob("*_detections.geojson")):
        stem = det.name.replace("_detections.geojson", "")
        # Find matching landmark file
        lm_path = LANDMARKS / f"{stem}_landmarks.geojson"
        if not lm_path.exists():
            # Try with space quirk
            lm_path = LANDMARKS / f"{stem} _landmarks.geojson"
        if not lm_path.exists():
            continue
        # Find image
        img_path = None
        for ext in (".tif", ".bmp"):
            p = OGPICS / f"{stem}{ext}"
            if p.exists():
                img_path = p
                break
        if img_path is None:
            continue
        stems.append((stem, det, lm_path, img_path))
    return stems


def render_overlay(stem, img_path, veins, regions):
    """Render veins + named regions overlay."""
    img = cv2.imread(str(img_path))
    if img is None:
        return
    overlay = img.copy()

    # Draw regions as semi-transparent fills
    for r in regions:
        if r.polygon is None:
            continue
        color_key = r.name.split(" + ")[0]  # Use first region name for color
        rgb = REGION_COLORS.get(color_key, [128, 128, 128])
        bgr = (rgb[2], rgb[1], rgb[0])
        coords = np.array(r.polygon.exterior.coords, dtype=np.int32)
        cv2.fillPoly(overlay, [coords], bgr)

    # Blend regions
    img_out = cv2.addWeighted(overlay, 0.4, img, 0.6, 0)

    # Draw vein centerlines
    for v in veins:
        if v.centerline is None:
            continue
        rgb = VEIN_COLORS.get(v.vein_id, [128, 128, 128])
        bgr = (rgb[2], rgb[1], rgb[0])
        pts = np.array(v.centerline.coords, dtype=np.int32)
        cv2.polylines(img_out, [pts], False, bgr, 3)

    # Add region labels
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
        cv2.putText(img_out, label, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), thickness_bg)
        cv2.putText(img_out, label, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), thickness_fg)

    out_path = VIZ_OUT / f"{stem}_regions.png"
    cv2.imwrite(str(out_path), img_out)


def main():
    specimens = find_specimens()
    print(f"Found {len(specimens)} specimens\n")

    summary = []
    for stem, det_path, lm_path, img_path in specimens:
        try:
            vein_polys, intervein_polys = load_detection_geojson(det_path)
            landmarks = load_landmarks_geojson(lm_path)
            all_polys = vein_polys + intervein_polys
            wing_outline = _compute_wing_outline(all_polys)
            img = cv2.imread(str(img_path))
            image_shape = (img.shape[0], img.shape[1])

            skel = build_skeleton_graph(vein_polys, image_shape, config)
            anchor_landmarks(skel, landmarks, config)
            wing_axis = compute_wing_axis(landmarks)
            veins = trace_veins_from_landmarks(skel, landmarks, wing_outline, config, wing_axis=wing_axis)
            regions = name_intervein_regions(
                intervein_polys,
                veins,
                landmarks,
                config,
                skel.median_vein_width_px,
                wing_outline,
                wing_axis,
            )

            region_names = sorted(r.name for r in regions)
            merged = [r.name for r in regions if r.status == "merged"]
            l6_found = any(v.vein_id == "L6" for v in veins if v.centerline is not None)

            render_overlay(stem, img_path, veins, regions)

            line = f"{stem}: {len(regions)}/7 regions"
            if merged:
                line += " | merged: " + ", ".join(merged)
            if l6_found:
                line += " | L6 detected"
            print(line)
            summary.append((stem, len(regions), region_names, merged, l6_found))

        except Exception as e:
            print(f"{stem}: ERROR - {e}")
            summary.append((stem, 0, [], [], False))

    # Summary
    print(f"\n--- Summary ---")
    counts = {}
    for _, n, _, _, _ in summary:
        counts[n] = counts.get(n, 0) + 1
    for n in sorted(counts.keys(), reverse=True):
        print(f"  {counts[n]} specimens with {n}/7 regions")

    # Show which regions are missing
    all_expected = {"marginal", "submarginal", "1st basal", "1st posterior", "discal", "2nd posterior", "3rd posterior"}
    print(f"\nMissing regions breakdown:")
    for stem, n, names, merged, l6 in summary:
        # Flatten merged names for checking
        found = set()
        for name in names:
            for part in name.split(" + "):
                found.add(part)
        missing = all_expected - found
        if missing:
            print(f"  {stem}: missing {missing}" + (" (L6 found)" if l6 else ""))


if __name__ == "__main__":
    main()
