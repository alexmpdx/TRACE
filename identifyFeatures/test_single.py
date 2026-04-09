"""Test intervein region naming on a single specimen."""

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
from identify_features.models.intervein_splitter import split_merged_intervein_polygons
from identify_features.models.landmark_anchor import anchor_landmarks
from identify_features.models.skeleton import build_skeleton_graph
from identify_features.models.topology import REGION_COLORS, VEIN_COLORS
from identify_features.models.vein_tracer import trace_veins_from_landmarks
from identify_features.models.wing_axis import compute_wing_axis

logging.basicConfig(level=logging.INFO)

BASE = Path(__file__).parent
stem = "0003"

det_path = BASE / "geojsons" / f"{stem}_detections.geojson"
lm_path = BASE / "LandmarkLocator_output" / f"{stem}_landmarks.geojson"
img_path = BASE / "OGpics" / f"{stem}.tif"
if not img_path.exists():
    img_path = BASE / "OGpics" / f"{stem}.bmp"

config = PipelineConfig()

vein_polys, intervein_polys = load_detection_geojson(det_path)
landmarks = load_landmarks_geojson(lm_path)
all_polys = vein_polys + intervein_polys
wing_outline = _compute_wing_outline(all_polys)
img = cv2.imread(str(img_path))
image_shape = (img.shape[0], img.shape[1])

print(f"Intervein polygons: {len(intervein_polys)}")

skel = build_skeleton_graph(vein_polys, image_shape, config)
anchor_landmarks(skel, landmarks, config)
wing_axis = compute_wing_axis(landmarks)
veins = trace_veins_from_landmarks(skel, landmarks, wing_outline, config, wing_axis=wing_axis)

print(f"\nVeins found:")
for v in veins:
    if v.centerline is not None:
        print(f"  {v.vein_id}: {v.centerline.length:.0f}px")

VIZ_OUT = BASE / "viz_output"
VIZ_OUT.mkdir(exist_ok=True)
intervein_polys = split_merged_intervein_polygons(
    intervein_polys,
    veins,
    wing_outline,
    image_shape,
    skel.median_vein_width_px,
    config,
    debug_out=VIZ_OUT / f"{stem}_splitter_debug.png",
    debug_base_image=img,
)
print(f"\nIntervein polygons after splitter: {len(intervein_polys)}")

regions = name_intervein_regions(
    intervein_polys,
    veins,
    landmarks,
    config,
    skel.median_vein_width_px,
    wing_outline,
    wing_axis,
)

print(f"\nRegions found ({len(regions)}):")
for r in regions:
    print(f"  {r.name} [{r.status}]: {r.area_px2:.0f}px², veins={r.bounding_veins}")

# Render overlay
overlay = img.copy()
for r in regions:
    if r.polygon is None:
        continue
    color_key = r.name.split(" + ")[0]
    rgb = REGION_COLORS.get(color_key, [128, 128, 128])
    bgr = (rgb[2], rgb[1], rgb[0])
    coords = np.array(r.polygon.exterior.coords, dtype=np.int32)
    cv2.fillPoly(overlay, [coords], bgr)
img_out = cv2.addWeighted(overlay, 0.4, img, 0.6, 0)
for v in veins:
    if v.centerline is None:
        continue
    if v.vein_id.startswith("EV"):
        bgr = (255, 0, 255)  # magenta for ectopic
    else:
        rgb = VEIN_COLORS.get(v.vein_id, [128, 128, 128])
        bgr = (rgb[2], rgb[1], rgb[0])
    pts = np.array(v.centerline.coords, dtype=np.int32)
    cv2.polylines(img_out, [pts], False, bgr, 10)
    if v.vein_id.startswith("EV"):
        mx, my = int(v.centerline.centroid.x), int(v.centerline.centroid.y)
        cv2.putText(img_out, v.vein_id, (mx + 20, my - 20), cv2.FONT_HERSHEY_SIMPLEX, 3.0, (255, 255, 255), 12)
        cv2.putText(img_out, v.vein_id, (mx + 20, my - 20), cv2.FONT_HERSHEY_SIMPLEX, 3.0, (255, 0, 255), 5)
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
print(f"\nOverlay saved: {out_path}")
