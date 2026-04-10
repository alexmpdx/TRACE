"""Test intervein region naming on a single specimen."""

import logging
import sys
from pathlib import Path

import cv2
from identify_features.config import PipelineConfig
from identify_features.models.geojson_io import (
    _compute_wing_outline,
    load_detection_geojson,
    load_landmarks_geojson,
)
from identify_features.models.intervein_namer import name_intervein_regions
from identify_features.models.intervein_splitter import (
    assign_vein_tissue_polygons,
    split_merged_intervein_polygons,
)
from identify_features.models.landmark_anchor import anchor_landmarks
from identify_features.models.skeleton import build_skeleton_graph
from identify_features.models.vein_tracer import trace_veins_from_landmarks
from identify_features.models.wing_axis import compute_wing_axis
from identify_features.views.geojson_export import export_geojson
from identify_features.views.overlay import render_overlay_to_file

logging.basicConfig(level=logging.INFO)

BASE = Path(__file__).parent
stem = sys.argv[1] if len(sys.argv) > 1 else "0003"

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

print(f"\nVein tracer output ({len(veins)} entries):")
for v in veins:
    length_str = f"{v.centerline.length:.0f}px" if v.centerline is not None else "no centerline"
    print(f"  {v.vein_id:8s} status={v.status.value:12s} type={v.vein_type.value:12s} {length_str}")

assign_vein_tissue_polygons(veins, skel.median_vein_width_px, config, wing_outline)

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
out_path = VIZ_OUT / f"{stem}_regions.png"
render_overlay_to_file(img, veins, regions, out_path)
print(f"\nOverlay saved: {out_path}")

# Export GeoJSON
geojson_path = VIZ_OUT / f"{stem}_output.geojson"
export_geojson(veins, regions, geojson_path, um_per_px=config.um_per_px)
print(f"GeoJSON saved: {geojson_path}")
