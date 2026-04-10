"""Visualize input polygons, h-maxima seeds, and barrier-topology seeds."""

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
from identify_features.models.landmark_anchor import anchor_landmarks
from identify_features.models.skeleton import build_skeleton_graph
from identify_features.models.vein_tracer import trace_veins_from_landmarks
from identify_features.models.wing_axis import compute_wing_axis
from identify_features.utils.image_utils import rasterize_polygons
from scipy.ndimage import distance_transform_edt
from skimage.segmentation import watershed

BASE = Path(__file__).parent
stem = sys.argv[1] if len(sys.argv) > 1 else "-CTRL_PknRNAi_108870_0008"

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
H, W = img.shape[:2]

skel = build_skeleton_graph(vein_polys, (H, W), config)
anchor_landmarks(skel, landmarks, config)
wing_axis = compute_wing_axis(landmarks)
veins = trace_veins_from_landmarks(skel, landmarks, wing_outline, config, wing_axis=wing_axis)

mvw = skel.median_vein_width_px
print(f"Median vein width: {mvw:.1f}px")
h_maxima_h = max(1, round(mvw * config.intervein_split_h_vw))
vein_barrier_px = int(round(mvw * config.intervein_split_vein_barrier_vw))
wing_buffer_px = int(round(mvw * config.intervein_split_wing_buffer_vw))
print(
    f"h-maxima h: {h_maxima_h}px ({config.intervein_split_h_vw}×vw), vein barrier: {vein_barrier_px}px, wing inset: {wing_buffer_px}px"
)

# Build interior mask (same logic as splitter)
wing_mask_raw = rasterize_polygons([wing_outline], (H, W)) > 0
wing_edt = distance_transform_edt(wing_mask_raw)
wing_mask = wing_edt >= wing_buffer_px

barrier_centerlines = np.zeros((H, W), dtype=np.uint8)
for v in veins:
    if v.centerline is None:
        continue
    if v.vein_id.startswith("EV") or v.vein_id == "L6":
        continue
    coords = np.array(v.centerline.coords, dtype=np.int32)
    if len(coords) >= 2:
        cv2.polylines(barrier_centerlines, [coords], False, 1, 1)

vein_edt = distance_transform_edt(barrier_centerlines == 0)
vein_barrier = vein_edt <= vein_barrier_px
interior_mask = wing_mask & ~vein_barrier
print(f"Interior mask: {interior_mask.sum():,} pixels")

# --- Image 1: Input polygons colored ---
distinct_colors = [
    (255, 100, 100),  # P0 light red
    (255, 180, 80),  # P1 orange
    (255, 240, 100),  # P2 yellow
    (100, 255, 120),  # P3 light green
    (100, 200, 255),  # P4 light blue
    (220, 130, 255),  # P5 light purple
]

img1 = img.copy()
overlay = img.copy()
for i, p in enumerate(intervein_polys):
    color = distinct_colors[i % len(distinct_colors)]
    bgr = (color[2], color[1], color[0])
    coords = np.array(p.exterior.coords, dtype=np.int32)
    cv2.fillPoly(overlay, [coords], bgr)
img1 = cv2.addWeighted(overlay, 0.5, img1, 0.5, 0)
for i, p in enumerate(intervein_polys):
    cx, cy = int(p.centroid.x), int(p.centroid.y)
    label = f"P{i}: {int(p.area):,}px²"
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 1.2, 2)
    cv2.putText(img1, label, (cx - tw // 2, cy + th // 2), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 6)
    cv2.putText(img1, label, (cx - tw // 2, cy + th // 2), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 0), 2)

cv2.imwrite(str(BASE / "viz_output" / f"{stem}_diag1_inputs.png"), img1)
print(f"Saved: viz_output/{stem}_diag1_inputs.png")

# --- Image 2: Current h-maxima seeds ---
from scipy.ndimage import gaussian_filter
from skimage.morphology import h_maxima as h_max_func

img2 = img.copy()
overlay = img.copy()
# Tint barrier mask red
barrier_only = vein_barrier & wing_mask_raw
overlay[barrier_only] = (60, 60, 220)
img2 = cv2.addWeighted(overlay, 0.35, img2, 0.65, 0)

hmaxima_seeds = np.zeros((H, W), dtype=np.int32)
next_label = 1
hmaxima_summary = []
for i, p in enumerate(intervein_polys):
    poly_mask = rasterize_polygons([p], (H, W)) > 0
    if not poly_mask.any():
        continue
    edt = distance_transform_edt(poly_mask)
    edt_smooth = gaussian_filter(edt, sigma=mvw)
    peaks = h_max_func(edt_smooth, h_maxima_h)
    if not peaks.any():
        hmaxima_summary.append((i, 0, "LOST → reseeded"))
        continue
    num, comp_labels = cv2.connectedComponents((peaks > 0).astype(np.uint8))
    hmaxima_summary.append((i, num - 1, f"{num - 1} seeds"))
    for comp in range(1, num):
        hmaxima_seeds[comp_labels == comp] = next_label
        next_label += 1

# Draw seeds in cyan
seed_mask = hmaxima_seeds > 0
img2[seed_mask] = (255, 255, 0)  # cyan in BGR

for i, p in enumerate(intervein_polys):
    cx, cy = int(p.centroid.x), int(p.centroid.y)
    label = f"P{i}: {hmaxima_summary[i][2] if i < len(hmaxima_summary) else '?'}"
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 2)
    cv2.putText(img2, label, (cx - tw // 2, cy + th // 2), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 5)
    cv2.putText(img2, label, (cx - tw // 2, cy + th // 2), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 2)

cv2.imwrite(str(BASE / "viz_output" / f"{stem}_diag2_hmaxima_seeds.png"), img2)
print(f"Saved: viz_output/{stem}_diag2_hmaxima_seeds.png")
print("h-maxima seeds per input polygon:")
for i, n, desc in hmaxima_summary:
    print(f"  P{i}: {desc}")

# --- Image 3: Proposed barrier-topology seeds ---
img3 = img.copy()
overlay = img.copy()
overlay[barrier_only] = (60, 60, 220)
img3 = cv2.addWeighted(overlay, 0.35, img3, 0.65, 0)

barrier_seeds = np.zeros((H, W), dtype=np.int32)
next_label = 1
barrier_summary = []
for i, p in enumerate(intervein_polys):
    poly_mask = rasterize_polygons([p], (H, W)) > 0
    if not poly_mask.any():
        continue
    inner = poly_mask & interior_mask
    if not inner.any():
        barrier_summary.append((i, 0, "LOST → reseeded"))
        continue
    num, comp_labels = cv2.connectedComponents(inner.astype(np.uint8))
    barrier_summary.append((i, num - 1, f"{num - 1} seeds"))
    for comp in range(1, num):
        barrier_seeds[comp_labels == comp] = next_label
        next_label += 1

seed_mask = barrier_seeds > 0
img3[seed_mask] = (255, 255, 0)  # cyan

for i, p in enumerate(intervein_polys):
    cx, cy = int(p.centroid.x), int(p.centroid.y)
    label = f"P{i}: {barrier_summary[i][2] if i < len(barrier_summary) else '?'}"
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 2)
    cv2.putText(img3, label, (cx - tw // 2, cy + th // 2), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 5)
    cv2.putText(img3, label, (cx - tw // 2, cy + th // 2), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 2)

cv2.imwrite(str(BASE / "viz_output" / f"{stem}_diag3_barrier_seeds.png"), img3)
print(f"Saved: viz_output/{stem}_diag3_barrier_seeds.png")
print("Barrier-topology seeds per input polygon:")
for i, n, desc in barrier_summary:
    print(f"  P{i}: {desc}")


# --- Image 4: Run watershed under both schemes, compare label counts ---
def run_watershed(seeds_arr, mask):
    surface = -distance_transform_edt(mask)
    return watershed(surface, markers=seeds_arr, mask=mask)


hmaxima_labels = run_watershed(hmaxima_seeds, interior_mask)
barrier_labels = run_watershed(barrier_seeds, interior_mask)

print(f"\nh-maxima-based watershed: {hmaxima_labels.max()} labels")
print(f"Barrier-based watershed: {barrier_labels.max()} labels")

# Render barrier-based labels with random colors
rng = np.random.default_rng(42)
img4 = img.copy()
overlay = img.copy()
for label in range(1, barrier_labels.max() + 1):
    mask = barrier_labels == label
    color = rng.integers(80, 255, size=3).tolist()
    overlay[mask] = color
img4 = cv2.addWeighted(overlay, 0.5, img4, 0.5, 0)
overlay2 = img4.copy()
overlay2[barrier_only] = (60, 60, 220)
img4 = cv2.addWeighted(overlay2, 0.35, img4, 0.65, 0)
cv2.imwrite(str(BASE / "viz_output" / f"{stem}_diag4_barrier_watershed.png"), img4)
print(f"Saved: viz_output/{stem}_diag4_barrier_watershed.png")

# And h-maxima-based for comparison
img5 = img.copy()
overlay = img.copy()
for label in range(1, hmaxima_labels.max() + 1):
    mask = hmaxima_labels == label
    color = rng.integers(80, 255, size=3).tolist()
    overlay[mask] = color
img5 = cv2.addWeighted(overlay, 0.5, img5, 0.5, 0)
overlay2 = img5.copy()
overlay2[barrier_only] = (60, 60, 220)
img5 = cv2.addWeighted(overlay2, 0.35, img5, 0.65, 0)
cv2.imwrite(str(BASE / "viz_output" / f"{stem}_diag5_hmaxima_watershed.png"), img5)
print(f"Saved: viz_output/{stem}_diag5_hmaxima_watershed.png")
