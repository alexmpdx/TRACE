"""Visualize EDT heatmap + h-maxima peaks for each input polygon."""

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
from skimage.morphology import h_maxima

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
h = max(1, round(mvw * config.intervein_split_h_vw))
print(f"Median vein width: {mvw:.0f}px, h={h}px ({config.intervein_split_h_vw}×vw)")

VIZ_OUT = BASE / "viz_output"

# --- Build combined EDT heatmap across all polygons ---
combined_edt = np.zeros((H, W), dtype=np.float64)
combined_peaks = np.zeros((H, W), dtype=bool)
combined_saddles = np.zeros((H, W), dtype=bool)
combined_poly_mask = np.zeros((H, W), dtype=bool)

for i, p in enumerate(intervein_polys):
    poly_mask = rasterize_polygons([p], (H, W)) > 0
    if not poly_mask.any():
        continue
    edt = distance_transform_edt(poly_mask)
    from scipy.ndimage import gaussian_filter

    edt_smooth = gaussian_filter(edt, sigma=mvw)
    peaks = h_maxima(edt_smooth, h)

    combined_edt = np.maximum(combined_edt, edt)
    combined_peaks |= peaks > 0
    combined_poly_mask |= poly_mask

    n_peaks = 0
    if peaks.any():
        n_comps, peak_labels = cv2.connectedComponents((peaks > 0).astype(np.uint8))
        n_peaks = n_comps - 1

        # Find saddle lines: watershed the polygon's own EDT using peaks as seeds
        if n_peaks >= 2:
            from skimage.segmentation import find_boundaries
            from skimage.segmentation import watershed as ws

            # Seeds from the h-maxima peaks, labeled per connected component
            poly_seeds = np.zeros((H, W), dtype=np.int32)
            for c in range(1, n_comps):
                poly_seeds[peak_labels == c] = c
            # Watershed on inverted EDT (flood from peaks outward through valleys)
            poly_labels = ws(-edt_smooth, markers=poly_seeds, mask=poly_mask)
            saddle_boundary = find_boundaries(poly_labels, mode="thick") & poly_mask
            combined_saddles |= saddle_boundary

    print(f"  P{i}: max_r={edt.max():.0f}px, {n_peaks} peaks")

# --- Render heatmap ---
# Normalize EDT to 0-255 within polygon areas
edt_norm = np.zeros((H, W), dtype=np.uint8)
if combined_edt.max() > 0:
    edt_norm[combined_poly_mask] = (combined_edt[combined_poly_mask] / combined_edt.max() * 255).astype(np.uint8)

# Apply colormap (COLORMAP_JET: blue=low, red=high inscribed radius)
heatmap = cv2.applyColorMap(edt_norm, cv2.COLORMAP_JET)

# Composite: heatmap inside polygons, original image outside
canvas = img.copy()
alpha = 0.7
canvas[combined_poly_mask] = (
    canvas[combined_poly_mask].astype(np.float32) * (1 - alpha) + heatmap[combined_poly_mask].astype(np.float32) * alpha
).astype(np.uint8)

# Draw h threshold contour: where EDT == h (the "split threshold" isoline)
h_contour = np.abs(combined_edt - h) < 1.5
h_contour &= combined_poly_mask
canvas[h_contour] = (255, 255, 255)  # white contour at EDT=h

# Draw saddle lines (watershed boundaries between peaks) in cyan, thickened
from scipy.ndimage import binary_dilation

saddles_thick = binary_dilation(combined_saddles, iterations=3)
canvas[saddles_thick] = (255, 255, 0)  # BGR cyan

# Draw peak regions in bright magenta
canvas[combined_peaks] = (255, 0, 255)  # BGR magenta

# Draw vein centerlines in thin black for reference
for v in veins:
    if v.centerline is None:
        continue
    if v.vein_id.startswith("EV") or v.vein_id == "L6":
        continue
    pts = np.array(v.centerline.coords, dtype=np.int32)
    cv2.polylines(canvas, [pts], False, (0, 0, 0), 2)

# Add legend
y0 = 60
cv2.putText(canvas, f"EDT heatmap (blue=thin, red=fat)", (30, y0), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 6)
cv2.putText(canvas, f"EDT heatmap (blue=thin, red=fat)", (30, y0), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 0), 2)
cv2.putText(
    canvas, f"White contour = h threshold ({h}px)", (30, y0 + 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 6
)
cv2.putText(canvas, f"White contour = h threshold ({h}px)", (30, y0 + 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 0), 2)
cv2.putText(
    canvas, f"Magenta = h-maxima peaks (seeds)", (30, y0 + 100), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 6
)
cv2.putText(
    canvas, f"Magenta = h-maxima peaks (seeds)", (30, y0 + 100), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 0, 255), 2
)
cv2.putText(
    canvas, f"Cyan = saddle lines (split boundaries)", (30, y0 + 150), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 6
)
cv2.putText(
    canvas, f"Cyan = saddle lines (split boundaries)", (30, y0 + 150), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 0), 2
)

out_path = VIZ_OUT / f"{stem}_edt_heatmap.png"
cv2.imwrite(str(out_path), canvas)
print(f"\nSaved: {out_path}")
