#!/usr/bin/env python3
"""HingeChopper — blacks out the hinge (proximal) region of wing images using landmark GeoJSONs."""

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
from hinge_psd_loader import imread_any


def _write_chopped_image(path: Path, image: np.ndarray) -> None:
    """Write a chopped image; coerce TIFF outputs to OME-TIFF to preserve metadata.

    Falls back to cv2.imwrite for non-TIFF formats and for any failure inside
    tifffile (e.g. if the optional dep isn't installed).
    """
    name_low = path.name.lower()
    is_tiff = path.suffix.lower() in (".tif", ".tiff") or name_low.endswith((".ome.tif", ".ome.tiff"))
    if is_tiff:
        try:
            import tifffile

            if image.ndim == 3 and image.shape[-1] == 3:
                rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                tifffile.imwrite(str(path), rgb, ome=True, photometric="rgb")
                return
            if image.ndim == 3 and image.shape[-1] == 4:
                rgba = cv2.cvtColor(image, cv2.COLOR_BGRA2RGBA)
                tifffile.imwrite(str(path), rgba, ome=True, photometric="rgb")
                return
            if image.ndim == 2:
                tifffile.imwrite(str(path), image, ome=True, photometric="minisblack")
                return
            tifffile.imwrite(str(path), image, ome=True)
            return
        except Exception:
            pass
    cv2.imwrite(str(path), image)


def load_landmarks(path):
    """Parse GeoJSON landmarks, return dict of name → (x, y).

    Skips features where properties.reliable is explicitly False so optional
    landmarks flagged low-confidence by LandmarkLocator don't influence the hinge line.
    """
    with open(path) as f:
        data = json.load(f)
    landmarks = {}
    for feat in data["features"]:
        props = feat.get("properties", {})
        if props.get("reliable") is False:
            continue
        name = props["classification"]["name"]
        x, y = feat["geometry"]["coordinates"]
        landmarks[name] = (x, y)
    return landmarks


# Priority-ordered fallbacks for a "distal-side reference point" used to tell
# which side of the hinge line is anatomically distal (kept) vs proximal (chopped).
# All five are distal-margin landmarks; DTip is preferred but any one of them
# suffices for the side-test, so a single low-confidence dtip prediction no
# longer aborts the whole image when other distal landmarks are reliable.
_DISTAL_REFERENCE_PRIORITY = ("DTip", "L4.d", "L2.d", "L5.d", "L4-L5", "L1-Rs")


def _pick_distal_reference(landmarks):
    """Return the (x, y) of the first available distal-side landmark, or None."""
    for name in _DISTAL_REFERENCE_PRIORITY:
        if name in landmarks:
            return landmarks[name]
    return None


def build_hinge_line(landmarks):
    """Construct ordered hinge line: subcostal_break → [L1-Rs] → [L4-L5] → alula_notch.

    When both L1-Rs and L4-L5 are present, the segment between them is shifted
    proximally (away from a distal-side reference point) by 25% of the distance
    between them, perpendicular to their connecting line. DTip is the preferred
    reference but any distal-margin landmark in `_DISTAL_REFERENCE_PRIORITY` works.
    """
    for req in ("subcostal break", "alula notch"):
        if req not in landmarks:
            raise ValueError(f"Missing required landmark: {req}")
    distal_ref = _pick_distal_reference(landmarks)
    if distal_ref is None:
        raise ValueError(
            "Missing required distal-side landmark " f"(expected one of: {', '.join(_DISTAL_REFERENCE_PRIORITY)})"
        )

    points = [landmarks["subcostal break"]]

    if "L1-Rs" in landmarks and "L4-L5" in landmarks:
        l1rs = np.array(landmarks["L1-Rs"], dtype=np.float64)
        l4l5 = np.array(landmarks["L4-L5"], dtype=np.float64)
        dtip = np.array(distal_ref, dtype=np.float64)

        seg_vec = l4l5 - l1rs
        seg_len = np.linalg.norm(seg_vec)
        offset_dist = 0.30 * seg_len

        # Perpendicular direction (two options — pick the one away from DTip)
        perp = np.array([-seg_vec[1], seg_vec[0]], dtype=np.float64)
        perp = perp / np.linalg.norm(perp)

        # Test which perpendicular direction is proximal (away from DTip)
        midpoint = (l1rs + l4l5) / 2
        if np.dot(perp, dtip - midpoint) > 0:
            perp = -perp  # flip so it points away from DTip

        shifted_l1rs = l1rs + perp * offset_dist
        shifted_l4l5 = l4l5 + perp * offset_dist

        points.append(tuple(shifted_l1rs))
        points.append(tuple(shifted_l4l5))
    else:
        if "L1-Rs" in landmarks:
            points.append(landmarks["L1-Rs"])
        if "L4-L5" in landmarks:
            points.append(landmarks["L4-L5"])

    points.append(landmarks["alula notch"])
    return points


def _line_edge_intersection(px, py, dx, dy, width, height):
    """Find where ray from (px,py) in direction (dx,dy) hits image boundary."""
    candidates = []
    if dx != 0:
        # x = 0
        t = -px / dx
        if t > 0:
            y_at = py + t * dy
            if 0 <= y_at <= height:
                candidates.append((t, 0, y_at))
        # x = width
        t = (width - px) / dx
        if t > 0:
            y_at = py + t * dy
            if 0 <= y_at <= height:
                candidates.append((t, width, y_at))
    if dy != 0:
        # y = 0
        t = -py / dy
        if t > 0:
            x_at = px + t * dx
            if 0 <= x_at <= width:
                candidates.append((t, x_at, 0))
        # y = height
        t = (height - py) / dy
        if t > 0:
            x_at = px + t * dx
            if 0 <= x_at <= width:
                candidates.append((t, x_at, height))
    if not candidates:
        return (px, py)
    # Pick nearest intersection
    candidates.sort(key=lambda c: c[0])
    _, ex, ey = candidates[0]
    return (ex, ey)


def extend_to_image_edges(points, landmarks, width, height):
    """Extend first and last segments of the hinge line to image edges.

    Uses the slope between L1-Rs and L4-L5 for extension direction.
    Falls back to overall line direction if those landmarks are missing.
    """
    points = list(points)

    if "L1-Rs" in landmarks and "L4-L5" in landmarks:
        dx = landmarks["L4-L5"][0] - landmarks["L1-Rs"][0]
        dy = landmarks["L4-L5"][1] - landmarks["L1-Rs"][1]
    else:
        dx = points[-1][0] - points[0][0]
        dy = points[-1][1] - points[0][1]

    # Extend start in the reverse direction
    start_ext = _line_edge_intersection(points[0][0], points[0][1], -dx, -dy, width, height)
    points.insert(0, start_ext)

    # Extend end in the forward direction
    end_ext = _line_edge_intersection(points[-1][0], points[-1][1], dx, dy, width, height)
    points.append(end_ext)

    return points


def make_proximal_mask(hinge_points, dtip, width, height):
    """Create binary mask where True = proximal side (to be blacked out).

    The hinge line splits the image. DTip is on the distal side (keep).
    We black out the opposite side.
    """
    pts = np.array(hinge_points, dtype=np.float64)

    # Build a polygon covering one side of the hinge line by extending to corners.
    # Strategy: the hinge line goes from one edge to another. We need to figure out
    # which side DTip is on, then fill the OTHER side.

    # Use cv2.fillPoly: build polygon for each side by tracing line + image boundary.
    # Simpler approach: use the sign of the cross product to determine which side
    # DTip is on relative to the hinge line direction.

    # Create mask by filling below/above the hinge polyline
    # First, determine which side of the line DTip falls on
    # Use the signed area / cross product test with the line's overall direction

    # Overall direction: first point to last point
    line_dx = pts[-1][0] - pts[0][0]
    line_dy = pts[-1][1] - pts[0][1]

    # Vector from line start to DTip
    to_dtip_x = dtip[0] - pts[0][0]
    to_dtip_y = dtip[1] - pts[0][1]

    # Cross product: positive = DTip is on the left of the line direction
    cross = line_dx * to_dtip_y - line_dy * to_dtip_x

    # Build proximal polygon: hinge line + image boundary on the proximal side
    # The proximal side is opposite to DTip
    line_pts_int = np.round(pts).astype(np.int32)

    # We need to close the polygon along the image edges.
    # Strategy: trace hinge line, then follow image boundary back to start.
    corners = np.array(
        [
            [0, 0],
            [width, 0],
            [width, height],
            [0, height],
        ],
        dtype=np.int32,
    )

    # DTip side has cross > 0 (left) or cross < 0 (right)
    # We want the OTHER side for the proximal mask.
    # Build polygon: hinge line points + corners on the proximal side

    # Determine which corners are on the proximal side (opposite of DTip)
    proximal_corners = []
    for corner in corners:
        to_corner_x = corner[0] - pts[0][0]
        to_corner_y = corner[1] - pts[0][1]
        corner_cross = line_dx * to_corner_y - line_dy * to_corner_x
        # Proximal = opposite sign of DTip's cross
        if (cross > 0 and corner_cross <= 0) or (cross < 0 and corner_cross >= 0):
            proximal_corners.append(corner)

    # Order proximal corners to form a proper polygon boundary
    # Sort by angle from centroid of the proximal corners
    if len(proximal_corners) > 1:
        centroid = np.mean(proximal_corners, axis=0)
        proximal_corners.sort(key=lambda c: np.arctan2(c[1] - centroid[1], c[0] - centroid[0]))

    # Build the proximal polygon: hinge line + proximal corners
    # The polygon is: start_ext → ... → end_ext → proximal corners (in order that
    # traces the image boundary from end_ext back to start_ext)
    polygon_pts = list(line_pts_int)

    # We need corners ordered so they trace the boundary from the end of the line
    # back to the start of the line, on the proximal side.
    # Find which corners to traverse by walking the image boundary on the proximal side.

    # Image boundary order (clockwise): (0,0) → (W,0) → (W,H) → (0,H) → (0,0)
    all_corners_cw = [(0, 0), (width, 0), (width, height), (0, height)]

    # Find which boundary segment the line endpoints are on
    def boundary_position(pt):
        """Return a parameter 0-4 representing position along clockwise boundary."""
        x, y = pt[0], pt[1]
        # Top edge: y≈0
        if y <= 0:
            return x / width  # 0 to 1
        # Right edge: x≈width
        if x >= width:
            return 1 + y / height  # 1 to 2
        # Bottom edge: y≈height
        if y >= height:
            return 2 + (width - x) / width  # 2 to 3
        # Left edge: x≈0
        return 3 + (height - y) / height  # 3 to 4

    start_pos = boundary_position(polygon_pts[0])
    end_pos = boundary_position(polygon_pts[-1])

    # Corner positions
    corner_positions = [
        (0.0, (0, 0)),
        (1.0, (width, 0)),
        (2.0, (width, height)),
        (3.0, (0, height)),
    ]

    # We can go clockwise or counterclockwise from end to start.
    # One path is on the DTip side, the other on the proximal side.
    # Collect corners for both paths and check which is proximal.

    def collect_corners_cw(from_pos, to_pos):
        """Collect corners encountered going clockwise from from_pos to to_pos."""
        result = []
        for cp, corner in corner_positions:
            # Normalize: going clockwise, cp is between from_pos and to_pos
            if from_pos < to_pos:
                if from_pos < cp <= to_pos:
                    result.append((cp, corner))
            else:  # wraps around
                if cp > from_pos or cp <= to_pos:
                    result.append((cp, corner))
        result.sort(key=lambda x: (x[0] - from_pos) % 4)
        return [c for _, c in result]

    path_cw = collect_corners_cw(end_pos, start_pos)
    path_ccw = collect_corners_cw(start_pos, end_pos)
    path_ccw.reverse()  # reverse to go counterclockwise

    # Check which path is on the proximal side by testing a point on each path
    def is_proximal_path(path_corners):
        """Check if a boundary path (set of corners) is on the proximal side."""
        if not path_corners:
            # No corners on this path — test midpoint of the boundary segment
            mid_x = (polygon_pts[-1][0] + polygon_pts[0][0]) / 2
            mid_y = (polygon_pts[-1][1] + polygon_pts[0][1]) / 2
            # Clamp to boundary
            test_pt = (mid_x, mid_y)
        else:
            test_pt = path_corners[0]
        to_test_x = test_pt[0] - pts[0][0]
        to_test_y = test_pt[1] - pts[0][1]
        test_cross = line_dx * to_test_y - line_dy * to_test_x
        # Proximal = opposite sign of DTip
        return (cross > 0 and test_cross <= 0) or (cross < 0 and test_cross >= 0)

    if is_proximal_path(path_cw):
        boundary_corners = path_cw
    else:
        boundary_corners = path_ccw

    # Build final polygon: hinge line → boundary corners → back to start
    final_polygon = list(polygon_pts) + [np.array(c, dtype=np.int32) for c in boundary_corners]
    final_polygon = np.array(final_polygon, dtype=np.int32)

    mask = np.zeros((height, width), dtype=np.uint8)
    cv2.fillPoly(mask, [final_polygon], 255)
    return mask > 0


def chop_hinge(image_path, landmarks_path, output_path):
    """Load image and landmarks, black out proximal region, save result."""
    image = imread_any(image_path, cv2.IMREAD_UNCHANGED)
    if image is None:
        raise FileNotFoundError(f"Could not load image: {image_path}")

    height, width = image.shape[:2]
    landmarks = load_landmarks(landmarks_path)
    hinge_pts = build_hinge_line(landmarks)
    extended_pts = extend_to_image_edges(hinge_pts, landmarks, width, height)
    dtip = _pick_distal_reference(landmarks)
    proximal_mask = make_proximal_mask(extended_pts, dtip, width, height)

    # Black out proximal region
    image[proximal_mask] = 0

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    _write_chopped_image(output_path, image)
    print(f"Saved: {output_path}")


def predict_landmarks(image_path, checkpoint_path):
    """Run LandmarkLocator on an image and return landmarks dict with GeoJSON names.

    `checkpoint_path` may be a single .pt file or a fold folder (containing
    best_fold*.pt files), in which case an ensemble is used.
    """
    from landmark_locator import make_predictor

    predictor = predict_landmarks._predictor
    if predictor is None or predict_landmarks._checkpoint != checkpoint_path:
        predictor = make_predictor(Path(checkpoint_path))
        predict_landmarks._predictor = predictor
        predict_landmarks._checkpoint = checkpoint_path

    result = predictor.predict_from_path(Path(image_path))

    # Convert internal names to GeoJSON names
    landmark_to_geojson = {v: k for k, v in predictor.geojson_to_landmark.items()}
    landmarks = {}
    for internal_name, coords in result["landmarks"].items():
        geojson_name = landmark_to_geojson.get(internal_name, internal_name)
        landmarks[geojson_name] = coords
    return landmarks


predict_landmarks._predictor = None
predict_landmarks._checkpoint = None


def landmarks_to_geojson(landmarks):
    """Convert landmarks dict to GeoJSON FeatureCollection."""
    features = []
    for name, (x, y) in landmarks.items():
        features.append(
            {
                "type": "Feature",
                "geometry": {"type": "Point", "coordinates": [x, y]},
                "properties": {"classification": {"name": name}},
            }
        )
    return {"type": "FeatureCollection", "features": features}


def chop_hinge_from_landmarks(image_path, landmarks, output_path):
    """Black out proximal region using a landmarks dict (no GeoJSON file needed)."""
    image = imread_any(image_path, cv2.IMREAD_UNCHANGED)
    if image is None:
        raise FileNotFoundError(f"Could not load image: {image_path}")

    height, width = image.shape[:2]
    hinge_pts = build_hinge_line(landmarks)
    extended_pts = extend_to_image_edges(hinge_pts, landmarks, width, height)
    dtip = _pick_distal_reference(landmarks)
    proximal_mask = make_proximal_mask(extended_pts, dtip, width, height)

    image[proximal_mask] = 0

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    _write_chopped_image(output_path, image)
    print(f"Saved: {output_path}")


def pair_files(pics_folder, landmarks_folder):
    """Pair image files with landmark GeoJSONs by matching stem."""
    pics_folder = Path(pics_folder)
    landmarks_folder = Path(landmarks_folder)

    image_exts = {".tif", ".tiff", ".bmp", ".png", ".jpg", ".jpeg", ".psd", ".psb"}
    images = {f.stem: f for f in pics_folder.iterdir() if f.suffix.lower() in image_exts}

    pairs = []
    for lm_file in sorted(landmarks_folder.glob("*_landmarks.geojson")):
        # Strip _landmarks suffix to get the image stem
        stem = lm_file.stem.replace("_landmarks", "")
        if stem in images:
            pairs.append((images[stem], lm_file))
        else:
            print(f"Warning: no image found for {lm_file.name}", file=sys.stderr)

    return pairs


def main():
    parser = argparse.ArgumentParser(description="Black out hinge region of wing images using landmarks.")
    parser.add_argument("image_or_pics", help="Image path (single mode) or pics folder (batch/predict mode)")
    parser.add_argument(
        "landmarks",
        nargs="?",
        default=None,
        help="Landmarks GeoJSON path (single) or landmarks folder (batch). " "Not needed with --predict.",
    )
    parser.add_argument("-o", "--output", required=True, help="Output path (file or folder)")
    parser.add_argument("--batch", action="store_true", help="Batch mode: process folders")
    parser.add_argument(
        "--predict",
        metavar="CHECKPOINT",
        help="Run LandmarkLocator to predict landmarks from images. "
        "Provide path to model checkpoint (.pt). "
        "Saves GeoJSON files to <output>/landmarks/ and chopped images to <output>/.",
    )

    args = parser.parse_args()

    if args.predict:
        pics_folder = Path(args.image_or_pics)
        image_exts = {".tif", ".tiff", ".bmp", ".png", ".jpg", ".jpeg", ".psd", ".psb"}
        images = sorted(f for f in pics_folder.iterdir() if f.suffix.lower() in image_exts)
        if not images:
            print(f"No images found in {pics_folder}", file=sys.stderr)
            sys.exit(1)

        output_dir = Path(args.output)
        landmarks_dir = output_dir / "landmarks"
        output_dir.mkdir(parents=True, exist_ok=True)
        landmarks_dir.mkdir(parents=True, exist_ok=True)

        for img_path in images:
            try:
                landmarks = predict_landmarks(img_path, args.predict)
                # Save GeoJSON
                geojson_path = landmarks_dir / (img_path.stem + "_landmarks.geojson")
                with open(geojson_path, "w") as f:
                    json.dump(landmarks_to_geojson(landmarks), f, indent=2)
                print(f"Landmarks: {geojson_path}")
                # Chop hinge
                out_path = output_dir / (img_path.stem + ".tif")
                chop_hinge_from_landmarks(img_path, landmarks, out_path)
            except Exception as e:
                print(f"Error processing {img_path.name}: {e}", file=sys.stderr)
    elif args.batch:
        if not args.landmarks:
            print("Batch mode requires a landmarks folder argument.", file=sys.stderr)
            sys.exit(1)
        pairs = pair_files(args.image_or_pics, args.landmarks)
        if not pairs:
            print("No image-landmark pairs found.", file=sys.stderr)
            sys.exit(1)
        output_dir = Path(args.output)
        output_dir.mkdir(parents=True, exist_ok=True)
        for img_path, lm_path in pairs:
            out_path = output_dir / (img_path.stem + ".tif")
            try:
                chop_hinge(img_path, lm_path, out_path)
            except Exception as e:
                print(f"Error processing {img_path.name}: {e}", file=sys.stderr)
    else:
        if not args.landmarks:
            print("Single mode requires a landmarks GeoJSON argument.", file=sys.stderr)
            sys.exit(1)
        chop_hinge(args.image_or_pics, args.landmarks, args.output)


if __name__ == "__main__":
    main()
