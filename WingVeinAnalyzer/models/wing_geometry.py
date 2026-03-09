"""Wing outline construction, hinge detection, intervein partitioning, and compartments."""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np
from scipy import ndimage
from scipy.ndimage import gaussian_filter1d
from shapely.geometry import LineString, MultiLineString, MultiPolygon, Point, Polygon
from shapely.ops import linemerge, split, unary_union

from WingVeinAnalyzer.models.vein_map import (
    BUFFER_OUTLINE_UM,
    BUFFER_SMOOTH_UM,
    BUFFER_VEIN_UM,
    COMPARTMENT_EXTENSION_UM,
    COMPARTMENT_SIMPLIFY_UM,
    HINGE_EXTENSION_UM,
    MIDLINE_SIGMA_UM,
    MIDLINE_SPACING_UM,
    MIN_HALF_HEIGHT_UM,
    VLINE_EXTENSION_UM,
    um_to_px,
)

logger = logging.getLogger(__name__)


@dataclass
class WingOutline:
    """Complete wing outline as a closed polygon."""

    polygon: Polygon
    anterior_margin: Optional[LineString] = None
    posterior_margin: Optional[LineString] = None


@dataclass
class HingeLandmarks:
    """Landmarks used to define the hinge cut line."""

    subcostal_break: tuple[float, float]
    alula_notch: tuple[float, float]
    hinge_line: LineString


def build_wing_outline(
    polygons: list[Polygon],
    buffer_dist: float | None = None,
    vein_polygons: Optional[list[Polygon]] = None,
) -> WingOutline:
    """Build a wing outline from the union of intervein + vein polygons.

    Buffers each polygon slightly to bridge the vein gaps, then takes
    the union and extracts the outer boundary.  When vein_polygons are
    provided, they are included with a smaller buffer to extend the
    outline to the full wing tip where vein tissue exists.
    """
    if buffer_dist is None:
        buffer_dist = um_to_px(BUFFER_OUTLINE_UM)
    if not polygons:
        return WingOutline(polygon=Polygon())

    buffered = [p.buffer(buffer_dist) for p in polygons]

    # Include vein polygons with smaller buffer to capture distal wing tip
    if vein_polygons:
        for vp in vein_polygons:
            buffered.append(vp.buffer(um_to_px(BUFFER_VEIN_UM)))

    union = unary_union(buffered)

    # Extract the largest polygon from the union
    if isinstance(union, MultiPolygon):
        outline_poly = max(union.geoms, key=lambda p: p.area)
    else:
        outline_poly = union

    # Smooth the outline
    _sb = um_to_px(BUFFER_SMOOTH_UM)
    outline_poly = outline_poly.buffer(_sb).buffer(-_sb)

    return WingOutline(polygon=outline_poly)


@dataclass
class WingMidline:
    """Wing midline with per-sample half-heights for signed-distance normalization."""

    line: LineString
    half_heights: np.ndarray  # parallel array of (max_y - min_y) / 2 at each sample X
    ref_point: Optional[tuple[float, float]] = None  # 3/4-span reference point for vein scoring


def compute_wing_midline(
    polygons: list[Polygon],
    wing_bbox: tuple[float, float, float, float],
    sample_spacing: float | None = None,
    smooth_sigma: float | None = None,
    wing_polygon: Optional[Polygon] = None,
) -> Optional[WingMidline]:
    """Compute the anterior-posterior midline of the wing.

    For each X position across the wing, intersects a vertical line with the
    wing shape to find the Y extent, then takes the midpoint.  The midline
    Y values are Gaussian-smoothed to follow the wing's curvature.

    When *wing_polygon* is provided (from the GeoJSON "wing" annotation),
    it is used directly as the wing shape.  Otherwise falls back to building
    a shape from buffered intervein + vein polygons.
    """
    if sample_spacing is None:
        sample_spacing = um_to_px(MIDLINE_SPACING_UM)
    if smooth_sigma is None:
        smooth_sigma = um_to_px(MIDLINE_SIGMA_UM)

    # Determine wing shape
    if wing_polygon is not None and not wing_polygon.is_empty:
        wing_shape = wing_polygon
    elif polygons:
        buffered = [p.buffer(um_to_px(BUFFER_OUTLINE_UM)) for p in polygons]
        union = unary_union(buffered)
        if isinstance(union, MultiPolygon):
            wing_shape = max(union.geoms, key=lambda p: p.area)
        else:
            wing_shape = union
    else:
        return None

    if wing_shape.is_empty:
        return None

    min_x, min_y, max_x, max_y = wing_bbox
    xs: list[float] = []
    ys: list[float] = []
    hhs: list[float] = []

    x = min_x
    while x <= max_x:
        _vext = um_to_px(VLINE_EXTENSION_UM)
        vline = LineString([(x, min_y - _vext), (x, max_y + _vext)])
        inter = wing_shape.intersection(vline)

        if inter.is_empty:
            x += sample_spacing
            continue

        # Extract Y bounds from intersection
        if isinstance(inter, LineString):
            coords = np.array(inter.coords)
        elif isinstance(inter, MultiLineString):
            coords = np.concatenate([np.array(g.coords) for g in inter.geoms])
        elif isinstance(inter, Point):
            x += sample_spacing
            continue
        else:
            # GeometryCollection or other — extract all coordinates
            try:
                coords = np.array(inter.coords)
            except Exception:
                x += sample_spacing
                continue

        if len(coords) < 2:
            x += sample_spacing
            continue

        y_min_local = coords[:, 1].min()
        y_max_local = coords[:, 1].max()
        mid_y = (y_min_local + y_max_local) / 2.0
        half_h = (y_max_local - y_min_local) / 2.0

        if half_h < um_to_px(MIN_HALF_HEIGHT_UM):  # skip degenerate slices
            x += sample_spacing
            continue

        xs.append(x)
        ys.append(mid_y)
        hhs.append(half_h)

        x += sample_spacing

    if len(xs) < 10:
        logger.warning("Too few midline samples (%d) — skipping midline", len(xs))
        return None

    xs_arr = np.array(xs)
    ys_arr = np.array(ys)
    hhs_arr = np.array(hhs)

    # Smooth both midline Y and half-heights
    sigma_samples = smooth_sigma / sample_spacing
    ys_smooth = gaussian_filter1d(ys_arr, sigma=sigma_samples)
    hhs_smooth = gaussian_filter1d(hhs_arr, sigma=sigma_samples)

    coords = list(zip(xs_arr.tolist(), ys_smooth.tolist()))
    midline = LineString(coords)

    logger.info(
        "Wing midline: %d samples, mean half-height=%.0fpx, source=%s",
        len(xs),
        hhs_smooth.mean(),
        "wing annotation" if wing_polygon is not None else "buffered polygons",
    )

    # Reference point at 3/4 span toward the distal (narrow) end.
    # Determine distal direction from half-heights: the tapered end is distal.
    quarter = max(1, len(hhs_smooth) // 4)
    hh_low = float(hhs_smooth[:quarter].mean())  # mean half-height near min_x
    hh_high = float(hhs_smooth[-quarter:].mean())  # mean half-height near max_x
    # The narrower end is proximal (hinge), the broader end is distal (wing blade)
    if hh_low < hh_high:
        # min_x side is proximal → distal is max_x → ref at 0.75
        ref_frac = 0.75
    else:
        # max_x side is proximal → distal is min_x → ref at 0.25
        ref_frac = 0.25
    ref_x = float(xs_arr[0] + (xs_arr[-1] - xs_arr[0]) * ref_frac)
    ref_y = float(np.interp(ref_x, xs_arr, ys_smooth))

    return WingMidline(line=midline, half_heights=hhs_smooth, ref_point=(ref_x, ref_y))


def _detect_hinge_side(
    polygons: list[Polygon],
    poly_names: dict[int, str],
) -> str:
    """Detect whether the hinge is on the left or right side of the image.

    The hinge is the proximal end where small regions (1st_basal_cell,
    costal_cell) cluster.  Returns "left" or "right".
    """
    proximal_regions = {"1st_basal_cell", "costal_cell", "discal_cell"}
    distal_regions = {"3rd_posterior_cell", "2nd_posterior_cell", "marginal_cell"}

    proximal_xs: list[float] = []
    distal_xs: list[float] = []
    for idx, name in poly_names.items():
        if idx >= len(polygons):
            continue
        cx = polygons[idx].centroid.x
        if name in proximal_regions:
            proximal_xs.append(cx)
        elif name in distal_regions:
            distal_xs.append(cx)

    if proximal_xs and distal_xs:
        prox_mean = np.mean(proximal_xs)
        dist_mean = np.mean(distal_xs)
        side = "left" if prox_mean < dist_mean else "right"
        logger.info("Hinge side detected: %s (proximal X=%.0f, distal X=%.0f)", side, prox_mean, dist_mean)
        return side

    # Fallback: use smallest polygon centroids vs largest
    sorted_by_area = sorted(
        [(idx, polygons[idx]) for idx in poly_names if idx < len(polygons)],
        key=lambda t: t[1].area,
    )
    if len(sorted_by_area) >= 4:
        small_x = np.mean([polygons[i].centroid.x for i, _ in sorted_by_area[:2]])
        large_x = np.mean([polygons[i].centroid.x for i, _ in sorted_by_area[-2:]])
        return "left" if small_x < large_x else "right"

    return "left"  # default assumption


def detect_hinge_landmarks(
    outline: WingOutline,
    polygons: list[Polygon],
    poly_names: dict[int, str],
) -> Optional[HingeLandmarks]:
    """Detect the subcostal break and alula notch to define the hinge line.

    The hinge line separates the wing blade from the hinge/thorax region.
    Automatically detects whether the hinge is on the left or right side.
    """
    if outline.polygon.is_empty:
        return None

    wing_ring = np.array(outline.polygon.exterior.coords)
    min_x = wing_ring[:, 0].min()
    max_x = wing_ring[:, 0].max()

    hinge_side = _detect_hinge_side(polygons, poly_names)
    # When hinge is on the right, "proximal" = max-X; otherwise min-X
    use_max_x = hinge_side == "right"

    def _find_proximal_x(pts: np.ndarray) -> int:
        """Return index of the most proximal (hinge-side) point."""
        return int(pts[:, 0].argmax() if use_max_x else pts[:, 0].argmin())

    # Find the subcostal break: most proximal point of anterior margin
    costal_idx = None
    marginal_idx = None
    for idx, name in poly_names.items():
        if name == "costal_cell":
            costal_idx = idx
        elif name == "marginal_cell":
            marginal_idx = idx

    target = marginal_idx if marginal_idx is not None else costal_idx
    if target is not None:
        ring = np.array(polygons[target].exterior.coords)
        upper_mask = ring[:, 1] < np.median(ring[:, 1])
        if upper_mask.any():
            upper_pts = ring[upper_mask]
            idx_found = _find_proximal_x(upper_pts)
            subcostal = (float(upper_pts[idx_found, 0]), float(upper_pts[idx_found, 1]))
        else:
            idx_found = _find_proximal_x(ring)
            subcostal = (float(ring[idx_found, 0]), float(ring[idx_found, 1]))
    else:
        upper = wing_ring[wing_ring[:, 1] < np.percentile(wing_ring[:, 1], 30)]
        if len(upper) > 0:
            idx_found = _find_proximal_x(upper)
            subcostal = (float(upper[idx_found, 0]), float(upper[idx_found, 1]))
        else:
            px = float(max_x) if use_max_x else float(min_x)
            subcostal = (px, float(wing_ring[:, 1].min()))

    # Alula notch: most proximal indentation on the posterior margin
    posterior_idx = None
    for idx, name in poly_names.items():
        if name == "3rd_posterior_cell":
            posterior_idx = idx

    if posterior_idx is not None:
        ring = np.array(polygons[posterior_idx].exterior.coords)
        lower_mask = ring[:, 1] > np.median(ring[:, 1])
        if lower_mask.any():
            lower_pts = ring[lower_mask]
            idx_found = _find_proximal_x(lower_pts)
            alula = (float(lower_pts[idx_found, 0]), float(lower_pts[idx_found, 1]))
        else:
            idx_found = _find_proximal_x(ring)
            alula = (float(ring[idx_found, 0]), float(ring[idx_found, 1]))
    else:
        lower = wing_ring[wing_ring[:, 1] > np.percentile(wing_ring[:, 1], 70)]
        if len(lower) > 0:
            idx_found = _find_proximal_x(lower)
            alula = (float(lower[idx_found, 0]), float(lower[idx_found, 1]))
        else:
            px = float(max_x) if use_max_x else float(min_x)
            alula = (px, float(wing_ring[:, 1].max()))

    hinge_line = LineString([subcostal, alula])

    logger.info(
        "Hinge landmarks: subcostal=(%.0f, %.0f), alula=(%.0f, %.0f), side=%s",
        subcostal[0],
        subcostal[1],
        alula[0],
        alula[1],
        hinge_side,
    )

    return HingeLandmarks(
        subcostal_break=subcostal,
        alula_notch=alula,
        hinge_line=hinge_line,
    )


def remove_hinge(
    outline: WingOutline,
    landmarks: HingeLandmarks,
    polygons: list[Polygon] = (),
    poly_names: dict[int, str] = {},
) -> Polygon:
    """Remove the hinge region by splitting the wing along the hinge line.

    Returns the distal portion of the wing.  Uses hinge-side detection
    to determine which piece to keep.
    """
    wing = outline.polygon
    if wing.is_empty:
        return wing

    hinge_side = _detect_hinge_side(list(polygons), poly_names) if polygons else "left"

    # Extend the hinge line beyond the wing boundary
    p1 = np.array(landmarks.subcostal_break)
    p2 = np.array(landmarks.alula_notch)
    direction = p2 - p1
    direction = direction / (np.linalg.norm(direction) + 1e-9)

    _hext = um_to_px(HINGE_EXTENSION_UM)
    extended_start = p1 - direction * _hext
    extended_end = p2 + direction * _hext
    cut_line = LineString([extended_start.tolist(), extended_end.tolist()])

    try:
        result = split(wing, cut_line)
        pieces = list(result.geoms)
    except Exception:
        # If split fails, try with a small buffer on the line
        try:
            blade = wing.difference(cut_line.buffer(2))
            if isinstance(blade, MultiPolygon):
                pieces = list(blade.geoms)
            else:
                return blade if isinstance(blade, Polygon) else wing
        except Exception:
            return wing

    if not pieces:
        return wing

    # Keep the distal piece and filter out tiny fragments
    total_area = wing.area
    valid_pieces = [p for p in pieces if isinstance(p, Polygon) and p.area > total_area * 0.05]

    if not valid_pieces:
        return wing

    # Distal = opposite side from hinge
    if hinge_side == "right":
        return min(valid_pieces, key=lambda p: p.centroid.x)
    else:
        return max(valid_pieces, key=lambda p: p.centroid.x)


def partition_intervein_spaces(
    wing_polygon: Polygon,
    polygons: list[Polygon],
    poly_names: dict[int, str],
) -> dict[str, Polygon]:
    """Partition the wing into named intervein regions.

    Uses the pre-annotated intervein polygons, clipped to the wing boundary.
    """
    regions: dict[str, Polygon] = {}

    for idx, name in poly_names.items():
        if idx >= len(polygons):
            continue
        clipped = polygons[idx].intersection(wing_polygon)
        if isinstance(clipped, Polygon) and not clipped.is_empty:
            regions[name] = clipped
        elif isinstance(clipped, MultiPolygon):
            # Keep the largest piece
            largest = max(clipped.geoms, key=lambda g: g.area)
            if isinstance(largest, Polygon):
                regions[name] = largest

    return regions


def partition_by_vein_extension(
    wing_polygon: Polygon,
    vein_lines: dict[str, LineString],
    image_shape: tuple[int, int],
    line_width: int = 2,
    touch_dist: float = 5.0,
    tangent_points: int = 10,
    min_area_frac: float = 0.01,
) -> tuple[list[Polygon], dict[int, set[str]], dict[str, list[LineString]]]:
    """Partition wing into intervein regions by extending vein centerlines.

    Returns (polygons, poly_veins, extension_lines) where poly_veins maps
    polygon index to the set of vein names that border it, and
    extension_lines maps vein_id to the LineStrings grown from endpoints.
    """
    H, W = image_shape

    # --- 1. Rasterize wing as filled mask; outside = barrier ---
    barrier = np.zeros((H, W), dtype=np.int32)
    wing_coords = np.array(wing_polygon.exterior.coords, dtype=np.int32)
    cv2.fillPoly(barrier, [wing_coords], 0)
    # Mark outside wing as barrier (-1)
    wing_mask = np.zeros((H, W), dtype=np.uint8)
    cv2.fillPoly(wing_mask, [wing_coords], 1)
    barrier[wing_mask == 0] = -1
    # Mark wing boundary pixels as barrier (-1)
    cv2.polylines(barrier, [wing_coords], isClosed=True, color=-1, thickness=2)

    # --- 2. Draw vein centerlines as barriers with unique labels ---
    vein_label_map: dict[int, str] = {}  # label → vein_id
    label_counter = 1
    for vein_id, line in vein_lines.items():
        coords = np.array(line.coords, dtype=np.int32)
        if len(coords) < 2:
            continue
        lbl = label_counter
        vein_label_map[lbl] = vein_id
        label_counter += 1
        # Draw polyline with unique label
        for i in range(len(coords) - 1):
            cv2.line(barrier, tuple(coords[i]), tuple(coords[i + 1]), int(lbl), thickness=line_width)

    # --- 3. Identify endpoints needing extension ---
    wing_boundary = wing_polygon.exterior
    active_extensions: list[dict] = []  # {pos, direction, label, trace}

    for vein_id, line in vein_lines.items():
        coords = list(line.coords)
        n = len(coords)
        if n < 2:
            continue
        # Find this vein's label
        lbl = None
        for l, vid in vein_label_map.items():
            if vid == vein_id:
                lbl = l
                break
        if lbl is None:
            continue

        for ep_idx in (0, -1):
            ep = Point(coords[ep_idx])
            # Skip if already near wing boundary or another vein
            dist_to_boundary = wing_boundary.distance(ep)
            near_other = False
            for other_id, other_line in vein_lines.items():
                if other_id == vein_id:
                    continue
                if other_line.distance(ep) < touch_dist:
                    near_other = True
                    break
            if dist_to_boundary < touch_dist or near_other:
                continue

            # Compute tangent direction
            if ep_idx == 0:
                end_idx = min(tangent_points, n - 1)
                dx = coords[0][0] - coords[end_idx][0]
                dy = coords[0][1] - coords[end_idx][1]
            else:
                start_idx = max(-tangent_points - 1, -n)
                dx = coords[-1][0] - coords[start_idx][0]
                dy = coords[-1][1] - coords[start_idx][1]

            length = math.hypot(dx, dy)
            if length < 1e-6:
                continue
            dx /= length
            dy /= length

            start_pt = (float(coords[ep_idx][0]), float(coords[ep_idx][1]))
            active_extensions.append(
                {
                    "x": start_pt[0],
                    "y": start_pt[1],
                    "dx": dx,
                    "dy": dy,
                    "label": lbl,
                    "trace": [start_pt],
                }
            )

    # Keep references to all extension dicts for trace collection after growth
    all_extensions = list(active_extensions)

    # --- 4. Simultaneous growth loop ---
    max_iter = max(H, W)
    for _ in range(max_iter):
        if not active_extensions:
            break
        still_active = []
        for ext in active_extensions:
            nx = ext["x"] + ext["dx"]
            ny = ext["y"] + ext["dy"]
            ix, iy = int(round(nx)), int(round(ny))
            # Bounds check
            if ix < 0 or ix >= W or iy < 0 or iy >= H:
                continue  # out of image
            val = barrier[iy, ix]
            if val != 0 and val != ext["label"]:
                # Hit something (boundary, another vein, or another extension)
                continue
            if val == ext["label"]:
                # Walking over own vein pixels — advance but don't re-mark
                ext["x"] = nx
                ext["y"] = ny
                still_active.append(ext)
                continue
            # Mark as barrier with this vein's label
            barrier[iy, ix] = ext["label"]
            ext["x"] = nx
            ext["y"] = ny
            ext["trace"].append((nx, ny))
            still_active.append(ext)
        active_extensions = still_active

    # Collect extension traces as LineStrings keyed by vein_id
    extension_lines: dict[str, list[LineString]] = {}
    for ext in all_extensions:
        trace = ext["trace"]
        if len(trace) >= 2:
            vid = vein_label_map.get(ext["label"], "?")
            extension_lines.setdefault(vid, []).append(LineString(trace))

    # --- 5. Connected components on free pixels ---
    free_mask = (barrier == 0).astype(np.uint8)
    labeled, num_labels = ndimage.label(free_mask)

    # --- 6. Vectorize → Shapely polygons ---
    wing_area = wing_polygon.area
    result_polys: list[Polygon] = []
    result_poly_veins: dict[int, set[str]] = {}

    for comp_label in range(1, num_labels + 1):
        comp_mask = (labeled == comp_label).astype(np.uint8)
        contours, _ = cv2.findContours(comp_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            continue
        # Take largest contour
        cnt = max(contours, key=cv2.contourArea)
        if len(cnt) < 4:
            continue
        pts = cnt.squeeze()
        if pts.ndim != 2 or pts.shape[0] < 4:
            continue
        poly = Polygon(pts).simplify(1.0)
        if not poly.is_valid:
            poly = poly.buffer(0)
        if poly.is_empty or poly.area < wing_area * min_area_frac:
            continue

        idx = len(result_polys)
        result_polys.append(poly)

        # --- 7. Find adjacent vein labels ---
        dilated = ndimage.binary_dilation(comp_mask, structure=np.ones((3, 3)))
        neighbor_labels = barrier[dilated > 0]
        unique_labels = set(int(v) for v in np.unique(neighbor_labels) if v > 0)
        veins = set()
        for lbl in unique_labels:
            if lbl in vein_label_map:
                veins.add(vein_label_map[lbl])
        result_poly_veins[idx] = veins

    logger.info(
        "Vein-extension partition: %d regions from %d veins, %d extensions grown",
        len(result_polys),
        len(vein_lines),
        len(all_extensions),
    )

    return result_polys, result_poly_veins, extension_lines


def compute_compartments(
    wing_polygon: Polygon,
    l4_line: Optional[LineString],
) -> tuple[Optional[Polygon], Optional[Polygon]]:
    """Split the wing into anterior and posterior compartments along L4.

    Returns (anterior_compartment, posterior_compartment).
    """
    if l4_line is None or wing_polygon.is_empty:
        return None, None

    coords = np.array(l4_line.coords)
    if len(coords) < 2:
        return None, None

    # Simplify the L4 line to remove noise, then extend endpoints
    simplified = l4_line.simplify(um_to_px(COMPARTMENT_SIMPLIFY_UM))
    s_coords = np.array(simplified.coords)
    if len(s_coords) < 2:
        return None, None

    # Extend the line endpoints along their local direction
    ext = um_to_px(COMPARTMENT_EXTENSION_UM)  # extend well beyond wing boundary

    # Start direction (from 2nd point to 1st point)
    start_dir = s_coords[0] - s_coords[min(1, len(s_coords) - 1)]
    start_len = np.linalg.norm(start_dir) + 1e-9
    start_ext = s_coords[0] + (start_dir / start_len) * ext

    # End direction (from 2nd-to-last to last point)
    end_dir = s_coords[-1] - s_coords[max(0, len(s_coords) - 2)]
    end_len = np.linalg.norm(end_dir) + 1e-9
    end_ext = s_coords[-1] + (end_dir / end_len) * ext

    extended = LineString([start_ext.tolist()] + list(simplified.coords) + [end_ext.tolist()])

    # Use buffer-based splitting which is more robust than split()
    # Create a thin strip along L4 and use it to divide the wing
    try:
        result = split(wing_polygon, extended)
        pieces = [g for g in result.geoms if isinstance(g, Polygon)]
    except Exception:
        pieces = []

    # Fallback: use buffer to create a cutting strip
    if len(pieces) < 2:
        try:
            strip = extended.buffer(1.0)
            remainder = wing_polygon.difference(strip)
            if isinstance(remainder, MultiPolygon):
                pieces = [g for g in remainder.geoms if isinstance(g, Polygon) and g.area > 100]
            elif isinstance(remainder, Polygon):
                pieces = [remainder]
            else:
                return None, None
        except Exception:
            return None, None

    if len(pieces) < 2:
        return None, None

    # Filter out tiny fragments
    total_area = wing_polygon.area
    valid = [p for p in pieces if p.area > total_area * 0.05]
    if len(valid) < 2:
        return None, None

    # Anterior = lower Y centroid, posterior = higher Y centroid
    valid.sort(key=lambda p: p.centroid.y)
    return valid[0], valid[-1]


def find_a2_distal_tip(
    polygons: list[Polygon],
    poly_names: dict[int, str],
) -> Optional[tuple[float, float]]:
    """Find the distal tip of the second anal vein (A2).

    A2 terminates on the posterior wing margin. Its distal tip is the
    inflection point on the posterior boundary of the 3rd posterior cell
    where the margin transitions from running distally (roughly horizontal)
    to curving anteriorly (upward toward the wing tip).

    This is detected as the point on the lower boundary with the maximum
    Y-coordinate (most ventral/posterior) that is also past the midpoint
    of the wing span.
    """
    posterior_idx = None
    for idx, name in poly_names.items():
        if name == "3rd_posterior_cell":
            posterior_idx = idx
            break

    if posterior_idx is None or posterior_idx >= len(polygons):
        return None

    poly = polygons[posterior_idx]
    ring = np.array(poly.exterior.coords)

    # Get the wing X span from all polygons
    all_xs = []
    for p in polygons:
        b = p.bounds
        all_xs.extend([b[0], b[2]])
    wing_min_x = min(all_xs)
    wing_max_x = max(all_xs)
    wing_span = wing_max_x - wing_min_x

    # Walk the posterior boundary (lower/high-Y portion)
    # Find the point with maximum Y (most posterior) that is in the
    # distal half of the wing — this is where A2 meets the margin
    median_y = np.median(ring[:, 1])
    lower_mask = ring[:, 1] > median_y
    if not lower_mask.any():
        return None

    lower_pts = ring[lower_mask]
    # A2 tip is the most posterior (max Y) point in the wing
    max_y_idx = lower_pts[:, 1].argmax()
    return (float(lower_pts[max_y_idx, 0]), float(lower_pts[max_y_idx, 1]))
