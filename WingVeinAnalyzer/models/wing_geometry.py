"""Wing outline construction, hinge detection, intervein partitioning, and compartments."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np
from shapely.geometry import LineString, MultiPolygon, Point, Polygon
from shapely.ops import linemerge, split, unary_union

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
    buffer_dist: float = 20.0,
    vein_polygons: Optional[list[Polygon]] = None,
) -> WingOutline:
    """Build a wing outline from the union of intervein + vein polygons.

    Buffers each polygon slightly to bridge the vein gaps, then takes
    the union and extracts the outer boundary.  When vein_polygons are
    provided, they are included with a smaller buffer to extend the
    outline to the full wing tip where vein tissue exists.
    """
    if not polygons:
        return WingOutline(polygon=Polygon())

    buffered = [p.buffer(buffer_dist) for p in polygons]

    # Include vein polygons with smaller buffer to capture distal wing tip
    if vein_polygons:
        for vp in vein_polygons:
            buffered.append(vp.buffer(5.0))

    union = unary_union(buffered)

    # Extract the largest polygon from the union
    if isinstance(union, MultiPolygon):
        outline_poly = max(union.geoms, key=lambda p: p.area)
    else:
        outline_poly = union

    # Smooth the outline
    outline_poly = outline_poly.buffer(5).buffer(-5)

    return WingOutline(polygon=outline_poly)


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
        logger.info("Hinge side detected: %s (proximal X=%.0f, distal X=%.0f)",
                     side, prox_mean, dist_mean)
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
    use_max_x = (hinge_side == "right")

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

    logger.info("Hinge landmarks: subcostal=(%.0f, %.0f), alula=(%.0f, %.0f), side=%s",
                subcostal[0], subcostal[1], alula[0], alula[1], hinge_side)

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

    extended_start = p1 - direction * 100
    extended_end = p2 + direction * 100
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
    simplified = l4_line.simplify(10.0)
    s_coords = np.array(simplified.coords)
    if len(s_coords) < 2:
        return None, None

    # Extend the line endpoints along their local direction
    ext = 500  # extend well beyond wing boundary

    # Start direction (from 2nd point to 1st point)
    start_dir = s_coords[0] - s_coords[min(1, len(s_coords) - 1)]
    start_len = np.linalg.norm(start_dir) + 1e-9
    start_ext = s_coords[0] + (start_dir / start_len) * ext

    # End direction (from 2nd-to-last to last point)
    end_dir = s_coords[-1] - s_coords[max(0, len(s_coords) - 2)]
    end_len = np.linalg.norm(end_dir) + 1e-9
    end_ext = s_coords[-1] + (end_dir / end_len) * ext

    extended = LineString(
        [start_ext.tolist()] + list(simplified.coords) + [end_ext.tolist()]
    )

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
