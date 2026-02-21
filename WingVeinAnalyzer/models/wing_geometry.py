"""Wing outline construction, hinge detection, intervein partitioning, and compartments."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
from shapely.geometry import LineString, MultiPolygon, Point, Polygon
from shapely.ops import linemerge, split, unary_union


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
) -> WingOutline:
    """Build a wing outline from the union of intervein polygons.

    Buffers each polygon slightly to bridge the vein gaps, then takes
    the union and extracts the outer boundary.
    """
    if not polygons:
        return WingOutline(polygon=Polygon())

    buffered = [p.buffer(buffer_dist) for p in polygons]
    union = unary_union(buffered)

    # Extract the largest polygon from the union
    if isinstance(union, MultiPolygon):
        outline_poly = max(union.geoms, key=lambda p: p.area)
    else:
        outline_poly = union

    # Smooth the outline
    outline_poly = outline_poly.buffer(5).buffer(-5)

    return WingOutline(polygon=outline_poly)


def detect_hinge_landmarks(
    outline: WingOutline,
    polygons: list[Polygon],
    poly_names: dict[int, str],
) -> Optional[HingeLandmarks]:
    """Detect the subcostal break and alula notch to define the hinge line.

    The hinge line separates the wing blade from the hinge/thorax region.
    """
    if outline.polygon.is_empty:
        return None

    wing_ring = np.array(outline.polygon.exterior.coords)
    min_x = wing_ring[:, 0].min()
    max_x = wing_ring[:, 0].max()
    wing_span = max_x - min_x

    # Find the subcostal break: most proximal point of anterior margin
    # This is where the costa starts, typically in the upper-left area
    # Use the proximal (leftmost) boundary of the most anterior polygon
    costal_idx = None
    marginal_idx = None
    for idx, name in poly_names.items():
        if name == "costal_cell":
            costal_idx = idx
        elif name == "marginal_cell":
            marginal_idx = idx

    # Subcostal break: leftmost extent of the marginal/costal cell
    target = marginal_idx if marginal_idx is not None else costal_idx
    if target is not None:
        ring = np.array(polygons[target].exterior.coords)
        # Find point with minimum X in upper region
        upper_mask = ring[:, 1] < np.median(ring[:, 1])
        if upper_mask.any():
            upper_pts = ring[upper_mask]
            min_x_idx = upper_pts[:, 0].argmin()
            subcostal = (float(upper_pts[min_x_idx, 0]), float(upper_pts[min_x_idx, 1]))
        else:
            min_x_idx = ring[:, 0].argmin()
            subcostal = (float(ring[min_x_idx, 0]), float(ring[min_x_idx, 1]))
    else:
        # Fallback: upper-left region of wing outline
        upper = wing_ring[wing_ring[:, 1] < np.percentile(wing_ring[:, 1], 30)]
        if len(upper) > 0:
            min_x_idx = upper[:, 0].argmin()
            subcostal = (float(upper[min_x_idx, 0]), float(upper[min_x_idx, 1]))
        else:
            subcostal = (float(min_x), float(wing_ring[:, 1].min()))

    # Alula notch: find the most proximal indentation on the posterior margin
    # This is typically a concavity in the lower-left region of the wing
    posterior_idx = None
    for idx, name in poly_names.items():
        if name == "3rd_posterior_cell":
            posterior_idx = idx

    if posterior_idx is not None:
        ring = np.array(polygons[posterior_idx].exterior.coords)
        # Lower-left region
        lower_mask = ring[:, 1] > np.median(ring[:, 1])
        if lower_mask.any():
            lower_pts = ring[lower_mask]
            # Find the leftmost point in the lower region
            proximal_idx = lower_pts[:, 0].argmin()
            alula = (float(lower_pts[proximal_idx, 0]), float(lower_pts[proximal_idx, 1]))
        else:
            proximal_idx = ring[:, 0].argmin()
            alula = (float(ring[proximal_idx, 0]), float(ring[proximal_idx, 1]))
    else:
        # Fallback: lower-left of wing outline
        lower = wing_ring[wing_ring[:, 1] > np.percentile(wing_ring[:, 1], 70)]
        if len(lower) > 0:
            proximal_idx = lower[:, 0].argmin()
            alula = (float(lower[proximal_idx, 0]), float(lower[proximal_idx, 1]))
        else:
            alula = (float(min_x), float(wing_ring[:, 1].max()))

    hinge_line = LineString([subcostal, alula])

    return HingeLandmarks(
        subcostal_break=subcostal,
        alula_notch=alula,
        hinge_line=hinge_line,
    )


def remove_hinge(
    outline: WingOutline,
    landmarks: HingeLandmarks,
) -> Polygon:
    """Remove the hinge region by splitting the wing along the hinge line.

    Returns the distal portion (larger x centroid) of the wing.
    """
    wing = outline.polygon
    if wing.is_empty:
        return wing

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

    # Keep the distal piece (larger x centroid) and filter out tiny fragments
    total_area = wing.area
    valid_pieces = [p for p in pieces if isinstance(p, Polygon) and p.area > total_area * 0.05]

    if not valid_pieces:
        return wing

    # Return piece with largest x centroid (distal)
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
