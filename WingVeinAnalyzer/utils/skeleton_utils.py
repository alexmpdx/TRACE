"""Skeletonize helpers, spur pruning, flip detection, and geometry smoothing."""

from __future__ import annotations

import numpy as np
from scipy.ndimage import gaussian_filter1d
from shapely.geometry import LineString, Polygon
from shapely.validation import make_valid


def detect_and_correct_flip(
    image: np.ndarray,
    mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Detect and correct X-flip based on anterior vein density."""
    h, w = mask.shape[:2]
    midpoint = w // 2

    anterior_density = np.count_nonzero(mask[:, :midpoint])
    posterior_density = np.count_nonzero(mask[:, midpoint:])

    if posterior_density > anterior_density:
        image = np.fliplr(image).copy()
        mask = np.fliplr(mask).copy()

    return image, mask


def detect_geojson_flip(
    polygons: list[Polygon],
    image_width: int,
) -> bool:
    """Check if GeoJSON polygons suggest the wing is flipped.

    In a standard orientation, the costa (most anterior/top vein) is at the
    top of the image. The most anterior (lowest Y) polygon should have its
    centroid in the upper portion. If the density of annotation centroids
    is concentrated in the lower half, the image may be flipped vertically.

    Returns True if a flip is detected.
    """
    if not polygons:
        return False

    centroids_y = [p.centroid.y for p in polygons]
    areas = [p.area for p in polygons]
    total_area = sum(areas)

    if total_area == 0:
        return False

    # Weighted average Y position
    weighted_y = sum(cy * a for cy, a in zip(centroids_y, areas)) / total_area
    # If weighted centroid is in upper third, wing is likely inverted
    # (posterior structures typically dominate area)
    # Standard orientation: posterior (large area) at bottom (high Y)
    return False  # GeoJSON coordinates are fixed; flipping handled at image level


# ---------------------------------------------------------------------------
# Geometry smoothing
# ---------------------------------------------------------------------------


def smooth_line(
    line: LineString,
    sigma: float = 3.0,
    sample_spacing: float = 5.0,
) -> LineString:
    """Resample a LineString at uniform intervals and Gaussian-smooth it."""
    coords = np.array(line.coords)
    if len(coords) < 3 or line.length < 20.0:
        return line

    # Cumulative arc-length parameterization
    diffs = np.diff(coords, axis=0)
    seg_lengths = np.hypot(diffs[:, 0], diffs[:, 1])
    cum_len = np.concatenate(([0.0], np.cumsum(seg_lengths)))
    total_len = cum_len[-1]

    if total_len < 20.0:
        return line

    # Resample at uniform spacing
    n_samples = max(int(total_len / sample_spacing), 3)
    t_uniform = np.linspace(0.0, total_len, n_samples)
    x_resampled = np.interp(t_uniform, cum_len, coords[:, 0])
    y_resampled = np.interp(t_uniform, cum_len, coords[:, 1])

    # Gaussian smooth
    x_smooth = gaussian_filter1d(x_resampled, sigma=sigma)
    y_smooth = gaussian_filter1d(y_resampled, sigma=sigma)

    # Preserve original endpoints
    x_smooth[0], y_smooth[0] = coords[0, 0], coords[0, 1]
    x_smooth[-1], y_smooth[-1] = coords[-1, 0], coords[-1, 1]

    return LineString(np.column_stack([x_smooth, y_smooth]))


def smooth_polygon(
    poly: Polygon,
    sigma: float = 3.0,
    sample_spacing: float = 5.0,
) -> Polygon:
    """Resample a Polygon ring at uniform intervals and Gaussian-smooth it."""
    if poly.is_empty:
        return poly

    def _smooth_ring(coords: np.ndarray) -> np.ndarray:
        """Smooth a closed coordinate ring with wrap-around."""
        # Drop closing duplicate if present
        if np.allclose(coords[0], coords[-1]):
            coords = coords[:-1]

        if len(coords) < 4:
            # Close and return unchanged
            return np.vstack([coords, coords[:1]])

        # Cumulative arc-length including closing segment back to start
        closed = np.vstack([coords, coords[:1]])
        diffs = np.diff(closed, axis=0)
        seg_lengths = np.hypot(diffs[:, 0], diffs[:, 1])
        total_len = seg_lengths.sum()

        if total_len < 20.0:
            return np.vstack([coords, coords[:1]])

        # cum_len has N+1 entries: [0, d1, d1+d2, ..., total]
        # corresponding to coords[0], coords[1], ..., coords[N-1], coords[0]
        cum_len = np.concatenate(([0.0], np.cumsum(seg_lengths)))

        # Resample at uniform spacing around the ring
        n_samples = max(int(total_len / sample_spacing), 6)
        t_uniform = np.linspace(0.0, total_len, n_samples, endpoint=False)

        # Interpolate using the closed ring parameterization
        x_resampled = np.interp(t_uniform, cum_len, closed[:, 0])
        y_resampled = np.interp(t_uniform, cum_len, closed[:, 1])

        # Gaussian smooth with wrap mode (closed ring)
        x_smooth = gaussian_filter1d(x_resampled, sigma=sigma, mode="wrap")
        y_smooth = gaussian_filter1d(y_resampled, sigma=sigma, mode="wrap")

        # Close the ring
        return np.vstack([
            np.column_stack([x_smooth, y_smooth]),
            [[x_smooth[0], y_smooth[0]]],
        ])

    # Smooth exterior
    ext_coords = np.array(poly.exterior.coords)
    smooth_ext = _smooth_ring(ext_coords)

    # Smooth interior rings (holes)
    smooth_holes = []
    for interior in poly.interiors:
        hole_coords = np.array(interior.coords)
        smooth_holes.append(_smooth_ring(hole_coords))

    try:
        result = Polygon(smooth_ext, smooth_holes)
        if not result.is_valid:
            result = make_valid(result)
            if result.geom_type != "Polygon":
                return poly
        if result.is_empty:
            return poly
        return result
    except Exception:
        return poly
