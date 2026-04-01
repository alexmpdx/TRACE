"""Shapely geometry helper functions."""

from __future__ import annotations

import math

import numpy as np
from shapely.geometry import LineString, Point


def line_direction(line: LineString, sample_px: float = 80.0) -> tuple[float, float]:
    """Compute the direction vector of a line's first `sample_px` pixels.

    Returns a unit vector (dx, dy) pointing from the start toward the end.
    """
    total = line.length
    if total < 1e-6:
        return (0.0, 0.0)

    sample_dist = min(sample_px, total)
    pt_start = line.interpolate(0)
    pt_end = line.interpolate(sample_dist)

    dx = pt_end.x - pt_start.x
    dy = pt_end.y - pt_start.y
    mag = math.hypot(dx, dy)
    if mag < 1e-6:
        return (0.0, 0.0)
    return (dx / mag, dy / mag)


def line_end_direction(line: LineString, sample_px: float = 80.0) -> tuple[float, float]:
    """Compute the direction vector at the END of a line.

    Returns a unit vector (dx, dy) pointing in the direction the line
    is heading at its endpoint.
    """
    total = line.length
    if total < 1e-6:
        return (0.0, 0.0)

    sample_dist = min(sample_px, total)
    pt_end = line.interpolate(total)
    pt_before = line.interpolate(total - sample_dist)

    dx = pt_end.x - pt_before.x
    dy = pt_end.y - pt_before.y
    mag = math.hypot(dx, dy)
    if mag < 1e-6:
        return (0.0, 0.0)
    return (dx / mag, dy / mag)


def angle_between_vectors(
    v1: tuple[float, float],
    v2: tuple[float, float],
) -> float:
    """Angle in degrees between two 2D vectors (0-180)."""
    dot = v1[0] * v2[0] + v1[1] * v2[1]
    dot = max(-1.0, min(1.0, dot))  # clamp for numerical safety
    return math.degrees(math.acos(dot))


def direction_toward(
    from_pt: Point | tuple[float, float],
    to_pt: Point | tuple[float, float],
) -> tuple[float, float]:
    """Unit vector from one point toward another."""
    if isinstance(from_pt, Point):
        fx, fy = from_pt.x, from_pt.y
    else:
        fx, fy = from_pt
    if isinstance(to_pt, Point):
        tx, ty = to_pt.x, to_pt.y
    else:
        tx, ty = to_pt

    dx = tx - fx
    dy = ty - fy
    mag = math.hypot(dx, dy)
    if mag < 1e-6:
        return (0.0, 0.0)
    return (dx / mag, dy / mag)


def angle_from_pd_axis(
    line: LineString,
    pd_vector: tuple[float, float],
) -> float:
    """Angle (degrees) between a line's principal direction and the PD axis.

    The PD (proximal-distal) vector typically points from wing base to tip.
    Returns 0-90: 0 = parallel to PD axis, 90 = perpendicular.
    """
    d = line_direction(line, sample_px=line.length)
    angle = angle_between_vectors(d, pd_vector)
    # Normalize to 0-90 range (direction doesn't matter)
    if angle > 90:
        angle = 180 - angle
    return angle


def smooth_linestring(line: LineString, sigma: float = 5.0) -> LineString:
    """Gaussian-smooth a LineString's coordinates."""
    from scipy.ndimage import gaussian_filter1d

    coords = np.array(line.coords)
    if len(coords) < 3:
        return line
    smoothed_x = gaussian_filter1d(coords[:, 0], sigma=sigma)
    smoothed_y = gaussian_filter1d(coords[:, 1], sigma=sigma)
    return LineString(np.column_stack([smoothed_x, smoothed_y]))
