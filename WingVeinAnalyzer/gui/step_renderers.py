"""Render input/output BGR images for each pipeline step."""

from __future__ import annotations

from typing import Optional

import cv2
import numpy as np
from shapely.geometry import LineString, Polygon

from WingVeinAnalyzer.gui.step_runner import StepState
from WingVeinAnalyzer.models.vein_map import (
    INTERVEIN_COLORS,
    VEIN_COLORS,
)

# Distinct colors for polygon/segment visualization (BGR)
SEGMENT_COLORS = [
    (0, 0, 255),  # red
    (0, 200, 0),  # green
    (255, 0, 0),  # blue
    (0, 200, 255),  # yellow
    (255, 0, 255),  # magenta
    (255, 255, 0),  # cyan
    (0, 128, 255),  # orange
    (128, 0, 255),  # pink
    (128, 255, 0),  # spring
    (255, 128, 0),  # sky
]


def render_step(
    step_index: int,
    state: StepState,
    prev_state: Optional[StepState] = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Render the input (left) and output (right) images for a step.

    Returns (left_bgr, right_bgr).
    """
    renderers = {
        0: _render_load,
        1: _render_skeleton,
        2: _render_junctions,
        3: _render_merge,
        4: _render_split,
        5: _render_costa,
        6: _render_longitudinals,
        7: _render_crossveins,
        8: _render_regions,
        9: _render_poly_split,
        10: _render_validation,
        11: _render_outline,
        12: _render_hinge,
        13: _render_compartments,
        14: _render_measurements,
        15: _render_overlays,
    }
    renderer = renderers.get(step_index)
    if renderer is None:
        blank = _blank(state)
        return blank, blank
    return renderer(state, prev_state)


# ------------------------------------------------------------------
# Individual step renderers
# ------------------------------------------------------------------


def _render_load(state: StepState, prev: Optional[StepState]) -> tuple[np.ndarray, np.ndarray]:
    """Step 0: Raw image → image + polygon outlines + vein mask."""
    left = state.image.copy() if state.image is not None else _blank(state)

    right = state.image.copy() if state.image is not None else _blank(state)

    # Draw intervein polygon outlines
    if state.polygons:
        for i, poly in enumerate(state.polygons):
            color = SEGMENT_COLORS[i % len(SEGMENT_COLORS)]
            _draw_polygon_outline(right, poly, color, thickness=3)
            cx, cy = int(poly.centroid.x), int(poly.centroid.y)
            cv2.putText(
                right,
                f"P{i}",
                (cx - 20, cy),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                color,
                2,
                cv2.LINE_AA,
            )

    # Draw vein mask overlay (semi-transparent red)
    if state.vein_polygons:
        from WingVeinAnalyzer.models.vein_skeleton import _fill_polygon

        h, w = right.shape[:2]
        mask = np.zeros((h, w), dtype=np.uint8)
        for poly in state.vein_polygons:
            _fill_polygon(mask, poly, 255)
        overlay = np.zeros_like(right)
        overlay[:, :, 2] = mask  # red channel
        right = cv2.addWeighted(right, 0.7, overlay, 0.3, 0)

    return left, right


def _render_skeleton(state: StepState, prev: Optional[StepState]) -> tuple[np.ndarray, np.ndarray]:
    """Step 1: Vein mask → skeleton centerlines."""
    # Left: vein mask on image
    left = state.image.copy()
    if state.vein_mask is not None:
        overlay = np.zeros_like(left)
        overlay[:, :, 2] = (state.vein_mask > 0).astype(np.uint8) * 255
        left = cv2.addWeighted(left, 0.7, overlay, 0.3, 0)

    # Right: traced centerlines on image
    right = state.image.copy()
    bridge_keys = set((state.bridge_segments or {}).keys())
    if state.centerlines:
        for idx, (key, line) in enumerate(state.centerlines.items()):
            color = SEGMENT_COLORS[idx % len(SEGMENT_COLORS)]
            if key in bridge_keys:
                # Draw the original (non-bridged) portion solid, bridge dashed
                bridge_line = state.bridge_segments[key]
                _draw_linestring(right, line, color, thickness=3)
                _draw_dashed_linestring(right, bridge_line, (255, 255, 255), thickness=3, dash_len=3)
            else:
                _draw_linestring(right, line, color, thickness=3)
            # Label at midpoint
            coords = list(line.coords)
            mid = coords[len(coords) // 2]
            label_suffix = " [bridged]" if key in bridge_keys else ""
            cv2.putText(
                right,
                f"{key}{label_suffix}",
                (int(mid[0]) - 30, int(mid[1]) - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                color,
                2,
                cv2.LINE_AA,
            )

    return left, right


def _render_junctions(state: StepState, prev: Optional[StepState]) -> tuple[np.ndarray, np.ndarray]:
    """Step 5: Centerlines → centerlines + junction point markers."""
    # Left: just centerlines
    left = state.image.copy()
    if state.centerlines:
        for idx, (key, line) in enumerate(state.centerlines.items()):
            color = SEGMENT_COLORS[idx % len(SEGMENT_COLORS)]
            _draw_linestring(left, line, color, thickness=2)

    # Right: centerlines + junction markers
    right = left.copy()
    if state.junctions:
        for junc in state.junctions:
            cv2.circle(
                right,
                (int(junc.x), int(junc.y)),
                15,
                (0, 255, 255),
                -1,  # filled yellow circle
            )
            cv2.circle(
                right,
                (int(junc.x), int(junc.y)),
                15,
                (0, 0, 0),
                2,  # black border
            )
            # Show number of converging segments
            n = len(junc.segment_keys)
            cv2.putText(
                right,
                str(n),
                (int(junc.x) - 6, int(junc.y) + 6),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 0, 0),
                2,
                cv2.LINE_AA,
            )

    return left, right


def _render_merge(state: StepState, prev: Optional[StepState]) -> tuple[np.ndarray, np.ndarray]:
    """Step 6: Separate segments → merged paths (same-color groups)."""
    # Left: individual centerline segments
    left = state.image.copy()
    if state.centerlines:
        for idx, (key, line) in enumerate(state.centerlines.items()):
            color = SEGMENT_COLORS[idx % len(SEGMENT_COLORS)]
            _draw_linestring(left, line, color, thickness=2)

    # Right: merged paths with thicker lines
    right = state.image.copy()
    if state.merged_paths:
        for idx, mp in enumerate(state.merged_paths):
            color = SEGMENT_COLORS[idx % len(SEGMENT_COLORS)]
            _draw_linestring(right, mp.line, color, thickness=4)
            coords = list(mp.line.coords)
            mid = coords[len(coords) // 2]
            label = f"M{idx} ({mp.line.length:.0f}px)"
            cv2.putText(
                right,
                label,
                (int(mid[0]) - 40, int(mid[1]) - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                color,
                2,
                cv2.LINE_AA,
            )

    return left, right


def _render_split(state: StepState, prev: Optional[StepState]) -> tuple[np.ndarray, np.ndarray]:
    """Step 4: Merged paths → split paths with landmark annotations."""
    # Left: merged paths (before split) + fork landmarks
    left = state.image.copy()
    if state.merged_paths:
        for idx, mp in enumerate(state.merged_paths):
            color = SEGMENT_COLORS[idx % len(SEGMENT_COLORS)]
            _draw_linestring(left, mp.line, color, thickness=3)
    _draw_fork_landmarks(left, state)

    # Right: all paths after splitting (before classification) + fork landmarks
    right = state.image.copy()
    if state.split_paths:
        for idx, mp in enumerate(state.split_paths):
            color = SEGMENT_COLORS[idx % len(SEGMENT_COLORS)]
            _draw_linestring(right, mp.line, color, thickness=3)
            coords = list(mp.line.coords)
            mid = coords[len(coords) // 2]
            cv2.putText(
                right,
                f"S{idx} ({mp.length_px:.0f}px)",
                (int(mid[0]) - 40, int(mid[1]) - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                color,
                2,
                cv2.LINE_AA,
            )
    _draw_fork_landmarks(right, state)

    return left, right


def _render_costa(state: StepState, prev: Optional[StepState]) -> tuple[np.ndarray, np.ndarray]:
    """Step 5: Show costa region mask and extracted costa vein."""
    # Left: all split paths + costa region overlay
    left = state.image.copy()
    if state.costa_region is not None:
        costa_overlay = np.zeros_like(left)
        costa_overlay[state.costa_region] = (0, 255, 255)  # yellow
        left = cv2.addWeighted(left, 0.7, costa_overlay, 0.3, 0)
    if state.split_paths:
        for idx, mp in enumerate(state.split_paths):
            color = SEGMENT_COLORS[idx % len(SEGMENT_COLORS)]
            _draw_linestring(left, mp.line, color, thickness=3)

    # Right: costa vein highlighted, other paths dimmed
    right = state.image.copy()
    if state.costa_region is not None:
        costa_overlay = np.zeros_like(right)
        costa_overlay[state.costa_region] = (0, 255, 255)
        right = cv2.addWeighted(right, 0.7, costa_overlay, 0.3, 0)
    if state.assignments:
        # Draw non-costa veins in gray
        for a in state.assignments:
            if a.line is not None and a.vein_id != "costa":
                _draw_linestring(right, a.line, (128, 128, 128), thickness=2)
        # Draw costa in bright white
        for a in state.assignments:
            if a.line is not None and a.vein_id == "costa":
                _draw_linestring(right, a.line, (255, 255, 255), thickness=4)
                coords = list(a.line.coords)
                mid = coords[len(coords) // 2]
                cv2.putText(
                    right,
                    f"costa ({a.length_px:.0f}px)",
                    (int(mid[0]) - 60, int(mid[1]) - 15),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 255, 255),
                    2,
                    cv2.LINE_AA,
                )

    return left, right


def _render_crossveins(state: StepState, prev: Optional[StepState]) -> tuple[np.ndarray, np.ndarray]:
    """Step 7: L1-L5 colored + crossveins gray → all veins colored (ACV/PCV highlighted)."""
    # Left: L1-L5 colored, crossveins gray
    left = state.image.copy()
    if state.assignments:
        for a in state.assignments:
            if a.line is None:
                continue
            if a.vein_id in ("ACV", "PCV"):
                _draw_linestring(left, a.line, (128, 128, 128), thickness=2)
            else:
                color = VEIN_COLORS.get(a.vein_id, (128, 128, 128))
                _draw_linestring(left, a.line, color, thickness=4)

    # Right: all veins colored, ACV/PCV highlighted with labels
    right = state.image.copy()
    if state.assignments:
        for a in state.assignments:
            if a.line is None:
                continue
            color = VEIN_COLORS.get(a.vein_id, (128, 128, 128))
            thickness = 5 if a.vein_id in ("ACV", "PCV") else 3
            _draw_linestring(right, a.line, color, thickness=thickness)
            if a.vein_id in ("ACV", "PCV"):
                coords = list(a.line.coords)
                mid = coords[len(coords) // 2]
                cv2.putText(
                    right,
                    a.vein_id,
                    (int(mid[0]) - 20, int(mid[1]) - 15),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.0,
                    color,
                    3,
                    cv2.LINE_AA,
                )

    return left, right


def _draw_dtip(image: np.ndarray, state: StepState) -> None:
    """Draw the DTip landmark as a labeled circle."""
    if not state.landmark_points:
        return
    dtip = state.landmark_points.get("DTip")
    if dtip is None:
        return
    x, y = int(dtip[0]), int(dtip[1])
    cv2.circle(image, (x, y), 20, (0, 255, 0), 3)
    cv2.circle(image, (x, y), 4, (0, 255, 0), -1)
    cv2.putText(image, "DTip", (x + 24, y + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)


def _render_longitudinals(state: StepState, prev: Optional[StepState]) -> tuple[np.ndarray, np.ndarray]:
    """Step 5: All paths gray → L1-L5 colored, crossveins gray."""
    # Left: all paths in gray + DTip
    left = state.image.copy()
    if state.assignments:
        for a in state.assignments:
            if a.line is None:
                continue
            _draw_linestring(left, a.line, (128, 128, 128), thickness=3)
    _draw_dtip(left, state)

    # Right: L1-L5 colored with labels, crossveins gray + DTip
    right = state.image.copy()
    if state.assignments:
        for a in state.assignments:
            if a.line is None:
                continue
            if a.vein_id in ("ACV", "PCV"):
                _draw_linestring(right, a.line, (128, 128, 128), thickness=2)
            else:
                color = VEIN_COLORS.get(a.vein_id, (128, 128, 128))
                _draw_linestring(right, a.line, color, thickness=4)
                coords = list(a.line.coords)
                mid = coords[len(coords) // 2]
                cv2.putText(
                    right,
                    a.vein_id,
                    (int(mid[0]) - 20, int(mid[1]) - 15),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.9,
                    color,
                    2,
                    cv2.LINE_AA,
                )
    _draw_dtip(right, state)

    return left, right


def _render_regions(state: StepState, prev: Optional[StepState]) -> tuple[np.ndarray, np.ndarray]:
    """Step 7: Named veins → veins + colored/labeled regions."""
    # Left: classified veins only
    left = state.image.copy()
    if state.assignments:
        for a in state.assignments:
            if a.line is None:
                continue
            color = VEIN_COLORS.get(a.vein_id, (128, 128, 128))
            _draw_linestring(left, a.line, color, thickness=4)

    # Right: veins + colored regions
    right = state.image.copy()
    if state.poly_names and state.polygons:
        # Fill regions
        for idx, name in state.poly_names.items():
            if idx >= len(state.polygons):
                continue
            poly = state.polygons[idx]
            color = INTERVEIN_COLORS.get(name, (180, 180, 180))
            _fill_polygon_alpha(right, poly, color, alpha=0.4)
            cx, cy = int(poly.centroid.x), int(poly.centroid.y)
            cv2.putText(
                right,
                name.replace("_cell", "").replace("_", " "),
                (cx - 80, cy),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
        # Draw veins on top
        if state.assignments:
            for a in state.assignments:
                if a.line is None:
                    continue
                color = VEIN_COLORS.get(a.vein_id, (128, 128, 128))
                _draw_linestring(right, a.line, color, thickness=3)

    return left, right


def _render_poly_split(state: StepState, prev: Optional[StepState]) -> tuple[np.ndarray, np.ndarray]:
    """Step 11: Left = original veins (solid), Right = veins + extensions (dashed) + clipped regions."""
    left = state.image.copy()
    right = state.image.copy()

    # Left: original veins as solid lines with labels + endpoint markers
    if state.assignments:
        for a in state.assignments:
            if a.line is None:
                continue
            color = VEIN_COLORS.get(a.vein_id, (128, 128, 128))
            _draw_linestring(left, a.line, color, thickness=4)
            coords = list(a.line.coords)
            mid = coords[len(coords) // 2]
            cv2.putText(
                left,
                a.vein_id,
                (int(mid[0]) - 20, int(mid[1]) - 15),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                color,
                2,
                cv2.LINE_AA,
            )
            # Mark endpoints with circles
            for ep in (coords[0], coords[-1]):
                cv2.circle(left, (int(ep[0]), int(ep[1])), 8, color, 2)

    # Right: veins (solid) + extension lines (dashed) + clipped regions underneath
    # Draw clipped regions first (underneath)
    if state.poly_names and state.polygons:
        for idx, name in state.poly_names.items():
            if idx >= len(state.polygons):
                continue
            poly = state.polygons[idx]
            color = INTERVEIN_COLORS.get(name, (180, 180, 180))
            _fill_polygon_alpha(right, poly, color, alpha=0.3)
            cx, cy = int(poly.centroid.x), int(poly.centroid.y)
            label = name.replace("_cell", "")
            cv2.putText(
                right,
                label,
                (cx - 80, cy),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

    # Draw veins on top
    if state.assignments:
        for a in state.assignments:
            if a.line is None:
                continue
            color = VEIN_COLORS.get(a.vein_id, (128, 128, 128))
            _draw_linestring(right, a.line, color, thickness=4)
            coords = list(a.line.coords)
            mid = coords[len(coords) // 2]
            cv2.putText(
                right,
                a.vein_id,
                (int(mid[0]) - 20, int(mid[1]) - 15),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                color,
                2,
                cv2.LINE_AA,
            )

    # Draw extension lines as dashed lines with bright markers at start/end
    if state.extension_lines:
        for vein_id, lines in state.extension_lines.items():
            color = VEIN_COLORS.get(vein_id, (200, 200, 200))
            for line in lines:
                _draw_dashed_linestring(right, line, (255, 255, 255), thickness=3, dash_len=8)
                ext_coords = list(line.coords)
                if ext_coords:
                    # Bright circle at extension start (vein endpoint)
                    sp = ext_coords[0]
                    cv2.circle(right, (int(sp[0]), int(sp[1])), 10, (0, 255, 0), -1)
                    # Arrow at extension end
                    ep = ext_coords[-1]
                    cv2.circle(right, (int(ep[0]), int(ep[1])), 10, (0, 0, 255), -1)

    return left, right


def _render_validation(state: StepState, prev: Optional[StepState]) -> tuple[np.ndarray, np.ndarray]:
    """Step 12: Classification → warnings highlighted in red."""
    # Left: clean classified view
    left = state.image.copy()
    if state.assignments:
        for a in state.assignments:
            if a.line is None:
                continue
            color = VEIN_COLORS.get(a.vein_id, (128, 128, 128))
            _draw_linestring(left, a.line, color, thickness=4)
            coords = list(a.line.coords)
            mid = coords[len(coords) // 2]
            cv2.putText(
                left,
                a.vein_id,
                (int(mid[0]) - 20, int(mid[1]) - 15),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                color,
                2,
                cv2.LINE_AA,
            )

    # Right: same but with warnings text
    right = left.copy()
    warnings = []
    if state.id_result and state.id_result.validation_report:
        warnings = state.id_result.validation_report.warnings

    if warnings:
        y_pos = 40
        for w in warnings[:15]:  # max 15 warnings
            cv2.putText(
                right,
                f"! {w[:80]}",
                (30, y_pos),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 255),
                2,
                cv2.LINE_AA,
            )
            y_pos += 30
    else:
        cv2.putText(
            right,
            "No validation warnings",
            (30, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (0, 200, 0),
            2,
            cv2.LINE_AA,
        )

    return left, right


def _render_outline(state: StepState, prev: Optional[StepState]) -> tuple[np.ndarray, np.ndarray]:
    """Step 15: All polygons → outline polygon boundary."""
    # Left: all polygons outlined
    left = state.image.copy()
    if state.polygons:
        for i, poly in enumerate(state.polygons):
            color = SEGMENT_COLORS[i % len(SEGMENT_COLORS)]
            _draw_polygon_outline(left, poly, color, thickness=2)

    # Right: wing outline
    right = state.image.copy()
    if state.outline and state.outline.polygon and not state.outline.polygon.is_empty:
        pts = np.array(state.outline.polygon.exterior.coords, dtype=np.int32)
        cv2.polylines(right, [pts], isClosed=True, color=(0, 255, 0), thickness=4)

    return left, right


def _render_hinge(state: StepState, prev: Optional[StepState]) -> tuple[np.ndarray, np.ndarray]:
    """Step 16: Outline + hinge landmarks → wing blade."""
    # Left: outline with hinge landmarks
    left = state.image.copy()
    if state.outline and state.outline.polygon and not state.outline.polygon.is_empty:
        pts = np.array(state.outline.polygon.exterior.coords, dtype=np.int32)
        cv2.polylines(left, [pts], isClosed=True, color=(0, 255, 0), thickness=3)
    if state.hinge_landmarks:
        lm = state.hinge_landmarks
        sc = (int(lm.subcostal_break[0]), int(lm.subcostal_break[1]))
        al = (int(lm.alula_notch[0]), int(lm.alula_notch[1]))
        cv2.circle(left, sc, 20, (0, 0, 255), -1)
        cv2.circle(left, al, 20, (255, 0, 0), -1)
        cv2.putText(left, "Subcostal", (sc[0] + 25, sc[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        cv2.putText(left, "Alula", (al[0] + 25, al[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
        # Draw hinge line extended to wing edges along subcostal→alula direction
        hl_coords = list(lm.hinge_line.coords)
        ext = 500
        p_sc = np.array(lm.subcostal_break)
        p_al = np.array(lm.alula_notch)
        direction = p_al - p_sc
        direction = direction / (np.linalg.norm(direction) + 1e-9)
        draw_coords = [(p_sc - direction * ext).tolist()] + hl_coords + [(p_al + direction * ext).tolist()]
        hl_pts = np.array(draw_coords, dtype=np.int32)
        cv2.polylines(left, [hl_pts], isClosed=False, color=(0, 165, 255), thickness=3)

    # Right: wing blade only
    right = state.image.copy()
    if state.wing_blade and not state.wing_blade.is_empty:
        pts = np.array(state.wing_blade.exterior.coords, dtype=np.int32)
        cv2.polylines(right, [pts], isClosed=True, color=(0, 255, 0), thickness=4)
        # Dim area outside wing blade
        mask = np.zeros(right.shape[:2], dtype=np.uint8)
        cv2.fillPoly(mask, [pts], 255)
        right[mask == 0] = (right[mask == 0] * 0.3).astype(np.uint8)

    return left, right


def _render_compartments(state: StepState, prev: Optional[StepState]) -> tuple[np.ndarray, np.ndarray]:
    """Step 17: Wing blade + L4 → anterior/posterior colored."""
    # Left: wing blade
    left = state.image.copy()
    if state.wing_blade and not state.wing_blade.is_empty:
        pts = np.array(state.wing_blade.exterior.coords, dtype=np.int32)
        cv2.polylines(left, [pts], isClosed=True, color=(0, 255, 0), thickness=3)
    # Draw L4
    if state.assignments:
        l4 = next((a for a in state.assignments if a.vein_id == "L4"), None)
        if l4 and l4.line:
            _draw_linestring(left, l4.line, VEIN_COLORS["L4"], thickness=4)

    # Right: anterior/posterior colored
    right = state.image.copy()
    if state.anterior_compartment and not state.anterior_compartment.is_empty:
        _fill_polygon_alpha(right, state.anterior_compartment, (255, 200, 100), alpha=0.35)
        cx = int(state.anterior_compartment.centroid.x)
        cy = int(state.anterior_compartment.centroid.y)
        cv2.putText(right, "Anterior", (cx - 60, cy), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 200, 100), 3, cv2.LINE_AA)
    if state.posterior_compartment and not state.posterior_compartment.is_empty:
        _fill_polygon_alpha(right, state.posterior_compartment, (100, 200, 255), alpha=0.35)
        cx = int(state.posterior_compartment.centroid.x)
        cy = int(state.posterior_compartment.centroid.y)
        cv2.putText(right, "Posterior", (cx - 60, cy), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (100, 200, 255), 3, cv2.LINE_AA)

    return left, right


def _render_measurements(state: StepState, prev: Optional[StepState]) -> tuple[np.ndarray, np.ndarray]:
    """Step 18: Geometry → measurement values as text overlay."""
    # Left: classified veins + regions
    left = state.image.copy()
    if state.assignments:
        for a in state.assignments:
            if a.line is None:
                continue
            color = VEIN_COLORS.get(a.vein_id, (128, 128, 128))
            _draw_linestring(left, a.line, color, thickness=3)

    # Right: measurement text overlay
    right = state.image.copy()
    # Semi-transparent background for text
    h, w = right.shape[:2]
    overlay_box = np.zeros_like(right)
    cv2.rectangle(overlay_box, (10, 10), (600, 700), (255, 255, 255), -1)
    right = cv2.addWeighted(right, 0.7, overlay_box, 0.3, 0)

    y_pos = 40
    text_color = (30, 30, 30)

    if state.measurements:
        m = state.measurements
        cv2.putText(
            right, "=== Measurements ===", (20, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.8, text_color, 2, cv2.LINE_AA
        )
        y_pos += 35

        # Vein lengths
        for vein_id in ["L1", "L2", "L3", "L4", "L5", "ACV", "PCV", "costa"]:
            length = m.vein_lengths_px.get(vein_id)
            if length is not None:
                cv2.putText(
                    right,
                    f"{vein_id}: {length:.0f} px",
                    (30, y_pos),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    text_color,
                    1,
                    cv2.LINE_AA,
                )
                y_pos += 25

        y_pos += 10
        if m.crossvein_distance_px:
            cv2.putText(
                right,
                f"Crossvein distance: {m.crossvein_distance_px:.0f} px",
                (30, y_pos),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                text_color,
                1,
            )
            y_pos += 25
        if m.wing_length_px:
            cv2.putText(
                right,
                f"Wing length: {m.wing_length_px:.0f} px",
                (30, y_pos),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                text_color,
                1,
            )
            y_pos += 25
        if m.wing_width_px:
            cv2.putText(
                right,
                f"Wing width: {m.wing_width_px:.0f} px",
                (30, y_pos),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                text_color,
                1,
            )
            y_pos += 25
        if m.total_wing_area_px2:
            cv2.putText(
                right,
                f"Wing area: {m.total_wing_area_px2:.0f} px^2",
                (30, y_pos),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                text_color,
                1,
            )
            y_pos += 25

        y_pos += 10
        if m.anterior_compartment_area_px2:
            cv2.putText(
                right,
                f"Anterior: {m.anterior_compartment_area_px2:.0f} px^2",
                (30, y_pos),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                text_color,
                1,
            )
            y_pos += 25
        if m.posterior_compartment_area_px2:
            cv2.putText(
                right,
                f"Posterior: {m.posterior_compartment_area_px2:.0f} px^2",
                (30, y_pos),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                text_color,
                1,
            )
            y_pos += 25

        # Region areas
        y_pos += 10
        for name, area in m.intervein_areas_px2.items():
            short = name.replace("_cell", "").replace("_", " ")
            cv2.putText(right, f"{short}: {area:.0f} px^2", (30, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 1)
            y_pos += 22
    else:
        cv2.putText(right, "No measurements computed", (30, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 200), 2)

    return left, right


def _render_overlays(state: StepState, prev: Optional[StepState]) -> tuple[np.ndarray, np.ndarray]:
    """Step 19: Skeleton overlay (left) + Rainbow overlay (right)."""
    left = state.skeleton_overlay if state.skeleton_overlay is not None else _blank(state)
    right = state.rainbow_overlay if state.rainbow_overlay is not None else _blank(state)
    return left, right


# ------------------------------------------------------------------
# Drawing helpers
# ------------------------------------------------------------------


def _blank(state: StepState) -> np.ndarray:
    """Return a blank image sized to the loaded image, or 800x600 default."""
    if state.image is not None:
        return np.zeros_like(state.image)
    return np.zeros((600, 800, 3), dtype=np.uint8)


_FORK_LANDMARKS = ("L1-Rs", "L2-L3", "L4-L5")


def _draw_fork_landmarks(
    image: np.ndarray,
    state: StepState,
    radius: int = 18,
    color: tuple[int, int, int] = (0, 255, 255),  # yellow BGR
) -> None:
    """Draw L1-Rs, L2-L3, L4-L5 landmark points as labeled circles."""
    if not state.landmark_points:
        return
    for name in _FORK_LANDMARKS:
        pt = state.landmark_points.get(name)
        if pt is None:
            continue
        x, y = int(pt[0]), int(pt[1])
        cv2.circle(image, (x, y), radius, color, 3)
        cv2.circle(image, (x, y), 4, color, -1)
        cv2.putText(
            image,
            name,
            (x + radius + 4, y + 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            color,
            2,
            cv2.LINE_AA,
        )


def _draw_linestring(
    image: np.ndarray,
    line: LineString,
    color: tuple[int, int, int],
    thickness: int = 3,
) -> None:
    """Draw a Shapely LineString on a BGR image."""
    pts = np.array(line.coords, dtype=np.int32)
    if len(pts) >= 2:
        cv2.polylines(image, [pts], isClosed=False, color=color, thickness=thickness)


def _draw_dashed_linestring(
    image: np.ndarray,
    line: LineString,
    color: tuple[int, int, int],
    thickness: int = 2,
    dash_len: int = 10,
) -> None:
    """Draw a dashed Shapely LineString on a BGR image."""
    pts = np.array(line.coords, dtype=np.int32)
    if len(pts) < 2:
        return
    # Draw alternating dash segments
    draw = True
    for i in range(len(pts) - 1):
        if draw:
            cv2.line(image, tuple(pts[i]), tuple(pts[i + 1]), color, thickness)
        # Toggle every dash_len points
        if (i + 1) % dash_len == 0:
            draw = not draw


def _draw_polygon_outline(
    image: np.ndarray,
    poly: Polygon,
    color: tuple[int, int, int],
    thickness: int = 3,
) -> None:
    """Draw a polygon outline on a BGR image."""
    pts = np.array(poly.exterior.coords, dtype=np.int32)
    cv2.polylines(image, [pts], isClosed=True, color=color, thickness=thickness)


def _fill_polygon_alpha(
    image: np.ndarray,
    poly: Polygon,
    color: tuple[int, int, int],
    alpha: float = 0.4,
) -> None:
    """Fill a polygon on image with semi-transparent color (in-place)."""
    pts = np.array(poly.exterior.coords, dtype=np.int32)
    overlay = image.copy()
    cv2.fillPoly(overlay, [pts], color)
    # Handle holes
    for interior in poly.interiors:
        hole_pts = np.array(interior.coords, dtype=np.int32)
        cv2.fillPoly(overlay, [hole_pts], (0, 0, 0))
    # Create mask for this polygon
    mask = np.zeros(image.shape[:2], dtype=np.uint8)
    cv2.fillPoly(mask, [pts], 255)
    for interior in poly.interiors:
        hole_pts = np.array(interior.coords, dtype=np.int32)
        cv2.fillPoly(mask, [hole_pts], 0)
    mask_bool = mask > 0
    image[mask_bool] = cv2.addWeighted(image, 1 - alpha, overlay, alpha, 0)[mask_bool]
