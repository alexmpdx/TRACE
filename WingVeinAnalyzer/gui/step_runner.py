"""Execute the pipeline step-by-step, caching StepState at each stage."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Optional

import cv2
import numpy as np
from shapely.geometry import LineString, Polygon

from WingVeinAnalyzer.controllers.measurement_controller import (
    WingMeasurements,
    compile_results,
    compute_measurements,
)
from WingVeinAnalyzer.models.geojson_parser import ParsedAnnotations, parse_geojson
from WingVeinAnalyzer.models.vein_identifier import (
    IdentificationResult,
    JunctionPoint,
    MergedPath,
    ValidationReport,
    classify_merged_paths,
    cross_validate,
    extract_l1_from_mask,
    find_triple_junctions,
    identify_veins_and_regions,
    merge_segments_at_junctions,
    name_regions_from_veins,
    split_merged_polygons,
)
from WingVeinAnalyzer.models.vein_labeler import (
    VeinAssignment,
    VeinStatus,
)
from WingVeinAnalyzer.models.vein_map import (
    BRIDGE_THRESHOLD_UM,
    BUFFER_OUTLINE_UM,
    BUFFER_VEIN_UM,
    COMPARTMENT_EXTENSION_UM,
    COMPARTMENT_SIMPLIFY_UM,
    CV_NORM_DIST_UM,
    MIN_PATH_LENGTH_UM,
    MIN_SEGMENT_LENGTH_UM,
    MIN_SPLIT_LENGTH_UM,
    SMOOTH_SPACING_FINE_UM,
    SMOOTH_SPACING_UM,
    SNAP_RADIUS_UM,
    STEP_DIST_UM,
    set_scale,
    um_to_px,
)
from WingVeinAnalyzer.models.vein_skeleton import (
    extract_veins_from_mask,
)
from WingVeinAnalyzer.models.wing_geometry import (
    HingeLandmarks,
    WingOutline,
    build_wing_outline,
    compute_compartments,
    detect_hinge_landmarks,
    partition_by_vein_extension,
    partition_intervein_spaces,
    remove_hinge,
)
from WingVeinAnalyzer.utils.skeleton_utils import smooth_line, smooth_polygon
from WingVeinAnalyzer.views.overlay_view import render_rainbow_overlay, render_skeleton_overlay

logger = logging.getLogger(__name__)


def _trim_l1_at_subcostal(
    l1_line: LineString,
    subcostal: tuple[float, float],
    dtip: tuple[float, float],
) -> Optional[LineString]:
    """Trim L1 so it does not extend distally past the subcostal break.

    Uses DTip (distal wing tip) to determine which direction is distal,
    then removes any portion of L1 past the subcostal break X on that side.
    """
    sc_x = subcostal[0]
    coords = list(l1_line.coords)
    if len(coords) < 2:
        return None

    # DTip is the distal end of the wing; hinge is on the opposite side
    hinge_left = sc_x < dtip[0]

    if hinge_left:
        # Distal is right — keep coords with X <= sc_x
        trimmed = [(x, y) for x, y in coords if x <= sc_x]
    else:
        # Distal is left — keep coords with X >= sc_x
        trimmed = [(x, y) for x, y in coords if x >= sc_x]

    if len(trimmed) < 2:
        return None
    return LineString(trimmed)


def _load_landmarks(image_path: Path) -> Optional[dict[str, tuple[float, float]]]:
    """Load landmark points from a *_landmarks.geojson file next to the image."""
    import json

    if image_path is None:
        return None
    stem = image_path.stem  # e.g. "-CTRL_PknRNAi_108870_0007"
    landmarks_path = image_path.parent / f"{stem}_landmarks.geojson"
    if not landmarks_path.exists():
        return None
    try:
        with open(landmarks_path) as f:
            data = json.load(f)
        points: dict[str, tuple[float, float]] = {}
        for feat in data.get("features", []):
            name = feat.get("properties", {}).get("classification", {}).get("name")
            coords = feat.get("geometry", {}).get("coordinates")
            if name and coords and len(coords) >= 2:
                points[name] = (float(coords[0]), float(coords[1]))
        logger.info("Loaded %d landmarks from %s", len(points), landmarks_path.name)
        return points if points else None
    except Exception:
        logger.warning("Failed to parse landmarks file: %s", landmarks_path)
        return None


@dataclass
class StepState:
    """Accumulated pipeline state after each step."""

    # Inputs (from step 0)
    image: Optional[np.ndarray] = None
    annotations: Optional[ParsedAnnotations] = None
    wing_bbox: Optional[tuple[float, float, float, float]] = None
    polygons: Optional[list[Polygon]] = None
    vein_polygons: Optional[list[Polygon]] = None
    original_polygons: Optional[list[Polygon]] = None
    landmark_points: Optional[dict[str, tuple[float, float]]] = None

    # Centerline extraction (from step 1)
    vein_mask: Optional[np.ndarray] = None
    nearest_labels: Optional[np.ndarray] = None
    centerlines: Optional[dict[tuple[int, int], LineString]] = None
    bridge_segments: Optional[dict[tuple[int, int], LineString]] = None

    # Identification (from step 5, cached through step 12)
    id_result: Optional[IdentificationResult] = None
    junctions: Optional[list[JunctionPoint]] = None
    merged_paths: Optional[list[MergedPath]] = None
    split_paths: Optional[list[MergedPath]] = None
    costa_region: Optional[np.ndarray] = None
    vein_map: Optional[dict[str, MergedPath]] = None
    poly_names: Optional[dict[int, str]] = None

    # Vein-extension lines (from step 11)
    extension_lines: Optional[dict[str, list[LineString]]] = None

    # Assignments (from steps 8-13)
    assignments: Optional[list[VeinAssignment]] = None

    # Geometry (from steps 15-17)
    outline: Optional[WingOutline] = None
    hinge_landmarks: Optional[Any] = None
    wing_blade: Optional[Polygon] = None
    intervein_regions: Optional[dict[str, Polygon]] = None
    anterior_compartment: Optional[Polygon] = None
    posterior_compartment: Optional[Polygon] = None

    # Measurements (from step 18)
    measurements: Optional[WingMeasurements] = None

    # Overlays (from step 19)
    skeleton_overlay: Optional[np.ndarray] = None
    rainbow_overlay: Optional[np.ndarray] = None

    # Parameters used (for display)
    params_used: dict[str, Any] = field(default_factory=dict)


class StepRunner:
    """Executes pipeline steps sequentially and caches states."""

    def __init__(self):
        self._states: dict[int, StepState] = {}
        self._image_path: Optional[Path] = None
        self._geojson_path: Optional[Path] = None
        self._last_completed: int = -1
        self._smooth_sigma: float = 3.0
        self._um_per_px: float = 0.483

    @property
    def image_path(self) -> Optional[Path]:
        return self._image_path

    @property
    def geojson_path(self) -> Optional[Path]:
        return self._geojson_path

    @property
    def last_completed(self) -> int:
        return self._last_completed

    @property
    def smooth_sigma(self) -> float:
        return self._smooth_sigma

    @smooth_sigma.setter
    def smooth_sigma(self, value: float) -> None:
        self._smooth_sigma = value

    @property
    def um_per_px(self) -> float:
        return self._um_per_px

    @um_per_px.setter
    def um_per_px(self, value: float) -> None:
        self._um_per_px = value

    def load_inputs(self, image_path: Path, geojson_path: Path) -> None:
        """Reset and set new input files."""
        self._states.clear()
        self._image_path = image_path
        self._geojson_path = geojson_path
        self._last_completed = -1

    def state_at(self, index: int) -> Optional[StepState]:
        """Return cached state at the given step, or None if not yet computed."""
        return self._states.get(index)

    def invalidate_from(self, index: int) -> None:
        """Clear cached states from index onward so they will be recomputed."""
        keys_to_remove = [k for k in self._states if k >= index]
        for k in keys_to_remove:
            del self._states[k]
        if keys_to_remove:
            self._last_completed = min(self._last_completed, index - 1)

    def run_step(self, index: int) -> StepState:
        """Execute one step. Must be called sequentially (0, 1, 2, ...)."""
        if index in self._states:
            return self._states[index]

        if index > 0 and (index - 1) not in self._states:
            raise RuntimeError(f"Step {index - 1} must be completed before step {index}")

        prev = self._states.get(index - 1, StepState())
        state = self._run_one(index, prev)
        self._states[index] = state
        self._last_completed = max(self._last_completed, index)
        return state

    def run_through(self, target: int) -> StepState:
        """Run all steps from the current position up to target (inclusive)."""
        start = self._last_completed + 1
        state = None
        for i in range(start, target + 1):
            state = self.run_step(i)
        return state or self._states.get(target, StepState())

    def run_all(self) -> StepState:
        """Run all 20 steps."""
        from WingVeinAnalyzer.gui.step_definitions import NUM_STEPS

        return self.run_through(NUM_STEPS - 1)

    # ------------------------------------------------------------------
    # Step implementations
    # ------------------------------------------------------------------

    def _run_one(self, index: int, prev: StepState) -> StepState:
        """Dispatch to the appropriate step handler."""
        handlers = {
            0: self._step_load,
            1: self._step_skeleton,
            2: self._step_identify,
            3: self._step_merge_viz,
            4: self._step_split_viz,
            5: self._step_longitudinal_viz,
            6: self._step_crossvein_viz,
            7: self._step_regions_viz,
            8: self._step_poly_split_viz,
            9: self._step_validate_viz,
            10: self._step_outline,
            11: self._step_hinge,
            12: self._step_compartments,
            13: self._step_measurements,
            14: self._step_overlays,
        }
        handler = handlers.get(index)
        if handler is None:
            raise ValueError(f"Unknown step index: {index}")
        return handler(prev)

    def _step_load(self, prev: StepState) -> StepState:
        """Step 0: Load image and parse GeoJSON."""
        set_scale(self._um_per_px if self._um_per_px else None)
        state = StepState()

        image = cv2.imread(str(self._image_path))
        if image is None:
            raise FileNotFoundError(f"Could not load image: {self._image_path}")
        state.image = image

        annotations = parse_geojson(self._geojson_path)
        state.annotations = annotations
        state.polygons = list(annotations.intervein_polygons)
        state.vein_polygons = list(annotations.vein_polygons)
        state.original_polygons = list(annotations.intervein_polygons)

        # Load landmark points if available (from LandmarkLocator)
        state.landmark_points = _load_landmarks(self._image_path)

        # Compute wing bounding box
        all_bounds = [p.bounds for p in state.polygons]
        if all_bounds:
            state.wing_bbox = (
                min(b[0] for b in all_bounds),
                min(b[1] for b in all_bounds),
                max(b[2] for b in all_bounds),
                max(b[3] for b in all_bounds),
            )
        else:
            h, w = image.shape[:2]
            state.wing_bbox = (0, 0, w, h)

        state.params_used = {
            "image_size": f"{image.shape[1]}x{image.shape[0]}",
            "num_polygons": str(len(state.polygons)),
            "has_vein_mask": str(bool(state.vein_polygons)),
            "landmarks": str(len(state.landmark_points)) if state.landmark_points else "none",
        }
        return state

    def _step_skeleton(self, prev: StepState) -> StepState:
        """Step 1: Skeletonize vein mask, extract centerlines."""
        state = self._copy_forward(prev)

        if not state.vein_polygons:
            logger.warning("No vein polygons — skipping skeleton step")
            return state

        result = extract_veins_from_mask(
            state.vein_polygons,
            state.image.shape[:2],
            intervein_polygons=(list(state.annotations.intervein_polygons) if state.annotations else None),
        )
        state.centerlines = result.centerlines
        state.bridge_segments = result.bridge_segments
        state.nearest_labels = result.nearest_labels
        state.vein_mask = result.vein_mask

        state.params_used = {
            "closing_kernel_size": "11",
            "prune_threshold": "200 px",
            "min_line_length": f"{um_to_px(MIN_SEGMENT_LENGTH_UM):.0f} px ({MIN_SEGMENT_LENGTH_UM} µm)",
            "bridge_threshold": f"{um_to_px(BRIDGE_THRESHOLD_UM):.0f} px ({BRIDGE_THRESHOLD_UM} µm)",
            "num_centerlines": str(len(state.centerlines or {})),
        }
        return state

    def _step_identify(self, prev: StepState) -> StepState:
        """Step 2: Run full identify_veins_and_regions() and cache result."""
        state = self._copy_forward(prev)

        if state.centerlines is None:
            logger.warning("No centerlines — skipping identification")
            return state

        lp = state.landmark_points or {}
        id_result = identify_veins_and_regions(
            state.centerlines,
            state.polygons,
            state.vein_polygons or [],
            state.image.shape[:2],
            state.wing_bbox,
            original_polygons=list(state.original_polygons) if state.original_polygons else None,
            dtip=lp.get("DTip"),
            landmark_points=lp if lp else None,
            wing_polygon=state.annotations.wing_polygon if state.annotations else None,
        )
        state.id_result = id_result
        state.assignments = list(id_result.assignments)
        state.poly_names = dict(id_result.poly_names)
        state.split_paths = list(id_result.split_paths)
        state.costa_region = id_result.costa_region
        if id_result.polygons:
            state.polygons = list(id_result.polygons)

        # Extract L1 from vein mask using landmarks
        if state.vein_mask is not None and "subcostal break" in lp and "L1-Rs" in lp:
            l1_line = extract_l1_from_mask(
                state.vein_mask,
                lp["subcostal break"],
                lp["L1-Rs"],
            )
            if l1_line is not None:
                # Replace the ABSENT L1 assignment with the extracted one
                state.assignments = [a for a in state.assignments if a.vein_id != "L1"]
                coords = list(l1_line.coords)
                state.assignments.append(
                    VeinAssignment(
                        vein_id="L1",
                        status=VeinStatus.COMPLETE,
                        edge_ids=[],
                        confidence=0.9,
                        evidence=["landmark_mask_extraction"],
                        length_px=l1_line.length,
                        line=l1_line,
                        endpoints=[coords[0], coords[-1]],
                    )
                )
                logger.info("L1 extracted from vein mask: %.0f px", l1_line.length)

        # Extract sub-results for visualization steps
        # Re-run sub-steps to capture intermediate data
        junctions = find_triple_junctions(state.centerlines, snap_radius=um_to_px(SNAP_RADIUS_UM))
        state.junctions = junctions

        merged_paths, _merge_decisions = merge_segments_at_junctions(state.centerlines, junctions)
        state.merged_paths = merged_paths

        # Use the real vein_map from identification (preserves segment_keys)
        state.vein_map = dict(id_result.vein_map) if id_result.vein_map else {}

        state.params_used = {"snap_radius": f"{um_to_px(SNAP_RADIUS_UM):.0f} px ({SNAP_RADIUS_UM} µm)"}
        return state

    def _step_merge_viz(self, prev: StepState) -> StepState:
        """Step 6: Visualization-only — show merged paths."""
        state = self._copy_forward(prev)
        state.params_used = {
            "collinearity_threshold": "45°",
            "min_gap": "15°",
            "orientation_guard": "25°/55°",
            "num_merged": str(len(state.merged_paths or [])),
        }
        return state

    def _step_split_viz(self, prev: StepState) -> StepState:
        """Step 7: Visualization-only — show paths after sharp-turn splitting."""
        state = self._copy_forward(prev)
        state.params_used = {
            "angle_threshold": "70°",
            "step_dist": f"{um_to_px(STEP_DIST_UM):.0f} px ({STEP_DIST_UM} µm)",
            "min_path_length": f"{um_to_px(MIN_PATH_LENGTH_UM):.0f} px ({MIN_PATH_LENGTH_UM} µm)",
            "min_split_length": f"{um_to_px(MIN_SPLIT_LENGTH_UM):.0f} px ({MIN_SPLIT_LENGTH_UM} µm)",
        }
        return state

    def _step_crossvein_viz(self, prev: StepState) -> StepState:
        """Step 9: Visualization-only — highlight ACV/PCV."""
        state = self._copy_forward(prev)
        state.params_used = {
            "max_crossvein_len": "15% wing span",
            "orientation_cutoff": "60°",
            "proximity_threshold": f"{um_to_px(CV_NORM_DIST_UM):.0f} px ({CV_NORM_DIST_UM} µm)",
        }
        return state

    def _step_longitudinal_viz(self, prev: StepState) -> StepState:
        """Step 8: Classify L1-L5, trim L1 to subcostal break."""
        state = self._copy_forward(prev)

        # Trim L1 at the subcostal break if landmark is available
        lp = state.landmark_points or {}
        l1_trimmed = False
        if "subcostal break" in lp and "DTip" in lp and state.assignments:
            l1 = next((a for a in state.assignments if a.vein_id == "L1" and a.line), None)
            if l1:
                trimmed = _trim_l1_at_subcostal(l1.line, lp["subcostal break"], lp["DTip"])
                if trimmed is not None and trimmed.length < l1.line.length:
                    l1.line = trimmed
                    l1.length_px = trimmed.length
                    l1_trimmed = True

        has_cv = state.vein_map and ("ACV" in state.vein_map or "PCV" in state.vein_map)
        if has_cv:
            weights = "Y:0.40, Len:0.30, CV:0.30"
        else:
            weights = "Y:0.60, Len:0.40"
        state.params_used = {
            "scoring_weights": weights,
            "L1_trim": "at subcostal break" if l1_trimmed else "none",
        }
        return state

    def _step_regions_viz(self, prev: StepState) -> StepState:
        """Step 10: Visualization-only — show named regions."""
        state = self._copy_forward(prev)
        state.params_used = {
            "num_regions": str(len(state.poly_names or {})),
        }
        return state

    def _step_poly_split_viz(self, prev: StepState) -> StepState:
        """Step 11: Clip intervein regions using vein-extension boundaries."""
        state = self._copy_forward(prev)

        vein_lines = {
            a.vein_id: a.line
            for a in (state.assignments or [])
            if a.line is not None
            and a.vein_id != "costa"
            and not a.vein_id.startswith("EV")
            and a.status != VeinStatus.ABSENT
        }
        if vein_lines and state.polygons and state.poly_names:
            # Build temporary wing outline (wing_blade not available until step 16)
            outline = build_wing_outline(
                state.polygons,
                vein_polygons=state.vein_polygons or None,
            )
            image_shape = state.image.shape[:2] if state.image is not None else (1, 1)
            # Don't extend L1's distal end (nearest to DTip)
            skip_eps: dict[str, list[int]] = {}
            if "L1" in vein_lines:
                lp = state.landmark_points or {}
                dtip = lp.get("DTip")
                if dtip is not None:
                    l1_coords = list(vein_lines["L1"].coords)
                    d_start = (l1_coords[0][0] - dtip[0]) ** 2 + (l1_coords[0][1] - dtip[1]) ** 2
                    d_end = (l1_coords[-1][0] - dtip[0]) ** 2 + (l1_coords[-1][1] - dtip[1]) ** 2
                    skip_eps["L1"] = [0 if d_start < d_end else -1]
            ext_polys, ext_poly_veins, ext_lines = partition_by_vein_extension(
                outline.polygon,
                vein_lines,
                image_shape,
                skip_endpoints=skip_eps,
                landmark_points=state.landmark_points,
            )
            ext_names = name_regions_from_veins(
                ext_polys,
                state.vein_map or {},
                state.wing_bbox,
                poly_veins=ext_poly_veins,
            )
            if ext_names:
                state.polygons, state.poly_names = _clip_regions_by_extension(
                    state.polygons,
                    state.poly_names,
                    ext_polys,
                    ext_names,
                )
                state.extension_lines = ext_lines
                n_ext = sum(len(v) for v in ext_lines.values())
                ext_summary = ", ".join(
                    f"{vid}: {sum(ln.length for ln in lns):.0f}px" for vid, lns in ext_lines.items()
                )
                state.params_used = {
                    "method": "vein_extension_clip",
                    "num_regions": str(len(state.poly_names)),
                    "ext_regions": str(len(ext_names)),
                    "extensions": f"{n_ext} lines ({ext_summary})" if ext_summary else "0",
                }
            else:
                state.params_used = {
                    "method": "vein_extension_failed",
                    "num_regions": str(len(state.poly_names or {})),
                }
        else:
            state.params_used = {
                "method": "skipped (no vein lines)",
                "num_regions": str(len(state.poly_names or {})),
            }
        return state

    def _step_validate_viz(self, prev: StepState) -> StepState:
        """Step 12: Visualization-only — show cross-validation results."""
        state = self._copy_forward(prev)
        warnings = []
        if state.id_result and state.id_result.validation_report:
            warnings = state.id_result.validation_report.warnings
        state.params_used = {"num_warnings": str(len(warnings))}
        return state

    def _step_outline(self, prev: StepState) -> StepState:
        """Step 15: Build wing outline."""
        state = self._copy_forward(prev)

        outline = build_wing_outline(
            state.polygons,
            vein_polygons=state.vein_polygons or None,
        )
        state.outline = outline
        state.params_used = {
            "buffer_dist": f"{um_to_px(BUFFER_OUTLINE_UM):.0f} px ({BUFFER_OUTLINE_UM} µm)",
            "vein_buffer": f"{um_to_px(BUFFER_VEIN_UM):.0f} px ({BUFFER_VEIN_UM} µm)",
        }
        return state

    def _step_hinge(self, prev: StepState) -> StepState:
        """Step 15: Detect hinge landmarks and remove hinge."""
        state = self._copy_forward(prev)

        # Use deep-learning landmarks if available
        lp = state.landmark_points or {}
        if "subcostal break" in lp and "alula notch" in lp:
            sc = lp["subcostal break"]
            al = lp["alula notch"]
            # Build hinge line through intermediate landmarks if available
            hinge_pts = [sc]
            if "L1-Rs" in lp:
                hinge_pts.append(lp["L1-Rs"])
            if "L4-L5" in lp:
                hinge_pts.append(lp["L4-L5"])
            hinge_pts.append(al)
            landmarks = HingeLandmarks(
                subcostal_break=sc,
                alula_notch=al,
                hinge_line=LineString(hinge_pts),
            )
            source = "landmarks file"
        else:
            landmarks = detect_hinge_landmarks(
                state.outline,
                state.polygons,
                state.poly_names or {},
            )
            source = "polygon geometry"

        state.hinge_landmarks = landmarks
        if landmarks:
            wing_blade = remove_hinge(
                state.outline,
                landmarks,
                state.polygons,
                state.poly_names or {},
            )
            state.params_used = {
                "status": f"Hinge detected and removed (from {source})",
            }
        else:
            wing_blade = state.outline.polygon
            state.params_used = {"status": "No hinge detected"}
        state.wing_blade = wing_blade
        return state

    def _step_compartments(self, prev: StepState) -> StepState:
        """Step 17: Compute compartments and partition intervein spaces."""
        state = self._copy_forward(prev)

        # Partition intervein spaces (polygons may have been updated by step 11)
        all_regions = partition_intervein_spaces(
            state.wing_blade,
            state.polygons,
            state.poly_names or {},
        )
        regions = {k: v for k, v in all_regions.items() if k != "costal_cell"}
        state.intervein_regions = regions

        # Compute compartments
        l4_assignment = next(
            (a for a in (state.assignments or []) if a.vein_id == "L4"),
            None,
        )
        l4_line = l4_assignment.line if l4_assignment else None
        anterior, posterior = compute_compartments(state.wing_blade, l4_line)
        state.anterior_compartment = anterior
        state.posterior_compartment = posterior

        state.params_used = {
            "simplify": f"{um_to_px(COMPARTMENT_SIMPLIFY_UM):.0f} px ({COMPARTMENT_SIMPLIFY_UM} µm)",
            "extend": f"{um_to_px(COMPARTMENT_EXTENSION_UM):.0f} px ({COMPARTMENT_EXTENSION_UM} µm)",
            "num_regions": str(len(regions)),
        }
        return state

    def _step_measurements(self, prev: StepState) -> StepState:
        """Step 18: Compute all measurements."""
        state = self._copy_forward(prev)

        measurements = compute_measurements(
            state.assignments or [],
            wing_polygon=state.wing_blade,
            intervein_regions=state.intervein_regions,
            anterior_compartment=state.anterior_compartment,
            posterior_compartment=state.posterior_compartment,
            microns_per_pixel=self._um_per_px or None,
        )
        state.measurements = measurements
        state.params_used = {
            "wing_area": f"{measurements.total_wing_area_px2:.0f} px²" if measurements.total_wing_area_px2 else "N/A",
            "wing_length": f"{measurements.wing_length_px:.0f} px" if measurements.wing_length_px else "N/A",
        }
        return state

    def _step_overlays(self, prev: StepState) -> StepState:
        """Step 19: Render final skeleton and rainbow overlays with smoothing."""
        from copy import copy

        state = self._copy_forward(prev)
        sigma = self._smooth_sigma

        # Smooth copies of assignments for rendering
        smoothed_assignments = []
        for a in state.assignments or []:
            if a.line is not None and sigma > 0:
                sa = copy(a)
                if sa.vein_id in ("ACV", "PCV"):
                    sa.line = smooth_line(
                        sa.line, sigma=max(sigma * 0.67, 0.5), sample_spacing=um_to_px(SMOOTH_SPACING_FINE_UM)
                    )
                else:
                    sa.line = smooth_line(sa.line, sigma=sigma, sample_spacing=um_to_px(SMOOTH_SPACING_UM))
                smoothed_assignments.append(sa)
            else:
                smoothed_assignments.append(a)

        # Smooth outline for display
        outline_poly = state.outline.polygon if state.outline else None
        if outline_poly and sigma > 0:
            outline_poly = smooth_polygon(outline_poly, sigma=sigma * 1.67)

        state.skeleton_overlay = render_skeleton_overlay(
            state.image,
            smoothed_assignments,
            outline_polygon=outline_poly,
        )

        # Smooth region boundaries for display
        smoothed_regions = {}
        for k, v in (state.intervein_regions or {}).items():
            if sigma > 0:
                smoothed_regions[k] = smooth_polygon(v, sigma=sigma)
            else:
                smoothed_regions[k] = v

        state.rainbow_overlay = render_rainbow_overlay(
            state.image,
            smoothed_regions,
        )

        state.params_used = {
            "line_thickness": "6 px",
            "opacity": "0.75",
            "smooth_sigma": f"{sigma:.1f}",
        }
        return state

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _copy_forward(prev: StepState) -> StepState:
        """Create a new StepState carrying forward all persistent data from prev."""
        state = StepState()
        # Copy all fields except params_used
        state.image = prev.image
        state.annotations = prev.annotations
        state.wing_bbox = prev.wing_bbox
        state.polygons = prev.polygons
        state.vein_polygons = prev.vein_polygons
        state.original_polygons = prev.original_polygons
        state.landmark_points = prev.landmark_points
        state.vein_mask = prev.vein_mask
        state.nearest_labels = prev.nearest_labels
        state.centerlines = prev.centerlines
        state.bridge_segments = prev.bridge_segments
        state.id_result = prev.id_result
        state.junctions = prev.junctions
        state.merged_paths = prev.merged_paths
        state.split_paths = prev.split_paths
        state.costa_region = prev.costa_region
        state.vein_map = prev.vein_map
        state.poly_names = prev.poly_names
        state.extension_lines = prev.extension_lines
        state.assignments = [replace(a) for a in prev.assignments] if prev.assignments else prev.assignments
        state.outline = prev.outline
        state.hinge_landmarks = prev.hinge_landmarks
        state.wing_blade = prev.wing_blade
        state.intervein_regions = prev.intervein_regions
        state.anterior_compartment = prev.anterior_compartment
        state.posterior_compartment = prev.posterior_compartment
        state.measurements = prev.measurements
        state.skeleton_overlay = prev.skeleton_overlay
        state.rainbow_overlay = prev.rainbow_overlay
        return state


def _clip_regions_by_extension(
    polygons: list[Polygon],
    poly_names: dict[int, str],
    ext_polys: list[Polygon],
    ext_names: dict[int, str],
    min_area: float = 1000,
) -> tuple[list[Polygon], dict[int, str]]:
    """Split each polygon along vein-extension boundaries.

    For each original polygon, intersect with every vein-extension region.
    Significant overlaps become separate polygons with the extension region's
    name.  This divides oversized polygons into proper compartments while
    only making individual pieces smaller.
    """
    from shapely.geometry import MultiPolygon

    # Build name → union of vein-extension polygons
    ext_by_name: dict[str, Polygon] = {}
    for i, name in ext_names.items():
        if name in ext_by_name:
            ext_by_name[name] = ext_by_name[name].union(ext_polys[i])
        else:
            ext_by_name[name] = ext_polys[i]

    new_polygons: list[Polygon] = []
    new_names: dict[int, str] = {}

    for i, orig_poly in enumerate(polygons):
        if i not in poly_names:
            new_polygons.append(orig_poly)
            continue

        # Intersect with every extension region
        pieces: list[tuple[str, Polygon]] = []
        for ext_name, ext_poly in ext_by_name.items():
            inter = orig_poly.intersection(ext_poly)
            if inter.is_empty or inter.area < min_area:
                continue
            # Extract largest polygon from multi-geometry results
            if isinstance(inter, MultiPolygon):
                inter = max(inter.geoms, key=lambda g: g.area)
            if isinstance(inter, Polygon) and inter.area >= min_area:
                pieces.append((ext_name, inter))

        if pieces:
            if len(pieces) == 1:
                # Not split — keep original name
                idx = len(new_polygons)
                new_polygons.append(pieces[0][1])
                new_names[idx] = poly_names[i]
            else:
                orig_name = poly_names[i]
                pieces.sort(key=lambda p: p[1].area, reverse=True)
                main_area = pieces[0][1].area
                # Filter out tiny edge slivers
                kept = [p for p in pieces if p[1].area >= main_area * 0.15]
                if not kept:
                    kept = [pieces[0]]
                if len(kept) == 1:
                    # Only one substantial piece — keep original name
                    idx = len(new_polygons)
                    new_polygons.append(kept[0][1])
                    new_names[idx] = orig_name
                else:
                    # Multiple substantial pieces — genuine split of a
                    # merged region; trust extension names for all pieces
                    for ext_name, piece in kept:
                        idx = len(new_polygons)
                        new_polygons.append(piece)
                        new_names[idx] = ext_name
        else:
            # No significant overlaps — keep original
            idx = len(new_polygons)
            new_polygons.append(orig_poly)
            new_names[idx] = poly_names[i]

    return new_polygons, new_names
