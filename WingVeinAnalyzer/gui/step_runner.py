"""Execute the pipeline step-by-step, caching StepState at each stage."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
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
    find_triple_junctions,
    identify_veins_and_regions,
    merge_segments_at_junctions,
    name_regions_from_veins,
    split_merged_polygons,
)
from WingVeinAnalyzer.models.vein_labeler import (
    VeinAssignment,
    VeinStatus,
    _extract_costa,
)
from WingVeinAnalyzer.models.vein_map import (
    BRIDGE_THRESHOLD_UM,
    BUFFER_OUTLINE_UM,
    BUFFER_VEIN_UM,
    COMPARTMENT_EXTENSION_UM,
    COMPARTMENT_SIMPLIFY_UM,
    CV_NORM_DIST_UM,
    MIDLINE_SIGMA_UM,
    MIDLINE_SPACING_UM,
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
    VoronoiResult,
    extract_veins_from_mask,
)
from WingVeinAnalyzer.models.wing_geometry import (
    WingMidline,
    WingOutline,
    build_wing_outline,
    compute_compartments,
    compute_wing_midline,
    detect_hinge_landmarks,
    partition_by_vein_extension,
    partition_intervein_spaces,
    remove_hinge,
)
from WingVeinAnalyzer.utils.skeleton_utils import smooth_line, smooth_polygon
from WingVeinAnalyzer.views.overlay_view import render_rainbow_overlay, render_skeleton_overlay

logger = logging.getLogger(__name__)


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

    # Wing midline (from step 1)
    wing_midline: Optional[WingMidline] = None

    # Voronoi (from step 2)
    vein_mask: Optional[np.ndarray] = None
    nearest_labels: Optional[np.ndarray] = None
    centerlines: Optional[dict[tuple[int, int], LineString]] = None

    # Hull seeding intermediates (from step 2, displayed in step 3)
    hull_mask: Optional[np.ndarray] = None
    seed_labels: Optional[np.ndarray] = None

    # Identification (from step 5, cached through step 12)
    id_result: Optional[IdentificationResult] = None
    junctions: Optional[list[JunctionPoint]] = None
    merged_paths: Optional[list[MergedPath]] = None
    split_paths: Optional[list[MergedPath]] = None
    vein_map: Optional[dict[str, MergedPath]] = None
    poly_names: Optional[dict[int, str]] = None

    # Vein-extension lines (from step 11)
    extension_lines: Optional[dict[str, list[LineString]]] = None

    # Assignments (from steps 13-14)
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
            1: self._step_midline,
            2: self._step_voronoi,
            3: self._step_hull_seeds,
            4: self._step_centerlines,
            5: self._step_identify,
            6: self._step_merge_viz,
            7: self._step_split_viz,
            8: self._step_longitudinal_viz,
            9: self._step_crossvein_viz,
            10: self._step_regions_viz,
            11: self._step_poly_split_viz,
            12: self._step_validate_viz,
            13: self._step_l1_recovery,
            14: self._step_costa,
            15: self._step_outline,
            16: self._step_hinge,
            17: self._step_compartments,
            18: self._step_measurements,
            19: self._step_overlays,
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
        }
        return state

    def _step_voronoi(self, prev: StepState) -> StepState:
        """Step 2: Rasterize vein mask, hull-component seeding, Voronoi partition."""
        state = self._copy_forward(prev)

        if not state.vein_polygons:
            logger.warning("No vein polygons — skipping Voronoi step")
            return state

        voronoi_result = extract_veins_from_mask(
            state.vein_polygons,
            state.image.shape[:2],
            intervein_polygons=(list(state.annotations.intervein_polygons) if state.annotations else None),
        )
        state.polygons = voronoi_result.voronoi_polygons
        state.centerlines = voronoi_result.centerlines
        state.nearest_labels = voronoi_result.nearest_labels
        state.vein_mask = voronoi_result.vein_mask
        state.hull_mask = voronoi_result.hull_mask
        state.seed_labels = voronoi_result.seed_labels

        # Recompute wing bounding box from Voronoi-derived polygons
        valid_polys = [p for p in state.polygons if not p.is_empty]
        if valid_polys:
            all_bounds = [p.bounds for p in valid_polys]
            state.wing_bbox = (
                min(b[0] for b in all_bounds),
                min(b[1] for b in all_bounds),
                max(b[2] for b in all_bounds),
                max(b[3] for b in all_bounds),
            )

        state.params_used = {
            "closing_kernel_size": "11",
            "voronoi_polygons": str(len(state.polygons)),
        }
        return state

    def _step_hull_seeds(self, prev: StepState) -> StepState:
        """Step 3: Visualization-only — show hull seeding intermediates."""
        state = self._copy_forward(prev)
        n_seed_labels = 0
        if state.seed_labels is not None:
            unique = set(int(v) for v in np.unique(state.seed_labels) if v > 0)
            n_seed_labels = len(unique)
        n_voronoi_polys = len(state.polygons) if state.polygons else 0
        state.params_used = {
            "hull_coverage": f"{int(state.hull_mask.sum()) if state.hull_mask is not None else 0} px",
            "seed_components": str(n_seed_labels),
            "voronoi_polygons": str(n_voronoi_polys),
        }
        return state

    def _step_centerlines(self, prev: StepState) -> StepState:
        """Step 4: Viz-only — centerlines already extracted in step 2."""
        state = self._copy_forward(prev)
        state.params_used = {
            "min_line_length": f"{um_to_px(MIN_SEGMENT_LENGTH_UM):.0f} px ({MIN_SEGMENT_LENGTH_UM} µm)",
            "bridge_threshold": f"{um_to_px(BRIDGE_THRESHOLD_UM):.0f} px ({BRIDGE_THRESHOLD_UM} µm)",
            "num_centerlines": str(len(state.centerlines or {})),
        }
        return state

    def _step_midline(self, prev: StepState) -> StepState:
        """Step 1: Compute the wing midline for crossvein-independent identification."""
        state = self._copy_forward(prev)

        if state.polygons and state.wing_bbox:
            # Prefer "wing" annotation polygon for midline shape
            wing_poly = state.annotations.wing_polygon if state.annotations else None

            # Use original annotation polygons + bbox for midline (not hull-derived Voronoi)
            if state.annotations and state.annotations.intervein_polygons:
                midline_polys = list(state.annotations.intervein_polygons) + list(state.annotations.vein_polygons)
                ann_bounds = [p.bounds for p in midline_polys]
                midline_bbox = (
                    min(b[0] for b in ann_bounds),
                    min(b[1] for b in ann_bounds),
                    max(b[2] for b in ann_bounds),
                    max(b[3] for b in ann_bounds),
                )
            else:
                midline_polys = state.polygons
                midline_bbox = state.wing_bbox
            midline = compute_wing_midline(
                midline_polys,
                midline_bbox,
                wing_polygon=wing_poly,
            )
            state.wing_midline = midline
            if midline is not None:
                state.params_used = {
                    "sample_spacing": f"{um_to_px(MIDLINE_SPACING_UM):.0f} px ({MIDLINE_SPACING_UM} µm)",
                    "smooth_sigma": f"{um_to_px(MIDLINE_SIGMA_UM):.0f} px ({MIDLINE_SIGMA_UM} µm)",
                    "num_samples": str(len(midline.line.coords)),
                }
            else:
                state.params_used = {"status": "Failed (too few samples)"}
        else:
            state.params_used = {"status": "Skipped (no polygons)"}

        return state

    def _step_identify(self, prev: StepState) -> StepState:
        """Step 5: Run full identify_veins_and_regions() and cache result."""
        state = self._copy_forward(prev)

        if state.centerlines is None:
            logger.warning("No centerlines — skipping identification")
            return state

        id_result = identify_veins_and_regions(
            state.centerlines,
            state.polygons,
            state.vein_polygons or [],
            state.image.shape[:2],
            state.wing_bbox,
            midline=state.wing_midline,
            original_polygons=list(state.original_polygons) if state.original_polygons else None,
        )
        state.id_result = id_result
        state.assignments = list(id_result.assignments)
        state.poly_names = dict(id_result.poly_names)
        if id_result.polygons:
            state.polygons = list(id_result.polygons)

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
        """Step 8: Visualization-only — show L1-L5 classified."""
        state = self._copy_forward(prev)
        has_midline = state.wing_midline is not None
        has_cv = state.vein_map and ("ACV" in state.vein_map or "PCV" in state.vein_map)
        if has_midline and has_cv:
            weights = "Y:0.30, Len:0.25, CV:0.20, Mid:0.25"
        elif has_midline:
            weights = "Y:0.30, Len:0.25, Mid:0.45"
        elif has_cv:
            weights = "Y:0.40, Len:0.30, CV:0.30"
        else:
            weights = "Y:0.60, Len:0.40"
        state.params_used = {"scoring_weights": weights}
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
            if a.line is not None and a.vein_id != "costa" and a.status != VeinStatus.ABSENT
        }
        if vein_lines and state.polygons and state.poly_names:
            # Build temporary wing outline (wing_blade not available until step 16)
            outline = build_wing_outline(
                state.polygons,
                vein_polygons=state.vein_polygons or None,
            )
            image_shape = state.image.shape[:2] if state.image is not None else (1, 1)
            ext_polys, ext_poly_veins, ext_lines = partition_by_vein_extension(
                outline.polygon,
                vein_lines,
                image_shape,
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

    def _step_l1_recovery(self, prev: StepState) -> StepState:
        """Step 13: L1 recovery from marginal cell boundary (if needed)."""
        state = self._copy_forward(prev)

        has_costal = "costal_cell" in (state.poly_names or {}).values()
        if not has_costal and state.assignments and state.polygons and state.poly_names:
            from WingVeinAnalyzer.controllers.analysis_controller import (
                _recover_l1_from_marginal_cell,
            )

            _recover_l1_from_marginal_cell(
                state.assignments,
                state.polygons,
                state.poly_names,
                state.wing_bbox,
                logger,
                vein_polygons=state.vein_polygons,
                image_shape=state.image.shape[:2] if state.image is not None else None,
            )
            state.params_used = {"method": "marginal_cell_boundary", "status": "Ran"}
        else:
            state.params_used = {"status": "Skipped (costal cell present)"}

        return state

    def _step_costa(self, prev: StepState) -> StepState:
        """Step 14: Extract costa."""
        state = self._copy_forward(prev)

        has_costal = "costal_cell" in (state.poly_names or {}).values()
        if has_costal and state.polygons and state.poly_names:
            costa_line = _extract_costa(state.polygons, state.poly_names, state.wing_bbox)
            if costa_line:
                state.assignments.append(
                    VeinAssignment(
                        vein_id="costa",
                        status=VeinStatus.COMPLETE,
                        edge_ids=[],
                        confidence=0.9,
                        evidence=["anterior_margin"],
                        length_px=costa_line.length,
                        line=costa_line,
                        endpoints=[
                            list(costa_line.coords)[0],
                            list(costa_line.coords)[-1],
                        ],
                    )
                )
                state.params_used = {"status": "Extracted", "length": f"{costa_line.length:.0f} px"}
            else:
                state.params_used = {"status": "Failed to extract"}
        else:
            state.assignments.append(
                VeinAssignment(
                    vein_id="costa",
                    status=VeinStatus.ABSENT,
                    edge_ids=[],
                    confidence=0.0,
                    evidence=["no_costal_region"],
                )
            )
            state.params_used = {"status": "Absent (no costal region)"}

        # Apply scale calibration
        compile_results(state.assignments, self._um_per_px or None)

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
        """Step 16: Detect hinge landmarks and remove hinge."""
        state = self._copy_forward(prev)

        landmarks = detect_hinge_landmarks(
            state.outline,
            state.polygons,
            state.poly_names or {},
        )
        state.hinge_landmarks = landmarks
        if landmarks:
            wing_blade = remove_hinge(
                state.outline,
                landmarks,
                state.polygons,
                state.poly_names or {},
            )
            state.params_used = {"status": "Hinge detected and removed"}
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
            midline=state.wing_midline.line if state.wing_midline else None,
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
        state.vein_mask = prev.vein_mask
        state.nearest_labels = prev.nearest_labels
        state.centerlines = prev.centerlines
        state.hull_mask = prev.hull_mask
        state.seed_labels = prev.seed_labels
        state.wing_midline = prev.wing_midline
        state.id_result = prev.id_result
        state.junctions = prev.junctions
        state.merged_paths = prev.merged_paths
        state.split_paths = prev.split_paths
        state.vein_map = prev.vein_map
        state.poly_names = prev.poly_names
        state.extension_lines = prev.extension_lines
        state.assignments = list(prev.assignments) if prev.assignments else prev.assignments
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
