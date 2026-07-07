"""Modal dialog for editing an identifyFeatures PipelineConfig.

Hosts the advanced-only tabs: Landmarks, Models, Wing Graph (skeleton +
pruning + bridging — all the steps that build the vein graph from the
segmentation mask), Tracing, Intervein. The user-facing General + Custom
Distances
panels live on the main window's right-panel tab bar (InlineGeneralPanel /
InlineCustomDistancesPanel) and auto-apply edits — they never pass through
this dialog.

The dialog is dispatch-based: each field is registered via a small helper
(`_add_float`, `_add_opt_float`, `_add_int`, `_add_opt_int`,
`_add_enum_list`, `_add_float_list`) and the dispatch table records both
the widget kind and the widget reference. Read/write go through the
dispatch table so the accept/reset/load paths don't have to know
individual field types.
"""

from __future__ import annotations

from dataclasses import replace as dataclass_replace
from pathlib import Path
from typing import Any

from identify_features.config import PipelineConfig
from identify_features.models.datatypes import PruneMethod, SkeletonMethod
from PyQt5.QtCore import QSettings, Qt
from PyQt5.QtWidgets import (
    QButtonGroup,
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFormLayout,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QPushButton,
    QRadioButton,
    QScrollArea,
    QSpinBox,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)
from TRACE.theme import current_theme as _ct

from TRACE.presets_loader import load_presets

# Friendly display names for the GateConfigPanel landmark labels. The shorthand
# is what LandmarkLocator stores internally; the right-hand text is what the
# Landmarks tab should show users.
# Tooltips for every PipelineConfig field shown in the dialog. Keyed by the
# field name passed to _add_float / _add_int / _add_opt_float / _add_bool / etc.
# The helpers automatically look up the tooltip and apply it to both the widget
# and its form-row label so hovering either surfaces the description.
_FIELD_TOOLTIPS: dict[str, str] = {
    # -- General → Scale --
    "um_per_px": "Microns per pixel — used to convert every measurement to physical units (µm, µm²).",
    # -- Skeletonization & Pruning --
    "skeleton_methods": (
        "Which skeletonization method(s) to run when extracting vein centerlines from "
        "the vein-tissue mask. RIDGE (Frangi-like ridge response) is the production default."
    ),
    "smooth_sigma": "Gaussian sigma (px) used to smooth the vein-tissue mask before skeletonization.",
    "enable_basic_prune": (
        "Step 4: length-based pruning of skeleton branches. Disable to keep every short "
        "branch the skeletonizer produces (useful for debugging)."
    ),
    "enable_small_fragment_removal": (
        "Steps 11/14: discard isolated tiny skeleton components. Disable to keep every " "disconnected fragment."
    ),
    "min_component_edge_fraction": (
        "Final-pass orphan cull: drop any connected component whose total edge length is "
        "below this fraction of the graph's combined edge length. 0 = keep every component."
    ),
    "prune_methods": (
        "Optional additional pruning methods layered on top of the length-based prune. "
        "Most pipelines leave this empty."
    ),
    "prune_min_length_um": (
        "Minimum branch length to keep (µm). When unchecked, uses an auto threshold "
        "derived from the median vein width."
    ),
    "prune_min_length_vein_widths": "Auto prune threshold as a multiple of the median vein width.",
    "final_stub_vein_widths": "Final stub-removal threshold: branches shorter than this × median vein width are dropped.",
    "junction_merge_vein_widths": (
        "Tight junction merge radius: combine degree-2/3 nodes that are within this × "
        "median vein width. 0 disables the merge."
    ),
    "prune_radius_ratio_threshold": (
        "Distance-map pruning: an endpoint is treated as noise when its ridge radius is "
        "below this fraction of its junction's radius."
    ),
    "prune_scale_sigmas": "Sigmas (px) for multi-scale persistence pruning — comma-separated.",
    "prune_single_scale_sigma": "Sigma (px) used by single-scale pruning methods.",
    "collinear_min_angle": (
        "Minimum angle (deg) at which two edges meeting at a degree-2 node are merged "
        "into one. 180 = perfectly straight."
    ),
    # -- Bridging (3 passes) --
    "bridge_max_gap_um": "Pass 1: maximum absolute gap (µm) between two edge endpoints eligible for bridging.",
    "bridge_gap_fraction": "Pass 1: gap allowance as a fraction of max(edge lengths).",
    "bridge_direction_window_um": "Pass 1: distance (µm) along each edge used to compute its outgoing direction.",
    "bridge_min_combined_length_um": "Pass 1: minimum combined length (µm) of both edges to qualify.",
    "bridge_on_axis_max_angle": "Pass 1: strict on-axis angle (deg) — the longer edge's tangent must point within this of the gap vector.",
    "bridge_on_axis_relaxed_cap": "Pass 1: cap (deg) for the shorter edge's relaxed on-axis tolerance.",
    "bridge_min_facing_angle": "Pass 1: minimum angle (deg) between the two edges' outgoing directions (closer to 180 = facing each other).",
    "bridge_direction_max_edge_fraction": "Pass 1: cap on the direction-window length as a fraction of the edge length (long edges).",
    "bridge2_max_gap_um": "Pass 2: maximum absolute gap (µm) between endpoints.",
    "bridge2_gap_fraction": "Pass 2: gap allowance as a fraction of max(edge lengths).",
    "bridge2_min_gap_vw": "Pass 2: floor on the adaptive gap, expressed as × median vein width.",
    "bridge2_direction_window_um": "Pass 2: distance (µm) along each edge used to compute direction.",
    "bridge2_min_combined_length_um": "Pass 2: minimum combined edge length (µm). Used only when the × vein-width override is disabled.",
    "bridge2_min_combined_length_vw": (
        "Pass 2: minimum combined edge length as × median vein width. When checked, overrides the µm version."
    ),
    "bridge2_on_axis_max_angle": "Pass 2: on-axis max angle (deg).",
    "bridge2_on_axis_relaxed_cap": "Pass 2: relaxed on-axis cap (deg).",
    "bridge2_min_facing_angle": "Pass 2: minimum facing angle (deg) between the two edges.",
    "bridge3_max_gap_vw": "Pass 3: maximum gap as × median vein width — relaxed pass for short stubs.",
    "bridge3_short_edge_vw": "Pass 3: threshold (× median vein width) below which an edge counts as 'short' for this pass.",
    "bridge3_relaxed_facing_angle": "Pass 3: relaxed facing-angle threshold (deg) for qualifying short-stub pairs.",
    "bridge3_direction_window_um": "Pass 3: distance (µm) along each edge used to compute direction.",
    "bridge3_on_axis_max_angle": "Pass 3: on-axis max angle (deg).",
    "bridge3_on_axis_relaxed_cap": "Pass 3: relaxed on-axis cap (deg).",
    # -- Tracing — landmark anchoring + vein tracing --
    "snap_radius_um": (
        "Primary landmark-to-skeleton snap radius (µm). A landmark must lie within this "
        "distance of a skeleton node to anchor."
    ),
    "snap_radius_vw": "Fallback snap radius as × median vein width — used when no µm-per-pixel scale is set.",
    "departure_sample_um": (
        "Distance (µm) along an edge used to compute the outgoing departure direction " "from an anchored landmark."
    ),
    "departure_sample_vw": "Fallback departure sample distance as × median vein width — used when no µm-per-pixel scale is set.",
    "tangent_continuity_max_angle": (
        "Maximum tangent deflection (deg) allowed when continuing a vein through a junction. "
        "Larger = more permissive about veins making sharp turns."
    ),
    "merge_max_gap_um": "Maximum gap (µm) between two collinear line segments when merging them into one vein.",
    "distal_landmark_search_vw": "Search radius (× median vein width) for extending toward a distal landmark when its anchor edge is short.",
    "costa_min_in_band_fraction": "Minimum fraction of an edge's length that must lie inside the wing margin band for it to be classified as costa.",
    "costa_propagation_max_distance_vw": "Maximum distance (× median vein width) from the margin band for costa propagation through chained edges.",
    # -- Crossveins --
    "crossvein_min_angle": "Minimum angle (deg) between a candidate crossvein and the L4 it connects to.",
    "crossvein_max_length_frac": "Maximum crossvein length as a fraction of the wing's proximodistal axis.",
    "crossvein_min_length_vw": "Minimum crossvein length as × median vein width.",
    "crossvein_max_length_vw": "Maximum crossvein length as × median vein width.",
    "synthesize_missing_crossveins": (
        "Phase 5b: when graph detection can't find ACV/PCV, synthesize them from landmark "
        "positions. Disable to preserve the fused-region output."
    ),
    # -- Mutant phenotype reporting (Phase 5c — absent / partial assignment) --
    "assign_absent_partial_status": (
        "When on, canonical veins with no labelled path are emitted as explicit "
        "status=absent rows (length 0), and any traced vein that is gapped or doesn't "
        "reach its expected endpoints is downgraded to status=partial. Turn off to "
        "restore the legacy behaviour where only identified / inferred / ectopic statuses "
        "are written."
    ),
    "partial_endpoint_search_vw": (
        "Distance (× median vein width) a vein may fall short of its expected endpoint "
        "before it's flagged status=partial. Larger = more forgiving (fewer partials); "
        "smaller = stricter. Only affects the partial/absent check above, not tracing."
    ),
    # -- Ectopic detection --
    "ectopic_min_length_um": "Minimum length (µm) for a detected ectopic vein to be kept.",
    "ectopic_min_length_vw": "Fallback minimum ectopic length as × median vein width — used when no µm-per-pixel scale is set.",
    # -- Intervein labeling / naming --
    "skip_intervein_regions": (
        "Skip §6.1 polygon splitting and §6.2 region naming. Saves resources when only "
        "vein outputs are needed. Vein-tissue assignment still runs."
    ),
    "vein_buffer_vw": "Buffer radius around each vein centerline (× median vein width) used when assigning tissue polygons to veins.",
    "adjacency_min_length_vw": "Minimum shared boundary length (× median vein width) for two intervein regions to be considered adjacent.",
    "max_merge_size": "Maximum number of regions in an N-way merge. Uncheck for no cap.",
    "intervein_split_h_vw": "h-maxima depth threshold for the intervein splitter, as × median vein width.",
    "intervein_split_reseed_min_area_um2": (
        "Intervein splitter: when a large region gets absorbed during open-under-constraint, "
        "reseed it if its area exceeds this threshold (µm²)."
    ),
    "intervein_split_vein_barrier_vw": "Buffer radius around vein centerlines used as a barrier during intervein splitting (× median vein width).",
    "intervein_split_wing_buffer_vw": "Inset (× median vein width) from the wing outline during intervein splitting.",
    # -- Quality (garbage detector) --
    "solidity_filter_enabled": (
        "Reject wings whose outline solidity (area ÷ convex-hull area) falls outside the "
        "accepted range. Catches gross segmentation-shape failures. Aborts the wing with a reason."
    ),
    "solidity_min": (
        "Lower solidity bound. A wing below this is too concave (big bite / missing chunk). "
        "Real wings sit ~0.983–0.990; 0.95 catches only gross failures."
    ),
    "solidity_max": (
        "Upper solidity bound. A wing above this is suspiciously convex (featureless blob with "
        "none of a real wing's slight alula/hinge concavity)."
    ),
    "solidity_mode": (
        "fixed: use the min/max range above (primary). batch_mad: derive the range from the "
        "batch as median ± k·robust-σ (opt-in; needs enough wings)."
    ),
    "solidity_batch_k": (
        "batch_mad only: how many robust σ (median ± k·1.4826·MAD) from the batch median counts "
        "as an outlier. Higher = more permissive. Default 5."
    ),
    "solidity_min_batch_size": (
        "batch_mad only: minimum number of wings before the robust range is trusted; below this "
        "it falls back to the fixed min/max range."
    ),
    "fragmentation_filter_enabled": (
        "Reject wings with a large disconnected secondary region (a partial second wing or debris "
        "in frame) that the outline's largest-component step would otherwise silently discard."
    ),
    "fragmentation_max_secondary_frac": (
        "Abort when a disconnected secondary region exceeds this fraction of the main wing area. "
        "Good wings carry ≤0.6% specks; real second objects are ≥1.5%. Default 0.01 (1%)."
    ),
    "vein_association_filter_enabled": (
        "Reject wings where too much of the segmented vein tissue isn't covered by any vein the "
        "tracer identified — i.e. hallucinated vein tissue or failed tracing. Runs after tracing."
    ),
    "max_unassigned_vein_frac": (
        "Abort when this fraction of segmented vein area is not associated with any traced vein. "
        "Default 0.08 (8%)."
    ),
    "required_veins": (
        "Abort the wing if any checked vein is missing (not traced). Leave all unchecked (default) "
        "to never abort for a missing vein."
    ),
}


# Tooltips for the Intermediate-output checkboxes in the General tab.
_INTERMEDIATE_TOOLTIPS: dict[str, str] = {
    "wing_isolated_image": (
        "Keep the isolated wing image (the wingIsolator-masked input) in the output folder. "
        "Requires the wing isolation step to be enabled and a wing-isolation model to be "
        "configured in the Models tab."
    ),
    "chopped_image": ("Keep the hinge-removed image produced by HingeChopper in the output folder."),
    "landmarks_overlay": ("Keep the landmark-points overlay PNG (rendered over the input image) in the output folder."),
    "segmentation_overlay": ("Keep the raw vein/intervein semantic-segmentation overlay PNG in the output folder."),
    "geojson": ("Keep the per-wing GeoJSON file (named veins + intervein regions) in the output folder."),
}


_LANDMARK_DISPLAY_NAMES: dict[str, str] = {
    "acv_a": "ACV-L3 junction",
    "acv_p": "ACV-L4 junction",
    "alula_notch": "alula notch",
    "dtip": "L3 distal end",
    "l1_rs": "L1-Rs junction",
    "l2_d": "L2 distal end",
    "l2_l3": "L2-L3-Rs junction",
    "l4_d": "L4 distal end",
    "l4_l5": "L4-L5 junction",
    "l5_d": "L5 distal end",
    "pcv_a": "PCV-L4 junction",
    "pcv_p": "PCV-L5 junction",
    "subcostal_break": "subcostal break",
}


def _merge_gate_override(base: dict, override: dict) -> dict:
    """Merge a user gate override onto the sidecar gate config for UI rebuild.

    Delegates to the LandmarkLocator predictor's deep-merge so that every
    field in the override survives — including per-metric `enabled` flags
    and `min_metric_failures_to_reject` — and so future gate-config fields
    don't need a parallel allowlist edit here. Matches the merge the
    predictor itself performs at runtime, so the UI's initial state stays
    consistent with what the pipeline will actually apply.
    """
    from landmark_locator.inference.predict import _deep_merge

    return _deep_merge(base, override)


class PipelineConfigDialog(QDialog):
    """Edit a PipelineConfig via a tabbed form."""

    # Widget-kind constants for the dispatch table.
    _KIND_FLOAT = "float"
    _KIND_INT = "int"
    _KIND_OPT_FLOAT = "opt_float"
    _KIND_OPT_INT = "opt_int"
    _KIND_ENUM_LIST = "enum_list"
    _KIND_FLOAT_LIST = "float_list"
    _KIND_BOOL = "bool"
    _KIND_CHOICE = "choice"
    _KIND_STR_SET = "str_set"

    def __init__(
        self,
        config: PipelineConfig,
        parent=None,
        include_unreliable_landmarks: bool = False,
        input_path: str = "",
        landmark_model_path: str = "",
        segmentation_model_path: str = "",
        gate_override: dict | None = None,
        wing_expand_fraction: float = 0.05,
        wing_isolation_model_path: str = "",
        landmark_target_um_per_px: float | None = None,
        segmentation_target_um_per_px: float | None = None,
        wing_isolation_target_um_per_px: float | None = None,
        active_rescale_target: str = "segmentation",
        rescale_tolerance_low: float = 0.85,
        rescale_tolerance_high: float = 1.15,
    ):
        super().__init__(parent)
        self.setWindowTitle("Pipeline Settings")
        # Let users shrink the window below its natural size hint — tab
        # contents are wrapped in a QScrollArea in _build_ui so anything
        # that doesn't fit becomes scrollable.
        self.setMinimumSize(360, 240)
        self.resize(720, 640)
        # Restore the dialog geometry the user last left (saved in done());
        # the resize() above is the first-open default.
        self._settings = QSettings("TRACE", "WingAnalysisPipeline")
        _saved_geometry = self._settings.value("settings_dialog_geometry")
        if _saved_geometry is not None:
            self.restoreGeometry(_saved_geometry)
            # Reported v0.1.60 symptom: clicking Advanced Settings did
            # nothing visible. Root cause was a saved geometry pointing
            # at a monitor that no longer existed (laptop user
            # disconnected an external display), so the dialog was
            # technically shown but on coordinates outside any current
            # screen. Drop the restore + fall back to default when the
            # center point isn't on any available screen.
            try:
                from PyQt5.QtWidgets import QApplication

                center = self.frameGeometry().center()
                screens = QApplication.screens()
                on_screen = any(s.geometry().contains(center) for s in screens)
                if not on_screen:
                    self.resize(720, 640)
                    self.move(0, 0)  # let WM reposition on first show
                    self._settings.remove("settings_dialog_geometry")
            except Exception:
                # Defensive: don't let geometry validation block the
                # dialog from constructing.
                pass
        self._original_config = config
        self._calib_input_path = input_path
        self._calib_lm_path = landmark_model_path
        self._calib_seg_path = segmentation_model_path
        self._gate_panel = None  # populated by _populate_landmark_gate_section when a model is loaded
        self._initial_gate_override = gate_override
        # Holds the currently-open native file picker so Python's GC doesn't
        # free it between open() and the user clicking Open / Cancel. See
        # TRACE.gui._open_native_picker_async for the open()-vs-exec_()
        # rationale (avoids the napari × nested-event-loop interaction
        # that kills the file list inside Advanced Settings once the live
        # preview pane has loaded napari).
        self._active_picker = None
        # (kind, widget, extra) tuples indexed by PipelineConfig field name.
        self._widgets: dict[str, tuple[str, Any, Any]] = {}
        # Stage 1 (resolutionAdjust) — per-model training-µm/px targets, which
        # model's target drives the global rescale, and the tolerance band. None
        # for any per-model target = "not configured; do not rescale on its
        # behalf". Captured here so `_build_models_tab` can seed its widgets.
        self._initial_landmark_target_um_per_px = landmark_target_um_per_px
        self._initial_segmentation_target_um_per_px = segmentation_target_um_per_px
        self._initial_wing_isolation_target_um_per_px = wing_isolation_target_um_per_px
        self._initial_active_rescale_target = active_rescale_target or "segmentation"
        self._initial_rescale_tolerance_low = rescale_tolerance_low
        self._initial_rescale_tolerance_high = rescale_tolerance_high
        self._build_ui()
        self._load_from_config(config)
        self._include_unreliable_landmarks_chk.setChecked(include_unreliable_landmarks)
        self._wing_expand_spin.setValue(float(wing_expand_fraction))
        if wing_isolation_model_path:
            self._wing_model_edit.setText(wing_isolation_model_path)

        # Live vein-tuning preview (TODO #14). Additive and fully guarded: any
        # failure leaves the dialog working exactly as before, without a preview.
        # Implementation lives in the sibling liveSettings/ module so it doesn't
        # clutter identifyFeatures. The pane is collapsed until the user opts in.
        try:
            import sys as _sys
            from pathlib import Path as _Path

            _ls_dir = str(_Path(__file__).resolve().parent.parent / "liveSettings")
            if _ls_dir not in _sys.path:
                _sys.path.insert(0, _ls_dir)
            from live_tune.dialog_integration import attach_live_preview as _attach_live_preview

            _attach_live_preview(self)
        except Exception as _live_preview_exc:  # noqa: BLE001 - preview is optional; never block the dialog
            # IMPORTANT: this branch must not raise. The previous code
            # referenced ``logger`` which was never defined/imported in
            # this module, so any failure here (e.g. liveSettings/ not
            # bundled in the Windows installer → ModuleNotFoundError on
            # ``live_tune``) raised NameError instead of being swallowed,
            # which bubbled out of __init__ and prevented the entire
            # Advanced Settings dialog from opening (issue #15).
            import traceback as _tb_lp

            from TRACE.startup_log import log as _slog_lp

            _slog_lp(
                "settings_dialog: live preview unavailable, continuing without it\n"
                + "".join(_tb_lp.format_exception(type(_live_preview_exc), _live_preview_exc, _live_preview_exc.__traceback__))
            )

    def done(self, result: int) -> None:  # noqa: N802 — Qt API
        # Persist the dialog geometry (covers OK, Cancel, Esc and the window
        # close button) so it reopens at the size the user last left.
        self._settings.setValue("settings_dialog_geometry", self.saveGeometry())
        self._settings.sync()
        super().done(result)

    # -----------------------------------------------------------------------
    # Public API
    # -----------------------------------------------------------------------
    def get_config(self) -> PipelineConfig:
        """Build a new PipelineConfig from the current widget state."""
        kwargs: dict[str, Any] = {}
        for name, (kind, widget, extra) in self._widgets.items():
            if kind == self._KIND_FLOAT:
                # When a spinbox has a QLineEdit placeholder set (via the
                # _PlaceholderSpinBox.set_placeholder helper) and the user left
                # it at the minimum, treat that as None — the placeholder state
                # means "no value entered yet" (e.g. um_per_px before the user
                # has supplied a conversion factor).
                placeholder = widget.lineEdit().placeholderText() if hasattr(widget, "lineEdit") else ""
                if placeholder and widget.value() <= widget.minimum():
                    kwargs[name] = None
                else:
                    kwargs[name] = widget.value()
            elif kind == self._KIND_INT:
                kwargs[name] = widget.value()
            elif kind == self._KIND_OPT_FLOAT:
                check, spin = extra
                kwargs[name] = spin.value() if check.isChecked() else None
            elif kind == self._KIND_OPT_INT:
                check, spin = extra
                kwargs[name] = spin.value() if check.isChecked() else None
            elif kind == self._KIND_ENUM_LIST:
                enum_cls = extra
                selected: list = []
                for i in range(widget.count()):
                    item = widget.item(i)
                    if item.checkState() == Qt.Checked:
                        selected.append(enum_cls(item.data(Qt.UserRole)))
                kwargs[name] = selected
            elif kind == self._KIND_FLOAT_LIST:
                text = widget.text().strip()
                if not text:
                    kwargs[name] = []
                else:
                    kwargs[name] = [float(x.strip()) for x in text.split(",") if x.strip()]
            elif kind == self._KIND_BOOL:
                kwargs[name] = widget.isChecked()
            elif kind == self._KIND_CHOICE:
                kwargs[name] = widget.currentText()
            elif kind == self._KIND_STR_SET:
                boxes, options = extra
                kwargs[name] = [opt for opt in options if boxes[opt].isChecked()]
        # The main window's InlineGeneralPanel owns several PipelineConfig
        # fields that the dialog no longer renders (Scale + Output options).
        # Preserve them from the input config so OK doesn't wipe them.
        for field in ("um_per_px", "vein_opacity", "intervein_opacity", "vein_colors", "region_colors"):
            kwargs[field] = getattr(self._original_config, field)
        return PipelineConfig(**kwargs)

    # -----------------------------------------------------------------------
    # UI construction
    # -----------------------------------------------------------------------
    def _wrap_scrollable(self, content: QWidget) -> QScrollArea:
        # Lets the dialog shrink below the tab's natural size hint; a
        # vertical scroll bar appears whenever the tab content is taller
        # than the available space.
        area = QScrollArea()
        area.setWidgetResizable(True)
        area.setFrameShape(QScrollArea.NoFrame)
        area.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        area.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        area.setWidget(content)
        return area

    def _build_ui(self):
        layout = QVBoxLayout(self)

        # Preset row — applies every field listed in the preset dict (any
        # PipelineConfig field, not just pruning/bridging). Presets are loaded
        # from TRACE/presets/*.json, so adding a new preset is just dropping a
        # JSON file in that folder — no code change needed.
        self._presets = load_presets()
        preset_row = QHBoxLayout()
        preset_row.addWidget(QLabel("Settings preset:"))
        self._preset_combo = QComboBox()
        self._preset_combo.setToolTip(
            "Named bundles of pipeline-config settings stored as JSON in TRACE/presets/. "
            "Pick one and click Apply preset to overwrite the listed fields."
        )
        for preset_name in self._presets:
            self._preset_combo.addItem(preset_name)
        preset_row.addWidget(self._preset_combo, stretch=1)
        apply_btn = QPushButton("Apply preset")
        apply_btn.setToolTip("Overwrite all fields listed in the selected preset. Fields not in the preset are kept.")
        apply_btn.clicked.connect(self._apply_selected_preset)
        preset_row.addWidget(apply_btn)
        layout.addLayout(preset_row)

        # Config-file row: Import + Save sit under Apply preset because
        # they're the same kind of action (loading / persisting the whole
        # config), grouped visually with the preset row instead of buried
        # in the OK / Cancel / Restore Defaults strip at the bottom.
        config_row = QHBoxLayout()
        config_row.addStretch(1)
        import_btn = QPushButton("Import…")
        import_btn.setToolTip(
            "Load a previously-exported pipeline-config JSON file. Replaces the "
            "settings shown in this dialog (commit with OK, discard with Cancel)."
        )
        import_btn.clicked.connect(self._import_config)
        config_row.addWidget(import_btn)
        save_btn = QPushButton("Save…")
        save_btn.setToolTip(
            "Save the current pipeline-config (as edited in this dialog) to a JSON file "
            "for reuse (CLI --config or this dialog's Import)."
        )
        save_btn.clicked.connect(self._export_config)
        config_row.addWidget(save_btn)
        layout.addLayout(config_row)

        self._tabs = QTabWidget()
        layout.addWidget(self._tabs, stretch=1)

        # Landmark confidence gates and pipeline quality filters are both
        # "quality" — merged into a single Quality tab (was two separate
        # Landmarks / Quality tabs). The landmark-gate panel section
        # inside it is rebuilt in place when the landmark model path
        # changes, via _rebuild_landmark_gate_section, so the surrounding
        # quality-filter widgets don't lose in-progress edits.
        self._tabs.addTab(self._wrap_scrollable(self._build_quality_tab()), "Quality")
        self._tabs.addTab(self._wrap_scrollable(self._build_models_tab()), "Models")
        self._tabs.addTab(self._wrap_scrollable(self._build_wing_graph_tab()), "Wing Graph")
        self._tabs.addTab(self._wrap_scrollable(self._build_tracing_tab()), "Tracing")
        self._tabs.addTab(self._wrap_scrollable(self._build_intervein_tab()), "Intervein")

        # Rebuild the Landmarks tab whenever the landmark model path changes
        # so the gate panel re-reads gate_config.yaml from the new folder.
        # Connected here (not in _build_models_tab) because _lm_model_edit
        # only exists after the Models tab is built, and the Landmarks tab
        # needs to exist (we replace it in-place).
        self._lm_model_edit.textChanged.connect(self._rebuild_landmark_gate_section)

        # Import/Save sit at the top under Apply preset — see config_row
        # above. This bar is only OK / Cancel / Restore Defaults now.
        btns = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel | QDialogButtonBox.RestoreDefaults)
        btns.accepted.connect(self.accept)
        btns.rejected.connect(self.reject)
        btns.button(QDialogButtonBox.RestoreDefaults).clicked.connect(self._reset_defaults)
        layout.addWidget(btns)

    # -----------------------------------------------------------------------
    # Import / Save pipeline-config JSON
    # -----------------------------------------------------------------------
    def _import_config(self) -> None:
        # Always open at the bundled TRACE/presets/ folder — user explicitly
        # asked for a fixed landing spot instead of per-widget last-dir
        # memory (which the OS was defeating anyway via NSOpenPanel's
        # process-wide cache when napari was loaded).
        from TRACE.gui import _open_native_picker_async

        _open_native_picker_async(
            self,
            "Import Pipeline Config",
            self._bundled_presets_dir(),
            self._on_import_config_picked,
            name_filter="JSON (*.json);;All Files (*)",
        )

    def _on_import_config_picked(self, path: str) -> None:
        if not path:
            return
        from TRACE.config_io import load_settings

        try:
            new_config, gate_override, gui_state = load_settings(Path(path))
        except Exception as e:  # noqa: BLE001
            QMessageBox.critical(self, "Import failed", f"Could not load config:\n{e}")
            return
        # Refresh every dispatch-table widget from the imported config.
        self._load_from_config(new_config)
        # Replace the snapshot so get_config() preserves the new inline-panel
        # fields (um_per_px, opacities, colors) when OK is clicked.
        self._original_config = new_config
        # gate_override is None when the file predates the field — leave the
        # current override untouched in that case.
        if gate_override is not None:
            self._initial_gate_override = gate_override
            # Rebuild the Landmarks tab so the gate panel reflects the imported
            # override against the (possibly changed) landmark-model path.
            self._rebuild_landmark_gate_section()
        # Apply GUI-only state (Settings-tab toggles, model paths, custom
        # distances, etc.) directly to the host window. This is an immediate,
        # non-revertible operation — Cancel'ing the dialog does NOT undo
        # imported GUI flags. The PipelineConfig portion DOES still revert
        # on Cancel because it lives in the dialog's working buffer.
        host = self.parent()
        if gui_state and host is not None and hasattr(host, "apply_gui_state"):
            host.apply_gui_state(gui_state)
            # Re-seed dialog widgets that mirror host state we just updated.
            if hasattr(host, "_wing_expand_fraction"):
                self._wing_expand_spin.setValue(float(host._wing_expand_fraction))
            if hasattr(host, "_wing_isolation_model_path"):
                self._wing_model_edit.setText(str(host._wing_isolation_model_path or ""))
            if hasattr(host, "_landmark_model_path"):
                self._lm_model_edit.setText(str(host._landmark_model_path or ""))
            if hasattr(host, "_segmentation_model_path"):
                self._seg_model_edit.setText(str(host._segmentation_model_path or ""))
            if hasattr(host, "_include_unreliable_landmarks"):
                self._include_unreliable_landmarks_chk.setChecked(bool(host._include_unreliable_landmarks))
        self._refresh_preset_dropdown()
        QMessageBox.information(self, "Import complete", f"Imported pipeline-config from:\n{path}")
        # Keep the dialog focused after the picker + confirm message box
        # tear down — apply_gui_state pushes updates through the host's
        # inline panels, and on macOS the unparented picker + parented
        # message box can leave the host window as the active window
        # instead of returning focus to this dialog. Defer via singleShot
        # so it runs AFTER any pending focus-change events queued during
        # the message-box close.
        from PyQt5.QtCore import QTimer as _QTimer

        _QTimer.singleShot(0, self._reclaim_focus)

    def _export_config(self) -> None:
        # Always open at the bundled TRACE/presets/ folder — user asked
        # for a fixed landing spot. Also seeds the default filename so
        # saved presets show up in the preset dropdown at the top of
        # this dialog (which reads TRACE/presets/*.json).
        from TRACE.gui import _open_native_picker_async

        presets_dir = self._bundled_presets_dir()
        default_path = str(Path(presets_dir) / "pipeline_config.json") if presets_dir else "pipeline_config.json"

        _open_native_picker_async(
            self,
            "Export Pipeline Config",
            default_path,
            self._on_export_config_picked,
            name_filter="JSON (*.json);;All Files (*)",
            save=True,
        )

    @staticmethod
    def _bundled_presets_dir() -> str:
        """Absolute path to the TRACE/presets/ folder that ships in the app
        install, or "" if the folder is missing.

        Used to force-open Import / Save at a consistent, discoverable
        location instead of wherever the OS's picker cache last landed.
        """
        p = Path(__file__).resolve().parent / "presets"
        return str(p) if p.is_dir() else ""

    @staticmethod
    def _bundled_models_dir() -> str:
        """Absolute path to the TRACE/models/ folder that ships in the app
        install, or "" if the folder is missing.

        Used to force-open the wing-isolation / landmark / segmentation
        model browses at a consistent location containing the bundled
        model folders (landmarks/, vein-intervein/, wingIsolation/).
        """
        p = Path(__file__).resolve().parent / "models"
        return str(p) if p.is_dir() else ""

    def _on_export_config_picked(self, path: str) -> None:
        if not path:
            return
        from TRACE.config_io import save_settings

        # Pull GUI-only state from the host window so the saved file
        # round-trips every user-visible setting, not just PipelineConfig.
        host = self.parent()
        gui_state = host.get_gui_state() if host is not None and hasattr(host, "get_gui_state") else None
        # Dialog-side widgets that mirror host state (wing_expand, model paths,
        # include-unreliable-landmarks) override the host snapshot since the
        # user may have edited them in this dialog session without Apply.
        if gui_state is not None:
            gui_state["wing_expand_fraction"] = self.get_wing_expand_fraction()
            gui_state["wing_isolation_model_path"] = self.get_wing_isolation_model_path()
            gui_state["landmark_model_path"] = self.get_landmark_model_path()
            gui_state["segmentation_model_path"] = self.get_segmentation_model_path()
            gui_state["landmark_target_um_per_px"] = self.get_landmark_target_um_per_px()
            gui_state["segmentation_target_um_per_px"] = self.get_segmentation_target_um_per_px()
            gui_state["wing_isolation_target_um_per_px"] = self.get_wing_isolation_target_um_per_px()
            gui_state["active_rescale_target"] = self.get_active_rescale_target()
            gui_state["rescale_tolerance_low"] = self.get_rescale_tolerance_low()
            gui_state["rescale_tolerance_high"] = self.get_rescale_tolerance_high()
            gui_state["include_unreliable_landmarks"] = self.get_include_unreliable_landmarks()
        try:
            save_settings(self.get_config(), self.get_gate_override(), Path(path), gui_state=gui_state)
        except Exception as e:  # noqa: BLE001
            QMessageBox.critical(self, "Export failed", f"Could not save config:\n{e}")
            return
        self._refresh_preset_dropdown()
        QMessageBox.information(self, "Save complete", f"Saved pipeline-config to:\n{path}")
        # See _on_import_config_picked — same reclaim after the picker +
        # message-box teardown that otherwise lets the host window steal
        # activation on macOS.
        from PyQt5.QtCore import QTimer as _QTimer

        _QTimer.singleShot(0, self._reclaim_focus)

    def _reclaim_focus(self) -> None:
        """Bring this dialog back to the front + make it the active window.

        Used after Import / Save flows where the file picker (unparented to
        avoid the napari × modal-grab bug) + confirmation QMessageBox can
        leave the host TRACE window as the active window on macOS, pushing
        this dialog behind it. Only fires if we're still visible — if the
        user closed the dialog mid-picker, don't raise a dead window.
        """
        if self.isVisible():
            self.raise_()
            self.activateWindow()

    def _refresh_preset_dropdown(self) -> None:
        """Rescan TRACE/presets/*.json and repopulate the preset combo.

        Called after Import (in case the user pointed at a file inside the
        presets folder) and after Save (in case the user just saved a new
        preset there) so newly-added JSON files are selectable without
        closing and reopening TRACE. Preserves the current selection when
        it still exists in the refreshed list.
        """
        current = self._preset_combo.currentText()
        self._presets = load_presets()
        self._preset_combo.blockSignals(True)
        try:
            self._preset_combo.clear()
            for preset_name in self._presets:
                self._preset_combo.addItem(preset_name)
            if current in self._presets:
                self._preset_combo.setCurrentText(current)
        finally:
            self._preset_combo.blockSignals(False)

    def get_wing_expand_fraction(self) -> float:
        return float(self._wing_expand_spin.value())

    def get_wing_isolation_model_path(self) -> str:
        return self._wing_model_edit.text().strip()

    def _select_wing_model_folder(self):
        from TRACE.gui import _open_native_picker_async

        _open_native_picker_async(
            self,
            "Select Wing-Identification Model Folder",
            self._bundled_models_dir(),
            self._on_wing_model_folder_picked,
            folder=True,
        )

    def _on_wing_model_folder_picked(self, folder: str) -> None:
        if folder:
            self._wing_model_edit.setText(folder)

    # -----------------------------------------------------------------------
    # GUI-only flag accessors (not part of PipelineConfig)
    # -----------------------------------------------------------------------
    def get_include_unreliable_landmarks(self) -> bool:
        return self._include_unreliable_landmarks_chk.isChecked()

    def get_gate_override(self) -> dict | None:
        """Confidence-gate override built from the Landmarks tab, or None if untouched."""
        if self._gate_panel is None:
            return self._initial_gate_override
        return self._gate_panel.result_override()

    def _populate_landmark_gate_section(self) -> None:
        """Fill self._landmark_gate_container with the per-landmark gate panel
        (or a status message if no model is loaded / gate_config is unreadable).

        Called once at build time and again from _rebuild_landmark_gate_section
        whenever the landmark-model path changes — the surrounding widgets in
        the Quality tab (include-low-confidence flag, solidity / fragmentation /
        vein-association / required-veins groups) are NOT rebuilt, so their
        in-progress edits are preserved.
        """
        container = self._landmark_gate_container
        # Purge existing widgets from prior populate calls before rebuilding.
        while container.layout().count():
            item = container.layout().takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()

        layout = container.layout()

        if not self._calib_lm_path:
            msg = QLabel(
                "Select a landmark model (.pt or fold folder) on the main window first, "
                "then reopen this dialog to edit per-landmark gate thresholds."
            )
            msg.setWordWrap(True)
            msg.setStyleSheet(f"color: {_ct().text_muted}; padding: 12px;")
            layout.addWidget(msg)
            return

        # Bind the logger names BEFORE the outer try so they're defined in
        # the except clause regardless of whether the try ever reached the
        # inner import (the previous version crashed with UnboundLocalError
        # when read_gate_config raised before the inner try ran).
        try:
            from TRACE.startup_log import log as _log
            from TRACE.startup_log import log_exception as _log_exc
        except Exception:
            _log = None
            _log_exc = None
        try:
            from landmark_locator.scripts.gui import GateConfigPanel, read_gate_config

            if _log is not None:
                _log(f"Landmark gate section: reading gate config from {self._calib_lm_path}")
            gate_config, landmark_order = read_gate_config(Path(self._calib_lm_path))
            if _log is not None:
                _log("Landmark gate section: read_gate_config OK")
        except Exception as exc:
            if _log_exc is not None:
                _log_exc("Landmark gate section: read_gate_config failed", exc)
            err = QLabel(f"Could not read gate config from {self._calib_lm_path}: {exc}")
            err.setWordWrap(True)
            err.setStyleSheet(f"color: {_ct().error_text}; padding: 12px;")
            layout.addWidget(err)
            return

        if not landmark_order:
            # read_gate_config returned the library DEFAULT_GATE_CONFIG fallback
            # because the model folder has no gate_config.yaml. Tell the user
            # what's missing rather than rendering an empty panel.
            msg = QLabel(
                f"The selected landmark model folder has no gate_config.yaml:\n"
                f"  {self._calib_lm_path}\n\n"
                "Switch to a model that ships a sidecar gate_config.yaml "
                "(e.g. TRACE/models/landmarks), or click Restore Defaults to "
                "reset the landmark-model path to the bundled default."
            )
            msg.setWordWrap(True)
            msg.setStyleSheet(f"color: {_ct().warning}; padding: 12px;")
            layout.addWidget(msg)
            return

        # Merge any persisted GUI override on top so the panel shows the user's last edits.
        # NOTE: TRACE deliberately does not pass `sidecar_path` — gate edits here belong
        # to the project's TRACE settings, not the model's defaults. The model-folder
        # sidecar stays read-only from TRACE, serving as the safe fallback when the
        # user resets / wipes their TRACE settings.
        if self._initial_gate_override:
            gate_config = _merge_gate_override(gate_config, self._initial_gate_override)

        self._gate_panel = GateConfigPanel(
            gate_config,
            landmark_order,
            container,
            display_names=_LANDMARK_DISPLAY_NAMES,
        )
        layout.addWidget(self._gate_panel)

    def _make_target_row(
        self,
        initial_value: float | None,
        model_label: str,
        get_model_path_fn,
    ) -> tuple[QHBoxLayout, QDoubleSpinBox, QPushButton]:
        """Build a "Training µm/px: [____] [Auto-detect]" row for one model.

        `get_model_path_fn` is a no-arg callable returning the currently-selected
        model path — used by Auto-detect to seed the folder picker so the user
        often doesn't have to navigate at all.
        """
        row = QHBoxLayout()
        row.addWidget(QLabel("Training µm/px:"))
        spin = QDoubleSpinBox()
        spin.setRange(0.0, 100.0)
        spin.setDecimals(4)
        spin.setSingleStep(0.001)
        spin.setValue(float(initial_value) if initial_value and initial_value > 0 else 0.0)
        spin.setToolTip(
            f"µm/px of the images this {model_label} model was trained on. "
            "Stage 1 (resolutionAdjust) rescales each input toward this value when "
            "its scale differs from the tolerance band. Leave at 0 to disable."
        )
        row.addWidget(spin, stretch=1)
        btn = QPushButton("Auto-detect")
        btn.setToolTip(
            "Pick a folder of training images. Their TIFF metadata (XResolution / OME-XML "
            "PhysicalSizeX) is averaged into the field above. Images without metadata are "
            "skipped; the dialog will report n/total."
        )
        btn.clicked.connect(lambda: self._autodetect_target_um_per_px(spin, get_model_path_fn))
        row.addWidget(btn)
        return row, spin, btn

    def _build_models_tab(self) -> QWidget:
        """Pipeline model paths: landmark, segmentation, and (optional) wing isolation.

        Each model section also carries a "Training µm/px" field + Auto-detect
        button consumed by Stage 1 (resolutionAdjust). The bottom group picks
        which model's target drives the actual rescale and sets the tolerance
        band that decides when a rescale is worth doing.
        """
        w = QWidget()
        layout = QVBoxLayout(w)

        # -- Landmark model --
        gb = QGroupBox("Landmark points")
        lm_layout = QVBoxLayout(gb)
        lm_layout.addWidget(QLabel("Model folder (contains 5 folds and YAML):"))
        lm_row = QHBoxLayout()
        self._lm_model_edit = QLineEdit()
        self._lm_model_edit.setReadOnly(True)
        self._lm_model_edit.setPlaceholderText("Select fold folder...")
        self._lm_model_edit.setToolTip(
            "Pick a model folder containing best_fold*.pt directly (5-fold ensemble). "
            "The folder also hosts the optional gate_config.yaml sidecar with "
            "per-model gate defaults, alongside training_chart.png."
        )
        btn_lm_folder = QPushButton("Browse...")
        btn_lm_folder.setToolTip(
            "Pick a model folder containing best_fold*.pt directly (5-fold ensemble). "
            "The folder also hosts the optional gate_config.yaml sidecar."
        )
        btn_lm_folder.clicked.connect(self._select_landmark_model_folder)
        lm_row.addWidget(self._lm_model_edit, stretch=1)
        lm_row.addWidget(btn_lm_folder)
        lm_layout.addLayout(lm_row)
        if self._calib_lm_path:
            self._lm_model_edit.setText(self._calib_lm_path)
        lm_target_row, self._lm_target_spin, self._lm_target_btn = self._make_target_row(
            self._initial_landmark_target_um_per_px,
            "landmark",
            lambda: self._lm_model_edit.text(),
        )
        lm_layout.addLayout(lm_target_row)
        layout.addWidget(gb)

        # -- Segmentation model --
        gb = QGroupBox("Wing features")
        seg_layout = QVBoxLayout(gb)
        seg_layout.addWidget(QLabel("Model folder (contains metadata.json + weights):"))
        seg_row = QHBoxLayout()
        self._seg_model_edit = QLineEdit()
        self._seg_model_edit.setReadOnly(True)
        self._seg_model_edit.setPlaceholderText("Select segmentation model folder...")
        self._seg_model_edit.setToolTip(
            "modelTOjson model directory (contains metadata.json + weights). Produces the "
            "vein/intervein semantic segmentation used by identifyFeatures."
        )
        btn_seg = QPushButton("Browse...")
        btn_seg.setToolTip("Pick the segmentation model folder.")
        btn_seg.clicked.connect(self._select_segmentation_model_folder)
        seg_row.addWidget(self._seg_model_edit, stretch=1)
        seg_row.addWidget(btn_seg)
        seg_layout.addLayout(seg_row)
        if self._calib_seg_path:
            self._seg_model_edit.setText(self._calib_seg_path)
        seg_target_row, self._seg_target_spin, self._seg_target_btn = self._make_target_row(
            self._initial_segmentation_target_um_per_px,
            "wing-features",
            lambda: self._seg_model_edit.text(),
        )
        seg_layout.addLayout(seg_target_row)
        layout.addWidget(gb)

        # -- Wing isolation model (Stage 2, optional) --
        # The enable/disable checkbox lives on the General tab; this group
        # holds the model path + buffer parameter only.
        gb = QGroupBox("Wing isolation (optional)")
        wig_layout = QVBoxLayout(gb)
        wig_layout.addWidget(QLabel("Model folder (contains metadata.json + weights):"))
        wing_row = QHBoxLayout()
        self._wing_model_edit = QLineEdit()
        self._wing_model_edit.setReadOnly(True)
        self._wing_model_edit.setPlaceholderText("Select wing-identification model folder...")
        self._wing_model_edit.setToolTip(
            "modelTOjson model directory for wing/background segmentation. The model's "
            "metadata.json must declare a 'wing' class."
        )
        self._wing_model_browse = QPushButton("Browse...")
        self._wing_model_browse.clicked.connect(self._select_wing_model_folder)
        wing_row.addWidget(self._wing_model_edit, stretch=1)
        wing_row.addWidget(self._wing_model_browse)
        wig_layout.addLayout(wing_row)

        wing_form = QFormLayout()
        self._wing_expand_spin = QDoubleSpinBox()
        self._wing_expand_spin.setRange(0.0, 1.0)
        self._wing_expand_spin.setDecimals(3)
        self._wing_expand_spin.setSingleStep(0.01)
        self._wing_expand_spin.setValue(0.05)
        self._wing_expand_spin.setToolTip(
            "Stage 2 mask buffer, as a fraction of sqrt(wing area). "
            "0 = exact polygon (no buffer); 0.05 = ~5% expansion. "
            "Used only when wing isolation is enabled."
        )
        wing_form.addRow("Buffer (× √area)", self._wing_expand_spin)
        wig_layout.addLayout(wing_form)
        wing_target_row, self._wing_target_spin, self._wing_target_btn = self._make_target_row(
            self._initial_wing_isolation_target_um_per_px,
            "wing-isolation",
            lambda: self._wing_model_edit.text(),
        )
        wig_layout.addLayout(wing_target_row)
        layout.addWidget(gb)

        # -- Resolution adjustment (Stage 1) --
        ra_gb = QGroupBox("Resolution adjustment")
        ra_layout = QVBoxLayout(ra_gb)
        ra_layout.addWidget(
            QLabel(
                "Rescale each input image so the active model's training resolution is matched. "
                "Skipped when the input µm/px is inside the tolerance band."
            )
        )
        ra_layout.addWidget(QLabel("Use this model's target µm/px for the rescale:"))
        radio_row = QHBoxLayout()
        self._ra_radio_landmark = QRadioButton("Landmark")
        self._ra_radio_segmentation = QRadioButton("Wing features")
        self._ra_radio_wing = QRadioButton("Wing isolation")
        self._ra_radio_group = QButtonGroup(self)
        self._ra_radio_group.addButton(self._ra_radio_landmark)
        self._ra_radio_group.addButton(self._ra_radio_segmentation)
        self._ra_radio_group.addButton(self._ra_radio_wing)
        sel = (self._initial_active_rescale_target or "segmentation").lower()
        if sel == "landmark":
            self._ra_radio_landmark.setChecked(True)
        elif sel in ("wing_isolation", "wing"):
            self._ra_radio_wing.setChecked(True)
        else:
            self._ra_radio_segmentation.setChecked(True)
        radio_row.addWidget(self._ra_radio_landmark)
        radio_row.addWidget(self._ra_radio_segmentation)
        radio_row.addWidget(self._ra_radio_wing)
        radio_row.addStretch(1)
        ra_layout.addLayout(radio_row)

        tol_form = QFormLayout()
        self._ra_tol_low_spin = QDoubleSpinBox()
        self._ra_tol_low_spin.setRange(0.01, 10.0)
        self._ra_tol_low_spin.setDecimals(3)
        self._ra_tol_low_spin.setSingleStep(0.01)
        self._ra_tol_low_spin.setValue(float(self._initial_rescale_tolerance_low))
        self._ra_tol_low_spin.setToolTip(
            "Lower edge of the pass-through ratio band (input_µm/px ÷ target_µm/px). "
            "Inputs inside [low, high] are not rescaled. 0.85 ≈ 15% smaller-than-target "
            "still acceptable."
        )
        self._ra_tol_high_spin = QDoubleSpinBox()
        self._ra_tol_high_spin.setRange(0.01, 10.0)
        self._ra_tol_high_spin.setDecimals(3)
        self._ra_tol_high_spin.setSingleStep(0.01)
        self._ra_tol_high_spin.setValue(float(self._initial_rescale_tolerance_high))
        self._ra_tol_high_spin.setToolTip(
            "Upper edge of the pass-through ratio band. 1.15 ≈ 15% larger-than-target " "still acceptable."
        )
        # Each row is `[spinbox]  = <ratio × active-target> µm/px`. The right-hand
        # label updates live whenever the tolerance, the active-model radio, or
        # the active model's Training µm/px changes.
        low_row = QHBoxLayout()
        low_row.addWidget(self._ra_tol_low_spin)
        self._ra_tol_low_label = QLabel("")
        self._ra_tol_low_label.setStyleSheet(f"color: {_ct().text_placeholder};")
        low_row.addWidget(self._ra_tol_low_label, stretch=1)
        low_container = QWidget()
        low_container.setLayout(low_row)
        high_row = QHBoxLayout()
        high_row.addWidget(self._ra_tol_high_spin)
        self._ra_tol_high_label = QLabel("")
        self._ra_tol_high_label.setStyleSheet(f"color: {_ct().text_placeholder};")
        high_row.addWidget(self._ra_tol_high_label, stretch=1)
        high_container = QWidget()
        high_container.setLayout(high_row)
        tol_form.addRow("Tolerance low (ratio)", low_container)
        tol_form.addRow("Tolerance high (ratio)", high_container)
        ra_layout.addLayout(tol_form)
        layout.addWidget(ra_gb)

        for spin in (
            self._lm_target_spin,
            self._seg_target_spin,
            self._wing_target_spin,
            self._ra_tol_low_spin,
            self._ra_tol_high_spin,
        ):
            spin.valueChanged.connect(self._update_tolerance_um_labels)
        self._ra_radio_group.buttonClicked.connect(lambda *_: self._update_tolerance_um_labels())
        self._update_tolerance_um_labels()

        layout.addStretch(1)
        return w

    def _active_target_um_per_px(self) -> float | None:
        """Training µm/px of the model the rescale radio currently points at."""
        if self._ra_radio_landmark.isChecked():
            v = float(self._lm_target_spin.value())
        elif self._ra_radio_wing.isChecked():
            v = float(self._wing_target_spin.value())
        else:
            v = float(self._seg_target_spin.value())
        return v if v > 0 else None

    def _update_tolerance_um_labels(self) -> None:
        """Refresh the µm/px labels beside the tolerance ratio spinboxes."""
        target = self._active_target_um_per_px()
        if target is None:
            placeholder = "(set Training µm/px)"
            self._ra_tol_low_label.setText(placeholder)
            self._ra_tol_high_label.setText(placeholder)
            return
        low = self._ra_tol_low_spin.value() * target
        high = self._ra_tol_high_spin.value() * target
        self._ra_tol_low_label.setText(f"= {low:.4f} µm/px")
        self._ra_tol_high_label.setText(f"= {high:.4f} µm/px")

    # -----------------------------------------------------------------------
    # Stage 1 (resolutionAdjust) helpers
    # -----------------------------------------------------------------------
    def _autodetect_target_um_per_px(self, spin: QDoubleSpinBox, get_model_path_fn) -> None:
        """Open a folder picker, run autodetect, write the average into `spin`."""
        from TRACE.gui import _open_native_picker_async, _picker_initial_path

        seed_path = ""
        try:
            seed_path = get_model_path_fn() or ""
        except Exception:
            seed_path = ""
        # Bind ``spin`` (the per-stage target spinbox) into the callback
        # so the result lands in the right widget — the auto-detect
        # helper is shared by landmark / segmentation / wing isolation,
        # each with its own spinbox.
        _open_native_picker_async(
            self,
            "Select training-image folder",
            _picker_initial_path(seed_path),
            lambda folder: self._on_autodetect_folder_picked(folder, spin),
            folder=True,
            last_dir_key="training_images_autodetect",
        )

    def _on_autodetect_folder_picked(self, folder: str, spin: QDoubleSpinBox) -> None:
        if not folder:
            return
        try:
            from resolutionAdjust import autodetect_um_per_px_from_folder

            avg, n_with_meta, n_total = autodetect_um_per_px_from_folder(Path(folder))
        except Exception as exc:  # noqa: BLE001
            QMessageBox.critical(self, "Auto-detect failed", f"Could not scan folder:\n{exc}")
            return
        if avg is None:
            QMessageBox.warning(
                self,
                "No metadata found",
                f"None of the {n_total} TIFF file(s) in this folder had readable µm/px "
                "metadata (XResolution / OME-XML PhysicalSizeX). The field was left "
                "unchanged — enter a value manually.",
            )
            return
        spin.setValue(float(avg))
        QMessageBox.information(
            self,
            "Auto-detect complete",
            f"Set training µm/px to {avg:.4f} (averaged over {n_with_meta} of {n_total} "
            "TIFF file(s) with usable metadata).",
        )

    def get_landmark_target_um_per_px(self) -> float | None:
        v = float(self._lm_target_spin.value())
        return v if v > 0 else None

    def get_segmentation_target_um_per_px(self) -> float | None:
        v = float(self._seg_target_spin.value())
        return v if v > 0 else None

    def get_wing_isolation_target_um_per_px(self) -> float | None:
        v = float(self._wing_target_spin.value())
        return v if v > 0 else None

    def get_active_rescale_target(self) -> str:
        if self._ra_radio_landmark.isChecked():
            return "landmark"
        if self._ra_radio_wing.isChecked():
            return "wing_isolation"
        return "segmentation"

    def get_rescale_tolerance_low(self) -> float:
        return float(self._ra_tol_low_spin.value())

    def get_rescale_tolerance_high(self) -> float:
        return float(self._ra_tol_high_spin.value())

    def get_landmark_model_path(self) -> str:
        return self._lm_model_edit.text().strip()

    def get_segmentation_model_path(self) -> str:
        return self._seg_model_edit.text().strip()

    def _select_landmark_model_folder(self):
        from TRACE.gui import _open_native_picker_async

        _open_native_picker_async(
            self,
            "Select Model Folder (contains best_fold*.pt directly)",
            self._bundled_models_dir(),
            self._on_landmark_model_folder_picked,
            folder=True,
        )

    def _on_landmark_model_folder_picked(self, folder: str) -> None:
        if not folder:
            return
        from pathlib import Path as _P

        if not sorted(_P(folder).glob("best_fold*.pt")):
            QMessageBox.warning(
                self,
                "No fold checkpoints",
                f"No best_fold*.pt files in {folder}.\n\n"
                "Pick the model folder itself — the one that contains best_fold0.pt … "
                "best_fold4.pt directly (and, optionally, gate_config.yaml + training_chart.png).",
            )
            return
        self._lm_model_edit.setText(folder)

    def _select_segmentation_model_folder(self):
        from TRACE.gui import _open_native_picker_async

        _open_native_picker_async(
            self,
            "Select Segmentation Model Folder",
            self._bundled_models_dir(),
            self._on_segmentation_model_folder_picked,
            folder=True,
        )

    def _on_segmentation_model_folder_picked(self, folder: str) -> None:
        if folder:
            self._seg_model_edit.setText(folder)

    def _build_wing_graph_tab(self) -> QWidget:
        """All the steps that build the vein graph from the segmentation mask.

        Order of GroupBoxes follows the actual pipeline order:
        skeletonize → prune branches → merge collinear runs → 3 bridging
        passes to reconnect over gaps. Previously split across two tabs
        (Skeletonization & Pruning + Bridging) but they're really one
        concept — turning the binary vein mask into a clean topological
        graph — so merging them removes a needless navigation hop.
        """
        w = QWidget()
        layout = QVBoxLayout(w)

        gb = QGroupBox("Skeletonization")
        form = QFormLayout(gb)
        self._add_enum_list(form, "skeleton_methods", "Methods", SkeletonMethod, allowed_values={"ridge"})
        self._add_float(form, "smooth_sigma", "Smoothing sigma", 0.0, 100.0, 2, 0.1)
        layout.addWidget(gb)

        gb = QGroupBox("Pruning")
        form = QFormLayout(gb)
        self._add_bool(form, "enable_basic_prune", "Basic length-based prune (step 4)")
        self._add_bool(form, "enable_small_fragment_removal", "Small-fragment removal (steps 11 / 14)")
        self._add_float(
            form,
            "min_component_edge_fraction",
            "Orphan component cull (min fraction of total length)",
            0.0,
            1.0,
            3,
            0.01,
        )
        self._add_enum_list(
            form, "prune_methods", "Methods (empty = length-based only)", PruneMethod, allowed_values={"distance-map"}
        )
        self._add_opt_float(
            form, "prune_min_length_um", "Min branch length (µm)", 0.0, 50000.0, 2, 1.0, "auto (median vw)"
        )
        self._add_float(form, "prune_min_length_vein_widths", "Min length (× vein width)", 0.0, 100.0, 2, 0.1)
        self._add_float(form, "final_stub_vein_widths", "Final stub removal (× vein width)", 0.0, 100.0, 2, 0.1)
        self._add_float(form, "junction_merge_vein_widths", "Junction merge (× vein width)", 0.0, 100.0, 2, 0.1)
        self._add_float(form, "prune_radius_ratio_threshold", "Radius ratio threshold", 0.0, 1.0, 3, 0.01)
        self._add_float_list(form, "prune_scale_sigmas", "Multi-scale sigmas (comma-separated)")
        self._add_float(form, "prune_single_scale_sigma", "Single-scale sigma", 0.0, 100.0, 2, 0.1)
        layout.addWidget(gb)

        gb = QGroupBox("Collinear merging")
        form = QFormLayout(gb)
        self._add_float(form, "collinear_min_angle", "Min collinear angle (deg)", 0.0, 180.0, 1, 1.0)
        layout.addWidget(gb)

        gb = QGroupBox("Bridging — pass 1 (initial)")
        form = QFormLayout(gb)
        self._add_float(form, "bridge_max_gap_um", "Max gap (µm)", 0.0, 10000.0, 1, 10.0)
        self._add_float(form, "bridge_gap_fraction", "Gap fraction", 0.0, 1.0, 3, 0.01)
        self._add_float(form, "bridge_direction_window_um", "Direction window (µm)", 0.0, 10000.0, 1, 10.0)
        self._add_float(form, "bridge_min_combined_length_um", "Min combined length (µm)", 0.0, 10000.0, 1, 10.0)
        self._add_float(form, "bridge_on_axis_max_angle", "On-axis max angle (deg)", 0.0, 180.0, 1, 1.0)
        self._add_float(form, "bridge_on_axis_relaxed_cap", "On-axis relaxed cap (deg)", 0.0, 180.0, 1, 1.0)
        self._add_float(form, "bridge_min_facing_angle", "Min facing angle (deg)", 0.0, 180.0, 1, 1.0)
        self._add_float(
            form, "bridge_direction_max_edge_fraction", "Direction window max edge fraction", 0.0, 1.0, 3, 0.01
        )
        layout.addWidget(gb)

        gb = QGroupBox("Bridging — pass 2 (after cleanup)")
        form = QFormLayout(gb)
        self._add_float(form, "bridge2_max_gap_um", "Max gap (µm)", 0.0, 10000.0, 1, 10.0)
        self._add_float(form, "bridge2_gap_fraction", "Gap fraction", 0.0, 1.0, 3, 0.01)
        self._add_float(form, "bridge2_min_gap_vw", "Min gap floor (× vein width)", 0.0, 100.0, 2, 0.1)
        self._add_float(form, "bridge2_direction_window_um", "Direction window (µm)", 0.0, 10000.0, 1, 10.0)
        self._add_float(form, "bridge2_min_combined_length_um", "Min combined length (µm)", 0.0, 10000.0, 1, 10.0)
        self._add_opt_float(
            form, "bridge2_min_combined_length_vw", "Min combined (× vein width)", 0.0, 100.0, 2, 0.1, "use µm"
        )
        self._add_float(form, "bridge2_on_axis_max_angle", "On-axis max angle (deg)", 0.0, 180.0, 1, 1.0)
        self._add_float(form, "bridge2_on_axis_relaxed_cap", "On-axis relaxed cap (deg)", 0.0, 180.0, 1, 1.0)
        self._add_float(form, "bridge2_min_facing_angle", "Min facing angle (deg)", 0.0, 180.0, 1, 1.0)
        layout.addWidget(gb)

        gb = QGroupBox("Bridging — pass 3 (short-stub relaxed)")
        form = QFormLayout(gb)
        self._add_float(form, "bridge3_max_gap_vw", "Max gap (× vein width)", 0.0, 100.0, 2, 0.1)
        self._add_float(form, "bridge3_short_edge_vw", "Short edge threshold (× vein width)", 0.0, 100.0, 2, 0.1)
        self._add_float(form, "bridge3_relaxed_facing_angle", "Relaxed facing angle (deg)", 0.0, 180.0, 1, 1.0)
        self._add_float(form, "bridge3_direction_window_um", "Direction window (µm)", 0.0, 10000.0, 1, 10.0)
        self._add_float(form, "bridge3_on_axis_max_angle", "On-axis max angle (deg)", 0.0, 180.0, 1, 1.0)
        self._add_float(form, "bridge3_on_axis_relaxed_cap", "On-axis relaxed cap (deg)", 0.0, 180.0, 1, 1.0)
        layout.addWidget(gb)

        layout.addStretch()
        return w

    def _build_tracing_tab(self) -> QWidget:
        w = QWidget()
        layout = QVBoxLayout(w)

        gb = QGroupBox("Landmark anchoring")
        form = QFormLayout(gb)
        self._add_float(form, "snap_radius_um", "Snap radius (µm)", 0.0, 10000.0, 1, 10.0)
        self._add_float(form, "snap_radius_vw", "Snap radius fallback (× vein width)", 0.0, 100.0, 2, 0.1)
        layout.addWidget(gb)

        gb = QGroupBox("Vein tracing")
        form = QFormLayout(gb)
        self._add_float(form, "departure_sample_um", "Departure sample (µm)", 0.0, 10000.0, 1, 10.0)
        self._add_float(form, "departure_sample_vw", "Departure sample fallback (× vein width)", 0.0, 100.0, 2, 0.1)
        self._add_float(form, "tangent_continuity_max_angle", "Tangent continuity max angle (deg)", 0.0, 180.0, 1, 1.0)
        self._add_float(form, "merge_max_gap_um", "Segment merge max gap (µm)", 0.0, 10000.0, 1, 1.0)
        self._add_float(form, "distal_landmark_search_vw", "Distal landmark search (× vein width)", 0.0, 100.0, 2, 0.1)
        layout.addWidget(gb)

        gb = QGroupBox("Costa detection")
        form = QFormLayout(gb)
        self._add_float(form, "costa_min_in_band_fraction", "Min in-band fraction", 0.0, 1.0, 3, 0.01)
        self._add_float(
            form, "costa_propagation_max_distance_vw", "Propagation max distance (× vein width)", 0.0, 100.0, 2, 0.1
        )
        layout.addWidget(gb)

        gb = QGroupBox("Crossvein detection")
        form = QFormLayout(gb)
        self._add_float(form, "crossvein_min_angle", "Min angle (deg)", 0.0, 180.0, 1, 1.0)
        self._add_float(form, "crossvein_max_length_frac", "Max length fraction", 0.0, 1.0, 3, 0.01)
        self._add_float(form, "crossvein_min_length_vw", "Min length (× vein width)", 0.0, 100.0, 2, 0.1)
        self._add_float(form, "crossvein_max_length_vw", "Max length (× vein width)", 0.0, 100.0, 2, 0.1)
        self._add_bool(
            form,
            "synthesize_missing_crossveins",
            "Synthesize ACV/PCV from landmarks when not detected",
        )
        layout.addWidget(gb)

        gb = QGroupBox("Ectopic detection")
        form = QFormLayout(gb)
        self._add_float(form, "ectopic_min_length_um", "Min length (µm)", 0.0, 10000.0, 1, 1.0)
        self._add_float(form, "ectopic_min_length_vw", "Min length fallback (× vein width)", 0.0, 100.0, 2, 0.1)
        layout.addWidget(gb)

        gb = QGroupBox("Mutant phenotype reporting")
        form = QFormLayout(gb)
        self._add_bool(form, "assign_absent_partial_status", "Emit absent / partial vein statuses")
        self._add_float(
            form, "partial_endpoint_search_vw", "Endpoint reach tolerance (× vein width)", 0.0, 100.0, 2, 0.1
        )
        layout.addWidget(gb)

        layout.addStretch()
        return w

    def _build_intervein_tab(self) -> QWidget:
        w = QWidget()
        layout = QVBoxLayout(w)

        gb = QGroupBox("Region naming")
        form = QFormLayout(gb)
        self._add_float(form, "vein_buffer_vw", "Vein buffer (× vein width)", 0.0, 100.0, 2, 0.05)
        self._add_float(form, "adjacency_min_length_vw", "Min adjacency length (× vein width)", 0.0, 100.0, 2, 0.05)
        self._add_opt_int(form, "max_merge_size", "Max N-way merge size", 0, 100, "no cap")
        layout.addWidget(gb)

        gb = QGroupBox("Region splitter (watershed)")
        form = QFormLayout(gb)
        self._add_float(form, "intervein_split_h_vw", "h-maxima depth (× vein width)", 0.0, 100.0, 2, 0.1)
        self._add_float(form, "intervein_split_reseed_min_area_um2", "Reseed min area (µm²)", 0.0, 1e9, 1, 1000.0)
        self._add_float(form, "intervein_split_vein_barrier_vw", "Vein barrier (× vein width)", 0.0, 100.0, 2, 0.1)
        self._add_float(form, "intervein_split_wing_buffer_vw", "Wing outline inset (× vein width)", 0.0, 100.0, 2, 0.1)
        layout.addWidget(gb)

        layout.addStretch()
        return w

    def _build_quality_tab(self) -> QWidget:
        """Merged quality-gates editor: per-landmark confidence gates (rebuilt
        in place when the landmark model changes) + garbage-detector data-quality
        filters (solidity, fragmentation, vein-association, required veins).

        Both sections abort bad-data wings early with a reason logged. Landmark
        gates run first (Stage 1 preprocessing); the garbage-detector filters
        run during identifyFeatures (Stage 2)."""
        w = QWidget()
        layout = QVBoxLayout(w)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(6)

        # -- Landmark section (output flag + per-landmark confidence gates in one box) --
        gb = QGroupBox("Landmarks")
        gb_layout = QVBoxLayout(gb)
        self._include_unreliable_landmarks_chk = QCheckBox("Include low-confidence landmarks")
        self._include_unreliable_landmarks_chk.setToolTip(
            "When off (default), landmarks flagged low-confidence by LandmarkLocator are "
            "dropped from the output. When on, they are still emitted (marked reliable=false). "
            "Core-landmark failures abort the image regardless of this setting. "
            "Also enables soft-weighting in wingRotator: gate-failed landmarks contribute "
            "to the rotation fit at reduced weight instead of being dropped."
        )
        gb_layout.addWidget(self._include_unreliable_landmarks_chk)

        # Container for the per-landmark confidence-gate panel. Sits inside
        # the "Landmarks" groupbox so the visible border encircles both the
        # output-flag checkbox AND the gate panel. Populated now and
        # re-populated whenever the landmark model path changes, without
        # disturbing the checkbox above or the filter groups below.
        self._landmark_gate_container = QWidget()
        gate_container_layout = QVBoxLayout(self._landmark_gate_container)
        gate_container_layout.setContentsMargins(0, 0, 0, 0)
        gate_container_layout.setSpacing(6)
        gb_layout.addWidget(self._landmark_gate_container)
        layout.addWidget(gb)
        self._populate_landmark_gate_section()

        # -- Pipeline-level data-quality filters --
        gb = QGroupBox("Wing solidity (shape check)")
        form = QFormLayout(gb)
        self._add_bool(form, "solidity_filter_enabled", "Enable solidity filter")
        self._add_float(form, "solidity_min", "Min solidity", 0.0, 1.0, 4, 0.005)
        self._add_float(form, "solidity_max", "Max solidity", 0.0, 1.0, 4, 0.005)
        self._add_choice(form, "solidity_mode", "Threshold mode", ["fixed", "batch_mad"])
        self._add_float(form, "solidity_batch_k", "batch_mad: k (× robust σ)", 0.0, 100.0, 1, 0.5)
        self._add_int(form, "solidity_min_batch_size", "batch_mad: min batch size", 1, 100000)
        layout.addWidget(gb)

        gb = QGroupBox("Fragmentation (disconnected regions)")
        form = QFormLayout(gb)
        self._add_bool(form, "fragmentation_filter_enabled", "Enable fragmentation filter")
        self._add_float(
            form, "fragmentation_max_secondary_frac", "Max secondary-region fraction", 0.0, 1.0, 4, 0.005
        )
        layout.addWidget(gb)

        gb = QGroupBox("Vein association (unexplained vein tissue)")
        form = QFormLayout(gb)
        self._add_bool(form, "vein_association_filter_enabled", "Enable vein-association filter")
        self._add_float(form, "max_unassigned_vein_frac", "Max unassigned vein fraction", 0.0, 1.0, 3, 0.01)
        layout.addWidget(gb)

        gb = QGroupBox("Required veins (abort if missing)")
        form = QFormLayout(gb)
        from identify_features.models.topology import ALL_CANONICAL_VEINS

        self._add_checkbox_set(form, "required_veins", "Require", list(ALL_CANONICAL_VEINS))
        layout.addWidget(gb)

        layout.addStretch()
        return w

    # -----------------------------------------------------------------------
    # Widget factories
    # -----------------------------------------------------------------------
    def _apply_field_tooltip(self, form: QFormLayout, widget, name: str):
        """Apply the tooltip for field `name` (from _FIELD_TOOLTIPS) to widget + form-row label."""
        tooltip = _FIELD_TOOLTIPS.get(name)
        if not tooltip:
            return
        widget.setToolTip(tooltip)
        lbl = form.labelForField(widget)
        if lbl is not None:
            lbl.setToolTip(tooltip)

    def _add_float(
        self, form: QFormLayout, name: str, label: str, minv: float, maxv: float, decimals: int, step: float
    ):
        # Use _PlaceholderSpinBox so any caller can opt into a QLineEdit-native
        # placeholder via `.set_placeholder(...)`. Behaves identically to a plain
        # QDoubleSpinBox when no placeholder is configured.
        from TRACE.gui import _PlaceholderSpinBox

        spin = _PlaceholderSpinBox()
        spin.setRange(minv, maxv)
        spin.setDecimals(decimals)
        spin.setSingleStep(step)
        form.addRow(label, spin)
        self._apply_field_tooltip(form, spin, name)
        self._widgets[name] = (self._KIND_FLOAT, spin, None)

    def _add_int(self, form: QFormLayout, name: str, label: str, minv: int, maxv: int):
        spin = QSpinBox()
        spin.setRange(minv, maxv)
        form.addRow(label, spin)
        self._apply_field_tooltip(form, spin, name)
        self._widgets[name] = (self._KIND_INT, spin, None)

    def _add_opt_float(
        self,
        form: QFormLayout,
        name: str,
        label: str,
        minv: float,
        maxv: float,
        decimals: int,
        step: float,
        off_label: str,
    ):
        row = QWidget()
        hbox = QHBoxLayout(row)
        hbox.setContentsMargins(0, 0, 0, 0)
        check = QCheckBox()
        check.setToolTip(f"Uncheck to use default ({off_label})")
        spin = QDoubleSpinBox()
        spin.setRange(minv, maxv)
        spin.setDecimals(decimals)
        spin.setSingleStep(step)
        hbox.addWidget(check)
        hbox.addWidget(spin, stretch=1)
        check.toggled.connect(spin.setEnabled)
        form.addRow(label, row)
        tooltip = _FIELD_TOOLTIPS.get(name)
        if tooltip:
            row.setToolTip(tooltip)
            spin.setToolTip(tooltip)
            lbl = form.labelForField(row)
            if lbl is not None:
                lbl.setToolTip(tooltip)
        self._widgets[name] = (self._KIND_OPT_FLOAT, row, (check, spin))

    def _add_opt_int(self, form: QFormLayout, name: str, label: str, minv: int, maxv: int, off_label: str):
        row = QWidget()
        hbox = QHBoxLayout(row)
        hbox.setContentsMargins(0, 0, 0, 0)
        check = QCheckBox()
        check.setToolTip(f"Uncheck to use default ({off_label})")
        spin = QSpinBox()
        spin.setRange(minv, maxv)
        hbox.addWidget(check)
        hbox.addWidget(spin, stretch=1)
        check.toggled.connect(spin.setEnabled)
        form.addRow(label, row)
        tooltip = _FIELD_TOOLTIPS.get(name)
        if tooltip:
            row.setToolTip(tooltip)
            spin.setToolTip(tooltip)
            lbl = form.labelForField(row)
            if lbl is not None:
                lbl.setToolTip(tooltip)
        self._widgets[name] = (self._KIND_OPT_INT, row, (check, spin))

    def _add_enum_list(self, form: QFormLayout, name: str, label: str, enum_cls: type, allowed_values=None):
        lw = QListWidget()
        lw.setMaximumHeight(110)
        for member in enum_cls:
            item = QListWidgetItem(member.value)
            item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
            item.setCheckState(Qt.Unchecked)
            item.setData(Qt.UserRole, member.value)
            if allowed_values is not None and member.value not in allowed_values:
                item.setFlags(item.flags() & ~Qt.ItemIsEnabled)
            lw.addItem(item)
        form.addRow(label, lw)
        self._apply_field_tooltip(form, lw, name)
        self._widgets[name] = (self._KIND_ENUM_LIST, lw, enum_cls)

    def _add_float_list(self, form: QFormLayout, name: str, label: str):
        edit = QLineEdit()
        edit.setPlaceholderText("e.g. 2.0, 4.0, 8.0, 16.0")
        form.addRow(label, edit)
        self._apply_field_tooltip(form, edit, name)
        self._widgets[name] = (self._KIND_FLOAT_LIST, edit, None)

    def _add_bool(self, form: QFormLayout, name: str, label: str):
        check = QCheckBox()
        form.addRow(label, check)
        self._apply_field_tooltip(form, check, name)
        self._widgets[name] = (self._KIND_BOOL, check, None)

    def _add_choice(self, form: QFormLayout, name: str, label: str, choices: list[str]):
        """Bind a string-valued config field to a QComboBox over `choices`."""
        combo = QComboBox()
        combo.addItems(choices)
        form.addRow(label, combo)
        self._apply_field_tooltip(form, combo, name)
        self._widgets[name] = (self._KIND_CHOICE, combo, choices)

    def _add_checkbox_set(self, form: QFormLayout, name: str, label: str, options: list[str], columns: int = 4):
        """Bind a list-of-strings config field to a grid of checkboxes (one per option).

        get_config() returns the checked options as a list (preserving `options` order).
        """
        container = QWidget()
        grid = QGridLayout(container)
        grid.setContentsMargins(0, 0, 0, 0)
        boxes: dict[str, QCheckBox] = {}
        for i, opt in enumerate(options):
            cb = QCheckBox(opt)
            grid.addWidget(cb, i // columns, i % columns)
            boxes[opt] = cb
        form.addRow(label, container)
        self._apply_field_tooltip(form, container, name)
        self._widgets[name] = (self._KIND_STR_SET, container, (boxes, options))

    # -----------------------------------------------------------------------
    # Load / reset
    # -----------------------------------------------------------------------
    def _load_from_config(self, config: PipelineConfig):
        for name, (kind, widget, extra) in self._widgets.items():
            val = getattr(config, name)
            if kind == self._KIND_FLOAT:
                # Some PipelineConfig fields are float | None (e.g. um_per_px). The GUI
                # treats them as plain floats; fall back to the widget's current value
                # when the saved/imported config has None so we don't crash on float(None).
                if val is None:
                    continue
                widget.setValue(float(val))
            elif kind == self._KIND_INT:
                widget.setValue(int(val))
            elif kind == self._KIND_OPT_FLOAT:
                check, spin = extra
                if val is None:
                    check.setChecked(False)
                    spin.setEnabled(False)
                else:
                    check.setChecked(True)
                    spin.setEnabled(True)
                    spin.setValue(float(val))
            elif kind == self._KIND_OPT_INT:
                check, spin = extra
                if val is None:
                    check.setChecked(False)
                    spin.setEnabled(False)
                else:
                    check.setChecked(True)
                    spin.setEnabled(True)
                    spin.setValue(int(val))
            elif kind == self._KIND_ENUM_LIST:
                selected_values = {m.value for m in val}
                for i in range(widget.count()):
                    item = widget.item(i)
                    item.setCheckState(Qt.Checked if item.data(Qt.UserRole) in selected_values else Qt.Unchecked)
            elif kind == self._KIND_FLOAT_LIST:
                widget.setText(", ".join(f"{x:g}" for x in val))
            elif kind == self._KIND_BOOL:
                widget.setChecked(bool(val))
            elif kind == self._KIND_CHOICE:
                idx = widget.findText(str(val))
                if idx >= 0:
                    widget.setCurrentIndex(idx)
            elif kind == self._KIND_STR_SET:
                boxes, _options = extra
                selected = set(val or [])
                for opt, cb in boxes.items():
                    cb.setChecked(opt in selected)
        # Overlay color overrides (vein_colors / region_colors) are owned by
        # the main window's InlineGeneralPanel — the dialog no longer renders
        # color pickers, so nothing to restore here.

    def _reset_defaults(self):
        self._load_from_config(PipelineConfig())
        # Scale (um_per_px) lives in the main window's InlineGeneralPanel and
        # is preserved across this reset — only the advanced/dialog-owned
        # PipelineConfig fields are wiped.
        self._include_unreliable_landmarks_chk.setChecked(False)
        self._wing_expand_spin.setValue(0.05)
        # Reset model paths to the bundled defaults at TRACE/models/* so
        # Restore Defaults gives the user a runnable configuration on first
        # touch. Missing default folders fall back to "" (empty), which
        # forces the user to pick one manually.
        from TRACE.gui import _default_model_path

        self._lm_model_edit.setText(_default_model_path("landmark"))
        self._seg_model_edit.setText(_default_model_path("segmentation"))
        self._wing_model_edit.setText(_default_model_path("wing_isolation"))
        # Clear the persisted gate override BEFORE rebuilding the Landmarks
        # tab so the rebuilt panel reads the bundled model's YAML cleanly.
        # The force-clear flag also makes get_gate_override() return None on
        # OK, so the host's persisted override is wiped on close.
        self._initial_gate_override = None
        self._gate_override_force_clear = True
        # Rebuild the Landmarks tab so the GateConfigPanel reads the bundled
        # model's gate_config.yaml. Without this, the tab keeps showing the
        # prior model's state (or an error if it was missing a YAML).
        self._rebuild_landmark_gate_section()

    def _rebuild_landmark_gate_section(self) -> None:
        """Re-read gate_config.yaml from the currently-selected landmark folder
        and repopulate the landmark-gate section of the Quality tab in place.

        Called whenever the landmark model path changes (textChanged) and from
        _reset_defaults so the gate panel always reflects the model in the
        Models tab — including when Reset to Model Defaults is clicked, since
        the panel's _cfg comes from this rebuild. The rest of the Quality tab
        (filter groupboxes) is untouched so in-progress edits survive.
        """
        # Sync the dialog's stored landmark path with the line edit.
        self._calib_lm_path = self._lm_model_edit.text().strip()
        self._gate_panel = None
        self._populate_landmark_gate_section()

    def _apply_selected_preset(self):
        from dataclasses import fields as _dc_fields

        name = self._preset_combo.currentText()
        if not name or name not in self._presets:
            return
        preset = self._presets[name]
        # Presets are JSON dicts that may carry non-PipelineConfig keys
        # (e.g. `gate_override`, `gui_state`). Split so dataclass_replace
        # only sees PipelineConfig fields.
        valid_field_names = {f.name for f in _dc_fields(PipelineConfig)}
        config_overrides = {k: v for k, v in preset.items() if k in valid_field_names}
        new_config = dataclass_replace(self.get_config(), **config_overrides)
        self._load_from_config(new_config)
        # Replace the snapshot so get_config() preserves the preset's
        # um_per_px / vein_opacity / intervein_opacity / colors when OK is
        # clicked (those fields are no longer in the dispatch table).
        self._original_config = new_config
        # Apply the preset's gate_override (if any) and rebuild the Landmarks
        # tab so the gate panel reflects the new thresholds.
        gate_override = preset.get("gate_override")
        if gate_override is not None:
            self._initial_gate_override = gate_override
            self._rebuild_landmark_gate_section()
        # If the preset includes a saved gui_state block (new format), apply
        # it to the host window immediately so all the GUI-only flags (model
        # paths, intermediate outputs, workers, etc.) update too.
        gui_state = preset.get("gui_state")
        host = self.parent()
        if gui_state and host is not None and hasattr(host, "apply_gui_state"):
            host.apply_gui_state(gui_state)
            # Re-seed dialog widgets that mirror host state we just updated
            # (same handling as Import).
            if hasattr(host, "_wing_expand_fraction"):
                self._wing_expand_spin.setValue(float(host._wing_expand_fraction))
            if hasattr(host, "_wing_isolation_model_path"):
                self._wing_model_edit.setText(str(host._wing_isolation_model_path or ""))
            if hasattr(host, "_landmark_model_path"):
                self._lm_model_edit.setText(str(host._landmark_model_path or ""))
            if hasattr(host, "_segmentation_model_path"):
                self._seg_model_edit.setText(str(host._segmentation_model_path or ""))
            if hasattr(host, "_include_unreliable_landmarks"):
                self._include_unreliable_landmarks_chk.setChecked(bool(host._include_unreliable_landmarks))
