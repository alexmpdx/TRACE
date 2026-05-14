"""Modal dialog for editing an identifyFeatures PipelineConfig.

All PipelineConfig fields are grouped into 5 tabs (General,
Skeletonization & Pruning, Bridging, Tracing, Intervein). The dialog
takes a config as input, works on a copy, and returns the updated
config on accept.

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
from PyQt5.QtCore import QEvent, Qt, QTimer
from PyQt5.QtGui import QColor
from PyQt5.QtWidgets import (
    QApplication,
    QButtonGroup,
    QCheckBox,
    QColorDialog,
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
    QSpinBox,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from TRACE.pipeline import DEFAULT_MAX_WORKERS, INTERMEDIATE_OUTPUTS, OUTPUT_TYPES
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
    """Shallow-merge for the gate-config dict shape used by GateConfigPanel."""
    import copy

    out = copy.deepcopy(base)
    for section in ("peak", "sharpness", "second_peak_ratio"):
        if section in override and "per_landmark" in override[section]:
            out.setdefault(section, {}).setdefault("per_landmark", {}).update(override[section]["per_landmark"])
        if section in override and "global" in override[section]:
            out.setdefault(section, {})["global"] = override[section]["global"]
    if "core_landmarks" in override:
        out["core_landmarks"] = list(override["core_landmarks"])
    if "second_peak_suppression_radius_px" in override:
        out["second_peak_suppression_radius_px"] = override["second_peak_suppression_radius_px"]
    return out


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

    def __init__(
        self,
        config: PipelineConfig,
        parent=None,
        show_vein_tissue: bool = False,
        include_unreliable_landmarks: bool = False,
        workers: int = DEFAULT_MAX_WORKERS,
        input_path: str = "",
        landmark_model_path: str = "",
        segmentation_model_path: str = "",
        gate_override: dict | None = None,
        wing_expand_fraction: float = 0.05,
        wing_isolation_enabled: bool = False,
        wing_isolation_model_path: str = "",
        intermediate_outputs: dict[str, bool] | None = None,
        do_rotation: bool = False,
        rotation_mirror_correct: bool = False,
        user_landmark_distances: list[dict] | None = None,
        distance_sample_image: str = "",
        distance_sample_landmarks: str = "",
        landmark_target_um_per_px: float | None = None,
        segmentation_target_um_per_px: float | None = None,
        wing_isolation_target_um_per_px: float | None = None,
        active_rescale_target: str = "segmentation",
        rescale_tolerance_low: float = 0.85,
        rescale_tolerance_high: float = 1.15,
    ):
        super().__init__(parent)
        self.setWindowTitle("Pipeline Settings")
        self.resize(720, 640)
        self._original_config = config
        self._calib_input_path = input_path
        self._calib_lm_path = landmark_model_path
        self._calib_seg_path = segmentation_model_path
        self._gate_panel = None  # populated by _build_landmarks_tab when a model is loaded
        self._initial_gate_override = gate_override
        # (kind, widget, extra) tuples indexed by PipelineConfig field name.
        self._widgets: dict[str, tuple[str, Any, Any]] = {}
        # User-defined landmark distance pairs (TRACE-only post-CSV augmentation).
        # List of {name_a, name_b, label} dicts so QSettings/JSON round-trips cleanly.
        self._user_landmark_distances: list[dict] = list(user_landmark_distances or [])
        # Last-used sample image + landmarks GeoJSON for the picker — pre-fills
        # the file pickers on the Custom Distances tab so the user doesn't have
        # to re-browse each session.
        self._distance_sample_image = distance_sample_image
        self._distance_sample_landmarks = distance_sample_landmarks
        # Stage -1 (resolutionAdjust) — per-model training-µm/px targets, which
        # model's target drives the global rescale, and the tolerance band. None
        # for any per-model target = "not configured; do not rescale on its
        # behalf". Captured here so `_build_models_tab` can seed its widgets.
        self._initial_landmark_target_um_per_px = landmark_target_um_per_px
        self._initial_segmentation_target_um_per_px = segmentation_target_um_per_px
        self._initial_wing_isolation_target_um_per_px = wing_isolation_target_um_per_px
        self._initial_active_rescale_target = active_rescale_target or "segmentation"
        self._initial_rescale_tolerance_low = rescale_tolerance_low
        self._initial_rescale_tolerance_high = rescale_tolerance_high
        # {disabled_child_chk: parent_chk_to_pulse}. Populated by build steps
        # (rotate→flip, wing-isolation→isolated-wing-image). The dialog-level
        # event filter pulses the parent when the user clicks a grayed child.
        self._pulse_dependencies: dict[QCheckBox, QCheckBox] = {}
        # {child_chk: hint_label}. Shown alongside the child when the user
        # clicks it while the parent is off, hidden again when the parent
        # becomes checked.
        self._dependency_hints: dict[QCheckBox, QLabel] = {}
        # User-editable overlay color overrides (vein_id / region_name -> [R,G,B]).
        # Initialised from topology defaults; overwritten by _load_from_config when
        # the loaded PipelineConfig has non-None vein_colors / region_colors. Only
        # entries that DIFFER from the topology defaults are written back to the
        # PipelineConfig (keeps presets/JSON round-trips minimal).
        from identify_features.models.topology import REGION_COLORS, VEIN_COLORS

        self._topology_vein_defaults: dict[str, list[int]] = {k: list(v) for k, v in VEIN_COLORS.items()}
        self._topology_region_defaults: dict[str, list[int]] = {k: list(v) for k, v in REGION_COLORS.items()}
        self._vein_color_state: dict[str, list[int]] = {k: list(v) for k, v in VEIN_COLORS.items()}
        self._region_color_state: dict[str, list[int]] = {k: list(v) for k, v in REGION_COLORS.items()}
        # Filled in by _build_general_tab when the color-picker rows are built.
        self._vein_color_btns: dict[str, QPushButton] = {}
        self._region_color_btns: dict[str, QPushButton] = {}
        self._build_ui()
        self._load_from_config(config)
        self._show_vein_tissue_chk.setChecked(show_vein_tissue)
        self._include_unreliable_landmarks_chk.setChecked(include_unreliable_landmarks)
        self._do_rotation_chk.setChecked(bool(do_rotation))
        self._rotation_mirror_correct_chk.setChecked(bool(rotation_mirror_correct))
        # Mirror-correct only meaningful when rotation is enabled. Auto-enable
        # rotation if the user (or a config import) checks flip while rotate
        # is off, so they can't end up requesting an output that won't run.
        self._rotation_mirror_correct_chk.setEnabled(self._do_rotation_chk.isChecked())
        self._do_rotation_chk.toggled.connect(self._rotation_mirror_correct_chk.setEnabled)
        self._rotation_mirror_correct_chk.toggled.connect(
            lambda checked: (
                self._do_rotation_chk.setChecked(True) if checked and not self._do_rotation_chk.isChecked() else None
            )
        )
        self._pulse_dependencies[self._rotation_mirror_correct_chk] = self._do_rotation_chk
        self._do_rotation_chk.toggled.connect(
            lambda checked: self._hide_hint(self._rotation_mirror_correct_chk) if checked else None
        )
        # Block signals so the initial-sync setValue doesn't re-trigger the
        # parallel-workers warning every time the Settings dialog opens.
        self._workers_spin.blockSignals(True)
        self._workers_spin.setValue(int(workers))
        self._workers_spin.blockSignals(False)
        self._wing_expand_spin.setValue(float(wing_expand_fraction))
        self._wing_enable_chk.setChecked(bool(wing_isolation_enabled))
        if wing_isolation_model_path:
            self._wing_model_edit.setText(wing_isolation_model_path)
        # Sync enabled state of model-picker widgets to the checkbox.
        self._on_wing_isolation_toggled(bool(wing_isolation_enabled))
        # Apply current intermediate-output state to the checkboxes.
        if intermediate_outputs:
            for key, chk in self._intermediate_output_chks.items():
                chk.setChecked(bool(intermediate_outputs.get(key, False)))
        # Catch mouse-press events on disabled dependent checkboxes so we can
        # pulse the parent. Disabled widgets don't receive events themselves
        # (and the event propagates only to the nearest enabled ancestor, not
        # the dialog), so the filter must sit on the application and route by
        # global position.
        app = QApplication.instance()
        if app is not None:
            app.installEventFilter(self)

    # -----------------------------------------------------------------------
    # Dependent-checkbox pulse
    # -----------------------------------------------------------------------
    def eventFilter(self, obj, event):
        if event.type() == QEvent.MouseButtonPress and self._pulse_dependencies:
            global_pos = event.globalPos()
            for child, parent_chk in self._pulse_dependencies.items():
                if not child.isEnabled() and child.isVisible():
                    local = child.mapFromGlobal(global_pos)
                    if child.rect().contains(local):
                        self._pulse_parent_text(parent_chk)
                        hint = self._dependency_hints.get(child)
                        if hint is not None:
                            hint.show()
                        break
        return super().eventFilter(obj, event)

    def closeEvent(self, event):
        app = QApplication.instance()
        if app is not None:
            app.removeEventFilter(self)
        super().closeEvent(event)

    def _pulse_parent_text(self, chk: QCheckBox) -> None:
        """Single brief text-color flash on a QCheckBox. Indicator untouched."""
        if chk.property("_pulse_active"):
            return
        saved = chk.styleSheet()
        chk.setProperty("_pulse_active", True)
        chk.setStyleSheet("QCheckBox { color: #4aa3ff; }")

        def _restore():
            chk.setStyleSheet(saved)
            chk.setProperty("_pulse_active", False)

        QTimer.singleShot(350, _restore)

    def _hide_hint(self, child: QCheckBox) -> None:
        hint = self._dependency_hints.get(child)
        if hint is not None:
            hint.hide()

    # -----------------------------------------------------------------------
    # Overlay color pickers
    # -----------------------------------------------------------------------
    @staticmethod
    def _swatch_style(rgb: list[int]) -> str:
        r, g, b = int(rgb[0]), int(rgb[1]), int(rgb[2])
        return f"background-color: rgb({r},{g},{b}); border: 1px solid #444;"

    # Display-name overrides for color picker labels. Internal keys (used as
    # dict keys in VEIN_COLORS / REGION_COLORS and as override-dict keys) stay
    # short; the GUI shows a friendlier label.
    _COLOR_LABEL_OVERRIDES = {
        "EV": "ectopic vein (EV)",
    }

    def _populate_color_grid(
        self,
        gb: QGroupBox,
        state: dict[str, list[int]],
        btn_map: dict[str, QPushButton],
        *,
        kind: str,
    ) -> None:
        """Fill ``gb`` with a 3-column grid of (swatch, label) pairs.

        Swatch sits to the left of its label. Both widgets are explicitly
        AlignVCenter so the fixed-height swatch and auto-sized QLabel line up
        in the same row regardless of column widths.
        Click a swatch → QColorDialog → write the new RGB triplet back into
        ``state`` and restyle the button.
        """
        grid = QGridLayout(gb)
        grid.setHorizontalSpacing(8)
        grid.setVerticalSpacing(6)
        cols = 3
        for idx, (key, rgb) in enumerate(state.items()):
            row, col_pair = divmod(idx, cols)
            col = col_pair * 2  # each pair occupies two columns (swatch + label)
            display = self._COLOR_LABEL_OVERRIDES.get(key, key)
            btn = QPushButton()
            btn.setFixedSize(48, 22)
            btn.setStyleSheet(self._swatch_style(rgb))
            btn.setCursor(Qt.PointingHandCursor)
            btn.setToolTip(f"Choose a color for {display}")
            btn.clicked.connect(lambda _checked=False, k=key, b=btn, kd=kind: self._on_color_swatch_clicked(k, b, kd))
            grid.addWidget(btn, row, col, Qt.AlignVCenter | Qt.AlignLeft)
            label = QLabel(display)
            label.setAlignment(Qt.AlignVCenter | Qt.AlignLeft)
            grid.addWidget(label, row, col + 1, Qt.AlignVCenter | Qt.AlignLeft)
            btn_map[key] = btn
        # Push (swatch, label) pairs to the left so trailing space absorbs
        # any extra width rather than stretching the label columns.
        grid.setColumnStretch(cols * 2, 1)

    def _on_color_swatch_clicked(self, key: str, btn: QPushButton, kind: str) -> None:
        state = self._vein_color_state if kind == "vein" else self._region_color_state
        current = state.get(key, [128, 128, 128])
        initial = QColor(int(current[0]), int(current[1]), int(current[2]))
        display = self._COLOR_LABEL_OVERRIDES.get(key, key)
        chosen = QColorDialog.getColor(initial, self, f"Pick color for {display}")
        if not chosen.isValid():
            return
        rgb = [chosen.red(), chosen.green(), chosen.blue()]
        state[key] = rgb
        btn.setStyleSheet(self._swatch_style(rgb))

    def _wrap_with_hint(self, chk: QCheckBox, hint_text: str) -> tuple[QWidget, QLabel]:
        """Pair a checkbox with a 'requires X' label without altering its size.

        Returns (row_widget, hint_label). The HBox uses zero margins and a
        trailing stretch so the checkbox keeps its natural sizeHint — the
        stretch absorbs any extra horizontal space.
        """
        row = QWidget()
        row_layout = QHBoxLayout(row)
        row_layout.setContentsMargins(0, 0, 0, 0)
        row_layout.setSpacing(8)
        row_layout.addWidget(chk)
        hint = QLabel(hint_text)
        hint.setStyleSheet("color: #4aa3ff;")
        hint.hide()
        row_layout.addWidget(hint)
        row_layout.addStretch(1)
        return row, hint

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
        # Overlay color overrides: only emit entries that differ from the
        # topology defaults so unchanged sessions round-trip with None (and
        # presets/JSON stay minimal).
        vein_diffs = {k: list(v) for k, v in self._vein_color_state.items() if v != self._topology_vein_defaults.get(k)}
        region_diffs = {
            k: list(v) for k, v in self._region_color_state.items() if v != self._topology_region_defaults.get(k)
        }
        kwargs["vein_colors"] = vein_diffs or None
        kwargs["region_colors"] = region_diffs or None
        return PipelineConfig(**kwargs)

    # -----------------------------------------------------------------------
    # UI construction
    # -----------------------------------------------------------------------
    def _build_ui(self):
        layout = QVBoxLayout(self)

        # Preset row — applies every field listed in the preset dict (any
        # PipelineConfig field, not just pruning/bridging). Presets are loaded
        # from TRACE/presets/*.json, so adding a new preset is just dropping a
        # JSON file in that folder — no code change needed.
        self._presets = load_presets()
        preset_row = QHBoxLayout()
        preset_row.addWidget(QLabel("Pipeline preset:"))
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

        self._tabs = QTabWidget()
        layout.addWidget(self._tabs, stretch=1)

        self._tabs.addTab(self._build_general_tab(), "General")
        self._tabs.addTab(self._build_custom_distances_tab(), "Custom Distances")
        self._tabs.addTab(self._build_landmarks_tab(), "Landmarks")
        self._tabs.addTab(self._build_models_tab(), "Models")
        self._tabs.addTab(self._build_skel_pruning_tab(), "Skeletonization && Pruning")
        self._tabs.addTab(self._build_bridging_tab(), "Bridging")
        self._tabs.addTab(self._build_tracing_tab(), "Tracing")
        self._tabs.addTab(self._build_intervein_tab(), "Intervein")

        # Keep the General-tab mirror of "synthesize missing crossveins" in sync
        # with the canonical checkbox on the Tracing tab.
        canonical = self._widgets["synthesize_missing_crossveins"][1]
        mirror = self._synthesize_missing_crossveins_mirror
        canonical.toggled.connect(lambda v: mirror.setChecked(v) if mirror.isChecked() != v else None)
        mirror.toggled.connect(lambda v: canonical.setChecked(v) if canonical.isChecked() != v else None)

        btns = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel | QDialogButtonBox.RestoreDefaults)
        btns.accepted.connect(self.accept)
        btns.rejected.connect(self.reject)
        btns.button(QDialogButtonBox.RestoreDefaults).clicked.connect(self._reset_defaults)
        layout.addWidget(btns)

    def _build_general_tab(self) -> QWidget:
        w = QWidget()
        layout = QVBoxLayout(w)

        gb = QGroupBox("Scale")
        form = QFormLayout(gb)
        self._add_float(form, "um_per_px", "Microns per pixel", 0.0001, 100.0, 4, 0.001)
        # Show the same "[conversion factor]" placeholder as the main-window scale
        # spinner when the user hasn't entered a value. The _PlaceholderSpinBox
        # subclass (from _add_float) renders this via QLineEdit's native
        # placeholder mechanism, so clicking the field clears it automatically.
        self._widgets["um_per_px"][1].set_placeholder("conversion factor")
        layout.addWidget(gb)

        gb = QGroupBox("Optional preprocessing steps")
        opt_layout = QVBoxLayout(gb)
        self._wing_enable_chk = QCheckBox("Wing isolation")
        self._wing_enable_chk.setToolTip(
            "Isolates the main (image-centered) wing and masks out everything else "
            "before the rest of the pipeline sees the image. Useful when the frame "
            "contains multiple wings or stray tissue around the wing of interest. "
            "Requires a wing-isolation model — pick it in the Models tab."
        )
        self._wing_enable_chk.toggled.connect(self._on_wing_isolation_toggled)
        opt_layout.addWidget(self._wing_enable_chk)
        self._do_rotation_chk = QCheckBox("Rotate wing")
        self._do_rotation_chk.setToolTip(
            "When checked, each wing is rotated so it sits right-side-up rather than at "
            "a skewed angle (rotation only — no mirroring or flipping). Runs as the LAST "
            "preprocessing step: every model inference (wing isolation, landmark "
            "detection, segmentation) still happens on the original un-rotated image, "
            "and the image + every produced GeoJSON (landmarks, wing, segmentation) are "
            "rotated together so identifyFeatures sees a self-consistent set. Skipped "
            "automatically when fewer than 2 reliable landmarks are available."
        )
        opt_layout.addWidget(self._do_rotation_chk)
        self._rotation_mirror_correct_chk = QCheckBox("Flip wing to canonical orientation")
        self._rotation_mirror_correct_chk.setToolTip(
            "When checked AND wingRotator detects a wing of opposite chirality from the "
            "canonical (right-wing) template, apply a vertical reflection on top of the "
            "rotation so the wing ends up distal-right AND anterior-up. Useful for visual "
            "consistency across mixed left+right wing batches, but flips biological "
            "chirality (a left wing is mirrored to look like a right wing). Default off: "
            "rotation only, opposite-chirality wings end up distal-left, anterior-up."
        )
        flip_row, flip_hint = self._wrap_with_hint(self._rotation_mirror_correct_chk, "requires Rotate wing")
        opt_layout.addWidget(flip_row)
        self._dependency_hints[self._rotation_mirror_correct_chk] = flip_hint
        layout.addWidget(gb)

        gb = QGroupBox("Crossvein detection")
        form = QFormLayout(gb)
        self._synthesize_missing_crossveins_mirror = QCheckBox("Synthesize ACV/PCV from landmarks when not detected")
        self._synthesize_missing_crossveins_mirror.setToolTip(_FIELD_TOOLTIPS["synthesize_missing_crossveins"])
        form.addRow("", self._synthesize_missing_crossveins_mirror)
        layout.addWidget(gb)

        gb = QGroupBox("Intermediate outputs")
        gb.setToolTip(
            "Upstream/intermediate artifacts written before final overlays + CSV. "
            "Toggle which ones to keep alongside the final outputs."
        )
        im_layout = QVBoxLayout(gb)
        self._intermediate_output_chks: dict[str, QCheckBox] = {}
        for key, label in OUTPUT_TYPES.items():
            if key not in INTERMEDIATE_OUTPUTS:
                continue
            chk = QCheckBox(label)
            chk.setChecked(False)
            tip = _INTERMEDIATE_TOOLTIPS.get(key)
            if tip:
                chk.setToolTip(tip)
            self._intermediate_output_chks[key] = chk
            if key == "wing_isolated_image":
                iso_row, iso_hint = self._wrap_with_hint(chk, "requires Wing isolation")
                im_layout.addWidget(iso_row)
                self._dependency_hints[chk] = iso_hint
            else:
                im_layout.addWidget(chk)
        # The "Isolated wing image" intermediate output is only meaningful when
        # the Wing isolation preprocessing step runs. Mirror the same UX as
        # the Rotate-wing → Flip-wing pair: the dependent checkbox grays out
        # when the parent is off, and toggling the dependent on auto-enables
        # the parent so the user doesn't end up with an output they can't get.
        iso_chk = self._intermediate_output_chks.get("wing_isolated_image")
        if iso_chk is not None:
            iso_chk.setEnabled(self._wing_enable_chk.isChecked())
            self._wing_enable_chk.toggled.connect(iso_chk.setEnabled)
            iso_chk.toggled.connect(
                lambda checked: (
                    self._wing_enable_chk.setChecked(True)
                    if checked and not self._wing_enable_chk.isChecked()
                    else None
                )
            )
            self._pulse_dependencies[iso_chk] = self._wing_enable_chk
            self._wing_enable_chk.toggled.connect(lambda checked: self._hide_hint(iso_chk) if checked else None)
        layout.addWidget(gb)

        gb = QGroupBox("Output options")
        out_layout = QVBoxLayout(gb)
        # The checkbox sits directly in the QVBox so it's left-aligned with
        # the rest of the dialog's checkboxes — adding it via QFormLayout
        # would indent it under the (empty) label column.
        self._show_vein_tissue_chk = QCheckBox("Fill buffered vein tissue in overlay")
        self._show_vein_tissue_chk.setToolTip(
            "When off (default), the per-wing overlay only shows vein skeleton "
            "centerlines. When on, it also fills the buffered vein tissue polygons."
        )
        out_layout.addWidget(self._show_vein_tissue_chk)
        form = QFormLayout()
        # Opacity controls for the vein and intervein layers. 0 = invisible,
        # 1 = fully opaque. Registered via the standard _add_float dispatch
        # path so to_config / _load_from_config round-trip them automatically.
        self._add_float(form, "vein_opacity", "Vein opacity", 0.0, 1.0, 2, 0.05)
        self._add_float(form, "intervein_opacity", "Intervein opacity", 0.0, 1.0, 2, 0.05)
        out_layout.addLayout(form)
        # Vein and intervein region color pickers. Layout: 3-column grid of
        # (label, swatch) pairs so the 10/8 rows stay compact in the dialog.
        vein_gb = QGroupBox("Vein colors")
        vein_gb.setToolTip("Click a swatch to choose a custom color for that vein in the overlay.")
        self._populate_color_grid(vein_gb, self._vein_color_state, self._vein_color_btns, kind="vein")
        out_layout.addWidget(vein_gb)
        region_gb = QGroupBox("Intervein region colors")
        region_gb.setToolTip("Click a swatch to choose a custom color for that intervein region in the overlay.")
        self._populate_color_grid(region_gb, self._region_color_state, self._region_color_btns, kind="region")
        out_layout.addWidget(region_gb)
        layout.addWidget(gb)

        gb = QGroupBox("Parallel processing")
        gb_layout = QVBoxLayout(gb)

        from TRACE.calibrate_widget import CalibrateWidget

        self._calibrate_widget = CalibrateWidget(self)
        self._calibrate_widget.set_paths(self._calib_input_path, self._calib_lm_path, self._calib_seg_path)
        self._calibrate_widget.applied.connect(lambda v: self._workers_spin.setValue(int(v)))
        gb_layout.addWidget(self._calibrate_widget)

        form = QFormLayout()
        self._workers_spin = QSpinBox()
        self._workers_spin.setRange(1, 32)
        self._workers_spin.setValue(DEFAULT_MAX_WORKERS)
        self._workers_spin.setToolTip(
            "Number of wings to process in parallel.\n"
            "Applies to both Stage 1 (hinge chop, segmentation) and Stage 2 (analysis).\n"
            "The landmark forward pass is GPU-batched once upfront."
        )
        self._workers_spin.valueChanged.connect(self._on_workers_changed)
        form.addRow("Workers", self._workers_spin)
        gb_layout.addLayout(form)

        layout.addWidget(gb)

        layout.addStretch()
        return w

    def get_workers(self) -> int:
        return int(self._workers_spin.value())

    def select_tab(self, name: str) -> None:
        """Switch the active tab by visible label. No-op when `name` doesn't match."""
        for i in range(self._tabs.count()):
            if self._tabs.tabText(i) == name:
                self._tabs.setCurrentIndex(i)
                return

    def _on_workers_changed(self, val: int) -> None:
        """Forward worker-count bumps to the parent window's shared warning logic."""
        parent = self.parent()
        if parent is not None and hasattr(parent, "maybe_show_workers_warning"):
            parent.maybe_show_workers_warning(val)

    def get_wing_expand_fraction(self) -> float:
        return float(self._wing_expand_spin.value())

    def get_wing_isolation_enabled(self) -> bool:
        return bool(self._wing_enable_chk.isChecked())

    def get_wing_isolation_model_path(self) -> str:
        return self._wing_model_edit.text().strip()

    def get_intermediate_outputs(self) -> dict[str, bool]:
        return {key: chk.isChecked() for key, chk in self._intermediate_output_chks.items()}

    def get_distance_sample_image(self) -> str:
        """Last-entered sample-image path in the Custom Distances picker."""
        if self._distance_picker is None:
            return self._distance_sample_image
        return self._distance_picker.image_path()

    def get_distance_sample_landmarks(self) -> str:
        """Last-entered landmarks-GeoJSON path in the Custom Distances picker."""
        if self._distance_picker is None:
            return self._distance_sample_landmarks
        return self._distance_picker.landmarks_path()

    def get_user_landmark_distances(self) -> list[dict]:
        """Return the configured custom landmark-distance pairs (list of dicts)."""
        return list(self._user_landmark_distances)

    def _on_wing_isolation_toggled(self, checked: bool):
        self._wing_model_edit.setEnabled(checked)
        self._wing_model_browse.setEnabled(checked)
        self._wing_expand_spin.setEnabled(checked)

    def _select_wing_model_folder(self):
        from PyQt5.QtWidgets import QFileDialog

        from TRACE.gui import _picker_initial_path

        folder = QFileDialog.getExistingDirectory(
            self,
            "Select Wing-Identification Model Folder",
            _picker_initial_path(self._wing_model_edit.text()),
        )
        if folder:
            self._wing_model_edit.setText(folder)

    # -----------------------------------------------------------------------
    # GUI-only flag accessors (not part of PipelineConfig)
    # -----------------------------------------------------------------------
    def get_show_vein_tissue(self) -> bool:
        return self._show_vein_tissue_chk.isChecked()

    def get_include_unreliable_landmarks(self) -> bool:
        return self._include_unreliable_landmarks_chk.isChecked()

    def get_do_rotation(self) -> bool:
        return self._do_rotation_chk.isChecked()

    def get_rotation_mirror_correct(self) -> bool:
        return self._rotation_mirror_correct_chk.isChecked()

    def get_gate_override(self) -> dict | None:
        """Confidence-gate override built from the Landmarks tab, or None if untouched."""
        if self._gate_panel is None:
            return self._initial_gate_override
        return self._gate_panel.result_override()

    def _build_custom_distances_tab(self) -> QWidget:
        """Tab embedding the napari-based landmark-distance picker.

        The whole tab body is a `LandmarkPickerWidget`: file pickers for a
        sample image + landmarks GeoJSON, an embedded napari canvas, and a
        side panel for managing the configured pairs.
        """
        w = QWidget()
        layout = QVBoxLayout(w)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(6)

        info = QLabel(
            "Configure straight-line distances between any two landmarks. Each pair adds "
            "custom_<label>_px (and _um when scale is set) columns to the batch CSV. "
            "Pairs are stored by landmark name and applied to every wing in the batch."
        )
        info.setWordWrap(True)
        info.setStyleSheet("color: #aaa;")
        layout.addWidget(info)

        try:
            from measurement_maker import LandmarkPair, LandmarkPickerWidget, pairs_from_dicts
        except ImportError as exc:
            err = QLabel(
                "measurement_maker is not importable. Install with:\n\n"
                "    pip install -e measurementMaker\n\n"
                f"Import error: {exc}"
            )
            err.setWordWrap(True)
            err.setStyleSheet("color: #f88; padding: 12px;")
            layout.addWidget(err)
            layout.addStretch()
            self._distance_picker = None
            return w

        initial: list[LandmarkPair] = pairs_from_dicts(self._user_landmark_distances)
        self._distance_picker = LandmarkPickerWidget(
            parent=w,
            initial_pairs=initial,
            default_image_dir=self._calib_input_path or "",
            initial_image_path=self._distance_sample_image,
            initial_landmarks_path=self._distance_sample_landmarks,
        )
        self._distance_picker.pairs_changed.connect(self._on_distance_pairs_changed)
        layout.addWidget(self._distance_picker, stretch=1)
        return w

    def _on_distance_pairs_changed(self, pairs):
        """Mirror the embedded picker's pair list onto the dialog's serialized state.

        Stored as plain dicts for QSettings/JSON round-trip compatibility.
        """
        from dataclasses import asdict

        self._user_landmark_distances = [asdict(p) for p in pairs]

    def _build_landmarks_tab(self) -> QWidget:
        """Confidence-gate editor tab. Requires a landmark model path; otherwise placeholder."""
        w = QWidget()
        layout = QVBoxLayout(w)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(6)

        # Output flag (always visible — independent of whether a gate-panel model is loaded).
        gb = QGroupBox("Output options")
        form = QFormLayout(gb)
        self._include_unreliable_landmarks_chk = QCheckBox("Include low-confidence landmarks")
        self._include_unreliable_landmarks_chk.setToolTip(
            "When off (default), landmarks flagged low-confidence by LandmarkLocator are "
            "dropped from the output. When on, they are still emitted (marked reliable=false). "
            "Core-landmark failures abort the image regardless of this setting. "
            "Also enables soft-weighting in wingRotator: gate-failed landmarks contribute "
            "to the rotation fit at reduced weight instead of being dropped."
        )
        form.addRow("", self._include_unreliable_landmarks_chk)
        layout.addWidget(gb)

        if not self._calib_lm_path:
            msg = QLabel(
                "Select a landmark model (.pt or fold folder) on the main window first, "
                "then reopen this dialog to edit per-landmark gate thresholds."
            )
            msg.setWordWrap(True)
            msg.setStyleSheet("color: #aaa; padding: 12px;")
            layout.addWidget(msg)
            layout.addStretch()
            return w

        try:
            from landmark_locator.scripts.gui import GateConfigPanel, read_gate_config_from_checkpoint

            gate_config, landmark_order = read_gate_config_from_checkpoint(Path(self._calib_lm_path))
        except Exception as exc:
            err = QLabel(f"Could not read gate config from {self._calib_lm_path}: {exc}")
            err.setWordWrap(True)
            err.setStyleSheet("color: #f88; padding: 12px;")
            layout.addWidget(err)
            layout.addStretch()
            return w

        # Merge any persisted GUI override on top so the panel shows the user's last edits.
        if self._initial_gate_override:
            gate_config = _merge_gate_override(gate_config, self._initial_gate_override)

        self._gate_panel = GateConfigPanel(gate_config, landmark_order, w, display_names=_LANDMARK_DISPLAY_NAMES)
        layout.addWidget(self._gate_panel)
        layout.addStretch(1)
        return w

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
            "Stage -1 (resolutionAdjust) rescales each input toward this value when "
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
        button consumed by Stage -1 (resolutionAdjust). The bottom group picks
        which model's target drives the actual rescale and sets the tolerance
        band that decides when a rescale is worth doing.
        """
        w = QWidget()
        layout = QVBoxLayout(w)

        # -- Landmark model --
        gb = QGroupBox("Landmark points")
        lm_layout = QVBoxLayout(gb)
        lm_layout.addWidget(QLabel("Checkpoint (.pt) or fold folder (best_fold*.pt):"))
        lm_row = QHBoxLayout()
        self._lm_model_edit = QLineEdit()
        self._lm_model_edit.setReadOnly(True)
        self._lm_model_edit.setPlaceholderText("Select .pt checkpoint or fold folder...")
        self._lm_model_edit.setToolTip(
            "Pick a single .pt checkpoint for fast single-fold inference, "
            "or pick a folder containing best_fold*.pt for 5-fold ensemble "
            "(~5× slower, more robust)."
        )
        btn_lm_file = QPushButton("File...")
        btn_lm_file.setToolTip("Pick a single .pt checkpoint for fast single-fold landmark inference.")
        btn_lm_file.clicked.connect(self._select_landmark_model_file)
        btn_lm_folder = QPushButton("Folder...")
        btn_lm_folder.setToolTip("Pick a folder of best_fold*.pt checkpoints (5-fold ensemble).")
        btn_lm_folder.clicked.connect(self._select_landmark_model_folder)
        lm_row.addWidget(self._lm_model_edit, stretch=1)
        lm_row.addWidget(btn_lm_file)
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

        # -- Wing isolation model (Stage 0, optional) --
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
            "Stage 0 mask buffer, as a fraction of sqrt(wing area). "
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

        # -- Resolution adjustment (Stage -1) --
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
        self._ra_tol_low_label.setStyleSheet("color: #888;")
        low_row.addWidget(self._ra_tol_low_label, stretch=1)
        low_container = QWidget()
        low_container.setLayout(low_row)
        high_row = QHBoxLayout()
        high_row.addWidget(self._ra_tol_high_spin)
        self._ra_tol_high_label = QLabel("")
        self._ra_tol_high_label.setStyleSheet("color: #888;")
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
    # Stage -1 (resolutionAdjust) helpers
    # -----------------------------------------------------------------------
    def _autodetect_target_um_per_px(self, spin: QDoubleSpinBox, get_model_path_fn) -> None:
        """Open a folder picker, run autodetect, write the average into `spin`."""
        from PyQt5.QtWidgets import QFileDialog

        from TRACE.gui import _picker_initial_path

        seed_path = ""
        try:
            seed_path = get_model_path_fn() or ""
        except Exception:
            seed_path = ""
        folder = QFileDialog.getExistingDirectory(
            self,
            "Select training-image folder",
            _picker_initial_path(seed_path),
        )
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

    def _select_landmark_model_file(self):
        from PyQt5.QtWidgets import QFileDialog

        from TRACE.gui import _picker_initial_path

        path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Landmark Model Checkpoint",
            _picker_initial_path(self._lm_model_edit.text()),
            "PyTorch Checkpoint (*.pt);;All Files (*)",
        )
        if path:
            self._lm_model_edit.setText(path)

    def _select_landmark_model_folder(self):
        from pathlib import Path as _P

        from PyQt5.QtWidgets import QFileDialog

        from TRACE.gui import _picker_initial_path

        folder = QFileDialog.getExistingDirectory(
            self,
            "Select Fold Checkpoint Folder (contains best_fold*.pt)",
            _picker_initial_path(self._lm_model_edit.text()),
        )
        if folder:
            if not sorted(_P(folder).glob("best_fold*.pt")):
                QMessageBox.warning(
                    self,
                    "No fold checkpoints",
                    f"No best_fold*.pt files in {folder}. Pick a folder containing 5-fold CV checkpoints.",
                )
                return
            self._lm_model_edit.setText(folder)

    def _select_segmentation_model_folder(self):
        from PyQt5.QtWidgets import QFileDialog

        from TRACE.gui import _picker_initial_path

        folder = QFileDialog.getExistingDirectory(
            self,
            "Select Segmentation Model Folder",
            _picker_initial_path(self._seg_model_edit.text()),
        )
        if folder:
            self._seg_model_edit.setText(folder)

    def _build_skel_pruning_tab(self) -> QWidget:
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

        layout.addStretch()
        return w

    def _build_bridging_tab(self) -> QWidget:
        w = QWidget()
        layout = QVBoxLayout(w)

        gb = QGroupBox("Pass 1 — initial")
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

        gb = QGroupBox("Pass 2 — after cleanup")
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

        gb = QGroupBox("Pass 3 — short-stub relaxed")
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
        # Overlay color overrides: start from topology defaults, layer any
        # overrides on top, restyle the swatch buttons.
        for key, default_rgb in self._topology_vein_defaults.items():
            override = (config.vein_colors or {}).get(key)
            rgb = list(override) if override is not None else list(default_rgb)
            self._vein_color_state[key] = rgb
            btn = self._vein_color_btns.get(key)
            if btn is not None:
                btn.setStyleSheet(self._swatch_style(rgb))
        for key, default_rgb in self._topology_region_defaults.items():
            override = (config.region_colors or {}).get(key)
            rgb = list(override) if override is not None else list(default_rgb)
            self._region_color_state[key] = rgb
            btn = self._region_color_btns.get(key)
            if btn is not None:
                btn.setStyleSheet(self._swatch_style(rgb))

    def _reset_defaults(self):
        self._load_from_config(PipelineConfig())
        # Scale: PipelineConfig.um_per_px defaults to None; _load_from_config skips
        # None floats, so explicitly snap the spinbox to its minimum (which shows
        # the "[conversion factor]" placeholder).
        um_spin = self._widgets["um_per_px"][1]
        um_spin.setValue(um_spin.minimum())
        self._show_vein_tissue_chk.setChecked(False)
        self._include_unreliable_landmarks_chk.setChecked(False)
        self._do_rotation_chk.setChecked(False)
        self._rotation_mirror_correct_chk.setChecked(False)
        self._workers_spin.setValue(DEFAULT_MAX_WORKERS)
        self._wing_expand_spin.setValue(0.05)
        self._wing_enable_chk.setChecked(False)
        # Model paths (Landmark, Segmentation, Wing isolation) are intentionally
        # preserved across Restore Defaults — the user has to "wipe my memories"
        # on the main window to fall back to the bundled TRACE/models/* defaults.
        self._on_wing_isolation_toggled(False)
        for chk in self._intermediate_output_chks.values():
            chk.setChecked(False)
        parent = self.parent()
        if parent is not None and hasattr(parent, "reset_workers_warning"):
            parent.reset_workers_warning()

    def _apply_selected_preset(self):
        name = self._preset_combo.currentText()
        if not name or name not in self._presets:
            return
        # Apply the preset's overrides on top of the user's current widget
        # state — fields not listed in the preset are preserved.
        new_config = dataclass_replace(self.get_config(), **self._presets[name])
        self._load_from_config(new_config)
