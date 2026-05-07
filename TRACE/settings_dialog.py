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

from pathlib import Path
from typing import Any

from identify_features.config import PIPELINE_PRESETS, PipelineConfig, apply_preset
from identify_features.models.datatypes import PruneMethod, SkeletonMethod
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from TRACE.pipeline import DEFAULT_MAX_WORKERS, INTERMEDIATE_OUTPUTS, OUTPUT_TYPES


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
        user_landmark_distances: list[dict] | None = None,
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
        self._build_ui()
        self._load_from_config(config)
        self._show_vein_tissue_chk.setChecked(show_vein_tissue)
        self._include_unreliable_landmarks_chk.setChecked(include_unreliable_landmarks)
        self._do_rotation_chk.setChecked(bool(do_rotation))
        self._workers_spin.setValue(int(workers))
        self._wing_expand_spin.setValue(float(wing_expand_fraction))
        self._wing_enable_chk.setChecked(bool(wing_isolation_enabled))
        if wing_isolation_model_path:
            self._wing_model_edit.setText(wing_isolation_model_path)
        # Sync enabled state of model-picker widgets to the checkbox.
        self._on_wing_isolation_toggled(bool(wing_isolation_enabled))
        # Apply current intermediate-output state to the checkboxes.
        if intermediate_outputs:
            for key, chk in self._intermediate_output_chks.items():
                chk.setChecked(bool(intermediate_outputs.get(key, True)))

    # -----------------------------------------------------------------------
    # Public API
    # -----------------------------------------------------------------------
    def get_config(self) -> PipelineConfig:
        """Build a new PipelineConfig from the current widget state."""
        kwargs: dict[str, Any] = {}
        for name, (kind, widget, extra) in self._widgets.items():
            if kind == self._KIND_FLOAT:
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
        return PipelineConfig(**kwargs)

    # -----------------------------------------------------------------------
    # UI construction
    # -----------------------------------------------------------------------
    def _build_ui(self):
        layout = QVBoxLayout(self)

        # Preset row — replaces pruning + bridging fields with a named snapshot.
        preset_row = QHBoxLayout()
        preset_row.addWidget(QLabel("Pruning / bridging preset:"))
        self._preset_combo = QComboBox()
        for preset_name in PIPELINE_PRESETS:
            self._preset_combo.addItem(preset_name)
        preset_row.addWidget(self._preset_combo, stretch=1)
        apply_btn = QPushButton("Apply preset")
        apply_btn.clicked.connect(self._apply_selected_preset)
        preset_row.addWidget(apply_btn)
        layout.addLayout(preset_row)

        tabs = QTabWidget()
        layout.addWidget(tabs, stretch=1)

        tabs.addTab(self._build_general_tab(), "General")
        tabs.addTab(self._build_custom_distances_tab(), "Custom Distances")
        tabs.addTab(self._build_landmarks_tab(), "Landmarks")
        tabs.addTab(self._build_skel_pruning_tab(), "Skeletonization && Pruning")
        tabs.addTab(self._build_bridging_tab(), "Bridging")
        tabs.addTab(self._build_tracing_tab(), "Tracing")
        tabs.addTab(self._build_intervein_tab(), "Intervein")

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
        self._add_opt_float(form, "um_per_px", "Microns per pixel", 0.0, 100.0, 4, 0.001, "use pixels")
        layout.addWidget(gb)

        gb = QGroupBox("Wing isolation (optional)")
        wig_layout = QVBoxLayout(gb)
        self._wing_enable_chk = QCheckBox("Enable wing isolation (Stage 0)")
        self._wing_enable_chk.setToolTip(
            "When enabled, every input image is masked through wingIsolator before "
            "LandmarkLocator sees it. Useful when images contain multiple wings."
        )
        self._wing_enable_chk.toggled.connect(self._on_wing_isolation_toggled)
        wig_layout.addWidget(self._wing_enable_chk)

        wig_layout.addWidget(QLabel("Wing identification model folder:"))
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
        layout.addWidget(gb)

        gb = QGroupBox("Wing rotation (optional)")
        rot_layout = QVBoxLayout(gb)
        self._do_rotation_chk = QCheckBox("Rotate to canonical orientation (Stage 1.5)")
        self._do_rotation_chk.setToolTip(
            "When checked, every image is rotated to a canonical right-side-up, "
            "distal-right orientation after landmark detection (rotation only — no mirror). "
            "Skipped automatically when fewer than 2 reliable landmarks are available."
        )
        rot_layout.addWidget(self._do_rotation_chk)
        layout.addWidget(gb)

        gb = QGroupBox("Vein detection")
        form = QFormLayout(gb)
        self._synthesize_missing_crossveins_mirror = QCheckBox("Synthesize ACV/PCV from landmarks when not detected")
        self._synthesize_missing_crossveins_mirror.setToolTip(
            "Mirror of the same option on the Tracing tab. Toggling here updates both."
        )
        form.addRow("", self._synthesize_missing_crossveins_mirror)
        layout.addWidget(gb)

        gb = QGroupBox("Output options")
        form = QFormLayout(gb)
        self._show_vein_tissue_chk = QCheckBox("Fill buffered vein tissue in overlay")
        self._show_vein_tissue_chk.setToolTip(
            "When off (default), the per-wing overlay only shows vein skeleton "
            "centerlines. When on, it also fills the buffered vein tissue polygons."
        )
        form.addRow("", self._show_vein_tissue_chk)
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
            chk.setChecked(True)
            self._intermediate_output_chks[key] = chk
            im_layout.addWidget(chk)
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
        form.addRow("Workers", self._workers_spin)
        gb_layout.addLayout(form)

        layout.addWidget(gb)

        layout.addStretch()
        return w

    def get_workers(self) -> int:
        return int(self._workers_spin.value())

    def get_wing_expand_fraction(self) -> float:
        return float(self._wing_expand_spin.value())

    def get_wing_isolation_enabled(self) -> bool:
        return bool(self._wing_enable_chk.isChecked())

    def get_wing_isolation_model_path(self) -> str:
        return self._wing_model_edit.text().strip()

    def get_intermediate_outputs(self) -> dict[str, bool]:
        return {key: chk.isChecked() for key, chk in self._intermediate_output_chks.items()}

    def get_user_landmark_distances(self) -> list[dict]:
        """Return the configured custom landmark-distance pairs (list of dicts)."""
        return list(self._user_landmark_distances)

    def _on_wing_isolation_toggled(self, checked: bool):
        self._wing_model_edit.setEnabled(checked)
        self._wing_model_browse.setEnabled(checked)
        self._wing_expand_spin.setEnabled(checked)

    def _select_wing_model_folder(self):
        from PyQt5.QtWidgets import QFileDialog

        folder = QFileDialog.getExistingDirectory(self, "Select Wing-Identification Model Folder")
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
            "user_distance_<label>_px (and _um when scale is set) columns to the batch CSV. "
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

        self._gate_panel = GateConfigPanel(gate_config, landmark_order, w)
        layout.addWidget(self._gate_panel)
        layout.addStretch(1)
        return w

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
    def _add_float(
        self, form: QFormLayout, name: str, label: str, minv: float, maxv: float, decimals: int, step: float
    ):
        spin = QDoubleSpinBox()
        spin.setRange(minv, maxv)
        spin.setDecimals(decimals)
        spin.setSingleStep(step)
        form.addRow(label, spin)
        self._widgets[name] = (self._KIND_FLOAT, spin, None)

    def _add_int(self, form: QFormLayout, name: str, label: str, minv: int, maxv: int):
        spin = QSpinBox()
        spin.setRange(minv, maxv)
        form.addRow(label, spin)
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
        self._widgets[name] = (self._KIND_ENUM_LIST, lw, enum_cls)

    def _add_float_list(self, form: QFormLayout, name: str, label: str):
        edit = QLineEdit()
        edit.setPlaceholderText("e.g. 2.0, 4.0, 8.0, 16.0")
        form.addRow(label, edit)
        self._widgets[name] = (self._KIND_FLOAT_LIST, edit, None)

    def _add_bool(self, form: QFormLayout, name: str, label: str):
        check = QCheckBox()
        form.addRow(label, check)
        self._widgets[name] = (self._KIND_BOOL, check, None)

    # -----------------------------------------------------------------------
    # Load / reset
    # -----------------------------------------------------------------------
    def _load_from_config(self, config: PipelineConfig):
        for name, (kind, widget, extra) in self._widgets.items():
            val = getattr(config, name)
            if kind == self._KIND_FLOAT:
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

    def _reset_defaults(self):
        self._load_from_config(PipelineConfig())
        self._show_vein_tissue_chk.setChecked(False)
        self._include_unreliable_landmarks_chk.setChecked(False)
        self._do_rotation_chk.setChecked(False)
        self._workers_spin.setValue(DEFAULT_MAX_WORKERS)
        self._wing_expand_spin.setValue(0.05)
        self._wing_enable_chk.setChecked(False)
        self._wing_model_edit.clear()
        self._on_wing_isolation_toggled(False)
        for chk in self._intermediate_output_chks.values():
            chk.setChecked(True)
        parent = self.parent()
        if parent is not None and hasattr(parent, "reset_workers_warning"):
            parent.reset_workers_warning()

    def _apply_selected_preset(self):
        name = self._preset_combo.currentText()
        if not name:
            return
        new_config = apply_preset(self.get_config(), name)
        self._load_from_config(new_config)
