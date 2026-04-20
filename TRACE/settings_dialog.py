"""Modal dialog for editing an identifyFeatures PipelineConfig.

All PipelineConfig fields are grouped into 5 tabs (Scale & Skeleton,
Pruning, Bridging, Tracing, Intervein). The dialog takes a config as
input, works on a copy, and returns the updated config on accept.

The dialog is dispatch-based: each field is registered via a small helper
(`_add_float`, `_add_opt_float`, `_add_int`, `_add_opt_int`,
`_add_enum_list`, `_add_float_list`) and the dispatch table records both
the widget kind and the widget reference. Read/write go through the
dispatch table so the accept/reset/load paths don't have to know
individual field types.
"""

from __future__ import annotations

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
    QPushButton,
    QSpinBox,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)


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

    def __init__(self, config: PipelineConfig, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Pipeline Settings")
        self.resize(720, 640)
        self._original_config = config
        # (kind, widget, extra) tuples indexed by PipelineConfig field name.
        self._widgets: dict[str, tuple[str, Any, Any]] = {}
        self._build_ui()
        self._load_from_config(config)

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

        tabs.addTab(self._build_scale_skel_tab(), "Scale && Skeleton")
        tabs.addTab(self._build_pruning_tab(), "Pruning")
        tabs.addTab(self._build_bridging_tab(), "Bridging")
        tabs.addTab(self._build_tracing_tab(), "Tracing")
        tabs.addTab(self._build_intervein_tab(), "Intervein")

        btns = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel | QDialogButtonBox.RestoreDefaults)
        btns.accepted.connect(self.accept)
        btns.rejected.connect(self.reject)
        btns.button(QDialogButtonBox.RestoreDefaults).clicked.connect(self._reset_defaults)
        layout.addWidget(btns)

    def _build_scale_skel_tab(self) -> QWidget:
        w = QWidget()
        layout = QVBoxLayout(w)

        gb = QGroupBox("Scale")
        form = QFormLayout(gb)
        self._add_opt_float(form, "um_per_px", "Microns per pixel", 0.0, 100.0, 4, 0.001, "use pixels")
        layout.addWidget(gb)

        gb = QGroupBox("Skeletonization")
        form = QFormLayout(gb)
        self._add_enum_list(form, "skeleton_methods", "Methods", SkeletonMethod, allowed_values={"ridge"})
        self._add_float(form, "smooth_sigma", "Smoothing sigma", 0.0, 100.0, 2, 0.1)
        layout.addWidget(gb)

        layout.addStretch()
        return w

    def _build_pruning_tab(self) -> QWidget:
        w = QWidget()
        layout = QVBoxLayout(w)

        gb = QGroupBox("Pruning")
        form = QFormLayout(gb)
        self._add_bool(form, "enable_basic_prune", "Basic length-based prune (step 4)")
        self._add_bool(form, "enable_small_fragment_removal", "Small-fragment removal (steps 11 / 14)")
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

    def _apply_selected_preset(self):
        name = self._preset_combo.currentText()
        if not name:
            return
        new_config = apply_preset(self.get_config(), name)
        self._load_from_config(new_config)
