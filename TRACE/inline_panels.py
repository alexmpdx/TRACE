"""Inline tab panels for the TRACE main window's right-side QTabWidget.

The right panel hosts:
  - Main (image list + log) — built inline in gui.py
  - General — InlineGeneralPanel (this module)
  - Custom Distances — InlineCustomDistancesPanel (this module)
  - Help — InlineHelpPanel (this module)

These three panels used to live as tabs inside the modal PipelineConfigDialog
(General + Custom Distances) and were not exposed at all (Help). Moving them
into the main window means every edit auto-applies straight to TraceWindow
state — there is no OK/Cancel/Apply.

The three classes expose a `refresh_from_state()` method called by the host
window after operations that change TraceWindow state from outside the panel
(settings dialog OK, "wipe my memories", config import). Each call blocks
widget signals during the refresh so the panel does not echo state back to
itself.
"""

from __future__ import annotations

import sys
from dataclasses import asdict
from pathlib import Path
from typing import TYPE_CHECKING, Optional

from PyQt5.QtCore import QEvent, Qt, QTimer, QUrl
from PyQt5.QtGui import QColor, QFont
from PyQt5.QtWidgets import (
    QApplication,
    QCheckBox,
    QColorDialog,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QProgressDialog,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from TRACE.pipeline import DEFAULT_MAX_WORKERS, INTERMEDIATE_OUTPUTS, OUTPUT_TYPES

if TYPE_CHECKING:
    from TRACE.gui import TraceWindow


# Tooltips mirrored from settings_dialog._INTERMEDIATE_TOOLTIPS — kept locally
# to avoid importing from the dialog module just for a dict. The General tab
# is the only consumer of these strings.
_INTERMEDIATE_TOOLTIPS: dict[str, str] = {
    "rescaled_image": (
        "Stage 1 (resolutionAdjust) writes the rescaled-input TIFF when the "
        "input resolution falls outside the tolerance band."
    ),
    "wing_isolated_image": (
        "Stage 2 (wing isolation) writes the isolated-wing image after the "
        "wing-isolation model has masked everything outside the main wing."
    ),
    "wing_isolated_geojson": (
        "Stage 2 writes the wing-outline GeoJSON used to crop the input " "image to just the main wing."
    ),
    "chopped_image": (
        "Stage 2 (hinge chop) writes the hinge-removed image — the input " "to landmark detection and segmentation."
    ),
    "landmarks_image": "Stage 2 writes a per-image landmark-overlay PNG.",
    "landmarks_geojson": "Stage 2 writes per-image landmark predictions as GeoJSON points.",
    "segmentation_geojson": ("Stage 2 writes the vein/intervein semantic-segmentation polygons as GeoJSON."),
    "intervein_geojson": "Stage 3 (identifyFeatures) writes the named intervein-region polygons as GeoJSON.",
}


# Display-name overrides for color picker labels. Internal keys (used as dict
# keys in VEIN_COLORS / REGION_COLORS and as override-dict keys) stay short;
# the GUI shows a friendlier label.
_COLOR_LABEL_OVERRIDES = {
    "EV": "ectopic vein (EV)",
}


def _swatch_style(rgb: list[int]) -> str:
    r, g, b = int(rgb[0]), int(rgb[1]), int(rgb[2])
    return f"background-color: rgb({r},{g},{b}); border: 1px solid #444;"


def _pulse_text(chk: QCheckBox) -> None:
    """Briefly flash the checkbox label in the accent color and restore.

    Used as feedback when the user clicks a disabled dependent checkbox so the
    parent ("you need to enable THIS first") visibly draws attention.
    """
    if chk.property("_pulse_active"):
        return
    saved = chk.styleSheet()
    chk.setProperty("_pulse_active", True)
    chk.setStyleSheet("QCheckBox { color: #4aa3ff; }")

    def _restore() -> None:
        chk.setStyleSheet(saved)
        chk.setProperty("_pulse_active", False)

    QTimer.singleShot(350, _restore)


class _DependentRow(QWidget):
    """Wrapper widget for a checkbox whose enabled state is gated by another.

    Provides storage for the child / parent / hint references so the panel's
    app-level event filter can reach them. Mouse handling itself lives on
    the panel because disabled QCheckBox absorbs press events silently and
    does not propagate to ancestor widgets.
    """

    def __init__(self, child: QCheckBox, parent_chk: QCheckBox):
        super().__init__()
        self._child = child
        self._parent_chk = parent_chk
        self._hint: Optional[QLabel] = None

    def set_hint(self, hint: QLabel) -> None:
        self._hint = hint


def _wrap_scrollable(content: QWidget) -> QScrollArea:
    """Wrap a panel in a QScrollArea so it can shrink below its natural size."""
    area = QScrollArea()
    area.setWidgetResizable(True)
    area.setFrameShape(QScrollArea.NoFrame)
    area.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
    area.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
    area.setWidget(content)
    return area


# ---------------------------------------------------------------------------
# InlineGeneralPanel
# ---------------------------------------------------------------------------


class InlineGeneralPanel(QWidget):
    """General settings inline panel — auto-applies edits to TraceWindow state.

    Built from the same source layout that previously lived in
    `PipelineConfigDialog._build_general_tab`. Differences:
      - No OK/Cancel — every signal writes straight to `window.config` /
        `window._*` attributes.
      - Wing-isolation enable affects only `window._wing_isolation_enabled`;
        the wing-model path widgets live on the Models tab in the dialog
        and stay enabled there regardless of this checkbox.
      - The synthesize-crossveins checkbox is the canonical control for
        `window.config.synthesize_missing_crossveins` whenever the dialog
        is closed; when the dialog is open, the Tracing-tab checkbox is
        authoritative and we re-sync on dialog close via refresh_from_state().
    """

    def __init__(self, window: "TraceWindow"):
        super().__init__()
        self._window = window
        # {disabled_child_chk: (parent_chk_to_pulse, hint_label_to_show)} —
        # populated as the panel builds dependent rows. The app-level event
        # filter (see eventFilter below) watches every MouseButtonPress and
        # routes by global position, since disabled QCheckBox absorbs clicks
        # silently and does not propagate to its parent widget.
        self._pulse_dependencies: dict[QCheckBox, tuple[QCheckBox, Optional[QLabel]]] = {}
        # Color picker state — initialized from topology defaults, layered with
        # any overrides from window.config.vein_colors / region_colors.
        from identify_features.models.topology import REGION_COLORS, VEIN_COLORS

        self._topology_vein_defaults: dict[str, list[int]] = {k: list(v) for k, v in VEIN_COLORS.items()}
        self._topology_region_defaults: dict[str, list[int]] = {k: list(v) for k, v in REGION_COLORS.items()}
        self._vein_color_state: dict[str, list[int]] = {k: list(v) for k, v in VEIN_COLORS.items()}
        self._region_color_state: dict[str, list[int]] = {k: list(v) for k, v in REGION_COLORS.items()}
        self._vein_color_btns: dict[str, QPushButton] = {}
        self._region_color_btns: dict[str, QPushButton] = {}
        # Snapshot of the opacity + color defaults as they are at GUI launch —
        # those are the values "Restore Defaults" reverts to. Captured before
        # _restore_settings() may overlay persisted values from a prior session,
        # so the snapshot always reflects the factory starting state regardless
        # of what's been saved.
        self._default_vein_opacity: float = float(self._window.config.vein_opacity)
        self._default_intervein_opacity: float = float(self._window.config.intervein_opacity)
        self._default_vein_colors: dict[str, list[int]] = {k: list(v) for k, v in self._topology_vein_defaults.items()}
        self._default_region_colors: dict[str, list[int]] = {
            k: list(v) for k, v in self._topology_region_defaults.items()
        }
        # Reference-distance state for the Estimate button (lazy popup).
        self._scale_estimator_available = True
        try:
            from scale_estimator import DEFAULT_REFERENCE_DISTANCE_UM

            self._scale_ref_default = float(DEFAULT_REFERENCE_DISTANCE_UM)
        except ImportError:
            self._scale_ref_default = 2200.0
            self._scale_estimator_available = False
        self._scale_ref_dialog: QDialog | None = None
        self._build_ui()
        self.refresh_from_state()
        # App-level event filter so clicks on disabled dependent checkboxes
        # trigger pulse + hint. Required because disabled QCheckBox absorbs
        # MouseButtonPress silently; the filter routes by global position.
        app = QApplication.instance()
        if app is not None:
            app.installEventFilter(self)

    # -----------------------------------------------------------------------
    # App-level event filter for dependent-checkbox pulse
    # -----------------------------------------------------------------------
    def eventFilter(self, obj, event):  # noqa: N802 — Qt API
        if event.type() == QEvent.MouseButtonPress and self._pulse_dependencies:
            global_pos = event.globalPos()
            for child, (parent_chk, hint) in self._pulse_dependencies.items():
                if not child.isEnabled() and child.isVisible():
                    local = child.mapFromGlobal(global_pos)
                    if child.rect().contains(local):
                        _pulse_text(parent_chk)
                        if hint is not None:
                            hint.show()
                        break
        return super().eventFilter(obj, event)

    # -----------------------------------------------------------------------
    # UI construction
    # -----------------------------------------------------------------------
    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)

        self._build_preset_row(layout)
        self._build_scale_group(layout)
        self._build_optional_preprocessing_group(layout)
        self._build_crossvein_group(layout)
        self._build_intermediate_outputs_group(layout)
        self._build_output_options_group(layout)
        self._build_parallel_processing_group(layout)
        layout.addStretch(1)

        # Bottom row: Restore Defaults + Advanced Settings. Restore Defaults
        # wipes only what this Settings tab owns (scale, opacities, colors,
        # preprocessing toggles, intermediates, workers, synth-crossveins) —
        # leaves model paths, input/output folders, and advanced PipelineConfig
        # fields alone. Advanced Settings opens the modal PipelineConfigDialog.
        adv_row = QHBoxLayout()
        adv_row.addStretch(1)
        self.btn_restore_defaults = QPushButton("Restore Defaults")
        self.btn_restore_defaults.setToolTip(
            "Reset every control on this Settings tab to its factory default: scale, "
            "preprocessing toggles, intermediate outputs, overlay opacities and colors, "
            "and worker count. Does not touch model paths, input/output folders, or "
            "advanced PipelineConfig fields (those have their own reset)."
        )
        self.btn_restore_defaults.clicked.connect(self.restore_defaults)
        adv_row.addWidget(self.btn_restore_defaults)
        self.btn_advanced = QPushButton("Advanced Settings…")
        self.btn_advanced.setToolTip(
            "Open the advanced pipeline-settings dialog: per-model gate thresholds, "
            "skeletonization, bridging, tracing, intervein region detection."
        )
        self.btn_advanced.clicked.connect(self._window._open_settings_dialog)
        adv_row.addWidget(self.btn_advanced)
        self.btn_wipe_memories = QPushButton("wipe my memories")
        self.btn_wipe_memories.setToolTip(
            "Clear every persisted setting — input/output folders, model paths, scale, "
            "pipeline config, custom distance pairs, workers warning suppression — and "
            "snap every widget back to the state a first-time user would see."
        )
        self.btn_wipe_memories.clicked.connect(self._window._reset_gui_to_defaults)
        adv_row.addWidget(self.btn_wipe_memories)
        adv_row.addStretch(1)
        layout.addLayout(adv_row)

    def _build_preset_row(self, parent_layout: QVBoxLayout) -> None:
        """Preset selector at the top of the Settings tab.

        Mirrors the Settings-preset row inside the Advanced Settings dialog.
        Applying a preset here overwrites every field the preset lists
        (PipelineConfig + gate_override + gui_state when present); fields
        the preset omits are preserved.
        """
        from TRACE.presets_loader import load_presets

        self._presets = load_presets()
        row = QHBoxLayout()
        row.addWidget(QLabel("Settings preset:"))
        self._preset_combo = QComboBox()
        self._preset_combo.setToolTip(
            "Named bundles of settings stored as JSON in TRACE/presets/. "
            "Pick one and click Apply preset to overwrite the listed fields."
        )
        for preset_name in self._presets:
            self._preset_combo.addItem(preset_name)
        row.addWidget(self._preset_combo, stretch=1)
        apply_btn = QPushButton("Apply preset")
        apply_btn.setToolTip("Overwrite all fields listed in the selected preset. Fields not in the preset are kept.")
        apply_btn.clicked.connect(self._apply_selected_preset)
        row.addWidget(apply_btn)
        parent_layout.addLayout(row)

    def _apply_selected_preset(self) -> None:
        from dataclasses import fields as _dc_fields
        from dataclasses import replace as _dc_replace

        from identify_features.config import PipelineConfig

        name = self._preset_combo.currentText()
        if not name or name not in self._presets:
            return
        preset = self._presets[name]
        valid_field_names = {f.name for f in _dc_fields(PipelineConfig)}
        config_overrides = {k: v for k, v in preset.items() if k in valid_field_names}
        # Build a fresh PipelineConfig from the host's current config + preset overrides.
        self._window.config = _dc_replace(self._window.config, **config_overrides)
        # Apply the preset's gate_override (if any).
        gate_override = preset.get("gate_override")
        if gate_override is not None:
            self._window._gate_override = gate_override
        # If the preset includes a saved gui_state block (new format), apply
        # it; that helper also calls refresh_from_state() on both panels.
        gui_state = preset.get("gui_state")
        if gui_state:
            self._window.apply_gui_state(gui_state)
        else:
            # Old-format preset: just sync inline widgets to the new config.
            self.refresh_from_state()
            self._window.inline_custom_distances_panel.refresh_from_state()
        self._window.statusBar().showMessage(f"Applied settings preset: {name}")

    def _build_scale_group(self, parent_layout: QVBoxLayout) -> None:
        from TRACE.gui import _PlaceholderSpinBox

        gb = QGroupBox("Scale")
        form = QFormLayout(gb)
        self.scale_spin = _PlaceholderSpinBox()
        self.scale_spin.setDecimals(4)
        self.scale_spin.setRange(0.0001, 100.0)
        self.scale_spin.setSingleStep(0.001)
        self.scale_spin.set_placeholder("conversion factor")
        self.scale_spin.setToolTip("Microns per pixel — used to convert every measurement to physical units (µm, µm²).")
        self.scale_spin.valueChanged.connect(self._on_scale_changed)
        form.addRow("µm/px", self.scale_spin)

        # Estimate button — runs LandmarkLocator on the first input image (or
        # batch from the input folder) and writes the median µm/px into the
        # spinbox above. Lives in a separate row so it sits flush-left under
        # the scale field.
        self._scale_ref_um_spin = QDoubleSpinBox()
        self._scale_ref_um_spin.setRange(1.0, 100000.0)
        self._scale_ref_um_spin.setDecimals(1)
        self._scale_ref_um_spin.setSingleStep(10.0)
        self._scale_ref_um_spin.setValue(self._scale_ref_default)
        self._scale_ref_um_spin.setSuffix(" µm")
        self._scale_ref_um_spin.setToolTip(
            "Assumed real-world distance between the L3 distal end and the "
            "L1-Rs junction. The estimator divides this by the measured pixel "
            "distance to derive µm/px. Default 2200 µm matches a typical "
            "Drosophila wing."
        )
        est_btn = QToolButton()
        est_btn.setText("Estimate")
        est_btn.setToolTip(
            "Runs the landmark model on up to 100 images from the selected "
            "input folder (or on the input file if a single image is "
            "selected). For each image, the L3-distal-end ↔ L1-Rs-junction "
            "pixel distance is measured; the median µm/px across the batch "
            "(= reference µm / measured px) is written into the field above. "
            "Requires an input path on the main window and the landmark "
            "model set in Settings → Models."
        )
        est_btn.clicked.connect(self._estimate_um_per_px_from_sample)
        est_help_btn = QToolButton()
        est_help_btn.setText("?")
        est_help_btn.setToolTip("Configure the wing-length reference distance used by Estimate")
        est_help_btn.setAutoRaise(True)
        est_help_btn.clicked.connect(self._show_scale_reference_dialog)
        if not self._scale_estimator_available:
            for w in (est_btn, est_help_btn, self._scale_ref_um_spin):
                w.setEnabled(False)
            est_btn.setToolTip(
                "scale_estimator is not importable. Add scaleEstimator/ to sys.path "
                "(or install it with `pip install -e scaleEstimator`) and reopen the window."
            )
        est_row = QHBoxLayout()
        est_row.setContentsMargins(0, 0, 0, 0)
        est_row.addWidget(est_btn)
        est_row.addWidget(est_help_btn)
        est_row.addStretch(1)
        form.addRow(est_row)

        parent_layout.addWidget(gb)

    def _build_optional_preprocessing_group(self, parent_layout: QVBoxLayout) -> None:
        gb = QGroupBox("Optional preprocessing steps")
        v = QVBoxLayout(gb)

        self.wing_enable_chk = QCheckBox("Wing isolation")
        self.wing_enable_chk.setToolTip(
            "Isolates the main (image-centered) wing and masks out everything else "
            "before the rest of the pipeline sees the image. Useful when the frame "
            "contains multiple wings or stray tissue around the wing of interest. "
            "Requires a wing-isolation model — pick it in Settings → Models."
        )
        self.wing_enable_chk.toggled.connect(self._on_wing_isolation_toggled)
        v.addWidget(self.wing_enable_chk)

        self.do_rotation_chk = QCheckBox("Rotate wing")
        self.do_rotation_chk.setToolTip(
            "When checked, each wing is rotated so it sits right-side-up rather than at "
            "a skewed angle (rotation only — no mirroring or flipping). Runs as the LAST "
            "preprocessing step: every model inference (wing isolation, landmark "
            "detection, segmentation) still happens on the original un-rotated image, "
            "and the image + every produced GeoJSON (landmarks, wing, segmentation) are "
            "rotated together so identifyFeatures sees a self-consistent set. Skipped "
            "automatically when fewer than 2 reliable landmarks are available."
        )
        self.do_rotation_chk.toggled.connect(self._on_do_rotation_toggled)
        v.addWidget(self.do_rotation_chk)

        self.rotation_mirror_correct_chk = QCheckBox("Flip wing to canonical orientation")
        self.rotation_mirror_correct_chk.setToolTip(
            "When checked AND wingRotator detects a wing of opposite chirality from the "
            "canonical (right-wing) template, apply a vertical reflection on top of the "
            "rotation so the wing ends up distal-right AND anterior-up. Useful for visual "
            "consistency across mixed left+right wing batches, but flips biological "
            "chirality (a left wing is mirrored to look like a right wing). Default off: "
            "rotation only, opposite-chirality wings end up distal-left, anterior-up."
        )
        self.rotation_mirror_correct_chk.toggled.connect(self._on_mirror_correct_toggled)
        flip_row, self._flip_hint = self._wrap_with_hint(
            self.rotation_mirror_correct_chk, "requires Rotate wing", parent_chk=self.do_rotation_chk
        )
        v.addWidget(flip_row)

        parent_layout.addWidget(gb)

    def _build_crossvein_group(self, parent_layout: QVBoxLayout) -> None:
        gb = QGroupBox("Crossvein detection")
        form = QFormLayout(gb)
        self.synthesize_crossveins_chk = QCheckBox("Synthesize ACV/PCV from landmarks when not detected")
        self.synthesize_crossveins_chk.setToolTip(
            "Fall back to drawing synthetic ACV / PCV centerlines anchored to landmark "
            "positions when the segmentation finds no crossvein tissue. Off keeps strict "
            "detection-only behavior."
        )
        self.synthesize_crossveins_chk.toggled.connect(self._on_synthesize_crossveins_toggled)
        form.addRow("", self.synthesize_crossveins_chk)
        parent_layout.addWidget(gb)

    def _build_intermediate_outputs_group(self, parent_layout: QVBoxLayout) -> None:
        gb = QGroupBox("Intermediate outputs")
        gb.setToolTip(
            "Upstream/intermediate artifacts written before final overlays + CSV. "
            "Toggle which ones to keep alongside the final outputs."
        )
        v = QVBoxLayout(gb)
        self.intermediate_output_chks: dict[str, QCheckBox] = {}
        iso_hint: Optional[QLabel] = None
        for key, label in OUTPUT_TYPES.items():
            if key not in INTERMEDIATE_OUTPUTS:
                continue
            chk = QCheckBox(label)
            chk.setChecked(False)
            # Prefer an example-image tooltip when one is bundled for this
            # key; fall back to the text tooltip otherwise.
            from TRACE.output_tooltips import output_tooltip_html

            tooltip = output_tooltip_html(key, _INTERMEDIATE_TOOLTIPS.get(key, ""))
            if tooltip:
                chk.setToolTip(tooltip)
            chk.toggled.connect(lambda checked, k=key: self._on_intermediate_toggled(k, checked))
            self.intermediate_output_chks[key] = chk
            if key == "wing_isolated_image":
                iso_row, iso_hint = self._wrap_with_hint(
                    chk, "requires Wing isolation", parent_chk=self.wing_enable_chk
                )
                v.addWidget(iso_row)
            else:
                v.addWidget(chk)

        # Wing-isolated-image intermediate output depends on Wing isolation
        # being enabled — same UX as flip→rotate.
        iso_chk = self.intermediate_output_chks.get("wing_isolated_image")
        if iso_chk is not None:
            iso_chk.setEnabled(self.wing_enable_chk.isChecked())
            self.wing_enable_chk.toggled.connect(iso_chk.setEnabled)
            iso_chk.toggled.connect(
                lambda checked: (
                    self.wing_enable_chk.setChecked(True) if checked and not self.wing_enable_chk.isChecked() else None
                )
            )
            if iso_hint is not None:
                self._iso_hint = iso_hint
                self.wing_enable_chk.toggled.connect(lambda checked: self._iso_hint.hide() if checked else None)

        parent_layout.addWidget(gb)

    def _build_output_options_group(self, parent_layout: QVBoxLayout) -> None:
        gb = QGroupBox("Output options")
        v = QVBoxLayout(gb)

        self.show_vein_tissue_chk = QCheckBox("Fill buffered vein tissue in overlay")
        self.show_vein_tissue_chk.setToolTip(
            "When off (default), the per-wing overlay only shows vein skeleton "
            "centerlines. When on, it also fills the buffered vein tissue polygons."
        )
        self.show_vein_tissue_chk.toggled.connect(self._on_show_vein_tissue_toggled)
        v.addWidget(self.show_vein_tissue_chk)

        form = QFormLayout()
        self.vein_opacity_spin = QDoubleSpinBox()
        self.vein_opacity_spin.setRange(0.0, 1.0)
        self.vein_opacity_spin.setDecimals(2)
        self.vein_opacity_spin.setSingleStep(0.05)
        self.vein_opacity_spin.setToolTip("Alpha (0..1) for the vein overlay layer.")
        self.vein_opacity_spin.valueChanged.connect(self._on_vein_opacity_changed)
        form.addRow("Vein opacity", self.vein_opacity_spin)

        self.intervein_opacity_spin = QDoubleSpinBox()
        self.intervein_opacity_spin.setRange(0.0, 1.0)
        self.intervein_opacity_spin.setDecimals(2)
        self.intervein_opacity_spin.setSingleStep(0.05)
        self.intervein_opacity_spin.setToolTip("Alpha (0..1) for the intervein-region overlay layer.")
        self.intervein_opacity_spin.valueChanged.connect(self._on_intervein_opacity_changed)
        form.addRow("Intervein opacity", self.intervein_opacity_spin)
        v.addLayout(form)

        vein_gb = QGroupBox("Vein colors")
        vein_gb.setToolTip("Click a swatch to choose a custom color for that vein in the overlay.")
        self._populate_color_grid(vein_gb, self._vein_color_state, self._vein_color_btns, kind="vein")
        v.addWidget(vein_gb)

        region_gb = QGroupBox("Intervein region colors")
        region_gb.setToolTip("Click a swatch to choose a custom color for that intervein region in the overlay.")
        self._populate_color_grid(region_gb, self._region_color_state, self._region_color_btns, kind="region")
        v.addWidget(region_gb)

        parent_layout.addWidget(gb)

    def _build_parallel_processing_group(self, parent_layout: QVBoxLayout) -> None:
        gb = QGroupBox("Parallel processing")
        v = QVBoxLayout(gb)

        from TRACE.calibrate_widget import CalibrateWidget

        self._calibrate_widget = CalibrateWidget(self)
        self._refresh_calibrate_paths()
        self._calibrate_widget.applied.connect(lambda val: self.workers_spin.setValue(int(val)))
        v.addWidget(self._calibrate_widget)

        row = QHBoxLayout()
        row.addWidget(QLabel("Workers:"))
        self.workers_spin = QSpinBox()
        self.workers_spin.setRange(1, 32)
        self.workers_spin.setValue(DEFAULT_MAX_WORKERS)
        self.workers_spin.setToolTip(
            "Number of wings to process in parallel through every stage of the pipeline:\n"
            "  • Stage 1 — hinge chop and segmentation (per image)\n"
            "  • Stage 2 — identifyFeatures analysis (per image)\n"
            "Also sets the GPU batch size for the upfront landmark forward pass.\n"
            "Higher = more memory + more throughput."
        )
        self.workers_spin.valueChanged.connect(self._on_workers_changed)
        row.addWidget(self.workers_spin, stretch=1)
        self.workers_help_btn = QToolButton()
        self.workers_help_btn.setText("?")
        self.workers_help_btn.setToolTip("Show parallel-workers warning")
        self.workers_help_btn.setAutoRaise(True)
        self.workers_help_btn.clicked.connect(self._window._show_workers_warning_info)
        row.addWidget(self.workers_help_btn)
        v.addLayout(row)

        parent_layout.addWidget(gb)

    def _refresh_calibrate_paths(self) -> None:
        """Re-seed the CalibrateWidget with the window's current input + model paths."""
        if not hasattr(self, "_calibrate_widget"):
            return
        self._calibrate_widget.set_paths(
            self._window.input_edit.text() if hasattr(self._window, "input_edit") else "",
            getattr(self._window, "_landmark_model_path", ""),
            getattr(self._window, "_segmentation_model_path", ""),
        )

    # -----------------------------------------------------------------------
    # Helpers ported from settings_dialog (wrap-with-hint, color picker grid)
    # -----------------------------------------------------------------------
    def _wrap_with_hint(self, chk: QCheckBox, hint_text: str, parent_chk: QCheckBox) -> tuple[_DependentRow, QLabel]:
        """Pair a dependent checkbox with a 'requires X' hint label.

        Registers the dependency for the app-level event filter so clicking
        the (disabled) child pulses parent_chk and reveals the hint.
        """
        row = _DependentRow(chk, parent_chk)
        h = QHBoxLayout(row)
        h.setContentsMargins(0, 0, 0, 0)
        h.setSpacing(8)
        h.addWidget(chk)
        hint = QLabel(hint_text)
        hint.setStyleSheet("color: #4aa3ff;")
        hint.hide()
        row.set_hint(hint)
        h.addWidget(hint)
        h.addStretch(1)
        self._pulse_dependencies[chk] = (parent_chk, hint)
        return row, hint

    def _populate_color_grid(
        self,
        gb: QGroupBox,
        state: dict[str, list[int]],
        btn_map: dict[str, QPushButton],
        *,
        kind: str,
    ) -> None:
        grid = QGridLayout(gb)
        grid.setHorizontalSpacing(8)
        grid.setVerticalSpacing(6)
        cols = 3
        for idx, (key, rgb) in enumerate(state.items()):
            row, col_pair = divmod(idx, cols)
            col = col_pair * 2
            display = _COLOR_LABEL_OVERRIDES.get(key, key)
            btn = QPushButton()
            btn.setFixedSize(48, 22)
            btn.setStyleSheet(_swatch_style(rgb))
            btn.setCursor(Qt.PointingHandCursor)
            btn.setToolTip(f"Choose a color for {display}")
            btn.clicked.connect(lambda _checked=False, k=key, b=btn, kd=kind: self._on_color_swatch_clicked(k, b, kd))
            grid.addWidget(btn, row, col, Qt.AlignVCenter | Qt.AlignLeft)
            label = QLabel(display)
            label.setAlignment(Qt.AlignVCenter | Qt.AlignLeft)
            grid.addWidget(label, row, col + 1, Qt.AlignVCenter | Qt.AlignLeft)
            btn_map[key] = btn
        grid.setColumnStretch(cols * 2, 1)

    def _on_color_swatch_clicked(self, key: str, btn: QPushButton, kind: str) -> None:
        state = self._vein_color_state if kind == "vein" else self._region_color_state
        current = state.get(key, [128, 128, 128])
        initial = QColor(int(current[0]), int(current[1]), int(current[2]))
        display = _COLOR_LABEL_OVERRIDES.get(key, key)
        chosen = QColorDialog.getColor(initial, self, f"Pick color for {display}")
        if not chosen.isValid():
            return
        rgb = [chosen.red(), chosen.green(), chosen.blue()]
        state[key] = rgb
        btn.setStyleSheet(_swatch_style(rgb))
        self._commit_color_overrides()

    def _commit_color_overrides(self) -> None:
        """Emit only entries that differ from the topology defaults into window.config."""
        vein_diffs = {k: list(v) for k, v in self._vein_color_state.items() if v != self._topology_vein_defaults.get(k)}
        region_diffs = {
            k: list(v) for k, v in self._region_color_state.items() if v != self._topology_region_defaults.get(k)
        }
        self._window.config.vein_colors = vein_diffs or None
        self._window.config.region_colors = region_diffs or None

    # -----------------------------------------------------------------------
    # Signal handlers — every one writes a single window attribute / config field.
    # -----------------------------------------------------------------------
    def _on_scale_changed(self, val: float) -> None:
        # Route through the window so the left-panel mirror updates in lock-step.
        self._window._set_scale(val, source="inline")

    def _on_wing_isolation_toggled(self, checked: bool) -> None:
        self._window._wing_isolation_enabled = bool(checked)

    def _on_do_rotation_toggled(self, checked: bool) -> None:
        self._window._do_rotation = bool(checked)
        # Mirror-correct only meaningful when rotation is enabled.
        self.rotation_mirror_correct_chk.setEnabled(bool(checked))
        if checked:
            self._flip_hint.hide()

    def _on_mirror_correct_toggled(self, checked: bool) -> None:
        self._window._rotation_mirror_correct = bool(checked)
        if checked and not self.do_rotation_chk.isChecked():
            # Auto-enable rotation so the requested output is actually reachable.
            self.do_rotation_chk.setChecked(True)

    def _on_synthesize_crossveins_toggled(self, checked: bool) -> None:
        self._window.config.synthesize_missing_crossveins = bool(checked)

    def _on_intermediate_toggled(self, key: str, checked: bool) -> None:
        self._window._intermediate_outputs[key] = bool(checked)

    def _on_show_vein_tissue_toggled(self, checked: bool) -> None:
        self._window._show_vein_tissue = bool(checked)

    def _on_vein_opacity_changed(self, val: float) -> None:
        self._window.config.vein_opacity = float(val)

    def _on_intervein_opacity_changed(self, val: float) -> None:
        self._window.config.intervein_opacity = float(val)

    def _on_workers_changed(self, val: int) -> None:
        self._window.maybe_show_workers_warning(int(val))

    # -----------------------------------------------------------------------
    # refresh_from_state
    # -----------------------------------------------------------------------
    def restore_defaults(self) -> None:
        """Reset every Settings-tab-owned field on the host window to its default.

        Scope: scale, opacities, colors, synthesize-crossveins, preprocessing
        toggles (wing isolation / rotate / flip), intermediate outputs,
        show-vein-tissue, workers count. Does not touch model paths, input/
        output folders, recursive flag, output checkboxes, or any other
        PipelineConfig fields owned by the advanced settings dialog.
        """
        from identify_features.config import PipelineConfig

        defaults = PipelineConfig()
        cfg = self._window.config
        cfg.um_per_px = defaults.um_per_px
        # Opacities + colors come from the snapshots captured at panel
        # construction (i.e. the starting state the user saw on first launch),
        # not from PipelineConfig() — so any future change to PipelineConfig
        # defaults can't drift the Restore Defaults target.
        cfg.vein_opacity = self._default_vein_opacity
        cfg.intervein_opacity = self._default_intervein_opacity
        # vein_colors / region_colors round-trip as override dicts; setting to
        # None means "no overrides — fall back to topology defaults", which is
        # what the snapshot represents. Use None for the minimal config shape.
        cfg.vein_colors = None
        cfg.region_colors = None
        # Reset the in-panel color state dicts to the snapshots so swatches
        # restyle even if the renderer-side fallback ever drifts from topology.
        for key, rgb in self._default_vein_colors.items():
            self._vein_color_state[key] = list(rgb)
        for key, rgb in self._default_region_colors.items():
            self._region_color_state[key] = list(rgb)
        cfg.synthesize_missing_crossveins = defaults.synthesize_missing_crossveins

        self._window._wing_isolation_enabled = False
        self._window._do_rotation = False
        self._window._rotation_mirror_correct = False
        self._window._show_vein_tissue = False
        self._window._intermediate_outputs = {key: False for key in self._window._intermediate_outputs}
        self._window.settings.setValue("max_workers", DEFAULT_MAX_WORKERS)
        # Re-arm the parallel-workers warning so a fresh climb above the
        # default triggers it again post-reset.
        self._window._workers_warning_shown = False

        self.refresh_from_state()
        # Also nudge the left-panel scale mirror through _set_scale so it
        # matches the freshly-reset um_per_px (refresh_from_state already
        # handles this, but call defensively to be explicit).
        if hasattr(self._window, "_set_scale"):
            self._window._set_scale(self.scale_spin.value(), source="inline")
        self._window.statusBar().showMessage("Settings tab reset to defaults")

    def refresh_from_state(self) -> None:
        """Pull every widget value from the host window's current state.

        Called after operations that change TraceWindow state from outside the
        panel (settings-dialog OK, wipe-my-memories, config import). Blocks
        signals during the refresh so handlers don't echo state back to itself.
        """
        widgets = (
            self.scale_spin,
            self.wing_enable_chk,
            self.do_rotation_chk,
            self.rotation_mirror_correct_chk,
            self.synthesize_crossveins_chk,
            self.show_vein_tissue_chk,
            self.vein_opacity_spin,
            self.intervein_opacity_spin,
            self.workers_spin,
        )
        widgets = widgets + tuple(self.intermediate_output_chks.values())
        for w in widgets:
            w.blockSignals(True)
        try:
            cfg = self._window.config
            scale_val = float(cfg.um_per_px) if cfg.um_per_px is not None else self.scale_spin.minimum()
            self.scale_spin.setValue(scale_val)
            # Mirror to the left-panel spinbox too (also with blocked signals
            # so it doesn't re-fire _set_scale and bounce back).
            left = getattr(self._window, "scale_spin", None)
            if left is not None:
                left.blockSignals(True)
                left.setValue(scale_val)
                left.blockSignals(False)
            self.wing_enable_chk.setChecked(bool(self._window._wing_isolation_enabled))
            self.do_rotation_chk.setChecked(bool(self._window._do_rotation))
            self.rotation_mirror_correct_chk.setChecked(bool(self._window._rotation_mirror_correct))
            self.rotation_mirror_correct_chk.setEnabled(bool(self._window._do_rotation))
            self.synthesize_crossveins_chk.setChecked(bool(cfg.synthesize_missing_crossveins))
            self.show_vein_tissue_chk.setChecked(bool(self._window._show_vein_tissue))
            self.vein_opacity_spin.setValue(float(cfg.vein_opacity))
            self.intervein_opacity_spin.setValue(float(cfg.intervein_opacity))
            for key, chk in self.intermediate_output_chks.items():
                chk.setChecked(bool(self._window._intermediate_outputs.get(key, False)))
            iso_chk = self.intermediate_output_chks.get("wing_isolated_image")
            if iso_chk is not None:
                iso_chk.setEnabled(self.wing_enable_chk.isChecked())
            # Color picker swatches — start from topology defaults, layer config overrides.
            for key, default_rgb in self._topology_vein_defaults.items():
                override = (cfg.vein_colors or {}).get(key)
                rgb = list(override) if override is not None else list(default_rgb)
                self._vein_color_state[key] = rgb
                btn = self._vein_color_btns.get(key)
                if btn is not None:
                    btn.setStyleSheet(_swatch_style(rgb))
            for key, default_rgb in self._topology_region_defaults.items():
                override = (cfg.region_colors or {}).get(key)
                rgb = list(override) if override is not None else list(default_rgb)
                self._region_color_state[key] = rgb
                btn = self._region_color_btns.get(key)
                if btn is not None:
                    btn.setStyleSheet(_swatch_style(rgb))
            # Workers spinner — read from QSettings shared with the rest of
            # the window. The default is DEFAULT_MAX_WORKERS when no value
            # has been persisted yet.
            workers_val = self._window.settings.value("max_workers", DEFAULT_MAX_WORKERS)
            try:
                self.workers_spin.setValue(int(workers_val))
            except (TypeError, ValueError):
                self.workers_spin.setValue(DEFAULT_MAX_WORKERS)
        finally:
            for w in widgets:
                w.blockSignals(False)
        # CalibrateWidget paths can change when the user picks a new input
        # folder or new models — re-seed defensively.
        self._refresh_calibrate_paths()

    # -----------------------------------------------------------------------
    # Scale estimator — ported from settings_dialog
    # -----------------------------------------------------------------------
    _SCALE_ESTIMATOR_MAX_IMAGES = 100

    def _show_scale_reference_dialog(self) -> None:
        dlg = self._scale_ref_dialog
        if dlg is None:
            dlg = QDialog(self)
            dlg.setWindowTitle("Wing length reference")
            form = QFormLayout(dlg)
            form.addRow("Wing length reference", self._scale_ref_um_spin)
            pick_btn = QPushButton("Estimate from one wing…")
            pick_btn.setToolTip(
                "Pick a single wing image. The landmark model runs on that one wing and "
                "µm/px = (reference µm) / (measured L3-distal-end ↔ L1-Rs-junction "
                "distance in px) is written into the µm/px field above."
            )
            pick_btn.clicked.connect(lambda: self._estimate_um_per_px_from_picked_image(dlg))
            if not self._scale_estimator_available:
                pick_btn.setEnabled(False)
                pick_btn.setToolTip(
                    "scale_estimator is not importable. Add scaleEstimator/ to sys.path "
                    "(or install it with `pip install -e scaleEstimator`) and reopen the window."
                )
            form.addRow(pick_btn)
            btns = QDialogButtonBox(QDialogButtonBox.Close)
            btns.rejected.connect(dlg.accept)
            form.addRow(btns)
            self._scale_ref_dialog = dlg
        dlg.exec_()

    def _confirm_scale_estimator_caveats(self) -> bool:
        box = QMessageBox(self)
        box.setIcon(QMessageBox.Warning)
        box.setWindowTitle("Estimated scale — read first")
        box.setText(
            "This tool <b>estimates</b> µm/px by assuming a fixed wing length "
            "(default 2200 µm — the L3 distal end ↔ L1-Rs junction distance)."
        )
        box.setInformativeText(
            "Real wing length varies with genotype, sex, rearing conditions, "
            "and individual specimens, and the landmark predictions themselves "
            "carry residual error. The output is only as accurate as those "
            "assumptions allow.\n\n"
            "For publication-quality data, calibrate µm/px directly — e.g. "
            "from a stage-micrometer image taken under the same optics, or "
            "from the microscope's TIFF / OME-XML metadata — rather than "
            "relying on this estimate.\n\n"
            "Continue with the estimate?"
        )
        box.setStandardButtons(QMessageBox.Ok | QMessageBox.Cancel)
        box.setDefaultButton(QMessageBox.Cancel)
        return box.exec_() == QMessageBox.Ok

    def _picker_initial_path(self, current: str) -> str:
        from TRACE.gui import _picker_initial_path

        return _picker_initial_path(current)

    def _estimate_um_per_px_from_picked_image(self, host_dialog: QDialog | None = None) -> None:
        lm_path = (getattr(self._window, "_landmark_model_path", "") or "").strip()
        if not lm_path:
            QMessageBox.warning(
                self,
                "No landmark model",
                "Set the Landmark model in Settings → Models first — the estimator "
                "needs it to detect the L3 distal end and the L1-Rs junction.",
            )
            return
        if not self._confirm_scale_estimator_caveats():
            return
        seed = self._window.input_edit.text() or lm_path
        image_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select a wing image",
            self._picker_initial_path(seed),
            "Images (*.tif *.tiff *.png *.jpg *.jpeg *.bmp *.psd *.ome.tif);;All Files (*)",
        )
        if not image_path:
            return
        reference_um = float(self._scale_ref_um_spin.value())
        QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            from scale_estimator import ScaleEstimate, ScaleEstimationError, estimate_um_per_px

            estimate: ScaleEstimate = estimate_um_per_px(
                Path(image_path),
                Path(lm_path),
                reference_distance_um=reference_um,
            )
        except ScaleEstimationError as exc:
            QApplication.restoreOverrideCursor()
            QMessageBox.warning(self, "Estimate failed", str(exc))
            return
        except Exception as exc:  # noqa: BLE001
            QApplication.restoreOverrideCursor()
            QMessageBox.critical(
                self,
                "Estimate failed",
                f"Could not run the landmark model on this image:\n{exc}",
            )
            return
        finally:
            QApplication.restoreOverrideCursor()

        self.scale_spin.setValue(float(estimate.um_per_px))
        flags = []
        if not estimate.dtip_reliable:
            flags.append("L3 distal end low confidence")
        if not estimate.l1_rs_reliable:
            flags.append("L1-Rs junction low confidence")
        flag_line = ("\n\nWarning: " + "; ".join(flags) + ".") if flags else ""
        QMessageBox.information(
            self,
            "Scale estimated",
            f"Sample image:        {Path(image_path).name}\n"
            f"L3 distal end ↔ L1-Rs junction:   {estimate.distance_px:.1f} px\n"
            f"Reference distance:  {estimate.reference_distance_um:.1f} µm\n"
            f"Estimated scale:     {estimate.um_per_px:.4f} µm/px"
            f"{flag_line}",
        )
        if host_dialog is not None:
            host_dialog.accept()

    def _estimate_um_per_px_from_sample(self) -> None:
        lm_path = (getattr(self._window, "_landmark_model_path", "") or "").strip()
        if not lm_path:
            QMessageBox.warning(
                self,
                "No landmark model",
                "Set the Landmark model in Settings → Models first — the estimator "
                "needs it to detect the L3 distal end and the L1-Rs junction.",
            )
            return
        if not self._confirm_scale_estimator_caveats():
            return
        input_path_str = (self._window.input_edit.text() or "").strip()
        if not input_path_str:
            QMessageBox.warning(
                self,
                "No input folder",
                "Pick an input image or folder on the main window first — the "
                "estimator runs on the images in that folder.",
            )
            return
        input_path = Path(input_path_str)
        if input_path.is_file():
            image_paths = [input_path]
        elif input_path.is_dir():
            try:
                from preprocessing.pipeline import discover_images
            except ImportError as exc:
                QMessageBox.critical(
                    self,
                    "Cannot list folder",
                    f"preprocessing.discover_images is not importable:\n{exc}",
                )
                return
            discovered = discover_images(input_path, recursive=False)
            if not discovered:
                QMessageBox.warning(
                    self,
                    "No images found",
                    f"No supported image files in:\n{input_path}",
                )
                return
            image_paths = discovered[: self._SCALE_ESTIMATOR_MAX_IMAGES]
        else:
            QMessageBox.warning(
                self,
                "Input not found",
                f"Input path does not exist:\n{input_path_str}",
            )
            return

        reference_um = float(self._scale_ref_um_spin.value())
        progress = QProgressDialog(
            f"Estimating scale on {len(image_paths)} image(s)…",
            "Cancel",
            0,
            len(image_paths),
            self,
        )
        progress.setWindowTitle("Estimating scale")
        progress.setWindowModality(Qt.WindowModal)
        progress.setMinimumDuration(0)
        progress.setValue(0)
        QApplication.processEvents()

        def _on_progress(done: int, total: int) -> bool:
            progress.setMaximum(total)
            progress.setValue(done)
            QApplication.processEvents()
            return progress.wasCanceled()

        try:
            from scale_estimator import (
                FolderScaleEstimate,
                ScaleEstimationError,
                estimate_um_per_px_from_paths,
            )

            result: FolderScaleEstimate = estimate_um_per_px_from_paths(
                image_paths,
                Path(lm_path),
                reference_distance_um=reference_um,
                progress_callback=_on_progress,
            )
        except ScaleEstimationError as exc:
            progress.close()
            QMessageBox.warning(self, "Estimate failed", str(exc))
            return
        except Exception as exc:  # noqa: BLE001
            progress.close()
            QMessageBox.critical(
                self,
                "Estimate failed",
                f"Could not run the landmark model:\n{exc}",
            )
            return
        finally:
            progress.close()

        self.scale_spin.setValue(float(result.um_per_px))
        notes = []
        if result.cancelled:
            notes.append("Cancelled by user — median computed over the images processed so far.")
        n_low_conf = sum(
            1 for _, est, _ in result.per_image if est is not None and (not est.dtip_reliable or not est.l1_rs_reliable)
        )
        if n_low_conf:
            notes.append(f"{n_low_conf} of {result.n_used} estimate(s) had a low-confidence landmark.")
        note_line = ("\n\n" + "\n".join(notes)) if notes else ""
        QMessageBox.information(
            self,
            "Scale estimated",
            f"Images used:          {result.n_used} of {result.n_tried}\n"
            f"Reference distance:   {result.reference_distance_um:.1f} µm\n"
            f"Estimated scale:      {result.um_per_px:.4f} µm/px (median)"
            f"{note_line}",
        )


# ---------------------------------------------------------------------------
# InlineCustomDistancesPanel
# ---------------------------------------------------------------------------


class InlineCustomDistancesPanel(QWidget):
    """Custom-measurements picker (measurement_maker.LandmarkPickerWidget) inline."""

    # Bundled fallback wing for the picker — shows by default when the user
    # hasn't picked their own sample image yet. Resolved relative to this
    # module so it works regardless of CWD or install layout.
    _CARTOON_DIR = Path(__file__).resolve().parent / "GUI_images" / "cartoon"
    _CARTOON_IMAGE = _CARTOON_DIR / "wing_cartoon.png"
    _CARTOON_LANDMARKS = _CARTOON_DIR / "wing_cartoon_landmarks.geojson"

    def __init__(self, window: "TraceWindow"):
        super().__init__()
        self._window = window
        self._picker = None  # type: ignore[assignment]
        self._build_ui()

    def _resolve_paths(self) -> tuple[str, str]:
        """Return (image, landmarks) paths to seed the picker.

        User-saved paths take precedence; otherwise the bundled cartoon is
        used (if present). Either or both may end up empty when nothing's
        configured and the cartoon files are missing.
        """
        img = self._window._distance_sample_image or (str(self._CARTOON_IMAGE) if self._CARTOON_IMAGE.is_file() else "")
        lm = self._window._distance_sample_landmarks or (
            str(self._CARTOON_LANDMARKS) if self._CARTOON_LANDMARKS.is_file() else ""
        )
        return img, lm

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(6)

        info = QLabel(
            "Configure custom straight-line measurements between any two landmarks. Each pair "
            "adds custom_<label>_px (and _um when scale is set) columns to the batch CSV. "
            "Pairs are stored by landmark name and applied to every wing in the batch."
        )
        info.setWordWrap(True)
        info.setStyleSheet("color: #aaa;")
        layout.addWidget(info)

        try:
            from measurement_maker import LandmarkPickerWidget, pairs_from_dicts
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
            self._picker = None
            self._import_error = True
            return

        self._import_error = False
        initial = pairs_from_dicts(list(self._window._user_landmark_distances))
        img_path, lm_path = self._resolve_paths()
        # LandmarkPickerWidget constructs a napari viewer internally —
        # this is the most likely failure point on a fresh PyInstaller
        # bundle (missing OpenGL drivers, vispy backend, qt plugin, etc.).
        # Catch + log so the GUI as a whole still opens and we get a
        # readable traceback in trace_startup.log.
        try:
            from TRACE.startup_log import log, log_exception

            log("InlineCustomDistancesPanel: constructing LandmarkPickerWidget")
            self._picker = LandmarkPickerWidget(
                parent=self,
                initial_pairs=initial,
                default_image_dir=self._window.input_edit.text() if hasattr(self._window, "input_edit") else "",
                initial_image_path=img_path,
                initial_landmarks_path=lm_path,
            )
            log("InlineCustomDistancesPanel: LandmarkPickerWidget OK")
        except BaseException as exc:  # noqa: BLE001
            log_exception("LandmarkPickerWidget construction failed", exc)
            err = QLabel(
                f"Custom Measurements unavailable — failed to start the napari viewer:\n\n"
                f"{type(exc).__name__}: {exc}\n\n"
                "See trace_startup.log next to TRACE.exe for the full traceback."
            )
            err.setWordWrap(True)
            err.setStyleSheet("color: #f88; padding: 12px;")
            layout.addWidget(err)
            layout.addStretch()
            self._picker = None
            self._import_error = True
            return
        self._picker.pairs_changed.connect(self._on_pairs_changed)
        layout.addWidget(self._picker, stretch=1)
        # Auto-load when both paths point at real files — covers the bundled-
        # cartoon-default case and the user-restored-session case.
        if img_path and lm_path and Path(img_path).is_file() and Path(lm_path).is_file():
            QTimer.singleShot(0, self._picker.load_initial)

    def _on_pairs_changed(self, pairs) -> None:
        self._window._user_landmark_distances = [asdict(p) for p in pairs]
        if self._picker is not None:
            self._window._distance_sample_image = self._picker.image_path()
            self._window._distance_sample_landmarks = self._picker.landmarks_path()

    def refresh_from_state(self) -> None:
        """Rebuild the picker from current window state (after import / reset).

        LandmarkPickerWidget has no public setter for pairs / image / landmarks,
        so a full rebuild is the simplest correct way to reflect external state
        changes without leaving stale rows in the side panel.
        """
        if getattr(self, "_import_error", False):
            return
        # Tear down the existing picker and recreate.
        layout = self.layout()
        if self._picker is not None:
            layout.removeWidget(self._picker)
            self._picker.deleteLater()
            self._picker = None
        try:
            from measurement_maker import LandmarkPickerWidget, pairs_from_dicts
        except ImportError:
            return
        initial = pairs_from_dicts(list(self._window._user_landmark_distances))
        img_path, lm_path = self._resolve_paths()
        self._picker = LandmarkPickerWidget(
            parent=self,
            initial_pairs=initial,
            default_image_dir=self._window.input_edit.text() if hasattr(self._window, "input_edit") else "",
            initial_image_path=img_path,
            initial_landmarks_path=lm_path,
        )
        self._picker.pairs_changed.connect(self._on_pairs_changed)
        layout.addWidget(self._picker, stretch=1)
        if img_path and lm_path and Path(img_path).is_file() and Path(lm_path).is_file():
            QTimer.singleShot(0, self._picker.load_initial)


# ---------------------------------------------------------------------------
# InlineHelpPanel
# ---------------------------------------------------------------------------


class InlineHelpPanel(QWidget):
    """Help tab — clickable links to the TRACE README and project repo."""

    def __init__(self, window: "TraceWindow"):
        super().__init__()
        self._window = window
        self._build_ui()

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(12)

        title = QLabel("Documentation")
        title_font = QFont(title.font())
        title_font.setPointSize(title_font.pointSize() + 2)
        title_font.setBold(True)
        title.setFont(title_font)
        layout.addWidget(title)

        readme_path = Path(__file__).resolve().parent / "README.md"
        if readme_path.is_file():
            url = QUrl.fromLocalFile(str(readme_path)).toString()
            link = QLabel(f'<a href="{url}" style="color: #4aa3ff;">Open README.md in your default app</a>')
            link.setOpenExternalLinks(True)
            link.setTextInteractionFlags(Qt.TextBrowserInteraction)
            layout.addWidget(link)

            path_label = QLabel(f"<span style='color: #888;'>Location:</span> {readme_path}")
            path_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
            path_label.setWordWrap(True)
            layout.addWidget(path_label)
        else:
            missing = QLabel(f"<span style='color: #f88;'>README.md not found at:</span><br>{readme_path}")
            missing.setWordWrap(True)
            layout.addWidget(missing)

        # GitHub repo link — for source, issues, and contributions.
        github_link = QLabel(
            '<a href="https://github.com/alexmpdx/TRACE" style="color: #4aa3ff;">' "View TRACE on GitHub</a>"
        )
        github_link.setOpenExternalLinks(True)
        github_link.setTextInteractionFlags(Qt.TextBrowserInteraction)
        layout.addWidget(github_link)

        # Walkthrough replay — re-runs the same first-launch tour the user
        # saw on initial startup. Sits in its own labeled section so it's
        # obviously distinct from the README link above.
        layout.addSpacing(12)
        tour_title = QLabel("Walkthrough")
        tour_title_font = QFont(tour_title.font())
        tour_title_font.setPointSize(tour_title_font.pointSize() + 2)
        tour_title_font.setBold(True)
        tour_title.setFont(tour_title_font)
        layout.addWidget(tour_title)

        tour_blurb = QLabel(
            "Re-run the guided tour of the main controls — the same one that "
            "appeared the first time you opened TRACE."
        )
        tour_blurb.setWordWrap(True)
        tour_blurb.setStyleSheet("color: #aaa;")
        layout.addWidget(tour_blurb)

        replay_row = QHBoxLayout()
        self.btn_replay_walkthrough = QPushButton("Replay walkthrough")
        self.btn_replay_walkthrough.setToolTip(
            "Restart the first-launch walkthrough. Step through with Next / Previous, or skip with Esc."
        )
        self.btn_replay_walkthrough.clicked.connect(self._window._show_walkthrough)
        replay_row.addWidget(self.btn_replay_walkthrough)
        replay_row.addStretch(1)
        layout.addLayout(replay_row)

        # Update section — current installed version + button that opens the
        # GitHub Releases page so the user can grab the latest installer.
        # Installing over an existing TRACE upgrades it in place (Inno Setup
        # detects the AppId match) — no need to uninstall first.
        layout.addSpacing(12)
        update_title = QLabel("Update")
        update_title_font = QFont(update_title.font())
        update_title_font.setPointSize(update_title_font.pointSize() + 2)
        update_title_font.setBold(True)
        update_title.setFont(update_title_font)
        layout.addWidget(update_title)

        try:
            from TRACE import __version__ as _trace_version
        except Exception:
            _trace_version = "unknown"
        self._version_label = QLabel(
            f"<span style='color: #aaa;'>Installed version:</span> "
            f"<span style='color: #d0d0d0;'>{_trace_version}</span>"
        )
        layout.addWidget(self._version_label)

        update_blurb = QLabel(
            "Get the latest TRACE installer from the project's Releases page. "
            "Running the new installer upgrades your existing TRACE in place — "
            "no need to uninstall first. Your settings and downloaded models "
            "are preserved."
        )
        update_blurb.setWordWrap(True)
        update_blurb.setStyleSheet("color: #aaa;")
        layout.addWidget(update_blurb)

        self._update_status_label = QLabel("")
        self._update_status_label.setWordWrap(True)
        self._update_status_label.setOpenExternalLinks(True)
        self._update_status_label.setTextInteractionFlags(Qt.TextBrowserInteraction)
        layout.addWidget(self._update_status_label)

        update_row = QHBoxLayout()
        self.btn_check_updates = QPushButton("Check for updates")
        self.btn_check_updates.setToolTip(
            "Query the GitHub Releases page for the latest TRACE installer and report whether you're up to date."
        )
        self.btn_check_updates.clicked.connect(self._check_for_updates)
        update_row.addWidget(self.btn_check_updates)

        # "Install Update" only appears (a) on the frozen build, and
        # (b) after a check that found a newer release. _check_for_updates
        # stores the asset URL + size in self._latest_update_* before
        # making this button visible.
        self.btn_install_update = QPushButton("Install Update")
        self.btn_install_update.setToolTip(
            "Download the latest TRACE installer and launch it. The new "
            "installer upgrades over the current install — your settings "
            "and downloaded models are preserved."
        )
        self.btn_install_update.setVisible(False)
        self.btn_install_update.clicked.connect(self._install_update)
        update_row.addWidget(self.btn_install_update)

        self.btn_open_releases = QPushButton("View all releases…")
        self.btn_open_releases.setToolTip(
            "Open the Releases page in your default browser. Download the "
            "latest TRACE-Setup.exe from there and run it to upgrade."
        )
        self.btn_open_releases.clicked.connect(self._open_releases_page)
        update_row.addWidget(self.btn_open_releases)
        update_row.addStretch(1)
        layout.addLayout(update_row)

        # Populated by _check_for_updates when a newer release is found.
        self._latest_update_url: Optional[str] = None
        self._latest_update_size: Optional[int] = None
        self._latest_update_version: Optional[str] = None

        layout.addStretch(1)

        footer = QLabel(
            "<span style='color: #888;'>For pipeline-internal docs, see the comments in "
            "<code>TRACE/pipeline.py</code> and <code>TRACE/gui.py</code>.</span>"
        )
        footer.setWordWrap(True)
        layout.addWidget(footer)

    # -----------------------------------------------------------------------
    # Update check
    # -----------------------------------------------------------------------
    _RELEASES_PAGE_URL = "https://github.com/alexmpdx/TRACE/releases"
    _LATEST_RELEASE_API = "https://api.github.com/repos/alexmpdx/TRACE/releases/latest"

    def _open_releases_page(self) -> None:
        QDesktopServices_open = None
        try:
            from PyQt5.QtCore import QUrl as _QUrl
            from PyQt5.QtGui import QDesktopServices

            QDesktopServices_open = QDesktopServices.openUrl
            QDesktopServices_open(_QUrl(self._RELEASES_PAGE_URL))
        except Exception:
            # Fall back to webbrowser stdlib if Qt's URL opener trips.
            import webbrowser

            webbrowser.open(self._RELEASES_PAGE_URL)

    def _check_for_updates(self) -> None:
        """Query GitHub's REST API for the latest release tag and compare.

        Runs synchronously — the request is small (a few KB of JSON) and
        completes in well under a second on a normal connection. Network
        failures show a friendly message instead of crashing.
        """
        try:
            from TRACE import __version__ as installed_version
        except Exception:
            installed_version = "unknown"

        self._update_status_label.setText("<span style='color: #888;'>Checking for updates…</span>")
        # Force a repaint before the blocking HTTP request so the user
        # sees the "checking" message right away.
        self._update_status_label.repaint()

        import json
        import urllib.request

        try:
            req = urllib.request.Request(
                self._LATEST_RELEASE_API,
                headers={"User-Agent": "TRACE-update-check"},
            )
            with urllib.request.urlopen(req, timeout=10) as resp:
                data = json.load(resp)
            latest_tag = str(data.get("tag_name") or "")
            html_url = str(data.get("html_url") or self._RELEASES_PAGE_URL)
            # Find the TRACE-Setup.exe asset on this release so the
            # "Install Update" button can download it directly.
            asset_url: Optional[str] = None
            asset_size: Optional[int] = None
            for asset in data.get("assets") or []:
                if asset.get("name") == "TRACE-Setup.exe":
                    asset_url = asset.get("browser_download_url") or asset.get("url")
                    try:
                        asset_size = int(asset.get("size") or 0) or None
                    except Exception:
                        asset_size = None
                    break
        except Exception as e:
            self._update_status_label.setText(
                f"<span style='color: #f88;'>Could not check for updates: {e}</span><br>"
                f"<a href='{self._RELEASES_PAGE_URL}' style='color: #4aa3ff;'>Open the Releases page manually</a>"
            )
            return

        # Strip a "windows-v" / "v" prefix from the tag for comparison.
        latest_version = latest_tag
        for prefix in ("windows-v", "v"):
            if latest_version.startswith(prefix):
                latest_version = latest_version[len(prefix) :]
                break

        if not latest_version:
            self._update_status_label.setText(
                f"<span style='color: #aaa;'>No releases found on GitHub yet. "
                f"<a href='{self._RELEASES_PAGE_URL}' style='color: #4aa3ff;'>"
                f"Check the Releases page</a>.</span>"
            )
            return

        if latest_version == installed_version:
            self._update_status_label.setText(
                f"<span style='color: #6c6;'>✓ You're up to date (installed: {installed_version}).</span>"
            )
            self.btn_install_update.setVisible(False)
            self._latest_update_url = None
            self._latest_update_size = None
            self._latest_update_version = None
            return

        # Newer (or just different) version available. Stash the asset
        # URL + size for the Install Update button. If we couldn't find
        # a TRACE-Setup.exe asset, or we're running from source rather
        # than a frozen bundle (where launching an installer makes no
        # sense), fall back to the release-page link.
        self._latest_update_url = asset_url
        self._latest_update_size = asset_size
        self._latest_update_version = latest_version

        can_install_in_place = bool(asset_url) and getattr(sys, "frozen", False) and sys.platform == "win32"
        if can_install_in_place:
            self.btn_install_update.setText(f"Install update {latest_version}")
            self.btn_install_update.setVisible(True)
            size_mb = (asset_size or 0) // (1024 * 1024)
            size_blurb = f" ({size_mb} MB)" if size_mb else ""
            self._update_status_label.setText(
                f"<span style='color: #ffb05a;'>Update available: "
                f"<b>{latest_version}</b> (you have {installed_version}).</span><br>"
                f"<span style='color: #aaa;'>Click <b>Install update {latest_version}</b> "
                f"to download{size_blurb} and launch the new installer.</span>"
            )
        else:
            self.btn_install_update.setVisible(False)
            self._update_status_label.setText(
                f"<span style='color: #ffb05a;'>A different version is available: "
                f"<b>{latest_version}</b> (you have {installed_version}).</span><br>"
                f"<a href='{html_url}' style='color: #4aa3ff;'>"
                f"Open the release page and download TRACE-Setup.exe</a>"
            )

    def _install_update(self) -> None:
        """Download the latest TRACE-Setup.exe and launch it.

        Called from the Install Update button after _check_for_updates
        has populated self._latest_update_url. Shows a QProgressDialog
        for the download; on success, runs the installer in a detached
        process and exits the current TRACE app so the installer can
        overwrite the .exe.
        """
        url = self._latest_update_url
        version = self._latest_update_version or "?"
        if not url:
            self._update_status_label.setText(
                "<span style='color: #f88;'>No installer URL — run Check for updates first.</span>"
            )
            return

        import tempfile

        dst = Path(tempfile.gettempdir()) / "TRACE-Setup.exe"

        dlg = QProgressDialog(
            f"Downloading TRACE {version}…",
            "Cancel",
            0,
            100,
            self,
        )
        dlg.setWindowTitle("TRACE update")
        dlg.setWindowModality(Qt.WindowModal)
        dlg.setMinimumDuration(0)
        dlg.setAutoClose(False)
        dlg.setAutoReset(False)
        dlg.setValue(0)
        dlg.show()
        QApplication.processEvents()

        cancelled = False
        try:
            import urllib.request

            req = urllib.request.Request(url, headers={"User-Agent": "TRACE-update"})
            with urllib.request.urlopen(req, timeout=30) as resp:
                total = int(resp.headers.get("Content-Length") or 0) or (self._latest_update_size or 0)
                downloaded = 0
                chunk = 1 << 20  # 1 MB
                last_pct = -1
                with open(dst, "wb") as out:
                    while True:
                        if dlg.wasCanceled():
                            cancelled = True
                            break
                        buf = resp.read(chunk)
                        if not buf:
                            break
                        out.write(buf)
                        downloaded += len(buf)
                        if total:
                            pct = int(downloaded * 100 / total)
                            if pct != last_pct:
                                last_pct = pct
                                mb_done = downloaded // (1024 * 1024)
                                mb_total = total // (1024 * 1024)
                                dlg.setLabelText(f"Downloading TRACE {version}…\n{mb_done} / {mb_total} MB ({pct}%)")
                                dlg.setValue(pct)
                        QApplication.processEvents()
        except Exception as e:  # noqa: BLE001
            dlg.close()
            try:
                dst.unlink(missing_ok=True)
            except Exception:
                pass
            QMessageBox.critical(
                self,
                "Update download failed",
                f"Could not download TRACE {version}:\n\n{e}\n\n"
                f"Check your internet connection and try again, or use "
                f"'View all releases…' to download manually.",
            )
            return
        dlg.close()
        if cancelled:
            try:
                dst.unlink(missing_ok=True)
            except Exception:
                pass
            self._update_status_label.setText("<span style='color: #aaa;'>Update download cancelled.</span>")
            return

        # Sanity check on file size — guards against a truncated download
        # that didn't trigger an exception above.
        if self._latest_update_size and dst.stat().st_size != self._latest_update_size:
            QMessageBox.critical(
                self,
                "Update download incomplete",
                f"Downloaded file size ({dst.stat().st_size} bytes) does not "
                f"match the expected size ({self._latest_update_size} bytes). "
                f"Please try Check for updates again, or download manually.",
            )
            try:
                dst.unlink(missing_ok=True)
            except Exception:
                pass
            return

        # Hand off to the installer and exit the current app so it can
        # overwrite TRACE.exe without a "file in use" error. The
        # installer (Inno Setup) handles the rest from here.
        try:
            import os
            import subprocess

            if sys.platform == "win32":
                # DETACHED_PROCESS so the child isn't killed when we exit.
                DETACHED_PROCESS = 0x00000008
                subprocess.Popen([str(dst)], creationflags=DETACHED_PROCESS, close_fds=True)
            else:
                # Dev convenience — just open the file with the system
                # default handler. (Not the expected path; the Install
                # Update button is gated to sys.platform == 'win32'.)
                os.startfile(str(dst))  # noqa: SIM117
        except Exception as e:  # noqa: BLE001
            QMessageBox.critical(
                self,
                "Could not launch installer",
                f"Downloaded the installer to:\n\n{dst}\n\n"
                f"But couldn't launch it automatically:\n{e}\n\n"
                f"Run TRACE-Setup.exe manually from the location above.",
            )
            return

        # Exit TRACE so the installer can replace files.
        QApplication.quit()
