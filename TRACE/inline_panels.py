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

import json
import os
import sys
from dataclasses import asdict
from pathlib import Path
from typing import TYPE_CHECKING, Optional

from PyQt5.QtCore import QEvent, Qt, QThread, QTimer, QUrl, pyqtSignal
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
from TRACE.theme import current_theme as _ct

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
    chk.setStyleSheet(f"QCheckBox {{ color: {_ct().link}; }}")

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
        # Holds the currently-open async file picker (see
        # TRACE.gui._open_native_picker_async) so Python's GC doesn't
        # free it between open() and the user clicking Open / Cancel.
        # The scale-estimator wing-image picker sits next to napari
        # (the Custom Measurements tab embeds it via measurementMaker),
        # so once napari is loaded the nested-event-loop static-method
        # path is dead — async is mandatory here.
        self._active_picker = None
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
        # Theme live-switch wiring. _apply_theme_styles re-colors the
        # dependent-row hint labels stored in _pulse_dependencies — they
        # show during the pulse animation, so their build-time link
        # color would otherwise survive a theme switch.
        self._apply_theme_styles()
        from TRACE.theme import manager as _theme_manager

        _theme_manager().themeChanged.connect(self._apply_theme_styles)

    def _apply_theme_styles(self, *_args) -> None:
        """Re-color theme-dependent inline stylesheets."""
        t = _ct()
        for _parent, hint in self._pulse_dependencies.values():
            if hint is not None:
                hint.setStyleSheet(f"color: {t.link};")

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

        # Bottom row: Advanced Settings, then the Restore Defaults + wipe my
        # memories pair. Restore Defaults wipes only what this Settings tab owns
        # (scale, opacities, colors, preprocessing toggles, intermediates,
        # workers, synth-crossveins) — leaves model paths, input/output folders,
        # and advanced PipelineConfig fields alone. Advanced Settings opens the
        # modal PipelineConfigDialog.
        adv_row = QHBoxLayout()
        adv_row.addStretch(1)
        self.btn_advanced = QPushButton("Advanced Settings…")
        self.btn_advanced.setToolTip(
            "Open the advanced pipeline-settings dialog: per-model gate thresholds, "
            "skeletonization, bridging, tracing, intervein region detection."
        )
        self.btn_advanced.clicked.connect(self._window._open_settings_dialog)
        adv_row.addWidget(self.btn_advanced)

        # The two reset buttons sit together in one container so the TRACE
        # walkthrough can highlight them as a pair.
        self._reset_buttons_widget = QWidget()
        reset_row = QHBoxLayout(self._reset_buttons_widget)
        reset_row.setContentsMargins(0, 0, 0, 0)
        self.btn_restore_defaults = QPushButton("Restore Defaults")
        self.btn_restore_defaults.setToolTip(
            "Reset every control on this Settings tab to its factory default: scale, "
            "preprocessing toggles, intermediate outputs, overlay opacities and colors, "
            "and worker count. Does not touch model paths, input/output folders, or "
            "advanced PipelineConfig fields (those have their own reset)."
        )
        self.btn_restore_defaults.clicked.connect(self.restore_defaults)
        reset_row.addWidget(self.btn_restore_defaults)
        self.btn_wipe_memories = QPushButton("wipe my memories")
        self.btn_wipe_memories.setToolTip(
            "Clear every persisted setting — input/output folders, model paths, scale, "
            "pipeline config, custom distance pairs, workers warning suppression — and "
            "snap every widget back to the state a first-time user would see."
        )
        self.btn_wipe_memories.clicked.connect(self._window._reset_gui_to_defaults)
        reset_row.addWidget(self.btn_wipe_memories)
        adv_row.addWidget(self._reset_buttons_widget)
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
        self._scale_group = gb
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

        # Mirror of the main-window "Detect scale from image metadata" checkbox.
        # Kept in lockstep with the left-panel widget via _set_auto_detect_um_per_px
        # on TraceWindow so flipping either one updates the runtime config and
        # both checkboxes without re-firing each other's toggled signal.
        self.auto_detect_um_per_px_chk = QCheckBox("Detect scale from image metadata")
        self.auto_detect_um_per_px_chk.setToolTip(
            "When checked, each image's µm/px is read from its OWN metadata (TIFF "
            "XResolution + ResolutionUnit / OME-XML PhysicalSizeX) — measurements "
            "convert through that image's real scale rather than a shared value. "
            "The µm/px field above becomes the fallback used only when an image "
            "has no parseable metadata. If checked AND the µm/px field is empty "
            "AND any image lacks metadata, Run raises a pre-flight error so you "
            "don't discover the missing scale mid-batch. Mirrored to the main "
            "window's Scale group."
        )
        self.auto_detect_um_per_px_chk.toggled.connect(self._on_auto_detect_um_per_px_toggled)
        form.addRow(self.auto_detect_um_per_px_chk)

        parent_layout.addWidget(gb)

    def _build_optional_preprocessing_group(self, parent_layout: QVBoxLayout) -> None:
        gb = QGroupBox("Optional preprocessing steps")
        self._optional_preprocessing_group = gb
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
        self._output_options_group = gb
        v = QVBoxLayout(gb)

        # The checkboxes + spinboxes live in a child dialog ("More options…")
        # so the inline panel stays uncluttered. The widgets are still owned by
        # `self` so refresh_from_state() / restore_defaults() can address them
        # by name; we just put them inside a hidden QDialog instead of `v`.
        self._more_output_options_dialog = QDialog(self)
        self._more_output_options_dialog.setWindowTitle("More output options")
        self._more_output_options_dialog.setModal(False)
        dlg_v = QVBoxLayout(self._more_output_options_dialog)

        self.show_vein_tissue_chk = QCheckBox("Fill buffered vein tissue in overlay")
        self.show_vein_tissue_chk.setToolTip(
            "When off (default), the per-wing overlay only shows vein skeleton "
            "centerlines. When on, it also fills the buffered vein tissue polygons."
        )
        self.show_vein_tissue_chk.toggled.connect(self._on_show_vein_tissue_toggled)
        dlg_v.addWidget(self.show_vein_tissue_chk)

        self.show_color_key_chk = QCheckBox("Show vein color key in overlay")
        self.show_color_key_chk.setToolTip(
            "When on (default), the vein color legend is baked into the overlay's "
            "upper-left corner. Turn off for publication-style figures."
        )
        self.show_color_key_chk.toggled.connect(self._on_show_color_key_toggled)
        self.show_color_key_chk.toggled.connect(self._update_keys_and_labels_master_state)
        dlg_v.addWidget(self.show_color_key_chk)

        self.show_ectopic_labels_chk = QCheckBox("Show ectopic vein labels (EV1, EV2, …)")
        self.show_ectopic_labels_chk.setToolTip(
            "When on (default), each ectopic vein is annotated with its EV1/EV2… text. "
            "Turn off to keep the ectopic centerlines but hide the labels."
        )
        self.show_ectopic_labels_chk.toggled.connect(self._on_show_ectopic_labels_toggled)
        self.show_ectopic_labels_chk.toggled.connect(self._update_keys_and_labels_master_state)
        dlg_v.addWidget(self.show_ectopic_labels_chk)

        self.show_region_labels_chk = QCheckBox("Show intervein region labels")
        self.show_region_labels_chk.setToolTip(
            "When on (default), each intervein region is annotated with its name "
            "(plus [M]/[I] status suffixes). Turn off to keep the colored region "
            "fills without the text."
        )
        self.show_region_labels_chk.toggled.connect(self._on_show_region_labels_toggled)
        self.show_region_labels_chk.toggled.connect(self._update_keys_and_labels_master_state)
        dlg_v.addWidget(self.show_region_labels_chk)

        self.show_landmark_labels_chk = QCheckBox("Show landmark point labels")
        self.show_landmark_labels_chk.setToolTip(
            "When on (default), landmark points on the landmarks overlay AND the CV "
            "ratio overlay are annotated with their names (D-Tip, L1-Rs, ACV.p, "
            "PCV.a, …). Turn off to keep the colored dots without the text."
        )
        self.show_landmark_labels_chk.toggled.connect(self._on_show_landmark_labels_toggled)
        self.show_landmark_labels_chk.toggled.connect(self._update_keys_and_labels_master_state)
        dlg_v.addWidget(self.show_landmark_labels_chk)

        self.show_compartment_labels_chk = QCheckBox("Show A/P compartment labels")
        self.show_compartment_labels_chk.setToolTip(
            "When on (default), the anterior/posterior compartment overlay (when "
            "selected as an output) is annotated with ANT xx.x% / POST xx.x% text. "
            "Turn off to keep the tinted compartment fills without the percentage labels."
        )
        self.show_compartment_labels_chk.toggled.connect(self._on_show_compartment_labels_toggled)
        self.show_compartment_labels_chk.toggled.connect(self._update_keys_and_labels_master_state)
        dlg_v.addWidget(self.show_compartment_labels_chk)

        form = QFormLayout()
        self.ectopic_label_scale_spin = QDoubleSpinBox()
        self.ectopic_label_scale_spin.setRange(0.5, 10.0)
        self.ectopic_label_scale_spin.setDecimals(1)
        self.ectopic_label_scale_spin.setSingleStep(0.5)
        self.ectopic_label_scale_spin.setToolTip(
            "cv2 font scale for the EV1/EV2… ectopic-vein labels in the overlay. "
            "Default 1.0. Outline and fill thicknesses scale proportionally so the "
            "label stays legible at any size."
        )
        self.ectopic_label_scale_spin.valueChanged.connect(self._on_ectopic_label_scale_changed)
        form.addRow("Ectopic label size", self.ectopic_label_scale_spin)

        # Landmark point size. Multiplies the auto-derived dot radius / font /
        # halo used by draw_landmarks_on_image AND the CV-ratio overlay's
        # _draw_labeled_point, so every landmark-drawing output scales in
        # lockstep. 1.0 = defaults; the range mirrors the landmark_locator
        # GUI's on-screen slider so behaviour matches what users see there.
        self.landmark_size_spin = QDoubleSpinBox()
        self.landmark_size_spin.setRange(0.1, 5.0)
        self.landmark_size_spin.setDecimals(1)
        self.landmark_size_spin.setSingleStep(0.1)
        self.landmark_size_spin.setToolTip(
            "Scale factor for landmark points drawn on every landmark-carrying overlay "
            "(landmarks overlay + CV ratio overlay). Default 1.0. Applied to the "
            "dot radius, the point label font, and the halo thicknesses so the "
            "annotation stays readable at any size."
        )
        self.landmark_size_spin.valueChanged.connect(self._on_landmark_size_changed)
        form.addRow("Landmark point size", self.landmark_size_spin)

        self.vein_smooth_spin = QDoubleSpinBox()
        self.vein_smooth_spin.setRange(0.0, 50.0)
        self.vein_smooth_spin.setDecimals(1)
        self.vein_smooth_spin.setSingleStep(0.5)
        self.vein_smooth_spin.setToolTip(
            "Douglas-Peucker tolerance (px) for smoothing vein centerlines in the "
            "overlay. 0 (default) = draw the raw skeleton polyline; a few pixels is "
            "usually enough to remove staircasing. Affects only the rendered overlay, "
            "not the saved geometry."
        )
        self.vein_smooth_spin.valueChanged.connect(self._on_vein_smooth_changed)
        form.addRow("Vein smoothing (px)", self.vein_smooth_spin)

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
        dlg_v.addLayout(form)

        dlg_buttons = QDialogButtonBox(QDialogButtonBox.Close)
        dlg_buttons.rejected.connect(self._more_output_options_dialog.hide)
        dlg_v.addWidget(dlg_buttons)

        self.show_keys_and_labels_chk = QCheckBox("Show keys and labels")
        self.show_keys_and_labels_chk.setTristate(True)
        self.show_keys_and_labels_chk.setToolTip(
            "Master toggle for the four overlay labels: vein color key, ectopic "
            "vein labels, intervein region labels, and AP compartment labels. "
            "Use 'More options…' to toggle any of these individually."
        )
        self.show_keys_and_labels_chk.clicked.connect(self._on_keys_and_labels_master_clicked)
        v.addWidget(self.show_keys_and_labels_chk)

        self._more_output_options_btn = QPushButton("More options…")
        self._more_output_options_btn.setToolTip(
            "Open a window with the rest of the overlay rendering options "
            "(individual label toggles, label size, vein smoothing, layer opacities)."
        )
        self._more_output_options_btn.clicked.connect(self._show_more_output_options_dialog)
        v.addWidget(self._more_output_options_btn)

        vein_gb = QGroupBox("Vein colors")
        vein_gb.setToolTip("Click a swatch to choose a custom color for that vein in the overlay.")
        self._populate_color_grid(vein_gb, self._vein_color_state, self._vein_color_btns, kind="vein")
        v.addWidget(vein_gb)

        region_gb = QGroupBox("Intervein region colors")
        region_gb.setToolTip("Click a swatch to choose a custom color for that intervein region in the overlay.")
        self._populate_color_grid(region_gb, self._region_color_state, self._region_color_btns, kind="region")
        v.addWidget(region_gb)

        parent_layout.addWidget(gb)

    def _show_more_output_options_dialog(self) -> None:
        self._more_output_options_dialog.show()
        self._more_output_options_dialog.raise_()
        self._more_output_options_dialog.activateWindow()

    def _keys_and_labels_children(self) -> tuple[QCheckBox, ...]:
        return (
            self.show_color_key_chk,
            self.show_ectopic_labels_chk,
            self.show_region_labels_chk,
            self.show_landmark_labels_chk,
            self.show_compartment_labels_chk,
        )

    def _on_keys_and_labels_master_clicked(self, checked: bool) -> None:
        """User toggled the master — propagate to all four children.

        Each child's existing toggled handler runs and updates window state.
        Children also call back into _update_keys_and_labels_master_state(),
        which converges the master to Qt.Checked / Qt.Unchecked (never partial
        after a full set).
        """
        for chk in self._keys_and_labels_children():
            chk.setChecked(checked)

    def _update_keys_and_labels_master_state(self, _checked: bool = False) -> None:
        states = [chk.isChecked() for chk in self._keys_and_labels_children()]
        if all(states):
            new_state = Qt.Checked
        elif not any(states):
            new_state = Qt.Unchecked
        else:
            new_state = Qt.PartiallyChecked
        self.show_keys_and_labels_chk.blockSignals(True)
        try:
            self.show_keys_and_labels_chk.setCheckState(new_state)
        finally:
            self.show_keys_and_labels_chk.blockSignals(False)

    def _build_parallel_processing_group(self, parent_layout: QVBoxLayout) -> None:
        gb = QGroupBox("Parallel processing")
        self._parallel_processing_group = gb
        v = QVBoxLayout(gb)

        from TRACE.calibrate_widget import CalibrateWidget

        self._calibrate_widget = CalibrateWidget(self)
        # Register an explicit refresher BEFORE addWidget reparents the
        # widget into the QGroupBox — otherwise the widget's runtime
        # parent() call at click time would return the QGroupBox, which
        # doesn't have _refresh_calibrate_paths, and the widget would
        # silently keep the stale path stashed at construction. See the
        # 2026-08-19 follow-up on issue #35.
        self._calibrate_widget.set_refresher(self._refresh_calibrate_paths)
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
        recursive = False
        recursive_chk = getattr(self._window, "recursive_chk", None)
        if recursive_chk is not None:
            try:
                recursive = bool(recursive_chk.isChecked())
            except Exception:
                recursive = False
        self._calibrate_widget.set_paths(
            self._window.input_edit.text() if hasattr(self._window, "input_edit") else "",
            getattr(self._window, "_landmark_model_path", ""),
            getattr(self._window, "_segmentation_model_path", ""),
            recursive=recursive,
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
        hint.setStyleSheet(f"color: {_ct().link};")
        # Reserve room for the hint in the layout even while it's hidden,
        # so the row width accounts for its natural size from the start.
        # Without this the row commits to a narrower minimum during layout
        # (hidden widgets are excluded), and when the pulse then shows the
        # label its text gets clipped by the row's fixed width.
        _sp = hint.sizePolicy()
        _sp.setRetainSizeWhenHidden(True)
        hint.setSizePolicy(_sp)
        hint.setMinimumWidth(hint.sizeHint().width())
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

    def _on_auto_detect_um_per_px_toggled(self, checked: bool) -> None:
        # Route through the window so the left-panel checkbox and the runtime
        # config stay in lock-step. source="inline" tells the helper not to
        # echo back into this checkbox (which would re-fire toggled).
        self._window._set_auto_detect_um_per_px(bool(checked), source="inline")

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

    def _on_show_color_key_toggled(self, checked: bool) -> None:
        self._window._show_color_key = bool(checked)

    def _on_show_ectopic_labels_toggled(self, checked: bool) -> None:
        self._window._show_ectopic_labels = bool(checked)

    def _on_show_region_labels_toggled(self, checked: bool) -> None:
        self._window._show_region_labels = bool(checked)

    def _on_show_landmark_labels_toggled(self, checked: bool) -> None:
        self._window._show_landmark_labels = bool(checked)

    def _on_vein_smooth_changed(self, val: float) -> None:
        self._window._vein_simplify_tolerance_px = float(val)

    def _on_show_compartment_labels_toggled(self, checked: bool) -> None:
        self._window._show_compartment_labels = bool(checked)

    def _on_ectopic_label_scale_changed(self, val: float) -> None:
        self._window._ectopic_label_font_scale = float(val)

    def _on_landmark_size_changed(self, val: float) -> None:
        self._window._landmark_size_scale = float(val)

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
        self._window._show_color_key = True
        self._window._show_ectopic_labels = True
        self._window._show_region_labels = True
        self._window._show_landmark_labels = True
        self._window._vein_simplify_tolerance_px = 0.0
        self._window._ectopic_label_font_scale = 1.0
        self._window._landmark_size_scale = 1.0
        self._window._show_compartment_labels = True
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
            self.auto_detect_um_per_px_chk,
            self.wing_enable_chk,
            self.do_rotation_chk,
            self.rotation_mirror_correct_chk,
            self.synthesize_crossveins_chk,
            self.show_vein_tissue_chk,
            self.show_color_key_chk,
            self.show_ectopic_labels_chk,
            self.show_region_labels_chk,
            self.show_landmark_labels_chk,
            self.show_compartment_labels_chk,
            self.show_keys_and_labels_chk,
            self.ectopic_label_scale_spin,
            self.landmark_size_spin,
            self.vein_smooth_spin,
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
            self.auto_detect_um_per_px_chk.setChecked(bool(getattr(cfg, "auto_detect_um_per_px", False)))
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
            self.show_color_key_chk.setChecked(bool(self._window._show_color_key))
            self.show_ectopic_labels_chk.setChecked(bool(self._window._show_ectopic_labels))
            self.show_region_labels_chk.setChecked(bool(self._window._show_region_labels))
            self.show_landmark_labels_chk.setChecked(
                bool(getattr(self._window, "_show_landmark_labels", True))
            )
            self.show_compartment_labels_chk.setChecked(bool(self._window._show_compartment_labels))
            self._update_keys_and_labels_master_state()
            self.ectopic_label_scale_spin.setValue(float(self._window._ectopic_label_font_scale))
            self.landmark_size_spin.setValue(float(getattr(self._window, "_landmark_size_scale", 1.0)))
            self.vein_smooth_spin.setValue(float(self._window._vein_simplify_tolerance_px))
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
            QMessageBox.critical(
                self,
                "No landmark model",
                "Set the Landmark model in Settings → Models first — the estimator "
                "needs it to detect the L3 distal end and the L1-Rs junction.",
            )
            return
        if not self._confirm_scale_estimator_caveats():
            return
        seed = self._window.input_edit.text() or lm_path
        from TRACE.gui import _open_native_picker_async

        # Bind ``lm_path`` and ``host_dialog`` into the callback —
        # they're needed for the post-pick estimator run and the
        # caller's optional auto-accept of the wrapping dialog.
        _open_native_picker_async(
            self,
            "Select a wing image",
            self._picker_initial_path(seed),
            lambda image_path: self._on_scale_estimator_image_picked(image_path, lm_path, host_dialog),
            name_filter="Images (*.tif *.tiff *.png *.jpg *.jpeg *.bmp *.psd *.ome.tif);;All Files (*)",
            last_dir_key="scale_estimator_image",
        )

    def _on_scale_estimator_image_picked(self, image_path: str, lm_path: str, host_dialog: QDialog | None) -> None:
        if not image_path:
            return
        reference_um = float(self._scale_ref_um_spin.value())
        QApplication.setOverrideCursor(Qt.WaitCursor)
        # Route scale_estimator's logger to the GUI log panel for the
        # duration of this synchronous call. _CAPTURED_LOGGERS attaches
        # forwarders only when a TraceWorker is running; this interactive
        # scale-estimation runs on the GUI thread with no worker active,
        # so without this scope-local handler the "scaleEstimator: <img>
        # → x µm/px (L3 distal end ↔ L1-Rs junction = …)" info line never
        # surfaces.
        import logging as _logging

        scale_logger = _logging.getLogger("scale_estimator")
        prev_level = scale_logger.level

        class _DirectLogHandler(_logging.Handler):
            def __init__(self, log_method):
                super().__init__()
                self._log = log_method
                self.setFormatter(_logging.Formatter("%(name)s %(levelname)s: %(message)s"))

            def emit(self, record):
                try:
                    self._log(self.format(record))
                except Exception:
                    pass

        log_handler = _DirectLogHandler(self._window._log)
        log_handler.setLevel(_logging.INFO)
        if scale_logger.level == _logging.NOTSET or scale_logger.level > _logging.INFO:
            scale_logger.setLevel(_logging.INFO)
        scale_logger.addHandler(log_handler)
        try:
            from scale_estimator import ScaleEstimate, ScaleEstimationError, estimate_um_per_px

            estimate: ScaleEstimate = estimate_um_per_px(
                Path(image_path),
                Path(lm_path),
                reference_distance_um=reference_um,
            )
        except ScaleEstimationError as exc:
            QApplication.restoreOverrideCursor()
            QMessageBox.critical(self, "Estimate failed", str(exc))
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
            scale_logger.removeHandler(log_handler)
            if prev_level != scale_logger.level and prev_level == _logging.NOTSET:
                scale_logger.setLevel(_logging.NOTSET)

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
            QMessageBox.critical(
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
            QMessageBox.critical(
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
            # Honor the main-window "Include subfolders" checkbox so the
            # estimator sees the same images the user does. Previously
            # hardcoded recursive=False, which surfaced "No images
            # found" on any batch where the user pointed at a parent
            # whose images live in subfolders even with "Include
            # subfolders" checked (reported 2026-08-19).
            recursive = False
            recursive_chk = getattr(self._window, "recursive_chk", None)
            if recursive_chk is not None:
                try:
                    recursive = bool(recursive_chk.isChecked())
                except Exception:
                    recursive = False
            discovered = discover_images(input_path, recursive=recursive)
            if not discovered:
                QMessageBox.critical(
                    self,
                    "No images found",
                    f"No supported image files in:\n{input_path}"
                    + ("" if recursive else "\n\n(Check 'Include subfolders' on the main window if your images live in subfolders.)"),
                )
                return
            image_paths = discovered[: self._SCALE_ESTIMATOR_MAX_IMAGES]
        else:
            QMessageBox.critical(
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
            QMessageBox.critical(self, "Estimate failed", str(exc))
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
        # Long-lived labels stored on self for live theme re-styling.
        # Set inside _build_ui depending on which branch (info-only,
        # missing-import error, napari-construction error) the panel
        # took at startup.
        self._info_label: Optional[QLabel] = None
        self._import_err_label: Optional[QLabel] = None
        self._napari_err_label: Optional[QLabel] = None
        self._build_ui()
        self._apply_theme_styles()
        from TRACE.theme import manager as _theme_manager

        _theme_manager().themeChanged.connect(self._apply_theme_styles)

    def _apply_theme_styles(self, *_args) -> None:
        """Re-color the info / error labels for the active theme.

        Also reloads the cartoon wing variant when the picker is
        currently displaying the cartoon (signalled by an empty
        image_edit). The bundled wing_cartoon.png is white-on-
        transparent — readable on dark backgrounds, near-invisible on
        light. We invert pixel colors (alpha preserved) to a cached
        ``wing_cartoon_inverted.png`` for light mode.
        """
        t = _ct()
        if self._info_label is not None:
            self._info_label.setStyleSheet(f"color: {t.text_muted};")
        for err_label in (self._import_err_label, self._napari_err_label):
            if err_label is not None:
                err_label.setStyleSheet(f"color: {t.error_text}; padding: 12px;")
        self._reload_cartoon_if_displayed()

    def _cartoon_image_for_theme(self) -> Path:
        """Path to the cartoon-wing PNG variant matching the active theme.

        Dark mode → the bundled white-line PNG. Light mode → an inverted
        (black-line) variant generated on first use and cached to the
        user's writable cache dir. Falls back to the bundled original on
        any error (a black-on-light cartoon misses an inversion is uglier
        than a white-on-light cartoon, but losing the cartoon entirely is
        worse).
        """
        from TRACE.theme import current_theme

        if current_theme().name == "dark":
            return self._CARTOON_IMAGE
        inverted = self._ensure_inverted_cartoon()
        return inverted if inverted is not None else self._CARTOON_IMAGE

    def _ensure_inverted_cartoon(self) -> Optional[Path]:
        """Generate (once) or return the cached inverted cartoon PNG."""
        from PyQt5.QtCore import QStandardPaths
        from PyQt5.QtGui import QImage

        cache_base = QStandardPaths.writableLocation(QStandardPaths.CacheLocation)
        cache_dir = Path(cache_base) if cache_base else Path.home() / ".cache" / "TRACE"
        cache_dir = cache_dir / "asset_cache"
        inverted_path = cache_dir / "wing_cartoon_inverted.png"
        # Invalidate the cache if the source PNG is newer than the
        # cached variant — handles the case where a new release ships
        # a different cartoon image but the user still has an old cache.
        if inverted_path.is_file():
            try:
                if inverted_path.stat().st_mtime >= self._CARTOON_IMAGE.stat().st_mtime:
                    return inverted_path
            except OSError:
                return inverted_path
        if not self._CARTOON_IMAGE.is_file():
            return None
        img = QImage(str(self._CARTOON_IMAGE))
        if img.isNull():
            return None
        # InvertRgb flips R/G/B in place but leaves alpha untouched —
        # exactly what we want for a white-stroke-on-transparent PNG.
        img.invertPixels(QImage.InvertRgb)
        try:
            cache_dir.mkdir(parents=True, exist_ok=True)
            if img.save(str(inverted_path), "PNG"):
                return inverted_path
        except OSError:
            return None
        return None

    def _reload_cartoon_if_displayed(self) -> None:
        """Re-load the cartoon at theme-switch IFF it's currently showing.

        Signal: the picker's image_edit text is empty. This convention is
        established by _restore_cartoon (which explicitly clears the
        path fields before loading the cartoon) and by the auto-load
        branches in _build_ui / _rebuild_picker_for_window (which load
        the cartoon only when no saved user paths were provided).

        Loading a user image via the picker UI populates image_edit
        with the picked path, so an empty edit reliably means "we're
        on the cartoon."
        """
        if self._picker is None:
            return
        try:
            cur_text = self._picker._image_edit.text()
        except Exception:
            return
        if cur_text:
            return  # user image is loaded
        if not self._CARTOON_IMAGE.is_file() or not self._CARTOON_LANDMARKS.is_file():
            return
        self._picker.load_into_viewer(self._cartoon_image_for_theme(), self._CARTOON_LANDMARKS)

    def _resolve_paths(self) -> tuple[str, str]:
        """Return (image, landmarks) paths to seed the picker.

        Returns user-saved paths from QSettings, or empty strings when
        nothing is configured. The cartoon-wing default is NOT auto-loaded
        here — running LandmarkLocator on a cartoon drawing produces
        nonsense, and the cartoon path leaking in as a pre-fill would also
        block the auto-detect branch on Load. The user invokes the cartoon
        explicitly via the "Restore cartoon wing" button.
        """
        return (self._window._distance_sample_image or "", self._window._distance_sample_landmarks or "")

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(6)

        info = QLabel(
            "Configure custom straight-line measurements between any two landmark points. "
            "When Custom measurements is selected as an output, your measurements will be "
            "applied to every wing in the run (measured in both px and µm) and will appear "
            "in the final measurements CSV."
        )
        info.setWordWrap(True)
        info.setStyleSheet(f"color: {_ct().text_muted};")
        self._info_label = info
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
            err.setStyleSheet(f"color: {_ct().error_text}; padding: 12px;")
            self._import_err_label = err
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
                landmarks_generator=self._generate_landmarks_for_image,
                show_landmarks_picker=False,
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
            err.setStyleSheet(f"color: {_ct().error_text}; padding: 12px;")
            self._napari_err_label = err
            layout.addWidget(err)
            layout.addStretch()
            self._picker = None
            self._import_error = True
            return
        self._picker.pairs_changed.connect(self._on_pairs_changed)
        layout.addWidget(self._picker, stretch=1)

        # "Restore cartoon wing" — slotted into the picker's source area
        # via add_source_action so it sits directly below the Load button
        # (rather than below the napari canvas), keeping the related
        # actions grouped together.
        self._btn_restore_cartoon = QPushButton("Restore cartoon wing")
        self._btn_restore_cartoon.setToolTip(
            "Load the bundled cartoon wing + its hand-curated landmarks into the viewer. "
            "Useful for experimenting without running LandmarkLocator on a real wing."
        )
        self._btn_restore_cartoon.clicked.connect(self._restore_cartoon)
        self._picker.add_source_action(self._btn_restore_cartoon)

        # Auto-load behavior:
        #   - Saved real-wing paths from a prior session → restore them
        #     normally (fields + viewer both populated via load_initial).
        #   - Nothing saved → show the bundled cartoon in the viewer with
        #     the path fields left blank. load_into_viewer bypasses the
        #     edits so the next Browse + Load is treated as a fresh user
        #     pick (auto-detect can fire) rather than as a re-load of a
        #     pre-filled cartoon path.
        if img_path and lm_path and Path(img_path).is_file() and Path(lm_path).is_file():
            QTimer.singleShot(0, self._picker.load_initial)
        elif self._CARTOON_IMAGE.is_file() and self._CARTOON_LANDMARKS.is_file():
            QTimer.singleShot(
                0, lambda: self._picker.load_into_viewer(self._cartoon_image_for_theme(), self._CARTOON_LANDMARKS)
            )

    def _restore_cartoon(self) -> None:
        """Load the bundled cartoon wing into the viewer; clear the path fields."""
        if self._picker is None:
            return
        if not self._CARTOON_IMAGE.is_file() or not self._CARTOON_LANDMARKS.is_file():
            QMessageBox.critical(
                self,
                "Cartoon wing missing",
                "The bundled cartoon-wing files weren't found in this install:\n"
                f"  {self._CARTOON_IMAGE}\n  {self._CARTOON_LANDMARKS}",
            )
            return
        self._picker.set_image_path("")
        self._picker.set_landmarks_path("")
        self._picker.load_into_viewer(self._cartoon_image_for_theme(), self._CARTOON_LANDMARKS)
        # Forget any previously-saved user image so next session also
        # opens on the cartoon instead of restoring the prior pick.
        self._window._distance_sample_image = ""
        self._window._distance_sample_landmarks = ""

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
            landmarks_generator=self._generate_landmarks_for_image,
            show_landmarks_picker=False,
        )
        self._picker.pairs_changed.connect(self._on_pairs_changed)
        layout.addWidget(self._picker, stretch=1)
        # Re-add the Restore cartoon wing button into the new picker's
        # source area. The previous instance was a child of the deleted
        # picker and went away with it, so a fresh QPushButton is needed.
        self._btn_restore_cartoon = QPushButton("Restore cartoon wing")
        self._btn_restore_cartoon.setToolTip(
            "Load the bundled cartoon wing + its hand-curated landmarks into the viewer. "
            "Useful for experimenting without running LandmarkLocator on a real wing."
        )
        self._btn_restore_cartoon.clicked.connect(self._restore_cartoon)
        self._picker.add_source_action(self._btn_restore_cartoon)
        if img_path and lm_path and Path(img_path).is_file() and Path(lm_path).is_file():
            QTimer.singleShot(0, self._picker.load_initial)
        elif self._CARTOON_IMAGE.is_file() and self._CARTOON_LANDMARKS.is_file():
            QTimer.singleShot(
                0, lambda: self._picker.load_into_viewer(self._cartoon_image_for_theme(), self._CARTOON_LANDMARKS)
            )

    def _generate_landmarks_for_image(self, image_path: Path, *, disable_gates: bool = False) -> Path:
        """Run LandmarkLocator on `image_path` and write a *_landmarks.geojson.

        Used as LandmarkPickerWidget.landmarks_generator: lets the user pick
        only a sample image in the Custom Measurements tab and have its
        landmarks detected automatically instead of having to also pick a
        matching GeoJSON. Returns the path to the freshly-written file.

        ``disable_gates``: when True, every confidence gate is turned off so the
        model's best-guess landmarks are returned even on a borderline wing
        instead of raising ``LowConfidenceLandmarkError``. The landmark
        inspector uses this — an image you open to hand-correct is exactly the
        one likely to fail the gate, and you still need points to drag.
        """
        lm_path = (getattr(self._window, "_landmark_model_path", "") or "").strip()
        if not lm_path:
            raise RuntimeError(
                "No landmark model is configured. Set one in Settings → Models, "
                "or pick a *_landmarks.geojson file manually."
            )

        from landmark_locator import make_predictor
        from landmark_locator.data.psd_loader import imread_any
        from landmark_locator.scripts.predict import _result_to_geojson

        image = imread_any(image_path)
        if image is None:
            raise IOError(f"Could not load image: {image_path}")
        # Disabling each metric's ``enabled`` flag is enough — _gate_landmark
        # skips disabled metrics, so no landmark is rejected and no core
        # failure is raised. (Same mechanism as "Rerun failed, no gate aborts".)
        confidence_override = (
            {metric: {"enabled": False} for metric in ("peak", "sharpness", "second_peak_ratio")}
            if disable_gates
            else None
        )
        predictor = make_predictor(Path(lm_path), confidence_override=confidence_override)
        result = predictor.predict(image, include_unreliable=True)
        internal_to_geojson = {v: k for k, v in (predictor.geojson_to_landmark or {}).items()}
        geojson = _result_to_geojson(result, internal_to_geojson)

        # Drop the GeoJSON next to the source image when its directory is
        # writable, so the user can re-load the same wing later without
        # regenerating. Fall back to the system temp dir if not writable
        # (e.g. CD image, read-only network share).
        target = Path(image_path).with_name(f"{Path(image_path).stem}_landmarks.geojson")
        try:
            target.write_text(json.dumps(geojson, indent=2), encoding="utf-8")
        except OSError:
            import tempfile

            fd, tmp_str = tempfile.mkstemp(suffix="_landmarks.geojson", prefix=f"{Path(image_path).stem}_")
            os.close(fd)
            target = Path(tmp_str)
            target.write_text(json.dumps(geojson, indent=2), encoding="utf-8")
        return target


# ---------------------------------------------------------------------------
# InlineHelpPanel
# ---------------------------------------------------------------------------


def _version_is_newer(candidate: str, installed: str) -> bool:
    """Return True iff ``candidate`` is strictly newer than ``installed``.

    Parses semver-ish dotted version strings as tuples of ints and
    compares element-wise (so "0.2.0" > "0.1.44" — index 1 wins before
    index 2 is consulted). On any parse failure, falls back to "not
    newer" — preferring to under-announce updates over offering the
    user a downgrade. The empty / "unknown" cases both fall through
    to that conservative branch.
    """
    if not candidate or not installed:
        return False
    try:
        cand_parts = [int(x) for x in candidate.split(".")]
        inst_parts = [int(x) for x in installed.split(".")]
    except (ValueError, AttributeError):
        return False
    return cand_parts > inst_parts


class _UpdateCheckThread(QThread):
    """Runs the GitHub /releases/latest query off the GUI thread.

    The synchronous version of the update check froze the window for the
    duration of the HTTP round-trip — fine on a fast connection, bad on
    a slow / flaky one. This thread does the network + JSON parse in the
    background and emits a single ``result`` payload back to the GUI
    thread for rendering.

    Lifetime is managed by the caller: store the instance on the panel,
    wire ``finished`` to ``deleteLater``, and guard against starting a
    second thread while one is in-flight.
    """

    result = pyqtSignal(dict)

    def __init__(self, api_url: str, parent=None):
        super().__init__(parent)
        self._api_url = api_url

    def run(self):
        try:
            payload = InlineHelpPanel._fetch_latest_release_info(self._api_url)
            self.result.emit({"ok": True, **payload})
        except Exception as exc:  # noqa: BLE001
            # Caller decides whether to surface the error (manual click)
            # or swallow it (auto-launch check on a flaky network).
            self.result.emit({"ok": False, "error": str(exc)})


class InlineHelpPanel(QWidget):
    """Help tab — clickable links to the TRACE README and project repo."""

    def __init__(self, window: "TraceWindow"):
        super().__init__()
        self._window = window
        self._build_ui()

    def _build_ui(self) -> None:
        # Top-level: a horizontal split with the text/controls column on
        # the left (takes all leftover width) and the fly icon pinned to
        # the top-right. The icon column has stretch=0 and the icon is
        # a fixed-size QLabel, so the text column always reflows into
        # the remaining width — the icon can never overlap or push into
        # text content when the window is resized.
        outer = QHBoxLayout(self)
        outer.setContentsMargins(16, 16, 16, 16)
        outer.setSpacing(16)

        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(12)
        outer.addLayout(layout, stretch=1)

        # Help-tab fly icon — top-right corner at the same baseline as
        # the Documentation heading. We ship one SVG (flicon.svg, the
        # black-stroke variant the README also embeds) and invert the
        # rendered pixmap at runtime for dark mode so we don't carry
        # two source assets. _render_flicon re-runs on every theme
        # change so the icon hot-swaps in sync with the rest of the UI.
        self._flicon_label = QLabel()
        self._flicon_label.setFixedSize(96, 96)  # placeholder until first render
        icon_col = QVBoxLayout()
        icon_col.setContentsMargins(0, 0, 0, 0)
        icon_col.setSpacing(0)
        icon_col.addWidget(self._flicon_label)
        icon_col.addStretch(1)  # pin to top
        outer.addLayout(icon_col)
        self._render_flicon()
        from TRACE.theme import manager as _theme_manager

        _theme_manager().themeChanged.connect(self._render_flicon)

        title = QLabel("Documentation")
        title_font = QFont(title.font())
        title_font.setPointSize(title_font.pointSize() + 2)
        title_font.setBold(True)
        title.setFont(title_font)
        layout.addWidget(title)

        # Static labels are stored on self so _apply_theme_styles can
        # re-set their HTML when the user switches theme — colors like
        # text_muted and error_text resolve to very different shades per
        # theme (e.g. #aaa in dark vs #666 in light), so a build-time
        # capture would leave low-contrast text after a live switch.
        self._readme_path = Path(__file__).resolve().parent / "README.md"
        self._doc_link = None
        self._doc_path_label = None
        self._doc_missing_label = None
        if self._readme_path.is_file():
            self._doc_link = QLabel("")
            self._doc_link.setOpenExternalLinks(True)
            self._doc_link.setTextInteractionFlags(Qt.TextBrowserInteraction)
            layout.addWidget(self._doc_link)

            self._doc_path_label = QLabel("")
            self._doc_path_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
            self._doc_path_label.setWordWrap(True)
            layout.addWidget(self._doc_path_label)
        else:
            self._doc_missing_label = QLabel("")
            self._doc_missing_label.setWordWrap(True)
            layout.addWidget(self._doc_missing_label)

        # GitHub repo link — for source, issues, and contributions.
        self._github_link = QLabel("")
        self._github_link.setOpenExternalLinks(True)
        self._github_link.setTextInteractionFlags(Qt.TextBrowserInteraction)
        layout.addWidget(self._github_link)

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

        self._tour_blurb = QLabel(
            "Re-run the guided tour of the main controls — the same one that "
            "appeared the first time you opened TRACE."
        )
        self._tour_blurb.setWordWrap(True)
        layout.addWidget(self._tour_blurb)

        replay_row = QHBoxLayout()
        self.btn_replay_walkthrough = QPushButton("Replay walkthrough")
        self.btn_replay_walkthrough.setToolTip(
            "Restart the first-launch walkthrough. Step through with Next / Previous, or skip with Esc."
        )
        self.btn_replay_walkthrough.clicked.connect(self._window._show_walkthrough)
        replay_row.addWidget(self.btn_replay_walkthrough)
        replay_row.addStretch(1)
        layout.addLayout(replay_row)

        # Appearance — Follow system / Light / Dark theme picker. Stores
        # the choice in QSettings (via TRACE.theme.ThemeManager) and emits
        # themeChanged so every long-lived widget hot-swaps without a
        # restart. Sits between Walkthrough and Update because it pairs
        # naturally with the rest of the "user preferences" section.
        layout.addSpacing(12)
        appearance_title = QLabel("Appearance")
        appearance_title_font = QFont(appearance_title.font())
        appearance_title_font.setPointSize(appearance_title_font.pointSize() + 2)
        appearance_title_font.setBold(True)
        appearance_title.setFont(appearance_title_font)
        layout.addWidget(appearance_title)

        from TRACE.theme import ThemePreference
        from TRACE.theme import manager as _theme_manager

        theme_row = QHBoxLayout()
        theme_row.setContentsMargins(0, 0, 0, 0)
        theme_row.addWidget(QLabel("Theme:"))
        self._theme_combo = QComboBox()
        self._theme_combo.addItem("Follow system", ThemePreference.SYSTEM.value)
        self._theme_combo.addItem("Light", ThemePreference.LIGHT.value)
        self._theme_combo.addItem("Dark", ThemePreference.DARK.value)
        self._theme_combo.setToolTip(
            "Switch the TRACE color scheme. System matches the operating system's "
            "dark / light setting (and updates live when it changes). Light and "
            "Dark are explicit picks that override the system setting."
        )
        _mgr = _theme_manager()
        for idx in range(self._theme_combo.count()):
            if self._theme_combo.itemData(idx) == _mgr.preference.value:
                self._theme_combo.setCurrentIndex(idx)
                break
        self._theme_combo.currentIndexChanged.connect(self._on_theme_changed)
        theme_row.addWidget(self._theme_combo)
        theme_row.addStretch(1)
        layout.addLayout(theme_row)

        # App icon picker. Independent of the UI theme so a user on a
        # dark OS can still get a light TRACE icon if they prefer (e.g.
        # against a dark taskbar background that already provides
        # contrast against light icons).
        from TRACE._app_icon import IconPreference

        icon_row = QHBoxLayout()
        icon_row.setContentsMargins(0, 0, 0, 0)
        icon_row.addWidget(QLabel("App icon:"))
        self._icon_combo = QComboBox()
        self._icon_combo.addItem("Follow system", IconPreference.SYSTEM.value)
        self._icon_combo.addItem("Light", IconPreference.LIGHT.value)
        self._icon_combo.addItem("Dark", IconPreference.DARK.value)
        self._icon_combo.setToolTip(
            "Choose the TRACE app-icon color. System uses a dark-friendly icon "
            "when your OS is in dark mode and a light-friendly one otherwise. "
            "Light and Dark let you override the OS pick."
        )
        # Seed the combo with the saved preference.
        saved_pref = self._window.settings.value("app_icon/preference", IconPreference.SYSTEM.value, type=str)
        for idx in range(self._icon_combo.count()):
            if self._icon_combo.itemData(idx) == saved_pref:
                self._icon_combo.setCurrentIndex(idx)
                break
        self._icon_combo.currentIndexChanged.connect(self._on_icon_pref_changed)
        icon_row.addWidget(self._icon_combo)
        icon_row.addStretch(1)
        layout.addLayout(icon_row)

        # Desktop shortcut — Windows-only (uses pywin32's WScript Shell
        # COM to create a .lnk). Hidden on macOS / Linux / dev-mode so
        # we don't promise something we can't deliver.
        if sys.platform == "win32" and getattr(sys, "frozen", False):
            shortcut_row = QHBoxLayout()
            shortcut_row.setContentsMargins(0, 0, 0, 0)
            self.btn_desktop_shortcut = QPushButton("Add desktop shortcut")
            self.btn_desktop_shortcut.setToolTip(
                "Create a TRACE shortcut on your Windows Desktop pointing at the "
                "installed TRACE.exe. Useful if you didn't tick the Desktop-shortcut "
                "option during install or want to put it back after deleting it."
            )
            self.btn_desktop_shortcut.clicked.connect(self._create_desktop_shortcut)
            shortcut_row.addWidget(self.btn_desktop_shortcut)
            shortcut_row.addStretch(1)
            layout.addLayout(shortcut_row)

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

        self._version_label = QLabel("")
        layout.addWidget(self._version_label)

        self._update_blurb = QLabel(
            "Get the latest TRACE installer from the project's Releases page. "
            "Running the new installer upgrades your existing TRACE in place — "
            "no need to uninstall first. Your settings and downloaded models "
            "are preserved."
        )
        self._update_blurb.setWordWrap(True)
        layout.addWidget(self._update_blurb)

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

        # Auto-check opt-out. Default ON — the check is silent, throttled
        # to once per hour, and only surfaces UI when an update is found.
        self.chk_auto_check_updates = QCheckBox("Auto-check for updates on launch")
        self.chk_auto_check_updates.setToolTip(
            "When checked, TRACE silently queries the Releases page on launch "
            "(throttled to once per hour) and flags the Help tab if an update is available."
        )
        self.chk_auto_check_updates.setChecked(
            self._window.settings.value("auto_update_check_enabled", True, type=bool)
        )
        self.chk_auto_check_updates.toggled.connect(
            lambda on: self._window.settings.setValue("auto_update_check_enabled", bool(on))
        )
        layout.addWidget(self.chk_auto_check_updates)

        # Populated by _check_for_updates when a newer release is found.
        self._latest_update_url: Optional[str] = None
        self._latest_update_size: Optional[int] = None
        self._latest_update_version: Optional[str] = None
        # When the "Update now" button in the launch-time notification
        # dialog is clicked BEFORE the auto-check has populated the
        # asset URL, set this True. The next _apply_update_check_result
        # consumes it and immediately fires _install_update() instead
        # of just updating the Help-tab UI.
        self._install_after_next_check: bool = False
        # In-flight thread guard so rapid clicks (or an auto-check colliding
        # with a manual click) don't fire two concurrent requests.
        self._update_thread: Optional[_UpdateCheckThread] = None

        # --- Report a bug ---
        bug_row = QHBoxLayout()
        self.btn_report_bug = QPushButton("Report a bug…")
        self.btn_report_bug.setToolTip(
            "File a bug report. No GitHub account required — the report goes to "
            "a maintainer-controlled server that files it as a GitHub issue."
        )
        self.btn_report_bug.clicked.connect(self._open_report_bug_dialog)
        bug_row.addWidget(self.btn_report_bug)
        bug_row.addStretch(1)
        layout.addLayout(bug_row)

        layout.addStretch(1)

        self._footer = QLabel("")
        self._footer.setWordWrap(True)
        layout.addWidget(self._footer)

        # Initial render + wire the theme signal so static labels stay
        # in sync with the user's pick. _apply_theme_styles must come
        # after every label has been instantiated (we reference them
        # all by attribute name).
        self._apply_theme_styles()
        from TRACE.theme import manager as _theme_manager

        _theme_manager().themeChanged.connect(self._apply_theme_styles)

    # -----------------------------------------------------------------------
    # Update check
    # -----------------------------------------------------------------------
    _RELEASES_PAGE_URL = "https://github.com/alexmpdx/TRACE/releases"
    _LATEST_RELEASE_API = "https://api.github.com/repos/alexmpdx/TRACE/releases/latest"

    def _apply_theme_styles(self, *_args) -> None:
        """Re-render every static label's HTML / stylesheet from the
        active theme.

        Called once at panel construction and again on every
        ThemeManager.themeChanged so that text_muted / link / error_text
        colors (which differ substantially between themes — e.g.
        #aaaaaa in dark vs #666666 in light) follow the live switch
        instead of staying at their build-time captures.

        The dynamic ``_update_status_label`` is deliberately not
        rebuilt here — its text is rewritten on every auto/manual
        update check, so the next check refreshes it in the new theme.
        Touching it here would either need to track the last-rendered
        "state" or risk overwriting an in-flight "Checking…" message.
        """
        from TRACE.theme import current_theme

        t = current_theme()
        if self._doc_link is not None:
            url = QUrl.fromLocalFile(str(self._readme_path)).toString()
            self._doc_link.setText(f'<a href="{url}" style="color: {t.link};">Open README.md in your default app</a>')
        if self._doc_path_label is not None:
            self._doc_path_label.setText(
                f"<span style='color: {t.text_placeholder};'>Location:</span> {self._readme_path}"
            )
        if self._doc_missing_label is not None:
            self._doc_missing_label.setText(
                f"<span style='color: {t.error_text};'>README.md not found at:</span><br>{self._readme_path}"
            )
        self._github_link.setText(
            f'<a href="https://github.com/alexmpdx/TRACE" style="color: {t.link};">View TRACE on GitHub</a>'
        )
        self._tour_blurb.setStyleSheet(f"color: {t.text_muted};")
        try:
            from TRACE import __version__ as _trace_version
        except Exception:
            _trace_version = "unknown"
        self._version_label.setText(
            f"<span style='color: {t.text_muted};'>Installed version:</span> "
            f"<span style='color: {t.text};'>{_trace_version}</span>"
        )
        self._update_blurb.setStyleSheet(f"color: {t.text_muted};")
        self._footer.setText(
            f"<span style='color: {t.text_placeholder};'>For pipeline-internal docs, see the comments in "
            "<code>TRACE/pipeline.py</code> and <code>TRACE/gui.py</code>.</span>"
        )

    def _render_flicon(self, *_args) -> None:
        """Re-render the help-tab fly icon for the active theme.

        Renders flicon.svg (black strokes on transparent) into a
        QPixmap, then inverts the RGB channels for dark mode so the
        black strokes become white while the transparent background
        stays transparent. One source asset, both themes derived.
        Called once at panel construction and again on every theme
        switch via ThemeManager.themeChanged.
        """
        from TRACE.theme import current_theme

        if getattr(sys, "frozen", False) and hasattr(sys, "_MEIPASS"):
            svg_path = Path(sys._MEIPASS) / "TRACE" / "GUI_images" / "logo" / "flicon.svg"
        else:
            svg_path = Path(__file__).resolve().parent / "GUI_images" / "logo" / "flicon.svg"
        if not svg_path.is_file():
            # Missing asset shouldn't break the Help tab. Clear any
            # placeholder pixmap so the slot just renders empty.
            self._flicon_label.clear()
            return
        from PyQt5.QtGui import QImage, QPainter, QPixmap
        from PyQt5.QtSvg import QSvgRenderer

        renderer = QSvgRenderer(str(svg_path))
        default = renderer.defaultSize()
        icon_w = 96
        icon_h = max(1, int(icon_w * default.height() / max(1, default.width())))
        pixmap = QPixmap(icon_w, icon_h)
        pixmap.fill(Qt.transparent)
        painter = QPainter(pixmap)
        renderer.render(painter)
        painter.end()
        if current_theme().name == "dark":
            # InvertRgb flips R/G/B in place but leaves alpha untouched
            # — black strokes become white, transparent stays transparent.
            img = pixmap.toImage()
            img.invertPixels(QImage.InvertRgb)
            pixmap = QPixmap.fromImage(img)
        self._flicon_label.setPixmap(pixmap)
        self._flicon_label.setFixedSize(icon_w, icon_h)

    def _on_theme_changed(self, _idx: int) -> None:
        """User picked a new theme from the Help-tab combo — apply via ThemeManager."""
        from TRACE.theme import ThemePreference
        from TRACE.theme import manager as _theme_manager

        raw = self._theme_combo.currentData()
        try:
            pref = ThemePreference(str(raw))
        except ValueError:
            return
        _theme_manager().set_preference(pref)

    def _on_icon_pref_changed(self, _idx: int) -> None:
        """Persist + immediately re-apply the new app-icon variant.

        Re-applies on:
          - QApplication (window-list / Alt-Tab / Dock icon on macOS)
          - the main TRACE window (window title-bar icon)
        Per-window icons survive any later QApplication.setWindowIcon
        override (notably the one napari fires when LandmarkPickerWidget
        constructs its embedded viewer), so updating the main window is
        what makes the visible change in most cases.
        """
        raw = self._icon_combo.currentData()
        self._window.settings.setValue("app_icon/preference", str(raw))
        from PyQt5.QtWidgets import QApplication

        from TRACE._app_icon import ensure_app_icon_ico, make_app_icon

        icon = make_app_icon()
        if icon is not None:
            app = QApplication.instance()
            if app is not None:
                app.setWindowIcon(icon)
            self._window.setWindowIcon(icon)
        # Best-effort: regenerate the cached desktop-shortcut ICO so
        # an existing TRACE.lnk picks up the new variant the next time
        # Windows draws it. No-op if a shortcut hasn't been created yet
        # — the file just sits in the cache until needed.
        try:
            ensure_app_icon_ico()
        except Exception:
            pass

    def _create_desktop_shortcut(self) -> None:
        """Create a Windows .lnk shortcut on the user's Desktop.

        Only wired up on a frozen Windows build (button is hidden on
        macOS / Linux / dev mode). Uses pywin32's WScript.Shell COM
        adapter — already in the bundle's dependency tree — so no
        extra runtime install is needed.
        """
        from PyQt5.QtCore import QStandardPaths
        from PyQt5.QtWidgets import QMessageBox

        if sys.platform != "win32" or not getattr(sys, "frozen", False):
            QMessageBox.information(
                self,
                "Desktop shortcut",
                "Desktop-shortcut creation is currently supported only on the " "Windows installer build of TRACE.",
            )
            return
        target = Path(sys.executable)
        if not target.is_file():
            QMessageBox.critical(
                self,
                "Desktop shortcut",
                f"Couldn't locate TRACE.exe at {target}. The shortcut wasn't created.",
            )
            return
        desktop_str = QStandardPaths.writableLocation(QStandardPaths.DesktopLocation)
        if not desktop_str:
            QMessageBox.critical(
                self,
                "Desktop shortcut",
                "Couldn't find your Desktop folder. The shortcut wasn't created.",
            )
            return
        shortcut_path = Path(desktop_str) / "TRACE.lnk"
        try:
            # pywin32 is already a dependency (see requirements-windows.txt).
            from win32com.client import Dispatch  # type: ignore[import-not-found]

            from TRACE._app_icon import ensure_app_icon_ico

            shell = Dispatch("WScript.Shell")
            sc = shell.CreateShortcut(str(shortcut_path))
            sc.Targetpath = str(target)
            sc.WorkingDirectory = str(target.parent)
            # IconLocation prefers the cached ICO matching the user's
            # IconPreference. Falls back to TRACE.exe (which uses the
            # icon embedded by PyInstaller at build time) if ICO
            # generation failed for any reason.
            ico = ensure_app_icon_ico()
            sc.IconLocation = str(ico) if ico is not None else str(target)
            sc.save()
        except Exception as exc:  # noqa: BLE001
            QMessageBox.critical(
                self,
                "Desktop shortcut",
                f"Could not create the shortcut: {type(exc).__name__}: {exc}",
            )
            return
        QMessageBox.information(
            self,
            "Desktop shortcut",
            f"Created: {shortcut_path}",
        )

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

    def _open_report_bug_dialog(self) -> None:
        from TRACE.report_bug_dialog import ReportBugDialog

        dlg = ReportBugDialog(self._window)
        dlg.exec()

    @staticmethod
    def _fetch_latest_release_info(api_url: str) -> dict:
        """Pure network + parse helper. No UI access. Raises on failure.

        Returns a dict with keys: ``tag`` (raw GitHub tag, e.g.
        ``windows-v0.2.0``), ``latest_version`` (the bare semver with
        the ``windows-v`` / ``v`` prefix stripped, for direct comparison
        with ``TRACE.__version__``), ``html_url`` (the release page),
        ``asset_url`` (``TRACE-Setup.exe`` download URL, or None), and
        ``asset_size`` (bytes, or None).

        Lives as a static method so the worker thread can call it
        without holding a reference to a UI instance.
        """
        import json
        import urllib.request

        from TRACE.fetch_assets import make_ssl_context

        req = urllib.request.Request(api_url, headers={"User-Agent": "TRACE-update-check"})
        with urllib.request.urlopen(req, timeout=10, context=make_ssl_context()) as resp:
            data = json.load(resp)

        latest_tag = str(data.get("tag_name") or "")
        latest_version = latest_tag
        for prefix in ("windows-v", "v"):
            if latest_version.startswith(prefix):
                latest_version = latest_version[len(prefix) :]
                break

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

        return {
            "tag": latest_tag,
            "latest_version": latest_version,
            "html_url": str(data.get("html_url") or InlineHelpPanel._RELEASES_PAGE_URL),
            "asset_url": asset_url,
            "asset_size": asset_size,
        }

    def _check_for_updates(self, *, silent: bool = False) -> None:
        """Kick off the GitHub release-tag query on a background thread.

        ``silent=False`` (the manual-button path) shows a "Checking…" hint
        in the label and surfaces network/parse errors in red so the user
        knows their click did something. ``silent=True`` (the auto-launch
        path from TraceWindow._maybe_auto_check_updates) skips both —
        nothing visible happens unless an update is actually found.

        Concurrency: at most one check runs at a time. A second click
        while a check is in-flight is a no-op (the result slot will
        update the label when the first one lands).
        """
        # Defensive: any uncaught exception in this handler — including
        # the QThread.start() call, the lambda binding, or _ct() blowing
        # up on a missing theme token — would otherwise make the click
        # look like a no-op (no label change, no error dialog) in a
        # frozen build with no console. Surface it instead.
        import traceback as _tb

        from TRACE.startup_log import log as _slog

        try:
            # PyQt5 dangling-reference guard. After the previous thread
            # finished, deleteLater destroyed the C++ QThread but the
            # Python attribute kept pointing at the (now invalid) sip
            # wrapper. Calling isRunning() on a deleted C++ object
            # raises "RuntimeError: wrapped C/C++ object of type
            # _UpdateCheckThread has been deleted" — issue #16.
            # Treat the RuntimeError as "no live thread" and proceed
            # to start a fresh one. The new finished-slot below also
            # clears self._update_thread back to None so future clicks
            # never see a stale wrapper in the first place.
            try:
                prior_running = self._update_thread is not None and self._update_thread.isRunning()
            except RuntimeError:
                prior_running = False
                self._update_thread = None
            if prior_running:
                _slog(f"update_check: click ignored — previous check still running (silent={silent})")
                return
            _slog(f"update_check: starting (silent={silent}, api={self._LATEST_RELEASE_API})")
            if not silent:
                self._update_status_label.setText(
                    f"<span style='color: {_ct().text_placeholder};'>Checking for updates…</span>"
                )
            self._update_thread = _UpdateCheckThread(self._LATEST_RELEASE_API, parent=self)
            self._update_thread.result.connect(lambda payload: self._apply_update_check_result(payload, silent=silent))
            self._update_thread.finished.connect(self._update_thread.deleteLater)
            # Clear the Python-side reference once the thread finishes,
            # before deleteLater destroys the C++ object. Without this
            # the attribute keeps pointing at a sip wrapper whose C++
            # backing has been freed.
            self._update_thread.finished.connect(self._clear_update_thread_ref)
            self._update_thread.start()
            _slog("update_check: background thread started")
        except BaseException as exc:  # noqa: BLE001 — log absolutely everything
            tb_text = "".join(_tb.format_exception(type(exc), exc, exc.__traceback__))
            _slog(f"update_check: handler raised {type(exc).__name__}\n{tb_text}")
            if not silent:
                try:
                    self._update_status_label.setText(
                        f"<span style='color: {_ct().error_text};'>Update check failed to start: "
                        f"{type(exc).__name__}: {exc}</span>"
                    )
                except Exception:
                    pass

    def _clear_update_thread_ref(self) -> None:
        """Drop the Python reference to the finished _UpdateCheckThread.

        Without this, the attribute keeps pointing at a sip wrapper
        whose C++ backing is about to be freed by deleteLater, and the
        next click on Check-for-updates hits ``isRunning()`` on a dead
        object → RuntimeError (issue #16).
        """
        self._update_thread = None

    def _apply_update_check_result(self, payload: dict, *, silent: bool) -> None:
        """GUI-thread slot for _UpdateCheckThread.result. Updates the label,
        Install button, and the Help-tab attention indicator.

        Always stamps ``last_update_check_time`` so the hourly auto-throttle
        in _maybe_auto_check_updates works regardless of whether the check
        was manual or automatic.
        """
        import time

        from TRACE.startup_log import log as _slog

        _slog("update_check: result received (ok=%s, error=%s)" % (payload.get("ok"), payload.get("error", "")))

        self._window.settings.setValue("last_update_check_time", int(time.time()))

        if not payload.get("ok"):
            # Silent path: leave whatever label state was there before.
            # Better than flashing red on a flaky home Wi-Fi; if the prior
            # check succeeded, the "up to date" label remains visible.
            if not silent:
                err = payload.get("error", "")
                self._update_status_label.setText(
                    f"<span style='color: {_ct().error_text};'>Could not check for updates: {err}</span><br>"
                    f"<a href='{self._RELEASES_PAGE_URL}' style='color: {_ct().link};'>"
                    f"Open the Releases page manually</a>"
                )
            return

        try:
            from TRACE import __version__ as installed_version
        except Exception:
            installed_version = "unknown"

        latest_version = payload["latest_version"]
        if not latest_version:
            if not silent:
                self._update_status_label.setText(
                    f"<span style='color: {_ct().text_muted};'>No releases found on GitHub yet. "
                    f"<a href='{self._RELEASES_PAGE_URL}' style='color: {_ct().link};'>"
                    f"Check the Releases page</a>.</span>"
                )
            return

        # Decide which branch to take by comparing as semver-ish tuples
        # of integers, not by raw string equality. The user can be:
        #   (a) exactly matching the latest release → up to date,
        #   (b) behind it → genuine update available,
        #   (c) AHEAD of it (running a source / dev build whose version
        #       has been bumped locally but CI hasn't published the
        #       matching tag yet). String-equality treated (c) as
        #       "different = update available" and offered the user a
        #       downgrade, which is wrong; only (b) should surface the
        #       update UI.
        latest_is_newer = _version_is_newer(latest_version, installed_version)
        if not latest_is_newer:
            self._update_status_label.setText(
                f"<span style='color: {_ct().success_text};'>✓ You're up to date (installed: {installed_version}).</span>"
            )
            self.btn_install_update.setVisible(False)
            self._latest_update_url = None
            self._latest_update_size = None
            self._latest_update_version = None
            # An earlier check that flagged an update may have left the
            # Help tab badged AND cached its version in QSettings —
            # clear both so the badge doesn't come back on next launch.
            self._window.clear_update_available_indicator(clear_cache=True)
            return

        # Update available — stash for the Install Update button.
        self._latest_update_url = payload["asset_url"]
        self._latest_update_size = payload["asset_size"]
        self._latest_update_version = latest_version

        can_install_in_place = bool(payload["asset_url"]) and getattr(sys, "frozen", False) and sys.platform == "win32"
        if can_install_in_place:
            self.btn_install_update.setText(f"Install update {latest_version}")
            self.btn_install_update.setVisible(True)
            size_mb = (payload["asset_size"] or 0) // (1024 * 1024)
            size_blurb = f" ({size_mb} MB)" if size_mb else ""
            self._update_status_label.setText(
                f"<span style='color: {_ct().warning_text};'>Update available: "
                f"<b>{latest_version}</b> (you have {installed_version}).</span><br>"
                f"<span style='color: {_ct().text_muted};'>Click <b>Install update {latest_version}</b> "
                f"to download{size_blurb} and launch the new installer.</span>"
            )
        else:
            self.btn_install_update.setVisible(False)
            self._update_status_label.setText(
                f"<span style='color: {_ct().warning_text};'>A different version is available: "
                f"<b>{latest_version}</b> (you have {installed_version}).</span><br>"
                f"<a href='{payload['html_url']}' style='color: {_ct().link};'>"
                f"Open the release page and download TRACE-Setup.exe</a>"
            )
        self._window.show_update_available_indicator(latest_version)
        # Fresh-result paths get the centered notification dialog. The
        # cached-restore path on launch deliberately doesn't (the user
        # already saw + dismissed it last session; the badge is enough).
        self._window.show_update_available_dialog(latest_version)
        # If the user clicked "Update now" in the launch-time
        # notification BEFORE this check completed, the asset URL
        # wasn't populated yet — _install_update would have early-
        # returned. Now that we have the URL, fire the install.
        if self._install_after_next_check and can_install_in_place:
            self._install_after_next_check = False
            self._install_update()
        else:
            # If the chain was requested but in-place install isn't
            # available (non-Windows / non-frozen), clear the flag —
            # the user gets the release-page link instead.
            self._install_after_next_check = False

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
                f"<span style='color: {_ct().error_text};'>No installer URL — run Check for updates first.</span>"
            )
            return

        import os
        import tempfile

        # Use a unique filename per attempt instead of a fixed
        # %TEMP%\TRACE-Setup.exe. A stale handle on the fixed name from a
        # prior attempt (antivirus mid-scan, an installer process still
        # exiting, an aborted earlier download) locks the path and the
        # next open(dst, "wb") fails with [Errno 13] Permission denied.
        # mkstemp picks a guaranteed-fresh name; we close its fd straight
        # away so the subsequent open() owns the handle.
        fd, _dst_str = tempfile.mkstemp(suffix="-TRACE-Setup.exe")
        os.close(fd)
        dst = Path(_dst_str)

        dlg = QProgressDialog(
            f"Downloading TRACE {version}…",
            "Cancel",
            0,
            100,
            self,
        )
        # napari clobbers QApplication.windowIcon when its embedded viewer
        # initializes, so by the time the user clicks Install Update the
        # app-wide icon is already the napari logo. Pin the TRACE icon
        # directly on this dialog so its title bar / taskbar entry stays
        # branded TRACE regardless of any prior setWindowIcon calls.
        from TRACE._app_icon import make_app_icon

        _icon = make_app_icon()
        if _icon is not None:
            dlg.setWindowIcon(_icon)
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

            from TRACE.fetch_assets import make_ssl_context

            req = urllib.request.Request(url, headers={"User-Agent": "TRACE-update"})
            with urllib.request.urlopen(req, timeout=30, context=make_ssl_context()) as resp:
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
            self._update_status_label.setText(
                f"<span style='color: {_ct().text_muted};'>Update download cancelled.</span>"
            )
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
