"""Attach a LivePreviewPane to the existing TRACE PipelineConfigDialog.

All of the integration lives here so the dialog itself needs only a tiny call:

    from live_tune.dialog_integration import attach_live_preview
    attach_live_preview(self)          # inside PipelineConfigDialog, after _build_ui

``attach_live_preview`` performs three jobs from the outside:

1. **Layout surgery** — re-parents the dialog's existing vertical layout into the
   left half of a horizontal QSplitter and mounts the preview pane (collapsed)
   on the right, behind a "Show live preview" toggle.
2. **Widget wiring** — connects every parameter widget in ``dialog._widgets`` to
   the pane's ``on_config_changed`` (so any edit triggers the right cache tier)
   and the tab widget's ``currentChanged`` so the pane knows the active tier.
3. **Lifecycle + preset save** — stops the worker thread when the dialog closes,
   and writes "Save as preset…" output to TRACE/presets/<name>.json.

It is deliberately defensive: any failure leaves the dialog fully functional
without a preview (the feature is strictly additive).
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QInputDialog,
    QMessageBox,
    QPushButton,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from .preview_pane import LivePreviewPane
from .session import FIELD_TIER

logger = logging.getLogger(__name__)


def _tier_change_signals(kind: str, widget, extra):
    """Yield (signal, sends_value) for a dialog widget of the given kind.

    Mirrors PipelineConfigDialog's KIND constants. Returns the Qt signals whose
    emission means "this field changed". We connect them all to one slot.
    """
    sigs = []
    if kind in ("float", "int"):
        sigs.append(widget.valueChanged)
    elif kind in ("opt_float", "opt_int"):
        check, spin = extra
        sigs.append(check.toggled)
        sigs.append(spin.valueChanged)
    elif kind == "bool":
        sigs.append(widget.toggled)
    elif kind == "enum_list":
        sigs.append(widget.itemChanged)
    elif kind == "float_list":
        sigs.append(widget.editingFinished)
    return sigs


def _collect_model_paths(dialog) -> dict:
    """Pull landmark/segmentation model paths off the dialog for image mode."""
    paths = {}
    lm = getattr(dialog, "_calib_lm_path", "") or ""
    seg = getattr(dialog, "_calib_seg_path", "") or ""
    # The Models tab may hold edited paths; prefer those when present.
    lm_edit = getattr(dialog, "_lm_model_edit", None)
    seg_edit = getattr(dialog, "_seg_model_edit", None)
    if lm_edit is not None and lm_edit.text().strip():
        lm = lm_edit.text().strip()
    if seg_edit is not None and seg_edit.text().strip():
        seg = seg_edit.text().strip()
    if lm:
        paths["landmark_checkpoint"] = lm
    if seg:
        paths["segmentation_model_dir"] = seg
    return paths


def _build_preproc_getter(dialog, main_window):
    """Return a callable yielding current preprocessing options for the preview.

    Reads dialog-reachable knobs (wing model, wing expand) live, and the
    main-window snapshot for rotation / isolation-enable / rescale target.
    Everything is read defensively so a missing attribute degrades to a safe
    default rather than crashing the preview.
    """

    def getter() -> dict:
        pp: dict = {}
        # Wing isolation: only pass a model dir when isolation is enabled AND a
        # model path is set (mirrors gui.py's run-time resolution).
        iso_enabled = bool(getattr(main_window, "_wing_isolation_enabled", False))
        model_path = ""
        try:
            model_path = dialog.get_wing_isolation_model_path()
        except Exception:  # noqa: BLE001
            model_path = getattr(main_window, "_wing_isolation_model_path", "") or ""
        if iso_enabled and model_path.strip():
            pp["wing_model_dir"] = model_path.strip()
        try:
            pp["wing_expand_fraction"] = float(dialog.get_wing_expand_fraction())
        except Exception:  # noqa: BLE001
            pp["wing_expand_fraction"] = float(getattr(main_window, "_wing_expand_fraction", 0.05))
        pp["do_rotation"] = bool(getattr(main_window, "_do_rotation", False))
        pp["rotation_mirror_correct"] = bool(getattr(main_window, "_rotation_mirror_correct", False))
        # Stage-1 rescale target (active target's training µm/px). Optional.
        resolver = getattr(main_window, "_resolve_active_target_um_per_px", None)
        if callable(resolver):
            try:
                pp["target_um_per_px"] = resolver()
            except Exception:  # noqa: BLE001
                pp["target_um_per_px"] = None
        return pp

    return getter


def _save_preset(dialog, config) -> None:
    """Write the tuned config to TRACE/presets/<name>.json (config_io format)."""
    try:
        from config_io import config_to_dict  # TRACE/config_io.py (on sys.path)
    except Exception:  # noqa: BLE001
        QMessageBox.warning(dialog, "Save preset", "config_io is unavailable; cannot save preset.")
        return
    name, ok = QInputDialog.getText(dialog, "Save as preset", "Preset name:")
    if not ok or not name.strip():
        return
    safe = "".join(c for c in name.strip() if c.isalnum() or c in (" ", "-", "_")).strip().replace(" ", "_")
    presets_dir = Path(__file__).resolve().parents[2] / "TRACE" / "presets"
    presets_dir.mkdir(parents=True, exist_ok=True)
    out = presets_dir / f"{safe}.json"
    try:
        out.write_text(json.dumps(config_to_dict(config), indent=2), encoding="utf-8")
    except Exception as exc:  # noqa: BLE001
        QMessageBox.critical(dialog, "Save preset", f"Could not write preset:\n{exc}")
        return
    QMessageBox.information(dialog, "Save preset", f"Saved preset to:\n{out}")


def attach_live_preview(dialog) -> Optional[LivePreviewPane]:
    """Mount a collapsible live-preview pane on the right of ``dialog``.

    Returns the pane (already created but hidden) or None if attachment failed.
    Safe to call once after the dialog's ``_build_ui`` has run.
    """
    try:
        old_layout = dialog.layout()
        if old_layout is None:
            logger.warning("attach_live_preview: dialog has no layout; skipping")
            return None

        # Move the existing layout (with all its widgets) into a left container.
        left = QWidget()
        left.setLayout(old_layout)

        main_window = dialog.parent()
        pane = LivePreviewPane(
            get_config=dialog.get_config,
            model_paths=_collect_model_paths(dialog),
            default_image_dir=getattr(dialog, "_calib_input_path", "") or "",
            preproc_getter=_build_preproc_getter(dialog, main_window),
            parent=dialog,
        )
        pane.setVisible(False)
        pane.save_preset_requested.connect(lambda cfg: _save_preset(dialog, cfg))

        splitter = QSplitter(Qt.Horizontal)
        splitter.addWidget(left)
        splitter.addWidget(pane)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setChildrenCollapsible(False)

        wrapper = QVBoxLayout(dialog)  # dialog has no layout now → attaches cleanly
        wrapper.setContentsMargins(0, 0, 0, 0)

        # Toggle button to reveal/hide the preview (keeps the dialog compact by default).
        toggle = QPushButton("Show live preview ▸")
        toggle.setCheckable(True)
        toggle.setToolTip("Open a live vein-overlay preview that updates as you tune parameters")

        def _on_toggle(checked: bool) -> None:
            pane.setVisible(checked)
            toggle.setText("Hide live preview ◂" if checked else "Show live preview ▸")
            if checked and dialog.width() < 1100:
                dialog.resize(max(dialog.width() + 520, 1100), dialog.height())

        toggle.toggled.connect(_on_toggle)

        top = QWidget()
        top_l = QVBoxLayout(top)
        top_l.setContentsMargins(8, 6, 8, 0)
        top_l.addWidget(toggle)
        wrapper.addWidget(top)
        wrapper.addWidget(splitter, 1)

        # Wire every parameter widget to the pane's change handler.
        widgets = getattr(dialog, "_widgets", {})
        for name, (kind, widget, extra) in widgets.items():
            for sig in _tier_change_signals(kind, widget, extra):
                # default-arg capture of name avoids the late-binding closure bug
                sig.connect(lambda *a, _n=name: pane.on_config_changed(_n))

        # Wire the dialog's preprocessing widgets (NOT PipelineConfig fields, so
        # not in _widgets) to a preprocessing re-run. Changing the wing model or
        # expand fraction re-runs the DL preprocessing on the loaded image.
        wing_expand = getattr(dialog, "_wing_expand_spin", None)
        if wing_expand is not None:
            wing_expand.valueChanged.connect(lambda *a: pane.on_preproc_changed())
        wing_model = getattr(dialog, "_wing_model_edit", None)
        if wing_model is not None:
            wing_model.textChanged.connect(lambda *a: pane.on_preproc_changed())

        # Tell the pane the active tab's tier when the user switches tabs.
        tabs = getattr(dialog, "_tabs", None)
        if tabs is not None:
            def _on_tab(_idx, _tabs=tabs, _pane=pane):
                title = _tabs.tabText(_idx).lower()
                if "wing graph" in title:
                    _pane.set_active_tier("A")
                elif "tracing" in title:
                    _pane.set_active_tier("B")
                elif "intervein" in title:
                    _pane.set_active_tier("C")
            tabs.currentChanged.connect(_on_tab)

        # Stop the worker thread when the dialog closes (wrap existing done()).
        _orig_done = dialog.done

        def _done(result: int, _orig=_orig_done, _pane=pane):
            try:
                _pane.shutdown()
            except Exception:  # noqa: BLE001
                logger.exception("pane shutdown during dialog.done")
            return _orig(result)

        dialog.done = _done
        dialog._live_preview_pane = pane  # keep a reference
        return pane
    except Exception:  # noqa: BLE001
        logger.exception("attach_live_preview failed; dialog continues without preview")
        return None
