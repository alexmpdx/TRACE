"""Per-image inspector + editor for landmarks and vein/intervein segmentation.

Three entry points (Main tab), all converging on
``TraceWindow._open_landmark_inspector(image_path, cohort=...)``:
  - Right-click a single image           → "Inspect & Edit landmarks…"
  - Right-click on a multi-selection      → "Inspect & Edit landmarks (N selected)…"
  - Post-run "Review failed images" button alongside the rerun-failed buttons

The dialog hosts two tabs in a ``QTabWidget``:
  - **Landmarks** — drag/add/delete landmark points, save
    ``<image_dir>/<stem>_landmarks_override.geojson`` (Stage-3 short-circuit).
  - **Veins / Interveins** — reclassify / delete / draw vein & intervein
    polygons, save ``<image_dir>/<stem>_segmentation_override.geojson``
    (Stage-5 short-circuit). Both short-circuits live in preprocessing/pipeline.py.

Cohort mode (>=2 images) shows prev/next navigation + a clickable image list so
images can be edited in any order. Each tab owns one napari Viewer, created
lazily (the segmentation viewer only when its tab is first shown) and reused
across cohort images (construction is expensive).

The editors are read-write siblings of measurementMaker's read-only
``LandmarkPickerWidget`` — they share its napari setup pattern but omit the
snap-back guard, because editing is the whole point. Mirror that widget's
napari API (``border_color``/``border_width``, ``features=``, ``size=90``).
"""

from __future__ import annotations

import html
import json
import re
import shutil
import tempfile
from pathlib import Path
from typing import Optional

from PyQt5.QtCore import QObject, QRunnable, Qt, QThreadPool, QTimer, pyqtSignal, pyqtSlot
from PyQt5.QtGui import QKeySequence
from PyQt5.QtWidgets import (
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QShortcut,
    QSlider,
    QSpinBox,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

# Landmark point / label colors. Unreliable (gate-failed) points are drawn in a
# distinct color so the user can see at a glance which ones need correcting;
# the currently-selected point (and its label) turns the "selected" color.
LM_FACE_COLOR = "cyan"  # reliable, unselected point
LM_TEXT_COLOR = "yellow"  # reliable, unselected label
LM_SELECTED_COLOR = "orange"  # selected point + label (matches Custom Measurements tab)
LM_FAILED_COLOR = "#FF2D2D"  # gate-failed point + label

# Vein/intervein vocabulary + colors (class indices from the segmentation
# model's metadata.json). Only vein and intervein are user-editable: "hinge
# junk" is stripped from the pipeline's output entirely and is covered by the
# Erase tool, so it's never offered as a class.
SEG_EDIT_CLASSES = ["vein", "intervein"]
SEG_CLASS_COLORS = {
    "vein": "#00BBFF",
    "intervein": "#FF5E00",
}
SEG_CLASS_INDEX = {"intervein": 2, "vein": 3}


def _parse_landmarks_geojson(path: Path) -> dict:
    """Load a landmarks GeoJSON into ``{name: (x, y)}`` (raw names, pixel coords).

    Tolerant of either Stage-3's full schema (confidence/sharpness/etc.) or the
    minimal override schema (just classification.name + coordinates).
    """
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    out: dict = {}
    for feat in data.get("features", []):
        geom = feat.get("geometry") or {}
        if geom.get("type") != "Point":
            continue
        coords = geom.get("coordinates") or []
        if len(coords) < 2:
            continue
        props = feat.get("properties") or {}
        cls = props.get("classification") or {}
        name = cls.get("name") or props.get("name")
        if not name:
            continue
        out[str(name)] = (float(coords[0]), float(coords[1]))
    return out


def _parse_failed_gate_landmarks(message: str) -> set:
    """Extract the failed landmark names from a confidence-gate error message.

    The main GUI already records each failed image's (landmark-name-translated)
    error string on the TraceWindow; for a landmark confidence-gate abort it reads
    ``Core landmarks failed confidence gate: <name> (<reason>), <name> (<reason>)``
    (see LandmarkLocator ``LowConfidenceLandmarkError``). We parse the ``<name>``
    tokens so those exact points can be colored red on the canvas.

    Returns an empty set for any other message (quality-gate aborts, preprocessing
    or analysis failures) — those name no landmarks, so nothing is colored. Reasons
    join their sub-parts with ``; `` and never contain ``,`` or ``()``, so entries
    split cleanly on the ``name (reason)`` shape.
    """
    if not message or "Core landmarks failed confidence gate" not in message:
        return set()
    _, _, tail = message.partition(":")
    return {m.group(1).strip() for m in re.finditer(r"([^,]+?)\s*\(([^)]*)\)", tail) if m.group(1).strip()}


class _TaskSignals(QObject):
    # Carry the load token through the signal so the (main-thread) handler can
    # discard results from a superseded load.
    done = pyqtSignal(int, object)
    failed = pyqtSignal(int, str)


class _AsyncTask(QRunnable):
    """Run a no-Qt callable on the global thread pool and emit its result.

    Keeps the inspector responsive (so the busy progress bar actually animates)
    while heavy, non-Qt work runs — model inference, preprocessing, image decode,
    rasterization. The result is rendered into napari back on the GUI thread.
    """

    def __init__(self, fn, token):
        super().__init__()
        self._fn = fn
        self._token = token
        self.signals = _TaskSignals()

    @pyqtSlot()
    def run(self):
        try:
            result = self._fn()
        except Exception as exc:  # noqa: BLE001
            self.signals.failed.emit(self._token, str(exc))
            return
        self.signals.done.emit(self._token, result)


class _AsyncLoadMixin:
    """Shared async-load plumbing for the two editor widgets.

    The subclass supplies a ``_progress`` QProgressBar, an ``_loaded`` flag, a
    ``_fail_load(msg)``, and a ``_render_payload(payload)`` that builds the
    napari layers from the worker's result. ``_begin_load`` runs ``work_fn``
    off-thread (busy bar visible) and renders on the GUI thread when it
    finishes. The handlers are bound methods of the editor (a QObject living on
    the GUI thread), so Qt invokes them via a queued connection — the napari
    calls never run on the worker thread. A token discards results from a load
    that was superseded (cohort swap) or whose dialog has closed.
    """

    def _init_async(self) -> None:
        self._loading = False
        self._load_token = 0
        self._task = None

    def _settings(self):
        """The window's QSettings, used to persist toolbar choices."""
        return getattr(self._window, "settings", None)

    def _begin_load(self, work_fn) -> None:
        if self._loading:
            return
        self._loading = True
        self._load_token += 1
        self._set_loading(True)
        task = _AsyncTask(work_fn, self._load_token)
        task.signals.done.connect(self._on_async_done)
        task.signals.failed.connect(self._on_async_failed)
        self._task = task  # keep a ref until the run completes
        QThreadPool.globalInstance().start(task)

    def _on_async_done(self, token, payload) -> None:
        if token != self._load_token:
            return  # superseded by a newer load, or the dialog closed
        try:
            self._render_payload(payload)
            self._loaded = True
        except RuntimeError:
            return  # widget/viewer was torn down mid-flight
        finally:
            self._set_loading(False)
            self._loading = False

    def _on_async_failed(self, token, msg) -> None:
        if token != self._load_token:
            return
        self._loading = False
        self._loaded = True  # don't auto-retry on every tab switch
        try:
            self._set_loading(False)
        except RuntimeError:
            return
        self._fail_load(msg)

    def _build_loading_overlay(self) -> None:
        """Translucent overlay that dims + blocks the editor while a load runs.

        Covers the whole editor (controls + viewer), so the stale image can't be
        clicked and the wait reads clearly. Shows a centered "Loading <image>…"
        label and an indeterminate bar. Created once; positioned on show/resize.
        """
        self._overlay = QWidget(self)
        self._overlay.setStyleSheet("background-color: rgba(20, 20, 20, 150);")
        self._overlay.setVisible(False)
        ov = QVBoxLayout(self._overlay)
        ov.addStretch(1)
        self._loading_label = QLabel("Loading…", self._overlay)
        self._loading_label.setAlignment(Qt.AlignCenter)
        self._loading_label.setStyleSheet("color: #eee; font-size: 15px; background: transparent;")
        ov.addWidget(self._loading_label)
        self._progress = QProgressBar(self._overlay)
        self._progress.setRange(0, 0)
        self._progress.setTextVisible(False)
        self._progress.setMaximumWidth(320)
        bar_row = QHBoxLayout()
        bar_row.addStretch(1)
        bar_row.addWidget(self._progress)
        bar_row.addStretch(1)
        ov.addLayout(bar_row)
        ov.addStretch(1)

    def _position_overlay(self) -> None:
        ov = getattr(self, "_overlay", None)
        if ov is not None:
            ov.setGeometry(self.rect())

    def _set_canvas_enabled(self, enabled: bool) -> None:
        viewer = getattr(self, "_viewer", None)
        if viewer is None:
            return
        try:
            viewer.window.qt_viewer.setEnabled(enabled)
        except Exception:
            pass

    def _set_loading(self, on: bool) -> None:
        try:
            label = getattr(self, "_loading_label", None)
            if on and label is not None:
                try:
                    label.setText(f"Loading {self._image_path.name}…")
                except Exception:
                    label.setText("Loading…")
            # Clear (don't hide) the pre-first-load placeholder so its text
            # doesn't show through the translucent overlay — hiding it would
            # collapse its stretch slot and shove the toolbar to the middle.
            ph = getattr(self, "_viewer_placeholder", None)
            if on and ph is not None:
                try:
                    ph.setText("")
                except RuntimeError:
                    pass
            ov = getattr(self, "_overlay", None)
            if ov is not None:
                if on:
                    self._position_overlay()
                    ov.setVisible(True)
                    ov.raise_()
                else:
                    ov.setVisible(False)
            # Belt-and-suspenders: block canvas interaction even if the overlay
            # can't paint over napari's GL surface on some platforms.
            self._set_canvas_enabled(not on)
        except RuntimeError:
            pass

    def _bump_load_token(self) -> None:
        self._load_token += 1
        self._loading = False


class LandmarkInspectorDialog(QDialog):
    """Modal wrapper: tabbed editors + (cohort) navigation + buttons."""

    def __init__(self, window, image_path, cohort: Optional[list] = None):
        """
        Args:
            window: The TraceWindow (parent).
            image_path: First image to display.
            cohort: Optional list of images to step through. When given and
                len > 1, the footer shows prev/next navigation + a counter.
                A cohort of <=1 is normalized to single-image mode.
        """
        super().__init__(window)
        self._window = window
        self._image_path = Path(image_path)
        self._cohort = [Path(p) for p in cohort] if cohort and len(cohort) > 1 else None
        self._cohort_index = (
            self._cohort.index(self._image_path) if self._cohort is not None and self._image_path in self._cohort else 0
        )
        self.setWindowTitle(f"Inspect & Edit — {self._image_path.name}")
        self.resize(1320, 820)

        # Cohort bookkeeping: which images already have a saved override this
        # session (shown with a ✓ in the list), and a guard so programmatic
        # list-selection changes don't re-trigger navigation.
        self._saved_indices: set = set()
        self._suppress_list_signal = False

        layout = QVBoxLayout(self)
        content = QHBoxLayout()
        # Left sidebar: clickable list of cohort images, so the user can jump
        # to any image in any order instead of only stepping prev/next.
        self._build_image_list(content)

        # Two tabs, one editor each. Each editor owns a napari viewer, created
        # lazily on first show — opening the dialog only pays for the Landmarks
        # viewer; the segmentation viewer is built when its tab is first opened.
        self._tabs = QTabWidget()
        self._landmark_editor = LandmarkEditorWidget(self, self._image_path)
        self._seg_editor = SegmentationEditorWidget(self, self._image_path)
        self._tabs.addTab(self._landmark_editor, "Landmarks")
        self._tabs.addTab(self._seg_editor, "Veins / Interveins")
        self._editors = [self._landmark_editor, self._seg_editor]
        self._tabs.currentChanged.connect(self._on_tab_changed)
        content.addWidget(self._tabs, stretch=1)
        layout.addLayout(content, stretch=1)

        # Navigation row (cohort mode only).
        self._build_navigation()

        # Save / Restore / Cancel buttons (act on the active tab's editor).
        btns = QDialogButtonBox(self)
        self.btn_save = btns.addButton("Save override", QDialogButtonBox.AcceptRole)
        self.btn_restore = btns.addButton("Restore predictions", QDialogButtonBox.ResetRole)
        self.btn_cancel = btns.addButton(QDialogButtonBox.Cancel)
        btns.accepted.connect(self._on_save)
        btns.rejected.connect(self.reject)
        self.btn_restore.clicked.connect(self._on_restore)
        layout.addWidget(btns)

        # Defer the heavy load until after the dialog paints. napari Viewer
        # construction + model first-call init together can take several
        # seconds; showing the dialog first gives the user feedback.
        QTimer.singleShot(0, self._ensure_active_loaded)

    # -- tab + editor plumbing -------------------------------------------
    def _active_editor(self):
        return self._editors[self._tabs.currentIndex()]

    def _ensure_active_loaded(self) -> None:
        self._active_editor().ensure_loaded()

    def _on_tab_changed(self, _index: int) -> None:
        self._ensure_active_loaded()

    def _loaded_editors(self):
        return [e for e in self._editors if e.is_loaded()]

    def _dirty_editors(self):
        return [e for e in self._loaded_editors() if e.has_unsaved_changes()]

    def _on_restore(self) -> None:
        self._active_editor().restore()

    def persist_landmark_edits_for_pipeline(self) -> None:
        """Flush unsaved landmark edits to the override sidecar so on-demand
        segmentation preprocessing (Stage 3) uses the user's corrected
        landmarks for the hinge chop.

        Only deliberate edits are persisted — an untouched Landmarks tab is
        left alone so we never fabricate an override out of raw predictions.
        """
        ed = self._landmark_editor
        if ed.is_loaded() and ed.has_unsaved_changes():
            try:
                ed.persist_override_for_pipeline()
            except Exception:
                pass  # best-effort; preprocessing falls back to a fresh prediction

    # -- image list (cohort sidebar) -------------------------------------
    def _build_image_list(self, content_layout) -> None:
        if self._cohort is None:
            return
        col = QVBoxLayout()
        lbl = QLabel("Images to edit")
        lbl.setStyleSheet("color: #aaa; font-weight: bold;")
        col.addWidget(lbl)
        self.image_list = QListWidget()
        self.image_list.setMaximumWidth(280)
        self.image_list.setToolTip(
            "Click any image to jump straight to it. A ✓ marks images you've "
            "already saved an override for this session."
        )
        for path in self._cohort:
            item = QListWidgetItem(path.name)
            item.setToolTip(str(path))
            self.image_list.addItem(item)
        self.image_list.currentRowChanged.connect(self._on_list_row_changed)
        col.addWidget(self.image_list, stretch=1)
        content_layout.addLayout(col)

    def _refresh_image_list(self) -> None:
        """Sync the list's highlight + per-row ✓ markers to current state."""
        if self._cohort is None:
            return
        self._suppress_list_signal = True
        try:
            for i, path in enumerate(self._cohort):
                prefix = "✓ " if i in self._saved_indices else ""
                self.image_list.item(i).setText(f"{prefix}{path.name}")
            self.image_list.setCurrentRow(self._cohort_index)
        finally:
            self._suppress_list_signal = False

    def _on_list_row_changed(self, row: int) -> None:
        if self._suppress_list_signal or self._cohort is None:
            return
        if row == self._cohort_index or not (0 <= row < len(self._cohort)):
            return
        if not self._go_to_index(row):
            # Navigation was aborted (user cancelled the unsaved-changes
            # prompt) — snap the list selection back to the current image.
            self._suppress_list_signal = True
            try:
                self.image_list.setCurrentRow(self._cohort_index)
            finally:
                self._suppress_list_signal = False

    # -- navigation -------------------------------------------------------
    def _build_navigation(self) -> None:
        if self._cohort is None:
            return
        nav_row = QHBoxLayout()
        self.btn_prev = QPushButton("← Previous")
        self.btn_prev.clicked.connect(lambda: self._navigate(-1))
        nav_row.addWidget(self.btn_prev)

        self.lbl_position = QLabel("")
        self.lbl_position.setAlignment(Qt.AlignCenter)
        self.lbl_position.setStyleSheet("color: #aaa;")
        nav_row.addWidget(self.lbl_position, stretch=1)

        self.btn_save_next = QPushButton("Save & Next →")
        self.btn_save_next.clicked.connect(self._on_save_and_next)
        nav_row.addWidget(self.btn_save_next)

        self.btn_next = QPushButton("Next →")
        self.btn_next.clicked.connect(lambda: self._navigate(+1))
        nav_row.addWidget(self.btn_next)

        self.layout().addLayout(nav_row)
        self._refresh_nav_state()

    def _refresh_nav_state(self) -> None:
        if self._cohort is None:
            return
        n = len(self._cohort)
        i = self._cohort_index
        self.lbl_position.setText(f"Image {i + 1} of {n} — {self._image_path.name}")
        self.btn_prev.setEnabled(i > 0)
        self.btn_next.setEnabled(i < n - 1)
        self.btn_save_next.setEnabled(i < n - 1)
        self._refresh_image_list()

    def _navigate(self, delta: int) -> None:
        if self._cohort is None:
            return
        self._go_to_index(self._cohort_index + delta)

    def _go_to_index(self, new_idx: int) -> bool:
        """Switch to cohort image ``new_idx``. Returns False if aborted.

        Shared by prev/next, Save & Next, and the sidebar list. Prompts to
        save unsaved edits on ANY loaded tab before leaving the current image.
        """
        if self._cohort is None:
            return False
        if not (0 <= new_idx < len(self._cohort)) or new_idx == self._cohort_index:
            return False
        dirty = self._dirty_editors()
        if dirty:
            reply = QMessageBox.question(
                self,
                "Unsaved changes",
                "You have unsaved edits on this image. Save before moving on?",
                QMessageBox.Save | QMessageBox.Discard | QMessageBox.Cancel,
            )
            if reply == QMessageBox.Cancel:
                return False
            if reply == QMessageBox.Save:
                for ed in dirty:
                    try:
                        ed.save_override()
                    except Exception as exc:  # noqa: BLE001
                        QMessageBox.critical(self, "Save failed", f"{exc}")
                        return False
                self._saved_indices.add(self._cohort_index)
        self._cohort_index = new_idx
        self._image_path = self._cohort[new_idx]
        self.setWindowTitle(f"Inspect & Edit — {self._image_path.name}")
        # Point every editor at the new image; only the active tab reloads now,
        # the others reload lazily when next shown.
        for ed in self._editors:
            ed.set_image(self._image_path)
        self._refresh_nav_state()
        self._ensure_active_loaded()
        return True

    def _on_save_and_next(self) -> None:
        result = self._save_active()
        if result is None or result is False:
            return  # save failed or user declined an empty-save
        self._navigate(+1)

    def _on_save(self) -> None:
        result = self._save_active()
        if result is None or result is False:
            return  # nothing saved (declined confirm) or save failed
        QMessageBox.information(
            self,
            "Saved",
            f"Override saved to:\n\n{result}\n\n"
            "The next run that includes this image will use the override "
            "instead of running the model for that stage.",
        )
        # In cohort mode plain "Save" should NOT close the dialog — the user
        # may want to keep reviewing. In single-image mode it auto-closes, but
        # only once the OTHER tab has no unsaved edits, so a two-tab session
        # isn't cut short after saving just one tab.
        if self._cohort is None and not self._dirty_editors():
            self.accept()

    def _save_active(self):
        """Save the active tab's editor. Returns the Path, None (declined), or
        False (error already reported)."""
        editor = self._active_editor()
        try:
            override_path = editor.save_override()
        except Exception as exc:  # noqa: BLE001
            QMessageBox.critical(self, "Save failed", f"Could not write override: {exc}")
            return False
        if override_path is None:
            return None  # user declined an empty-save confirm
        if self._cohort is not None:
            self._saved_indices.add(self._cohort_index)
            self._refresh_image_list()
        return override_path

    def closeEvent(self, event) -> None:  # noqa: N802 (Qt override)
        for ed in self._editors:
            ed.shutdown()
        super().closeEvent(event)


class LandmarkEditorWidget(QWidget, _AsyncLoadMixin):
    """Napari embed + landmark editing controls.

    Public API used by the dialog:
      - load_or_generate()       — initial load (call after dialog paints)
      - swap_image(new_path)     — reuse the napari viewer for a new image
      - save_override() -> Path  — write sidecar GeoJSON, return its path
      - restore_predictions()    — reset points to the original predictions
      - has_unsaved_changes()    — for dirty-state navigation prompts
      - shutdown()               — close the napari viewer on dialog close

    napari stores landmark identity in two feature columns kept in lockstep
    with the point data: ``name`` (canonical/raw key, the source of truth
    written on save) and ``label`` (friendly display string shown on the
    canvas). Reading ``name`` straight off the layer at save time means even a
    keyboard delete on the canvas can't desync the saved file from what's shown.
    """

    def __init__(self, parent_dialog, image_path):
        super().__init__(parent_dialog)
        self._dialog = parent_dialog
        self._image_path = Path(image_path)
        self._window = parent_dialog._window
        self._viewer = None  # napari.Viewer, lazy
        self._points_layer = None
        self._predicted_positions: dict = {}  # {raw_name: (x, y)} for Restore
        # Per-image failure info, sourced from the GUI's already-extracted
        # TraceWindow._image_error_text (NOT the regenerated GeoJSON, whose gate
        # flags are wiped by disable_gates). Populated in _render on the GUI thread.
        self._failure_text: str = ""
        self._failed_names: set = set()  # landmark names (raw + display) that tripped the gate
        self._last_saved_snapshot: Optional[tuple] = None
        # Point size defaults to an image-adaptive value on first render (None
        # here). A user drag on the slider pins an absolute size that then carries
        # across cohort images. Label size is image-independent, so it's restored
        # from settings in _build_ui.
        self._point_size: Optional[int] = None
        self._label_size = 12
        self._loaded = False
        self._init_async()
        self._build_ui()

    # -- lazy load protocol (shared shape with SegmentationEditorWidget) ---
    def is_loaded(self) -> bool:
        return self._loaded

    def set_image(self, new_image_path) -> None:
        """Point the editor at a new image WITHOUT loading (cohort swap)."""
        self._image_path = Path(new_image_path)
        self._points_layer = None
        self._predicted_positions = {}
        self._failure_text = ""
        self._failed_names = set()
        self._last_saved_snapshot = None
        self._loaded = False
        self._bump_load_token()  # discard any in-flight load for the old image

    def ensure_loaded(self) -> None:
        if self._loaded or self._loading:
            return
        self.load_or_generate()

    def restore(self) -> None:
        self.restore_predictions()

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        controls = QHBoxLayout()
        controls.addWidget(QLabel("Add landmark:"))
        self.cmb_add_name = QComboBox()
        from measurement_maker.landmark_names import LANDMARK_DISPLAY_NAMES

        for raw_name, display in LANDMARK_DISPLAY_NAMES.items():
            self.cmb_add_name.addItem(f"{display} ({raw_name})", raw_name)
        controls.addWidget(self.cmb_add_name)
        self.btn_add = QPushButton("Add at image center")
        self.btn_add.setToolTip("Add the selected landmark at the image center, then drag it into place.")
        self.btn_add.clicked.connect(self._on_add_landmark)
        controls.addWidget(self.btn_add)
        self.btn_delete = QPushButton("Delete selected")
        self.btn_delete.setToolTip("Remove the landmark(s) currently selected on the canvas.")
        self.btn_delete.clicked.connect(self._on_delete_selected)
        controls.addWidget(self.btn_delete)

        self.btn_select = QPushButton("Select / move")
        self.btn_select.setToolTip("Click a landmark to select it, or drag to reposition it.")
        self.btn_select.clicked.connect(lambda: self._set_points_mode("select"))
        controls.addWidget(self.btn_select)

        self.btn_pan = QPushButton("Pan / zoom")
        self.btn_pan.setToolTip("Stop editing; drag to pan and scroll to zoom.")
        self.btn_pan.clicked.connect(lambda: self._set_points_mode("pan_zoom"))
        controls.addWidget(self.btn_pan)
        controls.addStretch(1)
        layout.addLayout(controls)

        # Second row: live point-size + label-size sliders. Point size seeds from
        # an image-adaptive default at render; both apply to the layer live.
        sizes = QHBoxLayout()
        sizes.addWidget(QLabel("Point size:"))
        self.sld_point_size = QSlider(Qt.Horizontal)
        self.sld_point_size.setRange(2, 400)
        self.sld_point_size.setMinimumWidth(140)
        self.sld_point_size.setToolTip("Diameter of the landmark points, in pixels.")
        self.sld_point_size.valueChanged.connect(self._on_point_size_changed)
        sizes.addWidget(self.sld_point_size)
        self.lbl_point_size_val = QLabel("—")
        self.lbl_point_size_val.setMinimumWidth(34)
        sizes.addWidget(self.lbl_point_size_val)

        sizes.addSpacing(20)
        sizes.addWidget(QLabel("Label size:"))
        self.sld_label_size = QSlider(Qt.Horizontal)
        self.sld_label_size.setRange(4, 48)
        self.sld_label_size.setMinimumWidth(120)
        self.sld_label_size.setToolTip("Font size of the landmark name labels.")
        self.sld_label_size.valueChanged.connect(self._on_label_size_changed)
        sizes.addWidget(self.sld_label_size)
        self.lbl_label_size_val = QLabel("—")
        self.lbl_label_size_val.setMinimumWidth(28)
        sizes.addWidget(self.lbl_label_size_val)
        sizes.addStretch(1)
        layout.addLayout(sizes)

        # Per-image failure summary: which landmarks failed the confidence gate
        # and why. Kept short (scrolls if it overflows) so it doesn't eat canvas.
        self.failures_scroll = QScrollArea()
        self.failures_scroll.setWidgetResizable(True)
        self.failures_scroll.setMaximumHeight(72)
        self.failures_scroll.setFrameShape(QScrollArea.NoFrame)
        self.lbl_failures = QLabel("")
        self.lbl_failures.setWordWrap(True)
        self.lbl_failures.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self.lbl_failures.setStyleSheet("padding: 2px 4px;")
        self.failures_scroll.setWidget(self.lbl_failures)
        layout.addWidget(self.failures_scroll)

        self._viewer_placeholder = QLabel(
            "Loading image and predicted landmarks…\n\n"
            "First open may take a few seconds while LandmarkLocator initializes."
        )
        self._viewer_placeholder.setAlignment(Qt.AlignCenter)
        self._viewer_placeholder.setStyleSheet("color: #888; padding: 40px;")
        layout.addWidget(self._viewer_placeholder, stretch=1)
        self._canvas_embedded = False

        # Dimming "Loading…" overlay shown while a load runs off-thread.
        self._build_loading_overlay()

        # Restore the last-used "Add landmark" choice, then persist on change.
        s = self._settings()
        if s is not None:
            saved = s.value("inspector/landmark_add_class", "", type=str)
            if saved:
                i = self.cmb_add_name.findData(saved)
                if i >= 0:
                    self.cmb_add_name.setCurrentIndex(i)
            self._label_size = int(s.value("inspector/landmark_label_size", 12, type=int))
        self.cmb_add_name.currentIndexChanged.connect(self._save_toolbar_settings)

        # Seed the label slider from the restored value (blocked so it doesn't
        # try to touch the not-yet-created layer). The point slider is seeded at
        # render time, once the adaptive default is known.
        self.sld_label_size.blockSignals(True)
        self.sld_label_size.setValue(self._label_size)
        self.sld_label_size.blockSignals(False)
        self.lbl_label_size_val.setText(str(self._label_size))

    def _save_toolbar_settings(self, *_args) -> None:
        s = self._settings()
        if s is None:
            return
        data = self.cmb_add_name.currentData()
        if data:
            s.setValue("inspector/landmark_add_class", data)

    def resizeEvent(self, event):  # noqa: N802 (Qt override)
        super().resizeEvent(event)
        self._position_overlay()

    # -- loading ----------------------------------------------------------
    def load_or_generate(self) -> None:
        """Find (or generate) predictions for self._image_path, then render.

        File search + image decode + any on-demand generation run off-thread
        (busy progress bar), then the napari layers are built on the GUI thread.
        Search order — reopening a previously-edited image starts from the
        override so iterative refinement works:
          1. <image_dir>/<stem>_landmarks_override.geojson  ← prior corrections
          2. <output_folder>/<stem>_landmarks.geojson       ← post-run output
          3. <image_dir>/<stem>_landmarks.geojson           ← Stage-3 sidecar
          4. Generate via _generate_landmarks_for_image()    ← on-demand model
        """
        image_path = self._image_path
        window = self._window
        # Read the Qt widget here (GUI thread); the worker must not touch it.
        try:
            out_text = window.output_edit.text().strip()
        except Exception:
            out_text = ""

        def work():
            import numpy as np

            from TRACE.psd_loader import imread_any

            stem = image_path.stem
            image_dir = image_path.parent
            candidates = [
                image_dir / f"{stem}_landmarks_override.geojson",
                (Path(out_text) / f"{stem}_landmarks.geojson") if out_text else None,
                image_dir / f"{stem}_landmarks.geojson",
            ]
            landmarks_dict = None
            for cand in candidates:
                if cand is None or not cand.is_file():
                    continue
                try:
                    landmarks_dict = _parse_landmarks_geojson(cand)
                    break
                except Exception:
                    continue

            if not landmarks_dict:
                panel = getattr(window, "inline_custom_distances_panel", None)
                if panel is None:
                    raise RuntimeError("Internal: landmark generation panel is unavailable.")
                try:
                    # disable_gates: this image likely has no override BECAUSE its
                    # landmarks failed the confidence gate — generate the model's
                    # best guess anyway so the user has points to drag into place.
                    generated = panel._generate_landmarks_for_image(image_path, disable_gates=True)
                    landmarks_dict = _parse_landmarks_geojson(generated)
                except Exception as exc:  # noqa: BLE001
                    raise RuntimeError(
                        f"No landmarks file found and on-demand generation failed:\n\n{exc}\n\n"
                        "Configure a landmark model in Settings → Models and try again."
                    )

            image = imread_any(image_path)
            if image is None:
                raise IOError(f"Could not load image: {image_path}")
            # imread_any returns BGR(A) (cv2 convention) but napari expects RGB(A).
            if image.ndim == 3 and image.shape[2] == 3:
                image = np.ascontiguousarray(image[..., ::-1])
            elif image.ndim == 3 and image.shape[2] == 4:
                image = np.ascontiguousarray(image[..., [2, 1, 0, 3]])
            return {"image": image, "landmarks": landmarks_dict}

        self._begin_load(work)

    def _render_payload(self, payload: dict) -> None:
        self._predicted_positions = dict(payload["landmarks"])
        self._render(payload["image"], payload["landmarks"])
        self._last_saved_snapshot = self._snapshot_current()

    def _fail_load(self, message: str) -> None:
        QMessageBox.critical(self, "Could not load landmarks", message)
        # In cohort mode the user can step to the next image; only bail out of
        # the whole dialog in single-image mode.
        if self._dialog._cohort is None:
            self._dialog.reject()

    def _render(self, image, landmarks_dict: dict) -> None:
        """Build image + points layers from a preloaded (RGB) image array."""
        import napari
        import numpy as np
        from measurement_maker.landmark_names import landmark_display_name

        if self._viewer is None:
            # show=False keeps napari's own QMainWindow hidden — we embed only
            # the canvas (qt_viewer) into the dialog. Heavy: created once and
            # reused across cohort images via swap_image().
            self._viewer = napari.Viewer(show=False)
            qt_viewer = self._viewer.window.qt_viewer
            self.layout().removeWidget(self._viewer_placeholder)
            self._viewer_placeholder.hide()
            self.layout().addWidget(qt_viewer, stretch=1)
            self._canvas_embedded = True

        self._viewer.layers.clear()
        self._viewer.add_image(image, name=self._image_path.name, rgb=image.ndim == 3)

        raw_names = list(landmarks_dict.keys())
        labels = [landmark_display_name(n) for n in raw_names]
        # napari expects (row, col) = (y, x) ordering for point coordinates.
        coords_yx = np.array([[y, x] for (x, y) in landmarks_dict.values()], dtype=float)
        if coords_yx.size == 0:
            coords_yx = np.empty((0, 2), dtype=float)
        # Pull the already-extracted failure reason for THIS image off the GUI
        # window (GUI thread — safe here). The regenerated GeoJSON's own gate
        # flags are unusable (disable_gates wipes them), so this is the source of
        # both the failure list and the red points.
        err_map = getattr(self._window, "_image_error_text", {}) or {}
        self._failure_text = err_map.get(self._image_path.name, "")
        self._failed_names = _parse_failed_gate_landmarks(self._failure_text)
        # Scale the point size to the image, mirroring the landmark-overlay output
        # (visualize.py: radius = min(h, w) / 125). napari `size` is a diameter, so
        # double the overlay radius — points then look the same relative to the wing
        # regardless of image resolution, instead of a fixed 90 px that dwarfs small
        # images and vanishes on large ones. Only used as the slider's default; a
        # user-set point size (self._point_size) overrides it and carries across
        # cohort images.
        h, w = image.shape[:2]
        if self._point_size is None:
            self._point_size = max(20, int(min(h, w) / 125 * 2))
        point_size = int(self._point_size)
        # Per-point reliability lives in a feature column so it stays in lockstep
        # with the point data across add/delete, driving the failed-point color. A
        # point is "failed" when its raw name or display label is in the parsed
        # gate-failure set (match both so un-translated keys still land).
        reliable_flags = self._reliable_flags(raw_names, labels)
        self._points_layer = self._viewer.add_points(
            coords_yx,
            name="landmarks",
            size=point_size,
            face_color=LM_FACE_COLOR,
            border_color="black",
            border_width=0.15,
            features={
                "name": list(raw_names),
                "label": labels,
                "reliable": reliable_flags,
            },
            text={
                "string": "{label}",
                "size": int(self._label_size),
                "color": LM_TEXT_COLOR,
                "translation": [-30, 0],
            },
        )
        # Seed the point-size slider from whatever size was actually applied.
        self.sld_point_size.blockSignals(True)
        self.sld_point_size.setValue(point_size)
        self.sld_point_size.blockSignals(False)
        self.lbl_point_size_val.setText(str(point_size))
        # CRITICAL DIFFERENCE from LandmarkPickerWidget: select mode stays on
        # and NO snap-back callback is installed. The user is here to edit.
        self._points_layer.mode = "select"
        # Selected point + label turn orange; gate-failed points stay red until
        # corrected (matches the Custom Measurements tab's selection color).
        try:
            self._points_layer.events.highlight.connect(self._update_colors)
        except Exception:
            pass
        self._update_colors()
        self._update_failures_panel()
        try:
            self._viewer.layers.selection.active = self._points_layer
        except Exception:
            pass

    def _reliable_flags(self, raw_names, labels) -> list:
        """Per-point reliability: False iff the point's raw name or display label
        is in the parsed gate-failure set (match both so un-translated keys land)."""
        failed = self._failed_names
        return [(n not in failed and lbl not in failed) for n, lbl in zip(raw_names, labels)]

    def _update_colors(self, _event=None) -> None:
        """Recolor points AND their labels by selection + gate state.

        Selected → orange (point + label); gate-failed (unreliable) → red;
        otherwise the reliable defaults (cyan point / yellow label). Selection
        takes precedence so the active point always reads as selected.
        """
        layer = self._points_layer
        if layer is None:
            return
        try:
            sel = layer.selected_data
            n = len(layer.data)
            reliable = list(layer.features.get("reliable", [True] * n))
            face_colors, text_colors = [], []
            for i in range(n):
                if i in sel:
                    face_colors.append(LM_SELECTED_COLOR)
                    text_colors.append(LM_SELECTED_COLOR)
                elif i < len(reliable) and not reliable[i]:
                    face_colors.append(LM_FAILED_COLOR)
                    text_colors.append(LM_FAILED_COLOR)
                else:
                    face_colors.append(LM_FACE_COLOR)
                    text_colors.append(LM_TEXT_COLOR)
            layer.face_color = face_colors
            # napari coerces a sequence of colors into a per-point (manual) text
            # color encoding; skip when empty so it doesn't fall back to constant.
            if n:
                layer.text.color = text_colors
        except Exception:
            pass

    def _on_point_size_changed(self, value: int) -> None:
        self._point_size = int(value)
        self.lbl_point_size_val.setText(str(int(value)))
        if self._points_layer is not None:
            try:
                self._points_layer.size = int(value)
                self._points_layer.current_size = int(value)
            except Exception:
                pass

    def _on_label_size_changed(self, value: int) -> None:
        self._label_size = int(value)
        self.lbl_label_size_val.setText(str(int(value)))
        if self._points_layer is not None:
            try:
                self._points_layer.text.size = int(value)
            except Exception:
                pass
        s = self._settings()
        if s is not None:
            s.setValue("inspector/landmark_label_size", int(value))

    def _update_failures_panel(self) -> None:
        """Show this image's failure reason(s), as already extracted by the GUI.

        Reads the TraceWindow's per-image error text (translated, human-readable)
        rather than re-deriving anything — this covers ALL failure types (landmark
        confidence gate, quality gate, no-wing, analysis crash, etc.). The message
        is HTML-escaped because gate reasons contain '<' (e.g. peak=0.12<0.20).
        """
        text = (self._failure_text or "").strip()
        if not text:
            self.lbl_failures.setText(
                "<span style='color:#888'>No recorded failure for this image.</span>"
            )
            return
        header = "<span style='color:#FF6B6B'>&#9888; This image failed:</span>"
        body = html.escape(text).replace("\n", "<br>")
        self.lbl_failures.setText(f"{header}<br>{body}")

    def _set_points_mode(self, mode: str) -> None:
        if self._points_layer is None:
            return
        try:
            self._points_layer.mode = mode
        except Exception:
            pass

    # -- editing actions --------------------------------------------------
    def _on_add_landmark(self) -> None:
        if self._points_layer is None:
            return
        import numpy as np
        from measurement_maker.landmark_names import landmark_display_name

        raw_name = self.cmb_add_name.currentData()
        current = list(self._points_layer.features.get("name", []))
        if raw_name in current:
            QMessageBox.information(
                self,
                "Already present",
                f"'{raw_name}' is already in the landmark list. Drag it instead.",
            )
            return
        image_layer = self._viewer.layers[self._image_path.name]
        h, w = image_layer.data.shape[:2]
        center = np.array([[h / 2.0, w / 2.0]], dtype=float)
        if len(self._points_layer.data):
            new_coords = np.vstack([self._points_layer.data, center])
        else:
            new_coords = center
        labels = list(self._points_layer.features.get("label", []))
        reliable = list(self._points_layer.features.get("reliable", []))
        current.append(raw_name)
        labels.append(landmark_display_name(raw_name))
        reliable.append(True)  # a hand-placed point is reliable by definition
        self._points_layer.data = new_coords
        self._points_layer.features = {"name": current, "label": labels, "reliable": reliable}
        self._update_colors()

    def _on_delete_selected(self) -> None:
        if self._points_layer is None:
            return
        selected = self._points_layer.selected_data
        if not selected:
            QMessageBox.information(
                self, "Nothing selected", "Click a landmark on the canvas first, then Delete selected."
            )
            return
        # napari keeps features in lockstep with data on removal.
        self._points_layer.remove_selected()
        self._update_colors()

    def restore_predictions(self) -> None:
        if self._points_layer is None:
            return
        import numpy as np
        from measurement_maker.landmark_names import landmark_display_name

        raw_names = list(self._predicted_positions.keys())
        labels = [landmark_display_name(n) for n in raw_names]
        coords_yx = np.array([[y, x] for (x, y) in self._predicted_positions.values()], dtype=float)
        if coords_yx.size == 0:
            coords_yx = np.empty((0, 2), dtype=float)
        self._points_layer.data = coords_yx
        self._points_layer.features = {
            "name": list(raw_names),
            "label": labels,
            "reliable": self._reliable_flags(raw_names, labels),
        }
        self._update_colors()

    # -- saving -----------------------------------------------------------
    def save_override(self) -> Optional[Path]:
        """Write the override sidecar; return its path (or None if user aborts)."""
        if self._points_layer is None:
            raise RuntimeError("Viewer not initialized; nothing to save.")
        coords_yx = self._points_layer.data
        names = list(self._points_layer.features.get("name", []))
        if len(names) != len(coords_yx):
            raise RuntimeError(f"Internal: name/coord length mismatch ({len(names)} vs {len(coords_yx)})")
        if len(names) == 0:
            reply = QMessageBox.question(
                self,
                "Save with no landmarks?",
                "There are no landmarks to save. An empty override will make "
                "Stage 4/5 fail for this image on the next run.\n\nSave anyway?",
                QMessageBox.Save | QMessageBox.Cancel,
            )
            if reply != QMessageBox.Save:
                return None
        features = []
        for name, (y, x) in zip(names, coords_yx):
            features.append(
                {
                    "type": "Feature",
                    "geometry": {"type": "Point", "coordinates": [float(x), float(y)]},
                    "properties": {
                        "classification": {"name": str(name)},
                        "reliable": True,
                        "gate_reason": "manual override",
                        "confidence": 1.0,
                    },
                }
            )
        payload = {"type": "FeatureCollection", "features": features}
        override_path = self._image_path.parent / f"{self._image_path.stem}_landmarks_override.geojson"
        override_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        self._last_saved_snapshot = self._snapshot_current()
        return override_path

    def persist_override_for_pipeline(self) -> Optional[Path]:
        """Write current landmarks to the override sidecar with no dialogs, so
        on-demand segmentation preprocessing picks them up. Skips writing when
        there are no landmarks (an empty override would break the hinge chop)."""
        if self._points_layer is None or len(self._points_layer.data) == 0:
            return None
        return self.save_override()

    # -- dirty state / lifecycle -----------------------------------------
    def has_unsaved_changes(self) -> bool:
        if self._points_layer is None or self._last_saved_snapshot is None:
            return False
        return self._snapshot_current() != self._last_saved_snapshot

    def _snapshot_current(self) -> tuple:
        """(names_tuple, coords_tuple) for dirty-state comparison."""
        if self._points_layer is None:
            return ((), ())
        names = tuple(self._points_layer.features.get("name", []))
        coords = tuple(tuple(round(float(v), 3) for v in row) for row in self._points_layer.data)
        return (names, coords)

    def shutdown(self) -> None:
        self._bump_load_token()  # discard any in-flight load result
        if self._viewer is not None:
            try:
                self._viewer.close()
            except Exception:
                pass
            self._viewer = None


def _parse_segmentation_fc(data: dict) -> list:
    """Parse a segmentation FeatureCollection into ``[{class, geometry}, ...]``.

    Reads ``properties.class`` and falls back to ``properties.classification.name``.
    """
    out: list = []
    for feat in data.get("features", []):
        geom = feat.get("geometry") or {}
        if geom.get("type") not in ("Polygon", "MultiPolygon") or not geom.get("coordinates"):
            continue
        props = feat.get("properties") or {}
        cls = props.get("class")
        if cls is None:
            cls = (props.get("classification") or {}).get("name")
        out.append({"class": str(cls) if cls else "intervein", "geometry": geom})
    return out


def _parse_segmentation_geojson(path: Path) -> list:
    return _parse_segmentation_fc(json.loads(Path(path).read_text(encoding="utf-8")))


def _hex_to_rgb(hex_color: str) -> list:
    h = (hex_color or "#888888").lstrip("#")
    if len(h) != 6:
        h = "888888"
    return [int(h[0:2], 16) / 255.0, int(h[2:4], 16) / 255.0, int(h[4:6], 16) / 255.0]


def _scale_geom(geom: dict, factor: float) -> dict:
    """Return a copy of a Polygon/MultiPolygon geometry with all coords scaled."""

    def _ring(ring):
        return [[c[0] * factor, c[1] * factor] for c in ring]

    t = geom.get("type")
    coords = geom.get("coordinates") or []
    if t == "Polygon":
        return {"type": "Polygon", "coordinates": [_ring(r) for r in coords]}
    if t == "MultiPolygon":
        return {"type": "MultiPolygon", "coordinates": [[_ring(r) for r in poly] for poly in coords]}
    return geom


class SegmentationEditorWidget(QWidget, _AsyncLoadMixin):
    """Napari Labels (paint-mask) editor for vein / intervein segmentation.

    Public API mirrors LandmarkEditorWidget so the dialog can drive either:
      - is_loaded() / set_image() / ensure_loaded()  — lazy cohort swap
      - load_or_generate()                            — initial load
      - save_override() -> Optional[Path]             — write sidecar GeoJSON
      - restore()                                     — revert to the loaded mask
      - has_unsaved_changes()                         — dirty-state prompts
      - shutdown()                                    — close the viewer

    A vein traces a thin branching network whose *holes* are the enclosed
    intervein regions — and a napari Shapes polygon can't represent holes (it
    would fill them in and bury the whole wing). So the segmentation is shown as
    a per-class label *mask* over the PREPROCESSED image (rescaled + isolated +
    hinge-chopped, pre-rotation) — the exact image the model saw — where holes
    are inherent.
    Editing is paint / fill / erase (napari Labels), the natural model for
    segmentation. On save the mask is re-vectorized to vein/intervein polygons
    via modelTOjson's ``mask_to_geojson`` — the exact converter the pipeline
    uses — so the override matches the pipeline's own output format.

    Coordinates: the model only works on the fully-preprocessed image, so
    first-time generation runs the configured chain (wing isolation + hinge
    chop, rescale if set) with rotation OFF, then maps the polygons into
    ORIGINAL-image pixel space (wing isolation / hinge chop only mask pixels —
    no coord change — so only the rescale factor matters). The mask is
    rasterized at the original image's resolution and the override is saved in
    original space; the Stage-5 short-circuit scales it back by the rescale
    factor on the next run. A saved override is already in original space, so
    reopening it skips preprocessing entirely.
    """

    def __init__(self, parent_dialog, image_path):
        super().__init__(parent_dialog)
        self._dialog = parent_dialog
        self._image_path = Path(image_path)
        self._window = parent_dialog._window
        self._viewer = None
        self._labels_layer = None
        self._baseline_mask = None  # mask as loaded (for Restore)
        self._last_saved_mask = None  # mask at last save (for dirty-state)
        # Rescale factor applied by preprocessing (Stage 1). The mask is edited in
        # the preprocessed (rescaled) pixel space the model actually saw; on save
        # the vectorized polygons are divided back to original space for the
        # sidecar. 1.0 when no rescale was applied.
        self._rescale_factor = 1.0
        self._loaded = False
        self._init_async()
        self._build_ui()

    # -- lazy load protocol ----------------------------------------------
    def is_loaded(self) -> bool:
        return self._loaded

    def set_image(self, new_image_path) -> None:
        self._image_path = Path(new_image_path)
        self._labels_layer = None
        self._baseline_mask = None
        self._last_saved_mask = None
        self._rescale_factor = 1.0
        self._loaded = False
        self._bump_load_token()  # discard any in-flight load for the old image

    def ensure_loaded(self) -> None:
        if self._loaded or self._loading:
            return
        self.load_or_generate()

    # -- ui --------------------------------------------------------------
    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        controls = QHBoxLayout()
        controls.addWidget(QLabel("Class:"))
        self.cmb_class = QComboBox()
        for c in SEG_EDIT_CLASSES:
            self.cmb_class.addItem(c, c)
        self.cmb_class.setToolTip("The class that Paint and Fill apply.")
        self.cmb_class.currentIndexChanged.connect(self._on_class_changed)
        controls.addWidget(self.cmb_class)

        self.btn_paint = QPushButton("Paint")
        self.btn_paint.setToolTip("Brush the chosen class onto the mask (set Brush size at right).")
        self.btn_paint.clicked.connect(lambda: self._set_mode("paint"))
        controls.addWidget(self.btn_paint)

        self.btn_fill = QPushButton("Fill")
        self.btn_fill.setToolTip("Flood-fill the contiguous region you click with the chosen class.")
        self.btn_fill.clicked.connect(lambda: self._set_mode("fill"))
        controls.addWidget(self.btn_fill)

        self.btn_erase = QPushButton("Erase")
        self.btn_erase.setToolTip("Brush a region back to background — e.g. remove spurious tissue.")
        self.btn_erase.clicked.connect(lambda: self._set_mode("erase"))
        controls.addWidget(self.btn_erase)

        self.btn_pan = QPushButton("Pan / zoom")
        self.btn_pan.setToolTip("Stop editing; drag to pan and scroll to zoom.")
        self.btn_pan.clicked.connect(lambda: self._set_mode("pan_zoom"))
        controls.addWidget(self.btn_pan)

        self.btn_undo = QPushButton("Undo")
        self.btn_undo.setToolTip("Undo the last paint / fill / erase stroke (Ctrl+Z).")
        self.btn_undo.clicked.connect(self._on_undo)
        controls.addWidget(self.btn_undo)

        self.btn_redo = QPushButton("Redo")
        self.btn_redo.setToolTip("Redo the last undone stroke (Ctrl+Shift+Z).")
        self.btn_redo.clicked.connect(self._on_redo)
        controls.addWidget(self.btn_redo)

        controls.addWidget(QLabel("Brush:"))
        self.spin_brush = QSpinBox()
        self.spin_brush.setRange(1, 2000)
        self.spin_brush.setValue(60)
        self.spin_brush.setToolTip("Paint / erase brush diameter, in pixels.")
        self.spin_brush.valueChanged.connect(self._on_brush_changed)
        controls.addWidget(self.spin_brush)
        controls.addStretch(1)
        layout.addLayout(controls)

        self._viewer_placeholder = QLabel(
            "Preprocessing this image for vein / intervein inference…\n\n"
            "This runs wing isolation + hinge removal + segmentation, so it can "
            "take a little while (the window may be unresponsive meanwhile)."
        )
        self._viewer_placeholder.setAlignment(Qt.AlignCenter)
        self._viewer_placeholder.setStyleSheet("color: #888; padding: 40px;")
        layout.addWidget(self._viewer_placeholder, stretch=1)

        # Ctrl+Z / Ctrl+Shift+Z work over the editor + its embedded canvas. We
        # register them ourselves because only the napari canvas is embedded
        # (not napari's own window), so napari's built-in keybindings don't fire.
        sc_undo = QShortcut(QKeySequence.Undo, self)
        sc_undo.setContext(Qt.WidgetWithChildrenShortcut)
        sc_undo.activated.connect(self._on_undo)
        sc_redo = QShortcut(QKeySequence.Redo, self)
        sc_redo.setContext(Qt.WidgetWithChildrenShortcut)
        sc_redo.activated.connect(self._on_redo)

        # Dimming "Loading…" overlay shown while preprocessing / segmentation runs.
        self._build_loading_overlay()

        # Restore the last-used class + brush size (persisted in the change
        # handlers below). setCurrentIndex / setValue here re-fire those handlers,
        # which simply re-save the same values — harmless.
        s = self._settings()
        if s is not None:
            i = self.cmb_class.findData(s.value("inspector/seg_class", "vein", type=str))
            if i >= 0:
                self.cmb_class.setCurrentIndex(i)
            self.spin_brush.setValue(int(s.value("inspector/seg_brush_size", 60, type=int)))

    def resizeEvent(self, event):  # noqa: N802 (Qt override)
        super().resizeEvent(event)
        self._position_overlay()

    # -- loading ---------------------------------------------------------
    def load_or_generate(self) -> None:
        """Obtain the segmentation for this image and render it as a paint mask.

        The vein/intervein model runs on the *preprocessed* image (rescaled +
        wing-isolated + hinge-chopped, rotation OFF), so the inspector shows and
        edits the mask over **that exact image** — pixel-identical to what the
        model saw. This matters when Stage 1 rescales heavily: painting over the
        original image (a prior design) diverged from the real inference.

        The configured preprocessing chain is always run on demand (rotation
        OFF, gates OFF) to obtain that preprocessed image + its rescale factor.
        A prior ``<stem>_segmentation_override.geojson`` (stored in original
        space) is loaded and scaled *into* preprocessed space for editing;
        otherwise the freshly-generated segmentation (already in preprocessed
        space) is used as-is. Any unsaved landmark edits are flushed first so
        Stage 3 picks them up and the hinge chop uses the corrected landmarks.
        """
        # Flush landmark edits on the GUI thread BEFORE the worker starts, so
        # Stage 3 picks them up (touches the landmark layer — must be main-thread).
        self._dialog.persist_landmark_edits_for_pipeline()

        image_path = self._image_path
        window = self._window
        override_path = image_path.parent / f"{image_path.stem}_segmentation_override.geojson"

        def work():
            import numpy as np

            from TRACE.psd_loader import imread_any

            have_override = override_path.is_file()
            tmp_dir = Path(tempfile.mkdtemp(prefix="trace_seg_inspect_"))
            try:
                # Always run preprocessing so we have the exact image the model
                # segments. Skip the segmentation forward pass when we already
                # have an override to render (we only need the preprocessed image
                # + rescale factor in that case).
                try:
                    result = window.run_single_image_preprocessing_for_segmentation(
                        image_path, tmp_dir, with_segmentation=not have_override
                    )
                except Exception as exc:  # noqa: BLE001
                    raise RuntimeError(
                        f"Could not preprocess this image for vein/intervein inference:\n\n{exc}\n\n"
                        "Check that a landmark model and a segmentation model are configured "
                        "in Settings → Models."
                    )

                rf = getattr(result, "rescale_factor", 1.0) or 1.0
                # The preprocessed image the segmentation model actually saw:
                # rescaled + isolated + hinge-chopped, pre-rotation. Fall back
                # through the chain if an optional stage didn't produce a file.
                preproc_path = None
                for attr in ("chopped_image_path", "wing_isolated_image_path", "processed_image_path"):
                    p = getattr(result, attr, None)
                    if p and Path(p).is_file():
                        preproc_path = Path(p)
                        break
                if preproc_path is None:
                    preproc_path = image_path

                if have_override:
                    # Sidecar is in original-image space; scale INTO the
                    # preprocessed (rescaled) space so it aligns with preproc_path.
                    parsed = _parse_segmentation_geojson(override_path)
                    if rf != 1.0:
                        parsed = [{"class": p["class"], "geometry": _scale_geom(p["geometry"], rf)} for p in parsed]
                else:
                    seg_path = getattr(result, "segmentation_geojson_path", None)
                    parsed = _parse_segmentation_geojson(seg_path) if seg_path and Path(seg_path).is_file() else []

                image = imread_any(preproc_path)
                if image is None:
                    raise IOError(f"Could not load preprocessed image: {preproc_path}")
                # imread_any returns BGR(A) (cv2 convention) but napari expects RGB(A).
                if image.ndim == 3 and image.shape[2] == 3:
                    image = np.ascontiguousarray(image[..., ::-1])
                elif image.ndim == 3 and image.shape[2] == 4:
                    image = np.ascontiguousarray(image[..., [2, 1, 0, 3]])
                mask = self._rasterize(parsed, image.shape[:2])
            finally:
                shutil.rmtree(tmp_dir, ignore_errors=True)

            return {"image": image, "mask": mask, "rescale_factor": rf}

        self._begin_load(work)

    def _render_payload(self, payload: dict) -> None:
        self._rescale_factor = payload.get("rescale_factor", 1.0) or 1.0
        self._render(payload["image"], payload["mask"])

    def _fail_load(self, message: str) -> None:
        QMessageBox.critical(self, "Could not load segmentation", message)
        # Don't tear the whole dialog down — the Landmarks tab may still be
        # usable, and in cohort mode the user can step to another image.

    def _color_dict(self) -> dict:
        d = {None: (0.0, 0.0, 0.0, 0.0), 0: (0.0, 0.0, 0.0, 0.0)}
        for cls, idx in SEG_CLASS_INDEX.items():
            r, g, b = _hex_to_rgb(SEG_CLASS_COLORS[cls])
            d[idx] = (r, g, b, 1.0)
        return d

    def _rasterize(self, parsed: list, shape: tuple):
        """Burn vein/intervein polygons (holes respected) into a label mask."""
        import numpy as np
        from rasterio.features import rasterize as rio_rasterize

        shapes = []
        for feat in parsed:
            val = SEG_CLASS_INDEX.get(feat.get("class"), 0)
            geom = feat.get("geometry")
            if not val or not geom or not geom.get("coordinates"):
                continue
            shapes.append((geom, val))
        if not shapes:
            return np.zeros(shape, dtype=np.uint8)
        # Burn lower class indices first so veins (3) win at shared borders.
        shapes.sort(key=lambda s: s[1])
        try:
            return rio_rasterize(shapes, out_shape=shape, fill=0, dtype="uint8")
        except Exception:
            return np.zeros(shape, dtype=np.uint8)

    def _render(self, image, mask) -> None:
        """Build the image + Labels layers from a preloaded (RGB) preprocessed image + mask."""
        import napari
        from napari.utils.colormaps import DirectLabelColormap

        if self._viewer is None:
            self._viewer = napari.Viewer(show=False)
            qt_viewer = self._viewer.window.qt_viewer
            self.layout().removeWidget(self._viewer_placeholder)
            self._viewer_placeholder.hide()
            self.layout().addWidget(qt_viewer, stretch=1)

        self._viewer.layers.clear()
        self._viewer.add_image(image, name=self._image_path.name, rgb=image.ndim == 3)
        self._labels_layer = self._viewer.add_labels(
            mask,
            name="vein / intervein",
            colormap=DirectLabelColormap(color_dict=self._color_dict()),
            opacity=0.5,
        )
        self._labels_layer.selected_label = self._current_class_value()
        self._labels_layer.brush_size = self.spin_brush.value()
        self._labels_layer.mode = "paint"
        try:
            self._viewer.layers.selection.active = self._labels_layer
        except Exception:
            pass
        self._baseline_mask = mask.copy()
        self._last_saved_mask = mask.copy()

    # -- editing actions -------------------------------------------------
    def _current_class_value(self) -> int:
        return SEG_CLASS_INDEX.get(self.cmb_class.currentData() or self.cmb_class.currentText(), 3)

    def _on_class_changed(self, *_args) -> None:
        if self._labels_layer is not None:
            self._labels_layer.selected_label = self._current_class_value()
        s = self._settings()
        if s is not None:
            data = self.cmb_class.currentData()
            if data:
                s.setValue("inspector/seg_class", data)

    def _on_brush_changed(self, value: int) -> None:
        if self._labels_layer is not None:
            self._labels_layer.brush_size = value
        s = self._settings()
        if s is not None:
            s.setValue("inspector/seg_brush_size", int(value))

    def _set_mode(self, mode: str) -> None:
        if self._labels_layer is None:
            return
        if mode in ("paint", "fill"):
            self._labels_layer.selected_label = self._current_class_value()
        try:
            self._labels_layer.mode = mode
        except Exception:
            pass

    def _on_undo(self) -> None:
        if self._labels_layer is not None:
            try:
                self._labels_layer.undo()
            except Exception:
                pass

    def _on_redo(self) -> None:
        if self._labels_layer is not None:
            try:
                self._labels_layer.redo()
            except Exception:
                pass

    def restore(self) -> None:
        if self._labels_layer is None or self._baseline_mask is None:
            return
        self._labels_layer.data = self._baseline_mask.copy()

    # -- saving ----------------------------------------------------------
    def save_override(self) -> Optional[Path]:
        import numpy as np

        if self._labels_layer is None:
            raise RuntimeError("Viewer not initialized; nothing to save.")
        mask = np.asarray(self._labels_layer.data)

        # Re-vectorize the painted mask with the SAME converter the pipeline
        # uses, so the override is byte-for-byte the format Stage 5 produces.
        from modeltojson import mask_to_geojson

        classes_meta = [
            {"index": idx, "name": name, "color": SEG_CLASS_COLORS.get(name, "#888888")}
            for name, idx in SEG_CLASS_INDEX.items()
        ]
        fc = mask_to_geojson(mask, classes_meta, str(self._image_path))
        features_out = list(fc.get("features", [])) if isinstance(fc, dict) else []
        # The mask is in preprocessed (rescaled) pixel space; the sidecar is
        # stored in ORIGINAL-image space (Stage 5 multiplies back by
        # rescale_factor). Divide the vectorized coords by the rescale factor so
        # the on-disk contract is unchanged and resolution-independent.
        rf = self._rescale_factor or 1.0
        for f in features_out:
            if rf != 1.0 and isinstance(f.get("geometry"), dict):
                f["geometry"] = _scale_geom(f["geometry"], 1.0 / rf)
            f.setdefault("properties", {})["gate_reason"] = "manual override"

        if not features_out:
            reply = QMessageBox.question(
                self,
                "Save with no polygons?",
                "There are no vein/intervein regions to save. An empty override "
                "means the next run finds no veins for this image.\n\nSave anyway?",
                QMessageBox.Save | QMessageBox.Cancel,
            )
            if reply != QMessageBox.Save:
                return None

        payload = {"type": "FeatureCollection", "features": features_out}
        override_path = self._image_path.parent / f"{self._image_path.stem}_segmentation_override.geojson"
        override_path.write_text(json.dumps(payload), encoding="utf-8")
        self._last_saved_mask = mask.copy()
        return override_path

    # -- dirty state / lifecycle -----------------------------------------
    def has_unsaved_changes(self) -> bool:
        import numpy as np

        if self._labels_layer is None or self._last_saved_mask is None:
            return False
        return not np.array_equal(np.asarray(self._labels_layer.data), self._last_saved_mask)

    def shutdown(self) -> None:
        self._bump_load_token()  # discard any in-flight load result
        if self._viewer is not None:
            try:
                self._viewer.close()
            except Exception:
                pass
            self._viewer = None
