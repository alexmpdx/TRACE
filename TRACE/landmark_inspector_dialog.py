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

import json
import shutil
import tempfile
from pathlib import Path
from typing import Optional

from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QKeySequence
from PyQt5.QtWidgets import (
    QApplication,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QPushButton,
    QShortcut,
    QSpinBox,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

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


class LandmarkEditorWidget(QWidget):
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
        self._last_saved_snapshot: Optional[tuple] = None
        self._loaded = False
        self._build_ui()

    # -- lazy load protocol (shared shape with SegmentationEditorWidget) ---
    def is_loaded(self) -> bool:
        return self._loaded

    def set_image(self, new_image_path) -> None:
        """Point the editor at a new image WITHOUT loading (cohort swap)."""
        self._image_path = Path(new_image_path)
        self._points_layer = None
        self._predicted_positions = {}
        self._last_saved_snapshot = None
        self._loaded = False

    def ensure_loaded(self) -> None:
        if self._loaded:
            return
        self.load_or_generate()
        self._loaded = True

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
        controls.addStretch(1)
        layout.addLayout(controls)

        self._viewer_placeholder = QLabel(
            "Loading image and predicted landmarks…\n\n"
            "First open may take a few seconds while LandmarkLocator initializes."
        )
        self._viewer_placeholder.setAlignment(Qt.AlignCenter)
        self._viewer_placeholder.setStyleSheet("color: #888; padding: 40px;")
        layout.addWidget(self._viewer_placeholder, stretch=1)
        self._canvas_embedded = False

    # -- loading ----------------------------------------------------------
    def load_or_generate(self) -> None:
        """Locate predictions for self._image_path and (re)render the viewer.

        Search order — reopening a previously-edited image starts from the
        override so iterative refinement works:
          1. <image_dir>/<stem>_landmarks_override.geojson  ← prior corrections
          2. <output_folder>/<stem>_landmarks.geojson       ← post-run output
          3. <image_dir>/<stem>_landmarks.geojson           ← Stage-3 sidecar
          4. Generate via _generate_landmarks_for_image()    ← on-demand model
        """
        stem = self._image_path.stem
        image_dir = self._image_path.parent
        candidates = [
            image_dir / f"{stem}_landmarks_override.geojson",
            None,  # filled below if an output folder is set
            image_dir / f"{stem}_landmarks.geojson",
        ]
        try:
            out_text = self._window.output_edit.text().strip()
            if out_text:
                candidates[1] = Path(out_text) / f"{stem}_landmarks.geojson"
        except Exception:
            pass

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
            # No usable file found — regenerate via LandmarkLocator on demand.
            panel = getattr(self._window, "inline_custom_distances_panel", None)
            if panel is None:
                self._fail_load("Internal: landmark generation panel is unavailable.")
                return
            try:
                # disable_gates: this image likely has no override BECAUSE its
                # landmarks failed the confidence gate — generate the model's
                # best guess anyway so the user has points to drag into place.
                generated = panel._generate_landmarks_for_image(self._image_path, disable_gates=True)
                landmarks_dict = _parse_landmarks_geojson(generated)
            except Exception as exc:  # noqa: BLE001
                self._fail_load(
                    f"No landmarks file found and on-demand generation failed:\n\n{exc}\n\n"
                    "Configure a landmark model in Settings → Models and try again."
                )
                return

        self._predicted_positions = dict(landmarks_dict)
        self._render(landmarks_dict)
        self._last_saved_snapshot = self._snapshot_current()

    def _fail_load(self, message: str) -> None:
        QMessageBox.critical(self, "Could not load landmarks", message)
        # In cohort mode the user can step to the next image; only bail out of
        # the whole dialog in single-image mode.
        if self._dialog._cohort is None:
            self._dialog.reject()

    def _render(self, landmarks_dict: dict) -> None:
        """Create/clear the viewer and (re)populate image + points layers."""
        import napari
        import numpy as np
        from measurement_maker.landmark_names import landmark_display_name

        from TRACE.psd_loader import imread_any

        image = imread_any(self._image_path)
        if image is None:
            self._fail_load(f"Could not load image: {self._image_path}")
            return

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

        # imread_any returns BGR(A) (cv2 convention) but napari expects RGB(A).
        # Without this flip, warm-toned brightfield images render with a blue tint.
        if image.ndim == 3 and image.shape[2] == 3:
            image = np.ascontiguousarray(image[..., ::-1])
        elif image.ndim == 3 and image.shape[2] == 4:
            image = np.ascontiguousarray(image[..., [2, 1, 0, 3]])

        self._viewer.layers.clear()
        self._viewer.add_image(image, name=self._image_path.name, rgb=image.ndim == 3)

        raw_names = list(landmarks_dict.keys())
        # napari expects (row, col) = (y, x) ordering for point coordinates.
        coords_yx = np.array([[y, x] for (x, y) in landmarks_dict.values()], dtype=float)
        if coords_yx.size == 0:
            coords_yx = np.empty((0, 2), dtype=float)
        self._points_layer = self._viewer.add_points(
            coords_yx,
            name="landmarks",
            size=90,
            face_color="cyan",
            border_color="black",
            border_width=0.15,
            features={
                "name": list(raw_names),
                "label": [landmark_display_name(n) for n in raw_names],
            },
            text={
                "string": "{label}",
                "size": 12,
                "color": "yellow",
                "translation": [-30, 0],
            },
        )
        # CRITICAL DIFFERENCE from LandmarkPickerWidget: select mode stays on
        # and NO snap-back callback is installed. The user is here to edit.
        self._points_layer.mode = "select"
        try:
            self._viewer.layers.selection.active = self._points_layer
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
        current.append(raw_name)
        labels.append(landmark_display_name(raw_name))
        self._points_layer.data = new_coords
        self._points_layer.features = {"name": current, "label": labels}

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

    def restore_predictions(self) -> None:
        if self._points_layer is None:
            return
        import numpy as np
        from measurement_maker.landmark_names import landmark_display_name

        raw_names = list(self._predicted_positions.keys())
        coords_yx = np.array([[y, x] for (x, y) in self._predicted_positions.values()], dtype=float)
        if coords_yx.size == 0:
            coords_yx = np.empty((0, 2), dtype=float)
        self._points_layer.data = coords_yx
        self._points_layer.features = {
            "name": list(raw_names),
            "label": [landmark_display_name(n) for n in raw_names],
        }

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


class SegmentationEditorWidget(QWidget):
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
    a per-class label *mask* over the ORIGINAL image, where holes are inherent.
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
        self._loaded = False
        self._build_ui()

    # -- lazy load protocol ----------------------------------------------
    def is_loaded(self) -> bool:
        return self._loaded

    def set_image(self, new_image_path) -> None:
        self._image_path = Path(new_image_path)
        self._labels_layer = None
        self._baseline_mask = None
        self._last_saved_mask = None
        self._loaded = False

    def ensure_loaded(self) -> None:
        if self._loaded:
            return
        self.load_or_generate()
        self._loaded = True

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

    # -- loading ---------------------------------------------------------
    def load_or_generate(self) -> None:
        """Obtain the segmentation for this image and render it as a paint mask.

        A prior ``<stem>_segmentation_override.geojson`` (already in original
        space) is loaded directly. Otherwise the configured preprocessing chain
        is run on demand (rotation OFF, gates OFF) and its segmentation mapped
        into original space. Any unsaved landmark edits are flushed first so
        Stage 3 picks them up and the hinge chop uses the corrected landmarks.
        """
        self._dialog.persist_landmark_edits_for_pipeline()

        override_path = self._image_path.parent / f"{self._image_path.stem}_segmentation_override.geojson"

        # Let the placeholder paint before the (possibly blocking) work.
        QApplication.processEvents()

        if override_path.is_file():
            # Saved overrides are already in original-image pixel space — no
            # preprocessing needed, just re-rasterize them.
            try:
                parsed = _parse_segmentation_geojson(override_path)
            except Exception as exc:  # noqa: BLE001
                self._fail_load(f"Could not read the saved segmentation override:\n\n{exc}")
                return
        else:
            tmp_dir = Path(tempfile.mkdtemp(prefix="trace_seg_inspect_"))
            try:
                try:
                    result = self._window.run_single_image_preprocessing_for_segmentation(
                        self._image_path, tmp_dir, with_segmentation=True
                    )
                except Exception as exc:  # noqa: BLE001
                    self._fail_load(
                        f"Could not preprocess this image for vein/intervein inference:\n\n{exc}\n\n"
                        "Check that a landmark model and a segmentation model are configured "
                        "in Settings → Models."
                    )
                    return
                seg_path = getattr(result, "segmentation_geojson_path", None)
                parsed = _parse_segmentation_geojson(seg_path) if seg_path and Path(seg_path).is_file() else []
                # Map preprocessed-space polygons into original-image space.
                # Wing isolation / hinge chop don't move coords; only rescale does.
                rf = getattr(result, "rescale_factor", 1.0) or 1.0
                if rf != 1.0:
                    parsed = [{"class": p["class"], "geometry": _scale_geom(p["geometry"], 1.0 / rf)} for p in parsed]
            finally:
                shutil.rmtree(tmp_dir, ignore_errors=True)

        self._render(parsed)

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

    def _render(self, parsed: list) -> None:
        import napari
        import numpy as np
        from napari.utils.colormaps import DirectLabelColormap

        from TRACE.psd_loader import imread_any

        image = imread_any(self._image_path)
        if image is None:
            self._fail_load(f"Could not load image: {self._image_path}")
            return

        if self._viewer is None:
            self._viewer = napari.Viewer(show=False)
            qt_viewer = self._viewer.window.qt_viewer
            self.layout().removeWidget(self._viewer_placeholder)
            self._viewer_placeholder.hide()
            self.layout().addWidget(qt_viewer, stretch=1)

        # imread_any returns BGR(A) (cv2 convention) but napari expects RGB(A).
        if image.ndim == 3 and image.shape[2] == 3:
            image = np.ascontiguousarray(image[..., ::-1])
        elif image.ndim == 3 and image.shape[2] == 4:
            image = np.ascontiguousarray(image[..., [2, 1, 0, 3]])

        mask = self._rasterize(parsed, image.shape[:2])

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

    def _on_brush_changed(self, value: int) -> None:
        if self._labels_layer is not None:
            self._labels_layer.brush_size = value

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
        for f in features_out:
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
        if self._viewer is not None:
            try:
                self._viewer.close()
            except Exception:
                pass
            self._viewer = None
