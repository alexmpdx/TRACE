"""Per-image landmark inspector + editor.

Three entry points (Main tab), all converging on
``TraceWindow._open_landmark_inspector(image_path, cohort=...)``:
  - Right-click a single image           → "Inspect & Edit landmarks…"
  - Right-click on a multi-selection      → "Inspect & Edit landmarks (N selected)…"
  - Post-run "Review failed images" button alongside the rerun-failed buttons

Cohort mode (>=2 images) shows prev/next navigation in the dialog footer and
reuses a single napari Viewer across images (construction is expensive).

The user drags predicted landmarks to corrected positions, optionally adds a
missing landmark from the canonical name list, hits Save → writes
``<image_dir>/<stem>_landmarks_override.geojson``. The next batch run picks up
the override automatically (Stage-3 short-circuit in preprocessing/pipeline.py).

This is a sibling of measurementMaker's ``LandmarkPickerWidget`` — it shares the
napari setup pattern (image + Points layer + text labels) but deliberately does
NOT install the snap-back guard that makes the picker read-only. Drag-to-correct
is the whole point here.

v1 scope: landmarks only. Vein/intervein editing is parked for v2; the dialog
structure leaves room for a future QTabWidget with a "Segmentation" tab.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtWidgets import (
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
    QWidget,
)


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
    """Modal wrapper: holds the editor widget + (cohort) navigation + buttons."""

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
        self.setWindowTitle(f"Inspect & Edit landmarks — {self._image_path.name}")
        self.resize(1320, 800)

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
        self._editor = LandmarkEditorWidget(self, self._image_path)
        content.addWidget(self._editor, stretch=1)
        layout.addLayout(content, stretch=1)

        # Navigation row (cohort mode only).
        self._build_navigation()

        # Save / Restore / Cancel buttons.
        btns = QDialogButtonBox(self)
        self.btn_save = btns.addButton("Save override", QDialogButtonBox.AcceptRole)
        self.btn_restore = btns.addButton("Restore predictions", QDialogButtonBox.ResetRole)
        self.btn_cancel = btns.addButton(QDialogButtonBox.Cancel)
        btns.accepted.connect(self._on_save)
        btns.rejected.connect(self.reject)
        self.btn_restore.clicked.connect(self._editor.restore_predictions)
        layout.addWidget(btns)

        # Defer the heavy load until after the dialog paints. napari Viewer
        # construction + LandmarkLocator first-call init together can take
        # several seconds; showing the dialog first gives the user feedback.
        QTimer.singleShot(0, self._editor.load_or_generate)

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
        save unsaved edits before leaving the current image.
        """
        if self._cohort is None:
            return False
        if not (0 <= new_idx < len(self._cohort)) or new_idx == self._cohort_index:
            return False
        if self._editor.has_unsaved_changes():
            reply = QMessageBox.question(
                self,
                "Unsaved changes",
                "You have unsaved edits on this image. Save before moving on?",
                QMessageBox.Save | QMessageBox.Discard | QMessageBox.Cancel,
            )
            if reply == QMessageBox.Cancel:
                return False
            if reply == QMessageBox.Save:
                try:
                    self._editor.save_override()
                except Exception as exc:  # noqa: BLE001
                    QMessageBox.critical(self, "Save failed", f"{exc}")
                    return False
                self._saved_indices.add(self._cohort_index)
        self._cohort_index = new_idx
        self._image_path = self._cohort[new_idx]
        self.setWindowTitle(f"Inspect & Edit landmarks — {self._image_path.name}")
        self._refresh_nav_state()
        self._editor.swap_image(self._image_path)
        return True

    def _on_save_and_next(self) -> None:
        try:
            self._editor.save_override()
        except Exception as exc:  # noqa: BLE001
            QMessageBox.critical(self, "Save failed", f"{exc}")
            return
        self._saved_indices.add(self._cohort_index)
        self._refresh_image_list()
        self._navigate(+1)

    def _on_save(self) -> None:
        try:
            override_path = self._editor.save_override()
        except Exception as exc:  # noqa: BLE001
            QMessageBox.critical(self, "Save failed", f"Could not write override: {exc}")
            return
        if override_path is None:
            return  # user declined the empty-landmark confirm
        if self._cohort is not None:
            self._saved_indices.add(self._cohort_index)
            self._refresh_image_list()
        QMessageBox.information(
            self,
            "Saved",
            f"Landmark override saved to:\n\n{override_path}\n\n"
            "The next run that includes this image will use the override "
            "instead of running LandmarkLocator on it.",
        )
        # In cohort mode plain "Save" should NOT close the dialog — the user
        # may want to keep reviewing. Only single-image mode auto-closes.
        if self._cohort is None:
            self.accept()

    def closeEvent(self, event) -> None:  # noqa: N802 (Qt override)
        self._editor.shutdown()
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
        self._build_ui()

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
                generated = panel._generate_landmarks_for_image(self._image_path)
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

        self._viewer.layers.clear()
        self._viewer.add_image(image, name=self._image_path.name)

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

    def swap_image(self, new_image_path) -> None:
        """Replace the current image + landmarks, reusing self._viewer."""
        self._image_path = Path(new_image_path)
        self._points_layer = None
        self._predicted_positions = {}
        self._last_saved_snapshot = None
        self.load_or_generate()

    def shutdown(self) -> None:
        if self._viewer is not None:
            try:
                self._viewer.close()
            except Exception:
                pass
            self._viewer = None
