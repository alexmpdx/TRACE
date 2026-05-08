"""Embeddable napari-based landmark distance picker.

Exposes a single QWidget that hosts:
  - file pickers for a sample wing image + matching landmarks GeoJSON
  - an embedded napari canvas (lazy-created on first load)
  - a pair-management panel (label entry, Add/Remove/Clear)

The widget owns the configured pairs as a `list[LandmarkPair]` accessible via
`pairs()`. Mutations emit `pairs_changed`. Use `set_pairs()` to seed.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

from measurement_maker.distance import load_landmarks_from_geojson
from measurement_maker.types import LandmarkPair
from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtWidgets import (
    QCheckBox,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QMessageBox,
    QPushButton,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

logger = logging.getLogger(__name__)


class LandmarkPickerWidget(QWidget):
    """Embeddable napari-based picker.

    The napari Viewer is created lazily on the first call to `_load_wing`
    (i.e. when the user has chosen both an image and a landmarks GeoJSON and
    clicks "Load wing into viewer"). Subsequent loads replace layer contents
    on the existing Viewer rather than recreating it, so the embedded canvas
    stays mounted in the tab.
    """

    pairs_changed = pyqtSignal(list)

    def __init__(
        self,
        parent: Optional[QWidget] = None,
        initial_pairs: Optional[list[LandmarkPair]] = None,
        default_image_dir: str = "",
    ):
        super().__init__(parent)
        self._pairs: list[LandmarkPair] = list(initial_pairs or [])
        self._default_image_dir = default_image_dir
        self._viewer = None  # napari.Viewer (lazy)
        self._points_layer = None
        self._line_layer = None  # Shapes layer holding the highlight line
        self._names: list[str] = []
        # Snap-back state: keep landmark positions immutable while still
        # allowing the user to click-select them. (Setting `editable=False`
        # would disable selection too in napari 0.7+.)
        self._original_coords = None
        self._snapping_back = False
        self._build_ui()
        self._refresh_list()

    # --- Public API ---------------------------------------------------------
    def pairs(self) -> list[LandmarkPair]:
        """Return a copy of the currently configured pairs."""
        return list(self._pairs)

    def set_pairs(self, pairs: list[LandmarkPair]):
        """Replace the configured pairs (e.g. when re-opening the dialog)."""
        self._pairs = list(pairs)
        self._refresh_list()
        self.pairs_changed.emit(list(self._pairs))

    def set_default_image_dir(self, path: str):
        """Suggest a starting directory for the file pickers."""
        self._default_image_dir = path or ""

    # --- UI construction ---------------------------------------------------
    def _build_ui(self):
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(6)

        # Row 1: image picker
        row = QHBoxLayout()
        row.addWidget(QLabel("Sample image:"))
        self._image_edit = QLineEdit()
        self._image_edit.setReadOnly(True)
        self._image_edit.setPlaceholderText("Pick a wing image…")
        btn_image = QPushButton("Browse…")
        btn_image.clicked.connect(self._select_image)
        row.addWidget(self._image_edit, stretch=1)
        row.addWidget(btn_image)
        outer.addLayout(row)

        # Row 2: landmarks picker
        row = QHBoxLayout()
        row.addWidget(QLabel("Landmarks GeoJSON:"))
        self._lm_edit = QLineEdit()
        self._lm_edit.setReadOnly(True)
        self._lm_edit.setPlaceholderText("Pick the matching *_landmarks.geojson…")
        btn_lm = QPushButton("Browse…")
        btn_lm.clicked.connect(self._select_landmarks)
        row.addWidget(self._lm_edit, stretch=1)
        row.addWidget(btn_lm)
        outer.addLayout(row)

        # Row 3: viewer options + load button
        row = QHBoxLayout()
        self._show_labels_chk = QCheckBox("Show landmark labels")
        self._show_labels_chk.setChecked(True)
        self._show_labels_chk.setToolTip("Toggle the yellow text labels next to each landmark.")
        self._show_labels_chk.toggled.connect(self._on_show_labels_toggled)
        row.addWidget(self._show_labels_chk)
        row.addStretch()
        self._load_btn = QPushButton("Load wing into viewer")
        self._load_btn.clicked.connect(self._load_clicked)
        row.addWidget(self._load_btn)
        outer.addLayout(row)

        # Vertical splitter: pair controls on top, viewer below.
        # Wings are much wider than tall, so giving the canvas the full row
        # width (rather than sharing it with a side panel) keeps the wing
        # readable without forcing the user to zoom every time.
        self._splitter = QSplitter(Qt.Vertical)
        outer.addWidget(self._splitter, stretch=1)

        # Top: pair controls laid out horizontally so the panel stays short.
        pair_panel = QWidget()
        pp = QHBoxLayout(pair_panel)
        pp.setContentsMargins(0, 0, 0, 0)
        pp.setSpacing(12)

        # Left column: instructions + selection state + label entry + Add.
        left_col = QVBoxLayout()
        left_col.setSpacing(4)
        left_col.addWidget(
            QLabel(
                "1. Click a landmark to select it (turns orange).\n"
                "2. SHIFT-click a second landmark.\n"
                "3. Type a label, click 'Add pair'."
            )
        )
        self._sel_label = QLabel("Selected: (none)")
        self._sel_label.setWordWrap(True)
        left_col.addWidget(self._sel_label)
        label_row = QHBoxLayout()
        label_row.addWidget(QLabel("Label:"))
        self._label_edit = QLineEdit()
        self._label_edit.setPlaceholderText("e.g. wing_span")
        label_row.addWidget(self._label_edit, stretch=1)
        left_col.addLayout(label_row)
        add_btn = QPushButton("Add pair")
        add_btn.clicked.connect(self._add_pair)
        left_col.addWidget(add_btn)
        # No bottom stretch — panel sizes to its natural content height so the
        # right column's bottom can align with the Add pair button.
        pp.addLayout(left_col, stretch=1)

        # Right column: configured pair list + Remove/Clear.
        # The list has a capped height so the column ends at roughly the same
        # height as the left column (instructions + selected + label + Add).
        right_col = QVBoxLayout()
        right_col.setSpacing(4)
        right_col.addWidget(QLabel("Configured pairs:"))
        self._list = QListWidget()
        self._list.setMaximumHeight(80)
        self._list.currentRowChanged.connect(self._on_pair_row_changed)
        right_col.addWidget(self._list)
        btns_row1 = QHBoxLayout()
        rem_btn = QPushButton("Remove")
        rem_btn.clicked.connect(self._remove_pair)
        clr_btn = QPushButton("Clear all")
        clr_btn.clicked.connect(self._clear_pairs)
        btns_row1.addWidget(rem_btn)
        btns_row1.addWidget(clr_btn)
        right_col.addLayout(btns_row1)

        btns_row2 = QHBoxLayout()
        save_btn = QPushButton("Save…")
        save_btn.setToolTip("Save the current pair list to a JSON file.")
        save_btn.clicked.connect(self._save_pairs_to_file)
        load_btn = QPushButton("Load…")
        load_btn.setToolTip("Load a previously-saved pair list from a JSON file.")
        load_btn.clicked.connect(self._load_pairs_from_file)
        btns_row2.addWidget(save_btn)
        btns_row2.addWidget(load_btn)
        right_col.addLayout(btns_row2)
        pp.addLayout(right_col, stretch=1)

        self._splitter.addWidget(pair_panel)

        # Bottom: container that will hold napari's QtViewer once a wing is loaded
        self._viewer_container = QWidget()
        viewer_layout = QVBoxLayout(self._viewer_container)
        viewer_layout.setContentsMargins(0, 0, 0, 0)
        self._viewer_placeholder = QLabel("Load a wing image + landmarks GeoJSON\nto view landmarks here.")
        self._viewer_placeholder.setAlignment(Qt.AlignCenter)
        self._viewer_placeholder.setStyleSheet("color: #888; background: #1e1e1e; padding: 40px;")
        viewer_layout.addWidget(self._viewer_placeholder)
        self._splitter.addWidget(self._viewer_container)

        # Pair panel small, viewer big — wing imagery dominates the tab.
        self._splitter.setStretchFactor(0, 1)
        self._splitter.setStretchFactor(1, 4)

    # --- File pickers ------------------------------------------------------
    # We force Qt's own (non-native) dialog because the macOS native dialog
    # mis-handles file selection when napari/qtpy is also loaded in the
    # process — clicks land but selection never commits. The Qt dialog is
    # ugly but reliable across platforms.
    _FILE_DIALOG_OPTS = QFileDialog.DontUseNativeDialog

    def _select_image(self):
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Select sample wing image",
            self._default_image_dir,
            "Wing images (*.tif *.tiff *.png *.jpg *.jpeg *.bmp *.psd);;All files (*)",
            options=self._FILE_DIALOG_OPTS,
        )
        if path:
            self._image_edit.setText(path)
            # Default the landmarks picker to the same folder.
            if not self._lm_edit.text():
                self._default_image_dir = str(Path(path).parent)

    def _select_landmarks(self):
        start_dir = str(Path(self._image_edit.text()).parent) if self._image_edit.text() else self._default_image_dir
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Select matching landmarks GeoJSON",
            start_dir,
            "Landmarks GeoJSON (*_landmarks.geojson *.geojson);;All files (*)",
            options=self._FILE_DIALOG_OPTS,
        )
        if path:
            self._lm_edit.setText(path)

    def _load_clicked(self):
        image_path = self._image_edit.text().strip()
        landmarks_path = self._lm_edit.text().strip()
        if not image_path or not landmarks_path:
            QMessageBox.warning(
                self,
                "Pick both files",
                "Choose a sample wing image AND a matching landmarks GeoJSON before loading.",
            )
            return
        try:
            self._load_wing(Path(image_path), Path(landmarks_path))
        except Exception as exc:
            logger.exception("Failed to load wing into picker")
            QMessageBox.critical(self, "Load failed", f"{exc}")

    # --- napari viewer management ------------------------------------------
    def _load_wing(self, image_path: Path, landmarks_path: Path):
        """Load (or reload) image + landmarks layers into the embedded viewer."""
        landmarks = load_landmarks_from_geojson(landmarks_path)
        if not landmarks:
            raise ValueError(f"No landmark points found in {landmarks_path}")

        from TRACE.psd_loader import imread_any

        image = imread_any(image_path)
        if image is None:
            raise ValueError(f"Could not load image: {image_path}")

        names = list(landmarks.keys())
        # napari expects (row, col) = (y, x) ordering for point coordinates.
        import numpy as np

        coords_yx = np.array([[y, x] for (x, y) in landmarks.values()], dtype=float)

        if self._viewer is None:
            self._create_viewer()

        # Replace any existing layers; this lets the user re-load a different
        # wing without tearing down the embedded canvas.
        self._viewer.layers.clear()
        self._viewer.add_image(image, name="wing")
        self._points_layer = self._viewer.add_points(
            coords_yx,
            name="landmarks",
            size=90,
            face_color="cyan",
            border_color="black",
            border_width=0.15,
            features={"name": names},
            text={
                "string": "{name}",
                "size": 12,
                "color": "yellow",
                "translation": [-30, 0],
            },
        )
        self._points_layer.mode = "select"
        self._names = names

        # Lock landmark positions via snap-back: any data mutation (drag, add,
        # delete) is reverted immediately. Setting editable=False would also
        # disable selection in napari 0.7+, which we still need.
        self._original_coords = coords_yx.copy()
        try:
            self._points_layer.events.data.connect(self._on_points_data_changed)
        except Exception:
            logger.debug("embedded_picker: could not connect data event", exc_info=True)

        try:
            self._points_layer.events.highlight.connect(self._update_selection_label)
            self._points_layer.events.highlight.connect(self._update_face_colors)
        except Exception:
            logger.debug("embedded_picker: could not connect highlight event", exc_info=True)
        self._update_selection_label()
        self._update_face_colors()
        # Apply the current label-toggle state to the freshly-created text layer.
        self._on_show_labels_toggled(self._show_labels_chk.isChecked())

        # Highlight line for the currently-selected configured pair. Created
        # last so it renders on top of the image and points.
        self._line_layer = self._viewer.add_shapes(
            name="distance",
            shape_type="line",
            edge_color="magenta",
            edge_width=8,
        )
        self._line_layer.editable = False
        # napari makes the most-recently-added layer the active one, which
        # routes mouse clicks away from the points layer and breaks landmark
        # selection. Force the points layer back to active so click-to-select
        # keeps working with the user-friendly select tool.
        try:
            self._viewer.layers.selection.active = self._points_layer
        except Exception:
            logger.debug("embedded_picker: could not set points layer active", exc_info=True)
        # If a row in the pair list is already selected (e.g. seeded from
        # initial_pairs and clicked before this load), redraw the line.
        self._on_pair_row_changed(self._list.currentRow())

    def _create_viewer(self):
        """Create the napari Viewer and reparent its canvas into the tab."""
        import napari

        # show=False keeps napari's own QMainWindow hidden — we only want the
        # embedded canvas (qt_viewer) inside our tab.
        self._viewer = napari.Viewer(show=False)
        qt_viewer = self._viewer.window.qt_viewer
        layout = self._viewer_container.layout()
        layout.removeWidget(self._viewer_placeholder)
        self._viewer_placeholder.hide()
        layout.addWidget(qt_viewer)

    def _on_points_data_changed(self, _event=None):
        """Revert any user-initiated mutation of the landmarks layer.

        Fires when coords change (drag), points are added (mode=add), or
        deleted. Guarded with a re-entrancy flag because reassigning
        `data` re-triggers this event.
        """
        if self._snapping_back or self._points_layer is None or self._original_coords is None:
            return
        import numpy as np

        if np.array_equal(self._points_layer.data, self._original_coords):
            return
        self._snapping_back = True
        try:
            self._points_layer.data = self._original_coords.copy()
        finally:
            self._snapping_back = False

    # --- Pair list management ----------------------------------------------
    def _selected_names(self) -> list[str]:
        if self._points_layer is None:
            return []
        indices = sorted(self._points_layer.selected_data)
        return [self._names[i] for i in indices if 0 <= i < len(self._names)]

    def _update_selection_label(self, _event=None):
        sel = self._selected_names()
        self._sel_label.setText("Selected: " + ", ".join(sel) if sel else "Selected: (none)")

    def _update_face_colors(self, _event=None):
        """Recolor selected landmarks orange; unselected stay cyan."""
        if self._points_layer is None or not self._names:
            return
        sel = self._points_layer.selected_data
        n = len(self._names)
        try:
            self._points_layer.face_color = ["orange" if i in sel else "cyan" for i in range(n)]
        except Exception:
            logger.debug("embedded_picker: face_color update failed", exc_info=True)

    def _on_show_labels_toggled(self, checked: bool):
        """Show/hide the yellow text labels next to each landmark."""
        if self._points_layer is None:
            return
        try:
            self._points_layer.text.visible = checked
        except Exception:
            logger.debug("embedded_picker: failed to toggle text visibility", exc_info=True)

    def _add_pair(self):
        if self._points_layer is None:
            QMessageBox.warning(self, "Load a wing first", "Load a sample wing image + landmarks before adding pairs.")
            return
        sel = self._selected_names()
        if len(sel) != 2:
            QMessageBox.warning(
                self,
                "Pick two landmarks",
                f"Select exactly two landmarks first (currently {len(sel)} selected).",
            )
            return
        if sel[0] == sel[1]:
            QMessageBox.warning(self, "Duplicate landmark", "Pick two distinct landmarks.")
            return
        label = self._label_edit.text().strip()
        if not label:
            QMessageBox.warning(self, "Missing label", "Type a label for this pair.")
            return
        self._pairs.append(LandmarkPair(name_a=sel[0], name_b=sel[1], label=label))
        self._label_edit.clear()
        self._refresh_list()
        # Select the new row so the highlight line draws immediately.
        self._list.setCurrentRow(len(self._pairs) - 1)
        self.pairs_changed.emit(list(self._pairs))

    def _remove_pair(self):
        row = self._list.currentRow()
        if 0 <= row < len(self._pairs):
            self._pairs.pop(row)
            self._refresh_list()
            self.pairs_changed.emit(list(self._pairs))

    def _clear_pairs(self):
        if not self._pairs:
            return
        confirm = QMessageBox.question(
            self,
            "Clear all pairs?",
            f"Remove all {len(self._pairs)} configured pair(s)?",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )
        if confirm == QMessageBox.Yes:
            self._pairs.clear()
            self._refresh_list()
            self.pairs_changed.emit(list(self._pairs))

    def _refresh_list(self):
        self._list.clear()
        for pair in self._pairs:
            self._list.addItem(f"{pair.label}: {pair.name_a} ↔ {pair.name_b}")

    # --- Highlight line ----------------------------------------------------
    def _on_pair_row_changed(self, row: int):
        """When the user picks a pair in the list, draw a line between its endpoints.

        Called with row=-1 when no pair is selected (e.g. after Clear all),
        in which case the line is removed.
        """
        if self._line_layer is None or self._original_coords is None:
            return
        self._clear_line()
        if row < 0 or row >= len(self._pairs):
            return
        pair = self._pairs[row]
        if pair.name_a not in self._names or pair.name_b not in self._names:
            # Pair references landmarks not present in the loaded wing —
            # leave the line cleared without bothering the user.
            return
        idx_a = self._names.index(pair.name_a)
        idx_b = self._names.index(pair.name_b)
        coords = self._original_coords
        y1, x1 = coords[idx_a]
        y2, x2 = coords[idx_b]
        import numpy as np

        self._line_layer.add(
            np.array([[y1, x1], [y2, x2]]),
            shape_type="line",
        )

    def _clear_line(self):
        if self._line_layer is None or len(self._line_layer.data) == 0:
            return
        self._line_layer.selected_data = set(range(len(self._line_layer.data)))
        self._line_layer.remove_selected()

    # --- Save / Load -------------------------------------------------------
    def _save_pairs_to_file(self):
        if not self._pairs:
            QMessageBox.information(self, "No pairs", "No pairs configured to save.")
            return
        path, _ = QFileDialog.getSaveFileName(
            self,
            "Save landmark distance pairs",
            self._default_image_dir,
            "JSON (*.json);;All files (*)",
            options=self._FILE_DIALOG_OPTS,
        )
        if not path:
            return
        if not path.lower().endswith(".json"):
            path = path + ".json"
        try:
            import json

            from measurement_maker import pairs_to_dicts

            with open(path, "w") as fh:
                json.dump(pairs_to_dicts(self._pairs), fh, indent=2)
        except Exception as exc:
            logger.exception("Failed to save pairs")
            QMessageBox.critical(self, "Save failed", str(exc))

    def _load_pairs_from_file(self):
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Load landmark distance pairs",
            self._default_image_dir,
            "JSON (*.json);;All files (*)",
            options=self._FILE_DIALOG_OPTS,
        )
        if not path:
            return
        try:
            import json

            from measurement_maker import pairs_from_dicts

            with open(path) as fh:
                data = json.load(fh)
            if not isinstance(data, list):
                raise ValueError("Expected a JSON list of {name_a, name_b, label} entries.")
            loaded = pairs_from_dicts(data)
            if not loaded:
                raise ValueError("File contains no valid pairs.")
        except Exception as exc:
            logger.exception("Failed to load pairs")
            QMessageBox.critical(self, "Load failed", str(exc))
            return
        if self._pairs:
            confirm = QMessageBox.question(
                self,
                "Replace configured pairs?",
                f"Replace the {len(self._pairs)} currently configured pair(s) with the "
                f"{len(loaded)} loaded pair(s)?",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.Yes,
            )
            if confirm != QMessageBox.Yes:
                return
        self.set_pairs(loaded)

    # --- Cleanup ------------------------------------------------------------
    def closeEvent(self, event):  # noqa: N802 — Qt API
        # Close the napari viewer to free GPU resources when the tab/dialog goes away.
        if self._viewer is not None:
            try:
                self._viewer.close()
            except Exception:
                logger.debug("embedded_picker: viewer.close() failed", exc_info=True)
        super().closeEvent(event)
