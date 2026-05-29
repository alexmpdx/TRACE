# Landmark Inspector & Editor — Implementation Spec

**Status:** Implementation-ready spec. v1 scope: landmarks only. Vein/intervein editing parked for v2.

**For the implementing Claude session:** This document is self-contained. Read it cold; everything you need to start work is below, including project layout, file:line citations, and decision rationale. The parallel session that wrote this spec is in a different working directory and you don't have access to its conversation memory — that's intentional. Treat this file as the source of truth.

---

## Project context

You're working inside **mapThemVeins**, a Drosophila wing-morphology analysis suite. Project root: `/Users/alexmurphy/claude_scripts/mapThemVeins`. Top-level orchestrator is `TRACE/`; sibling modules (`preprocessing/`, `LandmarkLocator/`, `identifyFeatures/`, `measurementMaker/`, etc.) implement individual pipeline stages.

Key modules you'll touch or reference:
- `TRACE/gui.py` — main PyQt5 window
- `TRACE/inline_panels.py` — Custom Measurements + Help tabs
- `TRACE/pipeline.py` — top-level orchestrator (Stage 1 preprocessing + Stage 2 identifyFeatures)
- `preprocessing/pipeline.py` — preprocessing stages 1–6
- `measurementMaker/measurement_maker/embedded_picker.py` — existing napari embed (do NOT refactor this; build a sibling)
- `measurementMaker/measurement_maker/landmark_names.py` — canonical landmark name dictionary
- `LandmarkLocator/landmark_locator/` — landmark detection model + inference
- `TRACE/run_gui.py` — entry point that sets up `sys.path` for all sibling modules; import patterns rely on this being run first

Code style (per project CLAUDE.md): black + isort + flake8 at 120-char line length. `pre-commit run --all-files` before committing.

---

## Background: what's around this feature

Three recently-shipped TODOs are relevant context for this implementation:

- **TODO #10** (shipped): right-click context menu on `TraceWindow.image_list` for skip/unskip. The handler `_on_image_list_context_menu(pos)` lives at `TRACE/gui.py:1272`. You'll extend it with a new menu action.
- **TODO #11** (shipped, commit `2da1ac33`): "Rerun failed images" + "Rerun failed (no gate aborts)" buttons that appear below the Run Pipeline button after a run with failed images. They're driven by `self._last_run_failed_set` and a `_refresh_rerun_buttons()` method. You'll add a sibling "Review failed images" button alongside these — same visibility logic, same row, same walkthrough-hint pattern.
- **TODO #6** (shipped, commit `e91b370e`): `landmarks_geojson` and `segmentation_geojson` are now user-selectable outputs at `TRACE/pipeline.py:39, 41`. The landmarks GeoJSON for a completed image lands at `<output_folder>/<stem>_landmarks.geojson` (`TRACE/pipeline.py:976–989`).

The existing napari embed (`measurementMaker/measurement_maker/embedded_picker.py:LandmarkPickerWidget`) is **read-only by design** — it has a snap-back guard at lines 497–514 that immediately reverts any data mutation. **Do not refactor it**. Build a sibling widget instead.

---

## v1 Primer

Right-click any image (or multi-select images) in the Main-tab image list → "Inspect & Edit landmarks…" → modal opens a napari viewer with the image + predicted landmarks. User drags any landmark to correct its position, optionally adds a missing landmark from the canonical 13-name dropdown, hits **Save**. A sidecar file `<image_dir>/<stem>_landmarks_override.geojson` is written next to the source image. On the next batch run, Stage 3 of preprocessing detects the override and uses it directly instead of running LandmarkLocator on that image.

The viewer works both pre-run (auto-generates predictions on first open via an existing helper) and post-run (loads from the output folder's GeoJSON if present, falls back to regeneration).

Three entry points, all converging on the same launcher `_open_landmark_inspector(image_path, cohort=None)`:
1. Right-click on a single row → single-image mode
2. Multi-select + right-click → cohort mode with all selected rows
3. Post-run "Review failed images (N)" button → cohort mode with the failed set

In cohort mode (≥2 images), the dialog footer shows Prev / Next / Save & Next navigation + a counter, and the user can step through without closing the dialog. A single napari viewer is reused across cohort images via `swap_image(new_path)` since napari construction is expensive.

v2 (parked) adds vein/intervein editing as a sibling tab inside the same dialog.

---

## Concrete edits

### 1. `preprocessing/pipeline.py` — Stage 3 override detection

At the Stage 3 entry point (around line 769–800, inside the `if do_landmarks:` block, immediately before the `run_landmarks(...)` call at line 788):

```python
# Manual override: if the user inspected/corrected this image's landmarks
# via the inspector dialog, a sidecar file lives next to the source image.
# Skip the predictor entirely and load the corrected landmarks.
override_path = image_path.parent / f"{image_path.stem}_landmarks_override.geojson"
if override_path.is_file():
    logger.info("%s: using manual landmark override from %s", stem, override_path)
    landmarks, landmark_metadata = load_landmarks_from_geojson(override_path)
    # The output path expected by downstream stages still needs to exist —
    # copy the override to the canonical landmarks_geojson location so
    # Stage 4 (hinge chop) and Stage 5 (segmentation) see the same data.
    shutil.copy2(override_path, landmarks_output_path)
else:
    landmarks, landmark_metadata = run_landmarks(image_path, ...)
```

Exact local variable names depend on the surrounding code; verify against the actual function. If `load_landmarks_from_geojson()` doesn't already exist as a Stage-3 helper, write a small `_load_landmark_dict_from_geojson(path) -> tuple[dict, dict]` returning `({name: (x, y)}, metadata_dict)`. The hinge-chop fallback at `preprocessing/pipeline.py:806–809` already follows the same image-dir-sidecar pattern — that's the precedent justifying the design.

### 2. `TRACE/gui.py` — extend the right-click context menu

The handler `_on_image_list_context_menu(pos)` already exists at `TRACE/gui.py:1272` (TODO #10 added it). Around lines 1284–1297 there's the menu-action setup and the `if chosen is …` dispatch.

Add a new menu item with a dynamic label and the launcher:

```python
# Compute the label based on selection count so the user knows what will happen.
n_selected = len(self.image_list.selectedItems())
inspect_label = (
    "Inspect & Edit landmarks…" if n_selected <= 1
    else f"Inspect & Edit landmarks ({n_selected} selected)…"
)
menu.addSeparator()
act_inspect = menu.addAction(inspect_label)

# ... after menu.exec_() ...

elif chosen is act_inspect:
    selected = self.image_list.selectedItems()
    target_item = self.image_list.itemAt(pos)
    # Multi-select cohort if ≥2 selected AND the clicked row is part of the
    # selection. Otherwise treat as single-image on the clicked row only.
    if len(selected) > 1 and target_item in selected:
        paths = []
        for item in selected:
            row = self.image_list.row(item)
            if 0 <= row < len(self._image_paths):
                paths.append(self._image_paths[row])
        if paths:
            self._open_landmark_inspector(paths[0], cohort=paths)
    elif target_item is not None:
        row = self.image_list.row(target_item)
        if 0 <= row < len(self._image_paths):
            self._open_landmark_inspector(self._image_paths[row])
```

The "clicked row must be part of the selection" guard matches typical OS behavior: right-clicking outside the selection operates on the clicked row only.

Add the launcher method on `TraceWindow`:

```python
def _open_landmark_inspector(self, image_path: Path, cohort: Optional[list[Path]] = None) -> None:
    from TRACE.landmark_inspector_dialog import LandmarkInspectorDialog
    dlg = LandmarkInspectorDialog(self, image_path, cohort=cohort)
    dlg.exec()
```

### 3. `TRACE/gui.py` — post-run "Review failed images" button

In the same area where TODO #11's `btn_rerun_failed` and `btn_rerun_failed_nogate` were added (find the row layout containing those — the exact line numbers aren't fixed in this spec since you'll need to inspect the post-#11 codebase), append:

```python
self.btn_review_failed = QPushButton("Review failed images")
self.btn_review_failed.setToolTip(
    "Open the landmark inspector on each failed image so you can correct "
    "the landmarks and save per-image overrides. The next run that includes "
    "these images will use your overrides instead of running LandmarkLocator."
)
self.btn_review_failed.clicked.connect(self._review_failed_images)
self.btn_review_failed.setVisible(False)
# Add to the same QHBoxLayout the rerun-failed buttons live in.
```

Extend `_refresh_rerun_buttons()` (or whatever TODO #11 named it) to also manage this button's visibility + label + first-time hint:

```python
def _refresh_rerun_buttons(self) -> None:
    failed = self._last_run_failed_set
    has_failed = bool(failed)
    self.btn_rerun_failed.setVisible(has_failed)
    # ... existing gate-failure check for btn_rerun_failed_nogate ...

    # NEW: review-failed button
    self.btn_review_failed.setVisible(has_failed)
    if has_failed:
        self.btn_review_failed.setText(f"Review failed images ({len(failed)})")

    # First-time walkthrough hint. Place after the existing rerun-button
    # hints in the chain so the user sees one hint per session in natural
    # feature progression.
    if has_failed and not self.settings.value(
        "review_failed_walkthrough_seen", False, type=bool
    ):
        QTimer.singleShot(300, lambda: self._show_button_hint(
            self.btn_review_failed,
            settings_key="review_failed_walkthrough_seen",
            title="New: Review failed images",
            body=(
                "You can also open the failed images one at a time in the "
                "landmark inspector to drag the landmarks into place manually. "
                "Saved corrections become per-image overrides that the next "
                "run picks up automatically — no need to retrain the model."
            ),
        ))
```

Add the launcher:

```python
def _review_failed_images(self) -> None:
    failed_basenames = sorted(self._last_run_failed_set)
    if not failed_basenames:
        return
    failed_paths: list[Path] = []
    for bn in failed_basenames:
        row = self._basename_to_row.get(bn)
        if row is not None and row < len(self._image_paths):
            failed_paths.append(self._image_paths[row])
    if not failed_paths:
        QMessageBox.warning(
            self, "Review failed images",
            "Couldn't locate the failed images. Has the input folder changed "
            "since the last run?"
        )
        return
    self._open_landmark_inspector(failed_paths[0], cohort=failed_paths)
```

### 4. `TRACE/landmark_inspector_dialog.py` — new file

```python
"""Per-image landmark inspector + editor.

Three entry points (Main tab):
  - Right-click a single image → "Inspect & Edit landmarks…"
  - Right-click on a multi-selection → "Inspect & Edit landmarks (N selected)…"
  - Post-run "Review failed images" button alongside the TODO #11 rerun buttons

All three call self._window._open_landmark_inspector(image_path, cohort=...).
Cohort mode (≥2 images) shows prev/next navigation in the dialog footer
and reuses a single napari Viewer across images for performance.

User drags predicted landmarks to corrected positions, optionally adds a
missing landmark from the canonical name list, hits Save → writes
<image_dir>/<stem>_landmarks_override.geojson. The next batch run picks up
the override automatically (Stage 3 short-circuit in preprocessing/pipeline.py).

v1 scope: landmarks only. Vein/intervein editing parked for v2; dialog
structure leaves room for a future QTabWidget with a "Segmentation" tab.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtWidgets import (
    QComboBox, QDialog, QDialogButtonBox, QHBoxLayout, QLabel,
    QMessageBox, QPushButton, QVBoxLayout, QWidget,
)


class LandmarkInspectorDialog(QDialog):
    def __init__(self, window, image_path: Path, cohort: Optional[list[Path]] = None):
        """
        Args:
            window: The TraceWindow (parent).
            image_path: First image to display.
            cohort: Optional list of images to step through. When given and
                len > 1, the dialog footer shows prev/next navigation +
                a "Failed image N of M" counter. When None, single-image mode.
        """
        super().__init__(window)
        self._window = window
        self._image_path = Path(image_path)
        self._cohort = list(cohort) if cohort and len(cohort) > 1 else None
        self._cohort_index = (
            self._cohort.index(self._image_path)
            if self._cohort and self._image_path in self._cohort
            else 0
        )
        self.setWindowTitle(f"Inspect & Edit landmarks — {self._image_path.name}")
        self.resize(1200, 800)

        layout = QVBoxLayout(self)
        self._editor = LandmarkEditorWidget(self, self._image_path)
        layout.addWidget(self._editor, stretch=1)

        # Navigation row (cohort mode only)
        self._build_navigation()

        # Save / Restore / Cancel buttons
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

    def _navigate(self, delta: int) -> None:
        if self._cohort is None:
            return
        new_idx = self._cohort_index + delta
        if not (0 <= new_idx < len(self._cohort)):
            return
        if self._editor.has_unsaved_changes():
            reply = QMessageBox.question(
                self, "Unsaved changes",
                "You have unsaved edits on this image. Save before moving on?",
                QMessageBox.Save | QMessageBox.Discard | QMessageBox.Cancel,
            )
            if reply == QMessageBox.Cancel:
                return
            if reply == QMessageBox.Save:
                try:
                    self._editor.save_override()
                except Exception as exc:  # noqa: BLE001
                    QMessageBox.critical(self, "Save failed", f"{exc}")
                    return
        self._cohort_index = new_idx
        self._image_path = self._cohort[new_idx]
        self.setWindowTitle(f"Inspect & Edit landmarks — {self._image_path.name}")
        self._editor.swap_image(self._image_path)
        self._refresh_nav_state()

    def _on_save_and_next(self) -> None:
        try:
            self._editor.save_override()
        except Exception as exc:  # noqa: BLE001
            QMessageBox.critical(self, "Save failed", f"{exc}")
            return
        self._navigate(+1)

    def _on_save(self) -> None:
        try:
            override_path = self._editor.save_override()
        except Exception as exc:  # noqa: BLE001
            QMessageBox.critical(self, "Save failed", f"Could not write override: {exc}")
            return
        QMessageBox.information(
            self, "Saved",
            f"Landmark override saved to:\n\n{override_path}\n\n"
            f"The next run that includes this image will use the override "
            f"instead of running LandmarkLocator on it."
        )
        # In cohort mode, plain "Save" should NOT close the dialog — the user
        # may want to keep reviewing. Only single-image mode auto-closes.
        if self._cohort is None:
            self.accept()
```

```python
class LandmarkEditorWidget(QWidget):
    """Napari embed + landmark editing controls.

    Sibling of measurementMaker's LandmarkPickerWidget — shares the napari
    setup pattern (image + Points layer + text labels) but does NOT snap-back
    data mutations. Drag-to-correct is the whole point of this widget.

    Public API:
      - load_or_generate()       — initial load (call after dialog paints)
      - swap_image(new_path)     — reuse the napari viewer for a new image
      - save_override() -> Path  — write sidecar GeoJSON, return its path
      - restore_predictions()    — reset points to the original predictions
      - has_unsaved_changes()    — for dirty-state navigation prompts
    """

    def __init__(self, parent_dialog, image_path: Path):
        super().__init__(parent_dialog)
        self._dialog = parent_dialog
        self._image_path = image_path
        self._window = parent_dialog._window
        self._viewer = None  # napari.Viewer, lazy
        self._image_layer = None
        self._points_layer = None
        self._names: list[str] = []
        self._predicted_positions: dict[str, tuple[float, float]] = {}
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
        self.btn_add.clicked.connect(self._on_add_landmark)
        controls.addWidget(self.btn_add)
        controls.addStretch(1)
        layout.addLayout(controls)

        self._viewer_placeholder = QLabel(
            "Loading image and predicted landmarks…\n\n"
            "First open may take a few seconds while LandmarkLocator initializes."
        )
        self._viewer_placeholder.setAlignment(Qt.AlignCenter)
        self._viewer_placeholder.setStyleSheet("color: #888; padding: 40px;")
        layout.addWidget(self._viewer_placeholder, stretch=1)

    def load_or_generate(self) -> None:
        """Locate predictions for self._image_path and open the viewer.

        Search order — reopening a previously-edited image starts from the
        override so iterative refinement works:
          1. <image_dir>/<stem>_landmarks_override.geojson      ← user's prior corrections
          2. <output_folder>/<stem>_landmarks.geojson           ← post-run output
          3. <image_dir>/<stem>_landmarks.geojson               ← Stage 3 sidecar precedent
          4. Generate via _generate_landmarks_for_image()       ← on-demand LandmarkLocator
        """
        stem = self._image_path.stem
        image_dir = self._image_path.parent
        candidates = [
            image_dir / f"{stem}_landmarks_override.geojson",
            None,  # filled below if output folder is set
            image_dir / f"{stem}_landmarks.geojson",
        ]
        try:
            out_text = self._window.output_edit.text().strip()
            if out_text:
                candidates[1] = Path(out_text) / f"{stem}_landmarks.geojson"
        except Exception:
            pass

        landmarks_dict: Optional[dict] = None
        for cand in candidates:
            if cand is None or not cand.is_file():
                continue
            try:
                landmarks_dict = _parse_landmarks_geojson(cand)
                break
            except Exception:
                continue

        if landmarks_dict is None:
            # No file found — regenerate via LandmarkLocator.
            panel = self._window.inline_custom_distances_panel
            try:
                generated = panel._generate_landmarks_for_image(self._image_path)
                landmarks_dict = _parse_landmarks_geojson(generated)
            except Exception as exc:
                QMessageBox.critical(
                    self, "Could not load landmarks",
                    f"No landmarks file found and on-demand generation failed:\n\n{exc}\n\n"
                    f"Configure a landmark model in Settings → Models and try again."
                )
                self._dialog.reject()
                return

        self._predicted_positions = dict(landmarks_dict)
        self._open_napari(landmarks_dict)
        self._last_saved_snapshot = self._snapshot_current()

    def _open_napari(self, landmarks_dict: dict[str, tuple[float, float]]) -> None:
        import napari
        import numpy as np
        from landmark_locator.data.psd_loader import imread_any

        image = imread_any(self._image_path)
        self._viewer = napari.Viewer(show=False)
        self._image_layer = self._viewer.add_image(image, name=self._image_path.name)

        self._names = list(landmarks_dict.keys())
        coords_yx = np.array([[y, x] for (x, y) in landmarks_dict.values()])
        text = {
            "string": "{name}",
            "size": 12,
            "color": "yellow",
            "translation": [0, 15],
            "anchor": "lower_left",
        }
        properties = {"name": np.array(self._names)}
        self._points_layer = self._viewer.add_points(
            coords_yx, name="landmarks", size=12,
            face_color="cyan", edge_color="black",
            text=text, properties=properties,
        )
        # CRITICAL DIFFERENCE from LandmarkPickerWidget: leave drag-mode enabled
        # and do NOT install a snap-back callback. The user is here to edit.
        self._points_layer.mode = "select"
        self._points_layer.editable = True

        self.layout().removeWidget(self._viewer_placeholder)
        self._viewer_placeholder.deleteLater()
        self.layout().addWidget(self._viewer.window._qt_window, stretch=1)

    def _on_add_landmark(self) -> None:
        if self._points_layer is None:
            return
        raw_name = self.cmb_add_name.currentData()
        if raw_name in self._names:
            QMessageBox.information(
                self, "Already present",
                f"'{raw_name}' is already in the landmark list. Drag it instead."
            )
            return
        import numpy as np
        h, w = self._image_layer.data.shape[:2]
        cy, cx = h / 2, w / 2
        new_coords = np.vstack([self._points_layer.data, [cy, cx]])
        self._names.append(raw_name)
        self._points_layer.data = new_coords
        self._points_layer.properties = {"name": np.array(self._names)}

    def restore_predictions(self) -> None:
        if self._points_layer is None:
            return
        import numpy as np
        self._names = list(self._predicted_positions.keys())
        coords_yx = np.array([[y, x] for (x, y) in self._predicted_positions.values()])
        self._points_layer.data = coords_yx
        self._points_layer.properties = {"name": np.array(self._names)}

    def save_override(self) -> Path:
        if self._points_layer is None:
            raise RuntimeError("Viewer not initialized; nothing to save.")
        coords_yx = self._points_layer.data
        names = list(self._points_layer.properties.get("name", []))
        if len(names) != len(coords_yx):
            raise RuntimeError(
                f"Internal: name/coord length mismatch ({len(names)} vs {len(coords_yx)})"
            )
        features = []
        for name, (y, x) in zip(names, coords_yx):
            features.append({
                "type": "Feature",
                "geometry": {"type": "Point", "coordinates": [float(x), float(y)]},
                "properties": {
                    "classification": {"name": str(name)},
                    "reliable": True,
                    "gate_reason": "manual override",
                    "confidence": 1.0,
                },
            })
        payload = {"type": "FeatureCollection", "features": features}
        override_path = self._image_path.parent / f"{self._image_path.stem}_landmarks_override.geojson"
        override_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        self._last_saved_snapshot = self._snapshot_current()
        return override_path

    def has_unsaved_changes(self) -> bool:
        if self._points_layer is None or self._last_saved_snapshot is None:
            return False
        return self._snapshot_current() != self._last_saved_snapshot

    def _snapshot_current(self) -> tuple:
        """Tuple of (names_tuple, coords_tuple) for dirty-state comparison."""
        if self._points_layer is None:
            return ((), ())
        names = tuple(self._points_layer.properties.get("name", []))
        coords = tuple(tuple(map(float, row)) for row in self._points_layer.data)
        return (names, coords)

    def swap_image(self, new_image_path: Path) -> None:
        """Replace the current image + landmarks. Reuses self._viewer."""
        self._image_path = new_image_path
        if self._image_layer is not None and self._viewer is not None:
            self._viewer.layers.remove(self._image_layer)
        if self._points_layer is not None and self._viewer is not None:
            self._viewer.layers.remove(self._points_layer)
        self._image_layer = None
        self._points_layer = None
        self._predicted_positions = {}
        self._names = []
        self._last_saved_snapshot = None
        self.load_or_generate()


def _parse_landmarks_geojson(path: Path) -> dict[str, tuple[float, float]]:
    """Load a landmarks GeoJSON into {name: (x, y)}.

    Tolerant of either Stage 3's full schema (confidence/sharpness/etc.) or
    the minimal override schema (just classification.name + coordinates).
    """
    data = json.loads(path.read_text(encoding="utf-8"))
    out: dict[str, tuple[float, float]] = {}
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
```

---

## Key reference points

### Existing helpers you reuse

| Helper | Location | Purpose |
|---|---|---|
| `_generate_landmarks_for_image(image_path) -> Path` | `TRACE/inline_panels.py:1457–1498` | Runs LandmarkLocator on a single image; writes `<stem>_landmarks.geojson` to image dir or temp |
| `_on_image_list_context_menu(pos)` | `TRACE/gui.py:1272` | Right-click handler on `image_list`; extend with new action |
| `LANDMARK_DISPLAY_NAMES` | `measurementMaker/measurement_maker/landmark_names.py:24–38` | Canonical 13-landmark name dict — populates the Add-landmark dropdown |
| `landmarks_to_geojson()` | `preprocessing/pipeline.py:271–291` | Reference schema for the override sidecar |
| `imread_any(path)` | `landmark_locator/data/psd_loader.py` | Loads any image format the project supports |

### Existing things you mustn't break

- `LandmarkPickerWidget` in `measurementMaker/measurement_maker/embedded_picker.py:45` — the Custom Measurements tab depends on its read-only snap-back behavior. Build your widget as a sibling, not a refactor.
- TODO #11's rerun-failed buttons — share the row layout and the `_refresh_rerun_buttons` logic. Your new button appears alongside, not instead.
- The right-click context menu's existing skip/unskip actions — your new "Inspect & Edit landmarks…" action appears after a `menu.addSeparator()`, doesn't replace anything.

### Override sidecar invariants

- Path: `<image_dir>/<stem>_landmarks_override.geojson` (next to the source image, not in the output folder)
- Schema: FeatureCollection of Point features, each with `geometry.coordinates: [x_pixel, y_pixel]` and `properties.classification.name`
- All features marked `reliable: True`, `confidence: 1.0`, `gate_reason: "manual override"` so they sail through downstream confidence gates
- napari uses (row, col) = (y, x) internally; flip to GeoJSON's [x, y] on save

---

## Design decisions

### A. Sibling widget (not refactor) for the editor

`LandmarkPickerWidget`'s snap-back guard at `embedded_picker.py:497–514` is wired into its core data-flow. Refactoring it to support an "edit mode" would risk breakage in Custom Measurements' pair-picking. The sibling shares the napari setup pattern (image + Points layer + text labels) but reimplements ~80 lines without the guard.

### B. Sidecar file location: next to the source image

Three options were considered:
- **Image-adjacent sidecar (chosen).** Matches the hinge-chop fallback precedent at `preprocessing/pipeline.py:806–809`. Stays with the image when data moves. Doesn't pollute the output folder.
- Inside the output folder. Tied to one specific run — moving or sharing the image loses the corrections.
- A central registry in QSettings or a project file. Hard to migrate; data loss on wipe-memories.

Image-adjacent wins on portability + discoverability.

### C. Load-order priority in the dialog

`override → post-run output → image-dir sidecar → on-demand regen`. Reopening a previously-edited image starts from the override, not from a fresh ML prediction — so iterative refinement works.

### D. "Add at image center" instead of "click-to-place"

napari's add-point mode requires a UX dance (toggle layer mode → click → toggle back). Adding at the center and letting the user drag is simpler and uses the same primary interaction (drag) for both add and correct. Reconsider in v2 if users complain.

### E. Override file metadata

Every feature marked `reliable: True`, `confidence: 1.0`, `gate_reason: "manual override"`. This ensures the override sails through any confidence-gate logic downstream — a manually-placed landmark shouldn't be rejected as low-confidence. The `gate_reason` string is a diagnostic breadcrumb for anyone inspecting the file later.

### F. Stage 3 short-circuit doesn't re-validate the override

The spec just trusts the override: load, copy to canonical location, skip prediction. If the override is malformed (missing names, wildly out-of-image coords), Stage 4 will fail loudly downstream. A small validation pass (every name in the canonical 13, coords within image bounds) before trusting it would be cheap to add — defer to v2 if scope tightens.

### G. Cohort-aware constructor (vs. wrapping the dialog)

Considered building a wrapper that owns multiple `LandmarkInspectorDialog` instances and shows them sequentially. Rejected — the napari viewer is heavy to construct and tearing one down per image is wasteful. Keeping a single viewer and swapping the image/points layers via `swap_image()` is the right move (~5 napari calls per swap).

### H. "Save & Next" convenience

Plain Save in single-image mode closes the dialog. In cohort mode it should NOT close (the user is mid-review). "Save & Next" is a separate explicit action that saves AND advances. Disabled on the last image.

### I. Dirty-state confirmation on navigation

If the user has dragged landmarks but not saved, navigating away (Prev / Next / Close in cohort mode) should prompt with Save / Discard / Cancel. Prevents accidental loss of corrections.

### J. Walkthrough hint ordering

The first-time hint for the Review-failed button fires AFTER any rerun-button hints in the chain. User sees one hint per session in natural feature progression.

---

## Edge cases

- **No landmark model configured**: `_generate_landmarks_for_image` raises `RuntimeError`. Dialog catches it with a clear "Configure a landmark model in Settings → Models" message and closes.
- **Image format napari can't read natively**: use `imread_any` from `landmark_locator.data.psd_loader` — already handles every format the rest of the pipeline does.
- **User saves with zero landmarks**: writes an empty FeatureCollection. Stage 4/5 will fail downstream. Worth a confirm dialog ("Save with no landmarks? Stage 4/5 will likely fail."). Implementer's call.
- **Override exists but image is now at a different path**: sidecar is keyed on the source image directory, so this works correctly.
- **User wants to delete an override**: not in v1. They'd delete the file manually. A "Remove override" button (only enabled when an override is present) is a small follow-up.
- **Stage 3 cache hit + override exists**: override check goes BEFORE the prediction call, so a hit override fully short-circuits regardless of cache state.
- **Cohort: image in the cohort has been deleted from disk**: `swap_image` calls `load_or_generate`; the load fails. Show an error and let the user skip to the next image via the navigation buttons.
- **Cohort: user closes mid-review**: failed set on `TraceWindow` is unchanged. Buttons stay visible until the next run completes. No "this image is now corrected" feedback — out of scope for v1.
- **Cohort: input folder changes mid-review**: the cohort is captured at click time; dialog stays consistent with the snapshot. Button itself hides on input-folder change via the existing TODO #11 logic.
- **Single-image cohort (cohort of 1)**: constructor normalizes this to None (`cohort=None if len(cohort) <= 1 else cohort`) so navigation row doesn't appear.
- **Multi-select right-click outside the selection**: handler's `target_item in selected` guard demotes to single-image on the clicked row. Matches typical OS behavior.

---

## Test checklist

### Single-image mode

1. **Smoke**: right-click image → "Inspect & Edit landmarks…" → modal opens, napari embedded, image visible with cyan dots + yellow name labels.
2. **Drag**: drag one landmark to a new position → Save → confirm `<image_dir>/<stem>_landmarks_override.geojson` exists with the dragged coords.
3. **Restore predictions**: drag a landmark, click Restore → original position returns.
4. **Add missing**: dropdown shows all 13 canonical names → pick one not currently present → Add → landmark appears at center with the right text label.
5. **Duplicate guard**: pick a name already in the list → Add → informational dialog, no duplicate.
6. **Override fed into batch run**: with override in place, run a batch → run.log shows "using manual landmark override" → output landmarks_geojson matches override coords.
7. **Pre-run flow**: never run pipeline → right-click → inspector regenerates predictions on demand → edit → save → first batch uses override.
8. **Post-run flow**: run a batch with `landmarks_geojson` output ticked → right-click on an image → inspector loads from `<output_folder>/<stem>_landmarks.geojson`, not regenerated.
9. **Reopen previously-edited**: open inspector on image with existing override → viewer starts from the override (not fresh ML).
10. **No landmark model configured**: clear path → right-click → friendly error, dialog closes.
11. **Wipe memories**: after wipe → override files on disk untouched (they're in the image dir, not QSettings) → next run still picks them up.
12. **Cancel**: drag → Cancel → no override file written.

### Multi-select cohort (right-click on multi-selection)

13. **Cohort label**: select 3 rows in `image_list` → right-click on one of them → menu shows "Inspect & Edit landmarks (3 selected)…".
14. **Cohort opens**: click it → inspector opens on the first selected image, footer shows "Image 1 of 3 — <name>".
15. **Right-click outside selection**: select 3 rows → right-click on a fourth (unselected) row → menu shows "Inspect & Edit landmarks…" (no count) → click → single-image mode on the clicked row.

### Cohort navigation (both multi-select and Review-failed entries)

16. **Next**: footer "1 of 3" → Next → "2 of 3" loaded.
17. **Previous**: from "2 of 3" → Prev → "1 of 3" loaded.
18. **Save & Next**: drag a landmark, click Save & Next → override written for current image, dialog advances.
19. **Dirty-state prompt**: drag on image 2, don't save, click Previous → "Unsaved changes" dialog → Cancel → stay on 2; Save → override written, advance; Discard → no save, advance.
20. **Last image**: navigate to last → Save & Next disabled, plain Save still works (in cohort mode plain Save does NOT close the dialog).

### Post-run Review-failed button

21. **Button visibility**: run with one failed image → "Review failed images (1)" appears alongside rerun-failed buttons → click → inspector opens on that image, cohort row "Image 1 of 1".
22. **Walkthrough hint**: first batch with failed images → the Review-failed hint fires after the rerun hints have had their turn (sequencing depends on the existing chain logic).
23. **Cohort with corrections → rerun**: review all failed, save corrections for each → click "Rerun failed images" → run.log shows override path firing for each → previously-failed images succeed.
24. **Failed image deleted from disk**: delete an image file between run and review click → friendly error or graceful skip via navigation.

---

## v2 parking list

- **Vein/intervein editing** — polygon edit + class re-assignment via napari Shapes layer. Add a `QTabWidget` inside `LandmarkInspectorDialog` with "Landmarks" and "Segmentation" tabs.
- **Click-to-place add mode** using napari's native add-point tool.
- **Remove-override button** + visual indicator on the image list when an override exists (e.g., a new glyph in the existing per-image status indicator).
- **Override-validation pass** before Stage 3 trusts the sidecar (name set, coord bounds).
- **Batch inspector** — pick a folder of images and step through with arrow keys, without going through the Main tab.
- **Per-pixel segmentation paint** as an alternative to polygon editing.

---

## Project conventions for your reference

- All paths in code should use `pathlib.Path`, not strings.
- All imports needed across sibling modules: `run_gui.py` and `run_cli.py` set up `sys.path` by inserting `mapThemVeins/`, `mapThemVeins/HingeChopper/`, `mapThemVeins/modelTOjson/`, etc. before importing TRACE. If you write a script that uses the modules outside `run_*.py`, mirror that path setup.
- The `napari.Viewer` constructor is heavy — always lazy-create, and reuse instances when stepping through multiple images.
- For dialog modal feedback during long-running operations, use `QTimer.singleShot(0, fn)` to let Qt paint before the work starts.
- `pre-commit run --files <changed files>` before committing — the hooks run isort → black → flake8.
