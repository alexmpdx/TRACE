"""Train a landmark model on the user's own manual landmark edits.

Opened from the Main-tab image-list right-click menu ("Train landmark model from
manual edits…") or Settings → Models → "Your models". The dialog:

  1. collects every image with a saved landmark override under the chosen
     folders (``TRACE.landmark_training.find_override_samples``),
  2. fine-tunes the landmark model TRACE is currently using on them — or trains a
     blank one — in a background thread
     (``landmark_locator.training.user_training.train_from_samples``),
  3. offers to make the result TRACE's landmark model
     (``TraceWindow.use_landmark_model``).

Modal on purpose: training and a pipeline run would fight over the same GPU/CPU.
"""

from __future__ import annotations

import os
import shutil
import sys
import time
from pathlib import Path
from typing import Optional

from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDialog,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QMessageBox,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
)
from TRACE.landmark_training import (
    default_model_name,
    find_override_samples,
    is_landmark_model_dir,
    sanitize_model_name,
    user_models_dir,
)

SAVE_DIR_SETTING = "landmark_training/save_dir"

_MODE_FINETUNE = "finetune"
_MODE_SCRATCH = "scratch"

# Below these counts the dialog still lets the user train, but says what to expect.
_FEW_FINETUNE = 10
_FEW_SCRATCH = 100


class _LogCapture:
    """File-like stand-in for ``sys.stdout`` that forwards complete lines to a callback."""

    def __init__(self, emit) -> None:
        self._emit = emit
        self._partial = ""

    def write(self, text: str) -> int:
        if not text:
            return 0
        self._partial += text
        *lines, self._partial = self._partial.split("\n")
        if lines:
            self._emit("\n".join(lines))
        return len(text)

    def flush(self) -> None:
        if self._partial:
            self._emit(self._partial)
            self._partial = ""


class LandmarkTrainingThread(QThread):
    """Runs ``train_from_samples`` off the GUI thread."""

    log = pyqtSignal(str)
    epoch = pyqtSignal(object)  # dict from train_from_samples' epoch_callback
    succeeded = pyqtSignal(object)  # training_info dict
    cancelled = pyqtSignal()
    failed = pyqtSignal(str)

    def __init__(self, samples, output_dir: Path, base_model: Optional[Path], n_folds: int, epochs: int, parent=None):
        super().__init__(parent)
        self._samples = samples
        self._output_dir = Path(output_dir)
        self._base_model = base_model
        self._n_folds = n_folds
        self._epochs = epochs
        self._stop = False

    def request_stop(self) -> None:
        self._stop = True

    def run(self) -> None:
        # albumentations phones home for a version check on first import; not from a user's app.
        os.environ.setdefault("NO_ALBUMENTATIONS_UPDATE", "1")
        capture = _LogCapture(self.log.emit)
        old_stdout = sys.stdout
        sys.stdout = capture
        try:
            from landmark_locator.training.train import TrainingCancelled
            from landmark_locator.training.user_training import train_from_samples

            try:
                info = train_from_samples(
                    self._samples,
                    self._output_dir,
                    base_model=self._base_model,
                    n_folds=self._n_folds,
                    epochs=self._epochs,
                    epoch_callback=self.epoch.emit,
                    should_stop=lambda: self._stop,
                )
            except TrainingCancelled:
                capture.flush()
                self.cancelled.emit()
                return
            capture.flush()
            self.succeeded.emit(info)
        except Exception as exc:  # surfaced in the dialog; the log has the detail
            import traceback

            capture.flush()
            self.log.emit(traceback.format_exc())
            self.failed.emit(f"{type(exc).__name__}: {exc}")
        finally:
            sys.stdout = old_stdout


class LandmarkTrainingDialog(QDialog):
    """Pick training data + options, train, and switch TRACE to the new model."""

    def __init__(self, window) -> None:
        super().__init__(window)
        self._window = window
        self._thread: Optional[LandmarkTrainingThread] = None
        self._samples: list = []  # usable (override, image) pairs for the current mode
        self._base_order_cache: dict[str, Optional[list]] = {}
        self._output_dir: Optional[Path] = None
        self._created_output_dir = False
        self._trained_model: Optional[Path] = None
        self._close_when_stopped = False
        self._started_at = 0.0

        self.setWindowTitle("Train a landmark model")
        # No explicit minimum size: it would override the layout's own minimum and let
        # the option rows be squeezed on top of each other.
        self.resize(720, 800)
        self._build_ui()

        input_folder = ""
        if hasattr(window, "input_edit"):
            input_folder = window.input_edit.text().strip()
        if input_folder and Path(input_folder).is_dir():
            self.folder_list.addItem(input_folder)
        recursive_chk = getattr(window, "recursive_chk", None)
        if recursive_chk is not None:
            self.chk_recursive.setChecked(bool(recursive_chk.isChecked()))
        self.chk_recursive.toggled.connect(self._rescan)
        self._on_mode_changed()

    # -- construction -----------------------------------------------------
    def _base_model_path(self) -> Optional[Path]:
        """The landmark model TRACE is using now, when it is a loadable fold folder."""
        raw = (getattr(self._window, "_landmark_model_path", "") or "").strip()
        return Path(raw) if raw and is_landmark_model_dir(Path(raw)) else None

    def _wrapped_label(self, lines: int) -> QLabel:
        """Word-wrapped, top-aligned label with room reserved for `lines` lines.

        Reserving the height up front (instead of relying on Qt's height-for-width,
        which under-measures rich text inside nested layouts) keeps the last line of
        the longer messages from being clipped and stops the dialog jumping around.
        """
        label = QLabel()
        label.setWordWrap(True)
        label.setAlignment(Qt.AlignLeft | Qt.AlignTop)
        label.setMinimumHeight(label.fontMetrics().lineSpacing() * lines + 4)
        return label

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)

        intro = QLabel(
            "Every image whose landmarks you saved in <b>Edit model predictions…</b> is a training example. "
            "Landmarks you left out of an edit are skipped for that image."
        )
        intro.setWordWrap(True)
        layout.addWidget(intro)

        # Training data
        data_box = QGroupBox("Training data — folders containing edited images")
        data_layout = QVBoxLayout(data_box)
        row = QHBoxLayout()
        self.folder_list = QListWidget()
        self.folder_list.setMaximumHeight(64)
        row.addWidget(self.folder_list, stretch=1)
        btn_col = QVBoxLayout()
        self.btn_add_folder = QPushButton("Add folder…")
        self.btn_add_folder.clicked.connect(self._on_add_folder)
        self.btn_remove_folder = QPushButton("Remove")
        self.btn_remove_folder.clicked.connect(self._on_remove_folder)
        btn_col.addWidget(self.btn_add_folder)
        btn_col.addWidget(self.btn_remove_folder)
        btn_col.addStretch(1)
        row.addLayout(btn_col)
        data_layout.addLayout(row)
        self.chk_recursive = QCheckBox("Include subfolders")
        data_layout.addWidget(self.chk_recursive)
        self.lbl_samples = self._wrapped_label(3)
        data_layout.addWidget(self.lbl_samples)
        layout.addWidget(data_box)

        # Model options
        model_box = QGroupBox("New model")
        grid = QGridLayout(model_box)
        grid.setColumnStretch(1, 1)

        def _row(label: str, field) -> None:
            r = grid.rowCount()
            if label:
                grid.addWidget(QLabel(label), r, 0)
            if isinstance(field, QHBoxLayout):
                grid.addLayout(field, r, 1)
            else:
                grid.addWidget(field, r, 1)

        self.cmb_mode = QComboBox()
        self.cmb_mode.addItem("Improve TRACE's current model (recommended)", _MODE_FINETUNE)
        self.cmb_mode.addItem("Train a blank model from scratch", _MODE_SCRATCH)
        if self._base_model_path() is None:
            # No loadable current model → nothing to fine-tune from.
            self.cmb_mode.setCurrentIndex(1)
            self.cmb_mode.model().item(0).setEnabled(False)
        self.cmb_mode.currentIndexChanged.connect(self._on_mode_changed)
        # Explicit line breaks rather than word wrap: a wrapped label inside the grid is
        # under-measured by Qt and gets overdrawn by the row below it.
        self.lbl_mode_hint = QLabel()
        self.lbl_mode_hint.setAlignment(Qt.AlignLeft | Qt.AlignTop)
        self.lbl_mode_hint.setMinimumHeight(self.lbl_mode_hint.fontMetrics().lineSpacing() * 2 + 12)
        _row("Start from:", self.cmb_mode)
        _row("", self.lbl_mode_hint)

        self.edit_name = QLineEdit(default_model_name())
        _row("Name:", self.edit_name)

        save_row = QHBoxLayout()
        saved_dir = ""
        settings = getattr(self._window, "settings", None)
        if settings is not None:
            saved_dir = str(settings.value(SAVE_DIR_SETTING, "") or "")
        self.edit_save_dir = QLineEdit(saved_dir or str(user_models_dir()))
        self.edit_save_dir.setReadOnly(True)
        self.btn_save_dir = QPushButton("Browse…")
        self.btn_save_dir.clicked.connect(self._on_browse_save_dir)
        save_row.addWidget(self.edit_save_dir, stretch=1)
        save_row.addWidget(self.btn_save_dir)
        _row("Save in:", save_row)

        self.spin_epochs = QSpinBox()
        self.spin_epochs.setRange(1, 2000)
        self.spin_epochs.setToolTip(
            "Maximum passes over the training images per model. Training stops early once the "
            "held-out error stops improving."
        )
        _row("Epochs (max):", self.spin_epochs)
        self.spin_folds = QSpinBox()
        self.spin_folds.setRange(1, 5)
        self.spin_folds.setValue(5)
        self.spin_folds.setToolTip(
            "TRACE averages the predictions of several models trained on different subsets of your "
            "images. More models are more robust but take proportionally longer to train. Each one "
            "is scored on the images it did not train on."
        )
        _row("Models in ensemble:", self.spin_folds)
        layout.addWidget(model_box)

        # Run controls
        run_row = QHBoxLayout()
        self.btn_start = QPushButton("Start training")
        self.btn_start.setDefault(True)
        self.btn_start.clicked.connect(self._on_start)
        self.btn_stop = QPushButton("Stop")
        self.btn_stop.setEnabled(False)
        self.btn_stop.clicked.connect(self._on_stop)
        run_row.addWidget(self.btn_start)
        run_row.addWidget(self.btn_stop)
        run_row.addStretch(1)
        layout.addLayout(run_row)

        self.progress = QProgressBar()
        self.progress.setRange(0, 1)
        self.progress.setValue(0)
        layout.addWidget(self.progress)
        self.lbl_status = self._wrapped_label(2)
        layout.addWidget(self.lbl_status)

        self.txt_log = QPlainTextEdit()
        self.txt_log.setReadOnly(True)
        self.txt_log.setMaximumBlockCount(5000)
        self.txt_log.setMinimumHeight(60)
        layout.addWidget(self.txt_log, stretch=1)

        # Result
        result_row = QHBoxLayout()
        self.btn_use = QPushButton("Use this model in TRACE")
        self.btn_use.setEnabled(False)
        self.btn_use.setToolTip("Make the model you just trained the landmark model for every following run.")
        self.btn_use.clicked.connect(self._on_use_model)
        self.btn_close = QPushButton("Close")
        self.btn_close.clicked.connect(self.reject)
        result_row.addWidget(self.btn_use)
        result_row.addStretch(1)
        result_row.addWidget(self.btn_close)
        layout.addLayout(result_row)

    # -- training-data scan -------------------------------------------------
    def _mode(self) -> str:
        return self.cmb_mode.currentData()

    def _folders(self) -> list[Path]:
        return [Path(self.folder_list.item(i).text()) for i in range(self.folder_list.count())]

    def _base_landmark_order(self) -> Optional[list]:
        """Landmarks of the current model (cached; reading them loads a checkpoint)."""
        base = self._base_model_path()
        if base is None:
            return None
        key = str(base)
        if key not in self._base_order_cache:
            QApplication.setOverrideCursor(Qt.WaitCursor)
            try:
                from landmark_locator.training.user_training import base_model_landmarks

                self._base_order_cache[key] = base_model_landmarks(base)
            except Exception as exc:
                self._base_order_cache[key] = None
                self.txt_log.appendPlainText(f"Could not read the current landmark model: {exc}")
            finally:
                QApplication.restoreOverrideCursor()
        return self._base_order_cache[key]

    def _rescan(self, *_args) -> None:
        """Re-collect edited images and refresh the count line + Start button."""
        from landmark_locator.training.user_training import MIN_SAMPLES, summarize_samples

        found = find_override_samples(self._folders(), recursive=self.chk_recursive.isChecked())
        order = self._base_landmark_order() if self._mode() == _MODE_FINETUNE else None
        summary = summarize_samples(found, order)
        self._samples = summary["usable"]
        n = len(self._samples)

        if not self._folders():
            text = "Add a folder of images you have edited."
        elif n == 0:
            text = (
                "No edited images found here. Right-click an image in the main list, choose "
                "<b>Edit model predictions…</b>, correct the landmarks and save — then come back."
            )
        else:
            text = f"<b>{n} edited image{'s' if n != 1 else ''}</b> found."
            counts = summary["counts"]
            partial = sorted(name for name, c in counts.items() if c < n)
            if order is not None:
                partial += sorted(name for name in order if name not in counts)
            if partial:
                shown = ", ".join(f"{summary['display'].get(name, name)} ({counts.get(name, 0)})" for name in partial)
                text += f" Marked on only some images: {shown}."
            if summary["ignored"]:
                text += " Not in the current model, so ignored: " + ", ".join(sorted(summary["ignored"])) + "."
            few = _FEW_FINETUNE if self._mode() == _MODE_FINETUNE else _FEW_SCRATCH
            if n < MIN_SAMPLES:
                text += f" At least {MIN_SAMPLES} are needed to train."
            elif n < few:
                if self._mode() == _MODE_FINETUNE:
                    text += " That is few: treat the reported error as a rough guide, and edit more if you can."
                else:
                    text += " A blank model usually needs a few hundred images to match the bundled one."
        self.lbl_samples.setText(text)
        self.lbl_samples.setToolTip(text)
        self.spin_folds.setMaximum(max(1, min(5, n)))
        self.btn_start.setEnabled(n >= MIN_SAMPLES and self._thread is None)

    def _on_mode_changed(self, *_args) -> None:
        from landmark_locator.training.user_training import FINETUNE_EPOCHS, SCRATCH_EPOCHS

        if self._mode() == _MODE_FINETUNE:
            base = self._base_model_path()
            self.lbl_mode_hint.setText(
                f"Adjusts <b>{base.name if base else '—'}</b> toward your corrections.<br>"
                "Works with a few dozen images; never saved worse than it started."
            )
            self.spin_epochs.setValue(FINETUNE_EPOCHS)
        else:
            self.lbl_mode_hint.setText(
                "Learns only from your images and the landmarks you marked.<br>"
                "Needs a few hundred images, and internet access the first time."
            )
            self.spin_epochs.setValue(SCRATCH_EPOCHS)
        self._rescan()

    def _set_status(self, text: str) -> None:
        self.lbl_status.setText(text)

    # -- folder pickers -----------------------------------------------------
    def _pick_folder(self, caption: str, initial: str, on_picked, key: str) -> None:
        from TRACE.gui import _open_native_picker_async

        _open_native_picker_async(self, caption, initial, on_picked, folder=True, last_dir_key=key)

    def _on_add_folder(self) -> None:
        folders = self._folders()
        initial = str(folders[-1]) if folders else ""
        self._pick_folder("Add a folder of edited images", initial, self._on_folder_picked, "landmark_training_data")

    def _on_folder_picked(self, folder: str) -> None:
        if folder and folder not in [str(f) for f in self._folders()]:
            self.folder_list.addItem(folder)
            self._rescan()

    def _on_remove_folder(self) -> None:
        for item in self.folder_list.selectedItems():
            self.folder_list.takeItem(self.folder_list.row(item))
        self._rescan()

    def _on_browse_save_dir(self) -> None:
        self._pick_folder(
            "Folder to save trained models in", self.edit_save_dir.text(), self._on_save_dir_picked, "landmark_models"
        )

    def _on_save_dir_picked(self, folder: str) -> None:
        if folder:
            self.edit_save_dir.setText(folder)

    # -- run ------------------------------------------------------------------
    def _set_inputs_enabled(self, enabled: bool) -> None:
        for w in (
            self.folder_list,
            self.btn_add_folder,
            self.btn_remove_folder,
            self.chk_recursive,
            self.cmb_mode,
            self.edit_name,
            self.btn_save_dir,
            self.spin_epochs,
            self.spin_folds,
        ):
            w.setEnabled(enabled)

    def _on_start(self) -> None:
        name = sanitize_model_name(self.edit_name.text())
        if not name:
            QMessageBox.warning(self, "Name needed", "Give the new model a name.")
            return
        save_dir = Path(self.edit_save_dir.text().strip() or user_models_dir())
        output_dir = save_dir / name
        if output_dir.exists() and any(output_dir.iterdir()):
            QMessageBox.warning(
                self,
                "Name already used",
                f"{output_dir} already exists and is not empty.\n\nChoose a different name.",
            )
            return
        base = self._base_model_path() if self._mode() == _MODE_FINETUNE else None
        if self._mode() == _MODE_FINETUNE and base is None:
            QMessageBox.warning(self, "No model to improve", "TRACE has no landmark model selected to start from.")
            return
        try:
            self._created_output_dir = not output_dir.exists()
            output_dir.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            QMessageBox.critical(self, "Cannot create folder", f"Could not create {output_dir}:\n{exc}")
            return
        settings = getattr(self._window, "settings", None)
        if settings is not None:
            settings.setValue(SAVE_DIR_SETTING, str(save_dir))

        self._output_dir = output_dir
        self._trained_model = None
        self.btn_use.setEnabled(False)
        self.btn_use.setText("Use this model in TRACE")
        self.txt_log.clear()
        self.progress.setRange(0, 0)  # busy until the first epoch reports
        self._set_status("Preparing…")
        self._set_inputs_enabled(False)
        self.btn_start.setEnabled(False)
        self.btn_stop.setEnabled(True)
        self._started_at = time.monotonic()

        self._thread = LandmarkTrainingThread(
            list(self._samples), output_dir, base, self.spin_folds.value(), self.spin_epochs.value(), self
        )
        self._thread.log.connect(self.txt_log.appendPlainText)
        self._thread.epoch.connect(self._on_epoch)
        self._thread.succeeded.connect(self._on_succeeded)
        self._thread.cancelled.connect(self._on_cancelled)
        self._thread.failed.connect(self._on_failed)
        self._thread.start()

    def _on_stop(self) -> None:
        if self._thread is not None:
            self._thread.request_stop()
            self.btn_stop.setEnabled(False)
            self._set_status("Stopping…")

    def _on_epoch(self, data: dict) -> None:
        total = data["n_folds"] * data["epochs"]
        done = data["fold"] * data["epochs"] + data["epoch"] + 1
        self.progress.setRange(0, total)
        self.progress.setValue(done)
        self._set_status(
            f"Model {data['fold'] + 1} of {data['n_folds']} · epoch {data['epoch'] + 1}/{data['epochs']} · "
            f"held-out error {data['mean_error']:.1f} px"
        )

    def _finish_run(self) -> None:
        """Common teardown once the thread has reported (success, stop or failure)."""
        if self._thread is not None:
            self._thread.wait()
            self._thread = None
        self.btn_stop.setEnabled(False)
        self._set_inputs_enabled(True)
        self.progress.setRange(0, 1)
        self._rescan()  # re-enables Start when there is still enough data

    def _discard_output(self) -> None:
        """Remove the half-written model folder of a stopped / failed run."""
        if self._output_dir is not None and self._created_output_dir:
            shutil.rmtree(self._output_dir, ignore_errors=True)
        self._output_dir = None

    def _on_succeeded(self, info: dict) -> None:
        output_dir = self._output_dir
        self._finish_run()
        self.progress.setValue(1)
        try:
            (output_dir / "training.log").write_text(self.txt_log.toPlainText(), encoding="utf-8")
        except OSError:
            pass
        minutes = (time.monotonic() - self._started_at) / 60.0
        after = info["mean_pixel_error"]
        before = info.get("baseline_pixel_error")
        if before is None:
            verdict = f"Held-out error: <b>{after:.1f} px</b>."
        elif after < before:
            verdict = f"Held-out error went from {before:.1f} px to <b>{after:.1f} px</b>."
        else:
            verdict = (
                f"Training did not beat the starting model on the held-out images ({before:.1f} px), so the "
                "saved model is effectively the one you started from. More edited images would help."
            )
        self._set_status(
            f"Done in {minutes:.1f} min — trained on {info['n_images']} images. {verdict}<br>Saved to {output_dir}"
        )
        self._trained_model = output_dir
        self.btn_use.setEnabled(True)
        self.btn_use.setDefault(True)

    def _on_cancelled(self) -> None:
        self._finish_run()
        self._discard_output()
        self._set_status("Training stopped. Nothing was saved.")
        if self._close_when_stopped:
            super().reject()

    def _on_failed(self, message: str) -> None:
        self._finish_run()
        self._discard_output()
        self._set_status(f"Training failed: {message}")
        if self._close_when_stopped:
            super().reject()
            return
        QMessageBox.critical(self, "Training failed", f"{message}\n\nThe log in this window has the details.")

    def _on_use_model(self) -> None:
        if self._trained_model is None:
            return
        self._window.use_landmark_model(self._trained_model)
        self.btn_use.setEnabled(False)
        self.btn_use.setText("In use ✓")
        self._set_status(
            f"TRACE now uses <b>{self._trained_model.name}</b> for landmarks. Switch back any time in "
            "Settings → Models → Your models."
        )

    # -- closing ----------------------------------------------------------------
    def reject(self) -> None:
        if self._thread is None:
            super().reject()
            return
        reply = QMessageBox.question(
            self,
            "Stop training?",
            "Training is still running. Stop it and discard the unfinished model?",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )
        if reply == QMessageBox.Yes and self._thread is not None:
            self._close_when_stopped = True
            self._on_stop()

    def closeEvent(self, event) -> None:  # noqa: N802 (Qt override)
        if self._thread is not None:
            event.ignore()
            self.reject()
        else:
            super().closeEvent(event)
