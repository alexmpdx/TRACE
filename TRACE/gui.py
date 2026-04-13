"""PyQt5 GUI for the TRACE combined pipeline.

Dark Fusion theme matching the preprocessing app. Runs the pipeline in a
background QThread with progress reporting.
"""

import sys
import traceback
from collections import OrderedDict
from pathlib import Path

from PyQt5.QtCore import QSettings, Qt, QThread, pyqtSignal
from PyQt5.QtGui import QColor, QPalette
from PyQt5.QtWidgets import (
    QApplication,
    QCheckBox,
    QDoubleSpinBox,
    QFileDialog,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QMainWindow,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QSplitter,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from preprocessing.pipeline import discover_images
from TRACE.pipeline import OUTPUT_TYPES, trace_folder

# ---------------------------------------------------------------------------
# Worker thread
# ---------------------------------------------------------------------------


class TraceWorker(QThread):
    """Runs trace_folder() in a background thread."""

    progress = pyqtSignal(int, int, str, str, str)  # idx, total, name, stage, detail
    all_done = pyqtSignal(list)  # results
    error = pyqtSignal(str)

    def __init__(self, kwargs):
        super().__init__()
        self.kwargs = kwargs
        self._cancel = False

    def cancel(self):
        self._cancel = True

    def run(self):
        try:

            def _progress(idx, total, name, stage, detail):
                if self._cancel:
                    raise InterruptedError("Cancelled by user")
                self.progress.emit(idx, total, name, stage, detail)

            self.kwargs["progress_callback"] = _progress
            results = trace_folder(**self.kwargs)
            self.all_done.emit(results)
        except InterruptedError:
            self.all_done.emit([])
        except Exception as e:
            self.error.emit(f"{e}\n{traceback.format_exc()}")


# ---------------------------------------------------------------------------
# Main window
# ---------------------------------------------------------------------------


class TraceWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("TRACE — Wing Analysis Pipeline")
        self.settings = QSettings("TRACE", "WingAnalysisPipeline")
        self.resize(1050, 750)
        self.worker = None
        self._image_paths = []
        self._build_ui()
        self._restore_settings()

    # -----------------------------------------------------------------------
    # UI construction
    # -----------------------------------------------------------------------
    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QHBoxLayout(central)

        # --- Left panel ---
        left = QWidget()
        left.setMaximumWidth(380)
        left_layout = QVBoxLayout(left)
        left_layout.setContentsMargins(4, 4, 4, 4)

        # -- Folders --
        folder_group = QGroupBox("Folders")
        fg = QVBoxLayout(folder_group)

        fg.addWidget(QLabel("Input folder:"))
        row = QHBoxLayout()
        self.input_edit = QLineEdit()
        self.input_edit.setReadOnly(True)
        self.input_edit.setPlaceholderText("Select input folder...")
        btn = QPushButton("Browse...")
        btn.clicked.connect(self._select_input)
        row.addWidget(self.input_edit, stretch=1)
        row.addWidget(btn)
        fg.addLayout(row)

        fg.addWidget(QLabel("Output folder:"))
        row = QHBoxLayout()
        self.output_edit = QLineEdit()
        self.output_edit.setReadOnly(True)
        self.output_edit.setPlaceholderText("Select output folder...")
        btn = QPushButton("Browse...")
        btn.clicked.connect(self._select_output)
        row.addWidget(self.output_edit, stretch=1)
        row.addWidget(btn)
        fg.addLayout(row)

        left_layout.addWidget(folder_group)

        # -- Models --
        model_group = QGroupBox("Models")
        mg = QVBoxLayout(model_group)

        mg.addWidget(QLabel("Landmark model (.pt):"))
        row = QHBoxLayout()
        self.lm_edit = QLineEdit()
        self.lm_edit.setReadOnly(True)
        self.lm_edit.setPlaceholderText("Select landmark checkpoint...")
        btn = QPushButton("Browse...")
        btn.clicked.connect(self._select_landmark_model)
        row.addWidget(self.lm_edit, stretch=1)
        row.addWidget(btn)
        mg.addLayout(row)

        mg.addWidget(QLabel("Segmentation model folder:"))
        row = QHBoxLayout()
        self.seg_edit = QLineEdit()
        self.seg_edit.setReadOnly(True)
        self.seg_edit.setPlaceholderText("Select segmentation model folder...")
        btn = QPushButton("Browse...")
        btn.clicked.connect(self._select_seg_model)
        row.addWidget(self.seg_edit, stretch=1)
        row.addWidget(btn)
        mg.addLayout(row)

        left_layout.addWidget(model_group)

        # -- Scale --
        scale_group = QGroupBox("Scale")
        sg = QHBoxLayout(scale_group)
        sg.addWidget(QLabel("\u00b5m/px:"))
        self.scale_spin = QDoubleSpinBox()
        self.scale_spin.setDecimals(4)
        self.scale_spin.setRange(0.0, 100.0)
        self.scale_spin.setValue(0.483)
        self.scale_spin.setSingleStep(0.001)
        self.scale_spin.setSpecialValueText("pixel only")
        sg.addWidget(self.scale_spin, stretch=1)
        left_layout.addWidget(scale_group)

        # -- Output selection --
        out_group = QGroupBox("Outputs")
        ol = QVBoxLayout(out_group)
        self.output_checks: OrderedDict[str, QCheckBox] = OrderedDict()
        for key, label in OUTPUT_TYPES.items():
            chk = QCheckBox(label)
            chk.setChecked(True)
            self.output_checks[key] = chk
            ol.addWidget(chk)
        left_layout.addWidget(out_group)

        # -- Run / Cancel --
        btn_layout = QHBoxLayout()
        self.btn_run = QPushButton("Run Pipeline")
        self.btn_run.clicked.connect(self._run_pipeline)
        self.btn_cancel = QPushButton("Cancel")
        self.btn_cancel.clicked.connect(self._cancel_pipeline)
        self.btn_cancel.setEnabled(False)
        btn_layout.addWidget(self.btn_run)
        btn_layout.addWidget(self.btn_cancel)
        left_layout.addLayout(btn_layout)

        left_layout.addStretch()

        # --- Right panel ---
        right = QWidget()
        right_layout = QVBoxLayout(right)
        right_layout.setContentsMargins(4, 4, 4, 4)

        right_layout.addWidget(QLabel("Images:"))
        self.image_list = QListWidget()
        right_layout.addWidget(self.image_list, stretch=1)

        right_layout.addWidget(QLabel("Log:"))
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        right_layout.addWidget(self.log_text, stretch=1)

        self.progress = QProgressBar()
        right_layout.addWidget(self.progress)

        # --- Assemble ---
        splitter = QSplitter(Qt.Horizontal)
        splitter.addWidget(left)
        splitter.addWidget(right)
        splitter.setStretchFactor(1, 1)
        main_layout.addWidget(splitter)

        self.statusBar().showMessage("Ready")

    # -----------------------------------------------------------------------
    # Folder / model selection
    # -----------------------------------------------------------------------
    def _select_input(self):
        folder = QFileDialog.getExistingDirectory(self, "Select Input Folder")
        if not folder:
            return
        self.input_edit.setText(folder)
        self._image_paths = discover_images(Path(folder))
        self.image_list.clear()
        for p in self._image_paths:
            self.image_list.addItem(p.name)
        self.statusBar().showMessage(f"Found {len(self._image_paths)} images")

    def _select_output(self):
        folder = QFileDialog.getExistingDirectory(self, "Select Output Folder")
        if folder:
            self.output_edit.setText(folder)

    def _select_landmark_model(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Select Landmark Model Checkpoint", "", "PyTorch Checkpoint (*.pt);;All Files (*)"
        )
        if path:
            self.lm_edit.setText(path)

    def _select_seg_model(self):
        folder = QFileDialog.getExistingDirectory(self, "Select Segmentation Model Folder")
        if folder:
            self.seg_edit.setText(folder)

    # -----------------------------------------------------------------------
    # Settings persistence
    # -----------------------------------------------------------------------
    def _selected_outputs(self) -> set[str]:
        return {key for key, chk in self.output_checks.items() if chk.isChecked()}

    def _save_settings(self):
        s = self.settings
        s.setValue("input_folder", self.input_edit.text())
        s.setValue("output_folder", self.output_edit.text())
        s.setValue("landmark_model", self.lm_edit.text())
        s.setValue("segmentation_model", self.seg_edit.text())
        s.setValue("scale", self.scale_spin.value())
        for key, chk in self.output_checks.items():
            s.setValue(f"output/{key}", chk.isChecked())

    def _restore_settings(self):
        s = self.settings
        val = s.value("input_folder", "")
        if val:
            self.input_edit.setText(val)
            folder = Path(val)
            if folder.is_dir():
                self._image_paths = discover_images(folder)
                self.image_list.clear()
                for p in self._image_paths:
                    self.image_list.addItem(p.name)
        val = s.value("output_folder", "")
        if val:
            self.output_edit.setText(val)
        val = s.value("landmark_model", "")
        if val:
            self.lm_edit.setText(val)
        val = s.value("segmentation_model", "")
        if val:
            self.seg_edit.setText(val)
        scale = s.value("scale", None)
        if scale is not None:
            self.scale_spin.setValue(float(scale))
        for key, chk in self.output_checks.items():
            saved = s.value(f"output/{key}", None)
            if saved is not None:
                chk.setChecked(saved == "true" or saved is True)

    # -----------------------------------------------------------------------
    # Pipeline execution
    # -----------------------------------------------------------------------
    def _run_pipeline(self):
        # Validate required fields
        if not self.input_edit.text():
            QMessageBox.warning(self, "Missing Input", "Please select an input folder.")
            return
        if not self.output_edit.text():
            QMessageBox.warning(self, "Missing Output", "Please select an output folder.")
            return
        if not self.lm_edit.text():
            QMessageBox.warning(self, "Missing Model", "Please select a landmark model.")
            return
        if not self.seg_edit.text():
            QMessageBox.warning(self, "Missing Model", "Please select a segmentation model folder.")
            return

        scale = self.scale_spin.value()
        if scale == 0.0:
            scale = None

        # UI state
        self.btn_run.setEnabled(False)
        self.btn_cancel.setEnabled(True)
        self.progress.setValue(0)
        self.log_text.clear()
        self._save_settings()
        self._log("Starting TRACE pipeline...")

        for i in range(self.image_list.count()):
            self.image_list.item(i).setForeground(QColor(208, 208, 208))

        self.worker = TraceWorker(
            kwargs=dict(
                input_dir=Path(self.input_edit.text()),
                output_dir=Path(self.output_edit.text()),
                landmark_checkpoint=Path(self.lm_edit.text()),
                segmentation_model_dir=Path(self.seg_edit.text()),
                scale=scale,
                keep_intermediates=False,
                outputs=self._selected_outputs(),
            )
        )
        self.worker.progress.connect(self._on_progress)
        self.worker.all_done.connect(self._on_all_done)
        self.worker.error.connect(self._on_error)
        self.worker.start()

    def _cancel_pipeline(self):
        if self.worker:
            self.worker.cancel()
            self._log("Cancelling...")

    # -----------------------------------------------------------------------
    # Logging and callbacks
    # -----------------------------------------------------------------------
    def _log(self, msg):
        self.log_text.append(msg)
        scrollbar = self.log_text.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())

    def _on_progress(self, idx, total, name, stage, detail):
        self.progress.setMaximum(total)
        self.progress.setValue(idx + 1)
        msg = f"[{idx + 1}/{total}] {name}: {stage} - {detail}"
        self.statusBar().showMessage(msg)
        self._log(msg)

    def _on_all_done(self, results):
        self.btn_run.setEnabled(True)
        self.btn_cancel.setEnabled(False)

        if not results:
            self._log("\nPipeline cancelled or no results.")
            self.statusBar().showMessage("Cancelled")
            return

        succeeded = sum(1 for r in results if r.error is None)
        failed = sum(1 for r in results if r.error is not None)
        summary = f"Done: {succeeded} succeeded, {failed} failed out of {len(results)} images."
        self._log(f"\n{summary}")
        self.statusBar().showMessage(summary)

        if failed:
            self._log("\nFailed images:")
            for r in results:
                if r.error:
                    self._log(f"  {r.image_path.name} ({r.error_stage}): {r.error.splitlines()[0]}")

    def _on_error(self, msg):
        self.btn_run.setEnabled(True)
        self.btn_cancel.setEnabled(False)
        self._log(f"\nFatal error: {msg}")
        QMessageBox.critical(self, "Pipeline Error", msg)


# ---------------------------------------------------------------------------
# Dark Fusion theme + entry point
# ---------------------------------------------------------------------------
def _apply_dark_palette(app: QApplication):
    app.setStyle("Fusion")
    p = QPalette()
    p.setColor(QPalette.Window, QColor(45, 45, 45))
    p.setColor(QPalette.WindowText, QColor(208, 208, 208))
    p.setColor(QPalette.Base, QColor(30, 30, 30))
    p.setColor(QPalette.AlternateBase, QColor(45, 45, 45))
    p.setColor(QPalette.ToolTipBase, QColor(45, 45, 45))
    p.setColor(QPalette.ToolTipText, QColor(208, 208, 208))
    p.setColor(QPalette.Text, QColor(208, 208, 208))
    p.setColor(QPalette.Button, QColor(55, 55, 55))
    p.setColor(QPalette.ButtonText, QColor(208, 208, 208))
    p.setColor(QPalette.BrightText, QColor(255, 51, 51))
    p.setColor(QPalette.Link, QColor(66, 133, 244))
    p.setColor(QPalette.Highlight, QColor(66, 133, 244))
    p.setColor(QPalette.HighlightedText, QColor(255, 255, 255))
    p.setColor(QPalette.Disabled, QPalette.Text, QColor(128, 128, 128))
    p.setColor(QPalette.Disabled, QPalette.ButtonText, QColor(128, 128, 128))
    app.setPalette(p)


def main():
    app = QApplication(sys.argv)
    _apply_dark_palette(app)
    window = TraceWindow()
    window.show()
    sys.exit(app.exec_())
