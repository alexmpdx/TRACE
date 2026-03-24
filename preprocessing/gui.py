"""
PyQt5 GUI for the preprocessing pipeline.

Dark Fusion theme matching the modelTOjson app. Runs pipeline stages in a
background QThread with progress reporting and per-image error resilience.
"""

import sys
import traceback
from pathlib import Path

from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtGui import QColor, QPalette
from PyQt5.QtWidgets import (
    QApplication,
    QCheckBox,
    QFileDialog,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QSplitter,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from preprocessing.pipeline import (
    PipelineResult,
    _auto_device,
    discover_images,
    process_folder,
)


# ---------------------------------------------------------------------------
# Worker thread
# ---------------------------------------------------------------------------
class PipelineWorker(QThread):
    """Runs process_folder() in a background thread."""

    progress = pyqtSignal(int, int, str, str)  # idx, total, name, status
    image_done = pyqtSignal(int, bool, str)  # idx, success, message
    all_done = pyqtSignal(list)  # list[PipelineResult]
    error = pyqtSignal(str)  # fatal error

    def __init__(
        self, input_dir, output_dir, landmark_checkpoint, segmentation_model_dir, stages, device, keep_chopped
    ):
        super().__init__()
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.landmark_checkpoint = landmark_checkpoint
        self.segmentation_model_dir = segmentation_model_dir
        self.stages = stages
        self.device = device
        self.keep_chopped = keep_chopped
        self._cancel = False

    def cancel(self):
        self._cancel = True

    def run(self):
        try:
            results = process_folder(
                input_dir=self.input_dir,
                output_dir=self.output_dir,
                landmark_checkpoint=self.landmark_checkpoint,
                segmentation_model_dir=self.segmentation_model_dir,
                stages=self.stages,
                device=self.device,
                keep_chopped=self.keep_chopped,
                progress_callback=self._on_progress,
            )
            self.all_done.emit(results)
        except Exception as e:
            self.error.emit(f"{e}\n{traceback.format_exc()}")

    def _on_progress(self, idx, total, name, status):
        if self._cancel:
            raise InterruptedError("Cancelled by user")
        self.progress.emit(idx, total, name, status)
        if status == "done":
            self.image_done.emit(idx, True, name)
        elif status.startswith("error"):
            self.image_done.emit(idx, False, f"{name}: {status}")


# ---------------------------------------------------------------------------
# Main window
# ---------------------------------------------------------------------------
class PreprocessingWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Wing Preprocessing Pipeline")
        self.resize(1000, 700)
        self.worker = None
        self._image_paths = []
        self._build_ui()

    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QHBoxLayout(central)

        # --- Left panel ---
        left = QWidget()
        left.setMaximumWidth(360)
        left_layout = QVBoxLayout(left)
        left_layout.setContentsMargins(4, 4, 4, 4)

        # Folders
        folder_group = QGroupBox("Folders")
        fg_layout = QVBoxLayout(folder_group)

        fg_layout.addWidget(QLabel("Input folder:"))
        input_row = QHBoxLayout()
        self.input_edit = QLineEdit()
        self.input_edit.setReadOnly(True)
        self.input_edit.setPlaceholderText("Select input folder...")
        btn_input = QPushButton("Browse...")
        btn_input.clicked.connect(self._select_input)
        input_row.addWidget(self.input_edit, stretch=1)
        input_row.addWidget(btn_input)
        fg_layout.addLayout(input_row)

        fg_layout.addWidget(QLabel("Output folder:"))
        output_row = QHBoxLayout()
        self.output_edit = QLineEdit()
        self.output_edit.setReadOnly(True)
        self.output_edit.setPlaceholderText("Select output folder...")
        btn_output = QPushButton("Browse...")
        btn_output.clicked.connect(self._select_output)
        output_row.addWidget(self.output_edit, stretch=1)
        output_row.addWidget(btn_output)
        fg_layout.addLayout(output_row)
        left_layout.addWidget(folder_group)

        # Models
        model_group = QGroupBox("Models")
        mdl_layout = QVBoxLayout(model_group)

        mdl_layout.addWidget(QLabel("Landmark model (.pt):"))
        lm_row = QHBoxLayout()
        self.lm_edit = QLineEdit()
        self.lm_edit.setReadOnly(True)
        self.lm_edit.setPlaceholderText("Select landmark checkpoint...")
        self.btn_lm = QPushButton("Browse...")
        self.btn_lm.clicked.connect(self._select_landmark_model)
        lm_row.addWidget(self.lm_edit, stretch=1)
        lm_row.addWidget(self.btn_lm)
        mdl_layout.addLayout(lm_row)

        mdl_layout.addWidget(QLabel("Segmentation model folder:"))
        seg_row = QHBoxLayout()
        self.seg_edit = QLineEdit()
        self.seg_edit.setReadOnly(True)
        self.seg_edit.setPlaceholderText("Select segmentation model folder...")
        self.btn_seg = QPushButton("Browse...")
        self.btn_seg.clicked.connect(self._select_seg_model)
        seg_row.addWidget(self.seg_edit, stretch=1)
        seg_row.addWidget(self.btn_seg)
        mdl_layout.addLayout(seg_row)
        left_layout.addWidget(model_group)

        # Stages
        stage_group = QGroupBox("Stages")
        sg_layout = QVBoxLayout(stage_group)
        self.chk_landmarks = QCheckBox("Landmarks")
        self.chk_landmarks.setChecked(True)
        self.chk_landmarks.toggled.connect(self._on_stage_toggled)
        self.chk_hinge = QCheckBox("Hinge Chop")
        self.chk_hinge.setChecked(True)
        self.chk_hinge.toggled.connect(self._on_stage_toggled)
        self.chk_segment = QCheckBox("Segmentation")
        self.chk_segment.setChecked(True)
        self.chk_segment.toggled.connect(self._on_stage_toggled)
        self.chk_wing = QCheckBox("Add Wing")
        self.chk_wing.setChecked(True)
        sg_layout.addWidget(self.chk_landmarks)
        sg_layout.addWidget(self.chk_hinge)
        sg_layout.addWidget(self.chk_segment)
        sg_layout.addWidget(self.chk_wing)
        left_layout.addWidget(stage_group)

        # Run / Cancel
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

        # Image list
        right_layout.addWidget(QLabel("Images:"))
        self.image_list = QListWidget()
        right_layout.addWidget(self.image_list, stretch=1)

        # Log
        right_layout.addWidget(QLabel("Log:"))
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        right_layout.addWidget(self.log_text, stretch=1)

        # Progress bar
        self.progress = QProgressBar()
        right_layout.addWidget(self.progress)

        # Assemble
        splitter = QSplitter(Qt.Horizontal)
        splitter.addWidget(left)
        splitter.addWidget(right)
        splitter.setStretchFactor(1, 1)
        main_layout.addWidget(splitter)

        self.statusBar().showMessage("Ready")

    # --- Folder/model selection ---
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
            self,
            "Select Landmark Model Checkpoint",
            "",
            "PyTorch Checkpoint (*.pt);;All Files (*)",
        )
        if path:
            self.lm_edit.setText(path)

    def _select_seg_model(self):
        folder = QFileDialog.getExistingDirectory(self, "Select Segmentation Model Folder")
        if folder:
            self.seg_edit.setText(folder)

    # --- Stage toggling ---
    def _on_stage_toggled(self):
        needs_lm = self.chk_landmarks.isChecked() or self.chk_hinge.isChecked()
        needs_seg = self.chk_segment.isChecked()
        self.btn_lm.setEnabled(needs_lm)
        self.lm_edit.setEnabled(needs_lm)
        self.btn_seg.setEnabled(needs_seg)
        self.seg_edit.setEnabled(needs_seg)

    # --- Pipeline execution ---
    def _run_pipeline(self):
        # Validate
        if not self.input_edit.text():
            QMessageBox.warning(self, "Missing Input", "Please select an input folder.")
            return
        if not self.output_edit.text():
            QMessageBox.warning(self, "Missing Output", "Please select an output folder.")
            return

        stages = (
            self.chk_landmarks.isChecked(),
            self.chk_hinge.isChecked(),
            self.chk_segment.isChecked(),
            self.chk_wing.isChecked(),
        )
        if not any(stages):
            QMessageBox.warning(self, "No Stages", "Please select at least one stage to run.")
            return

        landmark_checkpoint = None
        segmentation_model_dir = None

        if stages[0] or stages[1]:
            if not self.lm_edit.text():
                QMessageBox.warning(self, "Missing Model", "Landmark model is required for landmarks/hinge stages.")
                return
            landmark_checkpoint = Path(self.lm_edit.text())

        if stages[2]:
            if not self.seg_edit.text():
                QMessageBox.warning(self, "Missing Model", "Segmentation model is required for segmentation stage.")
                return
            segmentation_model_dir = Path(self.seg_edit.text())

        # UI state
        self.btn_run.setEnabled(False)
        self.btn_cancel.setEnabled(True)
        self.progress.setValue(0)
        self.log_text.clear()
        self._log("Starting pipeline...")

        # Reset image list icons
        for i in range(self.image_list.count()):
            item = self.image_list.item(i)
            item.setForeground(QColor(208, 208, 208))

        device = _auto_device()

        self.worker = PipelineWorker(
            input_dir=Path(self.input_edit.text()),
            output_dir=Path(self.output_edit.text()),
            landmark_checkpoint=landmark_checkpoint,
            segmentation_model_dir=segmentation_model_dir,
            stages=stages,
            device=device,
            keep_chopped=False,
        )
        self.worker.progress.connect(self._on_progress)
        self.worker.image_done.connect(self._on_image_done)
        self.worker.all_done.connect(self._on_all_done)
        self.worker.error.connect(self._on_error)
        self.worker.start()

    def _cancel_pipeline(self):
        if self.worker:
            self.worker.cancel()
            self._log("Cancelling...")

    def _log(self, msg):
        self.log_text.append(msg)
        # Auto-scroll to bottom
        scrollbar = self.log_text.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())

    def _on_progress(self, idx, total, name, status):
        self.progress.setMaximum(total)
        self.progress.setValue(idx + 1)
        self.statusBar().showMessage(f"[{idx + 1}/{total}] {name}: {status}")
        self._log(f"[{idx + 1}/{total}] {name}: {status}")

    def _on_image_done(self, idx, success, message):
        if idx < self.image_list.count():
            item = self.image_list.item(idx)
            if success:
                item.setForeground(QColor(100, 200, 100))
            else:
                item.setForeground(QColor(255, 80, 80))

    def _on_all_done(self, results):
        self.btn_run.setEnabled(True)
        self.btn_cancel.setEnabled(False)

        succeeded = sum(1 for r in results if r.error is None)
        failed = sum(1 for r in results if r.error is not None)
        summary = f"Done: {succeeded} succeeded, {failed} failed out of {len(results)} images."
        self._log(f"\n{summary}")
        self.statusBar().showMessage(summary)

        if failed:
            self._log("\nFailed images:")
            for r in results:
                if r.error:
                    self._log(f"  {r.image_path.name}: {r.error.splitlines()[0]}")

    def _on_error(self, msg):
        self.btn_run.setEnabled(True)
        self.btn_cancel.setEnabled(False)
        self._log(f"\nFatal error: {msg}")
        QMessageBox.critical(self, "Pipeline Error", msg)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main():
    app = QApplication(sys.argv)
    app.setStyle("Fusion")

    palette = QPalette()
    palette.setColor(QPalette.Window, QColor(45, 45, 45))
    palette.setColor(QPalette.WindowText, QColor(208, 208, 208))
    palette.setColor(QPalette.Base, QColor(30, 30, 30))
    palette.setColor(QPalette.AlternateBase, QColor(45, 45, 45))
    palette.setColor(QPalette.ToolTipBase, QColor(45, 45, 45))
    palette.setColor(QPalette.ToolTipText, QColor(208, 208, 208))
    palette.setColor(QPalette.Text, QColor(208, 208, 208))
    palette.setColor(QPalette.Button, QColor(55, 55, 55))
    palette.setColor(QPalette.ButtonText, QColor(208, 208, 208))
    palette.setColor(QPalette.BrightText, QColor(255, 51, 51))
    palette.setColor(QPalette.Link, QColor(66, 133, 244))
    palette.setColor(QPalette.Highlight, QColor(66, 133, 244))
    palette.setColor(QPalette.HighlightedText, QColor(255, 255, 255))
    palette.setColor(QPalette.Disabled, QPalette.Text, QColor(128, 128, 128))
    palette.setColor(QPalette.Disabled, QPalette.ButtonText, QColor(128, 128, 128))
    app.setPalette(palette)

    window = PreprocessingWindow()
    window.show()
    sys.exit(app.exec_())
