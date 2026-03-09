"""Batch processing dialog and file chooser dialog."""

from __future__ import annotations

import traceback
from pathlib import Path
from typing import Optional

from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtGui import QFont
from PyQt5.QtWidgets import (
    QCheckBox,
    QDialog,
    QDoubleSpinBox,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QProgressBar,
    QPushButton,
    QTextEdit,
    QVBoxLayout,
)

from WingVeinAnalyzer.gui.file_selector import FilePair


class FileChooserDialog(QDialog):
    """Simple dialog to choose one file pair from a list."""

    def __init__(self, pairs: list[FilePair], parent=None):
        super().__init__(parent)
        self.setWindowTitle("Choose Wing Image")
        self.resize(500, 400)
        self._pairs = pairs
        self._selected: Optional[FilePair] = None

        layout = QVBoxLayout(self)
        layout.addWidget(QLabel("Select a wing image to analyze:"))

        self._list = QListWidget()
        self._list.setFont(QFont("Menlo", 11))
        for pair in pairs:
            gt_info = ""
            if pair.gt_intervein_path:
                gt_info += " [GT:regions]"
            if pair.gt_skeleton_path:
                gt_info += " [GT:veins]"
            self._list.addItem(f"{pair.display_name}{gt_info}")
        self._list.setCurrentRow(0)
        self._list.itemDoubleClicked.connect(self._on_double_click)
        layout.addWidget(self._list)

        btn_layout = QHBoxLayout()
        ok_btn = QPushButton("Open")
        ok_btn.clicked.connect(self._on_ok)
        cancel_btn = QPushButton("Cancel")
        cancel_btn.clicked.connect(self.reject)
        btn_layout.addStretch()
        btn_layout.addWidget(cancel_btn)
        btn_layout.addWidget(ok_btn)
        layout.addLayout(btn_layout)

    def selected_pair(self) -> Optional[FilePair]:
        return self._selected

    def _on_ok(self) -> None:
        row = self._list.currentRow()
        if 0 <= row < len(self._pairs):
            self._selected = self._pairs[row]
            self.accept()

    def _on_double_click(self, item: QListWidgetItem) -> None:
        self._on_ok()


class BatchWorker(QThread):
    """Worker thread for batch processing."""

    progress = pyqtSignal(int, str)  # file index, status message
    file_done = pyqtSignal(int, bool, str)  # file index, success, message
    all_done = pyqtSignal()

    def __init__(
        self, pairs: list[FilePair], output_dir: Path, smooth_sigma: float = 3.0, um_per_px: float = 0.483, parent=None
    ):
        super().__init__(parent)
        self._pairs = pairs
        self._output_dir = output_dir
        self._smooth_sigma = smooth_sigma
        self._um_per_px = um_per_px
        self._cancelled = False

    def cancel(self) -> None:
        self._cancelled = True

    def run(self):
        from WingVeinAnalyzer.controllers.analysis_controller import run_pipeline

        for i, pair in enumerate(self._pairs):
            if self._cancelled:
                break

            self.progress.emit(i, f"Processing {pair.display_name}...")

            try:
                result = run_pipeline(
                    image_path=pair.image_path,
                    geojson_path=pair.geojson_path,
                    output_dir=self._output_dir / pair.display_name,
                    microns_per_pixel=self._um_per_px,
                    smooth_sigma=self._smooth_sigma,
                )
                self.file_done.emit(i, True, f"{pair.display_name}: OK")
            except Exception:
                tb = traceback.format_exc()
                self.file_done.emit(i, False, f"{pair.display_name}: ERROR\n{tb[:500]}")

        self.all_done.emit()


class BatchDialog(QDialog):
    """Batch processing dialog with file checkboxes and progress."""

    def __init__(self, pairs: list[FilePair], parent=None, smooth_sigma: float = 3.0, um_per_px: float = 0.483):
        super().__init__(parent)
        self.setWindowTitle("Batch Processing")
        self.resize(700, 500)
        self._pairs = pairs
        self._smooth_sigma = smooth_sigma
        self._um_per_px = um_per_px
        self._worker: Optional[BatchWorker] = None

        layout = QVBoxLayout(self)

        # Scale input
        scale_layout = QHBoxLayout()
        scale_layout.addWidget(QLabel("\u00b5m/px:"))
        self._scale_spin = QDoubleSpinBox()
        self._scale_spin.setRange(0.001, 100.0)
        self._scale_spin.setDecimals(3)
        self._scale_spin.setValue(um_per_px)
        self._scale_spin.setSingleStep(0.01)
        self._scale_spin.setFixedWidth(90)
        self._scale_spin.setToolTip("Micrometers per pixel")
        scale_layout.addWidget(self._scale_spin)
        scale_layout.addStretch()
        layout.addLayout(scale_layout)

        # File list with checkboxes
        layout.addWidget(QLabel("Files to process:"))
        self._checkboxes: list[QCheckBox] = []
        for pair in pairs:
            cb = QCheckBox(pair.display_name)
            cb.setChecked(True)
            cb.setFont(QFont("Menlo", 11))
            self._checkboxes.append(cb)
            layout.addWidget(cb)

        # Output directory
        dir_layout = QHBoxLayout()
        dir_layout.addWidget(QLabel("Output:"))
        self._dir_label = QLabel(str(pairs[0].image_path.parent / "output"))
        self._dir_label.setFont(QFont("Menlo", 10))
        dir_layout.addWidget(self._dir_label, stretch=1)
        browse_btn = QPushButton("Browse...")
        browse_btn.clicked.connect(self._browse_output)
        dir_layout.addWidget(browse_btn)
        layout.addLayout(dir_layout)

        # Progress
        self._progress = QProgressBar()
        self._progress.setRange(0, len(pairs))
        self._progress.setValue(0)
        layout.addWidget(self._progress)

        # Log
        self._log = QTextEdit()
        self._log.setReadOnly(True)
        self._log.setFont(QFont("Menlo", 10))
        self._log.setMaximumHeight(200)
        layout.addWidget(self._log)

        # Buttons
        btn_layout = QHBoxLayout()
        self._run_btn = QPushButton("Run")
        self._run_btn.clicked.connect(self._on_run)
        self._cancel_btn = QPushButton("Cancel")
        self._cancel_btn.clicked.connect(self._on_cancel)
        self._cancel_btn.setEnabled(False)
        self._close_btn = QPushButton("Close")
        self._close_btn.clicked.connect(self.accept)
        btn_layout.addStretch()
        btn_layout.addWidget(self._close_btn)
        btn_layout.addWidget(self._cancel_btn)
        btn_layout.addWidget(self._run_btn)
        layout.addLayout(btn_layout)

    def _browse_output(self) -> None:
        folder = QFileDialog.getExistingDirectory(self, "Select Output Directory")
        if folder:
            self._dir_label.setText(folder)

    def _on_run(self) -> None:
        """Start batch processing."""
        selected = [pair for pair, cb in zip(self._pairs, self._checkboxes) if cb.isChecked()]
        if not selected:
            return

        output_dir = Path(self._dir_label.text())
        output_dir.mkdir(parents=True, exist_ok=True)

        self._progress.setRange(0, len(selected))
        self._progress.setValue(0)
        self._log.clear()
        self._run_btn.setEnabled(False)
        self._cancel_btn.setEnabled(True)
        for cb in self._checkboxes:
            cb.setEnabled(False)

        self._worker = BatchWorker(selected, output_dir, self._smooth_sigma, self._scale_spin.value(), self)
        self._worker.progress.connect(self._on_progress)
        self._worker.file_done.connect(self._on_file_done)
        self._worker.all_done.connect(self._on_all_done)
        self._worker.start()

    def _on_cancel(self) -> None:
        if self._worker:
            self._worker.cancel()
            self._log.append("Cancelling...")

    def _on_progress(self, index: int, msg: str) -> None:
        self._progress.setValue(index)
        self._log.append(msg)

    def _on_file_done(self, index: int, success: bool, msg: str) -> None:
        self._progress.setValue(index + 1)
        if success:
            self._log.append(f"  {msg}")
        else:
            self._log.append(f"  ERROR: {msg}")

    def _on_all_done(self) -> None:
        self._run_btn.setEnabled(True)
        self._cancel_btn.setEnabled(False)
        for cb in self._checkboxes:
            cb.setEnabled(True)
        self._log.append("\nBatch processing complete.")
