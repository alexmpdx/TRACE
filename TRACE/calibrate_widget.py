"""Reusable Calibrate-workers UI widget.

Owns the Calibrate button, status label, progress bar, time-remaining
estimate, and the background `_CalibrationThread`. Used in both the
settings dialog and the main-window "?" help dialog so the calibration
flow looks identical in either place.
"""

from __future__ import annotations

import time as _time
from pathlib import Path

from PyQt5.QtCore import QThread, QTimer, pyqtSignal
from PyQt5.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QProgressBar,
    QToolButton,
    QVBoxLayout,
    QWidget,
)


class _CalibrationThread(QThread):
    """Runs TRACE Stage 1 + Stage 2 calibration off the UI thread."""

    progress = pyqtSignal(str, str)
    finished_ok = pyqtSignal(dict)
    failed = pyqtSignal(str)

    def __init__(self, image_or_folder: Path, landmark_model: Path, segmentation_model: Path, parent=None):
        super().__init__(parent)
        self._image_or_folder = image_or_folder
        self._landmark_model = landmark_model
        self._segmentation_model = segmentation_model

    def run(self):
        try:
            from TRACE.calibrate_workers import calibrate_for_trace

            result = calibrate_for_trace(
                image_or_folder=self._image_or_folder,
                landmark_checkpoint=self._landmark_model,
                segmentation_model_dir=self._segmentation_model,
                progress_callback=lambda stage, detail: self.progress.emit(stage, detail),
            )
            self.finished_ok.emit(result)
        except Exception as e:
            self.failed.emit(str(e))


class CalibrateWidget(QWidget):
    """Calibrate button + progress bar + ETA, emitting `applied(int)` on accept.

    Callers must wire `set_paths()` whenever the input/landmark/segmentation
    paths change. Connect to the `applied` signal to receive the recommended
    workers value when the user clicks Apply on the result dialog.
    """

    applied = pyqtSignal(int)

    _PHASE_BUDGETS = {"preprocessing": 30.0, "calibration": 60.0}

    def __init__(self, parent=None):
        super().__init__(parent)
        self._input_path = ""
        self._landmark_path = ""
        self._seg_path = ""
        self._thread = None
        self._started_at = None
        self._phase_starts: dict[str, float] = {}
        self._current_phase: str | None = None

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)

        header = QHBoxLayout()
        self._calibrate_btn = QToolButton()
        self._calibrate_btn.setText("Calibrate")
        self._calibrate_btn.setToolTip(
            "Measure peak Stage 2 memory on one image from your input folder, "
            "then suggest a safe Workers value for this machine.\n"
            "Requires the Input folder (main window) and the Landmark + Segmentation "
            "models (Settings → Models) to be set. Takes ~1 minute."
        )
        self._calibrate_btn.clicked.connect(self._start_calibration)
        header.addWidget(self._calibrate_btn)
        self._status = QLabel("")
        self._status.setStyleSheet("color: #555;")
        header.addWidget(self._status, stretch=1)
        outer.addLayout(header)

        self._progress = QProgressBar()
        self._progress.setRange(0, 1000)
        self._progress.setValue(0)
        self._progress.setVisible(False)
        self._progress.setFormat("%p%")
        outer.addWidget(self._progress)
        self._eta = QLabel("")
        self._eta.setStyleSheet("color: #555;")
        self._eta.setVisible(False)
        outer.addWidget(self._eta)

        self._timer = QTimer(self)
        self._timer.setInterval(500)
        self._timer.timeout.connect(self._tick)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def set_paths(self, input_path: str, landmark_path: str, seg_path: str):
        self._input_path = input_path or ""
        self._landmark_path = landmark_path or ""
        self._seg_path = seg_path or ""

    # ------------------------------------------------------------------
    # Calibration lifecycle
    # ------------------------------------------------------------------
    def _start_calibration(self):
        missing = []
        if not self._input_path:
            missing.append("Input folder")
        if not self._landmark_path:
            missing.append("Landmark model")
        if not self._seg_path:
            missing.append("Segmentation model")
        if missing:
            QMessageBox.warning(
                self,
                "Calibration prerequisites missing",
                "Set the following in the main window before calibrating:\n  - " + "\n  - ".join(missing),
            )
            return
        if self._thread is not None and self._thread.isRunning():
            return

        self._calibrate_btn.setEnabled(False)
        self._status.setText("Calibrating… (~1 min)")
        self._begin_progress()

        thread = _CalibrationThread(
            image_or_folder=Path(self._input_path),
            landmark_model=Path(self._landmark_path),
            segmentation_model=Path(self._seg_path),
            parent=self,
        )
        thread.progress.connect(self._on_progress)
        thread.finished_ok.connect(self._on_finished)
        thread.failed.connect(self._on_failed)
        thread.finished.connect(self._on_thread_done)
        self._thread = thread
        thread.start()

    def _on_progress(self, stage: str, detail: str):
        self._status.setText(f"{stage}: {detail}")
        if stage in self._PHASE_BUDGETS and stage not in self._phase_starts:
            self._phase_starts[stage] = _time.monotonic()
            self._current_phase = stage
            self._refresh(stage, 0.0)

    def _on_finished(self, result: dict):
        from TRACE.calibrate_workers import format_report

        rec = result["recommendation"]
        suggested = int(rec["recommended_workers"])
        report = format_report(result)

        box = QMessageBox(self)
        box.setWindowTitle("Calibration result")
        box.setIcon(QMessageBox.Information)
        box.setText(f"Suggested Workers: {suggested}\n" f"(binding constraint: {rec.get('binding_constraint', 'n/a')})")
        box.setDetailedText(report)
        apply_btn = box.addButton("Apply", QMessageBox.AcceptRole)
        box.addButton("Keep current", QMessageBox.RejectRole)
        box.exec_()
        if box.clickedButton() is apply_btn:
            self.applied.emit(suggested)
        self._status.setText(f"Suggested Workers: {suggested}")

    def _on_failed(self, message: str):
        self._end_progress(completed=False)
        QMessageBox.critical(self, "Calibration failed", message)
        self._status.setText("Calibration failed")

    def _on_thread_done(self):
        self._calibrate_btn.setEnabled(True)
        self._thread = None
        if self._timer.isActive():
            self._end_progress(completed=True)

    # ------------------------------------------------------------------
    # Progress UI
    # ------------------------------------------------------------------
    def _begin_progress(self):
        self._started_at = _time.monotonic()
        self._phase_starts = {}
        self._current_phase = None
        self._progress.setValue(0)
        self._progress.setVisible(True)
        self._eta.setText("Estimating…")
        self._eta.setVisible(True)
        self._timer.start()

    def _end_progress(self, completed: bool):
        self._timer.stop()
        if completed:
            self._progress.setValue(self._progress.maximum())
            self._eta.setText("Done")
        QTimer.singleShot(1500, self._hide_progress)

    def _hide_progress(self):
        self._progress.setVisible(False)
        self._eta.setVisible(False)

    def _tick(self):
        if self._started_at is None:
            return
        phase = self._current_phase or "preprocessing"
        phase_start = self._phase_starts.get(phase, self._started_at)
        self._refresh(phase, max(0.0, _time.monotonic() - phase_start))

    def _refresh(self, phase: str, phase_elapsed: float):
        budgets = self._PHASE_BUDGETS
        total = sum(budgets.values())
        slice_starts: dict[str, float] = {}
        cum = 0.0
        for name, dur in budgets.items():
            slice_starts[name] = cum / total
            cum += dur
        slice_size = budgets.get(phase, 30.0) / total
        intra = min(phase_elapsed / max(budgets.get(phase, 30.0), 1e-3), 0.99)
        frac = max(0.0, min(slice_starts.get(phase, 0.0) + slice_size * intra, 0.99))
        self._progress.setValue(int(frac * self._progress.maximum()))

        elapsed_total = _time.monotonic() - (self._started_at or _time.monotonic())
        remaining = max(0.0, budgets.get(phase, 30.0) - phase_elapsed)
        seen = set(self._phase_starts.keys()) | {phase}
        for name, dur in budgets.items():
            if name not in seen:
                remaining += dur
        self._eta.setText(f"Elapsed {self._fmt_secs(elapsed_total)}  ·  ~{self._fmt_secs(remaining)} remaining")

    @staticmethod
    def _fmt_secs(s: float) -> str:
        s = max(0.0, s)
        if s < 60:
            return f"{int(round(s))}s"
        m, sec = divmod(int(round(s)), 60)
        return f"{m}m{sec:02d}s"
