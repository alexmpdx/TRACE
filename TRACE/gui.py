"""PyQt5 GUI for the TRACE combined pipeline.

Dark Fusion theme matching the preprocessing app. Runs the pipeline in a
background QThread with progress reporting.
"""

import logging
import sys
import time
import traceback
from collections import OrderedDict
from pathlib import Path

from identify_features.config import PipelineConfig
from PyQt5.QtCore import QSettings, Qt, QThread, pyqtSignal
from PyQt5.QtGui import QColor, QPalette
from PyQt5.QtWidgets import (
    QApplication,
    QCheckBox,
    QDialog,
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
    QSpinBox,
    QSplitter,
    QTextEdit,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from preprocessing.pipeline import discover_images
from TRACE.config_io import config_from_json, config_to_json, load_config, save_config
from TRACE.pipeline import DEFAULT_MAX_WORKERS, OUTPUT_TYPES, trace_folder
from TRACE.settings_dialog import PipelineConfigDialog

# ---------------------------------------------------------------------------
# Worker thread
# ---------------------------------------------------------------------------


_CAPTURED_LOGGERS = ("identify_features", "TRACE", "preprocessing")


class _SignalLogHandler(logging.Handler):
    """Logging handler that forwards records through a pyqtSignal."""

    def __init__(self, signal):
        super().__init__()
        self._signal = signal
        self.setFormatter(logging.Formatter("%(name)s %(levelname)s: %(message)s"))

    def emit(self, record):
        try:
            self._signal.emit(self.format(record))
        except Exception:
            pass


class TraceWorker(QThread):
    """Runs trace_folder() in a background thread."""

    progress = pyqtSignal(int, int, str, str, str)  # idx, total, name, stage, detail
    log_message = pyqtSignal(str)  # forwarded log records from captured loggers
    all_done = pyqtSignal(list)  # results
    error = pyqtSignal(str)

    def __init__(self, kwargs):
        super().__init__()
        self.kwargs = kwargs
        self._cancel = False

    def cancel(self):
        self._cancel = True

    def run(self):
        handler = _SignalLogHandler(self.log_message)
        handler.setLevel(logging.INFO)
        attached = []
        for name in _CAPTURED_LOGGERS:
            lg = logging.getLogger(name)
            if lg.level == logging.NOTSET or lg.level > logging.INFO:
                lg.setLevel(logging.INFO)
            lg.addHandler(handler)
            attached.append(lg)
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
        finally:
            for lg in attached:
                lg.removeHandler(handler)


# ---------------------------------------------------------------------------
# Main window
# ---------------------------------------------------------------------------


class TraceWindow(QMainWindow):
    _PARALLEL_WORKERS_DETAILS_TEXT = (
        "Each worker runs the full identifyFeatures pipeline on one wing "
        "and uses significant CPU and RAM. Setting this value too high can:\n\n"
        "  • Exhaust system RAM and cause the process to be killed\n"
        "  • Saturate all CPU cores and freeze other applications\n"
        "  • Trigger thermal throttling on laptops\n"
        "  • Produce no speedup past the number of physical cores\n\n"
        "A safe starting point is 2–4. Do not exceed your machine's "
        "physical core count unless you know what you're doing."
    )
    _PARALLEL_WORKERS_WARNING_TEXT = "You are enabling parallel Stage 2 analysis.\n\n" + _PARALLEL_WORKERS_DETAILS_TEXT

    def __init__(self):
        super().__init__()
        self.setWindowTitle("TRACE — Wing Analysis Pipeline")
        self.settings = QSettings("TRACE", "WingAnalysisPipeline")
        self.resize(1050, 750)
        self.worker = None
        self._image_paths = []
        self.config = PipelineConfig()
        self._show_vein_tissue = False
        self._include_unreliable_landmarks = False
        self._workers_warning_shown = False
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

        mg.addWidget(QLabel("Landmark model (.pt or fold folder):"))
        row = QHBoxLayout()
        self.lm_edit = QLineEdit()
        self.lm_edit.setReadOnly(True)
        self.lm_edit.setPlaceholderText("Select .pt checkpoint or fold folder...")
        self.lm_edit.setToolTip(
            "Pick a single .pt checkpoint for fast single-fold inference, "
            "or pick a folder containing best_fold*.pt for 5-fold ensemble "
            "(~5× slower, more robust)."
        )
        btn_file = QPushButton("File...")
        btn_file.clicked.connect(self._select_landmark_model)
        btn_folder = QPushButton("Folder...")
        btn_folder.setToolTip("Pick a folder of best_fold*.pt checkpoints (5-fold ensemble).")
        btn_folder.clicked.connect(self._select_landmark_model_folder)
        row.addWidget(self.lm_edit, stretch=1)
        row.addWidget(btn_file)
        row.addWidget(btn_folder)
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
        self.scale_spin.setValue(self.config.um_per_px or 0.0)
        self.scale_spin.setSingleStep(0.001)
        self.scale_spin.setSpecialValueText("pixel only")
        self.scale_spin.valueChanged.connect(self._on_scale_changed)
        sg.addWidget(self.scale_spin, stretch=1)
        left_layout.addWidget(scale_group)

        # -- Pipeline settings --
        settings_group = QGroupBox("Pipeline settings")
        stg = QVBoxLayout(settings_group)
        self.btn_settings = QPushButton("Edit settings...")
        self.btn_settings.clicked.connect(self._open_settings_dialog)
        stg.addWidget(self.btn_settings)
        io_row = QHBoxLayout()
        btn_import = QPushButton("Import...")
        btn_import.clicked.connect(self._import_config)
        btn_export = QPushButton("Export...")
        btn_export.clicked.connect(self._export_config)
        io_row.addWidget(btn_import)
        io_row.addWidget(btn_export)
        stg.addLayout(io_row)
        left_layout.addWidget(settings_group)

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

        # -- Parallel workers --
        # Custom header so the "?" help button sits directly next to the section title.
        workers_header = QHBoxLayout()
        workers_title = QLabel("Parallel processing")
        workers_title.setStyleSheet("font-weight: bold;")
        workers_header.addWidget(workers_title)
        self.workers_help_btn = QToolButton()
        self.workers_help_btn.setText("?")
        self.workers_help_btn.setToolTip("Show parallel-workers warning")
        self.workers_help_btn.setAutoRaise(True)
        self.workers_help_btn.clicked.connect(self._show_workers_warning_info)
        workers_header.addWidget(self.workers_help_btn)
        workers_header.addStretch()
        left_layout.addLayout(workers_header)

        workers_row = QHBoxLayout()
        workers_row.addWidget(QLabel("Workers:"))
        self.workers_spin = QSpinBox()
        self.workers_spin.setRange(1, 32)
        self.workers_spin.setValue(DEFAULT_MAX_WORKERS)
        self.workers_spin.setToolTip(
            "Number of wings to analyze in parallel during Stage 2 AND batch size for "
            "the Stage 1 landmark forward pass. Higher = more memory + more throughput."
        )
        self.workers_spin.valueChanged.connect(self._on_workers_changed)
        workers_row.addWidget(self.workers_spin, stretch=1)
        left_layout.addLayout(workers_row)

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

    def _select_landmark_model_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Select Fold Checkpoint Folder (contains best_fold*.pt)", "")
        if folder:
            from pathlib import Path as _P

            if not sorted(_P(folder).glob("best_fold*.pt")):
                QMessageBox.warning(
                    self,
                    "No fold checkpoints",
                    f"No best_fold*.pt files in {folder}. Pick a folder containing 5-fold CV checkpoints.",
                )
                return
            self.lm_edit.setText(folder)

    def _select_seg_model(self):
        folder = QFileDialog.getExistingDirectory(self, "Select Segmentation Model Folder")
        if folder:
            self.seg_edit.setText(folder)

    # -----------------------------------------------------------------------
    # Pipeline settings (PipelineConfig)
    # -----------------------------------------------------------------------
    def _on_scale_changed(self, val: float):
        self.config.um_per_px = val if val > 0 else None

    def reset_workers_warning(self):
        """Re-arm the spinner-change parallel-workers warning so it fires again on the next bump above 1.

        Skipped if the user has permanently suppressed the warning via the run-time dialog.
        """
        if self.settings.value("workers_warning_suppressed", False, type=bool):
            return
        self._workers_warning_shown = False

    def _on_workers_changed(self, val: int):
        if val <= DEFAULT_MAX_WORKERS or self._workers_warning_shown:
            return
        if self.settings.value("workers_warning_suppressed", False, type=bool):
            return
        self._workers_warning_shown = True
        QMessageBox.warning(self, "Parallel workers", self._PARALLEL_WORKERS_WARNING_TEXT)

    def _show_workers_warning_info(self):
        """Open the parallel-workers help dialog: details text + Calibrate panel."""
        from TRACE.calibrate_widget import CalibrateWidget

        dlg = QDialog(self)
        dlg.setWindowTitle("Parallel workers")
        layout = QVBoxLayout(dlg)
        msg = QLabel(self._PARALLEL_WORKERS_DETAILS_TEXT)
        msg.setWordWrap(True)
        layout.addWidget(msg)

        calib = CalibrateWidget(dlg)
        calib.set_paths(self.input_edit.text(), self.lm_edit.text(), self.seg_edit.text())
        calib.applied.connect(lambda v: self.workers_spin.setValue(int(v)))
        layout.addWidget(calib)

        close_row = QHBoxLayout()
        close_row.addStretch()
        close_btn = QPushButton("Close")
        close_btn.clicked.connect(dlg.accept)
        close_row.addWidget(close_btn)
        layout.addLayout(close_row)

        dlg.exec_()

    def _confirm_parallel_workers(self) -> bool:
        """Show the parallel-workers warning before a multi-worker run.

        Fires every time the pipeline is started with workers > 1, unless the
        user has previously checked "do not warn me again". Returns True if the
        run should proceed, False if the user cancelled.
        """
        if self.workers_spin.value() <= DEFAULT_MAX_WORKERS:
            return True
        if self.settings.value("workers_warning_suppressed", False, type=bool):
            return True

        box = QMessageBox(self)
        box.setIcon(QMessageBox.Warning)
        box.setWindowTitle("Parallel workers")
        box.setText(self._PARALLEL_WORKERS_WARNING_TEXT)
        box.setStandardButtons(QMessageBox.Ok | QMessageBox.Cancel)
        box.setDefaultButton(QMessageBox.Ok)
        suppress_chk = QCheckBox("Do not warn me again")
        box.setCheckBox(suppress_chk)
        result = box.exec_()
        if result != QMessageBox.Ok:
            return False
        if suppress_chk.isChecked():
            self.settings.setValue("workers_warning_suppressed", True)
        return True

    def _open_settings_dialog(self):
        dlg = PipelineConfigDialog(
            self.config,
            self,
            show_vein_tissue=self._show_vein_tissue,
            include_unreliable_landmarks=self._include_unreliable_landmarks,
            workers=self.workers_spin.value(),
            input_path=self.input_edit.text(),
            landmark_model_path=self.lm_edit.text(),
            segmentation_model_path=self.seg_edit.text(),
        )
        if dlg.exec_() == QDialog.Accepted:
            self.config = dlg.get_config()
            self._show_vein_tissue = dlg.get_show_vein_tissue()
            self._include_unreliable_landmarks = dlg.get_include_unreliable_landmarks()
            # Keep main-window scale spinner in sync.
            val = self.config.um_per_px if self.config.um_per_px is not None else 0.0
            self.scale_spin.blockSignals(True)
            self.scale_spin.setValue(val)
            self.scale_spin.blockSignals(False)
            # Keep main-window workers spinner in sync (block signals to avoid
            # re-firing the spinner-change warning).
            self.workers_spin.blockSignals(True)
            self.workers_spin.setValue(dlg.get_workers())
            self.workers_spin.blockSignals(False)

    def _import_config(self):
        path, _ = QFileDialog.getOpenFileName(self, "Import Pipeline Config", "", "JSON (*.json);;All Files (*)")
        if not path:
            return
        try:
            self.config = load_config(Path(path))
        except Exception as e:
            QMessageBox.critical(self, "Import failed", f"Could not load config:\n{e}")
            return
        val = self.config.um_per_px if self.config.um_per_px is not None else 0.0
        self.scale_spin.blockSignals(True)
        self.scale_spin.setValue(val)
        self.scale_spin.blockSignals(False)
        self._log(f"Imported config from {path}")

    def _export_config(self):
        path, _ = QFileDialog.getSaveFileName(
            self, "Export Pipeline Config", "pipeline_config.json", "JSON (*.json);;All Files (*)"
        )
        if not path:
            return
        try:
            save_config(self.config, Path(path))
        except Exception as e:
            QMessageBox.critical(self, "Export failed", f"Could not save config:\n{e}")
            return
        self._log(f"Exported config to {path}")

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
        s.setValue("pipeline_config_json", config_to_json(self.config))
        s.setValue("max_workers", self.workers_spin.value())
        s.setValue("show_vein_tissue", self._show_vein_tissue)
        s.setValue("include_unreliable_landmarks", self._include_unreliable_landmarks)
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
        cfg_json = s.value("pipeline_config_json", None)
        if cfg_json:
            try:
                self.config = config_from_json(cfg_json)
            except Exception:
                self.config = PipelineConfig()
        scale_val = self.config.um_per_px if self.config.um_per_px is not None else 0.0
        self.scale_spin.blockSignals(True)
        self.scale_spin.setValue(scale_val)
        self.scale_spin.blockSignals(False)
        for key, chk in self.output_checks.items():
            saved = s.value(f"output/{key}", None)
            if saved is not None:
                chk.setChecked(saved == "true" or saved is True)
        saved_svt = s.value("show_vein_tissue", None)
        if saved_svt is not None:
            self._show_vein_tissue = saved_svt == "true" or saved_svt is True
        saved_iul = s.value("include_unreliable_landmarks", None)
        if saved_iul is not None:
            self._include_unreliable_landmarks = saved_iul == "true" or saved_iul is True
        workers_val = s.value("max_workers", None)
        if workers_val is not None:
            try:
                self.workers_spin.blockSignals(True)
                self.workers_spin.setValue(int(workers_val))
            except (TypeError, ValueError):
                pass
            finally:
                self.workers_spin.blockSignals(False)

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

        # Warn on JPEG inputs (lossy compression)
        jpg_images = [p for p in self._image_paths if p.suffix.lower() in (".jpg", ".jpeg")]
        if jpg_images:
            box = QMessageBox(self)
            box.setIcon(QMessageBox.Warning)
            box.setWindowTitle("JPEG Input Detected")
            box.setTextFormat(Qt.RichText)
            box.setTextInteractionFlags(Qt.TextBrowserInteraction)
            box.setText(
                "CAUTION: You are using .jpg image(s). Due to the limited amount of information "
                "provided by this file type, TRACE may not function as intended."
                "<br><br>Read about why: "
                '<a href="https://pmc.ncbi.nlm.nih.gov/articles/PMC4210356/">'
                "https://pmc.ncbi.nlm.nih.gov/articles/PMC4210356/</a>"
            )
            box.setStandardButtons(QMessageBox.Ok | QMessageBox.Abort)
            box.setDefaultButton(QMessageBox.Abort)
            if box.exec_() == QMessageBox.Abort:
                return

        if not self._confirm_parallel_workers():
            return

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
                config=self.config,
                keep_intermediates=False,
                outputs=self._selected_outputs(),
                max_workers=self.workers_spin.value(),
                show_vein_tissue=self._show_vein_tissue,
                include_unreliable_landmarks=self._include_unreliable_landmarks,
                # Tie landmark batch size to the Workers spinbox so a single setting
                # controls Stage 1 batching and Stage 2 parallelism together.
                landmark_batch_size=self.workers_spin.value(),
            )
        )
        self.worker.progress.connect(self._on_progress)
        self.worker.log_message.connect(self._log)
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
        self.log_text.append(f"[{time.strftime('%H:%M:%S')}] {msg}")
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
