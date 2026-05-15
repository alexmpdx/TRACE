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
from typing import Optional

from identify_features.config import PipelineConfig
from PyQt5.QtCore import QEvent, QObject, QSettings, Qt, QThread, QTimer, pyqtSignal
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
from TRACE.config_io import config_from_json, config_to_json, load_settings, save_settings
from TRACE.pipeline import (
    DEFAULT_MAX_WORKERS,
    INTERMEDIATE_OUTPUTS,
    MEASUREMENT_GROUPS,
    OUTPUT_TOOLTIPS,
    OUTPUT_TYPES,
    compute_progress_weights,
    trace_folder,
)
from TRACE.settings_dialog import PipelineConfigDialog

# ---------------------------------------------------------------------------
# Worker thread
# ---------------------------------------------------------------------------


_CAPTURED_LOGGERS = ("identify_features", "TRACE", "preprocessing")


class _PlaceholderSpinBox(QDoubleSpinBox):
    """QDoubleSpinBox that shows a real QLineEdit placeholder (auto-dimmed, clears on
    focus) instead of `setSpecialValueText`. When the user hasn't entered a value
    (spinbox is at its minimum), `textFromValue` returns an empty string so
    QLineEdit's normal placeholder rendering kicks in.

    Configure by calling `set_placeholder("...")` on the instance.
    """

    def set_placeholder(self, text: str) -> None:
        self.lineEdit().setPlaceholderText(text)
        # Force the displayed text to refresh — without this the previous numeric
        # rendering can linger until the next value change.
        self.lineEdit().setText(self.textFromValue(self.value()))

    def textFromValue(self, value: float) -> str:  # noqa: N802 — Qt API
        if value == self.minimum() and self.lineEdit().placeholderText():
            return ""
        return super().textFromValue(value)


# Bundled default-model folders. Used to preload the three model paths on a
# first-time launch (no saved QSettings) and after "wipe my memories" so the
# user doesn't have to browse for them. Each entry is checked for existence
# at runtime — missing folders fall back to "" (empty), forcing the user to
# pick one in Settings → Models.
_DEFAULT_MODELS_DIR = Path(__file__).resolve().parent / "models"
_DEFAULT_MODEL_PATHS = {
    "landmark": _DEFAULT_MODELS_DIR / "landmarks",
    "segmentation": _DEFAULT_MODELS_DIR / "vein-intervein",
    "wing_isolation": _DEFAULT_MODELS_DIR / "wingIsolation",
}


def _default_model_path(key: str) -> str:
    """Return the bundled default model folder for `key` as a string, or "" if missing.

    Keys: "landmark", "segmentation", "wing_isolation".
    """
    path = _DEFAULT_MODEL_PATHS.get(key)
    if path is None:
        return ""
    return str(path) if path.is_dir() else ""


def _migrate_legacy_model_path(saved: str) -> str:
    """Auto-fall-back to a flat-layout model dir when a saved path is stale.

    LandmarkLocator used to ship checkpoints inside a `<name>_checkpoints/` (or
    `checkpoints/`) sub-folder. The flat layout puts `best_fold*.pt`,
    `gate_config.yaml`, `training_chart.png`, and `training.log` all directly in
    the model folder. If a user's saved path still points at the (now-deleted)
    nested sub-folder, transparently substitute the parent when the parent is a
    valid flat-layout model dir. Otherwise return the saved path unchanged so
    the user gets an obvious "missing path" signal in Settings.
    """
    if not saved:
        return saved
    p = Path(saved)
    if p.exists():
        return saved
    parent = p.parent
    if parent.exists() and parent.is_dir() and any(parent.glob("best_fold*.pt")):
        return str(parent)
    return saved


def _picker_initial_path(current: str) -> str:
    """Sane initial path for QFileDialog: walk up `current` to the first existing
    file/dir, or fall back to '/' (Finder's 'Computer' view) when nothing in the
    saved path exists. Empty input always returns '/'.
    """
    if current:
        p = Path(current)
        if p.exists():
            return str(p)
        while p != p.parent:
            p = p.parent
            if p.exists():
                return str(p)
    return "/"


class _SignalLogHandler(logging.Handler):
    """Logging handler that forwards records through a pyqtSignal."""

    def __init__(self, signal):
        super().__init__()
        self._signal = signal
        self.setFormatter(logging.Formatter("%(name)s %(levelname)s: %(message)s"))

    def emit(self, record):
        try:
            from preprocessing.pipeline import current_image

            text = self.format(record)
            img = current_image.get()
            if img:
                text = f"[{img}] {text}"
            self._signal.emit(text)
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
        "Each worker runs the full pipeline on one wing "
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
        self._do_rotation = False
        self._rotation_mirror_correct = False
        self._gate_override: dict | None = None
        self._wing_expand_fraction = 0.05
        self._wing_isolation_enabled = False
        self._wing_isolation_model_path = _default_model_path("wing_isolation")
        # Model paths (configured via Settings → Models). Plain strings rather
        # than QLineEdit widgets — the dialog owns the editing UI. Initialized
        # to the bundled TRACE/models/* folders so a first-time launch has
        # working defaults; overridden by saved QSettings values when present.
        self._landmark_model_path = _default_model_path("landmark")
        self._segmentation_model_path = _default_model_path("segmentation")
        # Stage 1 (resolutionAdjust) — per-model training-µm/px targets, the
        # active-model selector (default: wing features = segmentation), and the
        # ratio tolerance band. None means "not configured" for any target.
        # Wing features (segmentation) defaults to 0.483 µm/px — the resolution
        # the bundled segmentation model was trained at.
        self._landmark_target_um_per_px: Optional[float] = None
        self._segmentation_target_um_per_px: Optional[float] = 0.483
        self._wing_isolation_target_um_per_px: Optional[float] = None
        self._active_rescale_target: str = "segmentation"
        self._rescale_tolerance_low: float = 0.85
        self._rescale_tolerance_high: float = 1.15
        # Intermediate outputs (toggled in Settings → General → Intermediate outputs).
        # Default-off so a fresh batch only writes the final overlays + CSV; users
        # opt in to intermediates per-key in the Settings dialog.
        self._intermediate_outputs: dict[str, bool] = {key: False for key in INTERMEDIATE_OUTPUTS}
        # User-defined landmark distance pairs (TRACE-only post-CSV augmentation).
        # Configured via Settings → Custom Distances.
        self._user_landmark_distances: list[dict] = []
        # Last-used sample image + landmarks GeoJSON for the picker, so users
        # don't have to re-browse every session.
        self._distance_sample_image: str = ""
        self._distance_sample_landmarks: str = ""
        self._workers_warning_shown = False
        # Highest percentage shown on the progress bar this run — parallel
        # workers emit events out of order so we hold the bar at its peak
        # (never rewinds). Reset to 0 at the start of every run.
        self._progress_pct_high = 0
        # Stage1/Stage2 wall-time weight split on the unified progress bar.
        # Recomputed from the current output selection at run start so the
        # bar reflects "fraction of wall time done" rather than raw image count.
        self._progress_stage1_share = 1.0
        self._progress_stage2_share = 0.0
        # Wall-clock + smoothed ETA tracking.
        self._run_start_time: Optional[float] = None
        self._eta_smoothed_seconds: Optional[float] = None
        # Per-stage throughput tracking for the hybrid ETA. Reset on every
        # stage transition (preprocessing → analysis).
        self._current_stage: Optional[str] = None
        self._stage_first_event_time: Optional[float] = None
        self._stage_completions: int = 0
        self._stage_total: int = 0
        # Locked-in (not recomputed every tick) timestamps + average so the
        # smoothed bar can interpolate between "done" events without
        # collapsing to zero on every refresh.
        self._last_completion_time: Optional[float] = None
        self._avg_time_per_image: Optional[float] = None
        # 250ms QTimer that refreshes the bar + ETA between completion events.
        # Without this, parallel-worker batches arrive in clusters and the
        # bar sits frozen for tens of seconds between bursts of "done" events.
        self._progress_timer = QTimer(self)
        self._progress_timer.setInterval(250)
        self._progress_timer.timeout.connect(self._refresh_progress)
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
        folder_group = QGroupBox("Input and Output Folders")
        fg = QVBoxLayout(folder_group)

        fg.addWidget(QLabel("Images to analyze:"))
        row = QHBoxLayout()
        self.input_edit = QLineEdit()
        self.input_edit.setReadOnly(True)
        self.input_edit.setPlaceholderText("select folder")
        self.input_edit.setToolTip("Folder containing wing images to process. Click Browse... to select.")
        btn = QPushButton("Browse...")
        btn.setToolTip("Pick the folder containing wing images.")
        btn.clicked.connect(self._select_input)
        row.addWidget(self.input_edit, stretch=1)
        row.addWidget(btn)
        fg.addLayout(row)

        self.recursive_chk = QCheckBox("Include subfolders")
        self.recursive_chk.setToolTip("When checked, also discover images inside subdirectories of the input folder.")
        self.recursive_chk.toggled.connect(self._refresh_image_list)
        fg.addWidget(self.recursive_chk)

        fg.addWidget(QLabel("Save results to:"))
        row = QHBoxLayout()
        self.output_edit = QLineEdit()
        self.output_edit.setReadOnly(True)
        self.output_edit.setPlaceholderText("select folder")
        self.output_edit.setToolTip(
            "Folder where all outputs (overlays, CSV, GeoJSONs, intermediates) will be written."
        )
        btn = QPushButton("Browse...")
        btn.setToolTip("Pick the folder where outputs will be written.")
        btn.clicked.connect(self._select_output)
        row.addWidget(self.output_edit, stretch=1)
        row.addWidget(btn)
        fg.addLayout(row)

        left_layout.addWidget(folder_group)

        # Model paths live as plain instance state — configured in
        # Settings → Models (no widgets in the main window).

        # -- Scale --
        scale_group = QGroupBox("Scale")
        sg = QHBoxLayout(scale_group)
        sg.addWidget(QLabel("\u00b5m/px:"))
        self.scale_spin = _PlaceholderSpinBox()
        self.scale_spin.setDecimals(4)
        self.scale_spin.setRange(0.0001, 100.0)
        self.scale_spin.setSingleStep(0.001)
        # QLineEdit-native placeholder (auto-dimmed, clears on click) — shown when
        # the spinbox is at its minimum. Treated as "no scale entered" — Run
        # Pipeline refuses to start.
        self.scale_spin.setValue(self.config.um_per_px if self.config.um_per_px else self.scale_spin.minimum())
        self.scale_spin.set_placeholder("conversion factor")
        self.scale_spin.setToolTip("Microns per pixel — used to convert every measurement to physical units (µm, µm²).")
        self.scale_spin.valueChanged.connect(self._on_scale_changed)
        sg.addWidget(self.scale_spin, stretch=1)
        left_layout.addWidget(scale_group)

        # -- Pipeline settings --
        settings_group = QGroupBox("Pipeline settings")
        stg = QVBoxLayout(settings_group)
        self.btn_settings = QPushButton("Edit settings...")
        self.btn_settings.setToolTip(
            "Open the full settings dialog: tabs for General, Custom Distances, "
            "Landmarks, Models, Skeletonization & Pruning, Bridging, Tracing, Intervein."
        )
        self.btn_settings.clicked.connect(self._open_settings_dialog)
        stg.addWidget(self.btn_settings)
        io_row = QHBoxLayout()
        btn_import = QPushButton("Import...")
        btn_import.setToolTip("Load a previously-exported pipeline-config JSON file. Replaces the current settings.")
        btn_import.clicked.connect(self._import_config)
        btn_export = QPushButton("Save...")
        btn_export.setToolTip("Save the current pipeline-config to a JSON file for reuse (CLI --config or GUI Import).")
        btn_export.clicked.connect(self._export_config)
        io_row.addWidget(btn_import)
        io_row.addWidget(btn_export)
        stg.addLayout(io_row)
        self.btn_reset_gui = QPushButton("wipe my memories")
        self.btn_reset_gui.setToolTip(
            "Clear every persisted setting — input/output folders, model paths, scale, "
            "pipeline config, custom distance pairs, workers warning suppression — and "
            "snap every widget back to the state a first-time user would see."
        )
        self.btn_reset_gui.clicked.connect(self._reset_gui_to_defaults)
        stg.addWidget(self.btn_reset_gui)
        # settings_group is appended to left_layout at the bottom of this method
        # so it sits below the Outputs and Parallel processing sections.

        # -- Output selection (final outputs only; intermediates live in Settings → General) --
        out_group = QGroupBox("Outputs")
        ol = QVBoxLayout(out_group)
        self.output_checks: OrderedDict[str, QCheckBox] = OrderedDict()
        # Nested measurement-group checkboxes for the CSV output.
        self.csv_group_checks: OrderedDict[str, QCheckBox] = OrderedDict()
        self._csv_group_container: QWidget | None = None
        for key, label in OUTPUT_TYPES.items():
            if key in INTERMEDIATE_OUTPUTS:
                continue
            chk = QCheckBox(label)
            chk.setChecked(True)
            tip = OUTPUT_TOOLTIPS.get(key)
            if tip:
                chk.setToolTip(tip)
            self.output_checks[key] = chk
            ol.addWidget(chk)
            # Nest the measurement-group checkboxes directly under "csv".
            if key == "csv":
                self._csv_group_container = QWidget()
                cgl = QVBoxLayout(self._csv_group_container)
                cgl.setContentsMargins(24, 0, 0, 0)  # left-indent for hierarchy
                cgl.setSpacing(2)
                for gkey, glabel in MEASUREMENT_GROUPS.items():
                    gchk = QCheckBox(glabel)
                    gchk.setChecked(True)
                    self.csv_group_checks[gkey] = gchk
                    cgl.addWidget(gchk)
                # Custom landmark distances also writes only into the batch CSV,
                # so it sits alongside the measurement-group sub-checkboxes and
                # tracks the parent CSV checkbox's enabled state.
                cd_row = QHBoxLayout()
                cd_row.setContentsMargins(0, 0, 0, 0)
                self.include_custom_measurements_chk = QCheckBox("Custom distances")
                self.include_custom_measurements_chk.setChecked(True)
                self.include_custom_measurements_chk.setToolTip(
                    "Adds the pairs configured in Settings → Custom Distances to the batch CSV "
                    "as custom_<label>_px (and _um when scale is set) columns.\n\n"
                    "No effect when no pairs are configured."
                )
                cd_row.addWidget(self.include_custom_measurements_chk)
                self.btn_edit_custom_distances = QPushButton("Edit...")
                self.btn_edit_custom_distances.setToolTip(
                    "Open the Settings dialog at the Custom Distances tab to add/edit/remove "
                    "landmark distance pairs."
                )
                self.btn_edit_custom_distances.clicked.connect(
                    lambda: self._open_settings_dialog(initial_tab="Custom Distances")
                )
                cd_row.addWidget(self.btn_edit_custom_distances)
                cd_row.addStretch()
                cgl.addLayout(cd_row)
                ol.addWidget(self._csv_group_container)
                # Enable/disable nested checkboxes with parent CSV state.
                chk.toggled.connect(self._csv_group_container.setEnabled)
                self._csv_group_container.setEnabled(chk.isChecked())

        left_layout.addWidget(out_group)

        # -- Parallel processing --
        workers_group = QGroupBox("Parallel processing")
        wg = QVBoxLayout(workers_group)
        workers_row = QHBoxLayout()
        workers_row.addWidget(QLabel("Workers:"))
        self.workers_spin = QSpinBox()
        self.workers_spin.setRange(1, 32)
        self.workers_spin.setValue(DEFAULT_MAX_WORKERS)
        self.workers_spin.setToolTip(
            "Number of wings to process in parallel through every stage of the pipeline:\n"
            "  • Stage 1 — hinge chop and segmentation (per image)\n"
            "  • Stage 2 — identifyFeatures analysis (per image)\n"
            "Also sets the GPU batch size for the upfront landmark forward pass.\n"
            "Higher = more memory + more throughput."
        )
        self.workers_spin.valueChanged.connect(self._on_workers_changed)
        workers_row.addWidget(self.workers_spin, stretch=1)
        self.workers_help_btn = QToolButton()
        self.workers_help_btn.setText("?")
        self.workers_help_btn.setToolTip("Show parallel-workers warning")
        self.workers_help_btn.setAutoRaise(True)
        self.workers_help_btn.clicked.connect(self._show_workers_warning_info)
        workers_row.addWidget(self.workers_help_btn)
        wg.addLayout(workers_row)
        left_layout.addWidget(workers_group)

        # Pipeline settings sits at the bottom so the section the user touches
        # most often (Outputs + Parallel processing) is closer to eye-level.
        left_layout.addWidget(settings_group)

        # -- Run / Cancel --
        btn_layout = QHBoxLayout()
        self.btn_run = QPushButton("Run Pipeline")
        self.btn_run.setToolTip("Start processing every image in the input folder. Opens the output folder when done.")
        self.btn_run.clicked.connect(self._run_pipeline)
        self.btn_cancel = QPushButton("Cancel")
        self.btn_cancel.setToolTip("Stop the running batch at the next safe point. Partial results are kept.")
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
        self.progress.setRange(0, 100)
        # Snapshot the default Highlight color so we can revert after a green
        # completion paint without forcing the bar into stylesheet-rendering
        # mode (which subtly shifts indentation, border, and text position).
        self._progress_default_highlight = self.progress.palette().color(QPalette.Highlight)
        right_layout.addWidget(self.progress)
        self.eta_label = QLabel("")
        self.eta_label.setStyleSheet("color: #888;")
        right_layout.addWidget(self.eta_label)

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
        folder = QFileDialog.getExistingDirectory(
            self, "Select Input Folder", _picker_initial_path(self.input_edit.text())
        )
        if not folder:
            return
        self.input_edit.setText(folder)
        self._refresh_image_list()

    def _refresh_image_list(self):
        """Re-discover images using the current input-folder + recursive flag."""
        folder_text = self.input_edit.text()
        if not folder_text:
            self._image_paths = []
            self.image_list.clear()
            return
        folder = Path(folder_text)
        if not folder.is_dir():
            return
        recursive = self.recursive_chk.isChecked()
        self._image_paths = discover_images(folder, recursive=recursive)
        self.image_list.clear()
        for p in self._image_paths:
            # Show path relative to the input folder when recursing so subfolder
            # context is visible; otherwise just the name.
            label = str(p.relative_to(folder)) if recursive else p.name
            self.image_list.addItem(label)
        self.statusBar().showMessage(f"Found {len(self._image_paths)} images")

    def _select_output(self):
        folder = QFileDialog.getExistingDirectory(
            self, "Select Output Folder", _picker_initial_path(self.output_edit.text())
        )
        if folder:
            self.output_edit.setText(folder)

    # -----------------------------------------------------------------------
    # Pipeline settings (PipelineConfig)
    # -----------------------------------------------------------------------
    def _on_scale_changed(self, val: float):
        # The spinbox minimum doubles as the "[conversion factor]" placeholder
        # state — treat it as no scale entered.
        self.config.um_per_px = val if val > self.scale_spin.minimum() else None

    def reset_workers_warning(self):
        """Re-arm the spinner-change parallel-workers warning so it fires again on the next bump above 1.

        Skipped if the user has permanently suppressed the warning via the run-time dialog.
        """
        if self.settings.value("workers_warning_suppressed", False, type=bool):
            return
        self._workers_warning_shown = False

    def _on_workers_changed(self, val: int):
        self.maybe_show_workers_warning(val)

    def maybe_show_workers_warning(self, val: int) -> None:
        """Show the parallel-workers warning dialog once if `val` exceeds the
        default. Public entry point shared by the main-window Workers spinbox
        and the Settings → General → Parallel processing spinbox.

        No-op when the warning has already fired this session or the user has
        permanently suppressed it via the dialog's 'Don't show again' checkbox.
        """
        if val <= DEFAULT_MAX_WORKERS or self._workers_warning_shown:
            return
        if self.settings.value("workers_warning_suppressed", False, type=bool):
            return
        self._workers_warning_shown = True
        QMessageBox.warning(self, "Parallel workers", self._PARALLEL_WORKERS_WARNING_TEXT)

    def _show_workers_warning_info(self):
        """Open the parallel-workers help dialog: details text + Calibrate panel.

        Styled to match the QMessageBox warning that fires before a multi-worker
        run — bold message text, constrained width, right-aligned button row.
        """
        from PyQt5.QtGui import QFont

        from TRACE.calibrate_widget import CalibrateWidget

        dlg = QDialog(self)
        dlg.setWindowTitle("Parallel workers")
        dlg.setMinimumWidth(520)
        dlg.setMaximumWidth(560)
        layout = QVBoxLayout(dlg)
        layout.setContentsMargins(20, 20, 20, 16)
        layout.setSpacing(12)

        msg = QLabel(self._PARALLEL_WORKERS_DETAILS_TEXT)
        msg.setWordWrap(True)
        msg_font = QFont(msg.font())
        msg_font.setBold(True)
        msg.setFont(msg_font)
        layout.addWidget(msg)

        calib = CalibrateWidget(dlg)
        calib.set_paths(self.input_edit.text(), self._landmark_model_path, self._segmentation_model_path)
        calib.applied.connect(lambda v: self.workers_spin.setValue(int(v)))
        layout.addWidget(calib)

        close_row = QHBoxLayout()
        close_row.addStretch()
        close_btn = QPushButton("Close")
        close_btn.setDefault(True)
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

    def _open_settings_dialog(self, initial_tab: str | None = None):
        dlg = PipelineConfigDialog(
            self.config,
            self,
            show_vein_tissue=self._show_vein_tissue,
            include_unreliable_landmarks=self._include_unreliable_landmarks,
            workers=self.workers_spin.value(),
            input_path=self.input_edit.text(),
            landmark_model_path=self._landmark_model_path,
            segmentation_model_path=self._segmentation_model_path,
            gate_override=self._gate_override,
            wing_expand_fraction=self._wing_expand_fraction,
            wing_isolation_enabled=self._wing_isolation_enabled,
            wing_isolation_model_path=self._wing_isolation_model_path,
            intermediate_outputs=dict(self._intermediate_outputs),
            do_rotation=self._do_rotation,
            rotation_mirror_correct=self._rotation_mirror_correct,
            user_landmark_distances=list(self._user_landmark_distances),
            distance_sample_image=self._distance_sample_image,
            distance_sample_landmarks=self._distance_sample_landmarks,
            landmark_target_um_per_px=self._landmark_target_um_per_px,
            segmentation_target_um_per_px=self._segmentation_target_um_per_px,
            wing_isolation_target_um_per_px=self._wing_isolation_target_um_per_px,
            active_rescale_target=self._active_rescale_target,
            rescale_tolerance_low=self._rescale_tolerance_low,
            rescale_tolerance_high=self._rescale_tolerance_high,
        )
        if initial_tab:
            dlg.select_tab(initial_tab)
        if dlg.exec_() == QDialog.Accepted:
            self.config = dlg.get_config()
            self._show_vein_tissue = dlg.get_show_vein_tissue()
            self._include_unreliable_landmarks = dlg.get_include_unreliable_landmarks()
            self._do_rotation = dlg.get_do_rotation()
            self._rotation_mirror_correct = dlg.get_rotation_mirror_correct()
            self._gate_override = dlg.get_gate_override()
            self._wing_expand_fraction = dlg.get_wing_expand_fraction()
            self._wing_isolation_enabled = dlg.get_wing_isolation_enabled()
            self._wing_isolation_model_path = dlg.get_wing_isolation_model_path()
            self._landmark_model_path = dlg.get_landmark_model_path()
            self._segmentation_model_path = dlg.get_segmentation_model_path()
            self._landmark_target_um_per_px = dlg.get_landmark_target_um_per_px()
            self._segmentation_target_um_per_px = dlg.get_segmentation_target_um_per_px()
            self._wing_isolation_target_um_per_px = dlg.get_wing_isolation_target_um_per_px()
            self._active_rescale_target = dlg.get_active_rescale_target()
            self._rescale_tolerance_low = dlg.get_rescale_tolerance_low()
            self._rescale_tolerance_high = dlg.get_rescale_tolerance_high()
            self._intermediate_outputs = dlg.get_intermediate_outputs()
            self._user_landmark_distances = dlg.get_user_landmark_distances()
            self._distance_sample_image = dlg.get_distance_sample_image()
            self._distance_sample_landmarks = dlg.get_distance_sample_landmarks()
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
            self.config, gate_override = load_settings(Path(path))
        except Exception as e:
            QMessageBox.critical(self, "Import failed", f"Could not load config:\n{e}")
            return
        # gate_override is None when the file predates this field — leave the
        # current in-session override untouched in that case.
        if gate_override is not None:
            self._gate_override = gate_override
        val = self.config.um_per_px if self.config.um_per_px else self.scale_spin.minimum()
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
            save_settings(self.config, self._gate_override, Path(path))
        except Exception as e:
            QMessageBox.critical(self, "Export failed", f"Could not save config:\n{e}")
            return
        self._log(f"Exported config to {path}")

    def _reset_gui_to_defaults(self):
        """Wipe all persisted QSettings and snap every widget back to factory defaults.

        Equivalent to deleting the QSettings file and relaunching the GUI — useful
        when a user wants the first-time experience back without restarting.
        """
        reply = QMessageBox.warning(
            self,
            "Reset GUI to defaults",
            "This will clear every saved setting — input/output folders, model paths, "
            "scale, pipeline configuration, custom distance pairs, and any 'don't show "
            "again' warning suppressions — and snap every widget back to its first-launch "
            "state.\n\nContinue?",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )
        if reply != QMessageBox.Yes:
            return

        # Wipe persisted settings.
        self.settings.clear()
        self.settings.sync()

        # Reset instance state to __init__ defaults.
        self.config = PipelineConfig()
        self._show_vein_tissue = False
        self._include_unreliable_landmarks = False
        self._do_rotation = False
        self._rotation_mirror_correct = False
        self._gate_override = None
        self._wing_expand_fraction = 0.05
        self._wing_isolation_enabled = False
        # Wipe-my-memories re-applies the bundled defaults (TRACE/models/*)
        # so a fresh user gets working model paths without having to browse.
        # The Settings dialog's Restore Defaults button does NOT touch these.
        self._wing_isolation_model_path = _default_model_path("wing_isolation")
        self._landmark_model_path = _default_model_path("landmark")
        self._segmentation_model_path = _default_model_path("segmentation")
        self._landmark_target_um_per_px = None
        self._segmentation_target_um_per_px = 0.483
        self._wing_isolation_target_um_per_px = None
        self._active_rescale_target = "segmentation"
        self._rescale_tolerance_low = 0.85
        self._rescale_tolerance_high = 1.15
        self._intermediate_outputs = {key: False for key in INTERMEDIATE_OUTPUTS}
        self._user_landmark_distances = []
        self._distance_sample_image = ""
        self._distance_sample_landmarks = ""
        self._workers_warning_shown = False
        self._image_paths = []

        # Snap every widget back. blockSignals where the slot would re-fire a
        # warning or write back to a no-longer-stale value.
        self.input_edit.clear()
        self.output_edit.clear()
        self.recursive_chk.blockSignals(True)
        self.recursive_chk.setChecked(False)
        self.recursive_chk.blockSignals(False)
        self.image_list.clear()

        self.scale_spin.blockSignals(True)
        self.scale_spin.setValue(self.scale_spin.minimum())
        self.scale_spin.blockSignals(False)

        self.workers_spin.blockSignals(True)
        self.workers_spin.setValue(DEFAULT_MAX_WORKERS)
        self.workers_spin.blockSignals(False)

        for chk in self.output_checks.values():
            chk.setChecked(True)
        for gchk in self.csv_group_checks.values():
            gchk.setChecked(True)
        self.include_custom_measurements_chk.setChecked(True)

        self.log_text.clear()
        self.statusBar().showMessage("GUI reset to defaults")

    # -----------------------------------------------------------------------
    # Settings persistence
    # -----------------------------------------------------------------------
    def _selected_outputs(self) -> set[str]:
        finals = {key for key, chk in self.output_checks.items() if chk.isChecked()}
        intermediates = {key for key, on in self._intermediate_outputs.items() if on}
        return finals | intermediates

    def _save_settings(self):
        s = self.settings
        s.setValue("input_folder", self.input_edit.text())
        s.setValue("input_recursive", self.recursive_chk.isChecked())
        s.setValue("output_folder", self.output_edit.text())
        s.setValue("landmark_model", self._landmark_model_path)
        s.setValue("segmentation_model", self._segmentation_model_path)
        s.setValue("wing_isolation_enabled", self._wing_isolation_enabled)
        s.setValue("wing_isolation_model", self._wing_isolation_model_path)
        s.setValue("wing_expand_fraction", self._wing_expand_fraction)
        s.setValue(
            "models/landmark_target_um_per_px",
            "" if self._landmark_target_um_per_px is None else str(self._landmark_target_um_per_px),
        )
        s.setValue(
            "models/segmentation_target_um_per_px",
            "" if self._segmentation_target_um_per_px is None else str(self._segmentation_target_um_per_px),
        )
        s.setValue(
            "models/wing_isolation_target_um_per_px",
            "" if self._wing_isolation_target_um_per_px is None else str(self._wing_isolation_target_um_per_px),
        )
        s.setValue("models/active_rescale_target", self._active_rescale_target)
        s.setValue("resolution/tolerance_low", str(self._rescale_tolerance_low))
        s.setValue("resolution/tolerance_high", str(self._rescale_tolerance_high))
        s.setValue("pipeline_config_json", config_to_json(self.config))
        s.setValue("max_workers", self.workers_spin.value())
        s.setValue("show_vein_tissue", self._show_vein_tissue)
        s.setValue("include_unreliable_landmarks", self._include_unreliable_landmarks)
        s.setValue("do_rotation", self._do_rotation)
        s.setValue("rotation_mirror_correct", self._rotation_mirror_correct)
        import json as _json

        s.setValue(
            "gate_override_json",
            _json.dumps(self._gate_override) if self._gate_override else "",
        )
        s.setValue(
            "user_landmark_distances_json",
            _json.dumps(self._user_landmark_distances) if self._user_landmark_distances else "",
        )
        s.setValue("include_custom_measurements", self.include_custom_measurements_chk.isChecked())
        s.setValue("distance_sample_image", self._distance_sample_image)
        s.setValue("distance_sample_landmarks", self._distance_sample_landmarks)
        for key, chk in self.output_checks.items():
            s.setValue(f"output/{key}", chk.isChecked())
        for key, on in self._intermediate_outputs.items():
            s.setValue(f"output/{key}", on)
        for gkey, gchk in self.csv_group_checks.items():
            s.setValue(f"csv_group/{gkey}", gchk.isChecked())

    def _restore_settings(self):
        s = self.settings
        saved_recursive = s.value("input_recursive", None)
        if saved_recursive is not None:
            self.recursive_chk.setChecked(saved_recursive == "true" or saved_recursive is True)
        val = s.value("input_folder", "")
        if val:
            self.input_edit.setText(val)
            self._refresh_image_list()
        val = s.value("output_folder", "")
        if val:
            self.output_edit.setText(val)
        # Fall back to bundled defaults (TRACE/models/*) when no saved value
        # exists — first-time launch preloads the model paths so the user
        # only has to point them elsewhere if they want a different model.
        # `_migrate_legacy_model_path` rescues saved paths that pointed into a
        # now-deleted nested checkpoints folder by substituting the parent dir
        # when it's a valid flat-layout model.
        val = s.value("landmark_model", "")
        self._landmark_model_path = _migrate_legacy_model_path(val) if val else _default_model_path("landmark")
        val = s.value("segmentation_model", "")
        self._segmentation_model_path = _migrate_legacy_model_path(val) if val else _default_model_path("segmentation")
        val = s.value("wing_isolation_model", "")
        self._wing_isolation_model_path = (
            _migrate_legacy_model_path(val) if val else _default_model_path("wing_isolation")
        )

        def _parse_optional_float(raw) -> Optional[float]:
            if raw in (None, ""):
                return None
            try:
                v = float(raw)
            except (TypeError, ValueError):
                return None
            return v if v > 0 else None

        # `s.value(key, None)` returns None when the key was never written —
        # we use that to distinguish "first launch / pristine settings" (apply
        # the factory default) from "user deliberately cleared the field"
        # (saved as "", parse to None).
        raw_lm = s.value("models/landmark_target_um_per_px", None)
        if raw_lm is not None:
            self._landmark_target_um_per_px = _parse_optional_float(raw_lm)
        raw_seg = s.value("models/segmentation_target_um_per_px", None)
        if raw_seg is not None:
            self._segmentation_target_um_per_px = _parse_optional_float(raw_seg)
        raw_wing = s.value("models/wing_isolation_target_um_per_px", None)
        if raw_wing is not None:
            self._wing_isolation_target_um_per_px = _parse_optional_float(raw_wing)
        saved_active = s.value("models/active_rescale_target", "")
        if saved_active in ("landmark", "segmentation", "wing_isolation"):
            self._active_rescale_target = saved_active
        saved_tol_low = s.value("resolution/tolerance_low", "")
        try:
            if saved_tol_low not in (None, ""):
                self._rescale_tolerance_low = float(saved_tol_low)
        except (TypeError, ValueError):
            pass
        saved_tol_high = s.value("resolution/tolerance_high", "")
        try:
            if saved_tol_high not in (None, ""):
                self._rescale_tolerance_high = float(saved_tol_high)
        except (TypeError, ValueError):
            pass
        saved_wing = s.value("wing_isolation_enabled", None)
        if saved_wing is not None:
            self._wing_isolation_enabled = saved_wing == "true" or saved_wing is True
        saved_wef = s.value("wing_expand_fraction", None)
        if saved_wef is not None:
            try:
                self._wing_expand_fraction = float(saved_wef)
            except (TypeError, ValueError):
                pass
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
        # Migrate legacy "output/overlay" setting → both new keys (vein + intervein).
        legacy_overlay = s.value("output/overlay", None)
        if legacy_overlay is not None:
            legacy_on = legacy_overlay == "true" or legacy_overlay is True
            for new_key in ("vein_overlay", "intervein_overlay"):
                if s.value(f"output/{new_key}", None) is None:
                    s.setValue(f"output/{new_key}", legacy_on)
            s.remove("output/overlay")
        for key, chk in self.output_checks.items():
            saved = s.value(f"output/{key}", None)
            if saved is not None:
                chk.setChecked(saved == "true" or saved is True)
        for key in self._intermediate_outputs:
            saved = s.value(f"output/{key}", None)
            if saved is not None:
                self._intermediate_outputs[key] = saved == "true" or saved is True
        # Migrate legacy csv_group keys from the earlier split:
        #   wing_dimensions (area + length) → wing_area (just area)
        #   crossvein_distance              → cv_ratio (now also owns wing length)
        # Only write the new key if it hasn't already been saved.
        for old_key, new_key in (("wing_dimensions", "wing_area"), ("crossvein_distance", "cv_ratio")):
            legacy = s.value(f"csv_group/{old_key}", None)
            if legacy is not None:
                if s.value(f"csv_group/{new_key}", None) is None:
                    s.setValue(f"csv_group/{new_key}", legacy == "true" or legacy is True)
                s.remove(f"csv_group/{old_key}")
        for gkey, gchk in self.csv_group_checks.items():
            saved = s.value(f"csv_group/{gkey}", None)
            if saved is not None:
                gchk.setChecked(saved == "true" or saved is True)
        # Sync the nested-group container's enabled state with the parent CSV checkbox.
        if self._csv_group_container is not None and "csv" in self.output_checks:
            self._csv_group_container.setEnabled(self.output_checks["csv"].isChecked())
        saved_svt = s.value("show_vein_tissue", None)
        if saved_svt is not None:
            self._show_vein_tissue = saved_svt == "true" or saved_svt is True
        saved_dor = s.value("do_rotation", None)
        if saved_dor is not None:
            self._do_rotation = saved_dor == "true" or saved_dor is True
        saved_rmc = s.value("rotation_mirror_correct", None)
        if saved_rmc is not None:
            self._rotation_mirror_correct = saved_rmc == "true" or saved_rmc is True
        saved_iul = s.value("include_unreliable_landmarks", None)
        if saved_iul is not None:
            self._include_unreliable_landmarks = saved_iul == "true" or saved_iul is True
        saved_gate = s.value("gate_override_json", "")
        if saved_gate:
            try:
                import json as _json

                self._gate_override = _json.loads(saved_gate)
            except Exception:
                self._gate_override = None
        saved_uld = s.value("user_landmark_distances_json", "")
        if saved_uld:
            try:
                import json as _json

                parsed = _json.loads(saved_uld)
                if isinstance(parsed, list):
                    self._user_landmark_distances = [p for p in parsed if isinstance(p, dict)]
            except Exception:
                self._user_landmark_distances = []
        saved_icm = s.value("include_custom_measurements", None)
        if saved_icm is not None:
            self.include_custom_measurements_chk.setChecked(saved_icm == "true" or saved_icm is True)
        saved_dsi = s.value("distance_sample_image", "")
        if saved_dsi:
            self._distance_sample_image = saved_dsi
        saved_dsl = s.value("distance_sample_landmarks", "")
        if saved_dsl:
            self._distance_sample_landmarks = saved_dsl
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
        if not self._landmark_model_path:
            QMessageBox.warning(self, "Missing Model", "Please select a landmark model in Settings → Models.")
            return
        if not self._segmentation_model_path:
            QMessageBox.warning(
                self, "Missing Model", "Please select a segmentation model folder in Settings → Models."
            )
            return
        if self.config.um_per_px is None:
            QMessageBox.warning(
                self,
                "Missing Scale",
                "Please enter a µm/px conversion factor in the Scale field before running.",
            )
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
        self._progress_pct_high = 0
        self.progress.setValue(0)
        # Restore the default Highlight color (the bar turns green on a clean
        # finish — reset it here so the new run starts in the default color).
        pal = self.progress.palette()
        pal.setColor(QPalette.Highlight, self._progress_default_highlight)
        self.progress.setPalette(pal)
        self._run_start_time = time.monotonic()
        self._eta_smoothed_seconds = None
        self._current_stage = None
        self._stage_first_event_time = None
        self._stage_completions = 0
        self._stage_total = 0
        self._last_completion_time = None
        self._avg_time_per_image = None
        self.eta_label.setText("Estimating time until pipeline finishes…")
        self._progress_timer.start()
        # Pre-compute the Stage1/Stage2 wall-time weight split for the chosen
        # outputs so per-image progress events land on a unified 0–100 scale
        # that reflects relative wall-time cost (Stage 2 dominates by ~5×).
        outputs_now = self._selected_outputs()
        self._progress_stage1_share, self._progress_stage2_share = compute_progress_weights(
            outputs_now,
            wing_isolation_enabled=self._wing_isolation_enabled,
            skip_intervein_regions=getattr(self.config, "skip_intervein_regions", False),
        )
        self.log_text.clear()
        self._save_settings()
        self._log("Starting TRACE pipeline...")

        for i in range(self.image_list.count()):
            self.image_list.item(i).setForeground(QColor(208, 208, 208))

        wing_model_dir = None
        if self._wing_isolation_enabled and self._wing_isolation_model_path.strip():
            wing_model_dir = Path(self._wing_isolation_model_path)

        # Resolve the active per-model target. When the user hasn't entered one
        # for the active model, leave target_um_per_px None — preprocessing's
        # Stage 1 then no-ops cleanly.
        target_um_per_px: Optional[float]
        if self._active_rescale_target == "landmark":
            target_um_per_px = self._landmark_target_um_per_px
        elif self._active_rescale_target == "wing_isolation":
            target_um_per_px = self._wing_isolation_target_um_per_px
        else:
            target_um_per_px = self._segmentation_target_um_per_px

        self.worker = TraceWorker(
            kwargs=dict(
                input_dir=Path(self.input_edit.text()),
                output_dir=Path(self.output_edit.text()),
                landmark_checkpoint=Path(self._landmark_model_path),
                segmentation_model_dir=Path(self._segmentation_model_path),
                config=self.config,
                keep_intermediates=False,
                outputs=self._selected_outputs(),
                max_workers=self.workers_spin.value(),
                show_vein_tissue=self._show_vein_tissue,
                include_unreliable_landmarks=self._include_unreliable_landmarks,
                gate_override=self._gate_override,
                wing_isolation_model_dir=wing_model_dir,
                wing_expand_fraction=self._wing_expand_fraction,
                recursive=self.recursive_chk.isChecked(),
                do_rotation=self._do_rotation,
                rotation_mirror_correct=self._rotation_mirror_correct,
                user_landmark_distances=(
                    list(self._user_landmark_distances) if self.include_custom_measurements_chk.isChecked() else []
                ),
                csv_measurement_groups={gkey for gkey, gchk in self.csv_group_checks.items() if gchk.isChecked()},
                # Tie landmark batch size to the Workers spinbox so a single setting
                # controls Stage 1 batching and Stage 2 parallelism together.
                landmark_batch_size=self.workers_spin.value(),
                target_um_per_px=target_um_per_px,
                rescale_tolerance_low=self._rescale_tolerance_low,
                rescale_tolerance_high=self._rescale_tolerance_high,
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
        """Update internal stage-tracking state on each progress event.

        Doesn't touch the progress bar / ETA directly — that happens via
        _refresh_progress, which is called both here and from a 250ms QTimer
        so the bar interpolates smoothly between completion events.
        """
        # Detect stage transitions so the hybrid ETA can reset its throughput
        # tracker (Stage 2 per-image cost is wildly different from Stage 1).
        if stage in ("preprocessing", "analysis") and stage != self._current_stage:
            self._current_stage = stage
            self._stage_first_event_time = time.monotonic()
            self._stage_completions = 0
            self._stage_total = total
            self._last_completion_time = None
            self._avg_time_per_image = None
        self._stage_total = max(self._stage_total, total)

        # Each image fires multiple progress events ("starting", "landmarks: …",
        # "hinge: …", "segmentation: …", "done"). Only the "done" event
        # represents an actual completion. Lock in the average time per image
        # at each completion so the smoothing interpolator has a stable
        # per-tick value (otherwise recomputing avg = elapsed/K every tick
        # forces fractional-progress-since-last-completion to 0).
        is_done = isinstance(detail, str) and detail.startswith("done")
        if is_done and self._stage_first_event_time is not None:
            self._stage_completions = max(self._stage_completions, idx + 1)
            self._last_completion_time = time.monotonic()
            stage_elapsed = self._last_completion_time - self._stage_first_event_time
            self._avg_time_per_image = stage_elapsed / max(self._stage_completions, 1)

        self._refresh_progress()
        msg = f"[{idx + 1}/{total}] {name}: {stage} - {detail}"
        self.statusBar().showMessage(msg)
        self._log(msg)

    def _smoothed_within_stage_fraction(self) -> float:
        """Return 0.0..1.0 reflecting smoothed progress within the current stage.

        Between completion events the value advances linearly toward the next
        expected completion based on the locked-in average time per image,
        so the bar doesn't sit frozen for tens of seconds when parallel
        workers complete in clustered bursts.
        """
        if self._stage_total <= 0 or self._stage_first_event_time is None:
            return 0.0
        if self._last_completion_time is None or self._avg_time_per_image is None:
            # No completions yet in this stage — bar stays at the previous
            # stage's final position (or 0 for Stage 1). Honest: we don't
            # have a per-image time to extrapolate from yet.
            return 0.0
        time_since_last = max(0.0, time.monotonic() - self._last_completion_time)
        # Fractional advance toward the next completion (0 → 1 over avg).
        # Cap at 1.0 so we never appear to have completed an image we haven't.
        fractional = min(1.0, time_since_last / max(self._avg_time_per_image, 0.01))
        estimated = min(float(self._stage_total), self._stage_completions + fractional)
        return estimated / self._stage_total

    def _refresh_progress(self) -> None:
        """Compute the smoothed pct + ETA from current state. Called both
        when a progress event arrives and from a 250ms QTimer."""
        if self._current_stage is None:
            return
        within_stage = self._smoothed_within_stage_fraction()
        if self._current_stage == "analysis":
            pct_float = (self._progress_stage1_share + self._progress_stage2_share * within_stage) * 100.0
        else:
            pct_float = self._progress_stage1_share * within_stage * 100.0
        pct = min(99, int(pct_float))
        if pct > self._progress_pct_high:
            self._progress_pct_high = pct
            self.progress.setValue(pct)
        # ETA updates every tick whether or not pct ticked over an integer
        # boundary, so the displayed time also evolves smoothly.
        self._update_eta()

    @staticmethod
    def _format_eta(seconds: float) -> str:
        import math

        if not math.isfinite(seconds) or seconds <= 0:
            return "Estimating time until pipeline finishes…"
        # Sub-minute precision is noisy with per-image variance and parallelism;
        # round to minutes and use a "<1m" floor so the displayed value isn't
        # misleadingly specific.
        total_minutes = int(round(seconds / 60))
        if total_minutes < 1:
            return "<1m until pipeline finishes"
        if total_minutes < 60:
            return f"~{total_minutes}m until pipeline finishes"
        h, m = divmod(total_minutes, 60)
        return f"~{h}h {m}m until pipeline finishes"

    def _update_eta(self) -> None:
        """Recompute the smoothed ETA and update the label under the progress bar.

        Hybrid of two estimates:
          • Percent-based (prior): elapsed × (100 − pct) / pct. Stable from the
            first event because it leans on the Stage 1 / Stage 2 wall-time
            weights from `compute_progress_weights`, but uniformly off when a
            run's per-image cost deviates from the reference workload.
          • Throughput-based (empirical): observe images-per-second in the
            current stage and divide remaining images by it. Auto-adapts to
            this run's actual cost profile, but needs ≥2 samples to be stable.
        Blend weight grows from 0 → 1 as Stage samples accumulate (full
        throughput at 3 completions), then we exponentially smooth so the
        displayed value doesn't whiplash on outlier images.
        """
        if self._run_start_time is None or self._progress_pct_high <= 0:
            return
        elapsed = time.monotonic() - self._run_start_time
        pct = self._progress_pct_high
        percent_eta = elapsed * (100 - pct) / pct

        eta_raw = percent_eta
        if self._stage_first_event_time is not None and self._stage_completions >= 2 and self._stage_total > 0:
            stage_elapsed = time.monotonic() - self._stage_first_event_time
            if stage_elapsed > 0:
                throughput = self._stage_completions / stage_elapsed  # images/sec
                stage_remaining = max(0, self._stage_total - self._stage_completions)
                stage_remaining_seconds = stage_remaining / max(throughput, 1e-6)

                if self._current_stage == "preprocessing":
                    # Project full-Stage-1 wall time from observed throughput,
                    # then derive total run from the Stage 1 share prior. Stage 2
                    # remaining is what's left after now.
                    if self._progress_stage1_share > 0:
                        full_stage1 = self._stage_total / max(throughput, 1e-6)
                        total_predicted = full_stage1 / self._progress_stage1_share
                        throughput_eta = max(0.0, total_predicted - elapsed)
                    else:
                        throughput_eta = stage_remaining_seconds
                else:
                    # Stage 2 (analysis) — empirical throughput on remaining images.
                    throughput_eta = stage_remaining_seconds

                # Blend: w grows with completed samples in the current stage.
                # Full throughput-based at 3 completions; below that we blend
                # in the percent-based prior to absorb sample noise.
                threshold = 3.0
                w = min(1.0, self._stage_completions / threshold)
                eta_raw = w * throughput_eta + (1 - w) * percent_eta

        # Exponential moving average — α=0.3 favors stability over reactivity.
        alpha = 0.3
        if self._eta_smoothed_seconds is None:
            self._eta_smoothed_seconds = eta_raw
        else:
            self._eta_smoothed_seconds = alpha * eta_raw + (1 - alpha) * self._eta_smoothed_seconds
        self.eta_label.setText(self._format_eta(self._eta_smoothed_seconds))

    def _on_all_done(self, results):
        self.btn_run.setEnabled(True)
        self.btn_cancel.setEnabled(False)
        self._progress_timer.stop()

        if not results:
            self._log("\nPipeline cancelled or no results.")
            self.statusBar().showMessage("Cancelled")
            self.eta_label.setText("")
            return

        # Pipeline finished cleanly — release the 99% cap and snap to 100.
        self._progress_pct_high = 100
        self.progress.setValue(100)
        # Recolor the filled chunk green via palette (not stylesheet) so the
        # native rendering path stays in place — only the chunk color shifts,
        # no indentation/border/text changes. Reverted at the start of the
        # next run by _run_pipeline.
        pal = self.progress.palette()
        pal.setColor(QPalette.Highlight, QColor("#5cb85c"))
        self.progress.setPalette(pal)
        self.eta_label.setText("Done")

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

        # Open the output folder in the system file manager so the user can
        # see results without hunting for the path.
        out_dir = self.output_edit.text().strip()
        if out_dir and Path(out_dir).is_dir():
            from PyQt5.QtCore import QUrl
            from PyQt5.QtGui import QDesktopServices

            QDesktopServices.openUrl(QUrl.fromLocalFile(out_dir))

    def _on_error(self, msg):
        self.btn_run.setEnabled(True)
        self.btn_cancel.setEnabled(False)
        self._progress_timer.stop()
        self.eta_label.setText("")
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
    # Force tooltips to match the dark dialog look — without this, on macOS
    # they fall back to the unstyled native popup and ignore the palette.
    # QGroupBox border is bumped to a lighter shade (#7a7a7a) so the section
    # bounding boxes on the main window stand out against the dark background;
    # the title needs an explicit margin/padding so it doesn't overlap the
    # border line when we set an explicit border.
    app.setStyleSheet(
        "QToolTip { background-color: #2d2d2d; color: #d0d0d0;"
        " border: 1px solid #555555; padding: 4px; }"
        " QGroupBox { border: 1px solid #7a7a7a; border-radius: 4px;"
        " margin-top: 10px; padding-top: 6px; }"
        " QGroupBox::title { subcontrol-origin: margin;"
        " subcontrol-position: top left; left: 8px; padding: 0 4px;"
        " color: #d0d0d0; }"
    )


def main():
    app = QApplication(sys.argv)
    _apply_dark_palette(app)
    window = TraceWindow()
    window.show()
    sys.exit(app.exec_())
