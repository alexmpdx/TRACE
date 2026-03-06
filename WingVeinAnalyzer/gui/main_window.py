"""QMainWindow: layout, step sidebar, image panes, parameter bar, description panel."""

from __future__ import annotations

import logging
import traceback
from pathlib import Path
from typing import Optional

from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtGui import QFont
from PyQt5.QtWidgets import (
    QAction,
    QApplication,
    QDoubleSpinBox,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSlider,
    QSplitter,
    QStatusBar,
    QTextEdit,
    QToolBar,
    QVBoxLayout,
    QWidget,
)

from WingVeinAnalyzer.gui.file_selector import FilePair, discover_file_pairs
from WingVeinAnalyzer.gui.image_widget import ImageWidget
from WingVeinAnalyzer.gui.step_definitions import NUM_STEPS, STEP_DEFS
from WingVeinAnalyzer.gui.step_renderers import render_step
from WingVeinAnalyzer.gui.step_runner import StepRunner

logger = logging.getLogger(__name__)


class StepWorker(QThread):
    """Run a step in a background thread to keep the UI responsive."""

    finished = pyqtSignal(int)       # step index
    error = pyqtSignal(str, str)     # step name, traceback

    def __init__(self, runner: StepRunner, step_index: int, parent=None):
        super().__init__(parent)
        self._runner = runner
        self._step_index = step_index

    def run(self):
        try:
            self._runner.run_step(self._step_index)
            self.finished.emit(self._step_index)
        except Exception:
            tb = traceback.format_exc()
            name = STEP_DEFS[self._step_index].name if self._step_index < NUM_STEPS else "?"
            self.error.emit(name, tb)


class MainWindow(QMainWindow):
    """Main window for the WingVeinAnalyzer step-by-step GUI."""

    def __init__(self):
        super().__init__()
        self.setWindowTitle("WingVeinAnalyzer — Step-by-Step Pipeline")
        self.resize(1600, 1000)

        self._runner = StepRunner()
        self._current_step: int = -1
        self._worker: Optional[StepWorker] = None
        self._file_pairs: list[FilePair] = []
        self._current_pair: Optional[FilePair] = None

        self._setup_ui()
        self._setup_toolbar()
        self._setup_statusbar()
        self._update_nav_buttons()

    # ------------------------------------------------------------------
    # UI Setup
    # ------------------------------------------------------------------

    def _setup_ui(self) -> None:
        """Build the main layout."""
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QHBoxLayout(central)
        main_layout.setContentsMargins(4, 4, 4, 4)

        # Left: step sidebar
        self._step_list = QListWidget()
        self._step_list.setFixedWidth(200)
        self._step_list.setFont(QFont("Menlo", 11))
        # Block signals while populating to avoid spurious currentRowChanged
        self._step_list.blockSignals(True)
        for sdef in STEP_DEFS:
            item = QListWidgetItem(f"  {sdef.index:2d}. {sdef.short_name}")
            item.setFlags(
                (item.flags() & ~Qt.ItemIsSelectable) | Qt.ItemIsEnabled
            )
            self._step_list.addItem(item)
        self._step_list.setCurrentRow(-1)
        self._step_list.blockSignals(False)
        self._step_list.currentRowChanged.connect(self._on_step_selected)
        main_layout.addWidget(self._step_list)

        # Right: main content area
        right_layout = QVBoxLayout()
        right_layout.setContentsMargins(0, 0, 0, 0)

        # Image panes with labels
        left_container = QWidget()
        left_layout = QVBoxLayout(left_container)
        left_layout.setContentsMargins(0, 0, 0, 0)
        self._left_label = QLabel("INPUT")
        self._left_label.setAlignment(Qt.AlignCenter)
        self._left_label.setFont(QFont("Menlo", 10, QFont.Bold))
        self._left_label.setStyleSheet("color: #666; background: #f0f0f0; padding: 2px;")
        left_layout.addWidget(self._left_label)
        self._left_pane = ImageWidget()
        left_layout.addWidget(self._left_pane)

        right_container = QWidget()
        right_lay = QVBoxLayout(right_container)
        right_lay.setContentsMargins(0, 0, 0, 0)
        self._right_label = QLabel("OUTPUT")
        self._right_label.setAlignment(Qt.AlignCenter)
        self._right_label.setFont(QFont("Menlo", 10, QFont.Bold))
        self._right_label.setStyleSheet("color: #666; background: #f0f0f0; padding: 2px;")
        right_lay.addWidget(self._right_label)
        self._right_pane = ImageWidget()
        right_lay.addWidget(self._right_pane)

        img_splitter = QSplitter(Qt.Horizontal)
        img_splitter.addWidget(left_container)
        img_splitter.addWidget(right_container)
        img_splitter.setSizes([700, 700])
        right_layout.addWidget(img_splitter, stretch=5)

        # Parameter bar
        self._param_bar = QScrollArea()
        self._param_bar.setFixedHeight(50)
        self._param_bar.setWidgetResizable(True)
        self._param_bar.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self._param_bar.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self._param_widget = QWidget()
        self._param_layout = QHBoxLayout(self._param_widget)
        self._param_layout.setContentsMargins(8, 4, 8, 4)
        self._param_bar.setWidget(self._param_widget)
        right_layout.addWidget(self._param_bar)

        # Description panel (collapsible)
        desc_container = QWidget()
        desc_layout = QVBoxLayout(desc_container)
        desc_layout.setContentsMargins(0, 0, 0, 0)

        self._desc_toggle = QPushButton("Description [v]")
        self._desc_toggle.setCheckable(True)
        self._desc_toggle.setChecked(True)
        self._desc_toggle.setFixedHeight(24)
        self._desc_toggle.setStyleSheet("text-align: left; padding-left: 8px;")
        self._desc_toggle.toggled.connect(self._toggle_description)
        desc_layout.addWidget(self._desc_toggle)

        self._desc_text = QTextEdit()
        self._desc_text.setReadOnly(True)
        self._desc_text.setFont(QFont("Menlo", 11))
        self._desc_text.setMaximumHeight(180)
        desc_layout.addWidget(self._desc_text)

        right_layout.addWidget(desc_container)
        main_layout.addLayout(right_layout, stretch=1)

    def _setup_toolbar(self) -> None:
        """Build the navigation toolbar."""
        toolbar = QToolBar("Navigation")
        toolbar.setMovable(False)
        self.addToolBar(toolbar)

        # Open folder
        self._open_action = QAction("Open Folder", self)
        self._open_action.setShortcut("Ctrl+O")
        self._open_action.triggered.connect(self._on_open_folder)
        toolbar.addAction(self._open_action)

        toolbar.addSeparator()

        # File selector combo
        self._file_label = QLabel("  No files loaded  ")
        self._file_label.setFont(QFont("Menlo", 11))
        toolbar.addWidget(self._file_label)

        toolbar.addSeparator()

        # Navigation
        self._prev_action = QAction("<< Prev", self)
        self._prev_action.setShortcut("Left")
        self._prev_action.triggered.connect(self._on_prev)
        toolbar.addAction(self._prev_action)

        self._next_action = QAction("Next >>", self)
        self._next_action.setShortcut("Right")
        self._next_action.triggered.connect(self._on_next)
        toolbar.addAction(self._next_action)

        toolbar.addSeparator()

        # Fit views
        fit_action = QAction("Fit Views", self)
        fit_action.setShortcut("F")
        fit_action.triggered.connect(self._fit_views)
        toolbar.addAction(fit_action)

        toolbar.addSeparator()

        # Smoothing slider
        smooth_label = QLabel("  Smooth: ")
        smooth_label.setFont(QFont("Menlo", 11))
        toolbar.addWidget(smooth_label)

        self._smooth_slider = QSlider(Qt.Horizontal)
        self._smooth_slider.setRange(0, 100)  # 0.0 – 10.0 in 0.1 steps
        self._smooth_slider.setValue(30)       # default sigma = 3.0
        self._smooth_slider.setSingleStep(5)   # 0.5 increments
        self._smooth_slider.setPageStep(10)    # 1.0 increments
        self._smooth_slider.setFixedWidth(140)
        self._smooth_slider.setToolTip("Gaussian smoothing sigma for vein lines and region boundaries")
        self._smooth_slider.valueChanged.connect(self._on_smooth_changed)
        toolbar.addWidget(self._smooth_slider)

        self._smooth_value_label = QLabel(" 3.0 ")
        self._smooth_value_label.setFont(QFont("Menlo", 11))
        self._smooth_value_label.setFixedWidth(40)
        toolbar.addWidget(self._smooth_value_label)

        toolbar.addSeparator()

        # Scale (µm/px) input
        scale_label = QLabel("  \u00b5m/px: ")
        scale_label.setFont(QFont("Menlo", 11))
        toolbar.addWidget(scale_label)

        self._scale_spin = QDoubleSpinBox()
        self._scale_spin.setRange(0.001, 100.0)
        self._scale_spin.setDecimals(3)
        self._scale_spin.setValue(0.483)
        self._scale_spin.setSingleStep(0.01)
        self._scale_spin.setFixedWidth(90)
        self._scale_spin.setToolTip("Micrometers per pixel — used for all distance/area thresholds and output calibration")
        self._scale_spin.valueChanged.connect(self._on_scale_changed)
        toolbar.addWidget(self._scale_spin)

        toolbar.addSeparator()

        # Batch
        self._batch_action = QAction("Run All (Batch)", self)
        self._batch_action.triggered.connect(self._on_batch)
        toolbar.addAction(self._batch_action)

    def _setup_statusbar(self) -> None:
        """Set up the status bar."""
        self._statusbar = QStatusBar()
        self.setStatusBar(self._statusbar)
        self._statusbar.showMessage("Ready — Open a folder to begin")

    # ------------------------------------------------------------------
    # Navigation
    # ------------------------------------------------------------------

    def _on_prev(self) -> None:
        """Navigate to previous step."""
        if self._current_pair is None or self._current_step <= 0:
            return
        target = self._current_step - 1
        self._step_list.setCurrentRow(target)

    def _on_next(self) -> None:
        """Navigate to next step (computing if needed)."""
        if self._current_pair is None:
            return

        target = self._current_step + 1
        if target >= NUM_STEPS:
            return

        # Need to compute all steps up to target
        needs_computation = self._runner.state_at(target) is None
        if needs_computation:
            self._run_step_async(target)
        else:
            self._step_list.setCurrentRow(target)

    def _on_step_selected(self, row: int) -> None:
        """Handle step selection from the sidebar."""
        if row < 0 or row >= NUM_STEPS:
            return
        # Guard: no file loaded yet
        if self._current_pair is None:
            return

        state = self._runner.state_at(row)
        if state is None:
            # Need to compute — run through to this step
            self._run_step_async(row)
            return

        self._current_step = row
        self._display_step(row)
        self._update_nav_buttons()

    def _on_smooth_changed(self, value: int) -> None:
        """Handle smoothing slider change."""
        sigma = value / 10.0
        self._smooth_value_label.setText(f" {sigma:.1f} ")
        self._runner.smooth_sigma = sigma

        # If we're on the overlay step (19), re-render immediately
        if self._current_step == NUM_STEPS - 1 and self._current_pair is not None:
            self._runner.invalidate_from(NUM_STEPS - 1)
            self._run_step_async(NUM_STEPS - 1)

    def _on_scale_changed(self, value: float) -> None:
        """Handle µm/px scale change."""
        self._runner.um_per_px = value
        # Invalidate all steps since scale affects all distance thresholds
        if self._current_pair is not None:
            self._runner.invalidate_from(0)

    def _update_nav_buttons(self) -> None:
        """Enable/disable navigation based on current state."""
        has_file = self._current_pair is not None
        self._prev_action.setEnabled(has_file and self._current_step > 0)
        self._next_action.setEnabled(has_file and self._current_step < NUM_STEPS - 1)

        # Update step list icons
        for i in range(NUM_STEPS):
            item = self._step_list.item(i)
            state = self._runner.state_at(i)
            sdef = STEP_DEFS[i]
            if i == self._current_step:
                item.setText(f"> {sdef.index:2d}. {sdef.short_name}")
                item.setForeground(Qt.blue)
            elif state is not None:
                item.setText(f"  {sdef.index:2d}. {sdef.short_name}  [done]")
                item.setForeground(Qt.darkGreen)
            else:
                item.setText(f"  {sdef.index:2d}. {sdef.short_name}")
                item.setForeground(Qt.black)

    # ------------------------------------------------------------------
    # Step execution
    # ------------------------------------------------------------------

    def _run_step_async(self, target: int) -> None:
        """Run steps up to target synchronously with UI updates."""
        # Guard: no file loaded
        if self._current_pair is None:
            logger.warning("_run_step_async called with no file loaded")
            return

        if self._worker is not None and self._worker.isRunning():
            return

        # Find the first step that needs computation
        start = 0
        for i in range(target + 1):
            if self._runner.state_at(i) is None:
                start = i
                break
        else:
            # All cached — just display
            self._current_step = target
            self._display_step(target)
            self._update_nav_buttons()
            return

        self._statusbar.showMessage(f"Computing step {start}..{target}...")
        self._set_ui_busy(True)

        try:
            for i in range(start, target + 1):
                step_name = STEP_DEFS[i].name
                self._statusbar.showMessage(f"Computing step {i}: {step_name}...")
                logger.info("Running step %d: %s", i, step_name)
                QApplication.processEvents()
                self._runner.run_step(i)
                logger.info("Step %d complete", i)
        except Exception:
            tb = traceback.format_exc()
            logger.error("Step %d failed:\n%s", i, tb)
            self._on_step_error(STEP_DEFS[i].name, tb)
            self._set_ui_busy(False)
            return

        self._set_ui_busy(False)
        self._current_step = target
        self._step_list.blockSignals(True)
        self._step_list.setCurrentRow(target)
        self._step_list.blockSignals(False)
        self._display_step(target)
        self._update_nav_buttons()
        self._statusbar.showMessage(f"Step {target}: {STEP_DEFS[target].name} — done")

    def _on_step_error(self, step_name: str, tb: str) -> None:
        """Handle step execution error."""
        self._set_ui_busy(False)
        self._statusbar.showMessage(f"Error in {step_name}")
        QMessageBox.critical(
            self, f"Error in {step_name}",
            f"Step '{step_name}' failed:\n\n{tb[:1000]}",
        )

    def _set_ui_busy(self, busy: bool) -> None:
        """Disable/enable navigation during computation."""
        self._prev_action.setEnabled(not busy)
        self._next_action.setEnabled(not busy)
        self._open_action.setEnabled(not busy)
        self._batch_action.setEnabled(not busy)

    # ------------------------------------------------------------------
    # Display
    # ------------------------------------------------------------------

    def _display_step(self, index: int) -> None:
        """Render and display the images, params, and description for a step."""
        state = self._runner.state_at(index)
        if state is None:
            return

        sdef = STEP_DEFS[index]
        prev_state = self._runner.state_at(index - 1) if index > 0 else None

        # Render images
        try:
            left_img, right_img = render_step(index, state, prev_state)
            self._left_pane.set_image(left_img)
            self._right_pane.set_image(right_img)
            self._left_pane.fit_in_view()
            self._right_pane.fit_in_view()
        except Exception as e:
            logger.error("Render error at step %d: %s", index, e)
            self._statusbar.showMessage(f"Render error: {e}")

        # Update pane labels
        self._left_label.setText(f"INPUT — Step {index}: {sdef.name}")
        self._right_label.setText(f"OUTPUT — Step {index}: {sdef.name}")

        # Update parameter bar
        self._update_params(sdef, state)

        # Update description
        self._update_description(sdef)

    def _update_params(self, sdef, state) -> None:
        """Update the parameter bar with step params and runtime values."""
        # Clear existing
        while self._param_layout.count():
            child = self._param_layout.takeAt(0)
            if child.widget():
                child.widget().deleteLater()

        # Add definition params
        for param in sdef.params:
            label = QLabel(f"{param.name}: {param.value}")
            label.setFont(QFont("Menlo", 10))
            label.setStyleSheet(
                "background: #e8e8e8; padding: 2px 8px; border-radius: 3px; margin: 2px;"
            )
            if param.tooltip:
                label.setToolTip(param.tooltip)
            self._param_layout.addWidget(label)

        # Add runtime params from state
        if state.params_used:
            for key, val in state.params_used.items():
                label = QLabel(f"{key}: {val}")
                label.setFont(QFont("Menlo", 10))
                label.setStyleSheet(
                    "background: #d8e8f8; padding: 2px 8px; border-radius: 3px; margin: 2px;"
                )
                self._param_layout.addWidget(label)

        self._param_layout.addStretch()

    def _update_description(self, sdef) -> None:
        """Update the description panel."""
        text = f"<b>{sdef.name}</b><br><br>"
        text += sdef.description.replace("\n", "<br>")
        text += "<br><br><b>Pseudocode:</b><br>"
        text += f"<pre>{sdef.pseudocode}</pre>"
        self._desc_text.setHtml(text)

    def _toggle_description(self, checked: bool) -> None:
        """Toggle description panel visibility."""
        self._desc_text.setVisible(checked)
        self._desc_toggle.setText("Description [v]" if checked else "Description [>]")

    def _fit_views(self) -> None:
        """Reset zoom on both image panes."""
        self._left_pane.fit_in_view()
        self._right_pane.fit_in_view()

    # ------------------------------------------------------------------
    # File handling
    # ------------------------------------------------------------------

    def _on_open_folder(self) -> None:
        """Open a folder and discover TIFF+GeoJSON pairs."""
        folder = QFileDialog.getExistingDirectory(
            self, "Open Wing Data Folder",
        )
        if not folder:
            return

        folder_path = Path(folder)
        logger.info("Opening folder: %s", folder_path)
        pairs = discover_file_pairs(folder_path)

        if not pairs:
            QMessageBox.information(
                self, "No Files Found",
                f"No TIFF+GeoJSON pairs found in:\n{folder_path}",
            )
            return

        logger.info("Found %d file pairs", len(pairs))
        self._file_pairs = pairs

        # If multiple pairs, let user choose
        if len(pairs) > 1:
            from WingVeinAnalyzer.gui.batch_dialog import FileChooserDialog
            dialog = FileChooserDialog(pairs, self)
            if dialog.exec_():
                selected = dialog.selected_pair()
                if selected:
                    self._load_pair(selected)
            return

        self._load_pair(pairs[0])

    def _load_pair(self, pair: FilePair) -> None:
        """Load a file pair and start at step 0."""
        logger.info("Loading pair: %s", pair.display_name)
        logger.info("  Image: %s", pair.image_path)
        logger.info("  GeoJSON: %s", pair.geojson_path)

        self._current_pair = pair
        self._file_label.setText(f"  {pair.display_name}  ")
        self._runner.load_inputs(pair.image_path, pair.geojson_path)
        self._current_step = -1

        # Reset step list (block signals to avoid triggering _on_step_selected)
        self._step_list.blockSignals(True)
        for i in range(NUM_STEPS):
            item = self._step_list.item(i)
            sdef = STEP_DEFS[i]
            item.setText(f"  {sdef.index:2d}. {sdef.short_name}")
            item.setForeground(Qt.black)
        self._step_list.setCurrentRow(-1)
        self._step_list.blockSignals(False)

        self._left_pane.clear_image()
        self._right_pane.clear_image()

        self._statusbar.showMessage(f"Loaded: {pair.image_path.name}")

        # Auto-navigate to step 0
        self._run_step_async(0)

    # ------------------------------------------------------------------
    # Batch mode
    # ------------------------------------------------------------------

    def _on_batch(self) -> None:
        """Open batch processing dialog."""
        if not self._file_pairs:
            # Ask for folder first
            folder = QFileDialog.getExistingDirectory(
                self, "Select Folder for Batch Processing",
            )
            if not folder:
                return
            pairs = discover_file_pairs(Path(folder))
            if not pairs:
                QMessageBox.information(
                    self, "No Files Found",
                    "No TIFF+GeoJSON pairs found in the selected folder.",
                )
                return
            self._file_pairs = pairs

        from WingVeinAnalyzer.gui.batch_dialog import BatchDialog
        dialog = BatchDialog(
            self._file_pairs, self,
            smooth_sigma=self._runner.smooth_sigma,
            um_per_px=self._runner.um_per_px,
        )
        dialog.exec_()
