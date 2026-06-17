"""PyQt5 GUI for inspecting landmark predictions and ground truth."""

from __future__ import annotations

import io
import json
import re
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import yaml

from landmark_locator.data.dataset import _normalize_name, discover_landmarks

# Project root (LandmarkLocator/) for locating configs and data
_project_root = Path(__file__).resolve().parent.parent.parent
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg, NavigationToolbar2QT
from matplotlib.figure import Figure
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtGui import QColor, QFont, QImage, QPainter, QPixmap
from PyQt5.QtWidgets import (
    QAction,
    QApplication,
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFileDialog,
    QGraphicsPixmapItem,
    QGraphicsScene,
    QGraphicsView,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QListWidget,
    QMainWindow,
    QMenu,
    QMessageBox,
    QProgressDialog,
    QProxyStyle,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSlider,
    QSpinBox,
    QSplitter,
    QStatusBar,
    QStyle,
    QTableWidget,
    QTableWidgetItem,
    QTextEdit,
    QToolBar,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from landmark_locator.scripts.visualize import (
    _ensure_colors,
    draw_landmarks_on_image,
    generate_landmark_colors,
    load_ground_truth,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
MODEL_W, MODEL_H = 512, 352
HEATMAP_SIGMA = 5
HEATMAP_THUMB_W, HEATMAP_THUMB_H = 240, 165
IMAGE_EXTENSIONS = {
    ".tif",
    ".tiff",
    ".png",
    ".jpg",
    ".jpeg",
    ".bmp",
    ".psd",
    ".psb",
    ".heic",
    ".heif",
    ".svg",
    ".raw",
    ".dng",
    ".nef",
    ".cr2",
    ".cr3",
    ".arw",
    ".raf",
    ".orf",
    ".pef",
    ".rw2",
    ".srw",
    ".czi",
    ".nd2",
    ".lif",
    ".lsm",
}


# ---------------------------------------------------------------------------
# Native file-picker helpers (macOS click-fix)
# ---------------------------------------------------------------------------
def _pick_folder_native(caption: str, initial: str) -> str:
    """Show the native folder picker, working around the Qt-on-macOS bug.

    Symptom (Qt 5.15.x, macOS Big Sur+): when
    ``QFileDialog.getExistingDirectory`` is called with a Qt parent
    widget, Qt's modal-event grab intercepts mouse events that should
    land on the native NSOpenPanel's file-list pane. The path-navigator
    dropdown and the Open / Cancel buttons live in chrome outside the
    grab so they keep working — only the file list goes dead. Detaching
    the dialog (``parent=None``) restores click handling. Cost: the
    picker isn't centered on / sheeted to the main window. Windows /
    Linux behave the same as before.

    Returns the chosen absolute path, or an empty string if cancelled.
    """
    return QFileDialog.getExistingDirectory(None, caption, initial)


def _pick_file_native(caption: str, initial: str, name_filter: str = "") -> str:
    """File-open counterpart of :func:`_pick_folder_native` with the same parent=None workaround.

    ``name_filter`` follows the standard Qt syntax
    (``"YAML (*.yaml *.yml);;All Files (*)"``).

    Returns the chosen absolute path, or an empty string if cancelled.
    """
    path, _ = QFileDialog.getOpenFileName(None, caption, initial, name_filter)
    return path


def _pick_save_file_native(caption: str, initial: str, name_filter: str = "") -> str:
    """Save-as picker — same Qt-on-macOS workaround as the open / folder helpers.

    ``initial`` may be a pre-filled filename or a directory path.
    Returns the chosen absolute path, or an empty string if cancelled.
    """
    path, _ = QFileDialog.getSaveFileName(None, caption, initial, name_filter)
    return path


def _find_geojson_for_image(gt_dir: Path, image_name: str) -> tuple[Optional[Path], bool]:
    """Find the GeoJSON annotation file matching an image name, tolerating whitespace.

    Returns (path, fuzzy) where fuzzy is True if the match required whitespace tolerance.
    """
    # Exact match first
    candidate = gt_dir / (image_name + ".geojson")
    if candidate.exists():
        return candidate, False
    # Fuzzy match: compare with whitespace stripped
    target_clean = re.sub(r"\s+", "", image_name + ".geojson").lower()
    for f in gt_dir.iterdir():
        if not f.is_file() or f.suffix.lower() != ".geojson":
            continue
        if re.sub(r"\s+", "", f.name).lower() == target_clean:
            return f, True
    return None, False


def _truncate_pair(name_a: str, name_b: str, max_len: int = 30) -> tuple[str, str]:
    """Truncate two filenames to the same length for aligned display."""
    limit = max(max_len, 8)

    def _trunc(s: str) -> str:
        return s if len(s) <= limit else s[: limit - 3] + "..."

    return _trunc(name_a), _trunc(name_b)


def _make_display_name(internal_name: str) -> str:
    """Convert an internal snake_case name to a human-readable title."""
    return internal_name.replace("_", " ").title()


def _make_qcolors(names: list[str]) -> dict[str, QColor]:
    """Generate QColor map for a list of landmark names."""
    bgr_colors = generate_landmark_colors(names)
    return {name: QColor(bgr[2], bgr[1], bgr[0]) for name, bgr in bgr_colors.items()}


class _OutlinedCheckBoxStyle(QProxyStyle):
    """Proxy style that paints a light outline around every checkbox indicator.

    Overlays the outline after Qt's native indicator is drawn, so the native
    checkmark is preserved. Applied globally via QMainWindow.setStyle().
    """

    _OUTLINE = QColor("#888888")

    def drawPrimitive(self, element, option, painter, widget=None):
        super().drawPrimitive(element, option, painter, widget)
        if element in (QStyle.PE_IndicatorCheckBox, QStyle.PE_IndicatorItemViewItemCheck):
            painter.save()
            pen = painter.pen()
            pen.setColor(self._OUTLINE)
            pen.setWidth(1)
            painter.setPen(pen)
            painter.setBrush(Qt.NoBrush)
            rect = option.rect.adjusted(0, 0, -1, -1)
            painter.drawRect(rect)
            painter.restore()


# ---------------------------------------------------------------------------
# Data container
# ---------------------------------------------------------------------------
@dataclass
class ImageEntry:
    """Container for a single image's paths and cached results."""

    path: Path
    geojson_path: Optional[Path] = None
    prediction: Optional[dict] = None  # result from LandmarkPredictor.predict()
    gt: Optional[dict[str, tuple[float, float]]] = None
    gt_heatmaps: Optional[np.ndarray] = None  # (N, MODEL_H, MODEL_W)
    _gt_loaded: bool = field(default=False, repr=False)

    def load_gt(self) -> None:
        """Load ground truth from GeoJSON if available (cached)."""
        if self._gt_loaded:
            return
        self._gt_loaded = True
        if self.geojson_path and self.geojson_path.exists():
            self.gt = load_ground_truth(self.geojson_path)


# ---------------------------------------------------------------------------
# Standalone heatmap generation (mirrors LandmarkDataset._generate_heatmap)
# ---------------------------------------------------------------------------
def generate_gt_heatmaps(
    gt: dict[str, tuple[float, float]],
    landmark_order: list[str],
    width: int = MODEL_W,
    height: int = MODEL_H,
    sigma: float = HEATMAP_SIGMA,
    orig_w: Optional[int] = None,
    orig_h: Optional[int] = None,
) -> np.ndarray:
    """Render Gaussian heatmaps from ground-truth coordinates at model resolution."""
    heatmaps = np.zeros((len(landmark_order), height, width), dtype=np.float32)
    for i, name in enumerate(landmark_order):
        if name not in gt:
            continue
        kx, ky = gt[name]
        # Scale from original image coords to model resolution if needed
        if orig_w and orig_h:
            kx = kx * width / orig_w
            ky = ky * height / orig_h
        kx = np.clip(kx, 0, width - 1)
        ky = np.clip(ky, 0, height - 1)
        size = int(6 * sigma)
        x0, x1 = max(0, int(kx) - size), min(width, int(kx) + size + 1)
        y0, y1 = max(0, int(ky) - size), min(height, int(ky) + size + 1)
        if x0 >= x1 or y0 >= y1:
            continue
        xs = np.arange(x0, x1, dtype=np.float32)
        ys = np.arange(y0, y1, dtype=np.float32)
        xx, yy = np.meshgrid(xs, ys)
        gaussian = np.exp(-((xx - kx) ** 2 + (yy - ky) ** 2) / (2 * sigma**2))
        heatmaps[i, y0:y1, x0:x1] = gaussian
    return heatmaps


# ---------------------------------------------------------------------------
# Heatmap → QPixmap helper
# ---------------------------------------------------------------------------
def _colorize_heatmap(gray: np.ndarray) -> np.ndarray:
    """Colorize a uint8 grayscale array using OpenCV's HOT colormap (BGR output)."""
    return cv2.applyColorMap(gray, cv2.COLORMAP_HOT)


def heatmap_to_pixmap(
    channel: np.ndarray,
    thumb_w: int = HEATMAP_THUMB_W,
    thumb_h: int = HEATMAP_THUMB_H,
) -> QPixmap:
    """Convert a single-channel heatmap array to a colorized QPixmap thumbnail."""
    vmax = channel.max()
    if vmax > 0:
        norm = (channel / vmax * 255).astype(np.uint8)
    else:
        norm = np.zeros_like(channel, dtype=np.uint8)
    colored = _colorize_heatmap(norm)
    colored = cv2.resize(colored, (thumb_w, thumb_h))
    rgb = cv2.cvtColor(colored, cv2.COLOR_BGR2RGB)
    qimg = QImage(rgb.data, thumb_w, thumb_h, thumb_w * 3, QImage.Format_RGB888)
    return QPixmap.fromImage(qimg.copy())


# ---------------------------------------------------------------------------
# ImageWidget
# ---------------------------------------------------------------------------
class ImageWidget(QGraphicsView):
    """A QGraphicsView that displays a BGR numpy array with zoom and pan."""

    ZOOM_FACTOR = 1.15

    def __init__(self, parent=None):
        super().__init__(parent)
        self._scene = QGraphicsScene(self)
        self.setScene(self._scene)
        self._pixmap_item: QGraphicsPixmapItem | None = None
        self.setRenderHint(QPainter.SmoothPixmapTransform, True)
        self.setRenderHint(QPainter.Antialiasing, True)
        self.setDragMode(QGraphicsView.ScrollHandDrag)
        self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.AnchorUnderMouse)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self._placeholder = self._scene.addText("No image loaded")
        self._placeholder.setDefaultTextColor(Qt.gray)

    def set_image(self, bgr_array: np.ndarray) -> None:
        """Display a BGR numpy array (H, W, 3) as a pixmap."""
        if bgr_array is None:
            return
        h, w = bgr_array.shape[:2]
        channels = bgr_array.shape[2] if bgr_array.ndim == 3 else 1
        if channels == 3:
            rgb = bgr_array[:, :, ::-1].copy()
            qimage = QImage(rgb.data, w, h, w * 3, QImage.Format_RGB888)
        elif channels == 1:
            qimage = QImage(bgr_array.data, w, h, w, QImage.Format_Grayscale8)
        else:
            return
        pixmap = QPixmap.fromImage(qimage)
        self._scene.clear()
        self._pixmap_item = self._scene.addPixmap(pixmap)
        self._scene.setSceneRect(0, 0, w, h)
        self._placeholder = None

    def fit_in_view(self) -> None:
        """Reset zoom to fit the image in the viewport."""
        if self._pixmap_item is not None:
            self.fitInView(self._pixmap_item, Qt.KeepAspectRatio)

    def wheelEvent(self, event) -> None:
        """Zoom in/out with scroll wheel."""
        factor = self.ZOOM_FACTOR if event.angleDelta().y() > 0 else 1.0 / self.ZOOM_FACTOR
        self.scale(factor, factor)

    def clear_image(self) -> None:
        """Remove the current image and show placeholder."""
        self._scene.clear()
        self._pixmap_item = None
        self._placeholder = self._scene.addText("No image loaded")
        self._placeholder.setDefaultTextColor(Qt.gray)


# ---------------------------------------------------------------------------
# LegendWidget
# ---------------------------------------------------------------------------
class LegendWidget(QWidget):
    """Color legend for landmark types."""

    labels_toggled = pyqtSignal(bool)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._landmark_order: list[str] = []
        self._qcolors: dict[str, QColor] = {}
        self.setFixedHeight(50)
        # Overlay-labels toggle sits next to the painter-drawn "Legend" title.
        # Indicator styling comes from the main window's global QCheckBox stylesheet
        # so this checkbox matches every other one in the app.
        self._labels_chk = QCheckBox(self)
        self._labels_chk.setToolTip("Draw landmark name labels next to each point on the overlay.")
        self._labels_chk.setChecked(True)
        self._labels_chk.toggled.connect(self.labels_toggled.emit)
        self._labels_chk.adjustSize()
        self._labels_chk.raise_()
        self._labels_chk.show()
        self._position_labels_chk()

    def labels_enabled(self) -> bool:
        return self._labels_chk.isChecked()

    def resizeEvent(self, event) -> None:
        self._position_labels_chk()
        super().resizeEvent(event)

    def _position_labels_chk(self) -> None:
        # Anchor top-right, level with the "Legend" title drawn at y~5.
        hint = self._labels_chk.sizeHint()
        w = hint.width() if hint.width() > 0 else 18
        h = hint.height() if hint.height() > 0 else 18
        self._labels_chk.setGeometry(max(0, self.width() - w - 8), 4, w, h)
        self._labels_chk.raise_()

    def set_landmarks(self, landmark_order: list[str]) -> None:
        """Update the legend with a new set of landmarks."""
        self._landmark_order = landmark_order
        self._qcolors = _make_qcolors(landmark_order)
        self.setFixedHeight(len(landmark_order) * 22 + 50)
        self._position_labels_chk()
        self.update()

    def paintEvent(self, event) -> None:
        """Draw colored swatches and labels."""
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        font = QFont("sans-serif", 10)
        p.setFont(font)

        y = 5
        # Header
        p.setPen(Qt.white)
        p.drawText(10, y + 12, "Legend")
        y += 22

        # Prediction symbol
        p.setPen(Qt.lightGray)
        p.drawText(10, y + 12, "\u25cf Circle = Prediction")
        y += 18
        p.drawText(10, y + 12, "\u2716 Cross = Ground Truth")
        y += 22

        for name in self._landmark_order:
            color = self._qcolors.get(name, QColor(200, 200, 200))
            p.setBrush(color)
            p.setPen(Qt.NoPen)
            p.drawRect(10, y, 14, 14)
            p.setPen(Qt.lightGray)
            p.drawText(30, y + 12, _make_display_name(name))
            y += 20
        p.end()


# ---------------------------------------------------------------------------
# Clickable heatmap label
# ---------------------------------------------------------------------------
class _ClickableLabel(QLabel):
    """QLabel that emits a signal with its landmark name when clicked."""

    clicked = pyqtSignal(str)

    def __init__(self, name: str, parent=None):
        super().__init__(parent)
        self._name = name

    def mousePressEvent(self, event) -> None:
        self.clicked.emit(self._name)
        super().mousePressEvent(event)


# ---------------------------------------------------------------------------
# HeatmapPanel — right-side scrollable heatmap thumbnails
# ---------------------------------------------------------------------------
class HeatmapPanel(QScrollArea):
    """Scrollable panel showing predicted heatmap thumbnails with optional GT cross overlay."""

    heatmap_clicked = pyqtSignal(str)  # landmark name

    _STYLE_NORMAL = "background: #222; color: #666; border: 1px solid #444;"
    _STYLE_SELECTED = "background: #222; color: #666; border: 2px solid #2a82da;"

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWidgetResizable(True)
        self.setMinimumWidth(HEATMAP_THUMB_W + 40)
        self._container = QWidget()
        self._layout = QVBoxLayout(self._container)
        self._layout.setAlignment(Qt.AlignTop)
        self.setWidget(self._container)

        # GT directory label
        self._gt_dir_label = QLabel("GT: (none)")
        self._gt_dir_label.setStyleSheet("color: #aaa; font-size: 10px; padding: 2px;")
        self._gt_dir_label.setWordWrap(False)
        self._layout.addWidget(self._gt_dir_label)

        # GT cross toggle
        self._show_gt = QCheckBox("Show GT cross")
        self._show_gt.setChecked(True)
        self._show_gt.setStyleSheet("color: #ccc;")
        self._show_gt.toggled.connect(self._refresh)
        self._layout.addWidget(self._show_gt)

        # Color legend bar
        bar_row = QHBoxLayout()
        low_lbl = QLabel("Low")
        low_lbl.setStyleSheet("color: #888; font-size: 9px;")
        bar_row.addWidget(low_lbl)
        gradient = np.arange(256, dtype=np.uint8).reshape(1, 256)
        colored = _colorize_heatmap(gradient)  # (1, 256, 3) BGR
        colored = cv2.resize(colored, (HEATMAP_THUMB_W - 50, 14))
        rgb = cv2.cvtColor(colored, cv2.COLOR_BGR2RGB)
        h, w = rgb.shape[:2]
        qimg = QImage(rgb.data, w, h, w * 3, QImage.Format_RGB888)
        bar_lbl = QLabel()
        bar_lbl.setPixmap(QPixmap.fromImage(qimg.copy()))
        bar_lbl.setFixedHeight(14)
        bar_row.addWidget(bar_lbl)
        high_lbl = QLabel("High")
        high_lbl.setStyleSheet("color: #888; font-size: 9px;")
        bar_row.addWidget(high_lbl)
        self._layout.addLayout(bar_row)

        self._pred_labels: dict[str, _ClickableLabel] = {}
        self._headers: dict[str, QLabel] = {}
        self._selected: Optional[str] = None
        self._landmark_order: list[str] = []
        self._qcolors: dict[str, QColor] = {}
        self._stretch_item = self._layout.addStretch()

        # Cached state for refresh on toggle
        self._cur_pred_heatmaps: Optional[np.ndarray] = None
        self._cur_gt_coords: Optional[dict[str, tuple[float, float]]] = None

    def set_gt_dir_label(self, text: str) -> None:
        """Update the GT directory label."""
        self._gt_dir_label.setText(text)

    def mount_opacity_controls(self, slider: QSlider, value_label: QLabel) -> None:
        """Insert an Opacity slider row directly below the color gradient bar.

        Layout order at call time is: [gt_dir_label, show_gt, gradient_bar, stretch].
        We insert at index 3 so the row sits below the gradient and above the
        per-landmark thumbnails that get appended later by set_landmarks().
        """
        row = QHBoxLayout()
        opacity_lbl = QLabel("Opacity:")
        opacity_lbl.setStyleSheet("color: #aaa; font-size: 10px;")
        row.addWidget(opacity_lbl)
        slider.setParent(self._container)
        row.addWidget(slider)
        value_label.setParent(self._container)
        row.addWidget(value_label)
        self._layout.insertLayout(3, row)

    def set_landmarks(self, landmark_order: list[str]) -> None:
        """Rebuild heatmap thumbnail slots for a new set of landmarks."""
        # Remove old widgets
        for name in self._landmark_order:
            self._headers[name].deleteLater()
            self._pred_labels[name].deleteLater()
        self._headers.clear()
        self._pred_labels.clear()
        self._selected = None

        self._landmark_order = landmark_order
        self._qcolors = _make_qcolors(landmark_order)

        # Insert new widgets before the stretch
        insert_idx = self._layout.count() - 1  # before stretch
        for name in landmark_order:
            display = _make_display_name(name)
            color = self._qcolors.get(name, QColor(200, 200, 200))

            header = QLabel(f"<b style='color:{color.name()}'>{display}</b>")
            header.setStyleSheet("padding-top: 6px;")
            self._layout.insertWidget(insert_idx, header)
            self._headers[name] = header
            insert_idx += 1

            lbl = _ClickableLabel(name)
            lbl.setText("No model")
            lbl.setAlignment(Qt.AlignCenter)
            lbl.setFixedSize(HEATMAP_THUMB_W, HEATMAP_THUMB_H)
            lbl.setStyleSheet(self._STYLE_NORMAL)
            lbl.clicked.connect(self._on_label_clicked)
            self._layout.insertWidget(insert_idx, lbl)
            self._pred_labels[name] = lbl
            insert_idx += 1

    def _on_label_clicked(self, name: str) -> None:
        """Handle click on a heatmap thumbnail — toggle selection."""
        if self._selected == name:
            self._selected = None
        else:
            self._selected = name
        # Update border highlights
        for n, lbl in self._pred_labels.items():
            lbl.setStyleSheet(self._STYLE_SELECTED if n == self._selected else self._STYLE_NORMAL)
        self.heatmap_clicked.emit(self._selected or "")

    @staticmethod
    def _draw_cross(pixmap: QPixmap, x: int, y: int, color: QColor, arm: int = 8) -> QPixmap:
        """Draw a cross marker on a pixmap and return it."""
        pm = QPixmap(pixmap)
        p = QPainter(pm)
        p.setPen(color)
        p.drawLine(x - arm, y, x + arm, y)
        p.drawLine(x, y - arm, x, y + arm)
        p.end()
        return pm

    def _refresh(self) -> None:
        """Redraw thumbnails with current toggle state."""
        self.update_heatmaps(self._cur_pred_heatmaps, self._cur_gt_coords)

    def update_heatmaps(
        self,
        pred_heatmaps: Optional[np.ndarray],
        gt_coords: Optional[dict[str, tuple[float, float]]],
    ) -> None:
        """Update heatmap thumbnails. gt_coords are in model resolution (MODEL_W x MODEL_H)."""
        self._cur_pred_heatmaps = pred_heatmaps
        self._cur_gt_coords = gt_coords

        for i, name in enumerate(self._landmark_order):
            if name not in self._pred_labels:
                continue
            lbl = self._pred_labels[name]
            if pred_heatmaps is not None and i < pred_heatmaps.shape[0]:
                pm = heatmap_to_pixmap(pred_heatmaps[i])
                # Draw GT cross if enabled and available
                if self._show_gt.isChecked() and gt_coords and name in gt_coords:
                    gx, gy = gt_coords[name]
                    tx = int(gx * HEATMAP_THUMB_W / MODEL_W)
                    ty = int(gy * HEATMAP_THUMB_H / MODEL_H)
                    color = self._qcolors.get(name, QColor(255, 255, 255))
                    pm = self._draw_cross(pm, tx, ty, color)
                lbl.setPixmap(pm)
                lbl.setText("")
            else:
                lbl.clear()
                lbl.setText("No model")


# ---------------------------------------------------------------------------
# Training support: stdout capture, worker thread, log dialog
# ---------------------------------------------------------------------------
class _StdoutCapture(io.TextIOBase):
    """Redirect sys.stdout writes to a Qt signal, throttled to avoid GUI flood."""

    _INTERVAL = 0.15  # seconds between emissions

    def __init__(self, signal: pyqtSignal):
        super().__init__()
        self._signal = signal
        self._original = sys.stdout
        self._buffer: list[str] = []
        self._last_emit = 0.0

    def write(self, text: str) -> int:
        if not text:
            return 0
        if text.strip():
            self._buffer.append(text)
        now = time.monotonic()
        if now - self._last_emit >= self._INTERVAL and self._buffer:
            self._signal.emit("\n".join(self._buffer))
            self._buffer.clear()
            self._last_emit = now
        return len(text)

    def flush(self) -> None:
        if self._buffer:
            self._signal.emit("\n".join(self._buffer))
            self._buffer.clear()
            self._last_emit = time.monotonic()


class TrainingThread(QThread):
    """Background worker that trains all K folds of cross-validation."""

    progress = pyqtSignal(str)
    epoch_data = pyqtSignal(object)  # dict with epoch, mean_error, landmark_errors, fold
    finished_training = pyqtSignal(str)  # path to the run folder containing best_fold*.pt
    error = pyqtSignal(str)

    def __init__(self, model_name: str, gt_dir: Optional[Path] = None, parent=None):
        super().__init__(parent)
        self._model_name = model_name
        self._gt_dir = gt_dir

    def run(self) -> None:
        """Execute full K-fold CV training into trained_models/<model_name>/."""
        capture = _StdoutCapture(self.progress)
        old_stdout = sys.stdout
        sys.stdout = capture
        try:
            from landmark_locator.training.train import (
                _populate_landmark_config,
                create_cv_splits,
                get_device,
                log_config_summary,
                train_fold,
            )

            config_path = _project_root / "configs" / "default.yaml"
            with open(config_path) as f:
                cfg = yaml.safe_load(f)

            # Each run goes into its own named subfolder so folds stay grouped.
            run_dir = _project_root / "trained_models" / self._model_name
            run_dir.mkdir(parents=True, exist_ok=True)

            device = get_device()
            print(f"Using device: {device}")

            if self._gt_dir:
                annotation_dir = self._gt_dir
                cfg["data"]["annotation_dir"] = str(annotation_dir)
            else:
                annotation_dir = Path(cfg["data"]["annotation_dir"])
                if not annotation_dir.is_absolute():
                    annotation_dir = _project_root / annotation_dir

            _populate_landmark_config(cfg, annotation_dir)
            log_config_summary(cfg)
            splits = create_cv_splits(annotation_dir, cfg["cv"]["n_folds"])

            current_fold = {"value": 0}

            def _on_epoch(epoch, mean_error, landmark_errors, train_loss, val_loss):
                self.epoch_data.emit(
                    {
                        "epoch": epoch,
                        "mean_error": mean_error,
                        "landmark_errors": landmark_errors.copy(),
                        "train_loss": train_loss,
                        "val_loss": val_loss,
                        "fold": current_fold["value"],
                    }
                )

            for fold_idx, (train_idx, val_idx) in enumerate(splits):
                current_fold["value"] = fold_idx
                print()
                print("=" * 60)
                print(f"{self._model_name}_Fold{fold_idx}: {len(train_idx)} train, {len(val_idx)} val")
                print("=" * 60)
                train_fold(
                    cfg,
                    fold_idx,
                    train_idx,
                    val_idx,
                    run_dir,
                    device,
                    epoch_callback=_on_epoch,
                    checkpoint_name=None,  # use default best_fold{N}.pt naming so folds don't overwrite
                    interactive=False,
                    display_name=self._model_name,
                )

            # Flat layout: checkpoints live directly in run_dir.
            self.finished_training.emit(str(run_dir))
        except Exception as e:
            self.error.emit(str(e))
        finally:
            capture.flush()
            sys.stdout = old_stdout


class SweepRunnerThread(QThread):
    """Background worker that runs `python -m landmark_locator.scripts.sweep_aug` as a subprocess.

    Emits each stdout line via `progress`. Emits `finished_sweep` with the subprocess
    return code on completion.
    """

    progress = pyqtSignal(str)
    finished_sweep = pyqtSignal(int)
    error = pyqtSignal(str)

    def __init__(
        self,
        rotations: list[int],
        vflips: list[float],
        all_folds: bool,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self._rotations = rotations
        self._vflips = vflips
        self._all_folds = all_folds
        self._proc: Optional[subprocess.Popen] = None
        self._stop_requested = False

    def run(self) -> None:
        cmd = [
            sys.executable,
            "-u",
            "-m",
            "landmark_locator.scripts.sweep_aug",
            "--rotations",
            *[str(r) for r in self._rotations],
            "--vflips",
            *[str(v) for v in self._vflips],
        ]
        if self._all_folds:
            cmd.append("--all-folds")
        try:
            self._proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )
        except Exception as e:
            self.error.emit(f"Failed to start sweep: {e}")
            return
        assert self._proc.stdout is not None
        try:
            for line in self._proc.stdout:
                if self._stop_requested:
                    break
                self.progress.emit(line.rstrip("\n"))
            rc = self._proc.wait()
        except Exception as e:
            self.error.emit(str(e))
            return
        self.finished_sweep.emit(rc)

    def stop(self) -> None:
        self._stop_requested = True
        if self._proc and self._proc.poll() is None:
            try:
                self._proc.terminate()
            except Exception:
                pass


class SweepDialog(QDialog):
    """Configure and launch an augmentation sweep, streaming live progress to a log view."""

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Augmentation Sweep")
        self.resize(820, 620)
        self._thread: Optional[SweepRunnerThread] = None

        root = QVBoxLayout(self)
        hint = QLabel(
            "Trains every combination of rotation_limit × vertical_flip_p in sequence, then "
            "ranks them by validation pixel error. Fold 0 only is fast (~30 min/combo on MPS). "
            "All folds is the full 5-fold CV (~5× slower)."
        )
        hint.setWordWrap(True)
        hint.setStyleSheet("color: #aaa;")
        root.addWidget(hint)

        # Inputs
        form = QGridLayout()
        form.addWidget(QLabel("Rotation values (comma-separated):"), 0, 0)
        self._rot_edit = QLineEdit("30, 45, 60")
        form.addWidget(self._rot_edit, 0, 1)
        form.addWidget(QLabel("Vertical-flip probabilities (comma-separated):"), 1, 0)
        self._vflip_edit = QLineEdit("0.0, 0.25")
        form.addWidget(self._vflip_edit, 1, 1)
        form.addWidget(QLabel("Folds:"), 2, 0)
        self._folds_combo = QComboBox()
        self._folds_combo.addItem("Fold 0 only (fast triage)", False)
        self._folds_combo.addItem("All 5 folds (full CV)", True)
        form.addWidget(self._folds_combo, 2, 1)
        root.addLayout(form)

        # Output log
        self._log = QTextEdit()
        self._log.setReadOnly(True)
        self._log.setStyleSheet("background: #111; color: #ddd; font-family: monospace; font-size: 11px;")
        root.addWidget(self._log, stretch=1)

        # Buttons
        btn_row = QHBoxLayout()
        self._start_btn = QPushButton("Start Sweep")
        self._start_btn.clicked.connect(self._on_start)
        btn_row.addWidget(self._start_btn)
        self._stop_btn = QPushButton("Stop")
        self._stop_btn.setEnabled(False)
        self._stop_btn.clicked.connect(self._on_stop)
        btn_row.addWidget(self._stop_btn)
        btn_row.addStretch(1)
        self._close_btn = QPushButton("Close")
        self._close_btn.clicked.connect(self.reject)
        btn_row.addWidget(self._close_btn)
        root.addLayout(btn_row)

    def _parse_list(self, text: str, kind) -> list:
        out = []
        for part in text.split(","):
            part = part.strip()
            if not part:
                continue
            try:
                out.append(kind(part))
            except ValueError:
                raise ValueError(f"'{part}' is not a valid {kind.__name__}")
        return out

    def _on_start(self) -> None:
        try:
            rotations = self._parse_list(self._rot_edit.text(), int)
            vflips = self._parse_list(self._vflip_edit.text(), float)
        except ValueError as e:
            QMessageBox.warning(self, "Invalid input", str(e))
            return
        if not rotations or not vflips:
            QMessageBox.warning(self, "Invalid input", "Provide at least one rotation and one vflip value.")
            return
        all_folds = bool(self._folds_combo.currentData())

        n_combos = len(rotations) * len(vflips)
        n_folds = 5 if all_folds else 1
        self._log.append(f"Starting sweep: {n_combos} combos × {n_folds} fold(s)")
        self._log.append(f"  rotations: {rotations}")
        self._log.append(f"  vflips:    {vflips}")
        self._log.append("")

        self._start_btn.setEnabled(False)
        self._stop_btn.setEnabled(True)
        self._rot_edit.setEnabled(False)
        self._vflip_edit.setEnabled(False)
        self._folds_combo.setEnabled(False)

        self._thread = SweepRunnerThread(rotations, vflips, all_folds, self)
        self._thread.progress.connect(self._on_progress)
        self._thread.finished_sweep.connect(self._on_finished)
        self._thread.error.connect(self._on_error)
        self._thread.start()

    def _on_stop(self) -> None:
        if self._thread is not None:
            self._log.append("\n[GUI] Stop requested — terminating sweep subprocess…")
            self._thread.stop()
            self._stop_btn.setEnabled(False)

    def _on_progress(self, line: str) -> None:
        self._log.append(line)
        self._log.verticalScrollBar().setValue(self._log.verticalScrollBar().maximum())

    def _on_finished(self, rc: int) -> None:
        self._log.append(f"\n[GUI] Sweep finished (rc={rc}).")
        self._stop_btn.setEnabled(False)
        self._start_btn.setEnabled(True)
        self._rot_edit.setEnabled(True)
        self._vflip_edit.setEnabled(True)
        self._folds_combo.setEnabled(True)

    def _on_error(self, msg: str) -> None:
        self._log.append(f"\n[GUI] Error: {msg}")
        self._stop_btn.setEnabled(False)
        self._start_btn.setEnabled(True)


class TrainingDialog(QDialog):
    """Modal dialog showing live training log output and error chart."""

    def __init__(self, model_name: str = "", landmark_order: list[str] | None = None, max_epochs: int = 0, parent=None):
        super().__init__(parent)
        self._model_name = model_name
        title = f"Training — {model_name}" if model_name else "Training"
        self.setWindowTitle(title)
        self.resize(820, 600)
        layout = QVBoxLayout(self)

        self._landmark_order = landmark_order or []
        self._qcolors = _make_qcolors(self._landmark_order)
        self._max_epochs = max_epochs

        # Epoch counter label
        self._epoch_label = QLabel(f"Epoch 0/{max_epochs}" if max_epochs else "Epoch 0")
        self._epoch_label.setStyleSheet("color: #ddd; font-size: 12px; font-weight: bold; padding: 2px;")
        layout.addWidget(self._epoch_label)

        # Per-landmark data series (must init before _setup_chart)
        self._epochs: list[int] = []
        self._series: dict[str, list[float]] = {name: [] for name in self._landmark_order}
        self._mean_series: list[float] = []
        self._train_loss_series: list[float] = []
        self._val_loss_series: list[float] = []
        self._lines: dict[str, object] = {}
        self._last_chart_draw = 0.0
        self._chart_dirty = False

        # Matplotlib chart with navigation toolbar for pan/zoom
        self._fig = Figure(figsize=(7, 3), facecolor="#1e1e1e")
        self._ax_error = self._fig.add_subplot(121)
        self._ax_loss = self._fig.add_subplot(122)
        self._canvas = FigureCanvasQTAgg(self._fig)
        self._nav_toolbar = NavigationToolbar2QT(self._canvas, self)
        self._nav_toolbar.setStyleSheet("background: #333; border: none;")
        layout.addWidget(self._nav_toolbar)
        layout.addWidget(self._canvas, stretch=2)
        self._user_zoomed = False
        self._setup_chart()

        # Log output
        self._log = QTextEdit()
        self._log.setReadOnly(True)
        self._log.setStyleSheet("background: #1e1e1e; color: #ccc; font-family: monospace;")
        layout.addWidget(self._log, stretch=1)

        self._close_btn = QPushButton("Close")
        self._close_btn.setEnabled(False)
        self._close_btn.clicked.connect(self.accept)
        layout.addWidget(self._close_btn)

    def _setup_chart(self) -> None:
        """Configure the chart axes and style."""

        def _style_axis(ax, title, ylabel):
            ax.set_facecolor("#252526")
            ax.set_xlabel("Epoch", color="#aaa", fontsize=9)
            ax.set_ylabel(ylabel, color="#aaa", fontsize=9)
            ax.set_title(title, color="#ddd", fontsize=10)
            ax.tick_params(colors="#888", labelsize=8)
            for spine in ax.spines.values():
                spine.set_color("#444")
            ax.grid(True, color="#333", linewidth=0.5)

        # Left chart: validation error by landmark
        _style_axis(self._ax_error, "Validation Error by Landmark", "Pixel Error")
        for name in self._landmark_order:
            qc = self._qcolors.get(name, QColor(200, 200, 200))
            color = qc.name()
            display = _make_display_name(name)
            (line,) = self._ax_error.plot([], [], color=color, linewidth=1.2, label=display)
            self._lines[name] = line
        (mean_line,) = self._ax_error.plot([], [], color="#ffffff", linewidth=2, linestyle="--", label="Mean")
        self._lines["_mean"] = mean_line
        self._ax_error.legend(loc="upper right", fontsize=7, facecolor="#333", edgecolor="#555", labelcolor="#ccc")

        # Right chart: train/val loss
        _style_axis(self._ax_loss, "Loss", "MSE Loss")
        (train_line,) = self._ax_loss.plot([], [], color="#4ec9b0", linewidth=1.5, label="Train")
        (val_line,) = self._ax_loss.plot([], [], color="#ce9178", linewidth=1.5, label="Val")
        self._lines["_train_loss"] = train_line
        self._lines["_val_loss"] = val_line
        self._ax_loss.legend(loc="upper right", fontsize=7, facecolor="#333", edgecolor="#555", labelcolor="#ccc")

        self._fig.tight_layout()

        # Track when user manually zooms/pans so we stop auto-scaling
        self._canvas.mpl_connect("button_press_event", lambda e: setattr(self, "_user_zoomed", True))

    def update_chart(self, data: dict) -> None:
        """Add one epoch's data and redraw the chart (throttled to ~2 Hz)."""
        epoch = data["epoch"]
        self._epochs.append(epoch)
        if self._max_epochs:
            self._epoch_label.setText(f"Epoch {epoch + 1}/{self._max_epochs}")
        else:
            self._epoch_label.setText(f"Epoch {epoch + 1}")
        self._mean_series.append(data["mean_error"])
        self._train_loss_series.append(data.get("train_loss", 0.0))
        self._val_loss_series.append(data.get("val_loss", 0.0))
        for name in self._landmark_order:
            if name not in self._series:
                self._series[name] = []
            self._series[name].append(data["landmark_errors"].get(name, 0.0))

        # Update error chart lines
        for name in self._landmark_order:
            self._lines[name].set_data(self._epochs, self._series[name])
        self._lines["_mean"].set_data(self._epochs, self._mean_series)

        # Update loss chart lines
        self._lines["_train_loss"].set_data(self._epochs, self._train_loss_series)
        self._lines["_val_loss"].set_data(self._epochs, self._val_loss_series)

        # Auto-rescale unless user has manually zoomed/panned
        if not self._user_zoomed:
            xmax = max(self._epochs[-1], 1)

            self._ax_error.set_xlim(0, xmax)
            all_vals = self._mean_series + [v for s in self._series.values() for v in s]
            if all_vals:
                ymax = max(all_vals) * 1.1
                self._ax_error.set_ylim(0, max(ymax, 1))

            self._ax_loss.set_xlim(0, xmax)
            all_loss = self._train_loss_series + self._val_loss_series
            if all_loss:
                ymax = max(all_loss) * 1.1
                self._ax_loss.set_ylim(0, max(ymax, 1e-6))

        # Throttle redraws — matplotlib canvas draws are expensive
        now = time.monotonic()
        if now - self._last_chart_draw >= 0.5:
            self._canvas.draw_idle()
            self._last_chart_draw = now
            self._chart_dirty = False
        else:
            self._chart_dirty = True

    def append_log(self, text: str) -> None:
        """Append a line to the log view."""
        self._log.append(text)
        self._log.verticalScrollBar().setValue(self._log.verticalScrollBar().maximum())

    def enable_close(self) -> None:
        """Enable the close button after training completes."""
        if self._chart_dirty:
            self._canvas.draw_idle()
            self._chart_dirty = False
        self._close_btn.setEnabled(True)


# ---------------------------------------------------------------------------
# Gate-result overlay
# ---------------------------------------------------------------------------
def _draw_gate_overlay(
    image: np.ndarray,
    prediction: dict,
    landmark_order: list[str],
    size_scale: float = 1.0,
) -> np.ndarray:
    """Ring each predicted landmark in green (pass) or red (fail).

    `size_scale` mirrors the one passed to draw_landmarks_on_image so the ring stays
    consistent with the dot size at any user-chosen overlay size.
    """
    vis = image.copy()
    landmarks = prediction.get("landmarks", {})
    reliable = prediction.get("reliable", {})

    h, w = vis.shape[:2]
    base = min(h, w)
    # Quadratic response matches draw_landmarks_on_image so the ring grows and
    # shrinks in lockstep with the dot it's framing. Base divisors mirror that
    # function's so the gate ring stays a fixed proportion bigger than the dot.
    dot_scale = size_scale * size_scale
    dot_radius = max(5, int(base / 125 * dot_scale))
    ring_radius = dot_radius + max(2, int(base / 250 * dot_scale))
    thick = max(2, int(round(base / 400.0 * dot_scale)))

    color_ok = (0, 220, 0)
    color_fail = (0, 0, 255)

    for name in landmark_order:
        pt = landmarks.get(name)
        if pt is None:
            continue
        x, y = int(pt[0]), int(pt[1])
        ok = reliable.get(name, True)
        cv2.circle(vis, (x, y), ring_radius, color_ok if ok else color_fail, thick, cv2.LINE_AA)
    return vis


# ---------------------------------------------------------------------------
# Confidence gate editor dialog
# ---------------------------------------------------------------------------
# Tier names as they appear in the GUI dropdown. The order is the display order
# and "Standard" is the default ground-state when no override matches a named
# tier. All concrete threshold values come from the loaded gate_config.yaml's
# `tiers:` block — there are no hardcoded fallbacks. A model whose sidecar lacks
# a named tier block will leave the corresponding dropdown option dimmed-but-
# selectable, falling back to the active-gate globals so the editor still works.
_TIER_NAMES = ("Standard", "Permissive", "Strict")
_DEFAULT_TIER = "Standard"


class GateConfigPanel(QWidget):
    """Reusable panel with per-landmark tier + threshold editor and YAML import/export.

    Embeddable in any QDialog or QTabWidget; emits no signals on its own — read state
    via `result_override()` when the host dialog/window accepts.
    """

    def __init__(
        self,
        gate_config: dict,
        landmark_order: list[str],
        parent=None,
        *,
        display_names: dict[str, str] | None = None,
        sidecar_path: Path | None = None,
    ) -> None:
        super().__init__(parent)
        self._landmark_order = list(landmark_order)
        # Optional friendly labels keyed by internal name (e.g. "acv_a" → "ACV-L3 junction").
        # Falls back to the internal name when a mapping is missing for a given landmark.
        self._display_names = dict(display_names or {})
        # Deep-copy so the host can revert by discarding the panel.
        self._cfg = json.loads(json.dumps(gate_config))
        self._rows: dict[str, dict] = {}
        # If provided, enables the "Save as Model Default" button which writes the
        # current state to this path (the model-folder sidecar gate config).
        self._sidecar_path = sidecar_path

        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(4)

        # Global "minimum metric failures to reject" knob. 1 = today's behavior
        # (any single failed gate rejects the landmark). 2 = a landmark must
        # fail two of its enabled metric gates to be marked unreliable. Lets
        # users tighten individual thresholds without one overstrict gate
        # rejecting otherwise-good landmarks.
        min_fails_row = QHBoxLayout()
        min_fails_row.setContentsMargins(0, 0, 0, 0)
        min_fails_lbl = QLabel("Reject landmark after")
        self._min_fails_spin = QSpinBox()
        self._min_fails_spin.setRange(1, 3)
        self._min_fails_spin.setValue(int(self._cfg.get("min_metric_failures_to_reject", 1)))
        self._min_fails_spin.setToolTip(
            "How many of a landmark's metric gates (peak / sharp / sp_ratio) must fail "
            "before the landmark is marked unreliable.\n\n"
            "1 = any one failure rejects (default, original behavior).\n"
            "2 = need two independent failures — safer when tightening any one gate.\n"
            "3 = all three enabled metrics must fail.\n\n"
            "Disabled gates never count as failures, so setting N higher than the number "
            "of enabled gates effectively disables rejection for that landmark."
        )
        min_fails_suffix = QLabel("failed metric gate(s)")
        min_fails_row.addWidget(min_fails_lbl)
        min_fails_row.addWidget(self._min_fails_spin)
        min_fails_row.addWidget(min_fails_suffix)
        min_fails_row.addStretch(1)
        root.addLayout(min_fails_row)

        # Bulk-tier row: pick a tier and click Apply to set every landmark's
        # tier at once. Avoids hand-stepping each row's combobox when the user
        # wants a global sweep (e.g. "try the whole image at Permissive").
        # "Custom" is intentionally excluded — a global "set everything to
        # Custom" has no meaning since Custom is per-row hand-tuning.
        set_all_row = QHBoxLayout()
        set_all_row.setContentsMargins(0, 0, 0, 0)
        set_all_lbl = QLabel("Set all landmarks to tier:")
        self._set_all_combo = QComboBox()
        self._set_all_combo.addItems(list(_TIER_NAMES))
        self._set_all_combo.setCurrentText(_DEFAULT_TIER)
        self._set_all_combo.setToolTip(
            "Choose a tier to apply to every landmark. Click Apply to overwrite "
            "all per-landmark tiers and threshold values with the chosen tier's "
            "calibrated values. Rows currently set to Custom are also overwritten."
        )
        btn_apply_all = QPushButton("Apply")
        btn_apply_all.setToolTip("Apply the chosen tier to every landmark row.")
        btn_apply_all.clicked.connect(self._on_apply_tier_to_all)
        set_all_row.addWidget(set_all_lbl)
        set_all_row.addWidget(self._set_all_combo)
        set_all_row.addWidget(btn_apply_all)
        set_all_row.addStretch(1)
        root.addLayout(set_all_row)

        grid = QGridLayout()
        grid.setContentsMargins(0, 0, 0, 0)
        grid.setVerticalSpacing(2)
        grid.setHorizontalSpacing(8)
        headers = ["Landmark", "Tier", "peak ≥", "sharp ≥", "sp_ratio ≤", "Abort"]
        header_tooltips = {
            "Tier": (
                "Per-landmark confidence gate. 'Standard' is the calibrated default; "
                "'Permissive' admits weaker peaks; 'Strict' clamps tighter thresholds; "
                "'Custom' lets you set each metric."
            ),
            "Abort": "Check to fail the whole image when this landmark misses.",
        }
        # Tooltips explaining each metric, shown on hover of the column header.
        # Surfaces in TRACE's embedded panel as well as the LandmarkLocator GUI.
        metric_header_tooltips = {
            "peak ≥": (
                "Peak — the maximum value of the predicted heatmap at the landmark's "
                "location (0–1).\n\n"
                "Higher peak = the model is more confident something is there. Low peak "
                "means the feature is faint, missing, or occluded.\n\n"
                "A landmark passes this gate when its peak value is ≥ this threshold."
            ),
            "sharp ≥": (
                "Sharpness — how peaked the heatmap is: peak value divided by the mean "
                "value of an 11×11 window around it.\n\n"
                "High sharpness = a single confident spike. Low sharpness = a fuzzy blob "
                "where the model isn't sure exactly where the landmark sits.\n\n"
                "A landmark passes this gate when its sharpness ratio is ≥ this threshold."
            ),
            "sp_ratio ≤": (
                "Second-peak ratio — the next-highest heatmap value (outside a suppression "
                "radius around the main peak) divided by the main peak.\n\n"
                "High ratio = a comparably strong second candidate exists; the model is "
                "torn between two locations. Low ratio = the main peak is unambiguous.\n\n"
                "A landmark passes this gate when its second-peak ratio is ≤ this threshold."
            ),
        }
        # Metric columns get a checkbox to the left of the label so the user can
        # disable an entire gate column without losing its threshold values.
        # Default ON. Stored state lives on the panel and is emitted via
        # `result_override()` as `confidence.<metric>.enabled`.
        self._gate_enabled: dict[str, QCheckBox] = {}
        gate_enabled_tooltip = (
            "Uncheck to disable this gate. Disabled gates can't reject a "
            "landmark regardless of its threshold values (the values stay "
            "saved so re-enabling restores them)."
        )
        metric_by_header = {
            "peak ≥": "peak",
            "sharp ≥": "sharpness",
            "sp_ratio ≤": "second_peak_ratio",
        }
        for col, h in enumerate(headers):
            if h in metric_by_header:
                metric_key = metric_by_header[h]
                cell = QWidget()
                cell_layout = QHBoxLayout(cell)
                cell_layout.setContentsMargins(0, 0, 0, 0)
                cell_layout.setSpacing(4)
                # Tooltip on the cell catches hover over empty space; the label
                # below carries it too so hover over the text itself works
                # without falling through to the cell (Qt doesn't propagate).
                cell.setToolTip(metric_header_tooltips[h])
                chk = QCheckBox()
                chk.setChecked(bool(self._cfg.get(metric_key, {}).get("enabled", True)))
                chk.setToolTip(gate_enabled_tooltip)
                chk.toggled.connect(self._on_gate_enabled_toggled)
                self._gate_enabled[metric_key] = chk
                cell_layout.addWidget(chk)
                lbl = QLabel(h)
                lbl.setStyleSheet("font-weight: bold;")
                lbl.setToolTip(metric_header_tooltips[h])
                cell_layout.addWidget(lbl)
                cell_layout.addStretch(1)
                grid.addWidget(cell, 0, col)
            else:
                lbl = QLabel(h)
                lbl.setStyleSheet("font-weight: bold;")
                if h in header_tooltips:
                    lbl.setToolTip(header_tooltips[h])
                grid.addWidget(lbl, 0, col)

        core = set(self._cfg.get("core_landmarks", []) or [])
        peak_pl = self._cfg.get("peak", {}).get("per_landmark", {}) or {}
        sharp_pl = self._cfg.get("sharpness", {}).get("per_landmark", {}) or {}
        spr_pl = self._cfg.get("second_peak_ratio", {}).get("per_landmark", {}) or {}
        for i, name in enumerate(self._landmark_order, start=1):
            grid.addWidget(QLabel(self._display_names.get(name, name)), i, 0)

            combo = QComboBox()
            combo.addItems([*_TIER_NAMES, "Custom"])
            peak_spin = QDoubleSpinBox()
            peak_spin.setRange(0.0, 1.0)
            peak_spin.setSingleStep(0.05)
            peak_spin.setDecimals(3)
            sharp_spin = QDoubleSpinBox()
            sharp_spin.setRange(0.0, 20.0)
            sharp_spin.setSingleStep(0.1)
            sharp_spin.setDecimals(2)
            spr_spin = QDoubleSpinBox()
            spr_spin.setRange(0.0, 1.0)
            spr_spin.setSingleStep(0.05)
            spr_spin.setDecimals(2)
            abort_chk = QCheckBox()
            abort_chk.setChecked(name in core)

            cur_peak = peak_pl.get(name, self._cfg["peak"]["global"])
            cur_sharp = sharp_pl.get(name, self._cfg["sharpness"]["global"])
            cur_spr = spr_pl.get(name, self._cfg["second_peak_ratio"]["global"])
            peak_spin.setValue(float(cur_peak))
            sharp_spin.setValue(float(cur_sharp))
            spr_spin.setValue(float(cur_spr))

            # Decide the initial tier by matching the active-gate value against the
            # calibrated per-landmark presets. Standard is the default ground state.

            def _close(a: float, b: float, tol: float = 1e-3) -> bool:
                return abs(float(a) - float(b)) <= tol

            matched = "Custom"
            for tier_name in _TIER_NAMES:
                t_peak, t_sharp, t_spr = self._tier_values(tier_name, name)
                if _close(cur_peak, t_peak) and _close(cur_sharp, t_sharp) and _close(cur_spr, t_spr):
                    matched = tier_name
                    break
            combo.setCurrentText(matched)

            combo.currentTextChanged.connect(lambda tier, n=name: self._apply_tier_to_row(n, tier))
            grid.addWidget(combo, i, 1)
            grid.addWidget(peak_spin, i, 2)
            grid.addWidget(sharp_spin, i, 3)
            grid.addWidget(spr_spin, i, 4)
            grid.addWidget(abort_chk, i, 5)

            self._rows[name] = {
                "combo": combo,
                "peak": peak_spin,
                "sharpness": sharp_spin,
                "second_peak_ratio": spr_spin,
                "abort": abort_chk,
            }
            self._sync_row_editability(name)

        root.addLayout(grid)

        buttons_row = QHBoxLayout()
        btn_reset = QPushButton("Reset to Model Defaults")
        btn_reset.setToolTip(
            "Restore every landmark to its model-default (calibrated Standard tier).\n\n"
            "Useful when a stale saved override is masking the values you'd see fresh — "
            "click this, then OK, to bring the active gate back in line with the model's "
            "gate_config.yaml sidecar."
        )
        btn_reset.clicked.connect(self._on_reset_to_model_defaults)
        buttons_row.addWidget(btn_reset)

        btn_load = QPushButton("Import YAML…")
        btn_load.clicked.connect(self._on_import)
        btn_save = QPushButton("Export YAML…")
        btn_save.clicked.connect(self._on_export)
        buttons_row.addWidget(btn_load)
        buttons_row.addWidget(btn_save)

        # The "Save as Model Default" button only renders when the host provides a
        # sidecar path. LandmarkLocator GUI (the model-curation tool) passes it;
        # TRACE (the per-project settings UI) does not, since gate edits there are
        # persisted with the rest of the TRACE settings — the sidecar remains the
        # safe fallback that "Restore Defaults" / settings wipe falls back to.
        if self._sidecar_path is not None:
            btn_save_default = QPushButton("Save as Model Default")
            btn_save_default.setToolTip(
                f"Write the current gate config to:\n{self._sidecar_path}\n\n"
                "This file is auto-loaded whenever the model is loaded — by this GUI, "
                "TRACE, or landmark-predict on the CLI."
            )
            btn_save_default.clicked.connect(self._on_save_as_model_default)
            buttons_row.addWidget(btn_save_default)
        buttons_row.addStretch(1)
        root.addLayout(buttons_row)

    def _tier_values(self, tier: str, name: str) -> tuple[float, float, float]:
        """Return (peak, sharpness, second_peak_ratio) for `name` under the given tier.

        Looks up calibrated per-landmark thresholds in the loaded config's
        `tiers.{standard|permissive|strict}` block. Falls back to that tier's
        own globals, then to the active gate's globals when a tier block is
        absent. There are no hardcoded constants — a model with no `tiers:`
        block at all will just echo the active gate's values for every tier.
        """
        tier_block = (self._cfg.get("tiers") or {}).get(tier.lower()) or {}

        def _pick(metric_key: str) -> float:
            metric = tier_block.get(metric_key) or {}
            pl = metric.get("per_landmark") or {}
            if name in pl:
                return float(pl[name])
            if "global" in metric:
                return float(metric["global"])
            # Active-gate globals are the last layer.
            active = self._cfg.get(metric_key) or {}
            if "global" in active:
                return float(active["global"])
            return 0.0

        return (_pick("peak"), _pick("sharpness"), _pick("second_peak_ratio"))

    def _apply_tier_to_row(self, name: str, tier: str) -> None:
        row = self._rows[name]
        if tier in _TIER_NAMES:
            peak_v, sharp_v, spr_v = self._tier_values(tier, name)
            row["peak"].setValue(peak_v)
            row["sharpness"].setValue(sharp_v)
            row["second_peak_ratio"].setValue(spr_v)
        self._sync_row_editability(name)

    def _sync_row_editability(self, name: str) -> None:
        row = self._rows[name]
        custom = row["combo"].currentText() == "Custom"
        for key in ("peak", "sharpness", "second_peak_ratio"):
            # Editable iff the tier is Custom AND this gate column is enabled.
            gate_on = bool(self._gate_enabled[key].isChecked())
            row[key].setEnabled(custom and gate_on)

    def _on_gate_enabled_toggled(self) -> None:
        """Refresh every row's spinbox editability when a gate checkbox flips."""
        for name in self._landmark_order:
            self._sync_row_editability(name)

    def result_override(self) -> dict:
        """Build a confidence-override dict from the current widget state.

        Every landmark gets an explicit per-landmark entry — whether its tier is
        Standard, Permissive, Strict, or Custom — because the calibrated tier
        values are per-landmark (e.g. "Standard" ≠ "use global"; it means "use
        this landmark's standard-calibrated value"). The global stays as the
        safety fallback for any landmark missing from per_landmark.
        """
        peak_pl: dict[str, float] = {}
        sharp_pl: dict[str, float] = {}
        spr_pl: dict[str, float] = {}
        core: list[str] = []
        for name, row in self._rows.items():
            peak_pl[name] = float(row["peak"].value())
            sharp_pl[name] = float(row["sharpness"].value())
            spr_pl[name] = float(row["second_peak_ratio"].value())
            if row["abort"].isChecked():
                core.append(name)
        peak_enabled = bool(self._gate_enabled["peak"].isChecked())
        sharp_enabled = bool(self._gate_enabled["sharpness"].isChecked())
        spr_enabled = bool(self._gate_enabled["second_peak_ratio"].isChecked())
        return {
            "peak": {
                "enabled": peak_enabled,
                "global": self._cfg["peak"]["global"],
                "per_landmark": peak_pl,
            },
            "sharpness": {
                "enabled": sharp_enabled,
                "global": self._cfg["sharpness"]["global"],
                "per_landmark": sharp_pl,
            },
            "second_peak_ratio": {
                "enabled": spr_enabled,
                "global": self._cfg["second_peak_ratio"]["global"],
                "per_landmark": spr_pl,
            },
            "second_peak_suppression_radius_px": self._cfg.get("second_peak_suppression_radius_px", 30),
            "core_landmarks": sorted(core),
            "min_metric_failures_to_reject": int(self._min_fails_spin.value()),
        }

    def _on_export(self) -> None:
        path = _pick_save_file_native("Export gate YAML", "", "YAML (*.yaml *.yml)")
        if not path:
            return
        doc = {"confidence": self.result_override()}
        Path(path).write_text(yaml.safe_dump(doc, sort_keys=False))
        QMessageBox.information(self, "Exported", f"Wrote {path}")

    def _on_reset_to_model_defaults(self) -> None:
        """Snap every landmark back to the Standard tier (calibrated model defaults).

        Useful when a host has merged a stale per-user override on top of the panel's
        initial config — TRACE's `gate_override_json` from a previous session, for
        instance. After this, clicking OK persists the calibrated values, replacing
        the stale override.
        """
        for name in self._landmark_order:
            self._rows[name]["combo"].setCurrentText(_DEFAULT_TIER)
            # If the combo was already on the default tier, currentTextChanged didn't
            # fire — force-apply so the spinboxes refresh regardless.
            self._apply_tier_to_row(name, _DEFAULT_TIER)

    def _on_apply_tier_to_all(self) -> None:
        """Apply the bulk-tier combo's current selection to every landmark row."""
        tier = self._set_all_combo.currentText()
        if tier not in _TIER_NAMES:
            return
        for name in self._landmark_order:
            self._rows[name]["combo"].setCurrentText(tier)
            # currentTextChanged is a no-op when the row was already on `tier`;
            # force-apply so spinboxes refresh in that case too.
            self._apply_tier_to_row(name, tier)

    def _on_save_as_model_default(self) -> None:
        """Write the current gate to the model's sidecar YAML (auto-loaded on model load).

        Preserves the `tiers:` block (calibrated permissive/strict presets) from the
        existing sidecar so a user's edit to the active gate doesn't wipe the named
        tier definitions the GUI dropdown relies on.
        """
        if self._sidecar_path is None:
            QMessageBox.warning(self, "No model", "Load a model first.")
            return
        target = Path(self._sidecar_path)

        # Read the existing sidecar (if any) so we can carry forward the tiers block.
        existing_conf: dict = {}
        if target.exists():
            try:
                data = yaml.safe_load(target.read_text()) or {}
                existing_conf = data.get("confidence", {}) or {}
            except Exception:
                existing_conf = {}

        new_conf = self.result_override()
        if "tiers" in existing_conf:
            new_conf["tiers"] = existing_conf["tiers"]

        try:
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(yaml.safe_dump({"confidence": new_conf}, sort_keys=False))
        except Exception as e:
            QMessageBox.critical(self, "Save failed", f"Could not write {target}:\n{e}")
            return
        QMessageBox.information(
            self,
            "Saved as model default",
            f"Wrote {target}.\n\nThis gate will be loaded automatically whenever this "
            "model is used — in this GUI, TRACE, or via landmark-predict.",
        )

    def _on_import(self) -> None:
        path = _pick_file_native("Import gate YAML", "", "YAML (*.yaml *.yml)")
        if not path:
            return
        try:
            data = yaml.safe_load(Path(path).read_text()) or {}
        except Exception as e:
            QMessageBox.warning(self, "Import failed", str(e))
            return
        override = data.get("confidence", data)
        peak_pl = override.get("peak", {}).get("per_landmark", {}) or {}
        sharp_pl = override.get("sharpness", {}).get("per_landmark", {}) or {}
        spr_pl = override.get("second_peak_ratio", {}).get("per_landmark", {}) or {}
        core = set(override.get("core_landmarks", []) or [])
        # Gate-column enabled flags (default True for files written before this flag existed).
        self._gate_enabled["peak"].setChecked(bool(override.get("peak", {}).get("enabled", True)))
        self._gate_enabled["sharpness"].setChecked(bool(override.get("sharpness", {}).get("enabled", True)))
        self._gate_enabled["second_peak_ratio"].setChecked(
            bool(override.get("second_peak_ratio", {}).get("enabled", True))
        )
        # min_metric_failures_to_reject defaults to 1 for files written before this field existed.
        self._min_fails_spin.setValue(int(override.get("min_metric_failures_to_reject", 1)))
        for name, row in self._rows.items():
            if name in peak_pl or name in sharp_pl or name in spr_pl:
                row["combo"].setCurrentText("Custom")
                row["peak"].setValue(float(peak_pl.get(name, row["peak"].value())))
                row["sharpness"].setValue(float(sharp_pl.get(name, row["sharpness"].value())))
                row["second_peak_ratio"].setValue(float(spr_pl.get(name, row["second_peak_ratio"].value())))
            else:
                row["combo"].setCurrentText(_DEFAULT_TIER)
            row["abort"].setChecked(name in core)
            self._sync_row_editability(name)


def _find_gate_yaml(folder: Path) -> Optional[Path]:
    """Locate the gate-config YAML in `folder` with tiered name matching.

    Priority (highest → lowest):
      1. Exact stem match: `gate_config.yaml` or `gate_config.yml`.
      2. Prefix match: stem starts with `gate_config` (e.g. `gate_config_0p1.yaml`).
      3. Substring match: stem contains `gate_config` anywhere.
      4. Any `.yaml`/`.yml` file in the folder.
    Ties within a tier are broken by shortest stem (closer to the canonical
    name) then alphabetically.
    """
    if not folder.is_dir():
        return None
    yaml_files = sorted(
        [f for f in folder.iterdir() if f.is_file() and f.suffix.lower() in (".yaml", ".yml")],
        key=lambda f: (len(f.stem), f.name.lower()),
    )
    if not yaml_files:
        return None

    def _exact(f: Path) -> bool:
        return f.stem.lower() == "gate_config"

    def _prefix(f: Path) -> bool:
        return f.stem.lower().startswith("gate_config")

    def _contains(f: Path) -> bool:
        return "gate_config" in f.stem.lower()

    for predicate in (_exact, _prefix, _contains):
        match = next((f for f in yaml_files if predicate(f)), None)
        if match is not None:
            return match
    return yaml_files[0]


def read_gate_config(path: Path) -> tuple[dict, list[str]]:
    """Read the sidecar `gate_config.yaml` for a model and return (gate_config, landmark_order).

    `path` may be either a model folder or a specific checkpoint file. Used by
    UI callers (TRACE settings dialog, LandmarkLocator GUI) that want to
    populate a `GateConfigPanel` without instantiating a full predictor.

    The YAML alone is sufficient because:
      - Gate thresholds live in the sidecar (the previously-used
        checkpoint-embedded `config.confidence` block is no longer consulted).
      - `landmark_order` is derived from the YAML's `confidence.peak.per_landmark`
        keys. For UI display purposes any stable order works; if a caller
        needs the model's authoritative training order it should pull it
        from a live `Predictor` instance instead.
    """
    import yaml

    from landmark_locator.inference.predict import DEFAULT_GATE_CONFIG, _deep_merge

    p = Path(path).resolve()
    # Search for the sidecar YAML in two folders: alongside the model files
    # (or inside the model folder), and the parent folder (legacy nested
    # layout: <model>/checkpoints/best_fold*.pt + <model>/gate_config.yaml).
    search_dirs = [
        p if p.is_dir() else p.parent,
        p.parent if p.is_dir() else p.parent.parent,
    ]
    sidecar = None
    for d in search_dirs:
        match = _find_gate_yaml(d)
        if match is not None:
            sidecar = match
            break
    if sidecar is None:
        # No sidecar YAML — return library defaults so callers can still
        # populate a UI editor (the user will see fallback values instead
        # of a red error). Callers that strictly require the model's
        # calibrated values should check `landmark_order` for emptiness.
        return dict(DEFAULT_GATE_CONFIG), []

    data = yaml.safe_load(sidecar.read_text()) or {}
    # Accept both shapes: full doc with `confidence:` block, or just the block itself.
    sidecar_block = data.get("confidence", data)
    gate_config = _deep_merge(DEFAULT_GATE_CONFIG, sidecar_block or {})

    # Derive landmark_order from the YAML — every per-landmark block lists
    # every landmark, so taking the keys of `peak.per_landmark` gives a
    # stable display order that matches whatever the calibration script
    # emitted.
    pl = (gate_config.get("peak") or {}).get("per_landmark") or {}
    landmark_order = list(pl.keys())
    return gate_config, landmark_order


# Back-compat alias for older callers that still import the previous name.
read_gate_config_from_checkpoint = read_gate_config


class GateConfigDialog(QDialog):
    """Modal wrapper around `GateConfigPanel` for the LandmarkLocator inspection GUI."""

    def __init__(
        self,
        parent: "LandmarkGUI",
        gate_config: dict,
        landmark_order: list[str],
        sidecar_path: Path | None = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Confidence Gate Configuration")
        self.resize(820, 500)

        root = QVBoxLayout(self)
        self._panel = GateConfigPanel(gate_config, landmark_order, self, sidecar_path=sidecar_path)
        root.addWidget(self._panel)

        bb = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        bb.accepted.connect(self.accept)
        bb.rejected.connect(self.reject)
        root.addWidget(bb)

    def result_override(self) -> dict:
        return self._panel.result_override()


# ---------------------------------------------------------------------------
# Training-config editor + augmentation preview dialogs
# ---------------------------------------------------------------------------
_AUG_PREVIEW_COLORS_BGR = [
    (0, 0, 255),
    (0, 255, 0),
    (255, 0, 0),
    (0, 255, 255),
    (255, 0, 255),
    (255, 255, 0),
    (128, 128, 255),
    (255, 128, 128),
]


def _draw_kps_for_preview(image_rgb: np.ndarray, keypoints, names, landmark_order) -> np.ndarray:
    """Overlay landmark keypoints on an RGB image and return BGR for cv2/Qt display."""
    img = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR).copy()
    color_map = {n: _AUG_PREVIEW_COLORS_BGR[i % len(_AUG_PREVIEW_COLORS_BGR)] for i, n in enumerate(landmark_order)}
    h, w = img.shape[:2]
    for (x, y), name in zip(keypoints, names):
        xi, yi = int(round(x)), int(round(y))
        if not (0 <= xi < w and 0 <= yi < h):
            continue
        color = color_map.get(name, (255, 255, 255))
        cv2.circle(img, (xi, yi), 5, color, -1)
        cv2.circle(img, (xi, yi), 6, (0, 0, 0), 1)
        cv2.putText(img, name[:6], (xi + 7, yi - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1, cv2.LINE_AA)
    return img


def _bgr_to_qpixmap(img_bgr: np.ndarray) -> QPixmap:
    h, w = img_bgr.shape[:2]
    qimg = QImage(img_bgr.data, w, h, 3 * w, QImage.Format_BGR888)
    return QPixmap.fromImage(qimg.copy())


class TrainingConfigDialog(QDialog):
    """Editable form for the most-edited training-config keys in `configs/default.yaml`.

    Edits are applied to an in-memory copy; OK writes back to the YAML file. Cancel
    discards. A 'Preview Augmentations…' button opens a live-preview dialog using the
    current (unsaved) form state.
    """

    def __init__(self, parent: "LandmarkGUI", config_path: Path) -> None:
        super().__init__(parent)
        self.setWindowTitle("Training Configuration")
        self.resize(560, 640)
        self._parent_gui = parent
        self._config_path = config_path
        self._cfg = yaml.safe_load(config_path.read_text()) or {}

        root = QVBoxLayout(self)

        hint = QLabel(
            "Edit the augmentation and training knobs that drive `landmark-train`. "
            "Keys not exposed here can still be edited directly in "
            f"<code>{config_path}</code>."
        )
        hint.setWordWrap(True)
        hint.setStyleSheet("color: #aaa;")
        root.addWidget(hint)

        scroll = QScrollArea(self)
        scroll.setWidgetResizable(True)
        body = QWidget()
        body_layout = QVBoxLayout(body)
        body_layout.setContentsMargins(0, 0, 0, 0)

        aug = self._cfg.setdefault("augmentation", {})
        train = self._cfg.setdefault("training", {})
        opt = train.setdefault("optimizer", {})
        cv_cfg = self._cfg.setdefault("cv", {})

        self._fields: dict[str, QWidget] = {}

        def _add_int(grid, row, label, default, mn, mx, step=1):
            grid.addWidget(QLabel(label), row, 0)
            sb = QSpinBox()
            sb.setRange(mn, mx)
            sb.setSingleStep(step)
            sb.setValue(int(default))
            grid.addWidget(sb, row, 1)
            return sb

        def _add_float(grid, row, label, default, mn, mx, step=0.05, decimals=3):
            grid.addWidget(QLabel(label), row, 0)
            sb = QDoubleSpinBox()
            sb.setRange(mn, mx)
            sb.setSingleStep(step)
            sb.setDecimals(decimals)
            sb.setValue(float(default))
            grid.addWidget(sb, row, 1)
            return sb

        # --- Augmentation group ---
        aug_box = QGroupBox("Augmentation")
        aug_grid = QGridLayout(aug_box)
        aug_grid.setVerticalSpacing(4)
        aug_grid.setHorizontalSpacing(8)
        self._fields["augmentation.rotation_limit"] = _add_int(
            aug_grid, 0, "rotation_limit (±°)", aug.get("rotation_limit", 90), 0, 180, 5
        )
        self._fields["augmentation.horizontal_flip_p"] = _add_float(
            aug_grid, 1, "horizontal_flip_p", aug.get("horizontal_flip_p", 0.5), 0.0, 1.0, 0.05, 2
        )
        self._fields["augmentation.vertical_flip_p"] = _add_float(
            aug_grid, 2, "vertical_flip_p", aug.get("vertical_flip_p", 0.5), 0.0, 1.0, 0.05, 2
        )
        self._fields["augmentation.scale_limit"] = _add_float(
            aug_grid, 3, "scale_limit", aug.get("scale_limit", 0.15), 0.0, 1.0, 0.05, 2
        )
        self._fields["augmentation.blur_p"] = _add_float(
            aug_grid, 4, "blur_p", aug.get("blur_p", 0.3), 0.0, 1.0, 0.05, 2
        )
        self._fields["augmentation.coarse_dropout_p"] = _add_float(
            aug_grid, 5, "coarse_dropout_p", aug.get("coarse_dropout_p", 0.3), 0.0, 1.0, 0.05, 2
        )
        self._fields["augmentation.coarse_dropout_max_holes"] = _add_int(
            aug_grid, 6, "coarse_dropout_max_holes", aug.get("coarse_dropout_max_holes", 4), 0, 50, 1
        )
        self._fields["augmentation.coarse_dropout_max_height"] = _add_int(
            aug_grid, 7, "coarse_dropout_box_height (px)", aug.get("coarse_dropout_max_height", 40), 0, 512, 5
        )
        self._fields["augmentation.coarse_dropout_max_width"] = _add_int(
            aug_grid, 8, "coarse_dropout_box_width (px)", aug.get("coarse_dropout_max_width", 40), 0, 512, 5
        )
        body_layout.addWidget(aug_box)

        # --- Training group ---
        tr_box = QGroupBox("Training")
        tr_grid = QGridLayout(tr_box)
        tr_grid.setVerticalSpacing(4)
        tr_grid.setHorizontalSpacing(8)
        self._fields["training.epochs"] = _add_int(tr_grid, 0, "epochs", train.get("epochs", 300), 10, 5000, 10)
        self._fields["training.batch_size"] = _add_int(tr_grid, 1, "batch_size", train.get("batch_size", 4), 1, 64, 1)
        self._fields["training.encoder_freeze_epochs"] = _add_int(
            tr_grid, 2, "encoder_freeze_epochs", train.get("encoder_freeze_epochs", 20), 0, 500, 5
        )
        self._fields["training.early_stopping_patience"] = _add_int(
            tr_grid, 3, "early_stopping_patience", train.get("early_stopping_patience", 50), 0, 500, 5
        )
        # Learning rate via QLineEdit so scientific notation (1e-3) works cleanly.
        tr_grid.addWidget(QLabel("optimizer.lr"), 4, 0)
        lr_edit = QLineEdit()
        lr_edit.setText(str(opt.get("lr", 1.0e-3)))
        lr_edit.setPlaceholderText("e.g. 1e-3")
        tr_grid.addWidget(lr_edit, 4, 1)
        self._fields["training.optimizer.lr"] = lr_edit
        body_layout.addWidget(tr_box)

        # --- Cross-validation group ---
        cv_box = QGroupBox("Cross-validation")
        cv_grid = QGridLayout(cv_box)
        cv_grid.setVerticalSpacing(4)
        cv_grid.setHorizontalSpacing(8)
        self._fields["cv.n_folds"] = _add_int(cv_grid, 0, "n_folds", cv_cfg.get("n_folds", 5), 2, 10, 1)
        body_layout.addWidget(cv_box)

        body_layout.addStretch(1)
        scroll.setWidget(body)
        root.addWidget(scroll)

        # --- Buttons ---
        action_row = QHBoxLayout()
        preview_btn = QPushButton("Preview Augmentations…")
        preview_btn.clicked.connect(self._on_preview)
        action_row.addWidget(preview_btn)
        sweep_btn = QPushButton("Run Augmentation Sweep…")
        sweep_btn.setToolTip(
            "Train multiple combinations of rotation_limit × vertical_flip_p in sequence "
            "and rank them by validation pixel error. Saves config to default.yaml first."
        )
        sweep_btn.clicked.connect(self._on_sweep)
        action_row.addWidget(sweep_btn)
        action_row.addStretch(1)
        root.addLayout(action_row)

        bb = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel | QDialogButtonBox.Reset)
        bb.accepted.connect(self._on_save_and_close)
        bb.rejected.connect(self.reject)
        bb.button(QDialogButtonBox.Reset).clicked.connect(self._on_reset)
        root.addWidget(bb)

    def _form_into_cfg(self) -> dict:
        """Apply the current form-field values into a copy of self._cfg and return it."""
        cfg = json.loads(json.dumps(self._cfg))
        cfg.setdefault("augmentation", {})
        cfg.setdefault("training", {}).setdefault("optimizer", {})
        cfg.setdefault("cv", {})
        for key, widget in self._fields.items():
            if isinstance(widget, QSpinBox):
                value = int(widget.value())
            elif isinstance(widget, QDoubleSpinBox):
                value = float(widget.value())
            elif isinstance(widget, QLineEdit):
                txt = widget.text().strip()
                try:
                    value = float(txt)
                except ValueError:
                    raise ValueError(f"{key}: '{txt}' is not a valid number")
            else:
                continue
            parts = key.split(".")
            target = cfg
            for part in parts[:-1]:
                target = target.setdefault(part, {})
            target[parts[-1]] = value
        return cfg

    def _on_preview(self) -> None:
        try:
            cfg = self._form_into_cfg()
        except ValueError as e:
            QMessageBox.warning(self, "Invalid value", str(e))
            return
        dlg = AugPreviewDialog(self._parent_gui, cfg, parent=self)
        dlg.exec_()

    def _on_sweep(self) -> None:
        """Save the current config to disk, then open the SweepDialog.

        The sweep subprocess derives its base config from the YAML on disk, so any
        unsaved form edits would be ignored — save first to make the user's intent
        explicit. We do not auto-close this dialog so the user can come back to it.
        """
        try:
            cfg = self._form_into_cfg()
        except ValueError as e:
            QMessageBox.warning(self, "Invalid value", str(e))
            return
        try:
            self._config_path.write_text(yaml.safe_dump(cfg, sort_keys=False))
        except Exception as e:
            QMessageBox.critical(self, "Save failed", f"Could not write {self._config_path}:\n{e}")
            return
        dlg = SweepDialog(parent=self)
        dlg.exec_()

    def _on_save_and_close(self) -> None:
        try:
            cfg = self._form_into_cfg()
        except ValueError as e:
            QMessageBox.warning(self, "Invalid value", str(e))
            return
        try:
            self._config_path.write_text(yaml.safe_dump(cfg, sort_keys=False))
        except Exception as e:
            QMessageBox.critical(self, "Save failed", f"Could not write {self._config_path}:\n{e}")
            return
        self.accept()

    def _on_reset(self) -> None:
        """Re-read the YAML from disk and repopulate the fields."""
        self._cfg = yaml.safe_load(self._config_path.read_text()) or {}
        aug = self._cfg.get("augmentation", {})
        train = self._cfg.get("training", {})
        opt = train.get("optimizer", {})
        cv_cfg = self._cfg.get("cv", {})
        defaults = {
            "augmentation.rotation_limit": aug.get("rotation_limit", 90),
            "augmentation.horizontal_flip_p": aug.get("horizontal_flip_p", 0.5),
            "augmentation.vertical_flip_p": aug.get("vertical_flip_p", 0.5),
            "augmentation.scale_limit": aug.get("scale_limit", 0.15),
            "augmentation.blur_p": aug.get("blur_p", 0.3),
            "augmentation.coarse_dropout_p": aug.get("coarse_dropout_p", 0.3),
            "augmentation.coarse_dropout_max_holes": aug.get("coarse_dropout_max_holes", 4),
            "augmentation.coarse_dropout_max_height": aug.get("coarse_dropout_max_height", 40),
            "augmentation.coarse_dropout_max_width": aug.get("coarse_dropout_max_width", 40),
            "training.epochs": train.get("epochs", 300),
            "training.batch_size": train.get("batch_size", 4),
            "training.encoder_freeze_epochs": train.get("encoder_freeze_epochs", 20),
            "training.early_stopping_patience": train.get("early_stopping_patience", 50),
            "training.optimizer.lr": opt.get("lr", 1.0e-3),
            "cv.n_folds": cv_cfg.get("n_folds", 5),
        }
        for key, value in defaults.items():
            widget = self._fields[key]
            if isinstance(widget, QSpinBox):
                widget.setValue(int(value))
            elif isinstance(widget, QDoubleSpinBox):
                widget.setValue(float(value))
            elif isinstance(widget, QLineEdit):
                widget.setText(str(value))


class AugPreviewDialog(QDialog):
    """Render N augmented training samples with the current train transform.

    Picks an image+annotation from the parent GUI's loaded entries when available,
    otherwise falls back to the first GeoJSON in `cfg.data.annotation_dir`.
    """

    def __init__(self, parent_gui: "LandmarkGUI", cfg: dict, parent=None) -> None:
        super().__init__(parent or parent_gui)
        self.setWindowTitle("Augmentation Preview")
        self.resize(900, 720)
        self._parent_gui = parent_gui
        self._cfg = cfg

        root = QVBoxLayout(self)

        # --- Top controls ---
        top = QHBoxLayout()
        top.addWidget(QLabel("Image:"))
        self._image_combo = QComboBox()
        self._populate_image_choices()
        top.addWidget(self._image_combo, stretch=1)
        top.addWidget(QLabel("N samples:"))
        self._n_spin = QSpinBox()
        self._n_spin.setRange(1, 32)
        self._n_spin.setValue(8)
        top.addWidget(self._n_spin)
        gen_btn = QPushButton("Generate")
        gen_btn.clicked.connect(self._on_generate)
        top.addWidget(gen_btn)
        root.addLayout(top)

        # --- Status / hint ---
        self._status = QLabel("")
        self._status.setStyleSheet("color: #aaa;")
        root.addWidget(self._status)

        # --- Grid scroll area ---
        self._scroll = QScrollArea()
        self._scroll.setWidgetResizable(True)
        self._grid_holder = QWidget()
        self._grid = QGridLayout(self._grid_holder)
        self._grid.setSpacing(6)
        self._scroll.setWidget(self._grid_holder)
        root.addWidget(self._scroll, stretch=1)

        bb = QDialogButtonBox(QDialogButtonBox.Close)
        bb.rejected.connect(self.reject)
        bb.button(QDialogButtonBox.Close).clicked.connect(self.reject)
        root.addWidget(bb)

    def _populate_image_choices(self) -> None:
        """Fill the image picker with parent-GUI entries that have GT, then fall back to training_data."""
        seen: set[str] = set()
        # 1. Parent GUI entries that have a GT annotation
        for entry in getattr(self._parent_gui, "_entries", []) or []:
            if entry.geojson_path and entry.geojson_path.exists():
                key = str(entry.path)
                if key in seen:
                    continue
                seen.add(key)
                self._image_combo.addItem(entry.path.name, (str(entry.path), str(entry.geojson_path)))
        # 2. Fallback: first few annotated images from cfg.data.annotation_dir
        try:
            annotation_dir = Path(self._cfg.get("data", {}).get("annotation_dir", "training_data"))
            if not annotation_dir.is_absolute():
                annotation_dir = _project_root / annotation_dir
            image_dir = Path(self._cfg.get("data", {}).get("image_dir", "training_data_pics"))
            if not image_dir.is_absolute():
                image_dir = _project_root / image_dir
            for gj in sorted(annotation_dir.glob("*.geojson"))[:20]:
                img_path = image_dir / gj.stem
                if not img_path.exists():
                    continue
                key = str(img_path)
                if key in seen:
                    continue
                seen.add(key)
                self._image_combo.addItem(f"{img_path.name}  (training set)", (str(img_path), str(gj)))
        except Exception:
            pass

    def _clear_grid(self) -> None:
        while self._grid.count():
            item = self._grid.takeAt(0)
            w = item.widget()
            if w is not None:
                w.deleteLater()

    def _on_generate(self) -> None:
        import albumentations as A

        from landmark_locator.data.augmentation import get_train_transform
        from landmark_locator.data.psd_loader import imread_any

        data = self._image_combo.currentData()
        if not data:
            self._status.setText("No image available — load images or annotate one first.")
            return
        image_path = Path(data[0])
        geojson_path = Path(data[1]) if data[1] else None

        # Resolve landmark order: prefer parent's, else discover from cfg.data.annotation_dir.
        landmark_order = list(getattr(self._parent_gui, "_landmark_order", []) or [])
        geojson_to_landmark = dict(getattr(self._parent_gui, "_geojson_to_landmark", {}) or {})
        if not landmark_order:
            try:
                annotation_dir = Path(self._cfg.get("data", {}).get("annotation_dir", "training_data"))
                if not annotation_dir.is_absolute():
                    annotation_dir = _project_root / annotation_dir
                landmark_order, geojson_to_landmark = discover_landmarks(annotation_dir)
            except Exception as e:
                self._status.setText(f"Could not discover landmarks: {e}")
                return

        cfg = json.loads(json.dumps(self._cfg))
        cfg.setdefault("heatmap", {})["landmark_order"] = landmark_order
        cfg["heatmap"]["geojson_to_landmark"] = geojson_to_landmark
        cfg["heatmap"]["num_landmarks"] = len(landmark_order)
        # `get_train_transform` requires `input.height`/`width`; default if missing.
        cfg.setdefault("input", {}).setdefault("height", 352)
        cfg["input"].setdefault("width", 512)

        image = imread_any(image_path)
        if image is None:
            self._status.setText(f"Failed to load image: {image_path}")
            return
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Parse keypoints if a GT exists
        landmarks_dict: dict[str, tuple[float, float]] = {}
        if geojson_path and geojson_path.exists():
            try:
                geo = json.loads(geojson_path.read_text())
                for feat in geo.get("features", []):
                    geom = feat.get("geometry", {})
                    props = feat.get("properties", {})
                    classification = props.get("classification") or {}
                    name = classification.get("name", "")
                    internal = geojson_to_landmark.get(name) or geojson_to_landmark.get(_normalize_name(name))
                    if not internal:
                        continue
                    if geom.get("type") == "Point":
                        x, y = geom["coordinates"]
                    elif geom.get("type") == "MultiPoint":
                        x, y = geom["coordinates"][0]
                    else:
                        continue
                    landmarks_dict[internal] = (float(x), float(y))
            except Exception as e:
                self._status.setText(f"Annotation parse error: {e}")
                return

        keypoints = [landmarks_dict.get(n, (0.0, 0.0)) for n in landmark_order]
        names = list(landmark_order)
        present = [n for n in landmark_order if n in landmarks_dict]

        try:
            transform = get_train_transform(cfg)
        except Exception as e:
            QMessageBox.warning(self, "Transform error", f"Could not build transform: {e}")
            return

        n = int(self._n_spin.value())
        self._clear_grid()

        # Reference (resize-only)
        ref_t = A.Compose(
            [A.Resize(cfg["input"]["height"], cfg["input"]["width"])],
            keypoint_params=A.KeypointParams(format="xy", label_fields=["landmark_names"], remove_invisible=False),
        )
        ref = ref_t(image=image_rgb, keypoints=keypoints, landmark_names=names)
        ref_vis = _draw_kps_for_preview(ref["image"], ref["keypoints"], ref["landmark_names"], landmark_order)
        self._add_thumb(ref_vis, "REFERENCE", 0)

        for i in range(n):
            out = transform(image=image_rgb, keypoints=keypoints, landmark_names=names)
            vis = _draw_kps_for_preview(out["image"], out["keypoints"], out["landmark_names"], landmark_order)
            self._add_thumb(vis, f"#{i:02d}", i + 1)

        cols = 4
        # Lay out: keep simple 4-column grid; row computed by caller already used.
        self._status.setText(
            f"Image: {image_path.name}  ({image.shape[1]}×{image.shape[0]}) — "
            f"{len(present)}/{len(landmark_order)} GT landmarks present  "
            f"({n} augmented + 1 reference)"
        )

    def _add_thumb(self, img_bgr: np.ndarray, label: str, idx: int) -> None:
        """Place a (label, image) cell into the grid in row-major order, 4 cols wide."""
        cols = 4
        row = (idx // cols) * 2
        col = idx % cols
        cell = QWidget()
        cv = QVBoxLayout(cell)
        cv.setContentsMargins(0, 0, 0, 0)
        cv.setSpacing(2)
        title = QLabel(label)
        title.setAlignment(Qt.AlignCenter)
        title.setStyleSheet("color: #ddd; font-size: 11px;")
        pix = _bgr_to_qpixmap(img_bgr).scaledToWidth(200, Qt.SmoothTransformation)
        img_label = QLabel()
        img_label.setPixmap(pix)
        img_label.setAlignment(Qt.AlignCenter)
        cv.addWidget(title)
        cv.addWidget(img_label)
        self._grid.addWidget(cell, row, col)


# ---------------------------------------------------------------------------
# Main GUI
# ---------------------------------------------------------------------------
class LandmarkGUI(QMainWindow):
    """Main window for landmark verification."""

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("LandmarkLocator — Verification GUI")
        self.resize(1400, 900)
        # Outlined-checkbox proxy is applied at the QApplication level in main()
        # so it outlives any window. Setting it here too caused a use-after-free
        # in QToolBar's destructor when closing the window.
        self.setStyleSheet(
            "QMainWindow { background: #1e1e1e; color: #ddd; }"
            "QListWidget { background: #252526; color: #ddd; border: none; }"
            "QTableWidget { background: #252526; color: #ddd; gridline-color: #444; }"
            "QHeaderView::section { background: #333; color: #ddd; padding: 4px; }"
            "QScrollArea { background: #1e1e1e; border: none; }"
            "QToolBar { background: #333; border: none; spacing: 6px; }"
            "QStatusBar { background: #252526; color: #aaa; }"
        )

        self._entries: list[ImageEntry] = []
        self._current_idx: int = -1
        self._predictor = None  # LandmarkPredictor or None
        self._output_dir: Optional[Path] = None
        self._gt_dir: Optional[Path] = None
        self._img_dir: Optional[Path] = None
        self._landmark_order: list[str] = []
        self._geojson_to_landmark: dict[str, str] = {}

        # Heatmap overlay state
        self._base_vis: Optional[np.ndarray] = None  # BGR image without heatmap overlay
        self._overlay_heatmaps: Optional[np.ndarray] = None  # (N, MODEL_H, MODEL_W)
        self._overlay_orig_shape: Optional[tuple[int, int]] = None  # (orig_h, orig_w)
        self._selected_heatmap: Optional[str] = None

        self._build_toolbar()
        self._build_ui()
        self.setStatusBar(QStatusBar())
        self.statusBar().showMessage("Ready — load images or a model checkpoint")

        # Auto-load defaults
        self._auto_load()

    # ---- Toolbar ----
    def _build_toolbar(self) -> None:
        """Create the top toolbar."""
        tb = QToolBar("Main")
        tb.setMovable(False)
        self.addToolBar(tb)

        # Load Model is a split-less dropdown: clicking the button shows a menu
        # with two explicit choices, so the user picks single-fold vs ensemble
        # upfront instead of being asked after the file dialog.
        self._load_model_btn = QToolButton(self)
        self._load_model_btn.setText("Load Model")
        self._load_model_btn.setToolTip("Load a single fold .pt or an ensemble folder of fold checkpoints.")
        self._load_model_btn.setPopupMode(QToolButton.InstantPopup)
        model_menu = QMenu(self._load_model_btn)
        act_single = model_menu.addAction("Single fold (.pt file)…")
        act_single.setToolTip("Pick one best_fold*.pt checkpoint. No ensembling.")
        act_single.triggered.connect(self._on_load_single_fold)
        act_ensemble = model_menu.addAction("Ensemble folder…")
        act_ensemble.setToolTip(
            "Pick a folder containing best_fold*.pt checkpoints. Predictions average all folds."
        )
        act_ensemble.triggered.connect(self._on_load_ensemble_folder)
        self._load_model_btn.setMenu(model_menu)
        tb.addWidget(self._load_model_btn)

        act_images = QAction("Load Images", self)
        act_images.triggered.connect(self._on_load_folder)
        tb.addAction(act_images)

        act_gt = QAction("Set GT Dir", self)
        act_gt.triggered.connect(self._on_set_gt_dir)
        tb.addAction(act_gt)

        act_gate = QAction("Gate Config…", self)
        act_gate.triggered.connect(self._on_edit_gate_config)
        tb.addAction(act_gate)

        act_train_cfg = QAction("Train Config…", self)
        act_train_cfg.setToolTip("Edit augmentation + training hyperparameters in configs/default.yaml.")
        act_train_cfg.triggered.connect(self._on_edit_train_config)
        tb.addAction(act_train_cfg)

        act_save = QAction("Save", self)
        act_save.triggered.connect(self._on_save_all)
        tb.addAction(act_save)

        # Push Train Model button to the right
        spacer = QWidget()
        spacer.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        tb.addWidget(spacer)

        train_btn = QPushButton("TRAIN MODEL")
        train_btn.setStyleSheet(
            "QPushButton { background: #8b0000; color: #fff; font-weight: bold;"
            " padding: 4px 16px; border: none; border-radius: 3px; }"
            "QPushButton:hover { background: #a00000; }"
            "QPushButton:pressed { background: #600000; }"
        )
        train_btn.clicked.connect(self._on_train_model)
        tb.addWidget(train_btn)

    # ---- UI layout ----
    def _build_ui(self) -> None:
        """Build the three-panel layout."""
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QHBoxLayout(central)
        main_layout.setContentsMargins(4, 4, 4, 4)

        # Main horizontal splitter: [left | center | right]
        splitter = QSplitter(Qt.Horizontal)
        main_layout.addWidget(splitter)

        # -- Left panel: image list + legend --
        left = QWidget()
        left_layout = QVBoxLayout(left)
        left_layout.setContentsMargins(0, 0, 0, 0)

        self._img_dir_label = QLabel("Images: (none)")
        self._img_dir_label.setStyleSheet("color: #aaa; font-size: 10px; padding: 2px;")
        self._img_dir_label.setWordWrap(False)
        left_layout.addWidget(self._img_dir_label)

        self._search_edit = QLineEdit()
        self._search_edit.setPlaceholderText("Search images…")
        self._search_edit.setClearButtonEnabled(True)
        self._search_edit.setToolTip(
            "Filter images by substring (case-insensitive). Press Enter to jump to the first match."
        )
        self._search_edit.textChanged.connect(self._apply_search_filter)
        self._search_edit.returnPressed.connect(self._jump_to_first_match)
        left_layout.addWidget(self._search_edit)

        self._search_count_label = QLabel("")
        self._search_count_label.setStyleSheet("color: #888; font-size: 10px; padding: 0 2px;")
        left_layout.addWidget(self._search_count_label)

        # "Predict all" populates the per-entry prediction cache in one batch so
        # the pass-count and failed-image highlighting reflect the whole folder
        # instead of just the images the user has visited.
        self._predict_all_btn = QPushButton("Predict all")
        self._predict_all_btn.setToolTip(
            "Run the loaded model on every image in the list (batched). "
            "Populates the pass-count below and lets the highlight toggle "
            "color failed images. Requires a loaded model."
        )
        self._predict_all_btn.clicked.connect(self._on_predict_all)
        left_layout.addWidget(self._predict_all_btn)

        self._image_list = QListWidget()
        self._image_list.currentRowChanged.connect(self._on_image_selected)
        left_layout.addWidget(self._image_list)

        # Highlight-failed toggle and pass-count display sit directly under the
        # list so the user's eye finds them while scanning the images.
        self._highlight_failed_chk = QCheckBox("Highlight failed images")
        self._highlight_failed_chk.setChecked(False)
        self._highlight_failed_chk.setToolTip(
            "Color failed images (any core landmark unreliable) in red so they're "
            "easy to spot. Affects only images that have been predicted."
        )
        self._highlight_failed_chk.toggled.connect(self._on_toggle_highlight_failed)
        left_layout.addWidget(self._highlight_failed_chk)

        self._pass_count_label = QLabel("")
        self._pass_count_label.setStyleSheet("color: #888; font-size: 10px; padding: 0 2px;")
        left_layout.addWidget(self._pass_count_label)

        self._legend = LegendWidget()
        self._legend.labels_toggled.connect(self._on_labels_toggled)
        left_layout.addWidget(self._legend)
        left.setFixedWidth(220)
        splitter.addWidget(left)

        # -- Center panel: image view + info table --
        center = QWidget()
        center_layout = QVBoxLayout(center)
        center_layout.setContentsMargins(0, 0, 0, 0)

        self._image_widget = ImageWidget()
        center_layout.addWidget(self._image_widget, stretch=3)

        # Overlay size (points/rings/labels) slider — lives directly below the image view.
        overlay_row = QHBoxLayout()
        overlay_lbl = QLabel("Overlay size:")
        overlay_lbl.setStyleSheet("color: #aaa; font-size: 10px;")
        overlay_row.addWidget(overlay_lbl)
        self._overlay_size_slider = QSlider(Qt.Horizontal)
        self._overlay_size_slider.setRange(25, 300)  # percent of auto-computed size
        self._overlay_size_slider.setValue(100)
        self._overlay_size_slider.setFixedHeight(20)
        self._overlay_size_slider.valueChanged.connect(self._on_overlay_size_changed)
        overlay_row.addWidget(self._overlay_size_slider)
        self._overlay_size_label = QLabel("100%")
        self._overlay_size_label.setStyleSheet("color: #aaa; font-size: 10px;")
        self._overlay_size_label.setFixedWidth(40)
        overlay_row.addWidget(self._overlay_size_label)
        center_layout.addLayout(overlay_row)

        # Overall wing pass/fail status line (under the image).
        self._wing_status_label = QLabel("")
        self._wing_status_label.setAlignment(Qt.AlignCenter)
        self._wing_status_label.setStyleSheet(
            "padding: 6px; font-weight: bold; font-size: 14px;" "background: #252526; color: #aaa; border-radius: 4px;"
        )
        self._wing_status_label.setFixedHeight(32)
        center_layout.addWidget(self._wing_status_label)

        # Heatmap overlay opacity controls — built here so the slot can be wired, but
        # mounted below the color gradient inside HeatmapPanel (see _heatmap_panel.mount_opacity_controls).
        self._opacity_slider = QSlider(Qt.Horizontal)
        self._opacity_slider.setRange(0, 100)
        self._opacity_slider.setValue(50)
        self._opacity_slider.setFixedHeight(20)
        self._opacity_slider.valueChanged.connect(self._on_opacity_changed)
        self._opacity_label = QLabel("50%")
        self._opacity_label.setStyleSheet("color: #aaa; font-size: 10px;")
        self._opacity_label.setFixedWidth(32)

        self._model_label = QLabel("Model: (none)")
        self._model_label.setStyleSheet("color: #aaa; font-size: 10px; padding: 2px;")
        center_layout.addWidget(self._model_label)

        self._show_gate_chk = QCheckBox("Show gate ring on each landmark")
        self._show_gate_chk.setChecked(False)
        self._show_gate_chk.setToolTip(
            "Ring each predicted landmark in green (pass) or red (fail). "
            "Per-landmark metrics are in the table below."
        )
        self._show_gate_chk.toggled.connect(self._on_toggle_show_gate)
        center_layout.addWidget(self._show_gate_chk)

        self._info_table = QTableWidget(0, 7)
        self._info_table.setHorizontalHeaderLabels(
            ["Landmark", "Gate", "Peak", "Sharpness", "SP ratio", "Core?", "Reason"]
        )
        hdr = self._info_table.horizontalHeader()
        hdr.setSectionResizeMode(QHeaderView.ResizeToContents)
        hdr.setSectionResizeMode(0, QHeaderView.Stretch)  # landmark name column expands
        hdr.setSectionResizeMode(6, QHeaderView.Stretch)  # reason column expands
        self._info_table.setEditTriggers(QTableWidget.NoEditTriggers)
        self._info_table.setMaximumHeight(220)
        center_layout.addWidget(self._info_table, stretch=0)
        splitter.addWidget(center)

        # -- Right panel: heatmaps --
        self._heatmap_panel = HeatmapPanel()
        self._heatmap_panel.heatmap_clicked.connect(self._on_heatmap_clicked)
        self._heatmap_panel.mount_opacity_controls(self._opacity_slider, self._opacity_label)
        splitter.addWidget(self._heatmap_panel)

        splitter.setSizes([220, 800, 300])

    # ---- Landmark order management ----
    def _set_landmark_order(self, landmark_order: list[str], geojson_to_landmark: dict[str, str] | None = None) -> None:
        """Update the active landmark set and rebuild dependent widgets."""
        if landmark_order == self._landmark_order:
            return
        self._landmark_order = landmark_order
        if geojson_to_landmark is not None:
            self._geojson_to_landmark = geojson_to_landmark
        _ensure_colors(landmark_order)
        qcolors = _make_qcolors(landmark_order)

        # Rebuild info table rows
        self._info_table.setRowCount(len(landmark_order))
        for i, name in enumerate(landmark_order):
            item = QTableWidgetItem(_make_display_name(name))
            color = qcolors.get(name, QColor(200, 200, 200))
            item.setForeground(color)
            self._info_table.setItem(i, 0, item)

        # Update legend and heatmap panel
        self._legend.set_landmarks(landmark_order)
        self._heatmap_panel.set_landmarks(landmark_order)

    # ---- Auto-load ----
    def _auto_load(self) -> None:
        """No auto-load — user must select folders explicitly."""
        pass

    # ---- Gate-aware prediction (for inspection) ----
    def _predict_for_inspection(self, image: np.ndarray) -> dict:
        """Predict with include_unreliable=True and without aborting on core failure."""
        from landmark_locator.inference.predict import LowConfidenceLandmarkError

        try:
            return self._predictor.predict(image, include_unreliable=True)
        except LowConfidenceLandmarkError:
            # Temporarily drop the core set so the GUI can still render heatmaps.
            saved_core = self._predictor.gate_config.get("core_landmarks", [])
            self._predictor.gate_config["core_landmarks"] = []
            try:
                return self._predictor.predict(image, include_unreliable=True)
            finally:
                self._predictor.gate_config["core_landmarks"] = saved_core

    # ---- Gate config ----
    def _on_edit_gate_config(self) -> None:
        """Open a dialog to edit per-landmark confidence-gate thresholds."""
        if self._predictor is None:
            self.statusBar().showMessage("Load a model before editing gate config.")
            return
        from landmark_locator.inference.predict import canonical_sidecar_gate_path

        # Sidecar may not exist yet — canonical_sidecar_gate_path always returns the path it would be.
        sidecar_path = self._predictor.sidecar_gate_path or canonical_sidecar_gate_path(
            self._predictor.checkpoint_paths[0]
            if hasattr(self._predictor, "checkpoint_paths")
            else self._predictor.checkpoint_path if hasattr(self._predictor, "checkpoint_path") else Path(".")
        )
        dlg = GateConfigDialog(
            self,
            self._predictor.gate_config,
            self._predictor.landmark_order,
            sidecar_path=sidecar_path,
        )
        if dlg.exec_() != QDialog.Accepted:
            return
        override = dlg.result_override()
        # update_gate_config does a deep-merge, which would keep stale per_landmark
        # entries the user removed. Replace the gate_config wholesale instead.
        # Preserve the `tiers:` metadata block — result_override() doesn't emit it
        # (it isn't part of the active gate the predictor consumes), but the dialog
        # reads it back from gate_config on next open to populate the tier dropdown.
        # Dropping it here makes subsequent tier changes silently no-op until the
        # model is reloaded.
        preserved_tiers = self._predictor.gate_config.get("tiers")
        self._predictor.gate_config = override
        if preserved_tiers is not None:
            self._predictor.gate_config["tiers"] = preserved_tiers
        # Invalidate cached predictions so they re-run under the new gate
        for entry in self._entries:
            entry.prediction = None
        self._refresh_image_list_styling()
        if self._current_idx >= 0:
            self._on_image_selected(self._current_idx)
        self.statusBar().showMessage("Gate config updated.")

    def _on_edit_train_config(self) -> None:
        """Open a dialog to edit augmentation + training hyperparameters in default.yaml."""
        config_path = _project_root / "configs" / "default.yaml"
        if not config_path.exists():
            QMessageBox.warning(self, "Config missing", f"{config_path} does not exist.")
            return
        dlg = TrainingConfigDialog(self, config_path)
        if dlg.exec_() == QDialog.Accepted:
            self.statusBar().showMessage(f"Training config saved to {config_path.name}.")

    # ---- Actions ----
    def _on_load_single_fold(self) -> None:
        """Load a single fold checkpoint via file picker (no ensembling)."""
        start_dir = _project_root / "trained_models"
        if not start_dir.exists():
            start_dir = _project_root
        path = _pick_file_native("Select Fold Checkpoint", str(start_dir), "PyTorch (*.pt *.pth)")
        if not path:
            return
        self._load_predictor_at(Path(path))

    def _on_load_ensemble_folder(self) -> None:
        """Load an ensemble from a folder of `best_fold*.pt` checkpoints via folder picker."""
        from landmark_locator.inference.predict import _find_fold_checkpoints

        start_dir = _project_root / "trained_models"
        if not start_dir.exists():
            start_dir = _project_root
        folder = _pick_folder_native("Select Ensemble Folder", str(start_dir))
        if not folder:
            return
        folder_path = Path(folder)
        ckpts = _find_fold_checkpoints(folder_path)
        if not ckpts:
            QMessageBox.warning(
                self,
                "No fold checkpoints",
                f"No best_fold*.pt files found in:\n{folder_path}\n\n"
                "Pick a folder containing the cross-validation fold checkpoints, or use "
                "'Single fold (.pt file)…' to load one checkpoint directly.",
            )
            return
        self._load_predictor_at(folder_path)

    def _load_predictor_at(self, target: Path) -> None:
        """Instantiate the predictor for `target` (file → single, folder → ensemble) and finalize.

        Shared tail for both Load Model menu actions: builds the predictor via
        make_predictor, updates the model label and status bar, invalidates
        cached predictions, refreshes the image-list pass-count and styling,
        and re-runs the currently-selected image so the view reflects the new
        model.
        """
        from landmark_locator.inference.predict import _find_fold_checkpoints, make_predictor

        try:
            self._predictor = make_predictor(target)
        except Exception as e:
            self.statusBar().showMessage(f"Failed to load model: {e}")
            return

        self._set_landmark_order(
            self._predictor.landmark_order,
            self._predictor.geojson_to_landmark,
        )
        if target.is_dir():
            n = len(_find_fold_checkpoints(target))
            self._model_label.setText(f"Ensemble: {target.name}/ ({n} folds)")
            self.statusBar().showMessage(
                f"Ensemble loaded from {target.name}: {n} folds — predictions will average all of them"
            )
        else:
            self._model_label.setText(f"Model: {target.name}")
            self.statusBar().showMessage(f"Model loaded: {target.name}")

        for entry in self._entries:
            entry.prediction = None
        self._refresh_image_list_styling()
        if self._current_idx >= 0:
            self._on_image_selected(self._current_idx)

    def _on_toggle_show_gate(self, checked: bool) -> None:
        """Redraw the current image with/without the gate overlay."""
        if self._current_idx >= 0:
            self._on_image_selected(self._current_idx)

    def _on_labels_toggled(self, checked: bool) -> None:
        """Redraw the current image with/without overlay landmark labels."""
        if self._current_idx >= 0:
            self._on_image_selected(self._current_idx)

    def _on_overlay_size_changed(self, value: int) -> None:
        """Rescale and redraw overlay markers/labels/rings at the new size."""
        self._overlay_size_label.setText(f"{value}%")
        if self._current_idx >= 0:
            self._on_image_selected(self._current_idx)

    def _on_load_folder(self) -> None:
        """Load images from a user-selected folder."""
        start = str(self._img_dir) if self._img_dir else str(_project_root)
        folder = _pick_folder_native("Select Image Folder", start)
        if not folder:
            return
        folder = Path(folder)
        self._img_dir = folder
        self._img_dir_label.setText(f"Images: {folder.name}/")
        self._load_image_folder(folder, self._gt_dir)
        self.statusBar().showMessage(f"Loaded {len(self._entries)} images from {folder.name}/")

    def _on_set_gt_dir(self) -> None:
        """Let the user pick a ground-truth annotation folder."""
        start = str(self._gt_dir) if self._gt_dir else str(_project_root)
        folder = _pick_folder_native("Select GT Annotation Folder", start)
        if not folder:
            return
        self._gt_dir = Path(folder)
        self._heatmap_panel.set_gt_dir_label(f"GT: {self._gt_dir.name}/")
        self._apply_gt_dir()
        # Re-match GT paths to existing image entries
        fuzzy_matches: list[tuple[int, str, str]] = []
        for i, entry in enumerate(self._entries):
            entry.geojson_path = None
            entry.gt = None
            entry.gt_heatmaps = None
            entry._gt_loaded = False
            path, fuzzy = _find_geojson_for_image(self._gt_dir, entry.path.name)
            entry.geojson_path = path
            if fuzzy and path:
                fuzzy_matches.append((i, entry.path.name, path.name))
        if fuzzy_matches:
            self._confirm_fuzzy_matches(fuzzy_matches)
        gt_count = sum(1 for e in self._entries if e.geojson_path)
        self.statusBar().showMessage(f"GT dir: {self._gt_dir.name}/ — matched {gt_count}/{len(self._entries)} images")
        # Refresh current view
        if self._current_idx >= 0:
            self._on_image_selected(self._current_idx)

    def _apply_gt_dir(self) -> None:
        """Discover landmarks from the current GT dir and update the UI."""
        if not self._gt_dir:
            return
        try:
            landmark_order, geojson_to_landmark = discover_landmarks(self._gt_dir)
            if landmark_order:
                self._set_landmark_order(landmark_order, geojson_to_landmark)
        except Exception as e:
            self.statusBar().showMessage(f"Warning: could not discover landmarks: {e}")

    def _on_set_output(self) -> None:
        """Set the output directory for Save All."""
        folder = _pick_folder_native("Select Output Directory", str(_project_root))
        if folder:
            self._output_dir = Path(folder)
            self.statusBar().showMessage(f"Output directory: {self._output_dir}")

    def _on_save_all(self) -> None:
        """Save outputs for all entries with user-selected options."""
        try:
            self._do_save_all()
        except Exception as exc:
            import traceback

            self.statusBar().showMessage(f"Save failed: {exc}")
            print("Save failed:\n" + traceback.format_exc(), file=sys.stderr)

    def _do_save_all(self) -> None:
        if not self._entries:
            self.statusBar().showMessage("No images loaded")
            return

        # Build a folder dialog with save-option checkboxes. Force Qt's own dialog
        # (not macOS native) so we can add our checkboxes to its layout safely.
        dialog = QFileDialog(self, "Select Output Directory")
        dialog.setFileMode(QFileDialog.Directory)
        dialog.setOption(QFileDialog.ShowDirsOnly, True)
        dialog.setOption(QFileDialog.DontUseNativeDialog, True)
        start = str(self._output_dir) if self._output_dir else str(_project_root)
        dialog.setDirectory(start)

        # Add checkboxes to the dialog layout (guard: layout may not be a QGridLayout
        # on some Qt versions, in which case we fall back to separate checkbox dialogs).
        cb_geojson = QCheckBox("Save GeoJSONs")
        cb_geojson.setChecked(True)
        cb_images = QCheckBox("Save Labeled Images")
        cb_images.setChecked(True)
        opts_layout = QHBoxLayout()
        opts_layout.addWidget(cb_geojson)
        opts_layout.addWidget(cb_images)
        try:
            dlg_layout = dialog.layout()
            if dlg_layout is not None and hasattr(dlg_layout, "rowCount"):
                dlg_layout.addLayout(opts_layout, dlg_layout.rowCount(), 0, 1, -1)
            else:
                raise RuntimeError("dialog layout not a QGridLayout")
        except Exception:
            # Couldn't inject into the dialog — show an ad hoc options dialog first.
            picker = QDialog(self)
            picker.setWindowTitle("Save Options")
            pl = QVBoxLayout(picker)
            pl.addWidget(cb_geojson)
            pl.addWidget(cb_images)
            btns = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
            btns.accepted.connect(picker.accept)
            btns.rejected.connect(picker.reject)
            pl.addWidget(btns)
            if picker.exec_() != QDialog.Accepted:
                return

        if not dialog.exec_():
            return
        folders = dialog.selectedFiles()
        if not folders:
            return

        save_geojson = cb_geojson.isChecked()
        save_images = cb_images.isChecked()
        if not save_geojson and not save_images:
            self.statusBar().showMessage("Nothing selected to save")
            return

        self._output_dir = Path(folders[0])
        self._output_dir.mkdir(parents=True, exist_ok=True)

        progress = QProgressDialog("Saving...", "Cancel", 0, len(self._entries), self)
        progress.setWindowModality(Qt.WindowModal)
        progress.setMinimumDuration(0)

        saved = 0
        for i, entry in enumerate(self._entries):
            if progress.wasCanceled():
                break
            progress.setValue(i)

            from landmark_locator.data.psd_loader import imread_any

            image = imread_any(entry.path)
            if image is None:
                continue

            entry.load_gt()
            preds = {}
            if entry.prediction:
                preds = entry.prediction["landmarks"]
            elif self._predictor:
                try:
                    entry.prediction = self._predict_for_inspection(image)
                    preds = entry.prediction["landmarks"]
                except Exception:
                    pass

            if save_images:
                vis = draw_landmarks_on_image(
                    image,
                    preds,
                    entry.gt,
                    landmark_order=self._landmark_order,
                    show_labels=self._legend.labels_enabled(),
                    size_scale=self._overlay_size_slider.value() / 100.0,
                )
                out_path = self._output_dir / f"{entry.path.stem}_landmarks.jpg"
                cv2.imwrite(str(out_path), vis)

            if save_geojson and preds:
                pred = entry.prediction or {}
                reliable_map = pred.get("reliable", {})
                reason_map = pred.get("gate_reason", {})
                confidence_map = pred.get("confidences", {})
                sharpness_map = pred.get("sharpness", {})
                spr_map = pred.get("second_peak_ratio", {})

                features = []
                reverse_map = {v: k for k, v in self._geojson_to_landmark.items()}
                for name, (x, y) in preds.items():
                    geojson_name = reverse_map.get(name, name)
                    props = {
                        "classification": {"name": geojson_name},
                        "reliable": bool(reliable_map.get(name, True)),
                        "gate_reason": reason_map.get(name, ""),
                        "confidence": float(confidence_map.get(name, 0.0)),
                        "sharpness": float(sharpness_map.get(name, 0.0)),
                        "second_peak_ratio": float(spr_map.get(name, 0.0)),
                    }
                    features.append(
                        {
                            "type": "Feature",
                            "geometry": {"type": "Point", "coordinates": [x, y]},
                            "properties": props,
                        }
                    )
                geojson_path = self._output_dir / f"{entry.path.stem}_landmarks.geojson"
                with open(geojson_path, "w") as f:
                    json.dump({"type": "FeatureCollection", "features": features}, f, indent=2)

            saved += 1

        progress.setValue(len(self._entries))
        parts = []
        if save_images:
            parts.append("images")
        if save_geojson:
            parts.append("GeoJSON")
        self.statusBar().showMessage(f"Saved {saved} {' + '.join(parts)} to {self._output_dir}")

    def _on_train_model(self) -> None:
        """Launch training fold 0 in a background thread with live log dialog."""
        # Validate that GT annotations exist
        gt_count = sum(1 for e in self._entries if e.geojson_path and e.geojson_path.exists())
        if gt_count == 0:
            self.statusBar().showMessage("No ground-truth annotations found — cannot train")
            return

        # Pre-flight: check for geojson→image fuzzy matches (same direction as LandmarkDataset)
        from landmark_locator.data.dataset import _find_similar_file

        image_dir = Path(yaml.safe_load(open(_project_root / "configs" / "default.yaml"))["data"]["image_dir"])
        if not image_dir.is_absolute():
            image_dir = _project_root / image_dir
        gt_dir = self._gt_dir or (_project_root / "training_data")
        fuzzy_pairs: list[tuple[str, str]] = []
        for gj in sorted(gt_dir.glob("*.geojson")):
            img_name = gj.stem
            img_path = image_dir / img_name
            if not img_path.exists():
                img_path = image_dir / img_name.strip()
            if not img_path.exists():
                match = _find_similar_file(image_dir, img_name)
                if match:
                    fuzzy_pairs.append((gj.name, match.name))
        if fuzzy_pairs:
            from PyQt5.QtWidgets import QMessageBox

            lines = [f"  {a}  \u2192  {b}" for a, b in (_truncate_pair(gj, img) for gj, img in fuzzy_pairs)]
            msg = QMessageBox(self)
            msg.setStyleSheet("QLabel { min-width: 500px; }")
            msg.setWindowTitle("Approximate Image Matches")
            msg.setText(
                f"{len(fuzzy_pairs)} annotation(s) matched to images by ignoring whitespace:\n\n" + "\n".join(lines)
            )
            msg.setInformativeText("Continue training with these matches?")
            msg.setStandardButtons(QMessageBox.Yes | QMessageBox.No)
            msg.setDefaultButton(QMessageBox.Yes)
            if msg.exec_() != QMessageBox.Yes:
                return

        # Prompt for model name
        from PyQt5.QtWidgets import QInputDialog

        name, ok = QInputDialog.getText(
            self,
            "Model Name",
            "Enter a name for the model checkpoint:",
            text="landmark_model",
        )
        if not ok or not name.strip():
            return
        model_name = name.strip()

        config_path = _project_root / "configs" / "default.yaml"
        with open(config_path) as f:
            train_cfg = yaml.safe_load(f)
        max_epochs = train_cfg.get("training", {}).get("epochs", 0)

        self._train_dialog = TrainingDialog(model_name, self._landmark_order, max_epochs, self)
        self._train_thread = TrainingThread(model_name, self._gt_dir)
        self._train_thread.progress.connect(self._train_dialog.append_log)
        self._train_thread.epoch_data.connect(self._train_dialog.update_chart)
        self._train_thread.finished_training.connect(self._on_training_finished)
        self._train_thread.error.connect(self._on_training_error)
        self._train_thread.start()
        self._train_dialog.append_log(f"Starting training '{model_name}' with {gt_count} annotated images...")
        self._train_dialog.exec_()

    def _on_training_finished(self, ckpt_dir: str) -> None:
        """Handle successful training completion. ckpt_dir is the run dir with best_fold*.pt."""
        self._train_dialog.append_log(f"\nTraining complete! Fold checkpoints: {ckpt_dir}")
        self._train_dialog.enable_close()
        run_dir = Path(ckpt_dir)  # flat layout: ckpt_dir is the run dir itself
        # Save the error chart into the run folder
        chart_path = run_dir / "training_chart.png"
        try:
            self._train_dialog._fig.savefig(str(chart_path), dpi=150, facecolor="#1e1e1e")
            self._train_dialog.append_log(f"Error chart saved: {chart_path}")
        except Exception as e:
            self._train_dialog.append_log(f"Warning: could not save chart: {e}")
        # Save the run log as a text file
        log_path = run_dir / "training.log"
        try:
            log_path.write_text(self._train_dialog._log.toPlainText())
            self._train_dialog.append_log(f"Training log saved: {log_path}")
        except Exception as e:
            self._train_dialog.append_log(f"Warning: could not save log: {e}")
        # Auto-load fold 0 for quick inspection
        fold0 = Path(ckpt_dir) / "best_fold0.pt"
        if not fold0.exists():
            self.statusBar().showMessage(f"Training done but no best_fold0.pt in {ckpt_dir}")
            return
        try:
            from landmark_locator.inference.predict import LandmarkPredictor

            self._predictor = LandmarkPredictor(fold0)
            self._set_landmark_order(
                self._predictor.landmark_order,
                self._predictor.geojson_to_landmark,
            )
            for entry in self._entries:
                entry.prediction = None
            if self._current_idx >= 0:
                self._on_image_selected(self._current_idx)
            self._model_label.setText(f"Model: {run_dir.name}/best_fold0.pt")
            self.statusBar().showMessage(f"Loaded fold 0 from {run_dir.name}/; use predict-ensemble for all folds.")
        except Exception as e:
            self.statusBar().showMessage(f"Training done but failed to load checkpoint: {e}")

    def _on_training_error(self, msg: str) -> None:
        """Handle training failure."""
        self._train_dialog.append_log(f"\nERROR: {msg}")
        self._train_dialog.enable_close()
        self.statusBar().showMessage(f"Training failed: {msg}")
        # Save the run log as a text file even on failure — the log is what diagnoses the failure.
        try:
            run_dir = _project_root / "trained_models" / self._train_dialog._model_name
            run_dir.mkdir(parents=True, exist_ok=True)
            (run_dir / "training.log").write_text(self._train_dialog._log.toPlainText())
        except Exception:
            pass

    # ---- Heatmap overlay ----
    def _on_heatmap_clicked(self, name: str) -> None:
        """Toggle heatmap overlay on the main image."""
        self._selected_heatmap = name if name else None
        self._apply_heatmap_overlay()

    def _on_opacity_changed(self, value: int) -> None:
        """Update opacity label and reapply overlay."""
        self._opacity_label.setText(f"{value}%")
        if self._selected_heatmap:
            self._apply_heatmap_overlay()

    def _apply_heatmap_overlay(self) -> None:
        """Blend the selected heatmap channel onto the base visualization."""
        if self._base_vis is None:
            return
        if not self._selected_heatmap or self._overlay_heatmaps is None:
            self._image_widget.set_image(self._base_vis)
            return

        if self._selected_heatmap not in self._landmark_order:
            return
        idx = self._landmark_order.index(self._selected_heatmap)
        hm = self._overlay_heatmaps[idx]  # (MODEL_H, MODEL_W)
        orig_h, orig_w = self._overlay_orig_shape

        # Resize heatmap to original image size and colorize
        hm_resized = cv2.resize(hm, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)
        vmax = hm_resized.max()
        if vmax > 0:
            hm_norm = (hm_resized / vmax * 255).astype(np.uint8)
        else:
            hm_norm = np.zeros((orig_h, orig_w), dtype=np.uint8)
        colored = _colorize_heatmap(hm_norm)

        alpha = self._opacity_slider.value() / 100.0
        blended = cv2.addWeighted(self._base_vis, 1.0, colored, alpha, 0)
        self._image_widget.set_image(blended)

    # ---- Image loading ----
    def _load_image_folder(self, img_dir: Path, gt_dir: Optional[Path]) -> None:
        """Scan folder for images and match GT annotations."""
        self._entries.clear()
        self._image_list.clear()
        self._current_idx = -1

        image_files = sorted(f for f in img_dir.iterdir() if f.is_file() and f.suffix.lower() in IMAGE_EXTENSIONS)

        fuzzy_matches: list[tuple[int, str, str]] = []  # (index, image_name, geojson_name)
        for img_path in image_files:
            geojson_path = None
            if gt_dir:
                geojson_path, fuzzy = _find_geojson_for_image(gt_dir, img_path.name)
                if fuzzy and geojson_path:
                    fuzzy_matches.append((len(self._entries), img_path.name, geojson_path.name))
            entry = ImageEntry(path=img_path, geojson_path=geojson_path)
            self._entries.append(entry)
            self._image_list.addItem(img_path.name)

        if fuzzy_matches:
            self._confirm_fuzzy_matches(fuzzy_matches)

        # Re-apply any active search filter against the new list before selecting.
        self._apply_search_filter(self._search_edit.text())

        if self._entries:
            # Select the first visible row (respects the search filter).
            self._image_list.setCurrentRow(self._first_visible_row(default=0))
        self._refresh_image_list_styling()

    def _confirm_fuzzy_matches(self, fuzzy_matches: list[tuple[int, str, str]]) -> None:
        """Show a dialog asking the user to accept or reject fuzzy GT matches."""
        from PyQt5.QtWidgets import QMessageBox

        lines = [f"  {a}  \u2192  {b}" for a, b in (_truncate_pair(img, gj) for _, img, gj in fuzzy_matches)]
        msg = QMessageBox(self)
        msg.setStyleSheet("QLabel { min-width: 500px; }")
        msg.setWindowTitle("Approximate GT Matches")
        msg.setText(
            f"{len(fuzzy_matches)} annotation file(s) matched by ignoring whitespace differences:\n\n"
            + "\n".join(lines)
        )
        msg.setInformativeText("Accept these matches?")
        msg.setStandardButtons(QMessageBox.Yes | QMessageBox.No)
        msg.setDefaultButton(QMessageBox.Yes)
        if msg.exec_() != QMessageBox.Yes:
            for idx, _, _ in fuzzy_matches:
                self._entries[idx].geojson_path = None

    # ---- Image selection ----
    def _on_image_selected(self, row: int) -> None:
        """Handle image list selection change."""
        if row < 0 or row >= len(self._entries):
            return
        self._current_idx = row
        entry = self._entries[row]
        self.statusBar().showMessage(f"Loading {entry.path.name}...")
        QApplication.processEvents()

        # Load image
        from landmark_locator.data.psd_loader import imread_any

        image = imread_any(entry.path)
        if image is None:
            self.statusBar().showMessage(f"Failed to load {entry.path.name}")
            self._image_widget.clear_image()
            return

        orig_h, orig_w = image.shape[:2]

        # Load ground truth
        entry.load_gt()
        if entry.gt and entry.gt_heatmaps is None and self._landmark_order:
            entry.gt_heatmaps = generate_gt_heatmaps(entry.gt, self._landmark_order, orig_w=orig_w, orig_h=orig_h)

        # Run prediction if model available and not cached.
        # In the inspection GUI we show every predicted landmark even if it failed
        # the gate — so always include_unreliable, and bypass abort-on-core-fail so
        # inspecting a failing image still shows its heatmaps.
        if self._predictor and entry.prediction is None:
            try:
                entry.prediction = self._predict_for_inspection(image)
            except Exception as e:
                self.statusBar().showMessage(f"Prediction failed: {e}")

        # Draw overlay and cache base visualization
        preds = entry.prediction["landmarks"] if entry.prediction else {}
        size_scale = self._overlay_size_slider.value() / 100.0
        vis = draw_landmarks_on_image(
            image,
            preds,
            entry.gt,
            landmark_order=self._landmark_order,
            show_labels=self._legend.labels_enabled(),
            size_scale=size_scale,
        )
        if self._show_gate_chk.isChecked() and entry.prediction is not None and "reliable" in entry.prediction:
            vis = _draw_gate_overlay(vis, entry.prediction, self._landmark_order, size_scale=size_scale)
        self._base_vis = vis.copy()
        self._overlay_heatmaps = entry.prediction["heatmaps"] if entry.prediction else None
        self._overlay_orig_shape = (orig_h, orig_w)
        self._selected_heatmap = None
        self._heatmap_panel._selected = None
        for lbl in self._heatmap_panel._pred_labels.values():
            lbl.setStyleSheet(HeatmapPanel._STYLE_NORMAL)
        self._image_widget.set_image(vis)
        self._image_widget.fit_in_view()

        # Update info table
        self._update_info_table(entry)
        self._update_wing_status(entry)
        # Lazy single-image prediction may have just populated entry.prediction —
        # refresh the list styling so the new pass/fail status is reflected.
        self._refresh_image_list_styling()

        # Update heatmap panel — pass GT coords scaled to model resolution
        pred_hm = entry.prediction["heatmaps"] if entry.prediction else None
        gt_model_coords = None
        if entry.gt:
            gt_model_coords = {name: (x * MODEL_W / orig_w, y * MODEL_H / orig_h) for name, (x, y) in entry.gt.items()}
        self._heatmap_panel.update_heatmaps(pred_hm, gt_model_coords)

        gt_count = len(entry.gt) if entry.gt else 0
        pred_str = "yes" if entry.prediction else "no"
        self.statusBar().showMessage(
            f"{entry.path.name} — {orig_w}x{orig_h} — GT: {gt_count}/{len(self._landmark_order)} — Prediction: {pred_str}"
        )

    # ---- Info table ----
    def _update_info_table(self, entry: ImageEntry) -> None:
        """Populate the per-landmark gate table.

        Columns: Landmark, Gate (PASS/FAIL), Peak, Sharpness, SP ratio, Core?, Reason.
        Peak/Sharpness/SP-ratio cells are colored against their active threshold so
        the user can see at a glance which metric is near the edge.
        """
        pred = entry.prediction
        gate_cfg = self._predictor.gate_config if self._predictor else None
        core = set((gate_cfg or {}).get("core_landmarks", []))

        pass_fg = QColor(80, 255, 80)
        fail_fg = QColor(255, 80, 80)
        warn_fg = QColor(255, 200, 80)
        dim_fg = QColor(160, 160, 160)

        def _threshold_cell(value: float | None, threshold: float | None, higher_is_better: bool) -> QTableWidgetItem:
            if value is None:
                item = QTableWidgetItem("N/A")
                item.setForeground(dim_fg)
                return item
            text = f"{value:.3f}" if value < 10 else f"{value:.1f}"
            item = QTableWidgetItem(text)
            if threshold is None:
                return item
            passes = value >= threshold if higher_is_better else value <= threshold
            # Flag cells within 15% of the threshold on the passing side as "close"
            if not passes:
                item.setForeground(fail_fg)
            elif higher_is_better and threshold > 0 and value < threshold * 1.15:
                item.setForeground(warn_fg)
            elif not higher_is_better and threshold > 0 and value > threshold * 0.85:
                item.setForeground(warn_fg)
            else:
                item.setForeground(pass_fg)
            return item

        for i, name in enumerate(self._landmark_order):
            if pred is None:
                for col in range(1, 7):
                    cell = QTableWidgetItem("—")
                    cell.setForeground(dim_fg)
                    self._info_table.setItem(i, col, cell)
                continue

            reliable = pred.get("reliable", {}).get(name)
            peak = pred.get("confidences", {}).get(name)
            sharp = pred.get("sharpness", {}).get(name)
            spr = pred.get("second_peak_ratio", {}).get(name)
            reason = pred.get("gate_reason", {}).get(name, "")

            peak_thr = None
            sharp_thr = None
            spr_thr = None
            if gate_cfg is not None:
                peak_thr = gate_cfg["peak"]["per_landmark"].get(name, gate_cfg["peak"]["global"])
                sharp_thr = gate_cfg["sharpness"]["per_landmark"].get(name, gate_cfg["sharpness"]["global"])
                spr_thr = gate_cfg["second_peak_ratio"]["per_landmark"].get(
                    name, gate_cfg["second_peak_ratio"]["global"]
                )

            gate_item = QTableWidgetItem("PASS" if reliable else "FAIL")
            gate_item.setForeground(pass_fg if reliable else fail_fg)
            self._info_table.setItem(i, 1, gate_item)

            self._info_table.setItem(i, 2, _threshold_cell(peak, peak_thr, higher_is_better=True))
            self._info_table.setItem(i, 3, _threshold_cell(sharp, sharp_thr, higher_is_better=True))
            self._info_table.setItem(i, 4, _threshold_cell(spr, spr_thr, higher_is_better=False))

            core_item = QTableWidgetItem("core" if name in core else "")
            if name in core:
                core_item.setForeground(warn_fg)
            self._info_table.setItem(i, 5, core_item)

            reason_item = QTableWidgetItem(reason or ("" if reliable else "(no reason)"))
            if not reliable:
                reason_item.setForeground(fail_fg)
            else:
                reason_item.setForeground(dim_fg)
            self._info_table.setItem(i, 6, reason_item)

    # ---- Wing-level pass/fail ----
    # ---- Image-list search ----
    def _apply_search_filter(self, text: str | None = None) -> None:
        """Hide image-list rows whose filename does not contain the search substring."""
        if text is None:
            text = self._search_edit.text() if hasattr(self, "_search_edit") else ""
        query = (text or "").strip().lower()
        total = self._image_list.count()
        visible = 0
        for row in range(total):
            item = self._image_list.item(row)
            if item is None:
                continue
            matches = (not query) or (query in item.text().lower())
            item.setHidden(not matches)
            if matches:
                visible += 1
        if not query:
            self._search_count_label.setText("")
        else:
            self._search_count_label.setText(f"{visible} / {total} match")
        # If the currently selected row is hidden by the filter, jump to the first visible row.
        cur = self._image_list.currentRow()
        if cur >= 0 and cur < total:
            cur_item = self._image_list.item(cur)
            if cur_item is not None and cur_item.isHidden():
                first = self._first_visible_row(default=-1)
                if first >= 0:
                    self._image_list.setCurrentRow(first)

    def _jump_to_first_match(self) -> None:
        """Pressing Enter in the search box selects the first visible match."""
        first = self._first_visible_row(default=-1)
        if first >= 0:
            self._image_list.setCurrentRow(first)
            self._image_list.setFocus()

    def _first_visible_row(self, default: int = -1) -> int:
        """Return the row index of the first non-hidden image, or `default`."""
        for row in range(self._image_list.count()):
            item = self._image_list.item(row)
            if item is not None and not item.isHidden():
                return row
        return default

    def _update_wing_status(self, entry: ImageEntry) -> None:
        """Summarize the whole wing as pass / fail below the image view.

        Pass = every core landmark reliable. Non-core failures list as a warning but
        don't flip the overall status (matches the gate's abort semantics).
        """
        pred = entry.prediction
        if pred is None or not pred.get("reliable"):
            self._wing_status_label.setText("No prediction")
            self._wing_status_label.setStyleSheet(
                "padding: 6px; font-weight: bold; font-size: 14px;"
                "background: #252526; color: #888; border-radius: 4px;"
            )
            return

        gate_cfg = self._predictor.gate_config if self._predictor else {}
        core = set(gate_cfg.get("core_landmarks", []) or [])
        reliable = pred["reliable"]

        core_fail = sorted(n for n in core if n in reliable and not reliable[n])
        non_core_fail = sorted(n for n, ok in reliable.items() if n not in core and not ok)

        if core_fail:
            self._wing_status_label.setText(f"WING FAIL — core landmarks failed: {', '.join(core_fail)}")
            self._wing_status_label.setStyleSheet(
                "padding: 6px; font-weight: bold; font-size: 14px;"
                "background: #4a0e0e; color: #ffb0b0; border-radius: 4px;"
            )
        elif non_core_fail:
            self._wing_status_label.setText(
                f"WING PASS — but {len(non_core_fail)} non-core landmark(s) unreliable: " f"{', '.join(non_core_fail)}"
            )
            self._wing_status_label.setStyleSheet(
                "padding: 6px; font-weight: bold; font-size: 14px;"
                "background: #4a3c0e; color: #ffe08a; border-radius: 4px;"
            )
        else:
            self._wing_status_label.setText("WING PASS — all landmarks reliable")
            self._wing_status_label.setStyleSheet(
                "padding: 6px; font-weight: bold; font-size: 14px;"
                "background: #0e4a1a; color: #a0ffb0; border-radius: 4px;"
            )

    # ---- Image-list pass/fail summary ----
    def _entry_status(self, entry: ImageEntry) -> str:
        """Return 'pass' / 'fail' / 'unknown' for an entry's cached prediction.

        Mirrors `_update_wing_status`'s rule: pass = no core landmark unreliable.
        Non-core failures still count as a pass since the gate only aborts the
        image on core failures. Stub batch results carrying an `error` key (set
        when raise_on_core_fail=False intercepted a LowConfidenceLandmarkError)
        also count as fail.
        """
        pred = entry.prediction
        if pred is None:
            return "unknown"
        if pred.get("error") is not None:
            return "fail"
        reliable = pred.get("reliable") or {}
        if not reliable:
            return "unknown"
        gate_cfg = self._predictor.gate_config if self._predictor else {}
        core = set(gate_cfg.get("core_landmarks", []) or [])
        if any(n in reliable and not reliable[n] for n in core):
            return "fail"
        return "pass"

    def _refresh_image_list_styling(self) -> None:
        """Re-color list items per entry status and update the pass-count label.

        Safe to call any time the prediction cache, gate config, or highlight
        toggle changes. No-op when widgets aren't built yet.
        """
        if not hasattr(self, "_image_list") or not hasattr(self, "_pass_count_label"):
            return
        highlight = self._highlight_failed_chk.isChecked()
        # Warm red on the dark list background; default echoes the list's foreground style.
        fail_brush = QColor("#ff6b6b")
        default_brush = QColor("#dddddd")
        n_pass = n_fail = n_unknown = 0
        for row, entry in enumerate(self._entries):
            status = self._entry_status(entry)
            if status == "pass":
                n_pass += 1
            elif status == "fail":
                n_fail += 1
            else:
                n_unknown += 1
            item = self._image_list.item(row)
            if item is None:
                continue
            if highlight and status == "fail":
                item.setForeground(fail_brush)
            else:
                item.setForeground(default_brush)
        total = len(self._entries)
        if total == 0:
            self._pass_count_label.setText("")
        elif n_unknown == total:
            self._pass_count_label.setText(f"0 / {total} predicted")
        elif n_unknown == 0:
            self._pass_count_label.setText(f"{n_pass} / {total} passed · {n_fail} failed")
        else:
            self._pass_count_label.setText(
                f"{n_pass} passed · {n_fail} failed · {n_unknown} unpredicted"
            )

    def _on_toggle_highlight_failed(self, _checked: bool) -> None:
        """Re-apply list styling when the user toggles the highlight checkbox."""
        self._refresh_image_list_styling()

    def _on_predict_all(self) -> None:
        """Batch-predict every image lacking a cached prediction.

        Strips core_landmarks for the duration of the batch so that core
        failures don't drop heatmaps from the result (matches the lazy-path
        behavior in _predict_for_inspection). Status classification uses the
        original core set since gate_config is restored before refresh.
        """
        if self._predictor is None:
            self.statusBar().showMessage("Load a model before running Predict all.")
            return
        pending = [(i, e) for i, e in enumerate(self._entries) if e.prediction is None]
        if not pending:
            self.statusBar().showMessage("All images already predicted.")
            self._refresh_image_list_styling()
            return

        from landmark_locator.data.psd_loader import imread_any

        # Keep batches small so UI stays responsive and memory stays bounded
        # on the high-resolution training images (5440x3648).
        BATCH = 4
        saved_core = self._predictor.gate_config.get("core_landmarks", [])
        self._predictor.gate_config["core_landmarks"] = []
        try:
            self._predict_all_btn.setEnabled(False)
            done = 0
            for chunk_start in range(0, len(pending), BATCH):
                chunk = pending[chunk_start : chunk_start + BATCH]
                images: list[np.ndarray] = []
                valid: list[tuple[int, ImageEntry]] = []
                for idx, entry in chunk:
                    img = imread_any(entry.path)
                    if img is None:
                        continue
                    images.append(img)
                    valid.append((idx, entry))
                if images:
                    try:
                        results = self._predictor.predict_batch(
                            images, include_unreliable=True, raise_on_core_fail=False
                        )
                    except Exception as exc:
                        self.statusBar().showMessage(f"Predict all failed: {exc}")
                        return
                    for (_idx, entry), res in zip(valid, results):
                        entry.prediction = res
                done += len(chunk)
                self.statusBar().showMessage(f"Predicting… {done} / {len(pending)}")
                self._refresh_image_list_styling()
                QApplication.processEvents()
        finally:
            self._predictor.gate_config["core_landmarks"] = saved_core
            self._predict_all_btn.setEnabled(True)
        self.statusBar().showMessage(f"Predicted {len(pending)} image(s).")
        self._refresh_image_list_styling()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main() -> None:
    """Launch the GUI."""
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    # Apply the outlined-checkbox proxy at the application level so it outlives
    # any window. Stash a Python reference on the app instance so the wrapper
    # isn't garbage-collected before Qt is done with the C++ object — a missing
    # reference causes a use-after-free during QToolBar destruction on close.
    app._outlined_checkbox_style = _OutlinedCheckBoxStyle(app.style())
    app.setStyle(app._outlined_checkbox_style)
    # Dark palette
    from PyQt5.QtGui import QPalette

    palette = QPalette()
    palette.setColor(QPalette.Window, QColor(30, 30, 30))
    palette.setColor(QPalette.WindowText, QColor(220, 220, 220))
    palette.setColor(QPalette.Base, QColor(37, 37, 38))
    palette.setColor(QPalette.AlternateBase, QColor(45, 45, 48))
    palette.setColor(QPalette.Text, QColor(220, 220, 220))
    palette.setColor(QPalette.Button, QColor(51, 51, 51))
    palette.setColor(QPalette.ButtonText, QColor(220, 220, 220))
    palette.setColor(QPalette.Highlight, QColor(42, 130, 218))
    palette.setColor(QPalette.HighlightedText, QColor(255, 255, 255))
    app.setPalette(palette)

    window = LandmarkGUI()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
