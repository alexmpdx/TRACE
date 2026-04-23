"""PyQt5 GUI for inspecting landmark predictions and ground truth."""

from __future__ import annotations

import io
import json
import re
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
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QListWidget,
    QMainWindow,
    QMessageBox,
    QProgressDialog,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSlider,
    QSplitter,
    QStatusBar,
    QTableWidget,
    QTableWidgetItem,
    QTextEdit,
    QToolBar,
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
IMAGE_EXTENSIONS = {".tif", ".tiff", ".png", ".jpg", ".jpeg", ".bmp", ".psd", ".psb"}


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
# ImageWidget (inline copy from WingVeinAnalyzer/gui/image_widget.py)
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

    def __init__(self, parent=None):
        super().__init__(parent)
        self._landmark_order: list[str] = []
        self._qcolors: dict[str, QColor] = {}
        self.setFixedHeight(50)

    def set_landmarks(self, landmark_order: list[str]) -> None:
        """Update the legend with a new set of landmarks."""
        self._landmark_order = landmark_order
        self._qcolors = _make_qcolors(landmark_order)
        self.setFixedHeight(len(landmark_order) * 22 + 50)
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

            self.finished_training.emit(str(run_dir / "checkpoints"))
        except Exception as e:
            self.error.emit(str(e))
        finally:
            capture.flush()
            sys.stdout = old_stdout


class TrainingDialog(QDialog):
    """Modal dialog showing live training log output and error chart."""

    def __init__(self, model_name: str = "", landmark_order: list[str] | None = None, max_epochs: int = 0, parent=None):
        super().__init__(parent)
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
) -> np.ndarray:
    """Ring each predicted landmark in green (pass) or red (fail).

    The numeric gate details live in the info table — the overlay just gives an
    at-a-glance pass/fail for each landmark on the image.
    """
    vis = image.copy()
    landmarks = prediction.get("landmarks", {})
    reliable = prediction.get("reliable", {})

    h, w = vis.shape[:2]
    base = min(h, w)
    # Ring sits just outside the landmark dot drawn by draw_landmarks_on_image
    # (radius ~= base/500 there). Add a small gap, then outline.
    dot_radius = max(4, int(base / 500))
    ring_radius = dot_radius + max(2, int(base / 800))
    thick = max(2, int(round(base / 1200.0)))

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
_STRICT_DEFAULTS = {"peak": 0.20, "sharpness": 1.25, "second_peak_ratio": 0.65}
_PERMISSIVE_DEFAULTS = {"peak": 0.10, "sharpness": 1.15, "second_peak_ratio": 0.80}


class GateConfigDialog(QDialog):
    """Per-landmark threshold tier + abort checkbox editor."""

    def __init__(self, parent: "LandmarkGUI", gate_config: dict, landmark_order: list[str]) -> None:
        super().__init__(parent)
        self.setWindowTitle("Confidence Gate Configuration")
        self.resize(780, 480)
        self._landmark_order = list(landmark_order)
        # Deep-copy so Cancel discards changes
        self._cfg = json.loads(json.dumps(gate_config))
        # Per-row widgets keyed by landmark name
        self._rows: dict[str, dict] = {}

        root = QVBoxLayout(self)
        hint = QLabel(
            "Per-landmark confidence gate. 'Permissive' uses the global defaults; "
            "'Strict' clamps tighter thresholds (crossvein presets); 'Custom' lets you "
            "set each metric. Check 'Abort' to fail the whole image when this landmark misses."
        )
        hint.setWordWrap(True)
        root.addWidget(hint)

        # Header row
        grid = QGridLayout()
        headers = ["Landmark", "Tier", "peak ≥", "sharp ≥", "sp_ratio ≤", "Abort"]
        for col, h in enumerate(headers):
            lbl = QLabel(h)
            lbl.setStyleSheet("font-weight: bold;")
            grid.addWidget(lbl, 0, col)

        core = set(self._cfg.get("core_landmarks", []) or [])
        peak_pl = self._cfg.get("peak", {}).get("per_landmark", {}) or {}
        sharp_pl = self._cfg.get("sharpness", {}).get("per_landmark", {}) or {}
        spr_pl = self._cfg.get("second_peak_ratio", {}).get("per_landmark", {}) or {}
        for i, name in enumerate(self._landmark_order, start=1):
            grid.addWidget(QLabel(name), i, 0)

            combo = QComboBox()
            combo.addItems(["Permissive", "Strict", "Custom"])
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

            has_override = name in peak_pl or name in sharp_pl or name in spr_pl
            cur_peak = peak_pl.get(name, self._cfg["peak"]["global"])
            cur_sharp = sharp_pl.get(name, self._cfg["sharpness"]["global"])
            cur_spr = spr_pl.get(name, self._cfg["second_peak_ratio"]["global"])
            peak_spin.setValue(float(cur_peak))
            sharp_spin.setValue(float(cur_sharp))
            spr_spin.setValue(float(cur_spr))

            if has_override:
                # Detect strict tier by equality to strict defaults
                if (
                    peak_pl.get(name) == _STRICT_DEFAULTS["peak"]
                    and sharp_pl.get(name) == _STRICT_DEFAULTS["sharpness"]
                    and spr_pl.get(name) == _STRICT_DEFAULTS["second_peak_ratio"]
                ):
                    combo.setCurrentText("Strict")
                else:
                    combo.setCurrentText("Custom")
            else:
                combo.setCurrentText("Permissive")

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
        btn_load = QPushButton("Import YAML…")
        btn_load.clicked.connect(self._on_import)
        btn_save = QPushButton("Export YAML…")
        btn_save.clicked.connect(self._on_export)
        buttons_row.addWidget(btn_load)
        buttons_row.addWidget(btn_save)
        buttons_row.addStretch(1)
        root.addLayout(buttons_row)

        bb = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        bb.accepted.connect(self.accept)
        bb.rejected.connect(self.reject)
        root.addWidget(bb)

    def _apply_tier_to_row(self, name: str, tier: str) -> None:
        row = self._rows[name]
        if tier == "Strict":
            row["peak"].setValue(_STRICT_DEFAULTS["peak"])
            row["sharpness"].setValue(_STRICT_DEFAULTS["sharpness"])
            row["second_peak_ratio"].setValue(_STRICT_DEFAULTS["second_peak_ratio"])
        elif tier == "Permissive":
            row["peak"].setValue(self._cfg["peak"]["global"])
            row["sharpness"].setValue(self._cfg["sharpness"]["global"])
            row["second_peak_ratio"].setValue(self._cfg["second_peak_ratio"]["global"])
        self._sync_row_editability(name)

    def _sync_row_editability(self, name: str) -> None:
        row = self._rows[name]
        editable = row["combo"].currentText() == "Custom"
        for key in ("peak", "sharpness", "second_peak_ratio"):
            row[key].setEnabled(editable)

    def result_override(self) -> dict:
        """Build a confidence-override dict from the current widget state."""
        peak_pl: dict[str, float] = {}
        sharp_pl: dict[str, float] = {}
        spr_pl: dict[str, float] = {}
        core: list[str] = []
        for name, row in self._rows.items():
            tier = row["combo"].currentText()
            if tier != "Permissive":
                peak_pl[name] = float(row["peak"].value())
                sharp_pl[name] = float(row["sharpness"].value())
                spr_pl[name] = float(row["second_peak_ratio"].value())
            if row["abort"].isChecked():
                core.append(name)
        return {
            "peak": {"global": self._cfg["peak"]["global"], "per_landmark": peak_pl},
            "sharpness": {"global": self._cfg["sharpness"]["global"], "per_landmark": sharp_pl},
            "second_peak_ratio": {
                "global": self._cfg["second_peak_ratio"]["global"],
                "per_landmark": spr_pl,
            },
            "second_peak_suppression_radius_px": self._cfg.get("second_peak_suppression_radius_px", 30),
            "core_landmarks": sorted(core),
        }

    def _on_export(self) -> None:
        path, _ = QFileDialog.getSaveFileName(self, "Export gate YAML", "", "YAML (*.yaml *.yml)")
        if not path:
            return
        doc = {"confidence": self.result_override()}
        Path(path).write_text(yaml.safe_dump(doc, sort_keys=False))
        QMessageBox.information(self, "Exported", f"Wrote {path}")

    def _on_import(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Import gate YAML", "", "YAML (*.yaml *.yml)")
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
        for name, row in self._rows.items():
            if name in peak_pl or name in sharp_pl or name in spr_pl:
                row["combo"].setCurrentText("Custom")
                row["peak"].setValue(float(peak_pl.get(name, row["peak"].value())))
                row["sharpness"].setValue(float(sharp_pl.get(name, row["sharpness"].value())))
                row["second_peak_ratio"].setValue(float(spr_pl.get(name, row["second_peak_ratio"].value())))
            else:
                row["combo"].setCurrentText("Permissive")
            row["abort"].setChecked(name in core)
            self._sync_row_editability(name)


# ---------------------------------------------------------------------------
# Main GUI
# ---------------------------------------------------------------------------
class LandmarkGUI(QMainWindow):
    """Main window for landmark verification."""

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("LandmarkLocator — Verification GUI")
        self.resize(1400, 900)
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

        act_model = QAction("Load Model", self)
        act_model.triggered.connect(self._on_load_model)
        tb.addAction(act_model)

        act_ensemble = QAction("Load Fold Folder", self)
        act_ensemble.setToolTip("Load a folder of best_fold*.pt checkpoints and run ensemble prediction.")
        act_ensemble.triggered.connect(self._on_load_fold_folder)
        tb.addAction(act_ensemble)

        act_images = QAction("Load Images", self)
        act_images.triggered.connect(self._on_load_folder)
        tb.addAction(act_images)

        act_gt = QAction("Set GT Dir", self)
        act_gt.triggered.connect(self._on_set_gt_dir)
        tb.addAction(act_gt)

        act_save = QAction("Save", self)
        act_save.triggered.connect(self._on_save_all)
        tb.addAction(act_save)

        act_gate = QAction("Gate Config…", self)
        act_gate.triggered.connect(self._on_edit_gate_config)
        tb.addAction(act_gate)

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

        self._image_list = QListWidget()
        self._image_list.currentRowChanged.connect(self._on_image_selected)
        left_layout.addWidget(self._image_list)

        self._legend = LegendWidget()
        left_layout.addWidget(self._legend)
        left.setFixedWidth(220)
        splitter.addWidget(left)

        # -- Center panel: image view + info table --
        center = QWidget()
        center_layout = QVBoxLayout(center)
        center_layout.setContentsMargins(0, 0, 0, 0)

        self._image_widget = ImageWidget()
        center_layout.addWidget(self._image_widget, stretch=3)

        # Overall wing pass/fail status line (under the image).
        self._wing_status_label = QLabel("")
        self._wing_status_label.setAlignment(Qt.AlignCenter)
        self._wing_status_label.setStyleSheet(
            "padding: 6px; font-weight: bold; font-size: 14px;" "background: #252526; color: #aaa; border-radius: 4px;"
        )
        self._wing_status_label.setFixedHeight(32)
        center_layout.addWidget(self._wing_status_label)

        # Heatmap overlay opacity slider
        slider_row = QHBoxLayout()
        slider_label = QLabel("Heatmap opacity:")
        slider_label.setStyleSheet("color: #aaa; font-size: 10px;")
        slider_row.addWidget(slider_label)
        self._opacity_slider = QSlider(Qt.Horizontal)
        self._opacity_slider.setRange(0, 100)
        self._opacity_slider.setValue(50)
        self._opacity_slider.setFixedHeight(20)
        self._opacity_slider.valueChanged.connect(self._on_opacity_changed)
        slider_row.addWidget(self._opacity_slider)
        self._opacity_label = QLabel("50%")
        self._opacity_label.setStyleSheet("color: #aaa; font-size: 10px;")
        self._opacity_label.setFixedWidth(32)
        slider_row.addWidget(self._opacity_label)
        center_layout.addLayout(slider_row)

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
        dlg = GateConfigDialog(self, self._predictor.gate_config, self._predictor.landmark_order)
        if dlg.exec_() != QDialog.Accepted:
            return
        override = dlg.result_override()
        # update_gate_config does a deep-merge, which would keep stale per_landmark
        # entries the user removed. Replace the gate_config wholesale instead.
        self._predictor.gate_config = override
        # Invalidate cached predictions so they re-run under the new gate
        for entry in self._entries:
            entry.prediction = None
        if self._current_idx >= 0:
            self._on_image_selected(self._current_idx)
        self.statusBar().showMessage("Gate config updated.")

    # ---- Actions ----
    def _on_load_model(self) -> None:
        """Load a model checkpoint via file dialog."""
        path, _ = QFileDialog.getOpenFileName(
            self, "Select Checkpoint", str(_project_root / "checkpoints"), "PyTorch (*.pt *.pth)"
        )
        if not path:
            return
        try:
            from landmark_locator.inference.predict import LandmarkPredictor

            self._predictor = LandmarkPredictor(Path(path))
            # Update landmark order from the loaded model
            self._set_landmark_order(
                self._predictor.landmark_order,
                self._predictor.geojson_to_landmark,
            )
            self._model_label.setText(f"Model: {Path(path).name}")
            self.statusBar().showMessage(f"Model loaded: {Path(path).name}")
            # Clear cached predictions so they re-run
            for entry in self._entries:
                entry.prediction = None
            # Refresh current
            if self._current_idx >= 0:
                self._on_image_selected(self._current_idx)
        except Exception as e:
            self.statusBar().showMessage(f"Failed to load model: {e}")

    def _on_load_fold_folder(self) -> None:
        """Load a folder of best_fold*.pt checkpoints for ensemble prediction."""
        from landmark_locator.inference.predict import _find_fold_checkpoints, make_predictor

        start = str(_project_root / "trained_models")
        folder = QFileDialog.getExistingDirectory(
            self, "Select Fold Checkpoint Folder (contains best_fold0.pt, ...)", start
        )
        if not folder:
            return
        folder_path = Path(folder)
        checkpoints = _find_fold_checkpoints(folder_path)
        if not checkpoints:
            QMessageBox.warning(self, "No checkpoints", f"No best_fold*.pt in {folder_path}")
            return
        try:
            self._predictor = make_predictor(folder_path)
            self._set_landmark_order(
                self._predictor.landmark_order,
                self._predictor.geojson_to_landmark,
            )
            self._model_label.setText(f"Ensemble: {folder_path.name}/ ({len(checkpoints)} folds)")
            self.statusBar().showMessage(
                f"Ensemble loaded from {folder_path.name}: {len(checkpoints)} folds — predictions will average all of them"
            )
            for entry in self._entries:
                entry.prediction = None
            if self._current_idx >= 0:
                self._on_image_selected(self._current_idx)
        except Exception as e:
            self.statusBar().showMessage(f"Failed to load ensemble: {e}")

    def _on_toggle_show_gate(self, checked: bool) -> None:
        """Redraw the current image with/without the gate overlay."""
        if self._current_idx >= 0:
            self._on_image_selected(self._current_idx)

    def _on_load_folder(self) -> None:
        """Load images from a user-selected folder."""
        start = str(self._img_dir) if self._img_dir else str(_project_root)
        folder = QFileDialog.getExistingDirectory(self, "Select Image Folder", start)
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
        folder = QFileDialog.getExistingDirectory(self, "Select GT Annotation Folder", start)
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
        folder = QFileDialog.getExistingDirectory(self, "Select Output Directory", str(_project_root))
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
                vis = draw_landmarks_on_image(image, preds, entry.gt, landmark_order=self._landmark_order)
                out_path = self._output_dir / f"{entry.path.stem}_landmarks.jpg"
                cv2.imwrite(str(out_path), vis)

            if save_geojson and preds:
                features = []
                reverse_map = {v: k for k, v in self._geojson_to_landmark.items()}
                for name, (x, y) in preds.items():
                    geojson_name = reverse_map.get(name, name)
                    features.append(
                        {
                            "type": "Feature",
                            "geometry": {"type": "Point", "coordinates": [x, y]},
                            "properties": {"classification": {"name": geojson_name}},
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
        """Handle successful training completion. ckpt_dir contains best_fold*.pt."""
        self._train_dialog.append_log(f"\nTraining complete! Fold checkpoints: {ckpt_dir}")
        self._train_dialog.enable_close()
        run_dir = Path(ckpt_dir).parent  # trained_models/<name>/
        # Save the error chart into the run folder
        chart_path = run_dir / "training_chart.png"
        try:
            self._train_dialog._fig.savefig(str(chart_path), dpi=150, facecolor="#1e1e1e")
            self._train_dialog.append_log(f"Error chart saved: {chart_path}")
        except Exception as e:
            self._train_dialog.append_log(f"Warning: could not save chart: {e}")
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

        if self._entries:
            self._image_list.setCurrentRow(0)

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
        vis = draw_landmarks_on_image(image, preds, entry.gt, landmark_order=self._landmark_order)
        if self._show_gate_chk.isChecked() and entry.prediction is not None and "reliable" in entry.prediction:
            vis = _draw_gate_overlay(vis, entry.prediction, self._landmark_order)
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


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main() -> None:
    """Launch the GUI."""
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
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
