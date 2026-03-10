"""PyQt5 GUI for inspecting landmark predictions and ground truth."""

from __future__ import annotations

import io
import json
import sys
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
    QDialog,
    QFileDialog,
    QGraphicsPixmapItem,
    QGraphicsScene,
    QGraphicsView,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QListWidget,
    QMainWindow,
    QProgressDialog,
    QPushButton,
    QScrollArea,
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
IMAGE_EXTENSIONS = {".tif", ".tiff", ".png", ".jpg", ".jpeg", ".bmp"}


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
    """Redirect sys.stdout writes to a Qt signal for live GUI logging."""

    def __init__(self, signal: pyqtSignal):
        super().__init__()
        self._signal = signal
        self._original = sys.stdout

    def write(self, text: str) -> int:
        if text and text.strip():
            self._signal.emit(text)
        return len(text) if text else 0

    def flush(self) -> None:
        pass


class TrainingThread(QThread):
    """Background worker that trains a single fold."""

    progress = pyqtSignal(str)
    epoch_data = pyqtSignal(object)  # dict with epoch, mean_error, landmark_errors
    finished_training = pyqtSignal(str)  # checkpoint path
    error = pyqtSignal(str)

    def __init__(self, model_name: str, gt_dir: Optional[Path] = None, parent=None):
        super().__init__(parent)
        self._model_name = model_name
        self._gt_dir = gt_dir

    def run(self) -> None:
        """Execute training fold 0."""
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

            output_dir = _project_root / "trained_models"
            output_dir.mkdir(parents=True, exist_ok=True)

            device = get_device()
            print(f"Using device: {device}")

            if self._gt_dir:
                annotation_dir = self._gt_dir
                cfg["data"]["annotation_dir"] = str(annotation_dir)
            else:
                annotation_dir = _project_root / cfg["data"]["annotation_dir"]

            _populate_landmark_config(cfg, annotation_dir)
            splits = create_cv_splits(annotation_dir, cfg["cv"]["n_folds"])
            train_idx, val_idx = splits[0]
            print(f"Fold 0: {len(train_idx)} train, {len(val_idx)} val")

            def _on_epoch(epoch, mean_error, landmark_errors):
                self.epoch_data.emit(
                    {
                        "epoch": epoch,
                        "mean_error": mean_error,
                        "landmark_errors": landmark_errors.copy(),
                    }
                )

            train_fold(
                cfg,
                0,
                train_idx,
                val_idx,
                output_dir,
                device,
                epoch_callback=_on_epoch,
                checkpoint_name=self._model_name,
            )

            name = self._model_name if self._model_name.endswith(".pt") else self._model_name + ".pt"
            dest = output_dir / "checkpoints" / name
            self.finished_training.emit(str(dest))
        except Exception as e:
            self.error.emit(str(e))
        finally:
            sys.stdout = old_stdout


class TrainingDialog(QDialog):
    """Modal dialog showing live training log output and error chart."""

    def __init__(self, model_name: str = "", landmark_order: list[str] | None = None, parent=None):
        super().__init__(parent)
        title = f"Training — {model_name}" if model_name else "Training"
        self.setWindowTitle(title)
        self.resize(820, 600)
        layout = QVBoxLayout(self)

        self._landmark_order = landmark_order or []
        self._qcolors = _make_qcolors(self._landmark_order)

        # Per-landmark data series (must init before _setup_chart)
        self._epochs: list[int] = []
        self._series: dict[str, list[float]] = {name: [] for name in self._landmark_order}
        self._mean_series: list[float] = []
        self._lines: dict[str, object] = {}

        # Matplotlib chart with navigation toolbar for pan/zoom
        self._fig = Figure(figsize=(7, 3), facecolor="#1e1e1e")
        self._ax = self._fig.add_subplot(111)
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
        ax = self._ax
        ax.set_facecolor("#252526")
        ax.set_xlabel("Epoch", color="#aaa", fontsize=9)
        ax.set_ylabel("Pixel Error", color="#aaa", fontsize=9)
        ax.set_title("Validation Error by Landmark", color="#ddd", fontsize=10)
        ax.tick_params(colors="#888", labelsize=8)
        for spine in ax.spines.values():
            spine.set_color("#444")
        ax.grid(True, color="#333", linewidth=0.5)

        # Create lines for each landmark + mean
        for name in self._landmark_order:
            qc = self._qcolors.get(name, QColor(200, 200, 200))
            color = qc.name()
            display = _make_display_name(name)
            (line,) = ax.plot([], [], color=color, linewidth=1.2, label=display)
            self._lines[name] = line
        (mean_line,) = ax.plot([], [], color="#ffffff", linewidth=2, linestyle="--", label="Mean")
        self._lines["_mean"] = mean_line

        ax.legend(loc="upper right", fontsize=7, facecolor="#333", edgecolor="#555", labelcolor="#ccc")
        self._fig.tight_layout()

        # Track when user manually zooms/pans so we stop auto-scaling
        self._canvas.mpl_connect("button_press_event", lambda e: setattr(self, "_user_zoomed", True))

    def update_chart(self, data: dict) -> None:
        """Add one epoch's data and redraw the chart."""
        epoch = data["epoch"]
        self._epochs.append(epoch)
        self._mean_series.append(data["mean_error"])
        for name in self._landmark_order:
            if name not in self._series:
                self._series[name] = []
            self._series[name].append(data["landmark_errors"].get(name, 0.0))

        # Update line data
        for name in self._landmark_order:
            self._lines[name].set_data(self._epochs, self._series[name])
        self._lines["_mean"].set_data(self._epochs, self._mean_series)

        # Auto-rescale unless user has manually zoomed/panned
        if not self._user_zoomed:
            self._ax.set_xlim(0, max(self._epochs[-1], 1))
            all_vals = self._mean_series + [v for s in self._series.values() for v in s]
            if all_vals:
                ymax = max(all_vals) * 1.1
                self._ax.set_ylim(0, max(ymax, 1))

        self._canvas.draw_idle()

    def append_log(self, text: str) -> None:
        """Append a line to the log view."""
        self._log.append(text)
        self._log.verticalScrollBar().setValue(self._log.verticalScrollBar().maximum())

    def enable_close(self) -> None:
        """Enable the close button after training completes."""
        self._close_btn.setEnabled(True)


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

        act_images = QAction("Load Images", self)
        act_images.triggered.connect(self._on_load_folder)
        tb.addAction(act_images)

        act_gt = QAction("Set GT Dir", self)
        act_gt.triggered.connect(self._on_set_gt_dir)
        tb.addAction(act_gt)

        act_output = QAction("Set Output", self)
        act_output.triggered.connect(self._on_set_output)
        tb.addAction(act_output)

        act_train = QAction("Train Model", self)
        act_train.triggered.connect(self._on_train_model)
        tb.addAction(act_train)

        act_save = QAction("Save All", self)
        act_save.triggered.connect(self._on_save_all)
        tb.addAction(act_save)

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

        self._info_table = QTableWidget(0, 6)
        self._info_table.setHorizontalHeaderLabels(["Landmark", "Pred X", "Pred Y", "GT X", "GT Y", "Error (px)"])
        self._info_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self._info_table.setEditTriggers(QTableWidget.NoEditTriggers)
        self._info_table.setMaximumHeight(180)
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
        """Auto-load training_data_pics + training_data if they exist."""
        img_dir = _project_root / "training_data_pics"
        gt_dir = _project_root / "training_data"
        if img_dir.is_dir():
            self._img_dir = img_dir
            # Discover landmarks from annotations
            if gt_dir.is_dir():
                self._gt_dir = gt_dir
                landmark_order, geojson_to_landmark = discover_landmarks(gt_dir)
                if landmark_order:
                    self._set_landmark_order(landmark_order, geojson_to_landmark)
            self._load_image_folder(img_dir, self._gt_dir)
            self.statusBar().showMessage(f"Auto-loaded {len(self._entries)} images from training_data_pics/")

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
            self.statusBar().showMessage(f"Model loaded: {Path(path).name}")
            # Clear cached predictions so they re-run
            for entry in self._entries:
                entry.prediction = None
            # Refresh current
            if self._current_idx >= 0:
                self._on_image_selected(self._current_idx)
        except Exception as e:
            self.statusBar().showMessage(f"Failed to load model: {e}")

    def _on_load_folder(self) -> None:
        """Load images from a user-selected folder."""
        folder = QFileDialog.getExistingDirectory(self, "Select Image Folder", str(_project_root))
        if not folder:
            return
        folder = Path(folder)
        self._img_dir = folder
        # Look for GT annotations: sibling training_data/ or same folder
        gt_dir = None
        for name in ("training_data", "training_data_new"):
            candidate = folder.parent / name
            if candidate.is_dir():
                gt_dir = candidate
                break
        if gt_dir is None and list(folder.glob("*.geojson")):
            gt_dir = folder
        self._gt_dir = gt_dir
        self._apply_gt_dir()
        self._load_image_folder(folder, self._gt_dir)
        self.statusBar().showMessage(f"Loaded {len(self._entries)} images from {folder.name}/")

    def _on_set_gt_dir(self) -> None:
        """Let the user pick a ground-truth annotation folder."""
        start = str(self._gt_dir) if self._gt_dir else str(_project_root)
        folder = QFileDialog.getExistingDirectory(self, "Select GT Annotation Folder", start)
        if not folder:
            return
        self._gt_dir = Path(folder)
        self._apply_gt_dir()
        # Re-match GT paths to existing image entries
        for entry in self._entries:
            entry.geojson_path = None
            entry.gt = None
            entry.gt_heatmaps = None
            entry._gt_loaded = False
            candidate = self._gt_dir / (entry.path.name + ".geojson")
            if candidate.exists():
                entry.geojson_path = candidate
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
        """Save annotated images for all entries."""
        if not self._entries:
            self.statusBar().showMessage("No images loaded")
            return
        if self._output_dir is None:
            self._on_set_output()
            if self._output_dir is None:
                return
        self._output_dir.mkdir(parents=True, exist_ok=True)

        progress = QProgressDialog("Saving annotated images...", "Cancel", 0, len(self._entries), self)
        progress.setWindowModality(Qt.WindowModal)
        progress.setMinimumDuration(0)

        saved = 0
        for i, entry in enumerate(self._entries):
            if progress.wasCanceled():
                break
            progress.setValue(i)

            image = cv2.imread(str(entry.path))
            if image is None:
                continue

            entry.load_gt()
            preds = {}
            if entry.prediction:
                preds = entry.prediction["landmarks"]
            elif self._predictor:
                try:
                    entry.prediction = self._predictor.predict(image)
                    preds = entry.prediction["landmarks"]
                except Exception:
                    pass

            vis = draw_landmarks_on_image(image, preds, entry.gt, landmark_order=self._landmark_order)
            out_path = self._output_dir / f"{entry.path.stem}_landmarks.jpg"
            cv2.imwrite(str(out_path), vis)

            # Write GeoJSON with predicted landmarks
            if preds:
                features = []
                for name, (x, y) in preds.items():
                    # Reverse lookup: internal name → GeoJSON name
                    reverse_map = {v: k for k, v in self._geojson_to_landmark.items()}
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
        self.statusBar().showMessage(f"Saved {saved} images + GeoJSON to {self._output_dir}")

    def _on_train_model(self) -> None:
        """Launch training fold 0 in a background thread with live log dialog."""
        # Validate that GT annotations exist
        gt_count = sum(1 for e in self._entries if e.geojson_path and e.geojson_path.exists())
        if gt_count == 0:
            self.statusBar().showMessage("No ground-truth annotations found — cannot train")
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

        self._train_dialog = TrainingDialog(model_name, self._landmark_order, self)
        self._train_thread = TrainingThread(model_name, self._gt_dir)
        self._train_thread.progress.connect(self._train_dialog.append_log)
        self._train_thread.epoch_data.connect(self._train_dialog.update_chart)
        self._train_thread.finished_training.connect(self._on_training_finished)
        self._train_thread.error.connect(self._on_training_error)
        self._train_thread.start()
        self._train_dialog.append_log(f"Starting training '{model_name}' with {gt_count} annotated images...")
        self._train_dialog.exec_()

    def _on_training_finished(self, ckpt_path: str) -> None:
        """Handle successful training completion."""
        self._train_dialog.append_log(f"\nTraining complete! Checkpoint: {ckpt_path}")
        self._train_dialog.enable_close()
        # Save the error chart next to the checkpoint
        chart_path = Path(ckpt_path).with_suffix(".png")
        try:
            self._train_dialog._fig.savefig(str(chart_path), dpi=150, facecolor="#1e1e1e")
            self._train_dialog.append_log(f"Error chart saved: {chart_path}")
        except Exception as e:
            self._train_dialog.append_log(f"Warning: could not save chart: {e}")
        # Auto-load the trained model
        try:
            from landmark_locator.inference.predict import LandmarkPredictor

            self._predictor = LandmarkPredictor(Path(ckpt_path))
            self._set_landmark_order(
                self._predictor.landmark_order,
                self._predictor.geojson_to_landmark,
            )
            for entry in self._entries:
                entry.prediction = None
            if self._current_idx >= 0:
                self._on_image_selected(self._current_idx)
            self.statusBar().showMessage(f"Model loaded from training: {Path(ckpt_path).name}")
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

        for img_path in image_files:
            geojson_path = None
            if gt_dir:
                # Convention: image.tif → image.tif.geojson
                candidate = gt_dir / (img_path.name + ".geojson")
                if candidate.exists():
                    geojson_path = candidate
            entry = ImageEntry(path=img_path, geojson_path=geojson_path)
            self._entries.append(entry)
            self._image_list.addItem(img_path.name)

        if self._entries:
            self._image_list.setCurrentRow(0)

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
        image = cv2.imread(str(entry.path))
        if image is None:
            self.statusBar().showMessage(f"Failed to load {entry.path.name}")
            self._image_widget.clear_image()
            return

        orig_h, orig_w = image.shape[:2]

        # Load ground truth
        entry.load_gt()
        if entry.gt and entry.gt_heatmaps is None and self._landmark_order:
            entry.gt_heatmaps = generate_gt_heatmaps(entry.gt, self._landmark_order, orig_w=orig_w, orig_h=orig_h)

        # Run prediction if model available and not cached
        if self._predictor and entry.prediction is None:
            try:
                entry.prediction = self._predictor.predict(image)
            except Exception as e:
                self.statusBar().showMessage(f"Prediction failed: {e}")

        # Draw overlay and cache base visualization
        preds = entry.prediction["landmarks"] if entry.prediction else {}
        vis = draw_landmarks_on_image(image, preds, entry.gt, landmark_order=self._landmark_order)
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
        """Populate the per-landmark info table."""
        for i, name in enumerate(self._landmark_order):
            # Pred coords
            if entry.prediction and name in entry.prediction["landmarks"]:
                px, py = entry.prediction["landmarks"][name]
                conf = entry.prediction["confidences"][name]
                self._info_table.setItem(i, 1, QTableWidgetItem(f"{px:.1f}"))
                self._info_table.setItem(i, 2, QTableWidgetItem(f"{py:.1f}"))
            else:
                self._info_table.setItem(i, 1, QTableWidgetItem("N/A"))
                self._info_table.setItem(i, 2, QTableWidgetItem("N/A"))
                conf = None

            # GT coords
            if entry.gt and name in entry.gt:
                gx, gy = entry.gt[name]
                self._info_table.setItem(i, 3, QTableWidgetItem(f"{gx:.1f}"))
                self._info_table.setItem(i, 4, QTableWidgetItem(f"{gy:.1f}"))
            else:
                self._info_table.setItem(i, 3, QTableWidgetItem("N/A"))
                self._info_table.setItem(i, 4, QTableWidgetItem("N/A"))

            # Error
            if entry.prediction and name in entry.prediction["landmarks"] and entry.gt and name in entry.gt:
                px, py = entry.prediction["landmarks"][name]
                gx, gy = entry.gt[name]
                err = np.sqrt((px - gx) ** 2 + (py - gy) ** 2)
                item = QTableWidgetItem(f"{err:.1f}")
                if err > 100:
                    item.setForeground(QColor(255, 80, 80))
                elif err > 50:
                    item.setForeground(QColor(255, 200, 80))
                else:
                    item.setForeground(QColor(80, 255, 80))
                self._info_table.setItem(i, 5, item)
            else:
                self._info_table.setItem(i, 5, QTableWidgetItem("N/A"))


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
