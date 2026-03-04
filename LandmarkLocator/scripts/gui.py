"""PyQt5 GUI for inspecting landmark predictions and ground truth."""

from __future__ import annotations

import io
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import yaml

import cv2
import numpy as np

_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from PyQt5.QtCore import QThread, Qt, pyqtSignal
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
    QSplitter,
    QStatusBar,
    QTableWidget,
    QTableWidgetItem,
    QTextEdit,
    QToolBar,
    QVBoxLayout,
    QWidget,
)

from data.dataset import LANDMARK_ORDER
from scripts.visualize import LANDMARK_COLORS, draw_landmarks_on_image, load_ground_truth

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
MODEL_W, MODEL_H = 512, 352
HEATMAP_SIGMA = 5
HEATMAP_THUMB_W, HEATMAP_THUMB_H = 240, 165
IMAGE_EXTENSIONS = {".tif", ".tiff", ".png", ".jpg", ".jpeg"}

# Pretty display names for landmarks
LANDMARK_DISPLAY = {
    "subcostal_break": "Subcostal Break",
    "alula_notch": "Alula Notch",
    "l1_rs_junction": "L1-Rs Junction",
    "l4_l5_junction": "L4-L5 Junction",
    "wing_tip": "Wing Tip",
}

# BGR → QColor (RGB)
LANDMARK_QCOLORS: dict[str, QColor] = {
    name: QColor(bgr[2], bgr[1], bgr[0]) for name, bgr in LANDMARK_COLORS.items()
}


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
    gt_heatmaps: Optional[np.ndarray] = None  # (5, MODEL_H, MODEL_W)
    _gt_loaded: bool = field(default=False, repr=False)

    def load_gt(self) -> None:
        """Load ground truth from GeoJSON if available (cached)."""
        if self._gt_loaded:
            return
        self._gt_loaded = True
        if self.geojson_path and self.geojson_path.exists():
            self.gt = load_ground_truth(self.geojson_path)
            self.gt_heatmaps = generate_gt_heatmaps(self.gt)


# ---------------------------------------------------------------------------
# Standalone heatmap generation (mirrors LandmarkDataset._generate_heatmap)
# ---------------------------------------------------------------------------
def generate_gt_heatmaps(
    gt: dict[str, tuple[float, float]],
    width: int = MODEL_W,
    height: int = MODEL_H,
    sigma: float = HEATMAP_SIGMA,
    orig_w: Optional[int] = None,
    orig_h: Optional[int] = None,
) -> np.ndarray:
    """Render Gaussian heatmaps from ground-truth coordinates at model resolution."""
    heatmaps = np.zeros((len(LANDMARK_ORDER), height, width), dtype=np.float32)
    for i, name in enumerate(LANDMARK_ORDER):
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
        gaussian = np.exp(-((xx - kx) ** 2 + (yy - ky) ** 2) / (2 * sigma ** 2))
        heatmaps[i, y0:y1, x0:x1] = gaussian
    return heatmaps


# ---------------------------------------------------------------------------
# Heatmap → QPixmap helper
# ---------------------------------------------------------------------------
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
    colored = cv2.applyColorMap(norm, cv2.COLORMAP_HOT)  # BGR
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
        self.setFixedHeight(len(LANDMARK_ORDER) * 22 + 50)

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

        for name in LANDMARK_ORDER:
            color = LANDMARK_QCOLORS.get(name, QColor(200, 200, 200))
            p.setBrush(color)
            p.setPen(Qt.NoPen)
            p.drawRect(10, y, 14, 14)
            p.setPen(Qt.lightGray)
            p.drawText(30, y + 12, LANDMARK_DISPLAY.get(name, name))
            y += 20
        p.end()


# ---------------------------------------------------------------------------
# HeatmapPanel — right-side scrollable heatmap thumbnails
# ---------------------------------------------------------------------------
class HeatmapPanel(QScrollArea):
    """Scrollable panel showing predicted heatmaps with optional GT cross overlay."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWidgetResizable(True)
        self.setMinimumWidth(HEATMAP_THUMB_W + 40)
        self._container = QWidget()
        self._layout = QVBoxLayout(self._container)
        self._layout.setAlignment(Qt.AlignTop)
        self.setWidget(self._container)

        # Toggle for GT cross overlay
        self._show_gt = QCheckBox("Show GT cross")
        self._show_gt.setChecked(True)
        self._show_gt.setStyleSheet("color: #ddd; padding: 4px;")
        self._show_gt.toggled.connect(self._refresh)
        self._layout.addWidget(self._show_gt)

        self._pred_labels: dict[str, QLabel] = {}

        for name in LANDMARK_ORDER:
            display = LANDMARK_DISPLAY.get(name, name)
            color = LANDMARK_QCOLORS.get(name, QColor(200, 200, 200))

            header = QLabel(f"<b style='color:{color.name()}'>{display}</b>")
            header.setStyleSheet("padding-top: 6px;")
            self._layout.addWidget(header)

            pred_lbl = QLabel("No model")
            pred_lbl.setAlignment(Qt.AlignCenter)
            pred_lbl.setFixedSize(HEATMAP_THUMB_W, HEATMAP_THUMB_H)
            pred_lbl.setStyleSheet("background: #222; color: #666; border: 1px solid #444;")
            self._layout.addWidget(pred_lbl)
            self._pred_labels[name] = pred_lbl

        self._layout.addStretch()

        # Cache current data for re-draws on toggle
        self._cur_pred_heatmaps: Optional[np.ndarray] = None
        self._cur_gt_coords: Optional[dict[str, tuple[float, float]]] = None

    def update_heatmaps(
        self,
        pred_heatmaps: Optional[np.ndarray],
        gt_coords: Optional[dict[str, tuple[float, float]]],
    ) -> None:
        """Update heatmap thumbnails. gt_coords are in model-resolution pixels."""
        self._cur_pred_heatmaps = pred_heatmaps
        self._cur_gt_coords = gt_coords
        self._refresh()

    def _refresh(self) -> None:
        """Redraw all heatmap labels from cached data."""
        show_gt = self._show_gt.isChecked()
        for i, name in enumerate(LANDMARK_ORDER):
            lbl = self._pred_labels[name]
            if self._cur_pred_heatmaps is not None:
                pm = heatmap_to_pixmap(self._cur_pred_heatmaps[i])
                # Draw GT cross on top if available and toggled on
                if show_gt and self._cur_gt_coords and name in self._cur_gt_coords:
                    gx, gy = self._cur_gt_coords[name]
                    tx = gx * HEATMAP_THUMB_W / MODEL_W
                    ty = gy * HEATMAP_THUMB_H / MODEL_H
                    self._draw_cross(pm, tx, ty)
                lbl.setPixmap(pm)
                lbl.setText("")
            else:
                lbl.clear()
                lbl.setText("No model")

    @staticmethod
    def _draw_cross(pixmap: QPixmap, x: float, y: float, size: int = 8) -> None:
        """Draw a white cross with dark outline on a pixmap."""
        p = QPainter(pixmap)
        p.setRenderHint(QPainter.Antialiasing)
        ix, iy = int(round(x)), int(round(y))
        # Dark outline
        from PyQt5.QtGui import QPen
        p.setPen(QPen(QColor(0, 0, 0), 3))
        p.drawLine(ix - size, iy - size, ix + size, iy + size)
        p.drawLine(ix - size, iy + size, ix + size, iy - size)
        # White cross
        p.setPen(QPen(QColor(255, 255, 255), 1))
        p.drawLine(ix - size, iy - size, ix + size, iy + size)
        p.drawLine(ix - size, iy + size, ix + size, iy - size)
        p.end()


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
    finished_training = pyqtSignal(str)  # checkpoint path
    error = pyqtSignal(str)

    def run(self) -> None:
        """Execute training fold 0."""
        capture = _StdoutCapture(self.progress)
        old_stdout = sys.stdout
        sys.stdout = capture
        try:
            from training.train import create_cv_splits, get_device, train_fold

            config_path = _project_root / "configs" / "default.yaml"
            with open(config_path) as f:
                cfg = yaml.safe_load(f)

            output_dir = _project_root / "output"
            output_dir.mkdir(parents=True, exist_ok=True)

            device = get_device()
            print(f"Using device: {device}")

            annotation_dir = _project_root / cfg["data"]["annotation_dir"]
            splits = create_cv_splits(annotation_dir, cfg["cv"]["n_folds"])
            train_idx, val_idx = splits[0]
            print(f"Fold 0: {len(train_idx)} train, {len(val_idx)} val")

            train_fold(cfg, 0, train_idx, val_idx, output_dir, device)

            ckpt = output_dir / "checkpoints" / "best_fold0.pt"
            self.finished_training.emit(str(ckpt))
        except Exception as e:
            self.error.emit(str(e))
        finally:
            sys.stdout = old_stdout


class TrainingDialog(QDialog):
    """Modal dialog showing live training log output."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Training — Fold 0")
        self.resize(620, 450)
        layout = QVBoxLayout(self)

        self._log = QTextEdit()
        self._log.setReadOnly(True)
        self._log.setStyleSheet("background: #1e1e1e; color: #ccc; font-family: monospace;")
        layout.addWidget(self._log)

        self._close_btn = QPushButton("Close")
        self._close_btn.setEnabled(False)
        self._close_btn.clicked.connect(self.accept)
        layout.addWidget(self._close_btn)

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

        self._info_table = QTableWidget(len(LANDMARK_ORDER), 6)
        self._info_table.setHorizontalHeaderLabels(
            ["Landmark", "Pred X", "Pred Y", "GT X", "GT Y", "Error (px)"]
        )
        self._info_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self._info_table.setEditTriggers(QTableWidget.NoEditTriggers)
        self._info_table.setMaximumHeight(180)
        # Pre-fill landmark names
        for i, name in enumerate(LANDMARK_ORDER):
            item = QTableWidgetItem(LANDMARK_DISPLAY.get(name, name))
            color = LANDMARK_QCOLORS.get(name, QColor(200, 200, 200))
            item.setForeground(color)
            self._info_table.setItem(i, 0, item)
        center_layout.addWidget(self._info_table, stretch=0)
        splitter.addWidget(center)

        # -- Right panel: heatmaps --
        self._heatmap_panel = HeatmapPanel()
        splitter.addWidget(self._heatmap_panel)

        splitter.setSizes([220, 800, 300])

    # ---- Auto-load ----
    def _auto_load(self) -> None:
        """Auto-load training_data_pics + training_data if they exist."""
        img_dir = _project_root / "training_data_pics"
        gt_dir = _project_root / "training_data"
        if img_dir.is_dir():
            self._load_image_folder(img_dir, gt_dir if gt_dir.is_dir() else None)
            self.statusBar().showMessage(
                f"Auto-loaded {len(self._entries)} images from training_data_pics/"
            )

    # ---- Actions ----
    def _on_load_model(self) -> None:
        """Load a model checkpoint via file dialog."""
        path, _ = QFileDialog.getOpenFileName(
            self, "Select Checkpoint", str(_project_root / "checkpoints"), "PyTorch (*.pt *.pth)"
        )
        if not path:
            return
        try:
            from inference.predict import LandmarkPredictor

            self._predictor = LandmarkPredictor(Path(path))
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
        # Look for GT annotations: sibling training_data/ or same folder
        gt_dir = None
        candidate = folder.parent / "training_data"
        if candidate.is_dir():
            gt_dir = candidate
        else:
            # Check if geojson files exist alongside images
            if list(folder.glob("*.geojson")):
                gt_dir = folder
        self._load_image_folder(folder, gt_dir)
        self.statusBar().showMessage(f"Loaded {len(self._entries)} images from {folder.name}/")

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

            vis = draw_landmarks_on_image(image, preds, entry.gt)
            out_path = self._output_dir / f"{entry.path.stem}_landmarks.jpg"
            cv2.imwrite(str(out_path), vis)
            saved += 1

        progress.setValue(len(self._entries))
        self.statusBar().showMessage(f"Saved {saved} images to {self._output_dir}")

    def _on_train_model(self) -> None:
        """Launch training fold 0 in a background thread with live log dialog."""
        # Validate that GT annotations exist
        gt_count = sum(1 for e in self._entries if e.geojson_path and e.geojson_path.exists())
        if gt_count == 0:
            self.statusBar().showMessage("No ground-truth annotations found — cannot train")
            return

        self._train_dialog = TrainingDialog(self)
        self._train_thread = TrainingThread()
        self._train_thread.progress.connect(self._train_dialog.append_log)
        self._train_thread.finished_training.connect(self._on_training_finished)
        self._train_thread.error.connect(self._on_training_error)
        self._train_thread.start()
        self._train_dialog.append_log(f"Starting training with {gt_count} annotated images...")
        self._train_dialog.exec_()

    def _on_training_finished(self, ckpt_path: str) -> None:
        """Handle successful training completion."""
        self._train_dialog.append_log(f"\nTraining complete! Checkpoint: {ckpt_path}")
        self._train_dialog.enable_close()
        # Auto-load the trained model
        try:
            from inference.predict import LandmarkPredictor

            self._predictor = LandmarkPredictor(Path(ckpt_path))
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

    # ---- Image loading ----
    def _load_image_folder(self, img_dir: Path, gt_dir: Optional[Path]) -> None:
        """Scan folder for images and match GT annotations."""
        self._entries.clear()
        self._image_list.clear()
        self._current_idx = -1

        image_files = sorted(
            f for f in img_dir.iterdir()
            if f.is_file() and f.suffix.lower() in IMAGE_EXTENSIONS
        )

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
        if entry.gt and entry.gt_heatmaps is None:
            entry.gt_heatmaps = generate_gt_heatmaps(entry.gt, orig_w=orig_w, orig_h=orig_h)

        # Run prediction if model available and not cached
        if self._predictor and entry.prediction is None:
            try:
                entry.prediction = self._predictor.predict(image)
            except Exception as e:
                self.statusBar().showMessage(f"Prediction failed: {e}")

        # Draw overlay
        preds = entry.prediction["landmarks"] if entry.prediction else {}
        vis = draw_landmarks_on_image(image, preds, entry.gt)
        self._image_widget.set_image(vis)
        self._image_widget.fit_in_view()

        # Update info table
        self._update_info_table(entry)

        # Update heatmap panel — pass GT coords scaled to model resolution
        pred_hm = entry.prediction["heatmaps"] if entry.prediction else None
        gt_model_coords = None
        if entry.gt:
            gt_model_coords = {
                name: (x * MODEL_W / orig_w, y * MODEL_H / orig_h)
                for name, (x, y) in entry.gt.items()
            }
        self._heatmap_panel.update_heatmaps(pred_hm, gt_model_coords)

        gt_count = len(entry.gt) if entry.gt else 0
        pred_str = "yes" if entry.prediction else "no"
        self.statusBar().showMessage(
            f"{entry.path.name} — {orig_w}x{orig_h} — GT: {gt_count}/5 — Prediction: {pred_str}"
        )

    # ---- Info table ----
    def _update_info_table(self, entry: ImageEntry) -> None:
        """Populate the per-landmark info table."""
        for i, name in enumerate(LANDMARK_ORDER):
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
            if (
                entry.prediction
                and name in entry.prediction["landmarks"]
                and entry.gt
                and name in entry.gt
            ):
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
