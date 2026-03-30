"""
Segmentation inference GUI.

PyQt5 application that wraps the modeltojson library for interactive use.
"""

import os
import sys
import traceback
from pathlib import Path

import geojson
import numpy as np
import torch
from modeltojson import (
    SUPPORTED_EXTENSIONS,
    load_model,
    mask_to_geojson,
    read_image,
    run_inference,
)
from PyQt5.QtCore import QRectF, Qt, QThread, pyqtSignal
from PyQt5.QtGui import QColor, QImage, QPainter, QPixmap
from PyQt5.QtWidgets import (
    QApplication,
    QFileDialog,
    QGraphicsPixmapItem,
    QGraphicsScene,
    QGraphicsView,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QMainWindow,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QSlider,
    QSplitter,
    QStatusBar,
    QVBoxLayout,
    QWidget,
)


# ---------------------------------------------------------------------------
# Overlay rendering
# ---------------------------------------------------------------------------
def build_overlay(mask: np.ndarray, classes: list, alpha: int = 128) -> QImage:
    """Render the segmentation mask as a semi-transparent RGBA QImage."""
    h, w = mask.shape
    overlay = np.zeros((h, w, 4), dtype=np.uint8)

    for cls_info in classes:
        idx = cls_info["index"]
        if cls_info["name"].endswith("*"):
            continue
        color_hex = cls_info["color"].lstrip("#")
        r, g, b = int(color_hex[0:2], 16), int(color_hex[2:4], 16), int(color_hex[4:6], 16)
        where = mask == idx
        overlay[where, 0] = r
        overlay[where, 1] = g
        overlay[where, 2] = b
        overlay[where, 3] = alpha

    return QImage(overlay.data.tobytes(), w, h, w * 4, QImage.Format_RGBA8888).copy()


def numpy_to_qimage(img: np.ndarray) -> QImage:
    """Convert RGB numpy array to QImage."""
    h, w, c = img.shape
    bytes_per_line = w * c
    return QImage(img.data.tobytes(), w, h, bytes_per_line, QImage.Format_RGB888).copy()


# ---------------------------------------------------------------------------
# Worker thread
# ---------------------------------------------------------------------------
class InferenceWorker(QThread):
    progress = pyqtSignal(int, int)  # current, total
    image_done = pyqtSignal(str, np.ndarray, np.ndarray)  # path, image, mask
    finished_all = pyqtSignal()
    error = pyqtSignal(str)

    def __init__(self, model, metadata, image_paths, device):
        super().__init__()
        self.model = model
        self.metadata = metadata
        self.image_paths = image_paths
        self.device = device
        self._cancel = False

    def cancel(self):
        self._cancel = True

    def run(self):
        try:
            for path in self.image_paths:
                if self._cancel:
                    break
                img = read_image(path)
                mask = run_inference(
                    self.model,
                    img,
                    self.metadata,
                    self.device,
                    progress_callback=lambda cur, tot: self.progress.emit(cur, tot),
                )
                self.image_done.emit(path, img, mask)
        except Exception as e:
            self.error.emit(f"{e}\n{traceback.format_exc()}")
        self.finished_all.emit()


# ---------------------------------------------------------------------------
# Zoomable image viewer
# ---------------------------------------------------------------------------
class ImageViewer(QGraphicsView):
    def __init__(self):
        super().__init__()
        self._scene = QGraphicsScene(self)
        self.setScene(self._scene)
        self._base_item = None
        self._overlay_item = None
        self.setDragMode(QGraphicsView.ScrollHandDrag)
        self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
        self.setRenderHints(QPainter.Antialiasing | QPainter.SmoothPixmapTransform)

    def set_images(self, base_pixmap: QPixmap, overlay_pixmap: QPixmap):
        self._scene.clear()
        self._base_item = self._scene.addPixmap(base_pixmap)
        self._overlay_item = self._scene.addPixmap(overlay_pixmap)
        self._scene.setSceneRect(QRectF(base_pixmap.rect()))
        self.fitInView(self._scene.sceneRect(), Qt.KeepAspectRatio)

    def update_overlay(self, overlay_pixmap: QPixmap):
        if self._overlay_item:
            self._overlay_item.setPixmap(overlay_pixmap)

    def wheelEvent(self, event):
        factor = 1.15 if event.angleDelta().y() > 0 else 1 / 1.15
        self.scale(factor, factor)

    def resizeEvent(self, event):
        super().resizeEvent(event)


# ---------------------------------------------------------------------------
# Main window
# ---------------------------------------------------------------------------
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Segmentation Inference GUI")
        self.resize(1200, 800)

        self.model = None
        self.metadata = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.results = {}  # path -> (image, mask)
        self.worker = None

        self._build_ui()

    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QHBoxLayout(central)

        # --- Left panel ---
        left = QWidget()
        left_layout = QVBoxLayout(left)
        left_layout.setContentsMargins(4, 4, 4, 4)

        # Model selection
        model_group = QGroupBox("Model")
        mg_layout = QVBoxLayout(model_group)
        self.model_label = QLabel("No model loaded")
        self.model_label.setWordWrap(True)
        btn_model = QPushButton("Select Model Folder...")
        btn_model.clicked.connect(self._select_model)
        mg_layout.addWidget(self.model_label)
        mg_layout.addWidget(btn_model)
        left_layout.addWidget(model_group)

        # Image selection
        img_group = QGroupBox("Images")
        ig_layout = QVBoxLayout(img_group)
        btn_images = QPushButton("Select Image Folder...")
        btn_images.clicked.connect(self._select_images)
        self.image_list = QListWidget()
        self.image_list.currentRowChanged.connect(self._on_image_selected)
        ig_layout.addWidget(btn_images)
        ig_layout.addWidget(self.image_list)
        left_layout.addWidget(img_group)

        # Run / Export
        run_group = QGroupBox("Actions")
        rg_layout = QVBoxLayout(run_group)
        self.btn_run = QPushButton("Run Inference")
        self.btn_run.clicked.connect(self._run_inference)
        self.btn_run.setEnabled(False)
        self.btn_export = QPushButton("Export All as GeoJSON...")
        self.btn_export.clicked.connect(self._export_geojson)
        self.btn_export.setEnabled(False)
        self.btn_export_current = QPushButton("Export Current as GeoJSON...")
        self.btn_export_current.clicked.connect(self._export_current_geojson)
        self.btn_export_current.setEnabled(False)
        self.progress = QProgressBar()
        rg_layout.addWidget(self.btn_run)
        rg_layout.addWidget(self.btn_export)
        rg_layout.addWidget(self.btn_export_current)
        rg_layout.addWidget(self.progress)
        left_layout.addWidget(run_group)

        left.setMaximumWidth(320)

        # --- Right panel ---
        right = QWidget()
        right_layout = QVBoxLayout(right)
        right_layout.setContentsMargins(4, 4, 4, 4)

        self.viewer = ImageViewer()
        right_layout.addWidget(self.viewer, stretch=1)

        # Overlay controls
        overlay_bar = QHBoxLayout()
        overlay_bar.addWidget(QLabel("Overlay opacity:"))
        self.opacity_slider = QSlider(Qt.Horizontal)
        self.opacity_slider.setRange(0, 255)
        self.opacity_slider.setValue(128)
        self.opacity_slider.valueChanged.connect(self._update_overlay)
        overlay_bar.addWidget(self.opacity_slider)
        self.opacity_label = QLabel("50%")
        overlay_bar.addWidget(self.opacity_label)
        right_layout.addLayout(overlay_bar)

        # Legend
        self.legend_layout = QHBoxLayout()
        self.legend_layout.addWidget(QLabel("Classes:"))
        self.legend_layout.addStretch()
        right_layout.addLayout(self.legend_layout)

        splitter = QSplitter(Qt.Horizontal)
        splitter.addWidget(left)
        splitter.addWidget(right)
        splitter.setStretchFactor(1, 1)
        main_layout.addWidget(splitter)

        self.statusBar().showMessage("Ready — select a model folder to begin.")

    # --- Model loading ---
    def _select_model(self):
        folder = QFileDialog.getExistingDirectory(
            self, "Select Model Folder", "", QFileDialog.ShowDirsOnly | QFileDialog.DontUseNativeDialog
        )
        if not folder:
            return
        meta_path = os.path.join(folder, "metadata.json")
        if not os.path.exists(meta_path):
            QMessageBox.warning(self, "Error", "No metadata.json found in selected folder.")
            return
        try:
            self.statusBar().showMessage("Loading model...")
            QApplication.processEvents()
            self.model, self.metadata = load_model(folder, self.device)
            name = self.metadata.get("name", os.path.basename(folder))
            arch = self.metadata["architecture"]
            arch_desc = arch.get("type", "unknown")
            if "backbone" in arch:
                arch_desc += f" / {arch['backbone']}"
            self.model_label.setText(f"{name}\n{arch_desc} ({self.device})")
            self._update_legend()
            self._check_ready()
            self.statusBar().showMessage(f"Model loaded: {name}")
        except Exception as e:
            QMessageBox.critical(self, "Error loading model", str(e))
            self.statusBar().showMessage("Model loading failed.")

    def _update_legend(self):
        while self.legend_layout.count() > 2:
            item = self.legend_layout.takeAt(self.legend_layout.count() - 1)
            if item.widget():
                item.widget().deleteLater()

        if self.metadata:
            for cls in self.metadata["classes"]:
                if cls["name"].endswith("*"):
                    continue
                lbl = QLabel(f"  {cls['name']}  ")
                lbl.setStyleSheet(
                    f"background-color: {cls['color']}; color: white; "
                    f"padding: 2px 6px; border-radius: 3px; font-weight: bold;"
                )
                self.legend_layout.insertWidget(self.legend_layout.count() - 1, lbl)

    # --- Image selection ---
    def _select_images(self):
        folder = QFileDialog.getExistingDirectory(
            self, "Select Image Folder", "", QFileDialog.ShowDirsOnly | QFileDialog.DontUseNativeDialog
        )
        if not folder:
            return
        self.image_list.clear()
        self.results.clear()
        self._image_paths = []
        for f in sorted(os.listdir(folder)):
            if f.startswith("._") or f.startswith("."):
                continue
            ext = os.path.splitext(f)[1].lower()
            if ext in SUPPORTED_EXTENSIONS:
                full = os.path.join(folder, f)
                self._image_paths.append(full)
                self.image_list.addItem(f)
        self._check_ready()
        self.statusBar().showMessage(f"Found {len(self._image_paths)} images.")

    def _check_ready(self):
        ready = self.model is not None and hasattr(self, "_image_paths") and len(self._image_paths) > 0
        self.btn_run.setEnabled(ready)

    # --- Inference ---
    def _run_inference(self):
        self.btn_run.setEnabled(False)
        self.btn_export.setEnabled(False)
        self.btn_export_current.setEnabled(False)
        self.progress.setValue(0)
        self.results.clear()

        self.worker = InferenceWorker(self.model, self.metadata, self._image_paths, self.device)
        self.worker.progress.connect(self._on_tile_progress)
        self.worker.image_done.connect(self._on_image_done)
        self.worker.finished_all.connect(self._on_all_done)
        self.worker.error.connect(self._on_error)
        self.worker.start()

    def _on_tile_progress(self, current, total):
        self.progress.setMaximum(total)
        self.progress.setValue(current)

    def _on_image_done(self, path, image, mask):
        self.results[path] = (image, mask)
        idx = self._image_paths.index(path)
        self.image_list.setCurrentRow(idx)
        self.statusBar().showMessage(f"Done: {os.path.basename(path)}")

    def _on_all_done(self):
        self.btn_run.setEnabled(True)
        self.btn_export.setEnabled(len(self.results) > 0)
        self.btn_export_current.setEnabled(len(self.results) > 0)
        self.statusBar().showMessage(f"Inference complete — {len(self.results)} images processed.")

    def _on_error(self, msg):
        QMessageBox.critical(self, "Inference Error", msg)
        self.btn_run.setEnabled(True)

    # --- Display ---
    def _on_image_selected(self, row):
        if row < 0 or row >= len(self._image_paths):
            return
        path = self._image_paths[row]
        if path not in self.results:
            try:
                img = read_image(path)
                qimg = numpy_to_qimage(img)
                pix = QPixmap.fromImage(qimg)
                empty = QPixmap(pix.size())
                empty.fill(QColor(0, 0, 0, 0))
                self.viewer.set_images(pix, empty)
                self.btn_export_current.setEnabled(False)
            except Exception as e:
                self.statusBar().showMessage(f"Error reading image: {e}")
            return

        self.btn_export_current.setEnabled(True)
        image, mask = self.results[path]
        self._current_image = image
        self._current_mask = mask
        self._current_path = path

        qimg = numpy_to_qimage(image)
        base_pix = QPixmap.fromImage(qimg)

        alpha = self.opacity_slider.value()
        overlay_qimg = build_overlay(mask, self.metadata["classes"], alpha)
        overlay_pix = QPixmap.fromImage(overlay_qimg)

        self.viewer.set_images(base_pix, overlay_pix)

    def _update_overlay(self, value):
        pct = int(value / 255 * 100)
        self.opacity_label.setText(f"{pct}%")

        if not hasattr(self, "_current_mask") or self._current_mask is None:
            return
        if self.metadata is None:
            return

        overlay_qimg = build_overlay(self._current_mask, self.metadata["classes"], value)
        overlay_pix = QPixmap.fromImage(overlay_qimg)
        self.viewer.update_overlay(overlay_pix)

    # --- GeoJSON export ---
    def _export_geojson(self):
        folder = QFileDialog.getExistingDirectory(
            self,
            "Select Output Folder for GeoJSON files",
            "",
            QFileDialog.ShowDirsOnly | QFileDialog.DontUseNativeDialog,
        )
        if not folder:
            return
        count = 0
        for path, (image, mask) in self.results.items():
            fc = mask_to_geojson(mask, self.metadata["classes"], path)
            stem = Path(path).stem
            out_path = os.path.join(folder, f"{stem}_detections.geojson")
            with open(out_path, "w") as f:
                geojson.dump(fc, f, indent=2)
            count += 1
        self.statusBar().showMessage(f"Exported {count} GeoJSON files to {folder}")

    def _export_current_geojson(self):
        if not hasattr(self, "_current_path"):
            return
        stem = Path(self._current_path).stem
        path, _ = QFileDialog.getSaveFileName(
            self, "Save GeoJSON", f"{stem}_detections.geojson", "GeoJSON (*.geojson);;All Files (*)"
        )
        if not path:
            return
        image, mask = self.results[self._current_path]
        fc = mask_to_geojson(mask, self.metadata["classes"], self._current_path)
        with open(path, "w") as f:
            geojson.dump(fc, f, indent=2)
        self.statusBar().showMessage(f"Exported: {path}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main():
    app = QApplication(sys.argv)
    app.setStyle("Fusion")

    from PyQt5.QtCore import Qt
    from PyQt5.QtGui import QColor, QPalette

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

    window = MainWindow()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
