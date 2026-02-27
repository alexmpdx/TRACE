"""Zoomable/pannable QGraphicsView for displaying BGR numpy arrays."""

from __future__ import annotations

import numpy as np
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QImage, QPainter, QPixmap
from PyQt5.QtWidgets import QGraphicsPixmapItem, QGraphicsScene, QGraphicsView


class ImageWidget(QGraphicsView):
    """A QGraphicsView that displays a BGR numpy array with zoom and pan."""

    ZOOM_FACTOR = 1.15

    def __init__(self, parent=None):
        super().__init__(parent)
        self._scene = QGraphicsScene(self)
        self.setScene(self._scene)
        self._pixmap_item: QGraphicsPixmapItem | None = None

        # Smooth scrolling and rendering
        self.setRenderHint(QPainter.SmoothPixmapTransform, True)
        self.setRenderHint(QPainter.Antialiasing, True)
        self.setDragMode(QGraphicsView.ScrollHandDrag)
        self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.AnchorUnderMouse)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)

        # Placeholder text
        self._placeholder = self._scene.addText("No image loaded")
        self._placeholder.setDefaultTextColor(Qt.gray)

    def set_image(self, bgr_array: np.ndarray) -> None:
        """Display a BGR numpy array (H, W, 3) as a pixmap."""
        if bgr_array is None:
            return

        h, w = bgr_array.shape[:2]
        channels = bgr_array.shape[2] if bgr_array.ndim == 3 else 1

        if channels == 3:
            # BGR → RGB
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
        if event.angleDelta().y() > 0:
            factor = self.ZOOM_FACTOR
        else:
            factor = 1.0 / self.ZOOM_FACTOR
        self.scale(factor, factor)

    def clear_image(self) -> None:
        """Remove the current image and show placeholder."""
        self._scene.clear()
        self._pixmap_item = None
        self._placeholder = self._scene.addText("No image loaded")
        self._placeholder.setDefaultTextColor(Qt.gray)
