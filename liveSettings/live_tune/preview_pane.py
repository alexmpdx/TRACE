"""Embeddable live vein-overlay preview pane.

Hosted in the TRACE Advanced Settings dialog (right column). The host connects
every parameter widget to :meth:`on_config_changed` and the tab widget's
``currentChanged`` to :meth:`set_active_tier`; the pane debounces by tier and
dispatches recompute to a background :class:`LiveTuneWorker`.

Tier policy (debounce = idle wait after the last edit before the job starts;
recompute cost is measured on a 5440x3648 wing — specimen 0003):
    A (Wing Graph)  500 ms debounce   skeleton rebuild ~4 s
    B (Tracing)     200 ms debounce   anchor+trace+tissue ~5 s (trace dominates)
    C (Intervein)   manual button     split+name ~2 s
    D (Appearance)  30 ms debounce     render only ~30 ms (effectively instant)

So appearance/colour/opacity edits are truly live; tracing/skeleton edits are
"one slow recompute" rather than per-keystroke stutter — still far better than a
full batch run, and the cached skeleton means a Tier-B edit pays ~5 s, not ~9 s.
Reduced-resolution preview (downscale the input + track the scale factor) is the
lever that makes Tier A/B feel live on large wings: the "Preview res" combo runs
the stages at full / half / quarter resolution (default half). Compute is
~scale**2, so half-res ≈ 4× faster and quarter-res ≈ 16× faster. The real batch
run always uses full resolution — this only affects the preview (see ../STATUS.md).
"""

from __future__ import annotations

import logging
import tempfile
from pathlib import Path
from typing import Callable, Optional

import numpy as np
from PyQt5.QtCore import Qt, QTimer, pyqtSignal
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFileDialog,
    QFrame,
    QGraphicsPixmapItem,
    QGraphicsScene,
    QGraphicsView,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

from .session import (
    FIELD_TIER,
    TIER_A,
    TIER_B,
    TIER_C,
    TIER_D,
    VIEW_FINAL,
    VIEW_SKELETON,
    VIEW_TRACED,
    Appearance,
    LiveTuneSession,
)
from .worker import LiveTuneWorker

logger = logging.getLogger(__name__)

_DEBOUNCE_MS = {TIER_A: 500, TIER_B: 200, TIER_D: 30}
# Preprocessing re-run debounce: longer, since it triggers a full DL pass that
# can't be reduced-res'd. The user still gets one re-run after they settle.
_PREPROC_DEBOUNCE_MS = 700


def _bgr_to_qpixmap(img: np.ndarray) -> QPixmap:
    """Convert an OpenCV BGR (or grayscale) ndarray to a QPixmap."""
    if img is None:
        return QPixmap()
    arr = np.ascontiguousarray(img)
    if arr.ndim == 2:
        h, w = arr.shape
        qimg = QImage(arr.data, w, h, w, QImage.Format_Grayscale8)
    else:
        h, w, _ = arr.shape
        rgb = np.ascontiguousarray(arr[:, :, ::-1])  # BGR -> RGB
        qimg = QImage(rgb.data, w, h, 3 * w, QImage.Format_RGB888)
    return QPixmap.fromImage(qimg.copy())


class _ZoomView(QGraphicsView):
    """Minimal wheel-zoom / drag-pan image view."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._scene = QGraphicsScene(self)
        self.setScene(self._scene)
        self._item = QGraphicsPixmapItem()
        self._scene.addItem(self._item)
        self.setDragMode(QGraphicsView.ScrollHandDrag)
        self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
        self.setRenderHints(self.renderHints())
        self._has_pixmap = False

    def set_pixmap(self, pm: QPixmap, fit: bool = False) -> None:
        first = not self._has_pixmap
        self._item.setPixmap(pm)
        self._scene.setSceneRect(self._item.boundingRect())
        self._has_pixmap = True
        if first or fit:
            self.fit()

    def fit(self) -> None:
        if self._has_pixmap:
            self.fitInView(self._item, Qt.KeepAspectRatio)

    def wheelEvent(self, event):  # noqa: N802 (Qt signature)
        factor = 1.25 if event.angleDelta().y() > 0 else 0.8
        self.scale(factor, factor)


class LivePreviewPane(QWidget):
    """Live overlay preview + sample picker, embeddable in a settings dialog."""

    save_preset_requested = pyqtSignal(object)  # PipelineConfig

    def __init__(
        self,
        get_config: Callable[[], object],
        model_paths: Optional[dict] = None,
        default_image_dir: str = "",
        preproc_getter: Optional[Callable[[], dict]] = None,
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self._get_config = get_config
        self._model_paths = model_paths or {}
        self._default_image_dir = default_image_dir
        # Returns the current preprocessing options (wing isolation, rotation,
        # expand, rescale target) so a re-run reflects them. None → defaults.
        self._preproc_getter = preproc_getter or (lambda: {})
        self._appearance = Appearance()
        self._pending_tier: Optional[str] = None
        self._loaded = False
        self._tmp_dir: Optional[str] = None
        # Debounce timer for preprocessing re-runs (slow DL path; longer wait).
        self._preproc_debounce: Optional[QTimer] = None
        # Preview resolution factor (1.0 = full). Default to half-res so the
        # preview feels live on large wings out of the box; the user can pick
        # Full for a final check. The real batch run is always full resolution.
        self._preview_scale: float = 0.5
        # Active view mode (which pipeline product to show). Default Final
        # preserves the original behavior.
        self._view: str = VIEW_FINAL
        # Long-lived model caches so DL models load only once across samples.
        self._predictor_cache: dict = {}
        self._model_cache: dict = {}

        self._session = LiveTuneSession()
        self._worker = LiveTuneWorker(self._session, self)
        self._worker.result_ready.connect(self._on_result)
        self._worker.job_started.connect(self._on_job_started)
        self._worker.load_done.connect(self._on_load_done)
        self._worker.start()

        self._debounce = QTimer(self)
        self._debounce.setSingleShot(True)
        self._debounce.timeout.connect(self._fire_update)

        # Preprocessing re-runs are expensive (DL passes, seconds) and can't be
        # reduced-res'd, so they get a longer debounce than the tier cascade.
        self._preproc_debounce = QTimer(self)
        self._preproc_debounce.setSingleShot(True)
        self._preproc_debounce.timeout.connect(self._fire_preproc)

        self._build_ui()

    # -- UI construction --------------------------------------------------
    def _build_ui(self) -> None:
        root = QVBoxLayout(self)

        # Input source group — raw image only. Preprocessing (wing isolation,
        # rotation, rescale) runs on the image, so a pre-made GeoJSON can't
        # reflect preprocessing-setting changes; the image path is the source
        # of truth and preprocessing re-runs when its settings change.
        src = QGroupBox("Sample")
        sv = QVBoxLayout(src)
        sv.addWidget(QLabel("Pick a raw wing image. Preprocessing runs on Load and "
                            "re-runs when preprocessing settings change."))
        self.ed_image = self._file_row(sv, "Image:", self._pick_image)

        load_row = QHBoxLayout()
        self.btn_load = QPushButton("Load sample")
        self.btn_load.clicked.connect(self._on_load_clicked)
        load_row.addWidget(self.btn_load)
        load_row.addStretch(1)
        load_row.addWidget(QLabel("Preview res:"))
        self.cmb_res = QComboBox()
        for _label, _val in (("Full (slow)", 1.0), ("Half (≈4× faster)", 0.5), ("Quarter (≈16× faster)", 0.25)):
            self.cmb_res.addItem(_label, _val)
        self.cmb_res.setCurrentIndex(1)  # Half — matches self._preview_scale default
        self.cmb_res.setToolTip(
            "Resolution the live preview computes at. Lower = faster, but the skeleton "
            "and vein tracing run on a coarser raster, so the predicted veins/regions may "
            "differ from the full-resolution result.\n"
            "Use Full to judge a setting's true effect; the real batch run always uses "
            "full resolution regardless of this choice."
        )
        self.cmb_res.currentIndexChanged.connect(self._on_res_changed)
        load_row.addWidget(self.cmb_res)
        sv.addLayout(load_row)

        # Persistent accuracy warning, shown whenever the preview is below full
        # res. A tooltip alone is too easy to miss for something that changes the
        # predicted result; this keeps it in view while a reduced res is active.
        self.res_warning = QLabel()
        self.res_warning.setWordWrap(True)
        self.res_warning.setStyleSheet("color: #b8860b;")  # dark goldenrod — reads on light+dark
        sv.addWidget(self.res_warning)
        self._update_res_warning()

        root.addWidget(src)

        # View selector — which pipeline product to show. The skeleton view
        # needs only Tier A, so it skips the ~1s trace entirely; great for
        # tuning Wing Graph settings fast or pinpointing where the pipeline
        # goes wrong.
        view_row = QHBoxLayout()
        view_row.addWidget(QLabel("View:"))
        self.cmb_view = QComboBox()
        for _label, _val, _tip in (
            ("Wing graph (skeleton)", VIEW_SKELETON,
             "End of skeletonization: graph edges + nodes. No tracing — fastest; "
             "Tracing/Intervein settings don't affect this view."),
            ("Traced veins + landmarks", VIEW_TRACED,
             "End of vein tracing: labeled vein centerlines + snapped landmarks. "
             "No intervein regions."),
            ("Final output", VIEW_FINAL,
             "The full overlay: veins + intervein regions (regions via Refresh)."),
        ):
            self.cmb_view.addItem(_label, _val)
            self.cmb_view.setItemData(self.cmb_view.count() - 1, _tip, Qt.ToolTipRole)
        self.cmb_view.setCurrentIndex(2)  # Final — matches prior default behavior
        self.cmb_view.currentIndexChanged.connect(self._on_view_changed)
        view_row.addWidget(self.cmb_view, 1)
        root.addLayout(view_row)

        # Preview + static color key. The key is drawn here (UI-side) rather
        # than baked into the overlay image, so it never occludes the wing and
        # doesn't waste the limited preview canvas.
        preview_row = QHBoxLayout()
        self.view = _ZoomView(self)
        self.view.setMinimumSize(420, 360)
        preview_row.addWidget(self.view, 1)
        self.legend = self._build_legend()
        preview_row.addWidget(self.legend, 0)
        root.addLayout(preview_row, 1)

        # Status
        self.status = QLabel("Pick a sample and click Load to start the live preview.")
        self.status.setWordWrap(True)
        root.addWidget(self.status)
        self.progress = QProgressBar()
        self.progress.setRange(0, 0)  # indeterminate
        self.progress.hide()
        root.addWidget(self.progress)

        # Appearance + actions
        appearance = QGroupBox("Display")
        av = QHBoxLayout(appearance)
        self.cb_veins = QCheckBox("Veins")
        self.cb_veins.setChecked(True)
        self.cb_tissue = QCheckBox("Vein tissue")
        self.cb_regions = QCheckBox("Intervein regions")
        for cb in (self.cb_veins, self.cb_tissue, self.cb_regions):
            cb.toggled.connect(self._on_appearance_changed)
            av.addWidget(cb)
        av.addStretch(1)
        self.btn_fit = QPushButton("Fit")
        self.btn_fit.clicked.connect(self.view.fit)
        av.addWidget(self.btn_fit)
        root.addWidget(appearance)

        actions = QHBoxLayout()
        self.btn_intervein = QPushButton("Refresh intervein (slow)")
        self.btn_intervein.setToolTip("Run the intervein split + naming step from the current veins")
        self.btn_intervein.clicked.connect(self._on_intervein_clicked)
        self.btn_preset = QPushButton("Save as preset…")
        self.btn_preset.clicked.connect(lambda: self.save_preset_requested.emit(self._get_config()))
        actions.addWidget(self.btn_intervein)
        actions.addStretch(1)
        actions.addWidget(self.btn_preset)
        root.addLayout(actions)

        self._set_params_enabled(False)

    def _build_legend(self) -> QWidget:
        """Static color key whose contents switch with the active view.

        The vein-color key is meaningful for the traced + final views, but the
        skeleton view draws graph primitives (edges + degree-colored nodes), so
        it gets its own key. Both are built once into a QStackedWidget; the view
        selector flips between them via :meth:`_sync_legend`.
        """
        self._legend_stack = QStackedWidget()
        self._legend_stack.setSizePolicy(
            self._legend_stack.sizePolicy().Fixed, self._legend_stack.sizePolicy().Preferred
        )
        self._legend_vein = self._make_legend_box(self._vein_legend_entries())
        self._legend_skeleton = self._make_legend_box(self._skeleton_legend_entries())
        self._legend_stack.addWidget(self._legend_vein)      # index 0
        self._legend_stack.addWidget(self._legend_skeleton)  # index 1
        return self._legend_stack

    def _vein_legend_entries(self) -> list:
        """(label, rgb) rows for the vein-color key (traced / final views)."""
        from identify_features.models.topology import VEIN_AP_ORDER, VEIN_COLORS

        overrides = getattr(self._get_config(), "vein_colors", None) or {}
        entries = []
        for vid in list(VEIN_AP_ORDER) + ["EV"]:
            rgb = overrides.get(vid) or VEIN_COLORS.get(vid)
            if rgb is None:
                continue
            entries.append(("ectopic (EV)" if vid == "EV" else vid, rgb))
        return entries

    @staticmethod
    def _skeleton_legend_entries() -> list:
        """(label, rgb) rows matching render_skeleton's drawn colors.

        render_skeleton uses cv2 BGR tuples; the RGB equivalents are: edges
        (255,255,0), path nodes deg<=2 (255,128,0), junctions deg>=3
        (255,80,255). Kept in sync with preview_render.render_skeleton.
        """
        return [
            ("vein edge", [255, 255, 0]),
            ("node (path, deg ≤ 2)", [255, 128, 0]),
            ("junction (deg ≥ 3)", [255, 80, 255]),
        ]

    def _make_legend_box(self, entries: list) -> QWidget:
        box = QGroupBox("Key")
        box.setSizePolicy(box.sizePolicy().Fixed, box.sizePolicy().Preferred)
        v = QVBoxLayout(box)
        v.setSpacing(3)
        for label_text, rgb in entries:
            row = QHBoxLayout()
            row.setSpacing(6)
            swatch = QLabel()
            swatch.setFixedSize(16, 16)
            swatch.setStyleSheet(
                f"background-color: rgb({int(rgb[0])},{int(rgb[1])},{int(rgb[2])}); "
                "border: 1px solid #888;"
            )
            row.addWidget(swatch)
            row.addWidget(QLabel(label_text))
            row.addStretch(1)
            v.addLayout(row)
        v.addStretch(1)
        return box

    def _sync_legend(self) -> None:
        """Show the key matching the active view (skeleton vs vein)."""
        self._legend_stack.setCurrentIndex(1 if self._view == VIEW_SKELETON else 0)

    def _file_row(self, layout: QVBoxLayout, label: str, pick: Callable) -> QLineEdit:
        row = QHBoxLayout()
        row.addWidget(QLabel(label))
        ed = QLineEdit()
        row.addWidget(ed, 1)
        btn = QPushButton("…")
        btn.setFixedWidth(32)
        btn.clicked.connect(pick)
        row.addWidget(btn)
        holder = QFrame()
        holder.setLayout(row)
        ed._holder = holder  # type: ignore[attr-defined]
        layout.addWidget(holder)
        return ed

    # Native macOS file dialogs intermittently hang when opened from inside a
    # modal QDialog (the file list goes dead while the path bar / Cancel still
    # work) — the same bug measurementMaker's embedded picker hit. Force Qt's
    # own dialog everywhere in this widget.
    _FILE_DIALOG_OPTS = QFileDialog.DontUseNativeDialog

    def _pick_image(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Choose wing image", self._default_image_dir,
            "Images (*.tif *.tiff *.png *.bmp *.jpg *.jpeg);;All files (*)",
            options=self._FILE_DIALOG_OPTS)
        if path:
            self.ed_image.setText(path)

    # -- loading ----------------------------------------------------------
    def _build_loader(self):
        """Build a loader closure capturing the image + current preproc settings.

        Returns None (after warning the user) if the image or required models
        are missing. The returned callable runs preprocessing off the GUI thread
        and yields an InputBundle. Re-running with fresh preproc settings reuses
        the cached models, so only the forward passes re-run.
        """
        from . import input_loader as il

        img = self.ed_image.text().strip()
        if not img:
            QMessageBox.warning(self, "Live preview", "Choose an image first.")
            return None
        lm_ckpt = self._model_paths.get("landmark_checkpoint")
        seg_dir = self._model_paths.get("segmentation_model_dir")
        if not lm_ckpt or not seg_dir:
            QMessageBox.warning(
                self, "Live preview",
                "Landmark and segmentation models must be set in Settings "
                "before the live preview can preprocess an image.")
            return None
        if self._tmp_dir is None:
            self._tmp_dir = tempfile.mkdtemp(prefix="trace_livetune_")

        um = getattr(self._get_config(), "um_per_px", None)
        pp = dict(self._preproc_getter() or {})

        def loader():
            return il.load_from_raw_image(
                image_path=Path(img),
                output_dir=Path(self._tmp_dir),
                landmark_checkpoint=Path(lm_ckpt),
                segmentation_model_dir=Path(seg_dir),
                predictor_cache=self._predictor_cache,
                model_cache=self._model_cache,
                um_per_px=um,
                wing_model_dir=pp.get("wing_model_dir"),
                wing_expand_fraction=pp.get("wing_expand_fraction", 0.05),
                do_rotation=pp.get("do_rotation", False),
                rotation_mirror_correct=pp.get("rotation_mirror_correct", False),
                target_um_per_px=pp.get("target_um_per_px"),
            )

        return loader

    def _on_load_clicked(self) -> None:
        loader = self._build_loader()
        if loader is None:
            return
        self._set_params_enabled(False)
        self.btn_load.setEnabled(False)
        self.progress.show()
        self._worker.request_load(loader, self._preview_scale, self._get_config(), self._appearance)

    # -- preprocessing re-run (host calls on_preproc_changed) -------------
    def on_preproc_changed(self) -> None:
        """Host calls this when a preprocessing-affecting setting changes.

        Debounced because re-running preprocessing means a fresh DL pass.
        """
        if not self._loaded:
            return
        self._preproc_debounce.start(_PREPROC_DEBOUNCE_MS)

    def _fire_preproc(self) -> None:
        if not self._loaded:
            return
        loader = self._build_loader()
        if loader is None:
            return
        self.progress.show()
        self.status.setText("Re-running preprocessing…")
        self._worker.request_load(loader, self._preview_scale, self._get_config(), self._appearance)

    def _on_load_done(self, ok: bool, info: str) -> None:
        self.btn_load.setEnabled(True)
        if not ok:
            self.progress.hide()
            self._loaded = False
            self.status.setText(f"Load failed: {info}")
            QMessageBox.critical(self, "Live preview", f"Could not load sample:\n{info}")
            return
        self._loaded = True
        self.status.setText(f"Loaded {info} — tuning is live.")
        self._set_params_enabled(True)

    # -- view selection ---------------------------------------------------
    def _on_view_changed(self) -> None:
        self._view = self.cmb_view.currentData()
        self._worker.set_view(self._view)
        # Display checkboxes / intervein refresh only matter for the final view.
        self._sync_view_controls()
        # Swap the color key to match what this view draws.
        self._sync_legend()
        if self._loaded:
            # Re-render in the new view. If switching to a tracing view after
            # tuning on skeleton, the session runs the deferred trace now.
            self.progress.show()
            self._worker.request_update(self._get_config(), self._appearance)

    def _sync_view_controls(self) -> None:
        """Enable only the controls meaningful for the active view."""
        final = self._view == VIEW_FINAL
        for w in (self.cb_veins, self.cb_tissue, self.cb_regions, self.btn_intervein):
            w.setEnabled(self._loaded and final)

    # -- config change plumbing (called by the host dialog) --------------
    def on_config_changed(self, field_name: Optional[str] = None) -> None:
        """Host calls this whenever a parameter widget changes."""
        if not self._loaded:
            return
        tier = FIELD_TIER.get(field_name, TIER_A) if field_name else TIER_A
        # View-aware skip: a change the active view can't show needs no work.
        # Skeleton view shows only Tier A, so Tier-B/C/D changes are no-ops for
        # it (the session still defers the Tier-B recompute until a tracing
        # view asks for it). Traced view ignores Tier-C (intervein) changes.
        if self._view == VIEW_SKELETON and tier != TIER_A:
            return
        if tier == TIER_C:
            # Intervein params don't affect veins; flag and wait for Refresh
            # (only the final view shows regions).
            if self._view == VIEW_FINAL:
                self._mark_intervein_stale()
            return
        # Track the lowest (most-invalidating) pending tier; use its debounce.
        self._pending_tier = self._lowest_tier(self._pending_tier, tier)
        self._debounce.start(_DEBOUNCE_MS.get(self._pending_tier, 200))

    @staticmethod
    def _lowest_tier(a: Optional[str], b: str) -> str:
        order = {TIER_A: 0, TIER_B: 1, TIER_D: 2}
        if a is None:
            return b
        return a if order.get(a, 9) <= order.get(b, 9) else b

    def set_active_tier(self, tier: str) -> None:
        """Optional: host can report the active tab's tier (unused for correctness)."""
        # Invalidation is driven by FIELD_TIER; the tab tier is advisory only.
        pass

    def _fire_update(self) -> None:
        if not self._loaded:
            return
        self._pending_tier = None
        self._worker.request_update(self._get_config(), self._appearance)

    def _on_res_changed(self) -> None:
        self._preview_scale = float(self.cmb_res.currentData())
        self._update_res_warning()
        # Resolution changes the geometry the stages run on, so the whole sample
        # is re-scaled and recomputed from tier A. Cheap if a downscale; the
        # full bundle is cached on the worker so no reload/preprocess happens.
        if self._loaded:
            self.progress.show()
            self._worker.request_rescale(self._preview_scale, self._get_config(), self._appearance)

    def _update_res_warning(self) -> None:
        """Show/hide the reduced-resolution accuracy warning."""
        if self._preview_scale >= 0.999:
            self.res_warning.clear()
            self.res_warning.hide()
        else:
            self.res_warning.setText(
                "⚠ Reduced-resolution preview: veins/regions are computed on a coarser "
                "image and may differ from the full-resolution result. Switch to Full "
                "before judging a setting's final effect."
            )
            self.res_warning.show()

    def _on_appearance_changed(self) -> None:
        self._appearance = Appearance(
            show_veins=self.cb_veins.isChecked(),
            show_regions=self.cb_regions.isChecked(),
            show_vein_tissue=self.cb_tissue.isChecked(),
        )
        if self._loaded:
            self._worker.request_update(self._get_config(), self._appearance)

    def _on_intervein_clicked(self) -> None:
        if not self._loaded:
            return
        self.cb_regions.setChecked(True)
        self._appearance = Appearance(
            show_veins=self.cb_veins.isChecked(),
            show_regions=True,
            show_vein_tissue=self.cb_tissue.isChecked(),
        )
        self.progress.show()
        self._worker.request_intervein(self._get_config(), self._appearance)

    # -- results ----------------------------------------------------------
    def _on_job_started(self, msg: str) -> None:
        self.status.setText(msg)
        self.progress.show()

    def _on_result(self, result) -> None:
        self.progress.hide()
        if result.overlay_bgr is not None:
            self.view.set_pixmap(_bgr_to_qpixmap(result.overlay_bgr))
        if result.error:
            self.status.setText(f"⚠ {result.error} (showing last good overlay)")
            return
        bits = [f"{result.n_veins} veins"]
        if result.tier_ran in (TIER_A, TIER_B, TIER_C, TIER_D):
            t = result.timings_ms
            timing = ", ".join(f"{k.split('_')[0]} {v:.0f}ms" for k, v in t.items())
            bits.append(f"tier {result.tier_ran}" + (f" ({timing})" if timing else ""))
        if result.regions_stale and self.cb_regions.isChecked():
            bits.append("intervein stale — press Refresh")
        self.status.setText(" · ".join(bits))

    def _mark_intervein_stale(self) -> None:
        if self.cb_regions.isChecked():
            self.status.setText("Intervein params changed — press “Refresh intervein” to recompute.")

    # -- misc -------------------------------------------------------------
    def _set_params_enabled(self, enabled: bool) -> None:
        for w in (self.btn_preset, self.btn_fit):
            w.setEnabled(enabled)
        # View-dependent controls (vein/region display, intervein) are gated by
        # both load-state and the active view.
        self._sync_view_controls()

    def shutdown(self) -> None:
        """Stop the worker thread. Call from the host's closeEvent/accept/reject."""
        try:
            self._debounce.stop()
            self._worker.stop()
        except Exception:  # noqa: BLE001
            logger.exception("Error during pane shutdown")

    def closeEvent(self, event):  # noqa: N802
        self.shutdown()
        super().closeEvent(event)
