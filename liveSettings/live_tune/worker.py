"""Background recompute worker for the live-tuning preview.

All heavy work — sample loading, tier A/B/D recompute, and on-demand tier C
intervein — runs on a single serialized QThread so the :class:`LiveTuneSession`
is only ever touched from one thread (no locking inside the session needed).

Requests coalesce: only the latest pending job is kept, so a fast drag of a
slider never queues a backlog — intermediate configs are dropped and only the
most recent one is computed once the current job finishes.
"""

from __future__ import annotations

import logging
from typing import Callable, Optional

from PyQt5.QtCore import QMutex, QThread, QWaitCondition, pyqtSignal

from .session import VIEW_FINAL, Appearance, LiveTuneSession, RenderResult

logger = logging.getLogger(__name__)


class LiveTuneWorker(QThread):
    """Serialized worker thread that owns recompute against a LiveTuneSession."""

    # Emitted with a RenderResult after every job.
    result_ready = pyqtSignal(object)
    # Emitted just before a job starts, with a short human status string.
    job_started = pyqtSignal(str)
    # Emitted when a load job succeeds (provenance string) or fails (error).
    load_done = pyqtSignal(bool, str)

    def __init__(self, session: LiveTuneSession, parent=None) -> None:
        super().__init__(parent)
        self._session = session
        self._mutex = QMutex()
        self._cond = QWaitCondition()
        self._pending: Optional[tuple] = None
        self._abort = False
        # Most recently loaded FULL-resolution bundle, kept so a resolution
        # change can re-scale without re-running the (slow) loader/preprocessing.
        self._full_bundle = None
        # Active view mode (skeleton / traced / final). Set from the GUI thread
        # via set_view; read by every job so the session renders that product.
        self._view = VIEW_FINAL

    def set_view(self, view: str) -> None:
        """Set the view rendered by subsequent jobs (call from GUI thread)."""
        self._view = view

    # -- request API (call from GUI thread) ------------------------------
    def request_load(self, loader: Callable[[], object], scale: float, config, appearance: Appearance) -> None:
        """Queue a sample load. ``loader`` returns an InputBundle (called off-thread)."""
        self._enqueue(("load", loader, scale, config, appearance))

    def request_rescale(self, scale: float, config, appearance: Appearance) -> None:
        """Re-scale the already-loaded sample to a new preview resolution."""
        self._enqueue(("rescale", scale, config, appearance))

    def request_update(self, config, appearance: Appearance) -> None:
        self._enqueue(("update", config, appearance))

    def request_intervein(self, config, appearance: Appearance) -> None:
        self._enqueue(("intervein", config, appearance))

    def _enqueue(self, job: tuple) -> None:
        self._mutex.lock()
        self._pending = job  # coalesce: latest wins
        self._cond.wakeOne()
        self._mutex.unlock()

    def stop(self) -> None:
        self._mutex.lock()
        self._abort = True
        self._cond.wakeOne()
        self._mutex.unlock()
        self.wait(5000)

    # -- thread body -----------------------------------------------------
    def run(self) -> None:  # noqa: C901 - small state machine
        while True:
            self._mutex.lock()
            while self._pending is None and not self._abort:
                self._cond.wait(self._mutex)
            if self._abort:
                self._mutex.unlock()
                return
            job = self._pending
            self._pending = None
            self._mutex.unlock()

            try:
                self._run_job(job)
            except Exception:  # noqa: BLE001 - never let the thread die
                logger.exception("Worker job crashed")

    def _run_job(self, job: tuple) -> None:
        kind = job[0]
        if kind == "load":
            _, loader, scale, config, appearance = job
            self.job_started.emit("Loading sample…")
            try:
                from .input_loader import apply_to_session, scale_bundle

                bundle = loader()
                self._full_bundle = bundle
                apply_to_session(scale_bundle(bundle, scale), self._session)
                self.load_done.emit(True, bundle.source)
            except Exception as exc:  # noqa: BLE001
                logger.exception("Sample load failed")
                self.load_done.emit(False, f"{type(exc).__name__}: {exc}")
                return
            # Auto-run the first overlay after a successful load.
            self.job_started.emit("Building skeleton…")
            self.result_ready.emit(self._session.update(config, appearance, view=self._view))
            return

        if kind == "rescale":
            _, scale, config, appearance = job
            if self._full_bundle is None:
                return
            self.job_started.emit("Rescaling preview…")
            try:
                from .input_loader import apply_to_session, scale_bundle

                apply_to_session(scale_bundle(self._full_bundle, scale), self._session)
            except Exception as exc:  # noqa: BLE001
                logger.exception("Rescale failed")
                self.result_ready.emit(
                    RenderResult(
                        overlay_bgr=self._session._last_overlay,
                        tier_ran="error",
                        error=f"{type(exc).__name__}: {exc}",
                    )
                )
                return
            self.result_ready.emit(self._session.update(config, appearance, view=self._view))
            return

        if kind == "update":
            _, config, appearance = job
            self.job_started.emit("Recomputing…")
            self.result_ready.emit(self._session.update(config, appearance, view=self._view))
            return

        if kind == "intervein":
            _, config, appearance = job
            self.job_started.emit("Computing intervein regions…")
            result: RenderResult
            try:
                self._session.compute_intervein(config)
                overlay = self._session.render_current(config, appearance, view=self._view)
                result = RenderResult(
                    overlay_bgr=overlay,
                    tier_ran="C",
                    n_veins=sum(1 for v in self._session._veins if v.centerline is not None),
                    regions_stale=False,
                )
            except Exception as exc:  # noqa: BLE001
                logger.exception("Intervein compute failed")
                result = RenderResult(
                    overlay_bgr=self._session._last_overlay,
                    tier_ran="error",
                    error=f"{type(exc).__name__}: {exc}",
                )
            self.result_ready.emit(result)
            return
