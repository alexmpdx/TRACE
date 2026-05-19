#!/usr/bin/env python3
"""Entry point for the TRACE combined pipeline GUI."""

import sys
from pathlib import Path

# Add sibling package directories to sys.path
_project_root = str(Path(__file__).resolve().parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

_hinge_dir = str(Path(__file__).resolve().parent.parent / "HingeChopper")
if _hinge_dir not in sys.path:
    sys.path.insert(0, _hinge_dir)

_mtj_dir = str(Path(__file__).resolve().parent.parent / "modelTOjson")
if _mtj_dir not in sys.path:
    sys.path.insert(0, _mtj_dir)

_idf_dir = str(Path(__file__).resolve().parent.parent / "identifyFeatures")
if _idf_dir not in sys.path:
    sys.path.insert(0, _idf_dir)

_rot_dir = str(Path(__file__).resolve().parent.parent / "wingRotator")
if _rot_dir not in sys.path:
    sys.path.insert(0, _rot_dir)

_mm_dir = str(Path(__file__).resolve().parent.parent / "measurementMaker")
if _mm_dir not in sys.path:
    sys.path.insert(0, _mm_dir)

_se_dir = str(Path(__file__).resolve().parent.parent / "scaleEstimator")
if _se_dir not in sys.path:
    sys.path.insert(0, _se_dir)

# --- Startup logging ------------------------------------------------------
# Initialise the file logger as early as possible so anything that fails
# below (imports, Qt setup, napari plugin load, etc.) leaves a trace.
from TRACE.startup_log import (  # noqa: E402
    install_global_excepthook,
    install_logging_bridge,
    install_qt_message_handler,
    log,
    log_exception,
    log_path_str,
)

# Install the global Python excepthook first so any uncaught exception
# from this point on (including inside Qt slots) gets logged.
install_global_excepthook()
# Qt message handler captures qWarning / qCritical / qFatal emissions —
# napari uses Qt logging before its "Cannot show napari window" dialog
# fires, so the real underlying error lands here.
install_qt_message_handler()
# Python logging bridge — measurement_maker.embedded_picker calls
# logger.exception("Failed to load wing into picker") and then shows a
# user-facing QMessageBox. The traceback that exception emits is the
# real cause of the napari "Cannot show napari window" message.
install_logging_bridge()

log("run_gui.py: launcher entry")
log(f"sys.path[:8] = {sys.path[:8]}")


# --- Model bootstrap ------------------------------------------------------
# Download bundled DL model weights on first launch (no-op once installed).
# Runs BEFORE importing TRACE.gui so any model-path defaults that resolve
# at import time see a fully-populated TRACE/models/.
#
# For a PyInstaller windowed build (no console), we wrap the download in a
# Qt QProgressDialog so end users see progress instead of a frozen window
# during the 1.6 GB first-launch fetch. The QApplication created here is
# reused by gui.main().
def _bootstrap_models() -> None:
    log("bootstrap: importing fetch_assets")
    from TRACE.fetch_assets import DownloadCancelled, _has_models, ensure_assets  # noqa: E402

    if _has_models():
        log("bootstrap: models already present, skipping download")
        return  # Fast path — no UI needed.

    log("bootstrap: models not found, starting download")
    from PyQt5.QtCore import Qt
    from PyQt5.QtWidgets import QApplication, QMessageBox, QProgressDialog

    app = QApplication.instance() or QApplication(sys.argv)
    dlg = QProgressDialog(
        "Downloading TRACE models (~1.6 GB).\nThis only happens on first launch.",
        "Cancel",
        0,
        100,
    )
    dlg.setWindowTitle("Setting up TRACE")
    dlg.setWindowModality(Qt.WindowModal)
    dlg.setMinimumDuration(0)
    dlg.setAutoClose(False)
    dlg.setAutoReset(False)
    dlg.setValue(0)
    dlg.show()
    app.processEvents()

    def progress(downloaded: int, total: int) -> bool:
        if total:
            pct = int(downloaded * 100 / total)
            mb_done = downloaded // (1024 * 1024)
            mb_total = total // (1024 * 1024)
            dlg.setLabelText(f"Downloading TRACE models…\n{mb_done} / {mb_total} MB ({pct}%)")
            dlg.setValue(pct)
        app.processEvents()
        return dlg.wasCanceled()

    try:
        ensure_assets(progress_callback=progress)
    except DownloadCancelled:
        log("bootstrap: download cancelled by user")
        dlg.close()
        QMessageBox.information(
            None,
            "TRACE setup cancelled",
            "Model download cancelled. Re-launch TRACE to try again.",
        )
        sys.exit(0)
    except Exception as e:  # noqa: BLE001
        log_exception("bootstrap: model download failed", e)
        dlg.close()
        QMessageBox.critical(
            None,
            "TRACE setup failed",
            f"Could not download model weights:\n\n{e}\n\n"
            f"Check your internet connection and re-launch TRACE.\n"
            f"Log: {log_path_str()}",
        )
        sys.exit(1)
    log("bootstrap: download complete")
    dlg.close()


# --- Main ------------------------------------------------------------------
def _show_fatal_error(title: str, body: str) -> None:
    """Best-effort error dialog. Falls back silently if Qt isn't available."""
    try:
        from PyQt5.QtWidgets import QApplication, QMessageBox

        app = QApplication.instance() or QApplication(sys.argv)
        QMessageBox.critical(None, title, body)
        del app
    except Exception:
        pass


try:
    _bootstrap_models()

    log("import: TRACE.gui")
    from TRACE.gui import main  # noqa: E402

    log("import: TRACE.gui OK")
except BaseException as exc:  # noqa: BLE001
    log_exception("import-time fatal error", exc)
    _show_fatal_error(
        "TRACE failed to start",
        f"An error occurred during startup:\n\n{type(exc).__name__}: {exc}\n\n"
        f"A full traceback has been written to:\n{log_path_str()}\n\n"
        "Please send that file when reporting the problem.",
    )
    sys.exit(1)


if __name__ == "__main__":
    try:
        log("calling gui.main()")
        main()
    except SystemExit:
        raise
    except BaseException as exc:  # noqa: BLE001
        log_exception("runtime fatal error in gui.main()", exc)
        _show_fatal_error(
            "TRACE crashed",
            f"An error occurred while running:\n\n{type(exc).__name__}: {exc}\n\n" f"Log: {log_path_str()}",
        )
        sys.exit(1)
