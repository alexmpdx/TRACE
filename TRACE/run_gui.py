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

# Download bundled DL model weights on first launch (no-op once installed).
# Runs BEFORE importing TRACE.gui so any model-path defaults that resolve
# at import time see a fully-populated TRACE/models/.
#
# For a PyInstaller windowed build (no console), we wrap the download in a
# Qt QProgressDialog so end users see progress instead of a frozen window
# during the 1.6 GB first-launch fetch. The QApplication created here is
# reused by gui.main().
from TRACE.fetch_assets import DownloadCancelled, _has_models, ensure_assets  # noqa: E402


def _bootstrap_models() -> None:
    if _has_models():
        return  # Fast path — no UI needed.

    from PyQt5.QtCore import Qt
    from PyQt5.QtWidgets import QApplication, QMessageBox, QProgressDialog

    app = QApplication.instance() or QApplication(sys.argv)
    dlg = QProgressDialog(
        "Downloading TRACE models (~1.6 GB).\n" "This only happens on first launch.",
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
        dlg.close()
        QMessageBox.information(
            None,
            "TRACE setup cancelled",
            "Model download cancelled. Re-launch TRACE to try again.",
        )
        sys.exit(0)
    except Exception as e:  # noqa: BLE001
        dlg.close()
        QMessageBox.critical(
            None,
            "TRACE setup failed",
            f"Could not download model weights:\n\n{e}\n\n" "Check your internet connection and re-launch TRACE.",
        )
        sys.exit(1)
    dlg.close()


_bootstrap_models()

from TRACE.gui import main  # noqa: E402

if __name__ == "__main__":
    main()
