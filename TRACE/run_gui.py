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


def _probe_cpu_features() -> None:
    """Log Windows IsProcessorFeaturePresent flags for AVX/AVX2/SSE4.2.

    PyTorch 2.x CPU wheels for Windows are built requiring AVX2 — if the
    user's CPU lacks it, c10.dll's DllMain returns FALSE and the loader
    surfaces 'WinError 1114: DLL initialization routine failed'. This
    probe tells us up front whether the CPU is the problem.
    """
    try:
        import ctypes
        import platform

        log(f"probe: platform.processor() = {platform.processor()!r}")
        log(f"probe: platform.machine()   = {platform.machine()!r}")
        # IsProcessorFeaturePresent flag constants from winnt.h.
        _PF = {
            "SSE2": 10,
            "SSE3": 13,
            "SSE4_1": 37,
            "SSE4_2": 38,
            "AVX": 39,
            "AVX2": 40,
            "AVX512F": 41,
        }
        kernel32 = ctypes.windll.kernel32  # type: ignore[attr-defined]
        for name, flag in _PF.items():
            present = bool(kernel32.IsProcessorFeaturePresent(flag))
            log(f"probe: CPU.{name} = {present}")
    except Exception as e:  # noqa: BLE001
        log(f"probe: CPU feature probe failed: {e}")


def _probe_c10_direct_load() -> None:
    """Try to load c10.dll directly via ctypes.WinDLL.

    This bypasses Python's import machinery so the Windows loader error
    surfaces cleanly (with a GetLastError code we can map). Useful when
    the higher-level `import torch` traceback is opaque about which
    specific DLL is failing.
    """
    if not getattr(sys, "frozen", False):
        return  # only meaningful in the bundled layout
    try:
        import ctypes

        torch_lib_dir = Path(sys.executable).resolve().parent / "_internal" / "torch" / "lib"
        if not torch_lib_dir.is_dir():
            log(f"probe: c10.dll direct load skipped (no {torch_lib_dir})")
            return
        # add_dll_directory ensures torch_global_deps.dll etc. resolve
        # without polluting PATH.
        import os

        if hasattr(os, "add_dll_directory"):
            try:
                os.add_dll_directory(str(torch_lib_dir))
            except Exception:
                pass
        c10 = torch_lib_dir / "c10.dll"
        try:
            handle = ctypes.WinDLL(str(c10))
            log(f"probe: c10.dll direct load OK (handle={handle._handle:#x})")
        except OSError as e:
            log(f"probe: c10.dll direct load FAILED")
            log(f"    OSError.winerror = {e.winerror}")
            log(f"    OSError.strerror = {e.strerror}")
            log(f"    OSError.args     = {e.args}")
            # winerror 1114 = ERROR_DLL_INIT_FAILED
            # winerror 126  = ERROR_MOD_NOT_FOUND (missing dep)
            # winerror 127  = ERROR_PROC_NOT_FOUND (missing export)
            # winerror 193  = ERROR_BAD_EXE_FORMAT (arch mismatch)
            if e.winerror == 1114:
                log("    -> c10.dll's DllMain returned FALSE. Most common")
                log("       cause on modern PyTorch CPU wheels: CPU lacks")
                log("       AVX2. Check the CPU.AVX2 line above.")
    except Exception as e:  # noqa: BLE001
        log(f"probe: c10.dll direct-load probe failed: {e}")


def _probe_torch_environment() -> None:
    """Eagerly import torch and dump its bundled-DLL inventory.

    Torch's c10.dll fails to load on some Windows installs with
    'WinError 1114: A dynamic link library (DLL) initialization routine
    failed' — usually because the CPU lacks AVX2. Dumping the file list
    + CPU features lets us tell from the log whether PyInstaller bundled
    what we expect AND whether the host CPU can run it.
    """
    _probe_cpu_features()
    _probe_c10_direct_load()
    log("probe: importing torch")
    try:
        import torch  # type: ignore[import-untyped]  # noqa: F401

        log(f"probe: torch import OK (version={getattr(torch, '__version__', '?')})")
    except BaseException as exc:  # noqa: BLE001
        log_exception("probe: torch import FAILED", exc)
    # Inventory the torch/lib directory regardless of import success — even
    # if import fails for another reason, this tells us what's actually on
    # disk vs. what the error claims is missing.
    try:
        import torch as _torch  # noqa: F811

        torch_lib_dir = Path(_torch.__file__).resolve().parent / "lib"
    except Exception:
        # If we can't import torch, fall back to the bundled-frozen path
        # (run_gui.py lives at <exe-dir>/TRACE/run_gui.py for the source,
        # but at runtime _internal/TRACE for the frozen layout). Try a
        # few likely locations.
        candidates = []
        if getattr(sys, "frozen", False):
            base = Path(sys.executable).resolve().parent
            candidates += [
                base / "_internal" / "torch" / "lib",
                base / "torch" / "lib",
            ]
        for p in candidates:
            if p.is_dir():
                torch_lib_dir = p
                break
        else:
            torch_lib_dir = None
    if torch_lib_dir and torch_lib_dir.is_dir():
        try:
            dll_files = sorted(p.name for p in torch_lib_dir.iterdir() if p.suffix.lower() == ".dll")
            log(f"probe: torch/lib at {torch_lib_dir} contains {len(dll_files)} DLLs:")
            for name in dll_files:
                log(f"    {name}")
        except Exception as e:  # noqa: BLE001
            log(f"probe: could not list torch/lib: {e}")
    else:
        log(f"probe: torch/lib directory not found (looked at {torch_lib_dir})")
    # Microsoft Visual C++ runtime probe — c10.dll's DllMain depends on
    # vcruntime140.dll + msvcp140.dll. If PyInstaller bundled them they
    # land in the dist root; if not, c10.dll falls back to the system
    # Windows\System32 copy (or errors if absent there too).
    if getattr(sys, "frozen", False):
        exe_dir = Path(sys.executable).resolve().parent
        for runtime in ("vcruntime140.dll", "vcruntime140_1.dll", "msvcp140.dll", "msvcp140_1.dll"):
            for candidate in (exe_dir / runtime, exe_dir / "_internal" / runtime):
                if candidate.is_file():
                    log(f"probe: {runtime} -> bundled at {candidate}")
                    break
            else:
                # Check the system copy via ctypes — that's what Windows
                # loader will actually fall back to.
                try:
                    import ctypes

                    h = ctypes.WinDLL(runtime)
                    log(f"probe: {runtime} -> NOT bundled, system load OK (handle={h._handle:#x})")
                except OSError as e:
                    log(f"probe: {runtime} -> NOT bundled AND system load FAILED: {e}")


_probe_torch_environment()


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
