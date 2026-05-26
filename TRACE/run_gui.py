#!/usr/bin/env python3
"""Entry point for the TRACE combined pipeline GUI."""

# multiprocessing.freeze_support() must be called BEFORE any code that
# can spawn child processes. On Windows the default start method is
# "spawn", which re-executes the frozen entry point in a child process
# — without freeze_support the child re-runs the GUI launcher path
# instead of doing whatever the parent told it to do. Calling it
# unconditionally is safe in dev mode (it's a no-op when not frozen).
import multiprocessing
import sys
from pathlib import Path

multiprocessing.freeze_support()

# CLI dispatch sentinel. When TRACE.exe is invoked with this sentinel
# as argv[1], skip the GUI and run identify_features.cli on the
# remaining argv. recommend_workers (the Calibrate Workers backend)
# uses this to spawn a clean inference subprocess without re-opening
# the full TRACE main window — sys.executable inside the bundled exe
# is TRACE.exe, not python.exe, so `python -m identify_features.cli`
# isn't a valid invocation pattern there.
if len(sys.argv) > 1 and sys.argv[1] == "__identify_features_cli__":
    # Add sibling dirs to sys.path before the import — identify_features
    # lives in identifyFeatures/, alongside TRACE/, in the bundled layout.
    _bundle_root = Path(sys.executable).resolve().parent
    for _sub in ("identifyFeatures", "HingeChopper", "modelTOjson", "wingRotator", "preprocessing"):
        _p = str(_bundle_root / _sub)
        if (_bundle_root / _sub).is_dir() and _p not in sys.path:
            sys.path.insert(0, _p)
    sys.argv = sys.argv[:1] + sys.argv[2:]  # drop the sentinel before forwarding
    from identify_features.cli import main as _if_main

    sys.exit(_if_main())

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
    """Try to load c10.dll directly via kernel32.LoadLibraryW.

    We bypass `ctypes.WinDLL` because PyInstaller wraps it to add its
    own DLL search paths AND substitutes its own error message ("Most
    likely this dynlib/dll was not found when the application was
    frozen") — which swallows the real Windows GetLastError code.

    Calling LoadLibraryW directly returns NULL on failure, and we can
    pull the real Windows error number out via GetLastError + format it
    via FormatMessageW.
    """
    if not getattr(sys, "frozen", False):
        return  # only meaningful in the bundled layout
    try:
        import ctypes
        from ctypes import wintypes

        torch_lib_dir = Path(sys.executable).resolve().parent / "_internal" / "torch" / "lib"
        if not torch_lib_dir.is_dir():
            log(f"probe: c10.dll direct load skipped (no {torch_lib_dir})")
            return
        # add_dll_directory ensures sibling DLLs (torch_global_deps.dll
        # etc.) resolve without polluting PATH. Required on Python 3.8+.
        import os

        added_dirs = []
        if hasattr(os, "add_dll_directory"):
            try:
                added_dirs.append(os.add_dll_directory(str(torch_lib_dir)))
            except Exception:
                pass

        # LoadLibraryW signature
        kernel32 = ctypes.windll.kernel32  # type: ignore[attr-defined]
        kernel32.LoadLibraryW.argtypes = [wintypes.LPCWSTR]
        kernel32.LoadLibraryW.restype = wintypes.HMODULE
        kernel32.GetLastError.restype = wintypes.DWORD
        # LoadLibraryExW lets us use LOAD_WITH_ALTERED_SEARCH_PATH so
        # dependent DLL resolution searches torch/lib alongside c10.dll.
        LOAD_WITH_ALTERED_SEARCH_PATH = 0x00000008
        kernel32.LoadLibraryExW.argtypes = [wintypes.LPCWSTR, wintypes.HANDLE, wintypes.DWORD]
        kernel32.LoadLibraryExW.restype = wintypes.HMODULE

        c10 = str(torch_lib_dir / "c10.dll")
        handle = kernel32.LoadLibraryExW(c10, None, LOAD_WITH_ALTERED_SEARCH_PATH)
        if handle:
            log(f"probe: c10.dll LoadLibraryExW OK (handle={handle:#x})")
            return
        err = kernel32.GetLastError()
        # FormatMessageW for the human-readable string.
        FORMAT_MESSAGE_FROM_SYSTEM = 0x00001000
        FORMAT_MESSAGE_IGNORE_INSERTS = 0x00000200
        buf = ctypes.create_unicode_buffer(2048)
        kernel32.FormatMessageW(
            FORMAT_MESSAGE_FROM_SYSTEM | FORMAT_MESSAGE_IGNORE_INSERTS,
            None,
            err,
            0,
            buf,
            len(buf),
            None,
        )
        log(f"probe: c10.dll LoadLibraryExW FAILED")
        log(f"    GetLastError = {err} (0x{err:08x})")
        log(f"    message      = {buf.value.strip()!r}")
        if err == 1114:
            log("    -> ERROR_DLL_INIT_FAILED: c10.dll's DllMain returned FALSE.")
            log("       AVX2 was confirmed available, so this is most likely")
            log("       a missing transitive DLL or a torch-internal CPU init")
            log("       failure. Try pinning torch to a known-good version.")
        elif err == 126:
            log("    -> ERROR_MOD_NOT_FOUND: a dependency DLL is missing.")
        elif err == 127:
            log("    -> ERROR_PROC_NOT_FOUND: a dependency exports a missing symbol.")
        elif err == 193:
            log("    -> ERROR_BAD_EXE_FORMAT: 32/64-bit architecture mismatch.")
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
    # Set the app icon early so the first-launch download dialog gets the
    # TRACE logo in its title bar / taskbar entry instead of the default Qt
    # icon. Path resolution mirrors output_tooltips._GUI_IMAGES_DIR.
    from PyQt5.QtGui import QIcon

    if getattr(sys, "frozen", False) and hasattr(sys, "_MEIPASS"):
        _logo = Path(sys._MEIPASS) / "TRACE" / "GUI_images" / "logo" / "logo_dark.svg"
    else:
        _logo = Path(__file__).resolve().parent / "GUI_images" / "logo" / "logo_dark.svg"
    if _logo.is_file():
        app.setWindowIcon(QIcon(str(_logo)))
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


def _migrate_gate_defaults_to_permissive() -> None:
    """One-shot fixup: if the bundled gate_config.yaml's active gate
    still matches the Standard tier, replace it with the Permissive
    tier values.

    The model zip shipped via fetch_assets baked the Standard tier into
    the active `confidence:` block. Users have reported those thresholds
    gating out real landmarks on otherwise-fine wings. Without re-uploading
    a 1.6 GB models bundle, this migration rewrites the YAML in place on
    every launch. It's a no-op when the YAML has already been migrated
    (active gate != standard) OR when the user customised the gate.

    Long-term plan: LandmarkLocator owners retune + re-publish the models
    zip; this migration can then be removed.
    """
    try:
        if getattr(sys, "frozen", False) and hasattr(sys, "_MEIPASS"):
            models_dir = Path(sys.executable).resolve().parent / "TRACE" / "models"
        else:
            models_dir = Path(__file__).resolve().parent / "models"
        gate_path = models_dir / "landmarks" / "gate_config.yaml"
        if not gate_path.is_file():
            return
        import yaml

        data = yaml.safe_load(gate_path.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            return
        conf = data.get("confidence")
        if not isinstance(conf, dict):
            return
        tiers = conf.get("tiers") if isinstance(conf.get("tiers"), dict) else {}
        permissive = tiers.get("permissive") if isinstance(tiers, dict) else None
        standard = tiers.get("standard") if isinstance(tiers, dict) else None
        if not (isinstance(permissive, dict) and isinstance(standard, dict)):
            return
        # Heuristic: active gate counts as "still Standard" iff its peak/
        # sharpness/second_peak_ratio sub-dicts exactly match the standard
        # tier sub-block. Any user-side tweak breaks the equality and
        # leaves the file untouched.
        for section in ("peak", "sharpness", "second_peak_ratio"):
            if conf.get(section) != standard.get(section):
                log(f"gate-migration: active.{section} differs from standard - skipping")
                return
        for section in ("peak", "sharpness", "second_peak_ratio"):
            if section in permissive:
                conf[section] = permissive[section]
        gate_path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
        log(f"gate-migration: rewrote {gate_path} active gate Standard -> Permissive")
    except Exception as exc:  # noqa: BLE001
        log_exception("gate-migration: failed", exc)


try:
    _bootstrap_models()
    _migrate_gate_defaults_to_permissive()

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
