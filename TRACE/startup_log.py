"""Lightweight startup logger for diagnosing first-launch failures.

PyInstaller windowed builds have no console, so any traceback during
import/launch is invisible to the user. This module writes a plaintext
log to a predictable location alongside the .exe (or the TRACE/ source
folder in dev mode) so we can ask the user to send the log file when
something breaks.

Log location:
  - Frozen (PyInstaller):  <dir of TRACE.exe>/trace_startup.log
  - Dev:                   <TRACE source dir>/trace_startup.log

The log is truncated on every launch so the file always reflects the
most recent run, not a multi-session accumulation.
"""

from __future__ import annotations

import sys
import traceback
from datetime import datetime
from pathlib import Path
from typing import Optional

if getattr(sys, "frozen", False) and hasattr(sys, "_MEIPASS"):
    _LOG_DIR = Path(sys.executable).resolve().parent
else:
    _LOG_DIR = Path(__file__).resolve().parent

LOG_PATH = _LOG_DIR / "trace_startup.log"


def _ensure_log_initialised() -> None:
    """Open the log fresh on first call this session."""
    if getattr(_ensure_log_initialised, "_done", False):
        return
    try:
        LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(LOG_PATH, "w", encoding="utf-8") as f:
            f.write(f"TRACE startup log — {datetime.now().isoformat()}\n")
            f.write(f"  sys.frozen:   {getattr(sys, 'frozen', False)}\n")
            f.write(f"  sys.executable: {sys.executable}\n")
            f.write(f"  sys.version:  {sys.version}\n")
            f.write(f"  cwd:          {Path.cwd()}\n")
            f.write("-" * 70 + "\n")
        _ensure_log_initialised._done = True  # type: ignore[attr-defined]
    except Exception:
        # If we can't even write the log, there's nothing else to do —
        # silently degrade rather than crash the launcher.
        _ensure_log_initialised._done = True  # type: ignore[attr-defined]


def log(msg: str) -> None:
    """Append a timestamped line to the startup log."""
    _ensure_log_initialised()
    try:
        with open(LOG_PATH, "a", encoding="utf-8") as f:
            ts = datetime.now().strftime("%H:%M:%S.%f")[:-3]
            f.write(f"[{ts}] {msg}\n")
    except Exception:
        pass


def log_exception(prefix: str, exc: Optional[BaseException] = None) -> None:
    """Append a labeled traceback to the startup log.

    When `exc` is omitted, uses sys.exc_info() — call from inside an
    except block.
    """
    _ensure_log_initialised()
    try:
        with open(LOG_PATH, "a", encoding="utf-8") as f:
            ts = datetime.now().strftime("%H:%M:%S.%f")[:-3]
            f.write(f"[{ts}] {prefix}\n")
            if exc is not None:
                tb = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
            else:
                tb = traceback.format_exc()
            for line in tb.rstrip().splitlines():
                f.write(f"    {line}\n")
    except Exception:
        pass


def log_path_str() -> str:
    """Return the log path as a string for error-dialog messages."""
    return str(LOG_PATH)


def install_global_excepthook() -> None:
    """Route any uncaught Python exception into the startup log.

    PyInstaller windowed builds have no console, so a stray uncaught
    exception inside the Qt event loop (e.g. inside a signal handler)
    crashes the app silently with no traceback. This hook captures it.
    Idempotent.
    """
    if getattr(install_global_excepthook, "_done", False):
        return
    prev_hook = sys.excepthook

    def _hook(exc_type, exc_value, exc_tb) -> None:
        try:
            log_exception(f"UNCAUGHT EXCEPTION ({exc_type.__name__})", exc_value)
        except Exception:
            pass
        try:
            prev_hook(exc_type, exc_value, exc_tb)
        except Exception:
            pass

    sys.excepthook = _hook
    install_global_excepthook._done = True  # type: ignore[attr-defined]


def install_qt_message_handler() -> None:
    """Route Qt log messages (qDebug/qWarning/qCritical/qFatal) into the
    startup log. Napari emits its "Cannot show napari window" error via
    Qt's logging before raising/dialoging, so capturing this stream
    surfaces the real cause. Safe to call before QApplication exists.
    """
    if getattr(install_qt_message_handler, "_done", False):
        return
    try:
        from PyQt5.QtCore import QtCriticalMsg, QtDebugMsg, QtFatalMsg, QtInfoMsg, QtWarningMsg, qInstallMessageHandler
    except Exception:
        return

    _level_names = {
        QtDebugMsg: "DEBUG",
        QtInfoMsg: "INFO",
        QtWarningMsg: "WARNING",
        QtCriticalMsg: "CRITICAL",
        QtFatalMsg: "FATAL",
    }

    def _handler(mode, context, message: str) -> None:
        try:
            level = _level_names.get(mode, str(mode))
            log(f"QT[{level}] {message}")
        except Exception:
            pass

    qInstallMessageHandler(_handler)
    install_qt_message_handler._done = True  # type: ignore[attr-defined]


def install_logging_bridge() -> None:
    """Bridge Python's logging module into the startup log file.

    Libraries call ``logger.exception(...)`` to record a traceback before
    showing the user a sanitised error dialog. In a PyInstaller windowed
    build, those records normally vanish because no logging handler is
    attached and stderr is None. This handler captures every WARNING+
    record (and anything with exc_info, which covers logger.exception
    even at INFO level) and writes it through the startup log.
    Idempotent.
    """
    if getattr(install_logging_bridge, "_done", False):
        return
    import logging

    class _StartupLogHandler(logging.Handler):
        def emit(self, record):
            # Only capture warnings+ or anything with attached exc_info.
            # Otherwise the file fills up with INFO-level chatter from
            # urllib, torch, etc.
            if record.levelno < logging.WARNING and not record.exc_info:
                return
            try:
                msg = self.format(record)
                log(f"PYLOG[{record.levelname}] {record.name}: {msg}")
                if record.exc_info:
                    import traceback

                    tb = "".join(traceback.format_exception(*record.exc_info))
                    for line in tb.rstrip().splitlines():
                        log(f"    {line}")
            except Exception:
                pass

    handler = _StartupLogHandler()
    handler.setLevel(logging.DEBUG)  # filtering happens in emit() by level+exc_info
    handler.setFormatter(logging.Formatter("%(message)s"))
    root = logging.getLogger()
    root.addHandler(handler)
    # If the root logger's level is still WARNING (Python's default),
    # logger.exception(...) on a non-root logger may still get filtered.
    # Lower to INFO so logger.info() with exc_info=True propagates.
    if root.level > logging.INFO:
        root.setLevel(logging.INFO)
    install_logging_bridge._done = True  # type: ignore[attr-defined]
