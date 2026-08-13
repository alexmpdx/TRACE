"""PyQt5 GUI for the TRACE combined pipeline.

Dark Fusion theme matching the preprocessing app. Runs the pipeline in a
background QThread with progress reporting.
"""

import logging
import re
import sys
import time
import traceback
from collections import OrderedDict
from enum import Enum
from pathlib import Path
from typing import Optional

from identify_features.config import PipelineConfig
from PyQt5.QtCore import QEvent, QObject, QPoint, QRect, QSettings, Qt, QThread, QTimer, pyqtSignal
from PyQt5.QtGui import QColor, QPalette
from PyQt5.QtWidgets import (
    QApplication,
    QCheckBox,
    QDialog,
    QDoubleSpinBox,
    QFileDialog,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QMainWindow,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QSplitter,
    QTabWidget,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from preprocessing.pipeline import discover_images
from TRACE.config_io import config_from_json, config_to_json
from TRACE.inline_panels import (
    InlineCustomDistancesPanel,
    InlineGeneralPanel,
    InlineHelpPanel,
    _DependentRow,
    _pulse_text,
)
from TRACE.ood_check import format_report_line, preflight_batch
from TRACE.output_tooltips import output_tooltip_html
from TRACE.pipeline import (
    DEFAULT_MAX_WORKERS,
    INTERMEDIATE_OUTPUTS,
    MEASUREMENT_GROUP_TOOLTIPS,
    MEASUREMENT_GROUPS,
    OUTPUT_TOOLTIPS,
    OUTPUT_TYPES,
    _required_stages,
    compute_progress_weights,
    trace_folder,
)
from TRACE.run_state import (
    STATUS_COMPLETED,
    STATUS_PAUSED,
    STATUS_RUNNING,
    RunManifest,
    load_manifest,
    new_manifest,
    save_manifest,
)
from TRACE.settings_dialog import PipelineConfigDialog
from TRACE.walkthrough import WalkthroughOverlay, WalkthroughStep

# ---------------------------------------------------------------------------
# Worker thread
# ---------------------------------------------------------------------------


_CAPTURED_LOGGERS = ("identify_features", "TRACE", "preprocessing", "scale_estimator")


class _PlaceholderSpinBox(QDoubleSpinBox):
    """QDoubleSpinBox that shows a real QLineEdit placeholder (auto-dimmed, clears on
    focus) instead of `setSpecialValueText`. When the user hasn't entered a value
    (spinbox is at its minimum), `textFromValue` returns an empty string so
    QLineEdit's normal placeholder rendering kicks in.

    Configure by calling `set_placeholder("...")` on the instance.

    Also accepts empty text as "unset" so the user can Backspace / Delete a
    previously-entered value back to blank. Vanilla QDoubleSpinBox's default
    ``validate()`` refuses empty text and its ``fixup()`` reverts the widget
    to the last valid value — this override maps empty to ``minimum()``,
    which is what the rest of the code already treats as "not set" (e.g.
    ``_set_scale`` writes ``None`` to config when val <= minimum).
    """

    def set_placeholder(self, text: str) -> None:
        self.lineEdit().setPlaceholderText(text)
        # Force the displayed text to refresh — without this the previous numeric
        # rendering can linger until the next value change.
        self.lineEdit().setText(self.textFromValue(self.value()))

    def textFromValue(self, value: float) -> str:  # noqa: N802 — Qt API
        if value == self.minimum() and self.lineEdit().placeholderText():
            return ""
        return super().textFromValue(value)

    def valueFromText(self, text: str) -> float:  # noqa: N802 — Qt API
        # Empty text → minimum() so the placeholder shows again and the
        # existing "val > minimum → real value, else None" plumbing writes
        # None to config.um_per_px.
        if not text.strip():
            return self.minimum()
        return super().valueFromText(text)

    def validate(self, text: str, pos: int):  # noqa: N802 — Qt API
        # Accept blank as valid so the user can clear the field. Without
        # this, Qt calls fixup() and reverts to the last numeric value the
        # moment the user finishes editing an empty field.
        from PyQt5.QtGui import QValidator

        if not text.strip():
            return QValidator.Acceptable, text, pos
        return super().validate(text, pos)


# Bundled default-model folders. Used to preload the three model paths on a
# first-time launch (no saved QSettings) and after "wipe my memories" so the
# user doesn't have to browse for them. Each entry is checked for existence
# at runtime — missing folders fall back to "" (empty), forcing the user to
# pick one in Settings → Models.
#
# Resolution: in a PyInstaller onedir bundle, __file__ for this module
# lives inside the per-launch temp _internal/TRACE/ folder which is
# wiped between sessions. The actual writable models/ folder is next to
# TRACE.exe (this is where TRACE/fetch_assets.py puts the downloaded
# weights). Mirror that resolution here so the Models tab picks them up
# automatically on first launch instead of leaving the picker empty.
if getattr(sys, "frozen", False) and hasattr(sys, "_MEIPASS"):
    _DEFAULT_MODELS_DIR = Path(sys.executable).resolve().parent / "TRACE" / "models"
else:
    _DEFAULT_MODELS_DIR = Path(__file__).resolve().parent / "models"
_DEFAULT_MODEL_PATHS = {
    "landmark": _DEFAULT_MODELS_DIR / "landmarks",
    "segmentation": _DEFAULT_MODELS_DIR / "vein-intervein",
    "wing_isolation": _DEFAULT_MODELS_DIR / "wingIsolation",
}


def _default_model_path(key: str) -> str:
    """Return the bundled default model folder for `key` as a string, or "" if missing.

    Keys: "landmark", "segmentation", "wing_isolation".
    """
    path = _DEFAULT_MODEL_PATHS.get(key)
    if path is None:
        return ""
    return str(path) if path.is_dir() else ""


def _restore_model_path(saved: str, default_key: str) -> str:
    """Map a saved-QSettings model path to a usable path on disk.

    Resolution order, returning the first match:
      1. Saved path exists on disk → use as-is.
      2. Saved path is the legacy nested-checkpoints layout
         (parent dir contains ``best_fold*.pt``) → return the parent.
         LandmarkLocator used to ship checkpoints inside a
         ``<name>_checkpoints/`` sub-folder; the flat layout puts the
         weights directly in the model folder.
      3. Saved path is stale (project dir was moved or deleted between
         sessions) but a bundled default exists under ``TRACE/models/``
         → fall back to the bundled default. Without this rescue,
         relocating the repo silently breaks every saved path and the
         user has to re-pick three model folders by hand.
      4. Otherwise return the saved value unchanged so the missing path
         stays visible in Settings and the user can re-pick it.
    """
    if not saved:
        return _default_model_path(default_key)
    p = Path(saved)
    if p.exists():
        return saved
    parent = p.parent
    if parent.exists() and parent.is_dir() and any(parent.glob("best_fold*.pt")):
        return str(parent)
    fallback = _default_model_path(default_key)
    return fallback if fallback else saved


def _picker_initial_path(current: str) -> str:
    """Sane initial path for QFileDialog: walk up `current` to the first existing
    file/dir, or fall back to '/' (Finder's 'Computer' view) when nothing in the
    saved path exists. Empty input always returns '/'.
    """
    if current:
        p = Path(current)
        if p.exists():
            return str(p)
        while p != p.parent:
            p = p.parent
            if p.exists():
                return str(p)
    return "/"


def _pick_folder_native(caption: str, initial: str) -> str:
    """Show the native folder picker, working around the Qt-on-macOS bug.

    Symptom: when ``QFileDialog.getExistingDirectory`` is called with a Qt
    parent widget that's modally active, Qt's modal-event grab intercepts
    mouse events that should land on the native NSOpenPanel's file-list
    pane. The path-navigator dropdown and the Open / Cancel buttons live
    in chrome that isn't covered by the grab so they keep working — only
    the file list goes dead. Detaching the dialog from any Qt parent
    (passing ``None``) restores click handling. Cost: the picker isn't
    parented to the main window, so it appears at the system position
    rather than centered on / sheeted to TRACE — a one-time visual
    surprise rather than a persistent annoyance.

    Behaviour on Windows / Linux is unchanged in practice: their native
    pickers don't suffer from this and the unparented presentation is
    functionally identical for the user.

    Returns the chosen absolute path, or an empty string if cancelled.
    """
    return QFileDialog.getExistingDirectory(None, caption, initial)


def _pick_file_native(caption: str, initial: str, name_filter: str = "") -> str:
    """File-picker counterpart of :func:`_pick_folder_native`.

    Same root cause + same workaround: Qt's modal-event grab on a Qt
    parent blocks file-list clicks when ``QFileDialog.getOpenFileName``
    is called with one. Detaching (``parent=None``) restores click
    handling at the cost of an unparented dialog. ``name_filter``
    follows the standard Qt syntax (``"Images (*.tif *.png);;All Files (*)"``).

    Returns the chosen absolute path, or an empty string if cancelled.
    """
    path, _ = QFileDialog.getOpenFileName(None, caption, initial, name_filter)
    return path


def _pick_save_file_native(caption: str, initial: str, name_filter: str = "") -> str:
    """Save-as picker — same Qt-on-macOS workaround as the open / folder pickers.

    ``initial`` may be a pre-filled filename (the macOS save panel
    inherits the basename) or a directory path. Returns the chosen
    absolute path or an empty string if the user cancelled.
    """
    path, _ = QFileDialog.getSaveFileName(None, caption, initial, name_filter)
    return path


# QSettings sub-namespace for per-picker last-visited-directory memory.
# Each picker call site passes its own ``last_dir_key`` so its Browse
# button remembers ITS path independently of the others — without this,
# macOS NSOpenPanel falls back to a single per-process "last visited"
# cache and every picker opens at whichever folder was used most recently
# anywhere in TRACE (input picker landing at the output folder etc).
_PICKER_LAST_DIR_QSETTINGS_PREFIX = "picker_last_dir/"


def _get_picker_last_dir(key: str) -> str:
    if not key:
        return ""
    s = QSettings("TRACE", "WingAnalysisPipeline")
    return s.value(_PICKER_LAST_DIR_QSETTINGS_PREFIX + key, "", type=str)


def _save_picker_last_dir(key: str, path: str) -> None:
    if not key or not path:
        return
    s = QSettings("TRACE", "WingAnalysisPipeline")
    s.setValue(_PICKER_LAST_DIR_QSETTINGS_PREFIX + key, path)
    s.sync()


def _open_native_picker_async(
    holder,
    caption: str,
    initial: str,
    on_picked,
    *,
    name_filter: str = "",
    folder: bool = False,
    save: bool = False,
    last_dir_key: str = "",
    sync: Optional[bool] = None,
) -> None:
    """Native picker with per-picker last-directory memory.

    Two execution modes, chosen via ``sync``:

    - ``sync=True`` uses the static ``QFileDialog.getExistingDirectory``
      / ``getOpenFileName`` / ``getSaveFileName`` helpers with
      ``parent=None``. These block the calling code until the user
      picks or cancels, and — critically — macOS NSOpenPanel actually
      honors the passed-in initial directory + per-picker last-dir
      memory in this path. The blocking nested event loop is fine
      wherever napari isn't loaded.

    - ``sync=False`` uses ``QFileDialog.open()`` + the ``fileSelected``
      signal — asynchronous, no nested event loop. Required whenever
      napari IS loaded in the process (its application-wide event
      filters break the nested loop and kill file-list clicks — see
      [[qfiledialog-napari-gotcha]]). Downside: on macOS NSOpenPanel's
      process-wide "last visited" cache overrides the specified
      directoryURL for the async path, so the picker may open at
      whichever folder was used most recently anywhere in the app
      rather than at ``initial``.

    - ``sync=None`` (default) auto-selects: sync when napari isn't
      loaded, async when it is. Detection uses ``sys.modules`` since
      napari's Qt event filters are installed as a side effect of the
      module import. Callers that need to force a mode (e.g. main-window
      Browse buttons that are always sync-safe) can still pass True/False
      explicitly.

    Parameters:
      - ``folder=True`` — directory picker.
      - ``save=True`` — save dialog. Mutually exclusive with folder.
      - ``name_filter`` — Qt filter string for file pickers, ignored
        when ``folder=True``.
      - ``last_dir_key`` — QSettings sub-key used to remember THIS
        picker's last visited directory independently of the other
        pickers. Picked paths get persisted back under this key.

    Initial-directory precedence:
      1. The explicit ``initial`` arg if it points to an existing path
         (typically the value already in the associated widget).
      2. Per-picker QSettings memory under ``last_dir_key``.
      3. "/" as a last-resort fallback (Finder's Computer view).
    """
    resolved_initial = _picker_initial_path(initial)
    # When `initial` was empty / non-existent (resolved to "/"), prefer
    # the per-picker memory over the OS-wide cache.
    if resolved_initial == "/" and last_dir_key:
        saved = _get_picker_last_dir(last_dir_key)
        if saved:
            resolved_initial = _picker_initial_path(saved)

    # Auto-select sync/async by whether napari has been loaded — the
    # only condition that makes sync mode unsafe (napari's app-wide Qt
    # event filters kill the file list inside the nested loop). Static
    # helpers honor the initial directory + per-widget last-dir memory
    # on macOS; open() doesn't. See [[qfiledialog-napari-gotcha]].
    if sync is None:
        import sys as _sys

        sync = "napari" not in _sys.modules

    def _finalize(path: str) -> None:
        # Persist this picker's last-visited directory so the next click
        # on the same Browse button opens here, regardless of what other
        # pickers have been used in between.
        if path and last_dir_key:
            saved_path = path if folder else str(Path(path).parent)
            _save_picker_last_dir(last_dir_key, saved_path)
        on_picked(path)

    if sync:
        # Native macOS picker via the static helpers. parent=None avoids
        # Qt's modal-event grab from blocking file-list clicks on macOS
        # (see _pick_folder_native docstring). Static helpers run their
        # own blocking nested loop, which macOS NSOpenPanel handles
        # correctly and where directoryURL is honored — so per-picker
        # last-dir memory actually takes effect visually.
        if folder:
            path = QFileDialog.getExistingDirectory(None, caption, resolved_initial)
        elif save:
            path, _ = QFileDialog.getSaveFileName(None, caption, resolved_initial, name_filter)
        else:
            path, _ = QFileDialog.getOpenFileName(None, caption, resolved_initial, name_filter)
        _finalize(path)
        return

    dlg = QFileDialog(None, caption, resolved_initial, name_filter if not folder else "")
    if folder:
        dlg.setFileMode(QFileDialog.Directory)
        dlg.setOption(QFileDialog.ShowDirsOnly, True)
    else:
        if save:
            dlg.setFileMode(QFileDialog.AnyFile)
            dlg.setAcceptMode(QFileDialog.AcceptSave)
        else:
            dlg.setFileMode(QFileDialog.ExistingFile)
    dlg.setWindowModality(Qt.ApplicationModal)
    if resolved_initial and resolved_initial != "/":
        dlg.setDirectory(resolved_initial)
        dlg.selectFile(resolved_initial)

    dlg.fileSelected.connect(_finalize)
    # Stash the reference on the calling instance so the dialog lives
    # past this function return. The holder owns a single slot —
    # replaced on each new picker; safe because pickers are
    # application-modal so only one can be visible at a time.
    holder._active_picker = dlg
    dlg.open()


def _build_landmark_name_pattern():
    """Compile the regex used to swap raw landmark keys for anatomical names.

    Built lazily and cached on the function so the (cheap) compile happens
    at most once. Uses ALL_LANDMARK_KEY_DISPLAY_NAMES so both GeoJSON-style
    keys (``DTip``, ``L2-L3``, …) and gate-config snake_case keys
    (``dtip``, ``l4_l5``, …) get translated — the latter surface in
    ``LowConfidenceLandmarkError`` messages when a confidence gate trips.

    Sorted longest-first so multi-token keys ("L2-L3", "subcostal break",
    "alula_notch") match before any shorter substring that could fight
    them, and so future additions remain collision-safe.

    The capture group around the optional quote handles logging's %r
    formatter, which wraps string args in single quotes (or double quotes
    if the value itself contains a single quote). The backreference \\1
    forces a matched pair, so we don't strip a leading quote without a
    trailing one. Unquoted keys (e.g. the comma-separated list in
    LowConfidenceLandmarkError) match too because the quote group is
    optional.
    """
    import re

    from measurement_maker import ALL_LANDMARK_KEY_DISPLAY_NAMES

    if _build_landmark_name_pattern.cached is None:
        keys_alt = "|".join(re.escape(k) for k in sorted(ALL_LANDMARK_KEY_DISPLAY_NAMES, key=len, reverse=True))
        _build_landmark_name_pattern.cached = (
            re.compile(r"(['\"]?)(" + keys_alt + r")\1"),
            ALL_LANDMARK_KEY_DISPLAY_NAMES,
        )
    return _build_landmark_name_pattern.cached


_build_landmark_name_pattern.cached = None


def _translate_landmark_names(text: str) -> str:
    """Swap raw GeoJSON landmark keys in a log line for their anatomical names.

    Used by _SignalLogHandler.emit so messages like ``Landmark 'DTip'
    snapped to node 42`` surface in the TRACE GUI log as ``Landmark 'L3
    distal end' snapped to node 42``. identifyFeatures' own log output
    (CLI mode, no TRACE handler attached) is unaffected — only the GUI
    handler runs this transform.

    Identity-mapped keys ("alula notch", "subcostal break") still match
    and substitute, which is a no-op; that's intentional so the regex
    stays symmetric and the mapping stays authoritative.
    """
    pattern, names = _build_landmark_name_pattern()
    return pattern.sub(lambda m: f"{m.group(1)}{names[m.group(2)]}{m.group(1)}", text)


# Matches the LowConfidenceLandmarkError message prefix raised by
# LandmarkLocator/landmark_locator/inference/predict.py:27-34. Kept as
# a module-level constant so the dependency is named and one search away
# from the originating file. If that exception's message text ever
# changes upstream, update this pattern in lockstep — there's a
# back-pointer comment in predict.py.
_GATE_FAILURE_PATTERN = re.compile(r"Core landmarks failed confidence gate", re.IGNORECASE)
# Quality gates from identifyFeatures' garbage_detector (solidity / fragmentation /
# vein_association / vein_presence). TRACE/pipeline.py prefixes GarbageRejection
# error text with "Aborted by quality gate (<filter>):" — we match on that here.
_QUALITY_GATE_PATTERN = re.compile(r"Aborted by quality gate", re.IGNORECASE)


def _classify_preproc_failure(error_text: str) -> str:
    """Map a Stage-1 error message to a coarse failure category.

    Returns one of:
      "gate"          — landmark confidence gate abort
                        (LowConfidenceLandmarkError).
      "preproc_other" — any other Stage-1 failure (model load, file IO,
                        wing-isolation no-wing-found, etc).

    Stage-2 (analysis) failures go through _classify_analysis_failure.

    Parsing the message string is a pragmatic choice: the exception
    type doesn't survive the QThread boundary (only the stringified
    message is signalled). If a future refactor carries an exception
    class through the signal, swap this for an isinstance() check
    and drop the regex.
    """
    if error_text and _GATE_FAILURE_PATTERN.search(error_text):
        return "gate"
    return "preproc_other"


def _classify_analysis_failure(error_text: str) -> str:
    """Map a Stage-2 error message to a coarse failure category.

    Returns one of:
      "gate"     — garbage-detector quality gate abort (GarbageRejection,
                   prefixed in TRACE/pipeline.py). Lumped with landmark
                   confidence-gate aborts so the "Rerun failed (no quality
                   gates)" button picks them all up.
      "analysis" — any other Stage-2 failure (vein tracer crash, output
                   write error, etc).
    """
    if error_text and _QUALITY_GATE_PATTERN.search(error_text):
        return "gate"
    return "analysis"


def _format_compact_value(v) -> str:
    """Render a single value for the compact run-settings preamble."""
    if v is None:
        return "null"
    if isinstance(v, bool):
        return "True" if v else "False"
    if isinstance(v, (int, float)):
        return str(v)
    if isinstance(v, str):
        return v if v else "''"
    if isinstance(v, list):
        return "[" + ", ".join(_format_compact_value(x) for x in v) + "]"
    if isinstance(v, dict):
        return "{" + ", ".join(f"{k}: {_format_compact_value(val)}" for k, val in v.items()) + "}"
    return str(v)


def _emit_compact_section(d: dict, lines: list, wrap_width: int, *, indent: str, parent_path: str = "") -> None:
    """Emit one dict's contents as packed key=value lines + nested [section] blocks.

    Short scalars (bool, number, null, string ≤40 chars) pack onto wrapped
    lines at ``wrap_width``. Long strings and lists each get their own
    ``key = value`` line, with ``=`` column-aligned across the group.
    Nested dicts get their own ``[parent.child]`` section header, indented
    at the same depth as their parent's body (the dotted name carries the
    nesting; visual indent stays flat).
    """
    packable: list[tuple[str, str]] = []
    long_pairs: list[tuple[str, str]] = []
    nested: list[tuple[str, dict]] = []
    for k, v in d.items():
        if isinstance(v, dict):
            nested.append((k, v))
            continue
        rendered = _format_compact_value(v)
        # Anything long enough to crowd a packed line gets its own row.
        if (isinstance(v, str) and len(rendered) > 40) or isinstance(v, list):
            long_pairs.append((k, rendered))
        else:
            packable.append((k, rendered))

    if packable:
        cur = indent
        for k, rendered in packable:
            tok = f"{k}={rendered}"
            if cur != indent and len(cur) + 2 + len(tok) > wrap_width:
                lines.append(cur)
                cur = indent + tok
            elif cur == indent:
                cur = cur + tok
            else:
                cur = cur + "  " + tok
        lines.append(cur)

    if long_pairs:
        max_key = max(len(k) for k, _ in long_pairs)
        for k, rendered in long_pairs:
            lines.append(f"{indent}{k.ljust(max_key)} = {rendered}")

    for k, sub in nested:
        path = f"{parent_path}.{k}" if parent_path else k
        lines.append("")
        lines.append(f"[{path}]")
        _emit_compact_section(sub, lines, wrap_width, indent="  ", parent_path=path)


def _format_run_settings_compact(data: dict, wrap_width: int = 110) -> str:
    """Render the run-settings preamble dict in a compact, LLM-parseable form.

    Tops the log with packed ``key=value`` lines for scalars / long-path or
    list assignments; ``settings.pipeline_config``, ``settings.gui_state``,
    etc. become ``[name]``-headed sections. Designed so an LLM can pick up
    individual values without learning a custom grammar, while staying
    short enough that a human can verify "did I run with X enabled?" at a
    glance. The canonical machine-readable copy lives in settings.yaml.
    """
    lines: list[str] = []
    settings = data.get("settings") or {}
    top = {k: v for k, v in data.items() if k != "settings"}
    _emit_compact_section(top, lines, wrap_width, indent="")
    for k, v in settings.items():
        lines.append("")
        lines.append(f"[{k}]")
        if isinstance(v, dict):
            _emit_compact_section(v, lines, wrap_width, indent="  ", parent_path=k)
        else:
            lines.append(f"  {_format_compact_value(v)}")
    return "\n".join(lines)


class _SignalLogHandler(logging.Handler):
    """Logging handler that forwards records through a pyqtSignal."""

    def __init__(self, signal):
        super().__init__()
        self._signal = signal
        self.setFormatter(logging.Formatter("%(name)s %(levelname)s: %(message)s"))

    def emit(self, record):
        try:
            from preprocessing.pipeline import current_image

            text = self.format(record)
            # Translation runs on logger records only; direct self._log()
            # calls from _on_progress carry image filenames, not landmark
            # keys, and don't need this hook. If a future caller starts
            # routing landmark names through _log, run them through
            # _translate_landmark_names() at that call site too.
            text = _translate_landmark_names(text)
            img = current_image.get()
            if img:
                text = f"[{img}] {text}"
            self._signal.emit(text)
        except Exception:
            pass


class TraceWorker(QThread):
    """Runs trace_folder() in a background thread.

    Two distinct stop semantics:
      - cancel()  — hard abort: raises InterruptedError out of trace_folder,
                    emits all_done([]).
      - pause()   — clean stop between images: trace_folder returns with the
                    slice's partial results, paused(list) is emitted instead
                    of all_done. The host (TraceWindow) flips the Pause
                    button to Resume and re-launches a worker continuing
                    from where this one stopped.
    """

    progress = pyqtSignal(int, int, str, str, str)  # idx, total, name, stage, detail
    log_message = pyqtSignal(str)  # forwarded log records from captured loggers
    all_done = pyqtSignal(list)  # results — natural completion
    paused = pyqtSignal(list)  # results — pause-then-stop completion
    cancelled = pyqtSignal(list)  # results — hard-cancel completion (run discarded)
    image_completed = pyqtSignal(str)  # basename of an image whose Stage 2 succeeded
    image_failed_preproc = pyqtSignal(str, str)  # basename, error message (Stage 1)
    image_failed_analysis = pyqtSignal(str, str)  # basename, error message (Stage 2)
    error = pyqtSignal(str)

    def __init__(self, kwargs):
        super().__init__()
        self.kwargs = kwargs
        self._cancel = False
        # Threaded pause: trace_folder polls this between images. Owned by
        # the worker so the host can call pause()/is_paused() without
        # worrying about thread safety (Event is itself thread-safe).
        import threading as _threading

        self._pause = _threading.Event()

    def cancel(self):
        self._cancel = True

    def pause(self):
        self._pause.set()

    def is_paused(self) -> bool:
        return self._pause.is_set()

    def run(self):
        handler = _SignalLogHandler(self.log_message)
        handler.setLevel(logging.INFO)
        attached = []
        for name in _CAPTURED_LOGGERS:
            lg = logging.getLogger(name)
            if lg.level == logging.NOTSET or lg.level > logging.INFO:
                lg.setLevel(logging.INFO)
            lg.addHandler(handler)
            attached.append(lg)
        try:

            def _progress(idx, total, name, stage, detail):
                if self._cancel:
                    raise InterruptedError("Cancelled by user")
                self.progress.emit(idx, total, name, stage, detail)

            self.kwargs["progress_callback"] = _progress
            self.kwargs["pause_event"] = self._pause
            # on_image_complete is invoked inside trace_folder under a lock,
            # so emitting a Qt signal from here is safe regardless of how
            # many Stage 2 workers are running.
            self.kwargs["on_image_complete"] = lambda basename: self.image_completed.emit(basename)
            self.kwargs["on_image_failed_preproc"] = lambda basename, err: self.image_failed_preproc.emit(basename, err)
            self.kwargs["on_image_failed_analysis"] = lambda basename, err: self.image_failed_analysis.emit(
                basename, err
            )
            results = trace_folder(**self.kwargs)
            if self._pause.is_set():
                self.paused.emit(results)
            else:
                self.all_done.emit(results)
        except InterruptedError:
            # InterruptedError is the cancel path (TraceWorker.cancel sets
            # self._cancel; the progress callback raises). Emit the
            # cancelled signal so the host can distinguish hard-cancel
            # (discard run) from natural-empty-result completion.
            if self._cancel:
                self.cancelled.emit([])
            else:
                self.all_done.emit([])
        except Exception as e:
            self.error.emit(f"{e}\n{traceback.format_exc()}")
        finally:
            for lg in attached:
                lg.removeHandler(handler)


# ---------------------------------------------------------------------------
# Per-image status display (Main tab image list)
# ---------------------------------------------------------------------------


class ImageStatus(Enum):
    """Per-image state surfaced as a glyph + color in the Main tab list."""

    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    SKIPPED = "skipped"  # resume: image was already done in a prior run slice
    USER_SKIPPED = "user_skipped"  # user unchecked this row before the run


# Leading glyph rendered before each row's filename. PENDING keeps no
# prefix so the resting state of the list looks identical to the
# pre-status-indicator behavior.
#
# USER_SKIPPED uses ⊘ (U+2298 CIRCLED DIVISION SLASH) — distinct from
# resume SKIPPED's ↷ so the log/visual review can tell "user opted
# out" from "previously completed". Different cause, different
# reversibility (user can re-tick; resume can't).
_STATUS_GLYPH: dict[ImageStatus, str] = {
    ImageStatus.PENDING: "",
    ImageStatus.IN_PROGRESS: "→ ",
    ImageStatus.SUCCEEDED: "✓ ",
    ImageStatus.FAILED: "✗ ",
    ImageStatus.SKIPPED: "↷ ",
    ImageStatus.USER_SKIPPED: "⊘ ",
}


# Row foreground colors resolved from the active Theme. PENDING +
# IN_PROGRESS pick palette tones that read on both dark and light
# themes; SUCCEEDED/FAILED reuse the existing green/red palette already
# used by the progress bar fills. USER_SKIPPED is a warmer gray than
# the resume SKIPPED so the two are visually distinguishable on the
# same screen. Implemented as a function (not a module-level dict) so
# the colors track the live theme — switching from dark to light at
# runtime re-resolves each row's foreground on the next
# _update_image_status call without needing to rebuild the dict.
def _status_color(status: "ImageStatus") -> QColor:
    from TRACE.theme import current_theme

    t = current_theme()
    return {
        ImageStatus.PENDING: QColor(t.text),
        ImageStatus.IN_PROGRESS: QColor(t.accent),
        ImageStatus.SUCCEEDED: QColor(t.success),
        ImageStatus.FAILED: QColor(t.error),
        ImageStatus.SKIPPED: QColor(t.skip_gray),
        ImageStatus.USER_SKIPPED: QColor(t.user_skip),
    }[status]


# ---------------------------------------------------------------------------
# Main window
# ---------------------------------------------------------------------------


class TraceWindow(QMainWindow):
    _PARALLEL_WORKERS_DETAILS_TEXT = (
        "Each worker runs the full pipeline on one wing "
        "and uses significant CPU and RAM. Setting this value too high can:\n\n"
        "  • Exhaust system RAM and cause the process to be killed\n"
        "  • Saturate all CPU cores and freeze other applications\n"
        "  • Trigger thermal throttling on laptops\n"
        "  • Produce no speedup past the number of physical cores\n\n"
        "A safe starting point is 2–4. Do not exceed your machine's "
        "physical core count unless you know what you're doing."
    )
    _PARALLEL_WORKERS_WARNING_TEXT = "You are enabling parallel Stage 2 analysis.\n\n" + _PARALLEL_WORKERS_DETAILS_TEXT

    def __init__(self):
        super().__init__()
        self.setWindowTitle("TRACE — Wing Analysis Pipeline")
        self.settings = QSettings("TRACE", "WingAnalysisPipeline")
        # Pause/resume state: manifest tracks completed images + status.
        # Populated either by _run_pipeline (fresh run) or by
        # _maybe_offer_resume (resume from a prior session). None when no
        # run is active.
        self._manifest: Optional[RunManifest] = None
        self._resume_skip_set: set[str] = set()
        # Per-run folder under <output_dir>/. Holds the manifest, the
        # settings YAML(s), and the run.log. Created at run start so the
        # pause/resume bookkeeping always has a stable place to write.
        self._run_folder: Optional[Path] = None
        # True when the current launch is a resume of an existing run (vs.
        # a fresh start). Drives log-clearing + "Resuming run." log line.
        self._is_resuming: bool = False
        # Pulse-and-hint registry for the Measurements-CSV child checkboxes.
        # Populated by _wrap_csv_child in _build_ui; consumed by the app-level
        # eventFilter installed at the end of __init__. Parallel to (and
        # independent of) InlineGeneralPanel._pulse_dependencies — both
        # filters fire on every MouseButtonPress and iterate their own dicts.
        self._pulse_dependencies: dict[QCheckBox, tuple[QCheckBox, Optional[QLabel]]] = {}
        # Hints we own, so the CSV parent's toggled handler can hide them all
        # in one pass when the user re-enables Measurements CSV.
        self._csv_dependency_hints: list[QLabel] = []
        self.resize(1050, 750)
        # Restore the window geometry the user last left (saved in closeEvent);
        # the resize() above is the first-launch default.
        _saved_geometry = self.settings.value("main_window_geometry")
        if _saved_geometry is not None:
            self.restoreGeometry(_saved_geometry)
        self.worker = None
        self._image_paths = []
        # Main-tab image-list status tracking. _basename_to_row keeps an
        # O(1) lookup; _row_base_labels stores the prefix-less filename
        # per row so re-applying a glyph doesn't compound on top of an
        # earlier one. Rebuilt on every _refresh_image_list().
        self._image_status: dict[str, ImageStatus] = {}
        self._basename_to_row: dict[str, int] = {}
        self._row_base_labels: list[str] = []
        # Per-basename failure message — set when _on_image_failed_preproc
        # or _on_image_failed_analysis fires. _update_image_status reads
        # this and appends a short one-liner after the filename when the
        # row is in the FAILED state.
        self._image_error_text: dict[str, str] = {}
        # Failed-images bookkeeping for the rerun buttons. _failure_category
        # holds basename → "gate"|"analysis"|"preproc_other" so
        # _refresh_rerun_buttons can decide whether to show the no-gate
        # variant (visible only when ≥1 category=="gate" failure exists).
        # _last_run_failed_set is the union of manifest preproc failures and
        # in-memory Stage-2 (analysis) failures from the most recent run.
        self._failure_category: dict[str, str] = {}
        self._last_run_failed_set: set[str] = set()
        # One-shot pending state set by _start_rerun_failed and consumed
        # by the very next _run_pipeline call. Kept on self so we don't
        # have to refactor _run_pipeline's signature (it has none — all
        # state is on self). Each field is reset back to a falsy default
        # at the top of _run_pipeline so a subsequent normal Run click
        # doesn't accidentally inherit the rerun's overrides.
        self._pending_csv_filename_override: Optional[str] = None
        self._pending_rerun_skip_set: Optional[set[str]] = None
        self._pending_skip_workers_warning: bool = False
        # Set by _start_rerun_failed(disable_gates=True). _run_pipeline
        # splices these into the worker kwargs ONLY for this one launch
        # without mutating self.config or self._gate_override — those are
        # what _save_settings persists to QSettings, so any in-place flip
        # would survive across TRACE restarts. Both reset on consumption.
        self._pending_gate_override: Optional[dict] = None
        self._pending_disable_garbage_filters: bool = False
        # User-driven skip: basenames the user has explicitly unchecked
        # in the Main-tab image list. Held separately from
        # _resume_skip_set (which is derived from the run manifest and
        # rebuilt on every input-folder change) — the two lifecycles
        # are different. Both are merged into a single
        # skip_image_basenames arg at worker launch time so the pipeline
        # contract stays unchanged.
        # Known limitation: keyed by basename, so in recursive scans
        # marking foo.tif in one subdir skips foo.tif everywhere.
        # Mirrors the existing _basename_to_row caveat.
        self._user_skip_set: set[str] = set()
        # Re-entrancy guard around QListWidgetItem.setCheckState — we
        # programmatically toggle states during _refresh_image_list /
        # _apply_skip / _disable_skip_checkboxes, and each setCheckState
        # fires itemChanged. The slot would otherwise recurse-mutate the
        # user-skip set while we're rebuilding it.
        self._suppress_check_signal: bool = False
        # Holds the currently-open async file picker (see
        # _open_native_picker_async) so Python's GC doesn't free it
        # between open() and the user clicking Open / Cancel. Required
        # for the input / output / load-previous-run pickers to survive
        # napari's process-wide event filter intercepting nested-event-
        # loop pickers.
        self._active_picker = None
        self.config = PipelineConfig()
        self._show_vein_tissue = False
        self._show_color_key = True
        self._show_ectopic_labels = True
        self._show_region_labels = True
        self._show_landmark_labels = True
        # NB: __init__ + reset-defaults keep in sync — both instances updated
        # by this replace_all; see also the QSettings/gui_state persistence
        # sites just below and the pipeline pass-through in _run_pipeline.
        self._vein_simplify_tolerance_px = 0.0
        self._ectopic_label_font_scale = 1.0
        self._landmark_size_scale = 1.0
        self._show_compartment_labels = True
        self._include_unreliable_landmarks = False
        self._do_rotation = False
        self._rotation_mirror_correct = False
        self._gate_override: dict | None = None
        self._wing_expand_fraction = 0.05
        self._wing_isolation_enabled = False
        self._wing_isolation_model_path = _default_model_path("wing_isolation")
        # Model paths (configured via Settings → Models). Plain strings rather
        # than QLineEdit widgets — the dialog owns the editing UI. Initialized
        # to the bundled TRACE/models/* folders so a first-time launch has
        # working defaults; overridden by saved QSettings values when present.
        self._landmark_model_path = _default_model_path("landmark")
        self._segmentation_model_path = _default_model_path("segmentation")
        # Stage 1 (resolutionAdjust) — per-model training-µm/px targets, the
        # active-model selector (default: wing features = segmentation), and the
        # ratio tolerance band. None means "not configured" for any target.
        # Wing features (segmentation) defaults to 0.483 µm/px — the resolution
        # the bundled segmentation model was trained at.
        self._landmark_target_um_per_px: Optional[float] = None
        self._segmentation_target_um_per_px: Optional[float] = 0.483
        self._wing_isolation_target_um_per_px: Optional[float] = None
        self._active_rescale_target: str = "segmentation"
        self._rescale_tolerance_low: float = 0.85
        self._rescale_tolerance_high: float = 1.15
        # Intermediate outputs (toggled in Settings → General → Intermediate outputs).
        # Default-off so a fresh batch only writes the final overlays + CSV; users
        # opt in to intermediates per-key in the Settings dialog.
        self._intermediate_outputs: dict[str, bool] = {key: False for key in INTERMEDIATE_OUTPUTS}
        # User-defined landmark distance pairs (TRACE-only post-CSV augmentation).
        # Configured via Settings → Custom Distances.
        self._user_landmark_distances: list[dict] = []
        # Last-used sample image + landmarks GeoJSON for the picker, so users
        # don't have to re-browse every session.
        self._distance_sample_image: str = ""
        self._distance_sample_landmarks: str = ""
        self._workers_warning_shown = False
        # Highest percentage shown on the progress bar this run — parallel
        # workers emit events out of order so we hold the bar at its peak
        # (never rewinds). Reset to 0 at the start of every run.
        self._progress_pct_high = 0
        # Stage1/Stage2 wall-time weight split on the unified progress bar.
        # Recomputed from the current output selection at run start so the
        # bar reflects "fraction of wall time done" rather than raw image count.
        self._progress_stage1_share = 1.0
        self._progress_stage2_share = 0.0
        # Wall-clock + smoothed ETA tracking.
        self._run_start_time: Optional[float] = None
        self._eta_smoothed_seconds: Optional[float] = None
        # Per-stage throughput tracking for the hybrid ETA. Reset on every
        # stage transition (preprocessing → analysis).
        self._current_stage: Optional[str] = None
        self._stage_first_event_time: Optional[float] = None
        self._stage_completions: int = 0
        self._stage_total: int = 0
        # Locked-in (not recomputed every tick) timestamps + average so the
        # smoothed bar can interpolate between "done" events without
        # collapsing to zero on every refresh.
        self._last_completion_time: Optional[float] = None
        self._avg_time_per_image: Optional[float] = None
        # 250ms QTimer that refreshes the bar + ETA between completion events.
        # Without this, parallel-worker batches arrive in clusters and the
        # bar sits frozen for tens of seconds between bursts of "done" events.
        self._progress_timer = QTimer(self)
        self._progress_timer.setInterval(250)
        self._progress_timer.timeout.connect(self._refresh_progress)
        # Active walkthrough overlay (None when no walkthrough is showing).
        # Tracked so resizeEvent / splitterMoved can forward to it.
        self._walkthrough: Optional[WalkthroughOverlay] = None
        self._build_ui()
        self._restore_settings()
        # App-level event filter so clicks on the disabled CSV-child checkboxes
        # trigger pulse + hint. Mirrors InlineGeneralPanel's approach — both
        # filters coexist and iterate their own dicts.
        _app = QApplication.instance()
        if _app is not None:
            _app.installEventFilter(self)
        # First-launch auto-show. Deferred until after the event loop starts
        # so the window has been shown and every widget has a valid geometry.
        if not self.settings.value("walkthrough_completed", False, type=bool):
            QTimer.singleShot(0, self._show_walkthrough)
        # Re-apply the update-available badge from QSettings if the prior
        # session saw an update and the user hasn't upgraded yet. Runs
        # synchronously here so the dot is on the Help tab from the very
        # first paint — no network round-trip required.
        self._restore_update_badge_from_cache()
        # Auto-check for updates. Deferred so the window is shown first.
        # Throttled to once per hour via QSettings so a user relaunching
        # rapidly doesn't burn GitHub's anonymous rate budget. The Help
        # panel exposes an opt-out checkbox backed by the same setting.
        if self.settings.value("auto_update_check_enabled", True, type=bool):
            QTimer.singleShot(0, self._maybe_auto_check_updates)
        # Post-run restore prompt. Deferred to QTimer so the window is
        # visible before the dialog appears — otherwise the prompt
        # races the first paint and shows up against a blank background.
        QTimer.singleShot(0, self._maybe_offer_restore_post_run_state)

        # Theme live-switch wiring. _apply_theme_styles re-runs every
        # inline stylesheet that depends on theme tokens (eta_label,
        # transient_status_label, etc.) and invalidates cached resources
        # like the update-badge icon. Called once now so widgets get
        # their initial styling, and again on every Settings → Theme
        # change via the manager's themeChanged signal.
        from TRACE.theme import manager as _theme_manager

        self._apply_theme_styles()
        _theme_manager().themeChanged.connect(self._apply_theme_styles)

    def _apply_theme_styles(self, *_args) -> None:
        """Re-apply every inline stylesheet that depends on theme tokens.

        Called at end of __init__ and on every ThemeManager.themeChanged.
        Order: small inline styles first, then invalidate cached
        resources, finally repaint any visible image-list rows so the
        per-row foreground colors pick up the new theme.
        """
        from TRACE.theme import current_theme

        t = current_theme()
        self.eta_label.setStyleSheet(f"color: {t.text_placeholder};")
        self.transient_status_label.setStyleSheet(f"color: {t.warning};")
        # Re-color the "requires Measurements CSV" hint labels stashed
        # in _pulse_dependencies. They're hidden most of the time but
        # become visible during the pulse animation; without this they'd
        # show with the old theme's link color after a live switch.
        for _parent, hint in self._pulse_dependencies.values():
            if hint is not None:
                hint.setStyleSheet(f"color: {t.link};")
        # The update-available "●" badge pixmap is cached on first use;
        # invalidate so the next request rebuilds with the new accent.
        self._cached_update_badge_icon = None
        # Repaint image-list row foregrounds. _update_image_status pulls
        # _status_color (which already reads current_theme), so we just
        # re-trigger it for every row that has a current status.
        for basename, status in list(self._image_status.items()):
            self._update_image_status(basename, status)

    # -----------------------------------------------------------------------
    # App-level event filter for the CSV-child dependent-checkbox pulse
    # -----------------------------------------------------------------------
    def eventFilter(self, obj, event):  # noqa: N802 — Qt API
        if event.type() == QEvent.MouseButtonPress and self._pulse_dependencies:
            global_pos = event.globalPos()
            for child, (parent_chk, hint) in self._pulse_dependencies.items():
                if not child.isEnabled() and child.isVisible():
                    local = child.mapFromGlobal(global_pos)
                    if child.rect().contains(local):
                        _pulse_text(parent_chk)
                        if hint is not None:
                            hint.show()
                        break
        return super().eventFilter(obj, event)

    # -----------------------------------------------------------------------
    # UI construction
    # -----------------------------------------------------------------------
    def _build_ui(self):
        # No menu bar — the walkthrough is re-triggered from the Help tab's
        # "Replay walkthrough" button instead.
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QHBoxLayout(central)

        # --- Left panel ---
        left = QWidget()
        # A minimum width sized to fit the widest CSV-column row
        # ("A/P compartment areas" + the "requires Measurements CSV"
        # pulse hint + group-box padding + hierarchy indent). Without
        # this, the row's requested width isn't enforced by the splitter
        # and both the checkbox label and the pulse hint clip. NO
        # maximumWidth cap — an earlier fixed 380-px maximum silently
        # prevented the splitter divider from ever moving right, so
        # users reported it as "not adjustable".
        left.setMinimumWidth(460)
        left_layout = QVBoxLayout(left)
        left_layout.setContentsMargins(4, 4, 4, 4)

        # -- Folders --
        folder_group = QGroupBox("Input and Output Folders")
        fg = QVBoxLayout(folder_group)

        fg.addWidget(QLabel("Images to analyze:"))
        # Wrapped in self.input_row so the walkthrough can highlight the edit
        # field AND its Browse button together (otherwise the popup lands on
        # top of Browse when targeting just self.input_edit).
        self.input_row = QWidget()
        row = QHBoxLayout(self.input_row)
        row.setContentsMargins(0, 0, 0, 0)
        self.input_edit = QLineEdit()
        self.input_edit.setReadOnly(True)
        self.input_edit.setPlaceholderText("select folder")
        self.input_edit.setToolTip("Folder containing wing images to process. Click Browse... to select.")
        # Changing the input folder invalidates the most-recent run's
        # failed set (its basenames refer to the prior folder). Clear
        # the rerun state so the buttons hide; the user re-runs from
        # scratch in the new folder.
        self.input_edit.textChanged.connect(self._on_input_folder_changed)
        btn = QPushButton("Browse...")
        btn.setToolTip("Pick the folder containing wing images.")
        btn.clicked.connect(self._select_input)
        row.addWidget(self.input_edit, stretch=1)
        row.addWidget(btn)
        fg.addWidget(self.input_row)

        self.recursive_chk = QCheckBox("Include subfolders")
        self.recursive_chk.setToolTip("When checked, also discover images inside subdirectories of the input folder.")
        self.recursive_chk.toggled.connect(self._refresh_image_list)
        fg.addWidget(self.recursive_chk)

        fg.addWidget(QLabel("Save results to:"))
        self.output_row = QWidget()
        row = QHBoxLayout(self.output_row)
        row.setContentsMargins(0, 0, 0, 0)
        self.output_edit = QLineEdit()
        self.output_edit.setReadOnly(True)
        self.output_edit.setPlaceholderText("select folder")
        self.output_edit.setToolTip(
            "Folder where all outputs (overlays, CSV, GeoJSONs, intermediates) will be written."
        )
        btn = QPushButton("Browse...")
        btn.setToolTip("Pick the folder where outputs will be written.")
        btn.clicked.connect(self._select_output)
        row.addWidget(self.output_edit, stretch=1)
        row.addWidget(btn)
        fg.addWidget(self.output_row)

        left_layout.addWidget(folder_group)

        # Model paths live as plain instance state — configured in
        # Settings → Models (no widgets in the main window).

        # -- Scale (left-panel mirror of the inline General-tab spinbox) --
        # Kept on the left panel because µm/px is a per-batch setting users
        # change often; the mirror in the General tab is the authoritative
        # source for the value (both spinboxes write to self.config.um_per_px
        # via _set_scale, which keeps them in sync).
        scale_group = QGroupBox("Scale")
        # Two-row vertical layout: row 1 = label + spinbox, row 2 = the
        # per-image checkbox. The checkbox reads as an option that governs
        # the spinbox above rather than a peer widget on its right.
        scale_group_layout = QVBoxLayout(scale_group)
        scale_group_layout.setContentsMargins(9, 9, 9, 9)
        scale_group_layout.setSpacing(4)
        scale_row_widget = QWidget()
        sg = QHBoxLayout(scale_row_widget)
        sg.setContentsMargins(0, 0, 0, 0)
        sg.addWidget(QLabel("µm/px:"))
        self.scale_spin = _PlaceholderSpinBox()
        self.scale_spin.setDecimals(4)
        self.scale_spin.setRange(0.0001, 100.0)
        self.scale_spin.setSingleStep(0.001)
        self.scale_spin.setValue(self.config.um_per_px if self.config.um_per_px else self.scale_spin.minimum())
        self.scale_spin.set_placeholder("conversion factor")
        self.scale_spin.setToolTip(
            "Microns per pixel — used to convert every measurement to physical units (µm, µm²). "
            "Mirrored in the General tab on the right."
        )
        self.scale_spin.valueChanged.connect(lambda v: self._set_scale(v, source="left"))
        sg.addWidget(self.scale_spin, stretch=1)
        scale_group_layout.addWidget(scale_row_widget)
        self.auto_detect_um_per_px_chk = QCheckBox("Detect scale from image metadata")
        self.auto_detect_um_per_px_chk.setToolTip(
            "When checked, each image's µm/px is read from its OWN metadata (TIFF "
            "XResolution + ResolutionUnit / OME-XML PhysicalSizeX) — measurements "
            "convert through that image's real scale rather than a shared value. "
            "The scale field above becomes the fallback used only when an image "
            "has no parseable metadata. If checked AND the scale field is empty "
            "AND any image lacks metadata, Run raises a pre-flight error so you "
            "don't discover the missing scale mid-batch."
        )
        self.auto_detect_um_per_px_chk.setChecked(bool(getattr(self.config, "auto_detect_um_per_px", False)))
        self.auto_detect_um_per_px_chk.toggled.connect(self._on_auto_detect_um_per_px_toggled)
        scale_group_layout.addWidget(self.auto_detect_um_per_px_chk)
        left_layout.addWidget(scale_group)

        # Parallel processing lives in the right-panel General tab
        # (InlineGeneralPanel.workers_spin); no left-panel widget for it.

        # Pipeline settings entry points all live on the right-panel Settings
        # tab now: Restore Defaults, Advanced Settings…, wipe my memories.
        # Import/Save are inside the advanced dialog.

        # -- Output selection (final outputs only; intermediates live in Settings → General) --
        # Assigned to self so the first-launch walkthrough can target it.
        self.out_group = QGroupBox("Outputs")
        out_group = self.out_group
        ol = QVBoxLayout(out_group)
        self.output_checks: OrderedDict[str, QCheckBox] = OrderedDict()
        # Nested measurement-group checkboxes for the CSV output.
        self.csv_group_checks: OrderedDict[str, QCheckBox] = OrderedDict()
        self._csv_group_container: QWidget | None = None
        for key, label in OUTPUT_TYPES.items():
            if key in INTERMEDIATE_OUTPUTS:
                continue
            chk = QCheckBox(label)
            chk.setChecked(True)
            # output_tooltip_html returns <img> markup when a bundled example
            # exists for this key, or falls back to the text tooltip otherwise.
            tooltip = output_tooltip_html(key, OUTPUT_TOOLTIPS.get(key, ""))
            if tooltip:
                chk.setToolTip(tooltip)
            self.output_checks[key] = chk
            ol.addWidget(chk)
            # Nest the measurement-group checkboxes directly under "csv".
            if key == "csv":
                self._csv_group_container = QWidget()
                cgl = QVBoxLayout(self._csv_group_container)
                cgl.setContentsMargins(24, 0, 0, 0)  # left-indent for hierarchy
                cgl.setSpacing(2)
                for gkey, glabel in MEASUREMENT_GROUPS.items():
                    gchk = QCheckBox(glabel)
                    gchk.setChecked(True)
                    tip = MEASUREMENT_GROUP_TOOLTIPS.get(gkey)
                    if tip:
                        gchk.setToolTip(tip)
                    self.csv_group_checks[gkey] = gchk
                    cgl.addWidget(self._wrap_csv_child(gchk, chk))
                # Custom landmark distances also writes only into the batch CSV,
                # so it sits alongside the measurement-group sub-checkboxes and
                # tracks the parent CSV checkbox's enabled state.
                self.include_custom_measurements_chk = QCheckBox("Custom measurements")
                self.include_custom_measurements_chk.setChecked(True)
                self.include_custom_measurements_chk.setToolTip(
                    "Adds the pairs configured in the Custom Measurements tab to the batch CSV "
                    "as custom_<label>_px (and _um when scale is set) columns.\n\n"
                    "No effect when no pairs are configured."
                )
                self.btn_edit_custom_distances = QPushButton("Edit...")
                self.btn_edit_custom_distances.setToolTip(
                    "Jump to the Custom Measurements tab on the right to add/edit/remove " "landmark measurement pairs."
                )
                self.btn_edit_custom_distances.clicked.connect(
                    lambda: self.right_tabs.setCurrentWidget(self.inline_custom_distances_panel)
                )
                # The Custom measurements row uniquely also has an Edit... button
                # next to the checkbox. _wrap_csv_child can't handle that out of
                # the box, so build the row inline but still register the
                # checkbox in the pulse registry.
                cd_row_widget = _DependentRow(self.include_custom_measurements_chk, chk)
                cd_h = QHBoxLayout(cd_row_widget)
                cd_h.setContentsMargins(0, 0, 0, 0)
                cd_h.setSpacing(8)
                cd_h.addWidget(self.include_custom_measurements_chk)
                cd_h.addWidget(self.btn_edit_custom_distances)
                cd_hint = QLabel("requires Measurements CSV")
                # Hint label is only briefly visible (during the pulse
                # animation when the user clicks a disabled child). Reads
                # the theme at construction; live theme-switch while the
                # hint is mid-pulse would keep the old color until the
                # next pulse rebuilds it — acceptable.
                from TRACE.theme import current_theme as _ct

                cd_hint.setStyleSheet(f"color: {_ct().link};")
                # Reserve the hint's natural width in the layout even while
                # hidden — hidden widgets are normally excluded from sizing,
                # which caused the pulse-revealed hint to clip on narrow
                # panels. setRetainSizeWhenHidden + a min width based on
                # sizeHint keep the row wide enough from the outset.
                _sp = cd_hint.sizePolicy()
                _sp.setRetainSizeWhenHidden(True)
                cd_hint.setSizePolicy(_sp)
                cd_hint.setMinimumWidth(cd_hint.sizeHint().width())
                cd_hint.hide()
                cd_row_widget.set_hint(cd_hint)
                cd_h.addWidget(cd_hint)
                cd_h.addStretch(1)
                self._pulse_dependencies[self.include_custom_measurements_chk] = (chk, cd_hint)
                self._csv_dependency_hints.append(cd_hint)
                cgl.addWidget(cd_row_widget)
                ol.addWidget(self._csv_group_container)

                # Enable/disable nested checkboxes with parent CSV state.
                # Also hide any "requires Measurements CSV" hints that the
                # event filter may have surfaced — they'd otherwise linger
                # next to now-enabled children.
                def _on_csv_toggled(checked: bool) -> None:
                    self._csv_group_container.setEnabled(checked)
                    for _h in self._csv_dependency_hints:
                        _h.hide()

                chk.toggled.connect(_on_csv_toggled)
                self._csv_group_container.setEnabled(chk.isChecked())

        left_layout.addWidget(out_group)

        # -- Run / Pause / Resume (one button) + Cancel --
        # btn_run is the tri-state action button: it shows "Run Pipeline"
        # when idle, "Pause" while the worker is going, and "Resume" while
        # paused. Cancel is a separate hard-stop that discards the run.
        btn_layout = QHBoxLayout()
        self.btn_run = QPushButton("Run Pipeline")
        self.btn_run.setToolTip(
            "Run Pipeline — start processing every image in the input folder. "
            "While running, this button becomes Pause; once paused, Resume."
        )
        self.btn_run.clicked.connect(self._on_run_button_clicked)
        self.btn_cancel = QPushButton("Cancel")
        self.btn_cancel.setToolTip(
            "Stop the run and discard its progress. Per-image outputs already written stay on disk, but "
            "the run state is wiped so the next Run starts fresh (no Resume prompt)."
        )
        self.btn_cancel.clicked.connect(self._cancel_pipeline)
        self.btn_cancel.setEnabled(False)
        btn_layout.addWidget(self.btn_run)
        btn_layout.addWidget(self.btn_cancel)
        left_layout.addLayout(btn_layout)

        # Load-previous-run is always visible — it's the deliberate
        # fallback for "I dismissed the on-launch prompt" and for
        # picking older runs in a different output folder. Lives here
        # (below Run / Cancel) rather than in a menu bar because the
        # main window has no menu bar today, and grouping with the
        # other post-run actions matches the user's mental model.
        # Disabled while a run is in progress so the worker can't
        # race against a state restore.
        self.btn_load_previous = QPushButton("Load previous run…")
        self.btn_load_previous.setToolTip(
            "Reload the post-run state of a previous TRACE run so you can open the "
            "inspector on its failed images. Pick the output folder you used for that "
            "run; per-image overrides you saved before are still there."
        )
        self.btn_load_previous.clicked.connect(self._load_previous_run_dialog)
        left_layout.addWidget(self.btn_load_previous)

        # Rerun buttons appear after a run that left ≥1 failed image
        # (visibility recomputed in _refresh_rerun_buttons). The no-gate
        # variant only shows when at least one of those failures was a
        # landmark confidence-gate abort. Both stay hidden during a run
        # and on first launch.
        #
        # Stacked vertically rather than in a row because the combined
        # widths ("Rerun failed images", "Rerun failed (no quality gates)",
        # "Review failed images (N)") far exceed the left column at its
        # default width — Windows truncates the middle of each label
        # without showing ellipses, so the user sees garbled text like
        # "un failed (no quality g". Vertical stacking gives each button
        # its full label width regardless of the splitter position.
        rerun_layout = QVBoxLayout()
        rerun_layout.setContentsMargins(0, 0, 0, 0)
        rerun_layout.setSpacing(4)
        self.btn_rerun_failed = QPushButton("Rerun failed images")
        self.btn_rerun_failed.setToolTip(
            "Re-process only the images that failed in the last run, using current "
            "settings. You'll be asked whether to append the new rows to the "
            "existing measurements.csv or write a separate one."
        )
        self.btn_rerun_failed.clicked.connect(lambda: self._start_rerun_failed(disable_gates=False))
        self.btn_rerun_failed.setVisible(False)
        self.btn_rerun_failed_nogate = QPushButton("Rerun failed (no quality gates)")
        self.btn_rerun_failed_nogate.setToolTip(
            "Same as Rerun failed images, but with EVERY quality gate temporarily "
            "disabled — landmark confidence gates plus the garbage-detector filters "
            "(solidity, fragmentation, vein-association, vein-presence). Use when "
            "you suspect a gate is too strict. Your saved settings are not modified."
        )
        self.btn_rerun_failed_nogate.clicked.connect(lambda: self._start_rerun_failed(disable_gates=True))
        self.btn_rerun_failed_nogate.setVisible(False)
        self.btn_review_failed = QPushButton("Review failed images")
        self.btn_review_failed.setToolTip(
            "Open the landmark inspector on each failed image so you can correct "
            "the landmarks and save per-image overrides. The next run that includes "
            "these images will use your overrides instead of running LandmarkLocator."
        )
        self.btn_review_failed.clicked.connect(self._review_failed_images)
        self.btn_review_failed.setVisible(False)
        rerun_layout.addWidget(self.btn_rerun_failed)
        rerun_layout.addWidget(self.btn_rerun_failed_nogate)
        rerun_layout.addWidget(self.btn_review_failed)
        left_layout.addLayout(rerun_layout)

        # Tracks whether the worker is currently in a paused state. Drives
        # the button label (Pause ↔ Resume) and the next-Run-click decision
        # (continue current paused run vs. start fresh).
        self._is_paused = False
        # Set when the user clicks Cancel on a running worker. The worker
        # is left to exit cooperatively (no QThread.terminate — that
        # corrupts torch MPS internal state on Apple Silicon and segfaults
        # the next run). While this flag is set the Run/Cancel buttons
        # stay disabled, and whichever signal the worker emits on exit
        # (cancelled / all_done / error) is rerouted into _finalize_cancel.
        self._cancel_requested = False

        left_layout.addStretch()

        # --- Right panel: tab bar (Main / General / Custom Distances / Help)
        # with the progress bar + ETA always-visible below the tabs.
        right = QWidget()
        right_layout = QVBoxLayout(right)
        right_layout.setContentsMargins(4, 4, 4, 4)

        self.right_tabs = QTabWidget()
        right_layout.addWidget(self.right_tabs, stretch=1)

        # Tab 0 — Main (image list + log)
        main_tab = QWidget()
        main_tab_layout = QVBoxLayout(main_tab)
        main_tab_layout.setContentsMargins(4, 4, 4, 4)
        main_tab_layout.addWidget(QLabel("Images:"))
        self.image_list = QListWidget()
        # ExtendedSelection: click selects a single row, Shift-click
        # extends the selection to a contiguous range, Ctrl-click (Cmd
        # on macOS — Qt translates) toggles individual rows in/out.
        # The bulk Skip/Unskip context menu items operate on whatever
        # is selected here.
        self.image_list.setSelectionMode(QListWidget.ExtendedSelection)
        # Per-row check state drives _user_skip_set. itemChanged fires
        # whenever the user clicks a checkbox; _on_image_check_toggled
        # short-circuits when _suppress_check_signal is True (which we
        # set around programmatic refreshes). Context menu adds bulk
        # Skip/Unskip ops on the current selection.
        self.image_list.itemChanged.connect(self._on_image_check_toggled)
        self.image_list.setContextMenuPolicy(Qt.CustomContextMenu)
        self.image_list.customContextMenuRequested.connect(self._on_image_list_context_menu)
        main_tab_layout.addWidget(self.image_list, stretch=1)
        main_tab_layout.addWidget(QLabel("Log:"))
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        main_tab_layout.addWidget(self.log_text, stretch=1)
        self.right_tabs.addTab(main_tab, "Main")

        # Tab 1 — General (auto-apply controls; replaces dialog's General tab)
        self.inline_general_panel = InlineGeneralPanel(self)
        from TRACE.inline_panels import _wrap_scrollable as _wrap_scroll

        self.right_tabs.addTab(_wrap_scroll(self.inline_general_panel), "Settings")

        # Tab 2 — Custom Distances (LandmarkPickerWidget)
        self.inline_custom_distances_panel = InlineCustomDistancesPanel(self)
        self.right_tabs.addTab(self.inline_custom_distances_panel, "Custom Measurements")

        # Tab 3 — Help. Wrapped in _wrap_scroll so the panel scrolls
        # instead of clipping when the window is too narrow or short
        # to fit the heading-blurb-button rows at their natural size.
        # Stash the scroll wrapper as self._help_tab_widget because the
        # tab indexOf / widget lookups below need to compare against the
        # widget actually inserted into right_tabs (the QScrollArea),
        # not the inline panel inside it.
        self.inline_help_panel = InlineHelpPanel(self)
        self._help_tab_widget = _wrap_scroll(self.inline_help_panel)
        self.right_tabs.addTab(self._help_tab_widget, "Help")

        # Clear the update-available "●" badge whenever the user lands
        # on the Help tab. Connection lives here so the signal exists
        # regardless of whether an auto-check ever runs.
        self.right_tabs.currentChanged.connect(self._on_right_tab_changed)

        # Progress bar + ETA stay outside the tab widget so they're visible
        # regardless of which tab the user is viewing during a run.
        self.progress = QProgressBar()
        self.progress.setRange(0, 100)
        # Snapshot the default Highlight color so we can revert after a green
        # completion paint without forcing the bar into stylesheet-rendering
        # mode (which subtly shifts indentation, border, and text position).
        self._progress_default_highlight = self.progress.palette().color(QPalette.Highlight)
        right_layout.addWidget(self.progress)
        self.eta_label = QLabel("")
        # Styled in _apply_theme_styles so live theme switches update it.
        right_layout.addWidget(self.eta_label)
        # Transient one-line status under the ETA. Used to surface the
        # "Pause requested — finishing the current image first." and
        # "Cancelling — the current image will finish first." messages
        # so the user has visible confirmation that their click landed
        # while the worker finishes its in-flight image. Cleared once
        # the worker actually pauses / cancels / completes.
        self.transient_status_label = QLabel("")
        # Styled in _apply_theme_styles so live theme switches update it.
        self.transient_status_label.hide()
        right_layout.addWidget(self.transient_status_label)

        # --- Assemble ---
        # Assigned to self so the walkthrough can listen to splitterMoved and
        # reposition its highlight when the user drags the divider.
        self._splitter = QSplitter(Qt.Horizontal)
        self._splitter.addWidget(left)
        self._splitter.addWidget(right)
        self._splitter.setStretchFactor(1, 1)
        # The default QSplitter handle is nearly invisible on both macOS
        # (transparent 1-px seam between children) and Windows (~1 px wide)
        # — users can't see where to grab and report the left column as
        # "not adjustable". Widen the handle, style it as a subtle vertical
        # bar via a stylesheet so it's actually visible on macOS, and give
        # it an explicit split-cursor so the interaction is discoverable.
        self._splitter.setHandleWidth(8)
        self._splitter.setStyleSheet(
            "QSplitter::handle:horizontal {"
            "  background: palette(mid);"
            "  border-left: 1px solid palette(midlight);"
            "  border-right: 1px solid palette(midlight);"
            "}"
            "QSplitter::handle:horizontal:hover {"
            "  background: palette(highlight);"
            "}"
        )
        self._splitter.handle(1).setCursor(Qt.SplitHCursor)
        self._splitter.setChildrenCollapsible(False)
        # Start the left column wide enough to fit the widest CSV-row +
        # "requires Measurements CSV" pulse hint without clipping. The
        # hint sits in a horizontal row inside the Outputs frame and pops
        # in when the user clicks a disabled child checkbox; a narrower
        # default would clip both the checkbox label and the hint until
        # the user manually widened the panel.
        self._splitter.setSizes([460, 860])
        self._splitter.splitterMoved.connect(self._on_splitter_moved)
        main_layout.addWidget(self._splitter)

        self.statusBar().showMessage("Ready")

    def _wrap_csv_child(self, child_chk: QCheckBox, parent_chk: QCheckBox) -> _DependentRow:
        """Wrap a Measurements-CSV child checkbox in a pulse-and-hint row.

        Mirrors InlineGeneralPanel._wrap_with_hint but writes into TraceWindow's
        own _pulse_dependencies registry (consumed by self.eventFilter) and
        collects the hint label in _csv_dependency_hints so the parent-toggle
        handler can clear every hint in one pass.
        """
        row = _DependentRow(child_chk, parent_chk)
        h = QHBoxLayout(row)
        h.setContentsMargins(0, 0, 0, 0)
        h.setSpacing(8)
        h.addWidget(child_chk)
        hint = QLabel("requires Measurements CSV")
        # Short-lived: only visible during the pulse animation when the
        # user clicks the disabled child. Reads the theme at construction.
        from TRACE.theme import current_theme as _ct

        hint.setStyleSheet(f"color: {_ct().link};")
        # Reserve the hint's natural width in the layout even while hidden,
        # so the row's minimum width bakes the hint in from construction
        # onward. Hidden widgets are normally excluded from layout sizing,
        # which meant the row committed to a narrower minimum and the hint
        # got clipped by that fixed row width when the pulse revealed it.
        _sp = hint.sizePolicy()
        _sp.setRetainSizeWhenHidden(True)
        hint.setSizePolicy(_sp)
        hint.setMinimumWidth(hint.sizeHint().width())
        hint.hide()
        row.set_hint(hint)
        h.addWidget(hint)
        h.addStretch(1)
        self._pulse_dependencies[child_chk] = (parent_chk, hint)
        self._csv_dependency_hints.append(hint)
        return row

    # -----------------------------------------------------------------------
    # Folder / model selection
    # -----------------------------------------------------------------------
    def _select_input(self):
        _open_native_picker_async(
            self,
            "Select Input Folder",
            _picker_initial_path(self.input_edit.text()),
            self._on_input_folder_picked,
            folder=True,
            last_dir_key="input_folder",
            sync=True,
        )

    def _on_input_folder_picked(self, folder: str) -> None:
        if not folder:
            return
        self.input_edit.setText(folder)
        self._refresh_image_list()

    def _refresh_image_list(self):
        """Re-discover images using the current input-folder + recursive flag."""
        folder_text = self.input_edit.text()
        if not folder_text:
            self._image_paths = []
            self.image_list.clear()
            self._image_status.clear()
            self._basename_to_row.clear()
            self._row_base_labels.clear()
            self._image_error_text.clear()
            self._user_skip_set.clear()
            return
        folder = Path(folder_text)
        try:
            if not folder.is_dir():
                return
        except (PermissionError, OSError):
            return
        recursive = self.recursive_chk.isChecked()
        try:
            self._image_paths = discover_images(folder, recursive=recursive)
        except (PermissionError, OSError) as exc:
            # macOS TCC can deny iterdir() on protected locations (Desktop,
            # Documents, Downloads, iCloud) — don't let that abort startup.
            self._image_paths = []
            self.statusBar().showMessage(f"Cannot read folder: {exc}")
        # Pull this folder's saved user-skip set BEFORE the population
        # loop so each new row gets the correct initial checkState.
        self._restore_user_skip_set()
        # Programmatic setCheckState below would otherwise re-enter
        # _on_image_check_toggled and double-mutate _user_skip_set.
        self._suppress_check_signal = True
        try:
            self.image_list.clear()
            self._image_status.clear()
            self._basename_to_row.clear()
            self._row_base_labels.clear()
            self._image_error_text.clear()
            for idx, p in enumerate(self._image_paths):
                # Show path relative to the input folder when recursing so subfolder
                # context is visible; otherwise just the name.
                label = str(p.relative_to(folder)) if recursive else p.name
                self.image_list.addItem(label)
                self._row_base_labels.append(label)
                # Index by basename — that's what TraceWorker's image_* signals
                # carry. Known limitation: same-basename collisions across
                # different subdirs on a recursive scan land on the same row;
                # avoid by structuring inputs to keep basenames unique.
                self._basename_to_row[p.name] = idx
                # Checked = include in run; unchecked = skip. Reads more
                # naturally than the inverse (a ticked box is the thing
                # the user wants to process).
                item = self.image_list.item(idx)
                item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
                item.setCheckState(Qt.Unchecked if p.name in self._user_skip_set else Qt.Checked)
                # Paint the user-skip glyph for any pre-populated skips
                # so a freshly-restored folder shows ⊘ rows immediately.
                if p.name in self._user_skip_set:
                    self._update_image_status(p.name, ImageStatus.USER_SKIPPED)
        finally:
            self._suppress_check_signal = False
        self.statusBar().showMessage(f"Found {len(self._image_paths)} images")

    # -----------------------------------------------------------------------
    # User-driven skip (per-row checkbox + context menu)
    # -----------------------------------------------------------------------
    _TERMINAL_OR_LIVE_STATUSES = (
        ImageStatus.SUCCEEDED,
        ImageStatus.FAILED,
        ImageStatus.IN_PROGRESS,
    )

    def _on_image_check_toggled(self, item) -> None:
        """Single-row checkbox toggle → update _user_skip_set + glyph.

        Programmatic setCheckState calls during refresh / bulk-apply set
        ``self._suppress_check_signal`` so they don't recurse through
        here. The terminal-status guard keeps a user untick from
        downgrading a row that already finished a run.
        """
        if self._suppress_check_signal:
            return
        row = self.image_list.row(item)
        if not 0 <= row < len(self._image_paths):
            return
        basename = self._image_paths[row].name
        if item.checkState() == Qt.Unchecked:
            self._user_skip_set.add(basename)
            if self._image_status.get(basename) not in self._TERMINAL_OR_LIVE_STATUSES:
                self._update_image_status(basename, ImageStatus.USER_SKIPPED)
        else:
            self._user_skip_set.discard(basename)
            if self._image_status.get(basename) == ImageStatus.USER_SKIPPED:
                self._update_image_status(basename, ImageStatus.PENDING)
        self._persist_user_skip_set()

    def _on_image_list_context_menu(self, pos) -> None:
        """Right-click menu for bulk Skip / Unskip on the current selection.

        ``Skip selected`` / ``Unskip selected`` operate on currently
        selected rows (use Ctrl/Shift-click to multi-select first).
        ``Skip all`` / ``Unskip all`` cover the whole list.
        """
        from PyQt5.QtWidgets import QMenu

        menu = QMenu(self.image_list)
        act_skip = menu.addAction("Skip selected")
        act_unskip = menu.addAction("Unskip selected")
        menu.addSeparator()
        act_skip_all = menu.addAction("Skip all")
        act_unskip_all = menu.addAction("Unskip all")

        # Edit model predictions. Label reflects whether a multi-selection
        # (cohort) or a single clicked row will be opened.
        menu.addSeparator()
        n_selected = len(self.image_list.selectedItems())
        inspect_label = (
            "Edit model predictions…" if n_selected <= 1 else f"Edit model predictions ({n_selected} selected)…"
        )
        act_inspect = menu.addAction(inspect_label)

        chosen = menu.exec_(self.image_list.mapToGlobal(pos))
        if chosen is None:
            return
        if chosen is act_skip:
            self._apply_skip([self.image_list.row(i) for i in self.image_list.selectedItems()], skip=True)
        elif chosen is act_unskip:
            self._apply_skip([self.image_list.row(i) for i in self.image_list.selectedItems()], skip=False)
        elif chosen is act_skip_all:
            self._apply_skip(range(self.image_list.count()), skip=True)
        elif chosen is act_unskip_all:
            self._apply_skip(range(self.image_list.count()), skip=False)
        elif chosen is act_inspect:
            selected = self.image_list.selectedItems()
            target_item = self.image_list.itemAt(pos)
            # Cohort only if >=2 selected AND the clicked row is part of the
            # selection. Right-clicking outside the selection operates on the
            # clicked row only (matches typical OS behavior).
            if len(selected) > 1 and target_item in selected:
                paths = []
                for item in selected:
                    row = self.image_list.row(item)
                    if 0 <= row < len(self._image_paths):
                        paths.append(self._image_paths[row])
                if paths:
                    self._open_landmark_inspector(paths[0], cohort=paths)
            elif target_item is not None:
                row = self.image_list.row(target_item)
                if 0 <= row < len(self._image_paths):
                    self._open_landmark_inspector(self._image_paths[row])

    def _open_landmark_inspector(self, image_path: Path, cohort: Optional[list] = None) -> None:
        """Open the modal landmark inspector on one image (or a cohort)."""
        from TRACE.landmark_inspector_dialog import LandmarkInspectorDialog

        dlg = LandmarkInspectorDialog(self, image_path, cohort=cohort)
        dlg.exec_()

    def _apply_skip(self, rows, *, skip: bool) -> None:
        """Toggle multiple rows at once + persist + repaint glyphs."""
        self._suppress_check_signal = True
        try:
            for row in rows:
                item = self.image_list.item(row)
                if item is None or not 0 <= row < len(self._image_paths):
                    continue
                item.setCheckState(Qt.Unchecked if skip else Qt.Checked)
                basename = self._image_paths[row].name
                if skip:
                    self._user_skip_set.add(basename)
                    if self._image_status.get(basename) not in self._TERMINAL_OR_LIVE_STATUSES:
                        self._update_image_status(basename, ImageStatus.USER_SKIPPED)
                else:
                    self._user_skip_set.discard(basename)
                    if self._image_status.get(basename) == ImageStatus.USER_SKIPPED:
                        self._update_image_status(basename, ImageStatus.PENDING)
        finally:
            self._suppress_check_signal = False
        self._persist_user_skip_set()

    def _set_skip_checkboxes_enabled(self, enabled: bool) -> None:
        """Lock/unlock the per-row check toggles for the duration of a run.

        The worker captures the skip set at launch; mid-run toggles
        would be confusing because they'd silently no-op until the
        next run. Locking the checkboxes during a run makes the
        affordance honest.
        """
        self._suppress_check_signal = True
        try:
            for i in range(self.image_list.count()):
                item = self.image_list.item(i)
                if item is None:
                    continue
                flags = item.flags()
                if enabled:
                    item.setFlags(flags | Qt.ItemIsEnabled | Qt.ItemIsUserCheckable)
                else:
                    item.setFlags(flags & ~Qt.ItemIsEnabled)
        finally:
            self._suppress_check_signal = False

    def _user_skip_qsettings_key(self) -> Optional[str]:
        """QSettings key for the current input folder's user-skip set.

        Hashes the folder path so the key is bounded and registry/plist
        safe regardless of path length or characters. Returns None when
        no folder is selected — callers treat that as no-op.
        """
        folder = self.input_edit.text().strip()
        if not folder:
            return None
        import hashlib

        h = hashlib.sha1(folder.encode("utf-8")).hexdigest()[:16]
        return f"user_skip/{h}"

    def _persist_user_skip_set(self) -> None:
        """Write the current set to QSettings under the per-folder key."""
        key = self._user_skip_qsettings_key()
        if key is None:
            return
        import json

        self.settings.setValue(key, json.dumps(sorted(self._user_skip_set)))

    def _restore_user_skip_set(self) -> None:
        """Populate _user_skip_set from QSettings for the current folder.

        Called at the top of _refresh_image_list so the per-row
        checkState during the population loop already reflects the
        saved marks. Corruption / missing keys reset to empty.
        """
        self._user_skip_set.clear()
        key = self._user_skip_qsettings_key()
        if key is None:
            return
        raw = self.settings.value(key, "", type=str)
        if not raw:
            return
        import json

        try:
            names = json.loads(raw)
        except (json.JSONDecodeError, TypeError):
            return
        if isinstance(names, list):
            self._user_skip_set.update(str(n) for n in names)

    def _revert_in_progress_to_pending(self) -> None:
        """Roll back any rows still showing IN_PROGRESS to PENDING.

        Used by the pause/error/all-done handlers — if the worker
        exits while a Stage-2 image is mid-flight, the row was flipped
        to IN_PROGRESS by _on_progress but will never receive a
        succeeded/failed signal for that slice. Resetting it to PENDING
        gives an honest "not done yet" view.
        """
        for basename, status in list(self._image_status.items()):
            if status == ImageStatus.IN_PROGRESS:
                self._update_image_status(basename, ImageStatus.PENDING)

    def _mark_incomplete_as_skipped(self) -> None:
        """Flip PENDING + IN_PROGRESS rows to SKIPPED.

        Used by the cancel finalizer: a cancelled run is discarded, so
        anything that didn't reach SUCCEEDED or FAILED stays unprocessed
        for good (until the user starts a brand-new run). SKIPPED is the
        right resting state — the gray ↷ glyph signals "we didn't do
        this one" rather than the PENDING dot which would imply "still
        coming".
        """
        for basename, status in list(self._image_status.items()):
            if status in (ImageStatus.PENDING, ImageStatus.IN_PROGRESS):
                self._update_image_status(basename, ImageStatus.SKIPPED)

    @staticmethod
    def _shorten_error(error_text: str) -> str:
        """Squash a (potentially-multi-line, potentially-traceback) error to
        one line for inline display next to the row's filename.

        Takes the first non-empty line so multi-line tracebacks render
        tidily. Length is intentionally NOT capped here — QListWidget
        handles horizontal overflow on its own and the failure messages
        we surface (e.g. gate-confidence summaries with multiple
        landmarks) are far more useful when shown in full. The full
        text is also preserved in the run log either way.
        """
        if not error_text:
            return ""
        return next((line.strip() for line in error_text.splitlines() if line.strip()), "")

    def _update_image_status(self, basename: str, status: ImageStatus) -> None:
        """Set a row's glyph + foreground color from the given status.

        For FAILED rows, also appends a short single-line error message
        sourced from self._image_error_text (populated by the Stage-1 /
        Stage-2 failure slots). Non-FAILED states ignore the error map.
        """
        row = self._basename_to_row.get(basename)
        if row is None:
            return
        item = self.image_list.item(row)
        if item is None:
            return
        base_label = self._row_base_labels[row] if 0 <= row < len(self._row_base_labels) else basename
        self._image_status[basename] = status
        label = f"{_STATUS_GLYPH[status]}{base_label}"
        if status == ImageStatus.FAILED:
            err = self._shorten_error(self._image_error_text.get(basename, ""))
            if err:
                label = f"{label} — {err}"
        item.setText(label)
        item.setForeground(_status_color(status))

    def _select_output(self):
        _open_native_picker_async(
            self,
            "Select Output Folder",
            _picker_initial_path(self.output_edit.text()),
            self._on_output_folder_picked,
            folder=True,
            last_dir_key="output_folder",
            sync=True,
        )

    def _on_output_folder_picked(self, folder: str) -> None:
        if folder:
            self.output_edit.setText(folder)

    # -----------------------------------------------------------------------
    # Pipeline settings (PipelineConfig)
    # -----------------------------------------------------------------------
    def _set_scale(self, val: float, *, source: str) -> None:
        """Write um_per_px to the config and mirror the value to whichever
        scale spinbox didn't originate the change.

        Two spinboxes show µm/px — one on the left panel, one inside the
        General tab. They stay in lock-step; editing either drives the other.
        `source` is "left" or "inline" identifying the originating widget so
        we don't echo back into it (and re-fire its signal).
        """
        self.config.um_per_px = val if val > self.scale_spin.minimum() else None
        if source != "left":
            self.scale_spin.blockSignals(True)
            self.scale_spin.setValue(val)
            self.scale_spin.blockSignals(False)
        if source != "inline" and hasattr(self, "inline_general_panel"):
            self.inline_general_panel.scale_spin.blockSignals(True)
            self.inline_general_panel.scale_spin.setValue(val)
            self.inline_general_panel.scale_spin.blockSignals(False)

    def _on_auto_detect_um_per_px_toggled(self, checked: bool) -> None:
        """Left-panel checkbox handler — routes through _set_auto_detect_um_per_px
        so the inline General-tab mirror stays in sync.
        """
        self._set_auto_detect_um_per_px(bool(checked), source="left")

    def _set_auto_detect_um_per_px(self, checked: bool, *, source: str) -> None:
        """Sync ``auto_detect_um_per_px`` across the runtime config, the
        left-panel checkbox, and the inline General-tab checkbox.

        Two checkboxes show the flag; they stay in lock-step. ``source`` is
        "left" or "inline" identifying the originating widget so we don't
        echo back into it and re-fire its toggled signal. Enforcement
        (pre-flight scan for images without metadata) happens at Run-click
        time in ``_preflight_check_per_image_scale``.
        """
        self.config.auto_detect_um_per_px = bool(checked)
        if source != "left" and hasattr(self, "auto_detect_um_per_px_chk"):
            self.auto_detect_um_per_px_chk.blockSignals(True)
            self.auto_detect_um_per_px_chk.setChecked(bool(checked))
            self.auto_detect_um_per_px_chk.blockSignals(False)
        if source != "inline" and hasattr(self, "inline_general_panel"):
            inline_chk = getattr(self.inline_general_panel, "auto_detect_um_per_px_chk", None)
            if inline_chk is not None:
                inline_chk.blockSignals(True)
                inline_chk.setChecked(bool(checked))
                inline_chk.blockSignals(False)

    def _preflight_check_per_image_scale(self, effective_skips: Optional[set[str]] = None) -> bool:
        """Return True when the current config lets every image get a scale.

        Fires when ``auto_detect_um_per_px`` is on. Walks ``self._image_paths``
        (already filtered to supported image formats by ``discover_images``),
        drops any basenames in ``effective_skips`` (the pipeline's own skip
        set — user-unchecked rows, resume-cache hits, rerun's inverted set),
        and flags anything that's either a non-TIFF or a TIFF whose metadata
        isn't parseable. If any remain AND the manual scale field is empty,
        shows a QMessageBox and returns False so the run is aborted before
        any preprocessing starts. When the manual scale IS entered,
        missing-metadata images just fall through to that fallback (no
        error). When the flag is off, this method is a no-op that returns
        True.

        ``effective_skips`` should mirror the ``skip_image_basenames`` arg
        the caller ultimately passes to the worker: ``rerun_skip_set`` when
        set, otherwise ``resume_skip_set | user_skip_set``. Anything on that
        list won't be processed, so its missing metadata mustn't block Run.
        """
        if not bool(getattr(self.config, "auto_detect_um_per_px", False)):
            return True
        # If the user entered a manual scale, missing-metadata images just
        # use it as fallback — no pre-flight scan needed.
        manual_scale = self.config.um_per_px
        if manual_scale is not None and manual_scale > 0:
            return True
        # Nothing discovered yet → pipeline's own "no images" guard handles it.
        if not self._image_paths:
            return True
        try:
            from resolutionAdjust.auto_detect import (
                _MAX_PLAUSIBLE_UM_PER_PX,
                _MIN_PLAUSIBLE_UM_PER_PX,
                _TIFF_EXTS,
                _is_plausible_um_per_px,
                _read_um_per_px_from_tiff,
            )
        except Exception:  # pragma: no cover
            # If the detector can't be imported, we can't validate — allow
            # the run rather than block on our own tooling failure.
            return True
        skips = effective_skips or set()
        missing: list[str] = []
        implausible: list[tuple[str, float]] = []
        for p in self._image_paths:
            # Anything the pipeline itself won't process shouldn't block Run.
            if p.name in skips:
                continue
            if p.suffix.lower() not in _TIFF_EXTS:
                # Non-TIFF: this detector only reads TIFF tags / OME-XML, so
                # anything else counts as "no per-image scale available".
                missing.append(p.name)
                continue
            # Read the raw value so we can differentiate no-metadata from
            # bad-metadata in the error dialog. Runtime paths keep using
            # the filtered reader — the filter is what protects
            # resolutionAdjust from allocating terabytes when metadata is
            # a screen-DPI default.
            v = _read_um_per_px_from_tiff(p, allow_implausible=True)
            if v is None or v <= 0:
                missing.append(p.name)
            elif not _is_plausible_um_per_px(v):
                implausible.append((p.name, float(v)))
        if not missing and not implausible:
            return True
        # Truncate the lists so the dialog stays scannable.

        def _fmt(items: list[str]) -> str:
            preview = ", ".join(items[:8])
            more = f" (+{len(items) - 8} more)" if len(items) > 8 else ""
            return f"{preview}{more}"

        body_parts: list[str] = [
            "Per-image µm/px is enabled with no fallback scale entered.",
            "",
        ]
        if missing:
            body_parts.append(f"{len(missing)} image(s) don't carry parseable µm/px metadata:\n" f"  {_fmt(missing)}")
        if implausible:
            examples = _fmt([f"{name} (reported {v:.2f} µm/px)" for name, v in implausible])
            body_parts.append(
                f"{len(implausible)} image(s) reported a µm/px outside the "
                f"microscopy plausibility band [{_MIN_PLAUSIBLE_UM_PER_PX:.2f}, "
                f"{_MAX_PLAUSIBLE_UM_PER_PX:.2f}] — the metadata is almost "
                f"certainly a screen-DPI default (e.g. 96 dpi → 264.58 µm/px) "
                f"and not a real physical calibration:\n"
                f"  {examples}"
            )
        body_parts.extend(
            [
                "",
                "Either:",
                "  • Enter a µm/px value in the Scale field to use as the fallback, or",
                "  • Uncheck the images above in the list so they're skipped, or",
                "  • Uncheck 'Detect scale from image metadata' and enter a "
                "single value that applies to every image.",
            ]
        )
        title = "Missing per-image scale" if not implausible else "Per-image scale problem"
        QMessageBox.critical(self, title, "\n".join(body_parts))
        return False

    def reset_workers_warning(self):
        """Re-arm the spinner-change parallel-workers warning so it fires again on the next bump above 1.

        Skipped if the user has permanently suppressed the warning via the run-time dialog.
        """
        if self.settings.value("workers_warning_suppressed", False, type=bool):
            return
        self._workers_warning_shown = False

    def _on_workers_changed(self, val: int):
        self.maybe_show_workers_warning(val)

    def maybe_show_workers_warning(self, val: int) -> None:
        """Show the parallel-workers warning dialog once if `val` exceeds the
        default. Public entry point shared by the main-window Workers spinbox
        and the Settings → General → Parallel processing spinbox.

        No-op when the warning has already fired this session or the user has
        permanently suppressed it via the dialog's 'Don't show again' checkbox.
        """
        if val <= DEFAULT_MAX_WORKERS or self._workers_warning_shown:
            return
        if self.settings.value("workers_warning_suppressed", False, type=bool):
            return
        self._workers_warning_shown = True
        QMessageBox.warning(self, "Parallel workers", self._PARALLEL_WORKERS_WARNING_TEXT)

    def _show_workers_warning_info(self):
        """Open the parallel-workers help dialog: details text + Calibrate panel.

        Styled to match the QMessageBox warning that fires before a multi-worker
        run — bold message text, constrained width, right-aligned button row.
        """
        from PyQt5.QtGui import QFont

        from TRACE.calibrate_widget import CalibrateWidget

        dlg = QDialog(self)
        dlg.setWindowTitle("Parallel workers")
        dlg.setMinimumWidth(520)
        dlg.setMaximumWidth(560)
        layout = QVBoxLayout(dlg)
        layout.setContentsMargins(20, 20, 20, 16)
        layout.setSpacing(12)

        msg = QLabel(self._PARALLEL_WORKERS_DETAILS_TEXT)
        msg.setWordWrap(True)
        msg_font = QFont(msg.font())
        msg_font.setBold(True)
        msg.setFont(msg_font)
        layout.addWidget(msg)

        calib = CalibrateWidget(dlg)
        calib.set_paths(self.input_edit.text(), self._landmark_model_path, self._segmentation_model_path)
        calib.applied.connect(lambda v: self.inline_general_panel.workers_spin.setValue(int(v)))
        layout.addWidget(calib)

        close_row = QHBoxLayout()
        close_row.addStretch()
        close_btn = QPushButton("Close")
        close_btn.setDefault(True)
        close_btn.clicked.connect(dlg.accept)
        close_row.addWidget(close_btn)
        layout.addLayout(close_row)

        dlg.exec_()

    def _confirm_parallel_workers(self) -> bool:
        """Show the parallel-workers warning before a multi-worker run.

        Fires every time the pipeline is started with workers > 1, unless the
        user has previously checked "do not warn me again". Returns True if the
        run should proceed, False if the user cancelled.
        """
        if self.inline_general_panel.workers_spin.value() <= DEFAULT_MAX_WORKERS:
            return True
        if self.settings.value("workers_warning_suppressed", False, type=bool):
            return True

        box = QMessageBox(self)
        box.setIcon(QMessageBox.Warning)
        box.setWindowTitle("Parallel workers")
        box.setText(self._PARALLEL_WORKERS_WARNING_TEXT)
        box.setStandardButtons(QMessageBox.Ok | QMessageBox.Cancel)
        box.setDefaultButton(QMessageBox.Ok)
        suppress_chk = QCheckBox("Do not warn me again")
        box.setCheckBox(suppress_chk)
        result = box.exec_()
        if result != QMessageBox.Ok:
            return False
        if suppress_chk.isChecked():
            self.settings.setValue("workers_warning_suppressed", True)
        return True

    # -----------------------------------------------------------------------
    # GUI-only state snapshot for Save / Import (preset round-trip)
    # -----------------------------------------------------------------------
    # Field names that get serialized alongside the PipelineConfig + gate_override
    # so a saved preset captures every user-visible setting, not just the dataclass.
    _GUI_STATE_FIELDS = (
        "show_vein_tissue",
        "include_unreliable_landmarks",
        "do_rotation",
        "rotation_mirror_correct",
        "wing_isolation_enabled",
        "wing_expand_fraction",
        "wing_isolation_model_path",
        "landmark_model_path",
        "segmentation_model_path",
        "landmark_target_um_per_px",
        "segmentation_target_um_per_px",
        "wing_isolation_target_um_per_px",
        "active_rescale_target",
        "rescale_tolerance_low",
        "rescale_tolerance_high",
        "intermediate_outputs",
        "max_workers",
        "user_landmark_distances",
        "distance_sample_image",
        "distance_sample_landmarks",
    )

    def get_gui_state(self) -> dict:
        """Snapshot every GUI-only flag (everything not in PipelineConfig)."""
        return {
            "show_vein_tissue": bool(self._show_vein_tissue),
            "show_color_key": bool(self._show_color_key),
            "show_ectopic_labels": bool(self._show_ectopic_labels),
            "show_region_labels": bool(self._show_region_labels),
            "show_landmark_labels": bool(self._show_landmark_labels),
            "vein_simplify_tolerance_px": float(self._vein_simplify_tolerance_px),
            "ectopic_label_font_scale": float(self._ectopic_label_font_scale),
            "landmark_size_scale": float(self._landmark_size_scale),
            "show_compartment_labels": bool(self._show_compartment_labels),
            "include_unreliable_landmarks": bool(self._include_unreliable_landmarks),
            "do_rotation": bool(self._do_rotation),
            "rotation_mirror_correct": bool(self._rotation_mirror_correct),
            "wing_isolation_enabled": bool(self._wing_isolation_enabled),
            "wing_expand_fraction": float(self._wing_expand_fraction),
            "wing_isolation_model_path": str(self._wing_isolation_model_path or ""),
            "landmark_model_path": str(self._landmark_model_path or ""),
            "segmentation_model_path": str(self._segmentation_model_path or ""),
            "landmark_target_um_per_px": self._landmark_target_um_per_px,
            "segmentation_target_um_per_px": self._segmentation_target_um_per_px,
            "wing_isolation_target_um_per_px": self._wing_isolation_target_um_per_px,
            "active_rescale_target": str(self._active_rescale_target),
            "rescale_tolerance_low": float(self._rescale_tolerance_low),
            "rescale_tolerance_high": float(self._rescale_tolerance_high),
            "intermediate_outputs": dict(self._intermediate_outputs),
            # Final overlay outputs (Vein / Intervein / A-P compartment / CV
            # ratio / CSV / Custom measurements) — each is a top-level Outputs
            # checkbox. Saved so a config JSON round-trips the "what should
            # this run produce" selection along with the pipeline config
            # itself.
            "output_types": {key: chk.isChecked() for key, chk in self.output_checks.items()},
            # CSV measurement-group selections — the Wing area / Wing shape /
            # Vein lengths / etc. sub-checkboxes under Measurements CSV.
            "csv_measurement_groups": {key: chk.isChecked() for key, chk in self.csv_group_checks.items()},
            "include_custom_measurements": bool(self.include_custom_measurements_chk.isChecked()),
            "max_workers": int(self.inline_general_panel.workers_spin.value()),
            "user_landmark_distances": list(self._user_landmark_distances),
            "distance_sample_image": str(self._distance_sample_image or ""),
            "distance_sample_landmarks": str(self._distance_sample_landmarks or ""),
        }

    def apply_gui_state(self, state: dict) -> None:
        """Apply a snapshot from get_gui_state(). Unknown keys are ignored.

        After applying, refreshes the inline panels so their widgets reflect
        the new window state.
        """
        if "show_vein_tissue" in state:
            self._show_vein_tissue = bool(state["show_vein_tissue"])
        if "show_color_key" in state:
            self._show_color_key = bool(state["show_color_key"])
        if "show_ectopic_labels" in state:
            self._show_ectopic_labels = bool(state["show_ectopic_labels"])
        if "show_region_labels" in state:
            self._show_region_labels = bool(state["show_region_labels"])
        if "show_landmark_labels" in state:
            self._show_landmark_labels = bool(state["show_landmark_labels"])
        if "vein_simplify_tolerance_px" in state:
            try:
                self._vein_simplify_tolerance_px = float(state["vein_simplify_tolerance_px"])
            except (TypeError, ValueError):
                pass
        if "ectopic_label_font_scale" in state:
            try:
                self._ectopic_label_font_scale = float(state["ectopic_label_font_scale"])
            except (TypeError, ValueError):
                pass
        if "landmark_size_scale" in state:
            try:
                self._landmark_size_scale = float(state["landmark_size_scale"])
            except (TypeError, ValueError):
                pass
        if "show_compartment_labels" in state:
            self._show_compartment_labels = bool(state["show_compartment_labels"])
        if "include_unreliable_landmarks" in state:
            self._include_unreliable_landmarks = bool(state["include_unreliable_landmarks"])
        if "do_rotation" in state:
            self._do_rotation = bool(state["do_rotation"])
        if "rotation_mirror_correct" in state:
            self._rotation_mirror_correct = bool(state["rotation_mirror_correct"])
        if "wing_isolation_enabled" in state:
            self._wing_isolation_enabled = bool(state["wing_isolation_enabled"])
        if "wing_expand_fraction" in state:
            try:
                self._wing_expand_fraction = float(state["wing_expand_fraction"])
            except (TypeError, ValueError):
                pass
        if "wing_isolation_model_path" in state:
            self._wing_isolation_model_path = str(state["wing_isolation_model_path"] or "")
        if "landmark_model_path" in state:
            self._landmark_model_path = str(state["landmark_model_path"] or "")
        if "segmentation_model_path" in state:
            self._segmentation_model_path = str(state["segmentation_model_path"] or "")
        for key in ("landmark_target_um_per_px", "segmentation_target_um_per_px", "wing_isolation_target_um_per_px"):
            if key in state:
                val = state[key]
                try:
                    setattr(self, f"_{key}", float(val) if val is not None else None)
                except (TypeError, ValueError):
                    pass
        if "active_rescale_target" in state and state["active_rescale_target"] in (
            "landmark",
            "segmentation",
            "wing_isolation",
        ):
            self._active_rescale_target = str(state["active_rescale_target"])
        for key in ("rescale_tolerance_low", "rescale_tolerance_high"):
            if key in state:
                try:
                    setattr(self, f"_{key}", float(state[key]))
                except (TypeError, ValueError):
                    pass
        if "intermediate_outputs" in state and isinstance(state["intermediate_outputs"], dict):
            # Preserve the original key universe; only update keys present in both.
            for k, v in state["intermediate_outputs"].items():
                if k in self._intermediate_outputs:
                    self._intermediate_outputs[k] = bool(v)
        if "output_types" in state and isinstance(state["output_types"], dict):
            # Preserve the original key universe; only update keys present in
            # both. Setting the checkbox fires its toggled signal, which
            # propagates to any dependent widgets (e.g. the "requires
            # Measurements CSV" hint on child rows).
            for k, v in state["output_types"].items():
                if k in self.output_checks:
                    self.output_checks[k].setChecked(bool(v))
        if "csv_measurement_groups" in state and isinstance(state["csv_measurement_groups"], dict):
            for k, v in state["csv_measurement_groups"].items():
                if k in self.csv_group_checks:
                    self.csv_group_checks[k].setChecked(bool(v))
        if "include_custom_measurements" in state:
            self.include_custom_measurements_chk.setChecked(bool(state["include_custom_measurements"]))
        if "max_workers" in state:
            try:
                workers_val = int(state["max_workers"])
            except (TypeError, ValueError):
                workers_val = None
            if workers_val is not None:
                # Persist to QSettings first since refresh_from_state() reads
                # max_workers from there — without this, the spinbox would
                # snap back to the persisted (pre-import) value.
                self.settings.setValue("max_workers", workers_val)
        if "user_landmark_distances" in state and isinstance(state["user_landmark_distances"], list):
            self._user_landmark_distances = [p for p in state["user_landmark_distances"] if isinstance(p, dict)]
        if "distance_sample_image" in state:
            self._distance_sample_image = str(state["distance_sample_image"] or "")
        if "distance_sample_landmarks" in state:
            self._distance_sample_landmarks = str(state["distance_sample_landmarks"] or "")
        # Push window state into both inline panels so their widgets reflect
        # the imported snapshot.
        self.inline_general_panel.refresh_from_state()
        self.inline_custom_distances_panel.refresh_from_state()

    def _open_settings_dialog(self):
        """Open the advanced settings dialog (5 tabs: Landmarks, Models,
        Wing Graph, Tracing, Intervein).

        Non-modal: the user can still interact with the main window (run
        the pipeline, change inputs, etc.) while the dialog is open.
        Settings only apply when the user clicks OK — Cancel and window-
        close discard pending edits. If the dialog is already open when
        the user clicks Advanced again, the existing window is raised
        rather than spawning a second instance.

        General and Custom Distances live as right-panel tabs on the main
        window; they don't pass through this dialog.
        """
        existing = getattr(self, "_settings_dialog", None)
        if existing is not None and existing.isVisible():
            existing.raise_()
            existing.activateWindow()
            return
        # Trap dialog-construction errors so they land in trace_startup.log
        # instead of vanishing into Qt's event loop (which on a frozen
        # Windows build with no console means the user sees nothing
        # happen at all when they click Advanced Settings, then has no
        # error trail to point at). Surface as a QMessageBox so the
        # user gets immediate feedback too.
        try:
            dlg = PipelineConfigDialog(
                self.config,
                self,
                include_unreliable_landmarks=self._include_unreliable_landmarks,
                input_path=self.input_edit.text(),
                landmark_model_path=self._landmark_model_path,
                segmentation_model_path=self._segmentation_model_path,
                gate_override=self._gate_override,
                wing_expand_fraction=self._wing_expand_fraction,
                wing_isolation_model_path=self._wing_isolation_model_path,
                landmark_target_um_per_px=self._landmark_target_um_per_px,
                segmentation_target_um_per_px=self._segmentation_target_um_per_px,
                wing_isolation_target_um_per_px=self._wing_isolation_target_um_per_px,
                active_rescale_target=self._active_rescale_target,
                rescale_tolerance_low=self._rescale_tolerance_low,
                rescale_tolerance_high=self._rescale_tolerance_high,
            )
        except BaseException as exc:  # noqa: BLE001
            import traceback as _tb

            tb_text = "".join(_tb.format_exception(type(exc), exc, exc.__traceback__))
            try:
                from TRACE.startup_log import log as _slog

                _slog(f"settings_dialog: construction failed\n{tb_text}")
            except Exception:
                pass
            QMessageBox.critical(
                self,
                "Settings dialog failed",
                f"Could not open the settings dialog:\n\n{type(exc).__name__}: {exc}\n\n"
                "The full traceback is in trace_startup.log (next to TRACE.exe).",
            )
            return
        # Window-modal flag stays False (the QDialog default) so show()
        # leaves the parent interactive. setAttribute(WA_DeleteOnClose)
        # would simplify lifetime but we keep the reference around so
        # the raise-existing path above works regardless of how the
        # dialog was last dismissed.
        self._settings_dialog = dlg
        dlg.accepted.connect(lambda d=dlg: self._apply_settings_dialog_result(d))
        dlg.finished.connect(lambda _r: self._on_settings_dialog_finished())
        dlg.show()
        # Belt-and-braces: ensure the dialog is actually focused + on top
        # the moment it's shown, even when called from a non-active
        # context (Windows occasionally leaves the new window behind the
        # main one without explicit raise_/activateWindow calls).
        dlg.raise_()
        dlg.activateWindow()

    def _apply_settings_dialog_result(self, dlg) -> None:
        """Copy settings from the now-accepted dialog back onto self.

        Called by the dialog's ``accepted`` signal (OK button). On
        Cancel / close-window the dialog emits ``rejected`` and this is
        skipped — the existing self.config etc. stay untouched.
        """
        self.config = dlg.get_config()
        self._include_unreliable_landmarks = dlg.get_include_unreliable_landmarks()
        self._gate_override = dlg.get_gate_override()
        self._wing_expand_fraction = dlg.get_wing_expand_fraction()
        self._wing_isolation_model_path = dlg.get_wing_isolation_model_path()
        self._landmark_model_path = dlg.get_landmark_model_path()
        self._segmentation_model_path = dlg.get_segmentation_model_path()
        self._landmark_target_um_per_px = dlg.get_landmark_target_um_per_px()
        self._segmentation_target_um_per_px = dlg.get_segmentation_target_um_per_px()
        self._wing_isolation_target_um_per_px = dlg.get_wing_isolation_target_um_per_px()
        self._active_rescale_target = dlg.get_active_rescale_target()
        self._rescale_tolerance_low = dlg.get_rescale_tolerance_low()
        self._rescale_tolerance_high = dlg.get_rescale_tolerance_high()
        # Dialog may have edited fields the inline General panel mirrors
        # (e.g. synthesize_missing_crossveins on the Tracing tab) — pull
        # current state into the panel widgets so they stay in sync.
        self.inline_general_panel.refresh_from_state()
        # Same for the per-image-metadata checkbox on the left panel: the
        # dialog might have flipped auto_detect_um_per_px via config import.
        if hasattr(self, "auto_detect_um_per_px_chk"):
            self.auto_detect_um_per_px_chk.blockSignals(True)
            self.auto_detect_um_per_px_chk.setChecked(bool(getattr(self.config, "auto_detect_um_per_px", False)))
            self.auto_detect_um_per_px_chk.blockSignals(False)

    def _on_settings_dialog_finished(self) -> None:
        """Drop the cached dialog reference so the next Advanced click
        builds a fresh dialog rather than re-raising a hidden one."""
        self._settings_dialog = None

    # Import/Save pipeline-config JSON lives on PipelineConfigDialog (next to
    # Restore Defaults) — it's only useful in the context of editing the full
    # config and keeps the main window leaner.

    def _reset_gui_to_defaults(self):
        """Wipe all persisted QSettings and snap every widget back to factory defaults.

        Equivalent to deleting the QSettings file and relaunching the GUI — useful
        when a user wants the first-time experience back without restarting.
        """
        reply = QMessageBox.warning(
            self,
            "Reset GUI to defaults",
            "This will clear every saved setting — input/output folders, model paths, "
            "scale, pipeline configuration, custom distance pairs, and any 'don't show "
            "again' warning suppressions — and snap every widget back to its first-launch "
            "state.\n\nContinue?",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )
        if reply != QMessageBox.Yes:
            return

        # Wipe persisted settings.
        self.settings.clear()
        self.settings.sync()

        # Reset instance state to __init__ defaults.
        self.config = PipelineConfig()
        self._show_vein_tissue = False
        self._show_color_key = True
        self._show_ectopic_labels = True
        self._show_region_labels = True
        self._show_landmark_labels = True
        # NB: __init__ + reset-defaults keep in sync — both instances updated
        # by this replace_all; see also the QSettings/gui_state persistence
        # sites just below and the pipeline pass-through in _run_pipeline.
        self._vein_simplify_tolerance_px = 0.0
        self._ectopic_label_font_scale = 1.0
        self._landmark_size_scale = 1.0
        self._show_compartment_labels = True
        self._include_unreliable_landmarks = False
        self._do_rotation = False
        self._rotation_mirror_correct = False
        self._gate_override = None
        self._wing_expand_fraction = 0.05
        self._wing_isolation_enabled = False
        # Wipe-my-memories re-applies the bundled defaults (TRACE/models/*)
        # so a fresh user gets working model paths without having to browse.
        # The Settings dialog's Restore Defaults button does NOT touch these.
        self._wing_isolation_model_path = _default_model_path("wing_isolation")
        self._landmark_model_path = _default_model_path("landmark")
        self._segmentation_model_path = _default_model_path("segmentation")
        self._landmark_target_um_per_px = None
        self._segmentation_target_um_per_px = 0.483
        self._wing_isolation_target_um_per_px = None
        self._active_rescale_target = "segmentation"
        self._rescale_tolerance_low = 0.85
        self._rescale_tolerance_high = 1.15
        self._intermediate_outputs = {key: False for key in INTERMEDIATE_OUTPUTS}
        self._user_landmark_distances = []
        self._distance_sample_image = ""
        self._distance_sample_landmarks = ""
        self._workers_warning_shown = False
        self._image_paths = []
        # Wipe the per-row status tracking too — the list is about to be
        # repopulated from scratch by _refresh_image_list once the input
        # path is cleared.
        self._image_status.clear()
        self._basename_to_row.clear()
        self._row_base_labels.clear()
        self._image_error_text.clear()
        # User-skip set lives in QSettings under user_skip/<sha1>; the
        # settings.clear() at the top of this method already drops every
        # such key. Wipe the in-memory mirror to match.
        self._user_skip_set.clear()
        # Failed-images bookkeeping for the rerun buttons.
        self._failure_category.clear()
        self._last_run_failed_set.clear()
        self._pending_rerun_skip_set = None
        self._pending_csv_filename_override = None
        self._pending_skip_workers_warning = False
        self._pending_gate_override = None
        self._pending_disable_garbage_filters = False
        # Walkthrough seen-flags are stored in QSettings; settings.clear()
        # at the top of this method already cleared them.

        # Snap every widget back. blockSignals where the slot would re-fire a
        # warning or write back to a no-longer-stale value.
        self.input_edit.clear()
        self.output_edit.clear()
        self.recursive_chk.blockSignals(True)
        self.recursive_chk.setChecked(False)
        self.recursive_chk.blockSignals(False)
        self.image_list.clear()

        for chk in self.output_checks.values():
            chk.setChecked(True)
        for gchk in self.csv_group_checks.values():
            gchk.setChecked(True)
        self.include_custom_measurements_chk.setChecked(True)

        # Snap inline-panel widgets (scale, workers, intermediate outputs,
        # opacities, color pickers, custom distance pairs) back to the
        # freshly-reset window state.
        self.inline_general_panel.refresh_from_state()
        self.inline_custom_distances_panel.refresh_from_state()
        # Reset the per-image-metadata checkbox to the fresh-config default (off).
        if hasattr(self, "auto_detect_um_per_px_chk"):
            self.auto_detect_um_per_px_chk.blockSignals(True)
            self.auto_detect_um_per_px_chk.setChecked(False)
            self.auto_detect_um_per_px_chk.blockSignals(False)

        self.log_text.clear()
        self.statusBar().showMessage("GUI reset to defaults")

    # -----------------------------------------------------------------------
    # Settings persistence
    # -----------------------------------------------------------------------
    def _selected_outputs(self) -> set[str]:
        finals = {key for key, chk in self.output_checks.items() if chk.isChecked()}
        intermediates = {key for key, on in self._intermediate_outputs.items() if on}
        return finals | intermediates

    def _save_settings(self):
        s = self.settings
        s.setValue("input_folder", self.input_edit.text())
        s.setValue("input_recursive", self.recursive_chk.isChecked())
        s.setValue("output_folder", self.output_edit.text())
        s.setValue("landmark_model", self._landmark_model_path)
        s.setValue("segmentation_model", self._segmentation_model_path)
        s.setValue("wing_isolation_enabled", self._wing_isolation_enabled)
        s.setValue("wing_isolation_model", self._wing_isolation_model_path)
        s.setValue("wing_expand_fraction", self._wing_expand_fraction)
        s.setValue(
            "models/landmark_target_um_per_px",
            "" if self._landmark_target_um_per_px is None else str(self._landmark_target_um_per_px),
        )
        s.setValue(
            "models/segmentation_target_um_per_px",
            "" if self._segmentation_target_um_per_px is None else str(self._segmentation_target_um_per_px),
        )
        s.setValue(
            "models/wing_isolation_target_um_per_px",
            "" if self._wing_isolation_target_um_per_px is None else str(self._wing_isolation_target_um_per_px),
        )
        s.setValue("models/active_rescale_target", self._active_rescale_target)
        s.setValue("resolution/tolerance_low", str(self._rescale_tolerance_low))
        s.setValue("resolution/tolerance_high", str(self._rescale_tolerance_high))
        s.setValue("pipeline_config_json", config_to_json(self.config))
        s.setValue("max_workers", self.inline_general_panel.workers_spin.value())
        s.setValue("show_vein_tissue", self._show_vein_tissue)
        s.setValue("show_color_key", self._show_color_key)
        s.setValue("show_ectopic_labels", self._show_ectopic_labels)
        s.setValue("show_region_labels", self._show_region_labels)
        s.setValue("show_landmark_labels", self._show_landmark_labels)
        s.setValue("vein_simplify_tolerance_px", str(self._vein_simplify_tolerance_px))
        s.setValue("ectopic_label_font_scale", str(self._ectopic_label_font_scale))
        s.setValue("landmark_size_scale", str(self._landmark_size_scale))
        s.setValue("show_compartment_labels", self._show_compartment_labels)
        s.setValue("include_unreliable_landmarks", self._include_unreliable_landmarks)
        s.setValue("do_rotation", self._do_rotation)
        s.setValue("rotation_mirror_correct", self._rotation_mirror_correct)
        import json as _json

        s.setValue(
            "gate_override_json",
            _json.dumps(self._gate_override) if self._gate_override else "",
        )
        s.setValue(
            "user_landmark_distances_json",
            _json.dumps(self._user_landmark_distances) if self._user_landmark_distances else "",
        )
        s.setValue("include_custom_measurements", self.include_custom_measurements_chk.isChecked())
        s.setValue("distance_sample_image", self._distance_sample_image)
        s.setValue("distance_sample_landmarks", self._distance_sample_landmarks)
        for key, chk in self.output_checks.items():
            s.setValue(f"output/{key}", chk.isChecked())
        for key, on in self._intermediate_outputs.items():
            s.setValue(f"output/{key}", on)
        for gkey, gchk in self.csv_group_checks.items():
            s.setValue(f"csv_group/{gkey}", gchk.isChecked())

    def _restore_settings(self):
        s = self.settings
        saved_recursive = s.value("input_recursive", None)
        if saved_recursive is not None:
            self.recursive_chk.setChecked(saved_recursive == "true" or saved_recursive is True)
        val = s.value("input_folder", "")
        if val:
            self.input_edit.setText(val)
            self._refresh_image_list()
        val = s.value("output_folder", "")
        if val:
            self.output_edit.setText(val)
        # `_restore_model_path` handles all four cases at once: saved-and-valid,
        # legacy nested-checkpoints layout, stale path (project dir moved between
        # sessions) with a bundled default available, and pristine first-launch.
        self._landmark_model_path = _restore_model_path(s.value("landmark_model", ""), "landmark")
        self._segmentation_model_path = _restore_model_path(s.value("segmentation_model", ""), "segmentation")
        self._wing_isolation_model_path = _restore_model_path(s.value("wing_isolation_model", ""), "wing_isolation")

        def _parse_optional_float(raw) -> Optional[float]:
            if raw in (None, ""):
                return None
            try:
                v = float(raw)
            except (TypeError, ValueError):
                return None
            return v if v > 0 else None

        # `s.value(key, None)` returns None when the key was never written —
        # we use that to distinguish "first launch / pristine settings" (apply
        # the factory default) from "user deliberately cleared the field"
        # (saved as "", parse to None).
        raw_lm = s.value("models/landmark_target_um_per_px", None)
        if raw_lm is not None:
            self._landmark_target_um_per_px = _parse_optional_float(raw_lm)
        raw_seg = s.value("models/segmentation_target_um_per_px", None)
        if raw_seg is not None:
            self._segmentation_target_um_per_px = _parse_optional_float(raw_seg)
        raw_wing = s.value("models/wing_isolation_target_um_per_px", None)
        if raw_wing is not None:
            self._wing_isolation_target_um_per_px = _parse_optional_float(raw_wing)
        saved_active = s.value("models/active_rescale_target", "")
        if saved_active in ("landmark", "segmentation", "wing_isolation"):
            self._active_rescale_target = saved_active
        saved_tol_low = s.value("resolution/tolerance_low", "")
        try:
            if saved_tol_low not in (None, ""):
                self._rescale_tolerance_low = float(saved_tol_low)
        except (TypeError, ValueError):
            pass
        saved_tol_high = s.value("resolution/tolerance_high", "")
        try:
            if saved_tol_high not in (None, ""):
                self._rescale_tolerance_high = float(saved_tol_high)
        except (TypeError, ValueError):
            pass
        saved_wing = s.value("wing_isolation_enabled", None)
        if saved_wing is not None:
            self._wing_isolation_enabled = saved_wing == "true" or saved_wing is True
        saved_wef = s.value("wing_expand_fraction", None)
        if saved_wef is not None:
            try:
                self._wing_expand_fraction = float(saved_wef)
            except (TypeError, ValueError):
                pass
        cfg_json = s.value("pipeline_config_json", None)
        if cfg_json:
            try:
                self.config = config_from_json(cfg_json)
            except Exception:
                self.config = PipelineConfig()
        # Scale is owned by the inline General panel — its refresh_from_state()
        # (called at the end of _restore_settings) reads self.config.um_per_px
        # into its spinbox.
        # Migrate legacy "output/overlay" setting → both new keys (vein + intervein).
        legacy_overlay = s.value("output/overlay", None)
        if legacy_overlay is not None:
            legacy_on = legacy_overlay == "true" or legacy_overlay is True
            for new_key in ("vein_overlay", "intervein_overlay"):
                if s.value(f"output/{new_key}", None) is None:
                    s.setValue(f"output/{new_key}", legacy_on)
            s.remove("output/overlay")
        for key, chk in self.output_checks.items():
            saved = s.value(f"output/{key}", None)
            if saved is not None:
                chk.setChecked(saved == "true" or saved is True)
        for key in self._intermediate_outputs:
            saved = s.value(f"output/{key}", None)
            if saved is not None:
                self._intermediate_outputs[key] = saved == "true" or saved is True
        # Migrate legacy csv_group keys from the earlier split:
        #   wing_dimensions (area + length) → wing_area (just area)
        #   crossvein_distance              → cv_ratio (now also owns wing length)
        # Only write the new key if it hasn't already been saved.
        for old_key, new_key in (("wing_dimensions", "wing_area"), ("crossvein_distance", "cv_ratio")):
            legacy = s.value(f"csv_group/{old_key}", None)
            if legacy is not None:
                if s.value(f"csv_group/{new_key}", None) is None:
                    s.setValue(f"csv_group/{new_key}", legacy == "true" or legacy is True)
                s.remove(f"csv_group/{old_key}")
        for gkey, gchk in self.csv_group_checks.items():
            saved = s.value(f"csv_group/{gkey}", None)
            if saved is not None:
                gchk.setChecked(saved == "true" or saved is True)
        # Sync the nested-group container's enabled state with the parent CSV checkbox.
        if self._csv_group_container is not None and "csv" in self.output_checks:
            self._csv_group_container.setEnabled(self.output_checks["csv"].isChecked())
        saved_svt = s.value("show_vein_tissue", None)
        if saved_svt is not None:
            self._show_vein_tissue = saved_svt == "true" or saved_svt is True
        saved_sck = s.value("show_color_key", None)
        if saved_sck is not None:
            self._show_color_key = saved_sck == "true" or saved_sck is True
        saved_sel = s.value("show_ectopic_labels", None)
        if saved_sel is not None:
            self._show_ectopic_labels = saved_sel == "true" or saved_sel is True
        saved_srl = s.value("show_region_labels", None)
        if saved_srl is not None:
            self._show_region_labels = saved_srl == "true" or saved_srl is True
        saved_sll = s.value("show_landmark_labels", None)
        if saved_sll is not None:
            self._show_landmark_labels = saved_sll == "true" or saved_sll is True
        saved_vst = s.value("vein_simplify_tolerance_px", None)
        if saved_vst is not None:
            try:
                self._vein_simplify_tolerance_px = float(saved_vst)
            except (TypeError, ValueError):
                pass
        saved_elfs = s.value("ectopic_label_font_scale", None)
        if saved_elfs is not None:
            try:
                self._ectopic_label_font_scale = float(saved_elfs)
            except (TypeError, ValueError):
                pass
        saved_lss = s.value("landmark_size_scale", None)
        if saved_lss is not None:
            try:
                self._landmark_size_scale = float(saved_lss)
            except (TypeError, ValueError):
                pass
        saved_scl = s.value("show_compartment_labels", None)
        if saved_scl is not None:
            self._show_compartment_labels = saved_scl == "true" or saved_scl is True
        saved_dor = s.value("do_rotation", None)
        if saved_dor is not None:
            self._do_rotation = saved_dor == "true" or saved_dor is True
        saved_rmc = s.value("rotation_mirror_correct", None)
        if saved_rmc is not None:
            self._rotation_mirror_correct = saved_rmc == "true" or saved_rmc is True
        saved_iul = s.value("include_unreliable_landmarks", None)
        if saved_iul is not None:
            self._include_unreliable_landmarks = saved_iul == "true" or saved_iul is True
        saved_gate = s.value("gate_override_json", "")
        if saved_gate:
            try:
                import json as _json

                self._gate_override = _json.loads(saved_gate)
            except Exception:
                self._gate_override = None
        saved_uld = s.value("user_landmark_distances_json", "")
        if saved_uld:
            try:
                import json as _json

                parsed = _json.loads(saved_uld)
                if isinstance(parsed, list):
                    self._user_landmark_distances = [p for p in parsed if isinstance(p, dict)]
            except Exception:
                self._user_landmark_distances = []
        saved_icm = s.value("include_custom_measurements", None)
        if saved_icm is not None:
            self.include_custom_measurements_chk.setChecked(saved_icm == "true" or saved_icm is True)
        saved_dsi = s.value("distance_sample_image", "")
        if saved_dsi:
            self._distance_sample_image = saved_dsi
        saved_dsl = s.value("distance_sample_landmarks", "")
        if saved_dsl:
            self._distance_sample_landmarks = saved_dsl
        # Workers is owned by the inline General panel — its refresh_from_state()
        # reads the persisted max_workers value out of QSettings directly.

        # Inline panels mirror window state into widgets. Called last so every
        # earlier restore (config, intermediates, distance pairs, etc.) is
        # visible to the panels by the time they sync.
        self.inline_general_panel.refresh_from_state()
        self.inline_custom_distances_panel.refresh_from_state()

        # Sync the per-image-metadata checkbox with the restored config —
        # ``auto_detect_um_per_px`` round-trips through pipeline_config_json
        # like every other PipelineConfig field, but the checkbox widget
        # isn't automatically re-read; do it here.
        if hasattr(self, "auto_detect_um_per_px_chk"):
            self.auto_detect_um_per_px_chk.blockSignals(True)
            self.auto_detect_um_per_px_chk.setChecked(bool(getattr(self.config, "auto_detect_um_per_px", False)))
            self.auto_detect_um_per_px_chk.blockSignals(False)

    # -----------------------------------------------------------------------
    # Rerun-failed-images flow (TODO #11)
    # -----------------------------------------------------------------------
    def _refresh_rerun_buttons(self) -> None:
        """Show/hide the rerun buttons based on the most recent run's failure set.

        Called from every run-end handler (_on_all_done / _on_paused /
        _on_cancelled), from _run_pipeline at run start (hides both),
        and from the input-folder change signal (also hides — the
        failed set's basenames refer to the prior input folder).

        First-time visibility of either button also fires a one-step
        walkthrough hint, throttled by per-button QSettings flags so the
        hint shows at most once per user.
        """
        failed = self._last_run_failed_set
        has_failed = bool(failed)
        self.btn_rerun_failed.setVisible(has_failed)
        # No-gate variant only when ≥1 failure was a confidence-gate
        # abort. Stage-2 ("analysis") and other Stage-1 ("preproc_other")
        # failures don't surface this button — the no-gate rerun
        # specifically targets LowConfidenceLandmarkError cases.
        has_gate_failure = any(self._failure_category.get(name) == "gate" for name in failed)
        self.btn_rerun_failed_nogate.setVisible(has_failed and has_gate_failure)
        # Review-failed shares the rerun buttons' visibility — it opens the
        # landmark inspector on the failed set so the user can hand-correct.
        self.btn_review_failed.setVisible(has_failed)
        if has_failed:
            self.btn_review_failed.setText(f"Review failed images ({len(failed)})")

        # Failed-run walkthrough — multi-step overlay shown when ≥1 image
        # failed in the just-finished run. Step 1 (no highlight) summarizes
        # WHY images failed; steps 2-4 each highlight one of the post-run
        # buttons (Review, Rerun, Rerun no-quality-gates). The no-gates
        # step is conditional on a quality-gate failure being present —
        # otherwise the button is hidden and there's nothing to point at.
        # Deferred 300 ms so the buttons' freshly-set visibility is settled
        # before the overlay measures their geometry.
        if has_failed and not self.settings.value("failed_run_walkthrough_dismissed", False, type=bool):
            QTimer.singleShot(300, self._show_failed_run_walkthrough)

    def _review_failed_images(self) -> None:
        """Open the landmark inspector on the last run's failed images (cohort)."""
        failed_basenames = sorted(self._last_run_failed_set)
        if not failed_basenames:
            return
        failed_paths: list = []
        for bn in failed_basenames:
            row = self._basename_to_row.get(bn)
            if row is not None and 0 <= row < len(self._image_paths):
                failed_paths.append(self._image_paths[row])
        if not failed_paths:
            QMessageBox.warning(
                self,
                "Review failed images",
                "Couldn't locate the failed images. Has the input folder changed " "since the last run?",
            )
            return
        self._open_landmark_inspector(failed_paths[0], cohort=failed_paths)

    def _summarize_failure_reasons(self, failed: set[str]) -> list[str]:
        """Group the just-failed images into human-readable bullet points
        for the failed-run walkthrough's step-1 summary.

        Buckets:
          - Landmark confidence gate (Stage 1, regex on the legacy
            "Core landmarks failed confidence gate" prefix).
          - Garbage-detector quality gates, split per filter — solidity /
            fragmentation / vein-association / vein-presence — using the
            "Aborted by quality gate (<filter>):" prefix added in
            TRACE/pipeline.py's GarbageRejection handler.
          - Other Stage-1 (preprocessing) failures: model load, file IO,
            wing isolation with no wing found, etc.
          - Other Stage-2 (analysis) failures: vein-tracer crashes, output
            write errors, etc.
        """
        import re as _re

        landmark_gate = 0
        quality_gate_by_filter: dict[str, int] = {}
        other_preproc = 0
        other_analysis = 0
        for name in failed:
            cat = self._failure_category.get(name, "")
            msg = self._image_error_text.get(name, "")
            if cat == "gate":
                m = _re.search(r"Aborted by quality gate \(([^)]+)\)", msg, _re.IGNORECASE)
                if m:
                    label = m.group(1).strip()
                    quality_gate_by_filter[label] = quality_gate_by_filter.get(label, 0) + 1
                else:
                    landmark_gate += 1
            elif cat == "preproc_other":
                other_preproc += 1
            else:  # "analysis" or unknown
                other_analysis += 1

        def _img_count(n: int) -> str:
            return f"{n} image" if n == 1 else f"{n} images"

        reasons: list[str] = []
        if landmark_gate:
            reasons.append(f"{_img_count(landmark_gate)} failed a landmark confidence gate")
        for filter_name in sorted(quality_gate_by_filter):
            n = quality_gate_by_filter[filter_name]
            reasons.append(f"{_img_count(n)} failed the {filter_name} quality gate")
        if other_preproc:
            reasons.append(
                f"{_img_count(other_preproc)} failed during preprocessing "
                f"(model load / file read / no wing found / etc.)"
            )
        if other_analysis:
            reasons.append(f"{_img_count(other_analysis)} failed during analysis")
        return reasons

    def _show_failed_run_walkthrough(self) -> None:
        """Four-step walkthrough fired after a run with ≥1 failed image.

        Step 1 (no highlight): summary of WHY images failed.
        Step 2: Review failed images button.
        Step 3: Rerun failed images button.
        Step 4 (conditional): Rerun failed (no quality gates) button —
            only when a quality-gate abort is present, since that's the
            only case where the button is visible.

        Settings_key="failed_run_walkthrough_dismissed" replaces the
        old per-button hint keys; ticking "Don't show again" on any step
        permanently suppresses the whole walkthrough.

        If the main on-launch walkthrough is still up (rare — user
        finishes a run while still reading the tutorial), defer 500 ms
        so two overlays don't fight over the dim snapshot.
        """
        failed = self._last_run_failed_set
        if not failed:
            return
        if self._walkthrough is not None:
            try:
                still_visible = self._walkthrough.isVisible()
            except RuntimeError:
                still_visible = False
            if still_visible:
                QTimer.singleShot(500, self._show_failed_run_walkthrough)
                return

        reasons = self._summarize_failure_reasons(failed)
        # Format the summary body. PyQt's QLabel handles "<br>"; bullet
        # rendering via Unicode bullets is fine since the popup body
        # already supports HTML (the existing hints use <b>).
        summary_body = "At least one image failed because of the following reasons:<br><br>" + "<br>".join(
            f"• {r}" for r in reasons
        )
        steps: list[WalkthroughStep] = [
            WalkthroughStep(
                target_resolver=lambda _w: None,
                title="Some images failed",
                body=summary_body,
            ),
            WalkthroughStep(
                target_resolver=lambda _w: self.btn_review_failed,
                title="Review failed images",
                body=(
                    "Click <b>Review failed images</b> to open failed images in the "
                    "model inspector. Review and edit landmark point locations and "
                    "vein/intervein tissue detection — those corrections become "
                    "per-image overrides that the next run picks up automatically."
                ),
            ),
            WalkthroughStep(
                target_resolver=lambda _w: self.btn_rerun_failed,
                title="Rerun failed images",
                body=(
                    "Click <b>Rerun failed images</b> to re-process only the failed "
                    "images, e.g. after adjusting settings. You'll be asked whether to "
                    "append the new measurements to the existing CSV or write a new one."
                ),
            ),
        ]
        has_gate_failure = any(self._failure_category.get(n) == "gate" for n in failed)
        if has_gate_failure:
            steps.append(
                WalkthroughStep(
                    target_resolver=lambda _w: self.btn_rerun_failed_nogate,
                    title="Rerun without quality gates",
                    body=(
                        "Click <b>Rerun failed (no quality gates)</b> to reprocess the "
                        "failed images with all quality gates temporarily disabled. Use "
                        "when you suspect a gate is too strict. Your saved settings "
                        "won't be changed."
                    ),
                )
            )
        overlay = WalkthroughOverlay(
            self,
            steps,
            settings=self.settings,
            settings_key="failed_run_walkthrough_dismissed",
            dont_show_label="Don't show this again",
        )
        overlay.start()

    def _show_button_hint(
        self,
        target_btn,
        *,
        settings_key: str,
        title: str,
        body: str,
    ) -> None:
        """One-step WalkthroughOverlay highlighting a single button.

        Persistence is driven by the popup's "Don't show this again"
        checkbox — WalkthroughOverlay sets the QSettings key only when
        the user ticks the box on close. So a user who dismisses
        without ticking will see the hint again on the next run that
        ends with the same failure pattern, and one who ticks it
        suppresses it permanently.

        If the main walkthrough is still up, defer 500ms — two overlays
        in the same coordinate space would fight over the dim snapshot.
        """
        if self._walkthrough is not None:
            try:
                still_visible = self._walkthrough.isVisible()
            except RuntimeError:
                # C++ side already deleted; treat as not visible.
                still_visible = False
            if still_visible:
                QTimer.singleShot(
                    500,
                    lambda: self._show_button_hint(target_btn, settings_key=settings_key, title=title, body=body),
                )
                return

        step = WalkthroughStep(
            target_resolver=lambda _win: target_btn,
            title=title,
            body=body,
        )
        # Pass settings + settings_key so the overlay's "Don't show this
        # again" checkbox is what decides whether to suppress on the
        # next run. Without these, the overlay would just close and
        # re-fire every time (also acceptable, but no opt-out).
        # The default label ("...on launch") fits the main walkthrough
        # but reads wrong for a post-run hint; pass a friendlier one.
        overlay = WalkthroughOverlay(
            self,
            [step],
            settings=self.settings,
            settings_key=settings_key,
            dont_show_label="Don't show this again",
        )
        overlay.start()

    def _on_input_folder_changed(self, _text: str = "") -> None:
        """Clear the rerun state when the input folder changes.

        The last run's failed set holds basenames from the prior folder;
        offering to "rerun failed" against a different image universe
        would be wrong. Buttons hide on the next visibility refresh.
        """
        self._last_run_failed_set.clear()
        self._failure_category.clear()
        self._refresh_rerun_buttons()

    def _record_failures_from_run(self, results: list) -> None:
        """Populate _last_run_failed_set + refresh the rerun buttons.

        Combines manifest-tracked Stage-1 failures (failed_preproc_set)
        with Stage-2 (analysis) failures pulled from the in-memory
        results list. Analysis failures aren't recorded in the manifest
        today — Stage-2-failed images still get marked as completed for
        resume bookkeeping (see _on_image_completed) — so we read them
        from the worker's returned results to cover all categories.
        """
        manifest_failed: set[str] = set()
        manifest_analysis_failed: set[str] = set()
        if self._manifest is not None:
            try:
                manifest_failed = self._manifest.failed_preproc_set()
            except Exception:
                manifest_failed = set(getattr(self._manifest, "failed_preproc_images", []) or [])
            try:
                manifest_analysis_failed = self._manifest.analysis_failed_set()
            except Exception:
                # Older manifests may not have the analysis_failed_images
                # field at all; defaulting to empty is correct for that
                # case (those runs predate the field).
                manifest_analysis_failed = set(getattr(self._manifest, "analysis_failed_images", []) or [])
        # Stage-2 failures = a plain "analysis" error OR a garbage-filter abort (whose
        # error_stage is the specific filter label, e.g. "solidity"/"missing veins").
        from identify_features.garbage_detector import FILTER_LABELS

        _stage2_error_stages = {"analysis"} | set(FILTER_LABELS.values())
        analysis_failed: set[str] = {
            r.image_path.name
            for r in (results or [])
            if getattr(r, "error", None) and getattr(r, "error_stage", None) in _stage2_error_stages
        }
        self._last_run_failed_set = manifest_failed | manifest_analysis_failed | analysis_failed
        self._refresh_rerun_buttons()

    def _restore_post_run_state(self, manifest, run_folder: Path) -> None:
        """Repaint the GUI as if the worker had just finished ``manifest``.

        Drives both entry points (auto-prompt on launch and the
        Load-previous-run button). Without re-running the pipeline, this:

          - adopts ``manifest`` + ``run_folder`` so the rerun-failed paths
            (which look up self._manifest) work seamlessly;
          - points the input-folder edit at the manifest's saved input
            path and refreshes _image_paths / _basename_to_row so row
            lookups during state replay actually resolve;
          - replays each image's terminal state into _image_status,
            _image_error_text, _failure_category from the persisted
            manifest fields, then redraws the row labels via
            _update_image_status (which is how live failures get their
            tooltips today);
          - rebuilds _last_run_failed_set via _record_failures_from_run,
            which in turn calls _refresh_rerun_buttons — the Review +
            Rerun trio reappears with the correct counts.

        Output-section checkboxes and CSV-measurement groups are NOT
        restored: they're config for *running* the pipeline, not
        reviewing, and the user will adjust them deliberately if they
        decide to rerun anything. Leaving the current settings alone
        avoids surprising overwrites.
        """
        from TRACE.run_state import STATUS_COMPLETED

        # Block image-list signal handlers while we rebuild — the
        # input-folder change normally triggers Run-button enabling
        # heuristics that don't apply mid-restore.
        prior_input = self.input_edit.text()
        if manifest.input_dir and Path(manifest.input_dir).exists():
            self.input_edit.setText(manifest.input_dir)
        elif not prior_input:
            # Manifest references an input folder that's gone; show what
            # the manifest claims so the user can fix the path manually.
            self.input_edit.setText(manifest.input_dir)
        # The user's output folder (parent of run_<N>); if the manifest
        # lived at the top-level legacy location, run_folder IS the
        # output folder so .parent would back into the user's home —
        # use run_folder itself in that case.
        legacy_layout = (run_folder / "_run_state.json").is_file() and run_folder.name != ""
        if legacy_layout and not run_folder.name.startswith("run_"):
            self.output_edit.setText(str(run_folder))
        else:
            self.output_edit.setText(str(run_folder.parent))
        # Adopt the manifest. After this point, anything else that
        # reads self._manifest (rerun-failed flow, completion handlers)
        # sees the persisted state.
        self._manifest = manifest
        self._run_folder = run_folder

        # _refresh_image_list reads self.input_edit.text() so it picks
        # up the path we just set. Side effect: rebuilds _image_paths,
        # _basename_to_row, _row_base_labels — all required for
        # _update_image_status row lookups below.
        self._refresh_image_list()

        # Replay per-image state from the manifest.
        completed = set(manifest.completed_images or [])
        preproc_failed = set(manifest.failed_preproc_images or [])
        analysis_failed = set(manifest.analysis_failed_images or [])
        all_failed = preproc_failed | analysis_failed
        # Stage-2-failed images get marked completed too (see
        # _on_image_failed_analysis docstring), so flip them out of the
        # success-only set before painting.
        succeeded_only = completed - all_failed

        messages = manifest.failure_messages or {}
        for bn in succeeded_only:
            self._image_status[bn] = ImageStatus.SUCCEEDED
            self._update_image_status(bn, ImageStatus.SUCCEEDED)
        for bn in preproc_failed:
            msg = messages.get(bn, "")
            if msg:
                self._image_error_text[bn] = msg
            # Derive the category the same way the live failure slot
            # does. Older manifests stored no message → the regex sees
            # "" and falls into preproc_other, which is harmless (only
            # affects whether the "no gate aborts" rerun button shows
            # up — and for the user reviewing post-hoc, they'd see the
            # plain Rerun + Review buttons either way).
            self._failure_category[bn] = _classify_preproc_failure(msg)
            self._update_image_status(bn, ImageStatus.FAILED)
        for bn in analysis_failed:
            msg = messages.get(bn, "")
            if msg:
                self._image_error_text[bn] = msg
            # Same regex-on-message classification as the live failure
            # slot: garbage-detector quality aborts → "gate" so the
            # no-gates rerun button surfaces; everything else "analysis".
            self._failure_category[bn] = _classify_analysis_failure(msg)
            self._update_image_status(bn, ImageStatus.FAILED)

        # Tail end: rebuild _last_run_failed_set + paint the Review /
        # Rerun buttons. _record_failures_from_run consults the
        # manifest we just adopted and unions both failure lists.
        self._record_failures_from_run([])

        # Surface a one-line breadcrumb in the run log so the user
        # knows the restore happened (and can scroll back to it).
        try:
            self._log(f"--- Restored {len(all_failed)} failed image(s) from previous run at {run_folder} ---")
        except Exception:
            # Run-log widget isn't always there during early init.
            pass

        # Defensive: if find_completed_manifest mismatched the status
        # (shouldn't happen, but treat as best-effort), don't crash.
        if manifest.status != STATUS_COMPLETED:
            logging.getLogger(__name__).warning(
                "Restored manifest with non-completed status %s — UI is in a partially-restored state.",
                manifest.status,
            )

    def _maybe_offer_restore_post_run_state(self) -> None:
        """On launch, ask the user if they want to reload the previous run.

        Only fires when (a) the saved output folder has a COMPLETED
        manifest with at least one unreviewed failed image, AND (b) no
        run is currently in progress (defensive — auto-prompt runs
        once at init, before the user can start anything). Decline is
        silent; the Load-previous-run button at the bottom of the left
        column is the deliberate fallback for "I dismissed but changed
        my mind" and for picking older runs.
        """
        if self.worker is not None and self.worker.isRunning():
            return
        output_text = self.output_edit.text().strip()
        if not output_text:
            return
        output_dir = Path(output_text)
        if not output_dir.is_dir():
            return
        try:
            from TRACE.run_state import find_completed_manifest

            found = find_completed_manifest(output_dir)
        except Exception:
            return
        if found is None:
            return
        manifest, run_folder = found
        failed_count = len(set(manifest.failed_preproc_images or []) | set(manifest.analysis_failed_images or []))
        if failed_count <= 0:
            return  # defensive — find_completed_manifest already filters this
        reply = QMessageBox.question(
            self,
            "Reload previous run?",
            (
                f"Your last run in {output_dir} left {failed_count} image(s) flagged as failed "
                f"that you may not have reviewed yet.\n\n"
                "Reload that view so you can open the inspector on them?"
            ),
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.Yes,
        )
        if reply != QMessageBox.Yes:
            return
        self._restore_post_run_state(manifest, run_folder)
        # Open the inspector immediately — that's the user's actual
        # goal. _review_failed_images warns + bails if it can't resolve
        # paths, so a missing-input case is handled without an extra
        # check here.
        self._review_failed_images()

    def _load_previous_run_dialog(self) -> None:
        """Picker for the Load-previous-run button below the Run row.

        Lets the user point at any output folder (current or otherwise)
        and resurface its failed-image review state. Used for older
        runs that aren't the most recent, or for runs in a different
        output folder than the one the auto-prompt watches.
        """
        if self.worker is not None and self.worker.isRunning():
            QMessageBox.information(
                self,
                "Load previous run",
                "Wait for the current run to finish before loading a previous one.",
            )
            return
        seed = self.output_edit.text().strip() or str(Path.home())
        _open_native_picker_async(
            self,
            "Pick the output folder of a previous run",
            seed,
            self._on_load_previous_run_picked,
            folder=True,
            last_dir_key="load_previous_run",
            sync=True,
        )

    def _on_load_previous_run_picked(self, folder: str) -> None:
        if not folder:
            return
        try:
            from TRACE.run_state import find_completed_manifest

            found = find_completed_manifest(Path(folder))
        except Exception as exc:
            QMessageBox.warning(
                self,
                "Load previous run",
                f"Couldn't read run-state from {folder}:\n{type(exc).__name__}: {exc}",
            )
            return
        if found is None:
            QMessageBox.information(
                self,
                "Load previous run",
                (
                    f"No completed run with failed images was found under {folder}.\n\n"
                    "A reloadable run needs (a) a finished TRACE run in that folder "
                    "and (b) at least one image flagged as failed."
                ),
            )
            return
        manifest, run_folder = found
        self._restore_post_run_state(manifest, run_folder)
        self._review_failed_images()

    def _start_rerun_failed(self, *, disable_gates: bool) -> None:
        """Launch a rerun scoped to the last run's failed set.

        Inverts the user's failed_set into a skip_image_basenames arg so
        the existing pipeline code path needs no changes. When
        ``disable_gates`` is True, stashes one-shot pending overrides
        (``_pending_gate_override`` + ``_pending_disable_garbage_filters``)
        that ``_run_pipeline`` splices into the worker kwargs for this
        one launch — without mutating self.config or self._gate_override.
        That matters because _save_settings(...) inside _run_pipeline
        persists those two attributes to QSettings, so any in-place flip
        would survive across a TRACE restart. The overrides cover both
        landmark confidence gates and the four garbage-detector filters
        (solidity, fragmentation, vein_association, vein_presence).
        Prompts the user for append-to-existing-CSV vs. fresh CSV first.
        """
        if not self._last_run_failed_set:
            return  # defensive — button shouldn't be visible
        if self.worker is not None and self.worker.isRunning():
            QMessageBox.information(self, "TRACE", "A run is already in progress.")
            return
        # Sanity: if the input folder changed since the failed run, the
        # current _image_paths universe doesn't match _last_run_failed_set.
        all_image_names = {p.name for p in self._image_paths}
        if not self._last_run_failed_set.issubset(all_image_names):
            QMessageBox.warning(
                self,
                "Cannot rerun",
                "The input folder doesn't contain all the images that failed in the "
                "last run. Either re-point at the original folder or start a fresh run.",
            )
            return

        csv_choice = self._prompt_csv_branch()
        if csv_choice is None:
            return  # user cancelled

        # IMPORTANT: do NOT mutate self.config or self._gate_override here.
        # _run_pipeline calls _save_settings() right after launch, which
        # writes self.config + self._gate_override to QSettings — if we
        # flipped those flags in-place, the disabled state would persist
        # to disk and a TRACE restart would silently keep gates off.
        # Instead, stash one-shot overrides for _run_pipeline to splice
        # into the worker kwargs only for this launch. Both fields reset
        # to falsy defaults at the top of _run_pipeline so the next
        # normal Run inherits nothing.
        self._pending_gate_override = self._build_all_gates_disabled_override() if disable_gates else None
        self._pending_disable_garbage_filters = bool(disable_gates)

        # Stash the override-CSV filename on self so _run_pipeline can read
        # it; cleared back to None at the start of every _run_pipeline call.
        self._pending_csv_filename_override = csv_choice or None
        # Override the skip set for this single launch — _run_pipeline's
        # normal computation rebuilds from _resume_skip_set + _user_skip_set,
        # which doesn't capture "process only these". Stash an inverted-
        # set override that _run_pipeline picks up if present.
        self._pending_rerun_skip_set = all_image_names - self._last_run_failed_set
        # Suppress the parallel-workers warning on a rerun — the user
        # just ran with this same setting; re-confirming is friction.
        self._pending_skip_workers_warning = True
        self._run_pipeline()

    def _prompt_csv_branch(self):
        """Append-vs-new dialog. Returns "" (append), filename, or None (cancel)."""
        from datetime import datetime as _dt

        box = QMessageBox(self)
        box.setIcon(QMessageBox.Question)
        box.setWindowTitle("Rerun failed images")
        box.setText("Where should the new measurements go?")
        box.setInformativeText(
            "Append: new rows are merged into measurements.csv alongside the existing rows.\n\n"
            "New CSV: a separate measurements_rerun_<timestamp>.csv is written; the original is untouched."
        )
        btn_append = box.addButton("Append to existing CSV", QMessageBox.AcceptRole)
        btn_new = box.addButton("Write to new CSV", QMessageBox.AcceptRole)
        box.addButton(QMessageBox.Cancel)
        box.setDefaultButton(btn_append)
        box.exec_()
        clicked = box.clickedButton()
        if clicked is btn_append:
            return ""  # signals "use the default measurements.csv name; merge via resume path"
        if clicked is btn_new:
            ts = _dt.now().strftime("%Y%m%d_%H%M%S")
            return f"measurements_rerun_{ts}.csv"
        return None  # Cancel

    def _build_all_gates_disabled_override(self) -> dict:
        """Build a gate_override that disables every confidence-gate metric.

        Schema matches what GateConfigPanel emits in LandmarkLocator's
        scripts/gui.py:1365-1386: each of peak / sharpness /
        second_peak_ratio is a top-level key with an ``enabled`` flag.
        We preserve any existing per-landmark / global thresholds the
        user had configured — only the ``enabled`` bool flips. That way
        re-enabling later (by clearing the temporary override) restores
        the user's original tunings.
        """
        import copy as _copy

        base = _copy.deepcopy(self._gate_override) if self._gate_override else {}
        for metric in ("peak", "sharpness", "second_peak_ratio"):
            bucket = base.setdefault(metric, {})
            bucket["enabled"] = False
        return base

    def run_single_image_preprocessing_for_segmentation(self, image_path, output_dir, *, with_segmentation: bool):
        """Preprocess ONE image with the current GUI settings for the inspector.

        The vein/intervein model only works on the fully-preprocessed image
        (wing-isolated + hinge-removed, rescaled if a scale is set), so the
        Veins/Interveins tab calls this to run that exact chain on demand
        instead of segmenting the raw input. Mirrors the batch run's
        preprocessing config EXCEPT:
          - rotation is forced OFF, so the segmentation + chopped image stay in
            the pre-rotation pixel space the Stage-5 override is consumed in;
          - landmark gates are disabled, so a borderline wing doesn't abort
            before segmentation. Any saved landmark override is still honored by
            Stage 3, so the hinge chop uses the user's corrected landmarks.

        Returns a preprocessing PipelineResult; the inspector reads
        ``chopped_image_path`` (the exact image the segmentation model saw),
        ``segmentation_geojson_path`` and ``rescale_factor`` from it. The mask is
        shown/edited over that preprocessed image and the saved override is
        divided back into original-image space for the Stage-5 sidecar.
        """
        from preprocessing.pipeline import process_single_image

        wing_model_dir = None
        if self._wing_isolation_enabled and str(self._wing_isolation_model_path or "").strip():
            wing_model_dir = Path(self._wing_isolation_model_path)

        if self._active_rescale_target == "landmark":
            target_um_per_px = self._landmark_target_um_per_px
        elif self._active_rescale_target == "wing_isolation":
            target_um_per_px = self._wing_isolation_target_um_per_px
        else:
            target_um_per_px = self._segmentation_target_um_per_px

        return process_single_image(
            image_path=Path(image_path),
            output_dir=Path(output_dir),
            landmark_checkpoint=Path(self._landmark_model_path) if self._landmark_model_path else None,
            segmentation_model_dir=Path(self._segmentation_model_path) if self._segmentation_model_path else None,
            stages=(True, True, bool(with_segmentation)),
            include_unreliable_landmarks=self._include_unreliable_landmarks,
            wing_model_dir=wing_model_dir,
            wing_expand_fraction=self._wing_expand_fraction,
            do_rotation=False,
            gate_override=self._build_all_gates_disabled_override(),
            input_um_per_px=self.config.um_per_px,
            target_um_per_px=target_um_per_px,
            rescale_tolerance_low=self._rescale_tolerance_low,
            rescale_tolerance_high=self._rescale_tolerance_high,
        )

    # -----------------------------------------------------------------------
    # Pipeline execution
    # -----------------------------------------------------------------------
    def _run_pipeline(self):
        # Consume any one-shot pending-launch state set by
        # _start_rerun_failed (rerun via the failed-images buttons).
        # Reset back to defaults at the same time so a normal Run click
        # never inherits a rerun's overrides.
        rerun_skip_set = self._pending_rerun_skip_set
        csv_filename_override = self._pending_csv_filename_override
        skip_workers_warning = self._pending_skip_workers_warning
        # No-quality-gates rerun overlay — applied to the worker kwargs
        # below without touching self.config / self._gate_override so the
        # user's saved settings stay clean on next launch.
        pending_gate_override = self._pending_gate_override
        disable_garbage_filters = self._pending_disable_garbage_filters
        self._pending_rerun_skip_set = None
        self._pending_csv_filename_override = None
        self._pending_skip_workers_warning = False
        self._pending_gate_override = None
        self._pending_disable_garbage_filters = False
        is_rerun = rerun_skip_set is not None
        # A fresh Run click starts a new failure-tracking slate. Reruns
        # keep their _last_run_failed_set so the buttons stay visible
        # after the rerun (in case the rerun itself leaves some images
        # still failing, supporting a chain of progressively narrower
        # reruns).
        if not is_rerun:
            self._last_run_failed_set.clear()
            self._failure_category.clear()
        # Either way, hide the rerun buttons while the run is in flight.
        self.btn_rerun_failed.setVisible(False)
        self.btn_rerun_failed_nogate.setVisible(False)
        # Validate required fields
        if not self.input_edit.text():
            QMessageBox.warning(self, "Missing Input", "Please select an input folder.")
            return
        if not self.output_edit.text():
            QMessageBox.warning(self, "Missing Output", "Please select an output folder.")
            return
        if not self._landmark_model_path:
            QMessageBox.warning(self, "Missing Model", "Please select a landmark model in Settings → Models.")
            return
        if not self._segmentation_model_path:
            QMessageBox.warning(
                self, "Missing Model", "Please select a segmentation model folder in Settings → Models."
            )
            return
        # Scale validation branches on the per-image-metadata flag:
        #   - flag off  → single manual µm/px is required (legacy behaviour).
        #   - flag on   → manual µm/px is OPTIONAL, but if it's empty AND any
        #                 NON-SKIPPED image lacks parseable metadata, pre-flight
        #                 aborts here. Skipped images (user-unchecked, resume
        #                 cache hits, rerun's inverted set) never run so their
        #                 missing metadata must NOT block Run — we mirror the
        #                 same skip-set computation the worker uses below.
        if bool(getattr(self.config, "auto_detect_um_per_px", False)):
            preflight_skips = (
                rerun_skip_set if rerun_skip_set is not None else (self._resume_skip_set or set()) | self._user_skip_set
            )
            if not self._preflight_check_per_image_scale(effective_skips=preflight_skips):
                return
        elif self.config.um_per_px is None:
            QMessageBox.warning(
                self,
                "Missing Scale",
                "Please enter a µm/px conversion factor in the Scale field before running, "
                "or enable 'Detect scale from image metadata' to read the scale "
                "from each image's TIFF metadata.",
            )
            return

        # All-skipped guard: if every image is on a skip list (user
        # untick + resume cache combined), the pipeline would run a
        # no-op. Surface that intent before launching the worker so
        # the user doesn't watch an empty progress bar.
        if self._image_paths:
            effective_skips = (self._resume_skip_set or set()) | self._user_skip_set
            if all(p.name in effective_skips for p in self._image_paths):
                QMessageBox.warning(
                    self,
                    "All images skipped",
                    "Every image in the input folder is either marked to skip or already "
                    "completed in a prior run — there's nothing to process. Uncheck at "
                    "least one row (or right-click the list → Unskip all) and try again.",
                )
                return

        # Warn on JPEG inputs (lossy compression)
        jpg_images = [p for p in self._image_paths if p.suffix.lower() in (".jpg", ".jpeg")]
        if jpg_images:
            box = QMessageBox(self)
            box.setIcon(QMessageBox.Warning)
            box.setWindowTitle("JPEG Input Detected")
            box.setTextFormat(Qt.RichText)
            box.setTextInteractionFlags(Qt.TextBrowserInteraction)
            box.setText(
                "CAUTION: You are using .jpg image(s). Due to the limited amount of information "
                "provided by this file type, TRACE may not function as intended."
                "<br><br>Read about why: "
                '<a href="https://pmc.ncbi.nlm.nih.gov/articles/PMC4210356/">'
                "https://pmc.ncbi.nlm.nih.gov/articles/PMC4210356/</a>"
            )
            box.setStandardButtons(QMessageBox.Ok | QMessageBox.Abort)
            box.setDefaultButton(QMessageBox.Abort)
            if box.exec_() == QMessageBox.Abort:
                return

        # Reruns skip the parallel-workers warning — the user just ran with
        # this same setting, re-confirming would be friction.
        if not skip_workers_warning and not self._confirm_parallel_workers():
            return

        # OOD preflight: sample a few input images and compare per-channel pixel
        # stats against each enabled model's training distribution. Catches the
        # obvious failure modes (wrong stain, exposure blowup, missing channel)
        # before the user commits to a full run. Models without metadata.json
        # are silently skipped.
        outputs_now = self._selected_outputs()
        ood_needs_lm, _, ood_needs_seg = _required_stages(outputs_now)
        wing_model_dir = None
        if self._wing_isolation_enabled and self._wing_isolation_model_path.strip():
            wing_model_dir = Path(self._wing_isolation_model_path)
        ood_models: dict[str, Path] = {}
        if ood_needs_lm and self._landmark_model_path:
            ood_models["landmark"] = Path(self._landmark_model_path)
        if ood_needs_seg and self._segmentation_model_path:
            ood_models["vein/intervein"] = Path(self._segmentation_model_path)
        if wing_model_dir is not None:
            ood_models["wing isolation"] = wing_model_dir
        if ood_models and self._image_paths:
            try:
                ood_reports = preflight_batch(self._image_paths, ood_models, n_sample=3)
            except Exception as exc:
                logger.warning("OOD preflight raised: %s", exc)
                ood_reports = {}
            flagged = {n: r for n, r in ood_reports.items() if r.has_warnings}
            for _name, report in flagged.items():
                self._log(format_report_line(report))
            if flagged and not self._confirm_ood_warnings(flagged):
                return

        # Resume support: if the output folder has an in-progress manifest
        # from a prior run, prompt the user. Returns the set of basenames
        # to skip (empty for "start fresh") and the manifest object (None
        # if user opted to start fresh / there was no manifest). Returns
        # (None, None) when the user cancelled the dialog entirely.
        outputs_for_run = self._selected_outputs()
        csv_groups_for_run = {gkey for gkey, gchk in self.csv_group_checks.items() if gchk.isChecked()}
        resume_skip_set, self._manifest = self._maybe_offer_resume(
            output_dir=Path(self.output_edit.text()),
            outputs=outputs_for_run,
            csv_groups=csv_groups_for_run,
        )
        if resume_skip_set is None:
            # User cancelled; bail without starting.
            return

        # UI state — btn_run stays enabled but flips its label to Pause
        # while the worker is going. Cancel becomes available.
        self.btn_run.setEnabled(True)
        self.btn_run.setText("Pause")
        self.btn_cancel.setEnabled(True)
        # Defensive: any leftover transient status from a prior slice
        # (e.g. a "Pause requested…" that somehow stuck) gets cleared
        # at the start of a fresh run / resume.
        self.transient_status_label.hide()
        self._is_paused = False
        self._resume_skip_set = resume_skip_set
        self._progress_pct_high = 0
        self.progress.setValue(0)
        # Restore the default Highlight color (the bar turns green on a clean
        # finish — reset it here so the new run starts in the default color).
        pal = self.progress.palette()
        pal.setColor(QPalette.Highlight, self._progress_default_highlight)
        self.progress.setPalette(pal)
        self._run_start_time = time.monotonic()
        # Wall-clock start so the per-run metadata folder gets a
        # human-readable name. Captured here (not in __init__) because
        # we only need it for the duration of a run.
        from datetime import datetime as _dt

        self._run_start_wall = _dt.now()
        self._eta_smoothed_seconds = None
        self._current_stage = None
        self._stage_first_event_time = None
        self._stage_completions = 0
        self._stage_total = 0
        self._last_completion_time = None
        self._avg_time_per_image = None
        self.eta_label.setText("Estimating time until pipeline finishes…")
        self._progress_timer.start()
        # Pre-compute the Stage1/Stage2 wall-time weight split for the chosen
        # outputs so per-image progress events land on a unified 0–100 scale
        # that reflects relative wall-time cost (Stage 2 dominates by ~5×).
        outputs_now = self._selected_outputs()
        self._progress_stage1_share, self._progress_stage2_share = compute_progress_weights(
            outputs_now,
            wing_isolation_enabled=self._wing_isolation_enabled,
            skip_intervein_regions=getattr(self.config, "skip_intervein_regions", False),
        )
        # Preserve the existing log when resuming so the user can see what
        # happened in the prior slice. Fresh runs always start clean.
        if not self._is_resuming:
            self.log_text.clear()
        self._save_settings()
        if self._is_resuming:
            self._log("Resuming run.")
        else:
            self._log("Starting TRACE pipeline...")
        # Emit the active configuration as a YAML block right after the
        # banner. Done on every fresh-run AND resume so the log captures
        # any settings drift between slices in-place.
        self._log_run_settings()

        # Initialize per-image status. Single rule that covers fresh
        # runs, resumes, and rerun-failed launches:
        #   - A SUCCEEDED / FAILED row keeps its final glyph IFF the
        #     worker is going to skip this image in the upcoming run.
        #     That way the prior-run result the user just saw isn't
        #     overwritten with PENDING for an image that won't be
        #     reprocessed.
        #   - Otherwise reset based on skip-set membership:
        #     USER_SKIPPED > resume SKIPPED > PENDING.
        # Walking through the cases:
        #   - Fresh Run click: effective_skip_set is just the user-skip
        #     set. SUCCEEDED/FAILED rows from a prior run within the
        #     same session reset to PENDING because the worker is
        #     about to revisit them. (The "image list still shows old
        #     check marks on rerun" bug came from skipping this reset.)
        #   - Resume: resume_skip_set contains every previously-
        #     completed image, so all SUCCEEDED/FAILED rows are
        #     preserved — same end-state as before.
        #   - Rerun-failed: rerun_skip_set covers all images EXCEPT the
        #     failed ones being rerun. Green checks on the successful
        #     ones stay; the failed ones reset to PENDING because the
        #     worker is about to reprocess them.
        effective_skip_set = (
            rerun_skip_set if rerun_skip_set is not None else (self._resume_skip_set or set()) | self._user_skip_set
        )
        for basename in list(self._basename_to_row):
            current = self._image_status.get(basename)
            if current in (ImageStatus.SUCCEEDED, ImageStatus.FAILED) and basename in effective_skip_set:
                continue
            if basename in self._user_skip_set:
                initial = ImageStatus.USER_SKIPPED
            elif basename in self._resume_skip_set:
                initial = ImageStatus.SKIPPED
            else:
                initial = ImageStatus.PENDING
            self._update_image_status(basename, initial)

        # wing_model_dir was resolved above as part of the OOD preflight.

        # Resolve the active per-model target. When the user hasn't entered one
        # for the active model, leave target_um_per_px None — preprocessing's
        # Stage 1 then no-ops cleanly.
        target_um_per_px: Optional[float]
        if self._active_rescale_target == "landmark":
            target_um_per_px = self._landmark_target_um_per_px
        elif self._active_rescale_target == "wing_isolation":
            target_um_per_px = self._wing_isolation_target_um_per_px
        else:
            target_um_per_px = self._segmentation_target_um_per_px

        # No-quality-gates rerun: build a deep copy of self.config with the
        # four garbage-detector flags flipped off, just for this run. The
        # original self.config is untouched, so _save_settings(...) below
        # (and the QSettings write on TRACE close) persists the user's saved
        # values — restarting TRACE doesn't inherit the temporarily-disabled
        # state. Same shape of one-shot overlay we use for gate_override.
        run_config = self.config
        if disable_garbage_filters:
            import copy as _copy

            run_config = _copy.deepcopy(self.config)
            run_config.solidity_filter_enabled = False
            run_config.fragmentation_filter_enabled = False
            run_config.vein_association_filter_enabled = False
            run_config.required_veins = []
        run_gate_override = pending_gate_override if pending_gate_override is not None else self._gate_override

        self.worker = TraceWorker(
            kwargs=dict(
                input_dir=Path(self.input_edit.text()),
                output_dir=Path(self.output_edit.text()),
                landmark_checkpoint=Path(self._landmark_model_path),
                segmentation_model_dir=Path(self._segmentation_model_path),
                config=run_config,
                keep_intermediates=False,
                outputs=self._selected_outputs(),
                max_workers=self.inline_general_panel.workers_spin.value(),
                show_vein_tissue=self._show_vein_tissue,
                show_color_key=self._show_color_key,
                show_ectopic_labels=self._show_ectopic_labels,
                show_region_labels=self._show_region_labels,
                show_landmark_labels=self._show_landmark_labels,
                vein_simplify_tolerance_px=self._vein_simplify_tolerance_px,
                ectopic_label_font_scale=self._ectopic_label_font_scale,
                landmark_size_scale=self._landmark_size_scale,
                show_compartment_labels=self._show_compartment_labels,
                include_unreliable_landmarks=self._include_unreliable_landmarks,
                gate_override=run_gate_override,
                wing_isolation_model_dir=wing_model_dir,
                wing_expand_fraction=self._wing_expand_fraction,
                recursive=self.recursive_chk.isChecked(),
                do_rotation=self._do_rotation,
                rotation_mirror_correct=self._rotation_mirror_correct,
                user_landmark_distances=(
                    list(self._user_landmark_distances) if self.include_custom_measurements_chk.isChecked() else []
                ),
                csv_measurement_groups={gkey for gkey, gchk in self.csv_group_checks.items() if gchk.isChecked()},
                # Tie landmark batch size to the Workers spinbox so a single setting
                # controls Stage 1 batching and Stage 2 parallelism together.
                landmark_batch_size=self.inline_general_panel.workers_spin.value(),
                target_um_per_px=target_um_per_px,
                rescale_tolerance_low=self._rescale_tolerance_low,
                rescale_tolerance_high=self._rescale_tolerance_high,
                # Skip-set source of truth: rerun launches stash an
                # inverted set ({all_images} - {failed_set}) into
                # rerun_skip_set; normal launches merge resume-skip with
                # user-driven skips. The rerun set fully replaces the
                # normal merge because rerun semantically means "process
                # ONLY these" — including a successful prior image
                # would re-process it for no reason.
                skip_image_basenames=(
                    rerun_skip_set
                    if rerun_skip_set is not None
                    else (self._resume_skip_set or set()) | self._user_skip_set
                ),
                # Rerun-via-failed-images can route results to a fresh
                # measurements_rerun_<ts>.csv instead of merging into the
                # existing measurements.csv. None = default behavior.
                csv_filename_override=csv_filename_override,
            )
        )
        self.worker.progress.connect(self._on_progress)
        self.worker.log_message.connect(self._log)
        self.worker.all_done.connect(self._on_all_done)
        self.worker.paused.connect(self._on_paused)
        self.worker.cancelled.connect(self._on_cancelled)
        self.worker.image_completed.connect(self._on_image_completed)
        self.worker.image_failed_preproc.connect(self._on_image_failed_preproc)
        self.worker.image_failed_analysis.connect(self._on_image_failed_analysis)
        self.worker.error.connect(self._on_error)
        # Lock per-row check toggles for the duration of the run — the
        # worker captured the skip set above, so mid-run toggles would
        # silently no-op until the next run. _on_all_done / _on_paused /
        # _on_cancelled / _on_error each re-enable.
        self._set_skip_checkboxes_enabled(False)
        # Initialize the manifest now (after a possible resume merge), so
        # the first image completion already has somewhere to write.
        out_dir = Path(self.output_edit.text())
        if self._manifest is None:
            # Fresh run — make a new run_<timestamp>/ folder to hold the
            # manifest, settings snapshot, and run.log. Timestamp matches
            # _run_start_wall above so the folder name is human-readable.
            stamp = self._run_start_wall.strftime("%Y%m%d-%H%M%S")
            self._run_folder = out_dir / f"run_{stamp}"
            self._run_folder.mkdir(parents=True, exist_ok=True)
            total = max(1, len(self._image_paths))
            self._manifest = new_manifest(
                input_dir=Path(self.input_edit.text()),
                recursive=self.recursive_chk.isChecked(),
                outputs_selected=outputs_for_run,
                csv_measurement_groups=csv_groups_for_run,
                total_images=total,
                settings_snapshot_path="settings.yaml",
            )
            # Snapshot current settings into the run folder. Future resumes
            # diff against this; settings_partN.yaml gets written when the
            # user resumes with changed settings.
            self._write_settings_snapshot(self._run_folder / "settings.yaml")
        else:
            # Resume path: keep the existing completed list, mark running.
            # self._run_folder was populated by _maybe_offer_resume.
            self._manifest.status = STATUS_RUNNING
        save_manifest(self._run_folder, self._manifest)
        self.worker.start()

    def _confirm_ood_warnings(self, flagged: dict) -> bool:
        """Show the OOD warning dialog; return True if user wants to proceed."""
        lines: list[str] = []
        for _name, report in flagged.items():
            lines.append(format_report_line(report))
            for d in report.deviations[:5]:
                if d.metric == "missing_channel":
                    lines.append(f"    • channel {d.channel}: missing from image")
                else:
                    lines.append(
                        f"    • channel {d.channel} {d.metric}: image={d.image_value:.1f} "
                        f"vs training={d.training_value:.1f} ± {d.training_std:.1f}"
                    )
            extra = len(report.deviations) - 5
            if extra > 0:
                lines.append(f"    ... and {extra} more")
        body = (
            "These input images look unusual compared to what the models were trained on. "
            "Results may be unreliable. Common causes: wrong file type, fluorescence vs "
            "brightfield, dramatically different exposure, missing color channel.\n\n" + "\n".join(lines)
        )
        box = QMessageBox(self)
        box.setIcon(QMessageBox.Warning)
        box.setWindowTitle("Input images look out-of-distribution")
        box.setText(body)
        box.setStandardButtons(QMessageBox.Ok | QMessageBox.Abort)
        box.setDefaultButton(QMessageBox.Abort)
        return box.exec_() == QMessageBox.Ok

    def _on_run_button_clicked(self):
        """Tri-state dispatch for the single Run / Pause / Resume button.

        - Worker is running → Pause it. The worker finishes the current
          image cleanly and emits paused(...) when done. _on_paused then
          flips the button label to Resume.
        - Worker is None and self._is_paused is True → Resume by
          spawning a fresh worker that continues from the manifest's
          completed list.
        - Otherwise (idle) → kick off a new run via _run_pipeline.
        """
        if self.worker is not None:
            self.worker.pause()
            msg = "Pause requested — finishing the current image first."
            self._log(msg)
            # Persistent transient label under the ETA so the message
            # doesn't get buried in the scrolling log while the in-flight
            # image finishes. Cleared in _on_paused.
            self.transient_status_label.setText(msg)
            self.transient_status_label.show()
            # Disable during the transition (between click and the worker's
            # paused-ack) so the user can't spam-click and queue weird state.
            self.btn_run.setEnabled(False)
            return
        if self._is_paused:
            self._resume_paused_run()
            return
        if not self._confirm_no_pending_settings_edits():
            return
        self._run_pipeline()

    def _confirm_no_pending_settings_edits(self) -> bool:
        """Warn the user if the Advanced Settings dialog is open when Run
        is clicked — unapplied edits there won't take effect until OK.

        Returns True if the run should proceed, False if the user wants
        to go back to the dialog first. Always True when the dialog
        isn't open. Tracking "actual edits" vs. "dialog merely open" is
        not worth the per-widget dirty-flag plumbing — the warning fires
        whenever the dialog is visible, since the user already knows
        whether they made changes and the cost of an extra confirmation
        click is small compared to running with the wrong settings.
        """
        dlg = getattr(self, "_settings_dialog", None)
        if dlg is None or not dlg.isVisible():
            return True
        box = QMessageBox(self)
        box.setIcon(QMessageBox.Warning)
        box.setWindowTitle("Advanced Settings still open")
        box.setText("You have the Advanced Settings dialog open.")
        box.setInformativeText(
            "Any unapplied changes there won't affect this run — settings only "
            "apply after you click OK in the dialog.\n\n"
            "Run anyway, or go back to the settings dialog?"
        )
        btn_run = box.addButton("Run with current settings", QMessageBox.AcceptRole)
        btn_back = box.addButton("Go to settings", QMessageBox.RejectRole)
        box.setDefaultButton(btn_back)
        box.exec_()
        if box.clickedButton() is btn_run:
            return True
        # User chose to go back — raise the dialog to the front.
        dlg.raise_()
        dlg.activateWindow()
        return False

    def _cancel_pipeline(self) -> None:
        """Hard-stop the run and discard its resume state.

        Confirms with the user (because canceling can't be undone — the
        manifest gets marked cancelled so the next Run starts fresh).
        Per-image outputs already written stay on disk; the user can
        delete them manually if desired.
        """
        n_done = len(self._manifest.completed_images) if self._manifest is not None else 0
        n_total = self._manifest.total_images if self._manifest is not None else 0
        box = QMessageBox(self)
        box.setIcon(QMessageBox.Warning)
        box.setWindowTitle("Cancel run?")
        box.setText(
            f"Cancel this run and discard its resume state? " f"({n_done} of {n_total} images completed so far.)"
        )
        box.setInformativeText(
            "Per-image outputs already written to disk are kept. The next Run " "starts fresh — no Resume prompt."
        )
        box.setStandardButtons(QMessageBox.Yes | QMessageBox.No)
        box.setDefaultButton(QMessageBox.No)
        if box.exec_() != QMessageBox.Yes:
            return
        # Paused path: no worker thread is running, so we can finalize
        # immediately. No torch ops are in flight to corrupt.
        if self.worker is None:
            self._finalize_cancel()
            return
        # Running path: cooperative cancel only. Do NOT call terminate() —
        # killing a thread mid-torch-MPS op corrupts MPSGraphCache /
        # MetalShaderLibrary internal state, which then crashes the *next*
        # run when it touches MPS (observed: SIGSEGV in
        # MetalShaderLibrary::getLibraryPipelineState during hash rehash).
        # Instead: flip the cancel flag, mute the UI-facing signals so the
        # display freezes immediately, and let the worker exit at the next
        # per-image progress checkpoint. The worker's `cancelled` /
        # `all_done` / `error` signal lands in its handler, which routes
        # through _finalize_cancel because _cancel_requested is set.
        self._cancel_requested = True
        # Mute the chatty stream signals so the UI freezes immediately, but
        # leave the per-image completion signals connected. The image that
        # was mid-flight when Cancel was clicked finishes cleanly (artifacts
        # are written inside the analyze try-block before the next progress
        # tick raises InterruptedError); we want its row to land on
        # SUCCEEDED / FAILED, not the misleading SKIPPED that
        # _mark_incomplete_as_skipped would otherwise paint over the
        # still-IN_PROGRESS row.
        for sig in (
            self.worker.progress,
            self.worker.log_message,
        ):
            try:
                sig.disconnect()
            except (TypeError, RuntimeError):
                pass
        self.worker.cancel()
        # Lock the UI into Cancelling state. Both buttons disabled so the
        # user can't double-cancel or start a parallel run (two concurrent
        # TraceWorkers is the bug we're avoiding). _on_cancelled (or the
        # all_done/error fallbacks) re-enables Run when the worker exits.
        self.btn_cancel.setEnabled(False)
        self.btn_run.setEnabled(False)
        self.btn_run.setText("Cancelling…")
        self.transient_status_label.setText("Cancel requested — finishing the current image first.")
        self.transient_status_label.show()
        self.statusBar().showMessage("Cancelling…")

    def _finalize_cancel(self, results: Optional[list] = None) -> None:
        """Shared cleanup for both running-then-cancelled and paused-then-
        cancelled paths. Marks the manifest cancelled, flushes the log,
        and resets the UI to idle.

        ``results`` is the worker's emitted result list when available
        (passed through by _on_cancelled / _on_all_done's cancel branch).
        Used to capture Stage-2 (analysis) failures for the rerun buttons
        BEFORE the manifest reference is nulled below.
        """
        from TRACE.run_state import STATUS_CANCELLED

        # Discard semantics: every unfinished image becomes SKIPPED, not
        # PENDING, because the run is dead — a fresh Run is the only way
        # those images get processed now.
        self._mark_incomplete_as_skipped()
        self.transient_status_label.hide()
        # Capture failed-image set for the rerun buttons while the
        # manifest is still readable.
        self._record_failures_from_run(results or [])

        if self._manifest is not None and self._run_folder is not None:
            self._manifest.status = STATUS_CANCELLED
            save_manifest(self._run_folder, self._manifest)
        from datetime import datetime as _dt

        self._log(f"--- Cancelled at {_dt.now().isoformat(timespec='seconds')} ---")
        self._log("Run cancelled — resume state discarded.")
        self._write_run_metadata(status="cancelled")
        self._manifest = None
        self._is_paused = False
        self._is_resuming = False
        self._run_folder = None
        self.btn_run.setEnabled(True)
        self.btn_run.setText("Run Pipeline")
        self.btn_cancel.setEnabled(False)
        # Recolor the progress-bar fill red as a visual cue. _run_pipeline
        # resets it to the default highlight at the start of the next run.
        from TRACE.theme import current_theme as _ct

        pal = self.progress.palette()
        pal.setColor(QPalette.Highlight, QColor(_ct().cancel_highlight))
        self.progress.setPalette(pal)
        self.statusBar().showMessage("Cancelled")
        self.eta_label.setText("")
        self._progress_timer.stop()
        self._cancel_requested = False
        # Un-lock the per-row check toggles now that the run's done.
        self._set_skip_checkboxes_enabled(True)

    def _on_cancelled(self, results) -> None:
        """Worker exited via TraceWorker.cancelled (user clicked Cancel
        while the run was running). Flow into the shared finalizer.
        ``results`` is forwarded so any partial Stage-2 failures get
        picked up for the rerun-failed buttons.
        """
        self.worker = None
        self._finalize_cancel(results)

    def _on_image_completed(self, basename: str) -> None:
        """One image's Stage 2 succeeded — record + visually mark done.

        Called on the GUI thread via TraceWorker.image_completed signal,
        so no locking needed. Save-after-each-image is wasteful for very
        large batches but is the simplest way to keep the manifest in
        sync with on-disk artifacts; cost is one ~1 KB JSON write per
        image, dwarfed by the per-image overlay PNGs.

        Note: _signal_complete (TRACE/pipeline.py) fires image_completed for
        BOTH outcomes — on failure it emits image_failed_analysis first (which
        marks the row FAILED) and then image_completed for manifest bookkeeping.
        So this handler must do the manifest write but must NOT downgrade a row
        already marked FAILED back to SUCCEEDED — otherwise a Stage-2 abort
        (e.g. a garbage-filter rejection) would show a green check instead of
        the red ✗ that gate/preproc failures get.
        """
        if self._manifest is not None and self._run_folder is not None:
            self._manifest.mark_completed(basename)
            save_manifest(self._run_folder, self._manifest)
        if self._image_status.get(basename) == ImageStatus.FAILED:
            return
        self._update_image_status(basename, ImageStatus.SUCCEEDED)

    def _on_image_failed_preproc(self, basename: str, error_text: str = "") -> None:
        """One image's Stage 1 errored — record it for resume bookkeeping.

        On a resume with unchanged settings these get folded into the
        skip set so Stage 1 doesn't re-attempt them: gate failures
        won't change without a settings change. Picking "Continue with
        current settings" on a settings-drift dialog clears them so
        they're given another shot under the new gate thresholds.

        ``error_text`` is shown next to the row as a short one-liner.
        """
        # Categorize BEFORE translating landmark names — the gate-error
        # prefix ("Core landmarks failed confidence gate") isn't touched
        # by translation today, but classifying on the raw text keeps
        # the regex coupling tight to LowConfidenceLandmarkError's own
        # message format.
        self._failure_category[basename] = _classify_preproc_failure(error_text)
        # Translate raw landmark keys to anatomical names here, at
        # storage, so the inline row label (rendered via
        # _shorten_error → _update_image_status) shows friendly names
        # alongside the run log.
        translated = _translate_landmark_names(error_text) if error_text else ""
        if translated:
            self._image_error_text[basename] = translated
        if self._manifest is not None and self._run_folder is not None:
            # Persist the translated message so the post-run restore flow
            # can replay the same tooltip the user would have seen at the
            # end of the run.
            self._manifest.mark_failed_preproc(basename, translated)
            save_manifest(self._run_folder, self._manifest)
        self._update_image_status(basename, ImageStatus.FAILED)

    def _on_image_failed_analysis(self, basename: str, error_text: str = "") -> None:
        """One image's Stage 2 errored — mark the row failed.

        Stage-2 failures are persisted under analysis_failed_images so the
        post-run "Reload previous session" flow can resurface them. The
        companion image_completed signal that fires from _signal_complete
        in TRACE/pipeline.py still records the image in completed_images
        so resume's skip set keeps it out of a re-run (the manifest tracks
        "did we get to Stage 2?" + "did Stage 2 succeed?" independently).
        """
        # Stage-2 has two flavours: a garbage-detector quality-gate abort
        # (GarbageRejection — solidity / fragmentation / vein-association /
        # vein-presence) gets bucketed as "gate" alongside landmark
        # confidence-gate aborts, so the "Rerun failed (no quality gates)"
        # button picks it up. Everything else stays "analysis" (genuine
        # crash, output write error, etc).
        self._failure_category[basename] = _classify_analysis_failure(error_text)
        translated = _translate_landmark_names(error_text) if error_text else ""
        if translated:
            self._image_error_text[basename] = translated
        if self._manifest is not None and self._run_folder is not None:
            self._manifest.mark_failed_analysis(basename, translated)
            save_manifest(self._run_folder, self._manifest)
        self._update_image_status(basename, ImageStatus.FAILED)

    def _on_paused(self, results: list) -> None:
        """Worker stopped between images after a Pause click.

        Mark manifest paused, flip the button label so the user can resume
        in-session, and leave btn_run disabled (clicking Run while paused
        would conceptually mean "start over"; the user can wipe outputs +
        manifest if that's what they want).
        """
        self._revert_in_progress_to_pending()
        # The transient "Pause requested…" label served its purpose; hide
        # it now that the worker has actually paused.
        self.transient_status_label.hide()
        # Treat paused like an early return from a successful slice — no
        # error messaging, but acknowledge in the log.
        n_done = len(self._manifest.completed_images) if self._manifest is not None else 0
        n_total = self._manifest.total_images if self._manifest is not None else 0
        self._log(f"Paused at image {n_done} of {n_total}. Click Resume to continue.")
        self.statusBar().showMessage(f"Paused — {n_done} of {n_total} done")
        self._progress_timer.stop()
        # Persist paused status.
        if self._manifest is not None and self._run_folder is not None:
            self._manifest.status = STATUS_PAUSED
            save_manifest(self._run_folder, self._manifest)
        # Append a clear pause marker to the run log so the user (and a
        # later reader of run.log) sees exactly when each slice ended.
        from datetime import datetime as _dt

        self._log(f"--- Paused at {_dt.now().isoformat(timespec='seconds')} ---")
        # Flush the log to disk so a hard kill (close TRACE without Resume)
        # leaves run.log in sync with what the user just saw on screen.
        self._write_run_metadata(status="paused")
        # Worker thread has exited at this point (paused signal is emitted
        # at the end of run()). Clear the handle so the next Resume click
        # routes through _resume_paused_run.
        self.worker = None
        self._is_paused = True
        # The combo button now means "Resume" — click it to continue.
        self.btn_run.setText("Resume")
        self.btn_run.setEnabled(True)
        # Recolor the progress-bar fill yellow as a visual cue. Reset back
        # to the default highlight color happens in _run_pipeline on Resume.
        from TRACE.theme import current_theme as _ct

        pal = self.progress.palette()
        pal.setColor(QPalette.Highlight, QColor(_ct().warning))
        self.progress.setPalette(pal)
        # Cancel stays enabled while paused so the user can fully abandon
        # the run from the paused state (no worker is running, but the
        # manifest still claims the run is in-progress).
        self.btn_cancel.setEnabled(True)
        # Un-lock the per-row check toggles while paused — Resume picks
        # up the merged skip set fresh, so the user can adjust skips
        # between slices.
        self._set_skip_checkboxes_enabled(True)
        # Recompute rerun-button visibility from this slice's failures.
        self._record_failures_from_run(results)

    def _resume_paused_run(self) -> None:
        """User clicked Resume after pausing in-session — re-launch the worker
        with the manifest's completed list as the skip set."""
        if self._manifest is None:
            # Should never happen — _is_paused without a manifest is a bug,
            # not a state we should silently tolerate. Reset UI defensively.
            self._is_paused = False
            self.btn_run.setText("Run Pipeline")
            self.btn_run.setEnabled(True)
            return
        # Re-run _run_pipeline with the existing manifest available so
        # _maybe_offer_resume's "active manifest is current state, don't
        # re-prompt" branch kicks in. Simpler than duplicating the
        # validation + worker-construction code here.
        self._is_paused = False
        self.btn_run.setEnabled(True)  # _run_pipeline disables it again immediately
        self._run_pipeline()

    def _maybe_offer_resume(
        self,
        output_dir: Path,
        outputs: set,
        csv_groups: set,
    ) -> tuple[Optional[set[str]], Optional["RunManifest"]]:
        """Check for an existing in-progress manifest; ask the user what to do.

        Returns ``(skip_set, manifest)``:
          - skip_set is the set of basenames to skip on this run (empty for
            a fresh start),
          - manifest is the loaded RunManifest to continue with (None to
            create a new one),
          - ``(None, None)`` if the user cancelled.

        Three call paths:
          1. No manifest on disk → return ({}, None) silently (fresh run).
          2. In-session Resume click → existing self._manifest matches the
             output_dir, skip its completed set, return it directly.
          3. Manifest on disk from a prior session → prompt the user.
        """
        # Path 2: in-session resume after Pause. self._manifest is still
        # populated from the paused worker; self._run_folder still points
        # at the same place. Run the settings-drift check (the user may
        # have tweaked things between Pause and Resume), then hand the
        # data back.
        if self._manifest is not None and self._manifest.is_in_progress():
            if self._run_folder is not None and self._manifest.settings_snapshot_path:
                if not self._confirm_settings_drift(self._run_folder):
                    return None, None
            self._is_resuming = True
            return self._compute_resume_skip_set(self._manifest), self._manifest

        from TRACE.run_state import find_resumable_manifest

        found = find_resumable_manifest(output_dir)
        if found is None:
            # Fresh run — no resumable state on disk.
            self._is_resuming = False
            return set(), None
        manifest, run_folder = found
        # Keep self._run_folder pointed at the resumed folder so the
        # downstream save_manifest / settings-drift-snapshot calls land
        # alongside the existing files.
        self._run_folder = run_folder

        n_done = len(manifest.completed_images)
        n_total = manifest.total_images
        # Prompt: resume vs. start over vs. cancel.
        box = QMessageBox(self)
        box.setIcon(QMessageBox.Question)
        box.setWindowTitle("Resume previous run?")
        box.setText(
            f"This output folder has an unfinished TRACE run from "
            f"{manifest.started_at or '(unknown time)'}: "
            f"{n_done} of {n_total} images completed."
        )
        box.setInformativeText(
            "Resume from where it left off, start over (existing outputs are "
            "kept on disk but won't be skipped), or cancel?"
        )
        btn_resume = box.addButton("Resume", QMessageBox.AcceptRole)
        btn_fresh = box.addButton("Start over", QMessageBox.DestructiveRole)
        btn_cancel = box.addButton("Cancel", QMessageBox.RejectRole)
        box.setDefaultButton(btn_resume)
        box.exec_()
        clicked = box.clickedButton()
        if clicked is btn_cancel:
            self._run_folder = None
            return None, None
        if clicked is btn_fresh:
            self._run_folder = None  # Fresh run will create a new folder.
            self._is_resuming = False
            return set(), None
        # Resume — fold in a settings-drift check against the most recent
        # settings*.yaml in the run folder. If the user accepts new
        # settings, _confirm_settings_drift writes settings_partN.yaml
        # and logs a diff summary.
        if manifest.settings_snapshot_path:
            if not self._confirm_settings_drift(run_folder):
                self._run_folder = None
                return None, None
        self._is_resuming = True
        return self._compute_resume_skip_set(manifest), manifest

    def _compute_resume_skip_set(self, manifest: "RunManifest") -> set[str]:
        """Build the resume skip set from a manifest.

        Always includes the completed images. Includes the previously-
        failed-preproc images too unless the user chose "Continue with
        current settings" on the drift dialog (in which case the gate
        thresholds may have changed and those images deserve another
        shot).
        """
        skip = manifest.completed_set()
        if not getattr(self, "_resume_settings_changed", False):
            skip |= manifest.failed_preproc_set()
        return skip

    def _current_settings_snapshot_dict(self) -> dict:
        """Build the dict that gets persisted as _run_settings.yaml.

        Includes only the user-tunable run knobs (pipeline_config, gate
        override, gui_state) — no timestamps or status fields, so the dict
        is directly comparable between runs.
        """
        from TRACE.config_io import config_to_dict

        return {
            "pipeline_config": config_to_dict(self.config),
            "gate_override": self._gate_override or None,
            "gui_state": self.get_gui_state(),
        }

    def _write_settings_snapshot(self, path: Path) -> None:
        """Write the current settings snapshot to ``path`` as YAML.

        Best-effort: errors are logged but never block the run. Used by
        _run_pipeline at the start of a fresh run so a later resume can
        compare against current state.
        """
        try:
            import yaml as _yaml

            payload = self._current_settings_snapshot_dict()
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(
                _yaml.safe_dump(payload, sort_keys=False, default_flow_style=False),
                encoding="utf-8",
            )
        except Exception as exc:  # noqa: BLE001
            self._log(f"Warning: could not write settings snapshot: {exc}")

    def _run_log_preamble_dict(self) -> dict:
        """Build the configuration block written to the top of the run log.

        Superset of _current_settings_snapshot_dict — also captures
        input/output paths, recursive flag, final-output toggles, CSV
        measurement groups, run timestamp, and the TRACE version, so the
        saved run.log is self-describing on its own (without needing the
        sibling settings.yaml / manifest.json).

        Folder paths are included intentionally despite being potentially
        user-identifying — the log is meant to be sharable as a
        reproducibility artifact, and the absolute paths let a reviewer
        correlate a run with its input dataset.
        """
        from TRACE import __version__ as _trace_version

        return {
            "trace_version": _trace_version,
            "run_started": self._run_start_wall.isoformat(timespec="seconds"),
            "input_folder": self.input_edit.text(),
            "output_folder": self.output_edit.text(),
            "recursive": bool(self.recursive_chk.isChecked()),
            "final_outputs": sorted(k for k, chk in self.output_checks.items() if chk.isChecked()),
            "csv_measurement_groups": sorted(k for k, chk in self.csv_group_checks.items() if chk.isChecked()),
            "settings": self._current_settings_snapshot_dict(),
        }

    def _log_run_settings(self) -> None:
        """Write a compact summary of the active run config into the log.

        Emitted as one ``---``-fenced block (no per-line ``[HH:MM:SS]``
        prefixes) so the whole block reads as a coherent header. The
        canonical machine-readable copy is settings.yaml in the run
        folder; this preamble is the human/LLM-glance version.

        Called once at the start of every fresh run and again at the start
        of every resume — resumes stack a second block in-place, so a
        ``diff`` across blocks shows exactly what changed between slices.
        """
        try:
            text = _format_run_settings_compact(self._run_log_preamble_dict())
        except Exception as exc:  # noqa: BLE001
            # Best-effort — never block a run because the preamble can't render.
            self._log(f"Warning: could not format run-settings preamble: {exc}")
            return
        self._log("Run settings:")
        # Mirror the machine-parseable YAML block into run.log too so a
        # streamed log has the same "what config produced this run"
        # header the widget shows. Route the widget append and the file
        # append through the same lines to stay in sync.
        for _preamble_line in ("---", text, "---", ""):
            self.log_text.append(_preamble_line)
            self._stream_log_line(_preamble_line)
        sb = self.log_text.verticalScrollBar()
        sb.setValue(sb.maximum())

    def _latest_settings_yaml(self, run_folder: Path) -> tuple[Optional[Path], int]:
        """Locate the most-recent settings YAML in ``run_folder``.

        Returns (path, part_number). ``part_number`` is 1 for the base
        settings.yaml, 2+ for settings_partN.yaml. (path, 0) when nothing
        is found.
        """
        base = run_folder / "settings.yaml"
        parts: list[tuple[int, Path]] = []
        for p in run_folder.glob("settings_part*.yaml"):
            try:
                n = int(p.stem.removeprefix("settings_part"))
            except ValueError:
                continue
            parts.append((n, p))
        if parts:
            parts.sort(reverse=True)
            return parts[0][1], parts[0][0]
        if base.is_file():
            return base, 1
        return None, 0

    def _summarize_settings_diff(self, saved: dict, current: dict) -> list[str]:
        """Walk two nested settings dicts and return human-readable diff lines.

        Caps at ~10 lines so a wholesale settings rewrite doesn't flood the
        log. Each line is "<dotted.key>: <saved> → <current>".
        """
        diffs: list[str] = []

        def _walk(prefix: str, a, b) -> None:
            if isinstance(a, dict) and isinstance(b, dict):
                keys = sorted(set(a) | set(b))
                for k in keys:
                    _walk(f"{prefix}.{k}" if prefix else k, a.get(k), b.get(k))
            elif a != b:
                diffs.append(f"  {prefix}: {a!r} → {b!r}")

        _walk("", saved or {}, current or {})
        if len(diffs) > 10:
            extra = len(diffs) - 10
            diffs = diffs[:10] + [f"  … and {extra} more change(s)"]
        return diffs

    def _apply_original_settings(self, saved: dict) -> None:
        """Restore current GUI / config state from a saved settings dict.

        Inverse of _current_settings_snapshot_dict. Routes the three
        sub-dicts (pipeline_config, gate_override, gui_state) through the
        existing import-settings helpers, so inline panels refresh too
        (apply_gui_state ends with refresh_from_state() calls).
        """
        from TRACE.config_io import config_from_dict

        pcfg = saved.get("pipeline_config")
        if pcfg:
            try:
                self.config = config_from_dict(pcfg)
            except Exception as exc:  # noqa: BLE001
                self._log(f"Could not apply original pipeline_config: {exc}")
        self._gate_override = saved.get("gate_override") or None
        gs = saved.get("gui_state")
        if isinstance(gs, dict):
            self.apply_gui_state(gs)

    def _confirm_settings_drift(self, run_folder: Path) -> bool:
        """Compare current settings to the latest snapshot in ``run_folder``.

        On drift, offers three choices:
          - Use original — restore the saved snapshot into the GUI / config
            and resume as if nothing changed (no settings_partN.yaml
            written; the original part stays in force).
          - Continue with current — write settings_part<N+1>.yaml, log the
            diff, resume with the new settings applied to remaining images.
          - Cancel — bail; the GUI stays in its paused state and the user
            can re-tweak before clicking Resume again.

        Returns True when the resume should proceed (matched, restored, or
        user picked "Continue"), False to abort.

        Side effect: sets ``self._resume_settings_changed`` to True iff the
        caller chose "Continue with current settings" (i.e. accepted that
        new settings will apply to remaining images). The caller uses this
        flag to decide whether to retry previously-failed-preproc images
        — a settings change may unblock them, so the skip set drops them.
        """
        self._resume_settings_changed = False
        snapshot_path, current_part = self._latest_settings_yaml(run_folder)
        if snapshot_path is None:
            return True
        try:
            import yaml as _yaml

            saved = _yaml.safe_load(snapshot_path.read_text(encoding="utf-8"))
        except Exception as exc:  # noqa: BLE001
            self._log(f"Could not read settings snapshot: {exc}")
            return True
        current = self._current_settings_snapshot_dict()
        if saved == current:
            return True
        diff_lines = self._summarize_settings_diff(saved, current)
        box = QMessageBox(self)
        box.setIcon(QMessageBox.Warning)
        box.setWindowTitle("Settings changed since last run")
        box.setText(
            "The TRACE settings have changed since this run started. " "Choose how to handle the remaining images."
        )
        body = (
            "Use original — restore the saved settings before resuming.\n"
            "Continue with current — apply the new settings only to the "
            "remaining images (already-completed images keep their original "
            "results).\n"
            "Cancel — go back to the Paused state without resuming."
        )
        if diff_lines:
            body += "\n\nChanged:\n" + "\n".join(diff_lines)
        box.setInformativeText(body)
        btn_original = box.addButton("Use original", QMessageBox.AcceptRole)
        btn_current = box.addButton("Continue with current", QMessageBox.AcceptRole)
        btn_cancel = box.addButton("Cancel", QMessageBox.RejectRole)
        box.setDefaultButton(btn_cancel)
        box.exec_()
        clicked = box.clickedButton()
        if clicked is btn_cancel:
            return False
        if clicked is btn_original:
            # Restore the saved snapshot into current state. No new
            # settings_partN.yaml — we're conceptually returning to the
            # state the manifest already references.
            self._apply_original_settings(saved)
            self._log(f"Resumed with the original settings from {snapshot_path.name}.")
            return True
        # btn_current — snapshot the new settings as settings_part<N+1>.yaml
        # so the post-drift state is preserved alongside the original. Also
        # log the diff so a reader of the run.log knows the run was
        # processed under more than one settings configuration.
        self._resume_settings_changed = True
        new_part = max(2, current_part + 1)
        new_path = run_folder / f"settings_part{new_part}.yaml"
        self._write_settings_snapshot(new_path)
        if self._manifest is not None:
            self._manifest.settings_snapshot_path = new_path.name
        self._log(f"Settings changed since last slice; new settings saved to {new_path.name}.")
        for line in diff_lines:
            self._log(line)
        return True

    # -----------------------------------------------------------------------
    # Logging and callbacks
    # -----------------------------------------------------------------------
    def _log(self, msg):
        # Run every message through the landmark-name translator so direct
        # _log() callers (e.g. the "Failed images:" summary in _on_all_done,
        # which doesn't route through the logging-module handler) get the
        # same friendly anatomical names as records flowing through
        # _SignalLogHandler. _log_run_settings appends its YAML block via
        # log_text.append directly, intentionally bypassing this hook so
        # the block stays machine-parseable.
        line = f"[{time.strftime('%H:%M:%S')}] {_translate_landmark_names(msg)}"
        self.log_text.append(line)
        scrollbar = self.log_text.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())
        # Stream every logged line to run.log in the active run folder so
        # the running log persists incrementally alongside settings.yaml
        # and _run_state.json — no more losing progress info if TRACE is
        # killed / crashes / closed mid-run. _write_run_metadata still
        # runs at terminal transitions as a canonical full-snapshot
        # rewrite (from log_text.toPlainText()) covering anything that
        # slipped in before _run_folder was set.
        self._stream_log_line(line)

    def _stream_log_line(self, line: str) -> None:
        """Append `line` to run.log in the active run folder.

        No-op if the run folder isn't set (pre-run bookkeeping messages
        land in the log widget only) or the write fails (disk full,
        permissions, race with folder deletion). Silent failure is
        intentional — a log-write hiccup must never derail a run.
        """
        folder = self._run_folder
        if folder is None:
            return
        try:
            with (folder / "run.log").open("a", encoding="utf-8") as fh:
                fh.write(line)
                fh.write("\n")
        except Exception:  # noqa: BLE001
            pass

    def _on_progress(self, idx, total, name, stage, detail):
        """Update internal stage-tracking state on each progress event.

        Doesn't touch the progress bar / ETA directly — that happens via
        _refresh_progress, which is called both here and from a 250ms QTimer
        so the bar interpolates smoothly between completion events.
        """
        # First progress event for a PENDING image flips it to IN_PROGRESS.
        # Never downgrade a terminal status (SUCCEEDED / FAILED / SKIPPED)
        # — substep events can fire after the image has been marked done.
        if self._image_status.get(name) == ImageStatus.PENDING:
            self._update_image_status(name, ImageStatus.IN_PROGRESS)
        # Detect stage transitions so the hybrid ETA can reset its throughput
        # tracker (Stage 2 per-image cost is wildly different from Stage 1).
        if stage in ("preprocessing", "analysis") and stage != self._current_stage:
            self._current_stage = stage
            self._stage_first_event_time = time.monotonic()
            self._stage_completions = 0
            self._stage_total = total
            self._last_completion_time = None
            self._avg_time_per_image = None
        self._stage_total = max(self._stage_total, total)

        # Each image fires multiple progress events ("starting", "landmarks: …",
        # "hinge: …", "segmentation: …", "done"). Only the "done" event
        # represents an actual completion. Lock in the average time per image
        # at each completion so the smoothing interpolator has a stable
        # per-tick value (otherwise recomputing avg = elapsed/K every tick
        # forces fractional-progress-since-last-completion to 0).
        is_done = isinstance(detail, str) and detail.startswith("done")
        if is_done and self._stage_first_event_time is not None:
            self._stage_completions = max(self._stage_completions, idx + 1)
            self._last_completion_time = time.monotonic()
            stage_elapsed = self._last_completion_time - self._stage_first_event_time
            self._avg_time_per_image = stage_elapsed / max(self._stage_completions, 1)

        self._refresh_progress()
        msg = f"[{idx + 1}/{total}] {name}: {stage} - {detail}"
        self.statusBar().showMessage(msg)
        self._log(msg)

    def _smoothed_within_stage_fraction(self) -> float:
        """Return 0.0..1.0 reflecting smoothed progress within the current stage.

        Between completion events the value advances linearly toward the next
        expected completion based on the locked-in average time per image,
        so the bar doesn't sit frozen for tens of seconds when parallel
        workers complete in clustered bursts.
        """
        if self._stage_total <= 0 or self._stage_first_event_time is None:
            return 0.0
        if self._last_completion_time is None or self._avg_time_per_image is None:
            # No completions yet in this stage — bar stays at the previous
            # stage's final position (or 0 for Stage 1). Honest: we don't
            # have a per-image time to extrapolate from yet.
            return 0.0
        time_since_last = max(0.0, time.monotonic() - self._last_completion_time)
        # Fractional advance toward the next completion (0 → 1 over avg).
        # Cap at 1.0 so we never appear to have completed an image we haven't.
        fractional = min(1.0, time_since_last / max(self._avg_time_per_image, 0.01))
        estimated = min(float(self._stage_total), self._stage_completions + fractional)
        return estimated / self._stage_total

    def _refresh_progress(self) -> None:
        """Compute the smoothed pct + ETA from current state. Called both
        when a progress event arrives and from a 250ms QTimer."""
        if self._current_stage is None:
            return
        within_stage = self._smoothed_within_stage_fraction()
        if self._current_stage == "analysis":
            pct_float = (self._progress_stage1_share + self._progress_stage2_share * within_stage) * 100.0
        else:
            pct_float = self._progress_stage1_share * within_stage * 100.0
        pct = min(99, int(pct_float))
        if pct > self._progress_pct_high:
            self._progress_pct_high = pct
            self.progress.setValue(pct)
        # ETA updates every tick whether or not pct ticked over an integer
        # boundary, so the displayed time also evolves smoothly.
        self._update_eta()

    @staticmethod
    def _format_eta(seconds: float) -> str:
        import math

        if not math.isfinite(seconds) or seconds <= 0:
            return "Estimating time until pipeline finishes…"
        # Sub-minute precision is noisy with per-image variance and parallelism;
        # round to minutes and use a "<1m" floor so the displayed value isn't
        # misleadingly specific.
        total_minutes = int(round(seconds / 60))
        if total_minutes < 1:
            return "<1m until pipeline finishes"
        if total_minutes < 60:
            return f"~{total_minutes}m until pipeline finishes"
        h, m = divmod(total_minutes, 60)
        return f"~{h}h {m}m until pipeline finishes"

    def _update_eta(self) -> None:
        """Recompute the smoothed ETA and update the label under the progress bar.

        Hybrid of two estimates:
          • Percent-based (prior): elapsed × (100 − pct) / pct. Stable from the
            first event because it leans on the Stage 1 / Stage 2 wall-time
            weights from `compute_progress_weights`, but uniformly off when a
            run's per-image cost deviates from the reference workload.
          • Throughput-based (empirical): observe images-per-second in the
            current stage and divide remaining images by it. Auto-adapts to
            this run's actual cost profile, but needs ≥2 samples to be stable.
        Blend weight grows from 0 → 1 as Stage samples accumulate (full
        throughput at 3 completions), then we exponentially smooth so the
        displayed value doesn't whiplash on outlier images.
        """
        if self._run_start_time is None or self._progress_pct_high <= 0:
            return
        elapsed = time.monotonic() - self._run_start_time
        pct = self._progress_pct_high
        percent_eta = elapsed * (100 - pct) / pct

        eta_raw = percent_eta
        if self._stage_first_event_time is not None and self._stage_completions >= 2 and self._stage_total > 0:
            stage_elapsed = time.monotonic() - self._stage_first_event_time
            if stage_elapsed > 0:
                throughput = self._stage_completions / stage_elapsed  # images/sec
                stage_remaining = max(0, self._stage_total - self._stage_completions)
                stage_remaining_seconds = stage_remaining / max(throughput, 1e-6)

                if self._current_stage == "preprocessing":
                    # Project full-Stage-1 wall time from observed throughput,
                    # then derive total run from the Stage 1 share prior. Stage 2
                    # remaining is what's left after now.
                    if self._progress_stage1_share > 0:
                        full_stage1 = self._stage_total / max(throughput, 1e-6)
                        total_predicted = full_stage1 / self._progress_stage1_share
                        throughput_eta = max(0.0, total_predicted - elapsed)
                    else:
                        throughput_eta = stage_remaining_seconds
                else:
                    # Stage 2 (analysis) — empirical throughput on remaining images.
                    throughput_eta = stage_remaining_seconds

                # Blend: w grows with completed samples in the current stage.
                # Full throughput-based at 3 completions; below that we blend
                # in the percent-based prior to absorb sample noise.
                threshold = 3.0
                w = min(1.0, self._stage_completions / threshold)
                eta_raw = w * throughput_eta + (1 - w) * percent_eta

        # Exponential moving average — α=0.3 favors stability over reactivity.
        alpha = 0.3
        if self._eta_smoothed_seconds is None:
            self._eta_smoothed_seconds = eta_raw
        else:
            self._eta_smoothed_seconds = alpha * eta_raw + (1 - alpha) * self._eta_smoothed_seconds
        self.eta_label.setText(self._format_eta(self._eta_smoothed_seconds))

    def _write_run_metadata(self, status: str) -> Optional[Path]:
        """Flush the live Log panel contents to ``run.log`` in the run folder.

        The settings YAML(s) and the manifest are written at run START
        (and updated on resume drift) by _run_pipeline; this function no
        longer duplicates them here. The run folder itself was created at
        run start and lives at self._run_folder; if for some reason it
        wasn't (e.g. validation failed before run-folder creation), this
        is a no-op.

        Returns the run folder path, or None when nothing was written.
        """
        meta_dir = self._run_folder
        if meta_dir is None:
            return None
        try:
            meta_dir.mkdir(parents=True, exist_ok=True)
            (meta_dir / "run.log").write_text(self.log_text.toPlainText(), encoding="utf-8")
            return meta_dir
        except Exception as e:  # noqa: BLE001
            # Don't let a metadata-write failure mask the actual run result.
            self._log(f"Warning: could not write run log: {e}")
            return None

    def _on_all_done(self, results):
        # Cancel-pending fallback: the worker finished naturally between
        # the user's Cancel click and the next progress checkpoint (no
        # InterruptedError was raised). Reroute into _finalize_cancel so
        # the run gets discarded as requested instead of being treated as
        # a normal completion.
        if self._cancel_requested:
            self.worker = None
            self._finalize_cancel(results)
            return
        # Clear the worker handle so the next Run click is treated as
        # "start fresh" by _on_run_button_clicked (which dispatches on
        # `self.worker is not None` → Pause). The cancel / pause / error
        # paths all do this; the natural-completion branch was missing
        # it, so a second run within the same session triggered a
        # "Pause requested" against an already-exited thread instead of
        # kicking off a new pipeline. Reported after a fresh-completion
        # session: 2 hours after a 24-image run, clicking Run produced
        # only the "Pause requested — finishing the current image
        # first." status with no actual second run starting.
        self.worker = None
        self.btn_run.setEnabled(True)
        self.btn_run.setText("Run Pipeline")
        self.btn_cancel.setEnabled(False)
        self._is_paused = False
        self._progress_timer.stop()
        # Run is over — un-lock the per-row check toggles.
        self._set_skip_checkboxes_enabled(True)
        # Any image still marked IN_PROGRESS after the worker exits — e.g.
        # the slice ended early via cancel — gets rolled back so the
        # resting state of the list is honest.
        self._revert_in_progress_to_pending()
        self.transient_status_label.hide()

        # Capture failures BEFORE wiping the manifest reference below.
        # _record_failures_from_run reads manifest.failed_preproc_set().
        self._record_failures_from_run(results)

        # Mark the manifest as completed so the next run on this output
        # folder doesn't surface the resume prompt.
        if self._manifest is not None and self._run_folder is not None:
            self._manifest.status = STATUS_COMPLETED
            save_manifest(self._run_folder, self._manifest)
        # Clear so any subsequent fresh run rebuilds from scratch.
        self._manifest = None
        self._is_resuming = False
        # Keep self._run_folder set — _write_run_metadata below writes
        # run.log into it; it gets cleared after that.

        if not results:
            self._log("\nPipeline cancelled or no results.")
            self.statusBar().showMessage("Cancelled")
            self.eta_label.setText("")
            self._write_run_metadata(status="cancelled")
            self._run_folder = None
            return

        # Pipeline finished cleanly — release the 99% cap and snap to 100.
        self._progress_pct_high = 100
        self.progress.setValue(100)
        # Recolor the filled chunk green via palette (not stylesheet) so the
        # native rendering path stays in place — only the chunk color shifts,
        # no indentation/border/text changes. Reverted at the start of the
        # next run by _run_pipeline.
        from TRACE.theme import current_theme as _ct

        pal = self.progress.palette()
        pal.setColor(QPalette.Highlight, QColor(_ct().success))
        self.progress.setPalette(pal)
        self.eta_label.setText("Done")

        succeeded = sum(1 for r in results if r.error is None)
        failed = sum(1 for r in results if r.error is not None)
        summary = f"Done: {succeeded} succeeded, {failed} failed out of {len(results)} images."
        self._log(f"\n{summary}")
        self.statusBar().showMessage(summary)

        if failed:
            self._log("\nFailed images:")
            for r in results:
                if r.error:
                    self._log(f"  {r.image_path.name} ({r.error_stage}): {r.error.splitlines()[0]}")

        # Write per-run metadata folder BEFORE opening the output folder
        # so the log text already includes the final summary line.
        status = "ok" if failed == 0 else f"partial ({failed} failed)"
        self._write_run_metadata(status=status)
        self._run_folder = None

        # Open the output folder in the system file manager so the user can
        # see results without hunting for the path.
        out_dir = self.output_edit.text().strip()
        if out_dir and Path(out_dir).is_dir():
            from PyQt5.QtCore import QUrl
            from PyQt5.QtGui import QDesktopServices

            QDesktopServices.openUrl(QUrl.fromLocalFile(out_dir))

    def _on_error(self, msg):
        # Cancel-pending fallback: worker errored out after Cancel was
        # requested. Treat it as a cancel rather than a fatal error so the
        # user doesn't get a "Pipeline Error" dialog for a run they
        # already discarded. The actual error text is still logged below.
        if self._cancel_requested:
            self._log(f"\nWorker exited with error after cancel: {msg}")
            self.worker = None
            # No results list available on the error path; _finalize_cancel
            # still captures preproc failures from the manifest.
            self._finalize_cancel()
            return
        # Same fix as _on_all_done's natural-completion branch: clear
        # the worker handle so a follow-up Run click isn't dispatched
        # as Pause.
        self.worker = None
        self.btn_run.setEnabled(True)
        self.btn_run.setText("Run Pipeline")
        self.btn_cancel.setEnabled(False)
        self.transient_status_label.hide()
        self._progress_timer.stop()
        self.eta_label.setText("")
        # Un-lock the per-row check toggles after a fatal error too.
        self._set_skip_checkboxes_enabled(True)
        self._log(f"\nFatal error: {msg}")
        # Write the run-metadata folder even on fatal error so the
        # settings + log capture is preserved for debugging.
        self._write_run_metadata(status="error")
        QMessageBox.critical(self, "Pipeline Error", msg)

    # -----------------------------------------------------------------------
    # First-launch walkthrough
    # -----------------------------------------------------------------------
    def _walkthrough_steps(self) -> list[WalkthroughStep]:
        """Step-by-step tour of the main window's key controls."""

        def _settings_step(group_attr: str, title: str, body: str) -> WalkthroughStep:
            """Build a step targeting a group box on the (scrollable) Settings tab.

            The pre-action switches to the Settings tab and scrolls the target
            group box into view so the highlight ring lands on it even when the
            group sits below the fold.
            """

            def _pre(w: QWidget) -> None:
                w.right_tabs.setCurrentIndex(1)
                scroll = w.right_tabs.widget(1)
                group = getattr(w.inline_general_panel, group_attr, None)
                if group is not None and hasattr(scroll, "ensureWidgetVisible"):
                    scroll.ensureWidgetVisible(group, 0, 40)

            return WalkthroughStep(
                target_resolver=lambda w: getattr(w.inline_general_panel, group_attr),
                pre_action=_pre,
                title=title,
                body=body,
            )

        def _tab_intro(index: int, title: str, body: str) -> WalkthroughStep:
            """Build an overview step that highlights one tab in the right-panel
            tab bar before the steps that walk through that tab's controls."""

            def _tab_rect(w: QWidget) -> QRect:
                # Rect of just this tab within the tab bar, in central-widget
                # coordinates (the space the overlay highlights in).
                bar = w.right_tabs.tabBar()
                tab_rect = bar.tabRect(index)
                origin = bar.mapTo(w.centralWidget(), tab_rect.topLeft())
                return QRect(origin, tab_rect.size())

            return WalkthroughStep(
                target_resolver=lambda w: w.right_tabs.tabBar(),
                rect_resolver=_tab_rect,
                pre_action=lambda w: w.right_tabs.setCurrentIndex(index),
                title=title,
                body=body,
            )

        def _input_step_rect(w: QWidget) -> QRect:
            # Highlight both the input-folder picker AND the Include
            # subfolders checkbox as one region — they're a unit and the
            # walkthrough text mentions both.
            tl = w.input_row.mapTo(w.centralWidget(), QPoint(0, 0))
            chk = w.recursive_chk
            br = chk.mapTo(w.centralWidget(), QPoint(chk.width(), chk.height()))
            return QRect(tl, br)

        return [
            WalkthroughStep(
                target_resolver=lambda w: w.input_row,
                rect_resolver=_input_step_rect,
                title="Pick an input folder",
                body=(
                    "Click Browse to choose a folder of wing images. TRACE will "
                    "discover and queue all supported images. To search subfolders "
                    "for images, check the Include subfolders box."
                ),
            ),
            WalkthroughStep(
                target_resolver=lambda w: w.output_row,
                title="Pick an output folder",
                body=(
                    "Click Browse to choose where to save TRACE results. This folder "
                    "opens automatically when a run has been completed."
                ),
            ),
            WalkthroughStep(
                target_resolver=lambda w: w.scale_spin,
                title="Set µm/px",
                body=(
                    "Enter the microns-per-pixel conversion factor for your images. "
                    "<b>All images in a run must have the same scale.</b>"
                ),
            ),
            WalkthroughStep(
                target_resolver=lambda w: w.out_group,
                title="Choose your outputs",
                body=("Pick what to save: various feature overlays and measurements."),
            ),
            _tab_intro(
                0,
                "The Main tab",
                "Your run dashboard: it lists the queued images and streams the live " "log while TRACE works.",
            ),
            WalkthroughStep(
                target_resolver=lambda w: w.image_list,
                pre_action=lambda w: w.right_tabs.setCurrentIndex(0),
                title="Main tab — image queue",
                body=(
                    "All images discovered in the input folder. Each row updates with "
                    "its status as the run proceeds. <b>Uncheck</b> a row to skip that "
                    "image when you click Run; right-click for bulk Skip / Unskip "
                    "actions on the selection or the whole list."
                ),
            ),
            WalkthroughStep(
                target_resolver=lambda w: w.log_text,
                pre_action=lambda w: w.right_tabs.setCurrentIndex(0),
                title="Main tab — run log",
                body=(
                    "The Log streams progress, per-image messages, and warnings while the "
                    "pipeline runs. Check here first if a result looks off."
                ),
            ),
            _tab_intro(
                1,
                "The Settings tab",
                "Where you tune a run: scale calibration, optional preprocessing steps, "
                "output appearance, and processing throughput.",
            ),
            _settings_step(
                "_scale_group",
                "Settings — Scale",
                "Another place where you can set µm/px conversion factor. If you are "
                "unsure of your scale, click on the Estimate button and TRACE will "
                "make a guess based on the location of landmark points.",
            ),
            _settings_step(
                "_optional_preprocessing_group",
                "Settings — Optional preprocessing",
                "Extra steps that can be toggled on to improve TRACE's accuracy at the "
                "expense of time. Use wing isolation when neighboring wings are visible "
                "in your images and wing rotation if your wings are not right-side-up.",
            ),
            _settings_step(
                "_output_options_group",
                "Settings — Output options",
                "If you have selected vein and/or intervein overlay output(s), control "
                "the appearance of detected veins/intervein regions.",
            ),
            _settings_step(
                "_parallel_processing_group",
                "Settings — Parallel processing",
                "Set how many wings process at once, or use Calibrate to benchmark a "
                "safe worker count for your machine.",
            ),
            _settings_step(
                "btn_advanced",
                "Settings — Advanced Settings",
                "Opens a dialog of fine-grained pipeline controls: per-model gate "
                "thresholds, skeletonization, bridging, tracing, and intervein region "
                "detection. Most runs never need these.",
            ),
            _settings_step(
                "_reset_buttons_widget",
                "Settings — Reset options",
                "Restore Defaults resets this tab to its factory values. wipe my "
                "memories goes further — it clears every persisted setting (folders, "
                "model paths, custom pairs) and returns TRACE to a first-launch state.",
            ),
            _tab_intro(
                2,
                "The Custom Measurements tab",
                "Define your own landmark-to-landmark measurements — these can be added "
                "to the measurements CSV (selectable under Outputs as Custom measurements).",
            ),
            WalkthroughStep(
                target_resolver=lambda w: w.inline_custom_distances_panel._picker._source_widget,
                pre_action=lambda w: w.right_tabs.setCurrentIndex(2),
                title="Custom Measurements — sample wing",
                body=(
                    "Pick a sample wing image and load it into the viewer — TRACE "
                    "detects landmarks automatically. Click Restore cartoon wing for "
                    "a bundled example you can experiment with."
                ),
            ),
            WalkthroughStep(
                target_resolver=lambda w: w.inline_custom_distances_panel._picker._instructions_label,
                pre_action=lambda w: w.right_tabs.setCurrentIndex(2),
                title="Custom Measurements — how it works",
                body=(
                    "Select two points in the view to define a straight-line measurement. "
                    "Each custom measurement will appear as a separate column in the "
                    "measurements CSV."
                ),
            ),
            _tab_intro(
                3,
                "The Help tab",
                "Links to documentation, replays this walkthrough, and checks for TRACE " "updates.",
            ),
            WalkthroughStep(
                target_resolver=lambda w: w.inline_help_panel.btn_replay_walkthrough,
                pre_action=lambda w: w.right_tabs.setCurrentIndex(3),
                title="Help — replay this walkthrough",
                body="Re-run this guided tour anytime from here.",
            ),
            WalkthroughStep(
                target_resolver=lambda w: w.inline_help_panel.btn_check_updates,
                pre_action=lambda w: w.right_tabs.setCurrentIndex(3),
                title="Help — check for updates",
                body=(
                    "Check for a newer version of TRACE. When an update is available, an "
                    "Install Update button appears — click on that to update TRACE."
                ),
            ),
            WalkthroughStep(
                target_resolver=lambda w: w.btn_run,
                pre_action=lambda w: w.right_tabs.setCurrentIndex(0),
                title="Run the pipeline",
                body=("Click here when everything's set. Progress bar and ETA appear at the " "bottom of the window."),
            ),
        ]

    def _maybe_auto_check_updates(self) -> None:
        """Auto-check entrypoint with a short anti-spam throttle.

        Previously throttled to once per hour, which made the launch
        auto-check fire approximately never during normal testing
        (repeat-launches within an hour got silently suppressed, so
        neither the badge nor the notification dialog appeared). The
        throttle is now 60 seconds — enough to avoid hammering GitHub
        on a tight relaunch loop but short enough that the very next
        launch reliably surfaces a pending update.

        GitHub's anonymous rate limit is 60 requests/hour/IP. One
        request per launch under a 60s anti-spam window stays well
        within that even with aggressive launching.

        Manual button clicks bypass this entirely — they call
        ``InlineHelpPanel._check_for_updates`` directly.
        """
        import time as _time

        last = int(self.settings.value("last_update_check_time", 0, type=int) or 0)
        if _time.time() - last < 60:
            return
        self.inline_help_panel._check_for_updates(silent=True)

    def _update_badge_icon(self) -> "QIcon":
        """Cached blue-dot QIcon used as the Help-tab attention indicator.

        Drawn programmatically on a transparent QPixmap so the badge
        sits next to the tab text without re-coloring "Help" itself.
        Color matches the bootstrap-primary accent used elsewhere
        (IN_PROGRESS row glyphs, default progress bar fill).
        """
        from PyQt5.QtCore import Qt as _Qt
        from PyQt5.QtGui import QIcon, QPainter, QPixmap

        if getattr(self, "_cached_update_badge_icon", None) is None:
            from TRACE.theme import current_theme as _ct

            size = 12
            pix = QPixmap(size, size)
            pix.fill(_Qt.transparent)
            painter = QPainter(pix)
            painter.setRenderHint(QPainter.Antialiasing)
            painter.setBrush(QColor(_ct().accent))
            painter.setPen(_Qt.NoPen)
            painter.drawEllipse(0, 0, size, size)
            painter.end()
            self._cached_update_badge_icon = QIcon(pix)
        return self._cached_update_badge_icon

    def show_update_available_indicator(self, latest_version: str) -> None:
        """Mark the Help tab so the user notices an update.

        Sets a small blue dot as the Help tab's icon (tab text stays
        "Help" so the badge is the colored part, not the whole label).
        The badge persists across sessions: the latest-known version is
        cached in QSettings under ``cached_latest_release_version`` so a
        relaunch immediately re-applies the badge without waiting for
        the auto-check throttle window. Cleared on Help-tab activation
        for the session, and cleared from QSettings when the user
        upgrades (the next up-to-date result drops the cache key).

        The user-facing notification dialog is a separate call
        (``show_update_available_dialog``) so the launch-time cached
        restore path can re-apply the badge without re-popping the
        dialog the user already dismissed.
        """
        help_index = self.right_tabs.indexOf(self._help_tab_widget)
        if help_index < 0:
            return
        self.right_tabs.setTabIcon(help_index, self._update_badge_icon())
        self.settings.setValue("cached_latest_release_version", latest_version)

    def show_update_available_dialog(self, latest_version: str) -> None:
        """Centered, non-modal notification announcing an update.

        Fires on every TRACE launch when an update is detected (the
        launch-time cached-restore path calls this), and again on
        in-session auto-checks / manual clicks that discover a NEWER
        version than the one already shown this session. Within a
        single session the dialog is suppressed for a repeat of the
        same version so the hourly auto-check colliding with the
        launch-time fire doesn't re-pop the same dialog.

        Two buttons: Dismiss (close, badge stays as the passive
        reminder) and Update now (close + download the new installer
        and launch it — no extra clicks needed). On the launch-time
        cached-restore path the asset URL isn't populated yet; the
        Update-now handler chains the install onto the next auto-check
        result via InlineHelpPanel._install_after_next_check.
        """
        from PyQt5.QtCore import Qt as _Qt
        from PyQt5.QtGui import QFont
        from PyQt5.QtWidgets import QDialog, QHBoxLayout, QLabel, QPushButton, QVBoxLayout

        # Session-only de-dup. Per-launch repetition is exactly what
        # the user wants — they want the dialog every time TRACE
        # opens with a known pending update — so we do NOT persist
        # this attribute in QSettings.
        if getattr(self, "_dialog_fired_for_version", None) == latest_version:
            return
        try:
            from TRACE import __version__ as installed_version
        except Exception:
            installed_version = "unknown"

        # Window flags must be passed in the constructor — calling
        # setWindowFlags() AFTER construction re-parents the widget and
        # implicitly hides it, which on some macOS configurations races
        # with exec_()'s show() and leaves the dialog invisible. Frameless
        # + WindowStaysOnTop make it float over the main window like the
        # walkthrough popup. The blue accent border (matching the badge
        # dot) makes it visually distinct from system dialogs.
        # Read the active theme once at construction — the dialog is
        # short-lived (created, exec'd, deleted per launch fire) so it
        # doesn't need to re-style on theme change.
        from TRACE.theme import current_theme as _ct

        _t = _ct()
        dlg = QDialog(self, _Qt.Dialog | _Qt.FramelessWindowHint | _Qt.WindowStaysOnTopHint)
        dlg.setWindowTitle("Update available")
        dlg.setObjectName("UpdateAvailableDialog")
        dlg.setStyleSheet(
            f"#UpdateAvailableDialog {{ "
            f"background-color: {_t.dialog_bg}; "
            f"border: 2px solid {_t.dialog_border}; "
            f"border-radius: 6px; "
            f"}}"
        )
        dlg.setMinimumWidth(380)
        dlg.setMaximumWidth(480)

        layout = QVBoxLayout(dlg)
        layout.setContentsMargins(14, 12, 14, 12)
        layout.setSpacing(8)

        title = QLabel("Update available")
        title_font = QFont(title.font())
        title_font.setBold(True)
        title_font.setPointSize(title_font.pointSize() + 1)
        title.setFont(title_font)
        title.setStyleSheet(f"color: {_t.text};")
        layout.addWidget(title)

        body = QLabel(
            f"TRACE <b>{latest_version}</b> is now available — you're running <b>{installed_version}</b>.<br><br>"
            "Click <b>Update now</b> to download and launch the new installer. "
            "Your settings and downloaded models are preserved."
        )
        body.setWordWrap(True)
        body.setStyleSheet(f"color: {_t.text_body};")
        layout.addWidget(body)

        footer = QHBoxLayout()
        footer.setContentsMargins(0, 0, 0, 0)
        footer.addStretch(1)
        btn_dismiss = QPushButton("Dismiss")
        btn_dismiss.setStyleSheet(
            f"QPushButton {{ color: {_t.text_muted}; background-color: transparent; "
            f"border: 1px solid {_t.border_subtle}; border-radius: 4px; padding: 4px 10px; }} "
            f"QPushButton:hover {{ border-color: {_t.accent}; color: {_t.text_body}; }} "
            f"QPushButton:pressed {{ background-color: {_t.surface}; }}"
        )
        btn_dismiss.clicked.connect(dlg.reject)
        footer.addWidget(btn_dismiss)
        btn_update = QPushButton("Update now")
        btn_update.setDefault(True)
        btn_update.setStyleSheet(
            f"QPushButton {{ color: {_t.text}; background-color: {_t.surface}; "
            f"border: 1px solid {_t.accent}; border-radius: 4px; padding: 4px 12px; }} "
            f"QPushButton:hover {{ background-color: {_t.surface_alt}; }} "
            f"QPushButton:pressed {{ background-color: {_t.surface_pressed}; }}"
        )

        # "Update now" closes the dialog and triggers the actual
        # download + installer launch directly. Previously this was a
        # "View update" button that just switched to the Help tab so
        # the user could then click another button — one extra step
        # per upgrade for no added information.
        #
        # Two code paths reach the dialog:
        #   1. Launch-time cached-restore (_restore_update_badge_from_cache)
        #      — the asset URL isn't populated yet because the network
        #      auto-check hasn't completed. _install_update would
        #      early-return on a missing URL. We instead trigger a
        #      check + set _install_after_next_check so the install
        #      fires when results land.
        #   2. Post-auto-check (_apply_update_check_result) — URL is
        #      populated, fire the install immediately.
        def _on_update_now() -> None:
            dlg.accept()
            panel = self.inline_help_panel
            if panel._latest_update_url:
                panel._install_update()
            else:
                # No URL yet — chain the install to the next check's
                # result. _maybe_auto_check_updates may already be in
                # flight from launch; this either piggybacks on it
                # (via the in-flight guard in _check_for_updates) or
                # starts a fresh check.
                panel._install_after_next_check = True
                panel._check_for_updates(silent=True)

        btn_update.clicked.connect(_on_update_now)
        footer.addWidget(btn_update)
        layout.addLayout(footer)

        # Remember in-memory only (NOT persisted) so an hourly auto-check
        # firing the same version in the same session doesn't re-pop the
        # dialog. The next launch resets this to None and fires the
        # dialog again — by design, per user request.
        self._dialog_fired_for_version = latest_version

        # Use show() + signal handlers instead of exec_(). On Windows,
        # the combination of Qt.Dialog + FramelessWindowHint +
        # WindowStaysOnTopHint + exec_() is unreliable — the modal
        # nested event loop sometimes returns instantly without ever
        # rendering the dialog, so the user sees the Help-tab badge but
        # no notification. show() + raise_() + activateWindow() forces
        # the dialog to actually appear. v0.1.45 tried fixing this by
        # moving the flags into the constructor; that didn't address
        # the modal/frameless conflict, hence this rewrite.
        #
        # Keep a reference on the window so the dialog isn't garbage
        # collected the moment this method returns — show() is
        # non-blocking, so a local-only ref would die immediately.
        self._pending_update_dialog = dlg
        dlg.finished.connect(lambda _result: setattr(self, "_pending_update_dialog", None))

        # Center over the main window. adjustSize() resolves the
        # populated layout's sizeHint so width()/height() are accurate.
        dlg.adjustSize()
        host_geo = self.frameGeometry()
        dlg.move(
            host_geo.center().x() - dlg.width() // 2,
            host_geo.center().y() - dlg.height() // 2,
        )

        dlg.show()
        # raise_() and activateWindow() are belt-and-suspenders for
        # WindowStaysOnTopHint, which Windows ignores when the dialog
        # was constructed while the main window wasn't yet activated.
        dlg.raise_()
        dlg.activateWindow()

    def clear_update_available_indicator(self, *, clear_cache: bool = False) -> None:
        """Remove the blue dot from the Help tab.

        Default behavior (``clear_cache=False``) hides the dot for this
        session only — used by the Help-tab activation hook. The cached
        latest-version key stays in QSettings, so the next launch will
        re-apply the badge if the user still hasn't upgraded.

        ``clear_cache=True`` also drops the QSettings key — used when a
        check confirms the installed version IS the latest (so the
        badge doesn't come back on subsequent launches).
        """
        from PyQt5.QtGui import QIcon

        help_index = self.right_tabs.indexOf(self._help_tab_widget)
        if help_index >= 0:
            self.right_tabs.setTabIcon(help_index, QIcon())
        if clear_cache:
            self.settings.remove("cached_latest_release_version")

    def _restore_update_badge_from_cache(self) -> None:
        """Re-apply the Help-tab badge AND fire the dialog on launch when
        a prior session saw an update.

        Cleared-state-doesn't-persist semantics: if the user dismissed
        the badge by visiting the Help tab last session — or dismissed
        the notification dialog — but the installed version is still
        behind the cached latest, both the badge and the dialog
        reappear on relaunch. Once the user actually upgrades, the
        next ``_apply_update_check_result`` confirms up-to-date and
        drops the cache key — at which point neither the badge nor
        the dialog comes back on launch.

        Uses the same semver-tuple comparison as the fresh auto-check
        path (_apply_update_check_result → _version_is_newer). String
        inequality on raw version strings would offer a downgrade if
        the user upgraded past whatever the QSettings cache last saw
        (e.g. cached=0.1.51, installed=0.1.53 after skipping a broken
        intermediate release). A stale cache entry that's not strictly
        newer also gets dropped here so it stops re-firing on each
        launch until the next auto-check refreshes it.
        """
        cached = str(self.settings.value("cached_latest_release_version", "") or "")
        if not cached:
            return
        try:
            from TRACE import __version__ as installed_version
        except Exception:
            return
        from TRACE.inline_panels import _version_is_newer

        if not _version_is_newer(cached, installed_version):
            # Cache is stale (same or older than installed). Drop it so
            # we don't keep evaluating it on every launch — the next
            # auto-check will repopulate from GitHub.
            self.settings.remove("cached_latest_release_version")
            return
        self.show_update_available_indicator(cached)
        # Defer the dialog one tick so the main window finishes
        # showing before the centered notification appears over it.
        # The dialog dedups by version in-session, so the
        # immediately-following auto-check returning the same
        # cached version won't re-pop the dialog.
        QTimer.singleShot(0, lambda: self.show_update_available_dialog(cached))

    def _on_right_tab_changed(self, index: int) -> None:
        """Tab-change hook: clear the update-available badge on Help visit."""
        if self.right_tabs.widget(index) is self._help_tab_widget:
            self.clear_update_available_indicator()

    def _show_walkthrough(self) -> None:
        """Build a fresh overlay and start it. Called on first launch and
        from the Help tab's "Replay walkthrough" button."""
        # Tear down any existing walkthrough first — happens if the user
        # clicks "Replay walkthrough" while one is already running.
        if self._walkthrough is not None:
            try:
                self._walkthrough.finish()
            except Exception:
                pass
            self._walkthrough = None
        steps = self._walkthrough_steps()
        self._walkthrough = WalkthroughOverlay(self, steps, settings=self.settings)
        self._walkthrough.finished.connect(self._on_walkthrough_finished)
        self._walkthrough.start()

    def _on_walkthrough_finished(self) -> None:
        self._walkthrough = None

    def _on_splitter_moved(self, *_args) -> None:
        if self._walkthrough is not None:
            self._walkthrough.reposition()

    def resizeEvent(self, event):  # noqa: N802 — Qt API
        super().resizeEvent(event)
        if self._walkthrough is not None:
            self._walkthrough.reposition()

    def closeEvent(self, event):  # noqa: N802 — Qt API
        # Persist the window geometry so the next launch reopens at the size
        # and position the user last left.
        self.settings.setValue("main_window_geometry", self.saveGeometry())
        self.settings.sync()
        super().closeEvent(event)


# ---------------------------------------------------------------------------
# Dark Fusion theme + entry point
# ---------------------------------------------------------------------------
def _apply_theme(app: QApplication, theme=None) -> None:
    """Apply the active Theme's palette + app-wide stylesheet.

    Replaces the original _apply_dark_palette: pulls every color from
    the centralized Theme dataclass (TRACE/theme.py) so the same code
    path renders dark or light depending on the user's Settings pick.
    """
    from TRACE.theme import current_theme
    from TRACE.theme import manager as _theme_mgr
    from TRACE.theme import os_is_dark

    if theme is None:
        theme = current_theme()
    # Log the resolved theme + raw OS-detection signal so we can
    # diagnose "I have system set to dark but TRACE is light"-style
    # reports from the startup log without needing to re-instrument.
    try:
        from TRACE.startup_log import log as _slog

        _slog(f"theme: applying {theme.name} (pref={_theme_mgr().preference.value}, os_is_dark={os_is_dark()})")
    except Exception:
        pass
    app.setStyle("Fusion")
    p = QPalette()
    p.setColor(QPalette.Window, QColor(theme.bg))
    p.setColor(QPalette.WindowText, QColor(theme.text))
    # Base is the input-area background (line edits, list widgets, text
    # edits). Slightly darker than Window in dark mode; pure white in
    # light mode for a clean editor surface.
    p.setColor(QPalette.Base, QColor(theme.surface if theme.name == "light" else "#1e1e1e"))
    p.setColor(QPalette.AlternateBase, QColor(theme.bg))
    p.setColor(QPalette.ToolTipBase, QColor(theme.bg))
    p.setColor(QPalette.ToolTipText, QColor(theme.text))
    p.setColor(QPalette.Text, QColor(theme.text))
    p.setColor(QPalette.Button, QColor(theme.surface))
    p.setColor(QPalette.ButtonText, QColor(theme.text))
    p.setColor(QPalette.BrightText, QColor(theme.error))
    p.setColor(QPalette.Link, QColor(theme.link))
    p.setColor(QPalette.Highlight, QColor(theme.accent))
    # Highlighted text reads cleanly on a saturated accent fill in both
    # themes — keep it white rather than theme.text (which would be dark
    # in light mode and disappear into the highlight).
    p.setColor(QPalette.HighlightedText, QColor("#ffffff"))
    p.setColor(QPalette.Disabled, QPalette.Text, QColor(theme.text_disabled))
    p.setColor(QPalette.Disabled, QPalette.ButtonText, QColor(theme.text_disabled))
    app.setPalette(p)
    # Force tooltips to match the active theme — without this, on macOS
    # they fall back to the unstyled native popup and ignore the palette.
    # QGroupBox gets an explicit border so the section bounding boxes on
    # the main window stand out against the window background; the title
    # needs an explicit margin/padding so it doesn't overlap the border
    # line when we set an explicit border.
    groupbox_border = "#7a7a7a" if theme.name == "dark" else theme.border
    app.setStyleSheet(
        f"QToolTip {{ background-color: {theme.bg}; color: {theme.text};"
        f" border: 1px solid {theme.border}; padding: 4px; }}"
        f" QGroupBox {{ border: 1px solid {groupbox_border}; border-radius: 4px;"
        f" margin-top: 10px; padding-top: 6px; }}"
        f" QGroupBox::title {{ subcontrol-origin: margin;"
        f" subcontrol-position: top left; left: 8px; padding: 0 4px;"
        f" color: {theme.text}; }}"
    )


# Back-compat alias — kept so external entry points / tests that import
# _apply_dark_palette by name still work. Calls through to the
# theme-aware path with no theme override, so the user's preference wins.
_apply_dark_palette = _apply_theme


def main():
    # Reuse an existing QApplication if one was created by the launcher
    # (run_gui.py creates one early for the bootstrap progress dialog when
    # models aren't yet installed). Constructing a second QApplication in
    # the same process is undefined behavior in PyQt5.
    app = QApplication.instance() or QApplication(sys.argv)
    # Pin app + org names so QStandardPaths.AppLocalDataLocation /
    # CacheLocation return TRACE-specific paths instead of falling
    # back to the bare AppData/Cache root with no app component.
    # Without this, the cached cartoon-wing inverted PNG and the
    # cached desktop-shortcut ICO would land at e.g.
    # ~/Library/Application Support/icons/ rather than
    # ~/Library/Application Support/TRACE/icons/, polluting the
    # parent dir with files named like a sibling app.
    # Mirror QSettings("TRACE", "WingAnalysisPipeline") — org="TRACE",
    # app="WingAnalysisPipeline" — so QStandardPaths lands at the same
    # path-shape as the user's persisted settings dir.
    app.setOrganizationName("TRACE")
    app.setApplicationName("WingAnalysisPipeline")
    _apply_theme(app)
    # Re-apply the palette + app stylesheet whenever the user switches
    # themes from the Settings tab. Long-lived widgets handle their own
    # inline-stylesheet re-styling by connecting to the same signal in
    # their constructors.
    from TRACE.theme import manager as _theme_manager

    _theme_manager().themeChanged.connect(lambda t: _apply_theme(app, t))
    from TRACE._app_icon import make_app_icon

    icon = make_app_icon()
    if icon is not None:
        app.setWindowIcon(icon)
    window = TraceWindow()
    # napari's _QtMainWindow.__init__ unconditionally calls
    # QApplication.setWindowIcon during LandmarkPickerWidget construction,
    # which silently overwrites the TRACE icon set above. Re-set the app
    # icon now (post-construction) and also pin it on our main window —
    # per-window icons survive any subsequent QApplication.setWindowIcon
    # changes from anywhere in the process.
    if icon is not None:
        app.setWindowIcon(icon)
        window.setWindowIcon(icon)
    window.show()
    sys.exit(app.exec_())
