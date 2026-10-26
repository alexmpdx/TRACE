"""In-GUI MCP server: lets an agent (Claude Code, Claude Desktop, the MCP
Inspector, …) drive the running TRACE window.

Off by default. Enabled from Advanced Settings → Integrations, which flips the
QSettings keys ``mcp/enabled`` / ``mcp/port``. When on, a Streamable-HTTP
`MCPServer` is served by uvicorn in a daemon thread at
``http://127.0.0.1:<port>/mcp`` and every tool call is marshalled onto the Qt
main thread through :class:`GuiBridge`, so tool bodies touch widgets exactly
the way a click would.

Layout of this module:

* ``GuiBridge`` — queued-signal + ``Future`` marshaller (the only thread-safety
  mechanism; see the class docstring).
* ``_tool_*`` functions — plain ``def f(window, …) -> dict`` bodies that run on
  the main thread. Pure functions of the window, unit-testable with a stub.
* ``build_server(service)`` — registers the ``_tool_*`` bodies as MCP tools that
  close over a :class:`McpService`.
* ``McpService`` — owned by ``TraceWindow``; start / stop / status.

The ``mcp`` and ``uvicorn`` imports are deliberately lazy (inside ``build_server``
and ``McpService.start``) so a machine without the package still runs TRACE and
the toggle just reports "mcp package not installed".
"""

from __future__ import annotations

import dataclasses
import logging
import threading
import time
from concurrent.futures import Future
from concurrent.futures import TimeoutError as FutureTimeoutError
from pathlib import Path
from typing import Any, Callable, Optional

from PyQt5.QtCore import QObject, Qt, pyqtSignal
from PyQt5.QtWidgets import QApplication

logger = logging.getLogger(__name__)

DEFAULT_PORT = 8765
MIN_PORT = 1024
MAX_PORT = 65535
HOST = "127.0.0.1"
MCP_PATH = "/mcp"

# QSettings keys (TRACE / WingAnalysisPipeline). Deliberately NOT PipelineConfig
# fields — this is a machine-local app preference like auto_update_check_enabled,
# and adding it to PipelineConfig would also trip liveSettings' tier assertion.
QSETTINGS_ENABLED = "mcp/enabled"
QSETTINGS_PORT = "mcp/port"

# How long a tool waits for the GUI thread. Run-launching tools use the short
# one: _run_pipeline may open a modal QMessageBox whose nested exec_() keeps the
# event loop spinning but never returns to our slot until the user clicks.
CALL_TIMEOUT_S = 30.0
LAUNCH_TIMEOUT_S = 2.0
STARTUP_WAIT_S = 5.0
STOP_JOIN_S = 5.0


def server_url(port: int) -> str:
    return f"http://{HOST}:{int(port)}{MCP_PATH}"


def claude_code_add_command(port: int) -> str:
    """The one-liner a user pastes into a terminal to register TRACE with Claude Code."""
    return f"claude mcp add --transport http trace {server_url(port)}"


def clamp_port(value: Any) -> int:
    try:
        port = int(value)
    except (TypeError, ValueError):
        return DEFAULT_PORT
    return min(MAX_PORT, max(MIN_PORT, port))


# ---------------------------------------------------------------------------
# Thread marshalling
# ---------------------------------------------------------------------------
class GuiBridge(QObject):
    """Run callables on the Qt main thread from any other thread.

    Construct it on the main thread (its slot then lives there). ``submit``
    emits ``request`` with ``(callable, Future)``; because the emitting thread
    differs from the receiver's, Qt's AutoConnection becomes a queued
    connection and ``_on_request`` executes inside the GUI event loop. The
    caller blocks on the Future.
    """

    request = pyqtSignal(object)

    def __init__(self, parent: Optional[QObject] = None):
        super().__init__(parent)
        self.request.connect(self._on_request)

    def submit(self, fn: Callable[[], Any]) -> "Future[Any]":
        fut: "Future[Any]" = Future()
        self.request.emit((fn, fut))
        return fut

    def call(self, fn: Callable[[], Any], timeout: float = CALL_TIMEOUT_S) -> Any:
        return self.submit(fn).result(timeout)

    def _on_request(self, payload: Any) -> None:
        fn, fut = payload
        try:
            fut.set_result(fn())
        except BaseException as exc:  # noqa: BLE001 — forwarded to the caller
            fut.set_exception(exc)


# ---------------------------------------------------------------------------
# Tool bodies (main thread; take the TraceWindow as `window`)
# ---------------------------------------------------------------------------
class GuiToolError(Exception):
    """Raised by tool bodies for failures the model should read.

    Converted to ``mcp``'s ``ToolError`` in :func:`build_server` so the bodies
    stay importable without the ``mcp`` package (tests, machines without it).
    """


def _dialog_open(window) -> bool:
    dlg = getattr(window, "_settings_dialog", None)
    return dlg is not None and dlg.isVisible()


def _modal_prompt_open() -> bool:
    app = QApplication.instance()
    return app is not None and app.activeModalWidget() is not None


def _raw_run_state(window) -> str:
    if getattr(window, "worker", None) is not None:
        return "running"
    if getattr(window, "_is_paused", False):
        return "paused"
    return "idle"


def _tool_get_run_status(window) -> dict:
    raw = _raw_run_state(window)
    modal = _modal_prompt_open()
    manifest = getattr(window, "_manifest", None)
    run_folder = getattr(window, "_run_folder", None)
    progress = getattr(window, "progress", None)
    percent: Optional[int] = None
    if progress is not None and progress.maximum() > 0 and progress.value() >= 0:
        percent = int(progress.value())
    last = getattr(window, "_last_progress", None)
    return {
        "state": "awaiting_user_prompt" if modal else raw,
        "run_state": raw,
        "awaiting_user_prompt": modal,
        "run_folder": str(run_folder) if run_folder else None,
        "total": int(manifest.total_images) if manifest else len(getattr(window, "_image_paths", [])),
        "completed": len(manifest.completed_images) if manifest else 0,
        "failed_preproc": len(manifest.failed_preproc_images) if manifest else 0,
        "failed_analysis": len(manifest.analysis_failed_images) if manifest else 0,
        "current_image": last[2] if last else None,
        "current_stage": last[3] if last else None,
        "current_detail": last[4] if last else None,
        "progress_percent": percent,
        "eta": window.eta_label.text() if hasattr(window, "eta_label") else "",
    }


def _tool_get_state(window) -> dict:
    from TRACE import __version__

    gs = window.get_gui_state()
    cfg = window.config
    return {
        "version": __version__,
        "input_folder": window.input_edit.text(),
        "recursive": bool(window.recursive_chk.isChecked()),
        "output_folder": window.output_edit.text(),
        "scale_um_per_px": cfg.um_per_px,
        "auto_detect_um_per_px": bool(getattr(cfg, "auto_detect_um_per_px", False)),
        "landmark_model": gs.get("landmark_model_path", ""),
        "segmentation_model": gs.get("segmentation_model_path", ""),
        "wing_isolation_model": gs.get("wing_isolation_model_path", ""),
        "wing_isolation_enabled": bool(gs.get("wing_isolation_enabled", False)),
        "do_rotation": bool(gs.get("do_rotation", False)),
        "outputs": sorted(window._selected_outputs()),
        "csv_groups": sorted(k for k, v in gs.get("csv_measurement_groups", {}).items() if v),
        "include_custom_measurements": bool(gs.get("include_custom_measurements", False)),
        "max_workers": gs.get("max_workers"),
        "image_count": len(getattr(window, "_image_paths", [])),
        "settings_dialog_open": _dialog_open(window),
        "run": _tool_get_run_status(window),
    }


def _tool_list_images(window) -> dict:
    rows = []
    for p in getattr(window, "_image_paths", []):
        status = window._image_status.get(p.name)
        rows.append(
            {
                "basename": p.name,
                "path": str(p),
                "status": status.value if status is not None else "pending",
                "user_skipped": p.name in window._user_skip_set,
                "error": window._image_error_text.get(p.name, ""),
            }
        )
    return {"input_folder": window.input_edit.text(), "count": len(rows), "images": rows}


def _tool_set_input_folder(window, path: str, recursive: Optional[bool] = None) -> dict:
    folder = Path(path).expanduser()
    if not folder.is_dir():
        raise GuiToolError(f"Not a directory: {folder}")
    if _raw_run_state(window) == "running":
        raise GuiToolError("A run is in progress; pause or cancel it before changing the input folder.")
    if recursive is not None:
        window.recursive_chk.setChecked(bool(recursive))
    window.input_edit.setText(str(folder))
    # textChanged only clears rerun state (gui.py _on_input_folder_changed);
    # the image list is rebuilt explicitly, same as the Browse button does.
    window._refresh_image_list()
    return _tool_list_images(window)


def _tool_set_output_folder(window, path: str) -> dict:
    folder = Path(path).expanduser()
    if _raw_run_state(window) == "running":
        raise GuiToolError("A run is in progress; pause or cancel it before changing the output folder.")
    try:
        folder.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise GuiToolError(f"Cannot create output folder {folder}: {exc}") from exc
    window.output_edit.setText(str(folder))
    return {"output_folder": str(folder)}


def _tool_set_scale(window, um_per_px: Optional[float] = None, auto_detect: Optional[bool] = None) -> dict:
    if um_per_px is None and auto_detect is None:
        raise GuiToolError("Pass um_per_px (a positive number, or 0 to clear) and/or auto_detect.")
    if auto_detect is not None:
        window._set_auto_detect_um_per_px(bool(auto_detect), source="mcp")
    if um_per_px is not None:
        if um_per_px < 0:
            raise GuiToolError("um_per_px must be >= 0 (0 clears the manual scale).")
        spin = window.scale_spin
        # Below the spinbox minimum means "no manual scale" (placeholder
        # state); setValue fires valueChanged → _set_scale(source="left"),
        # which writes config.um_per_px and mirrors the General-tab spin.
        if um_per_px == 0:
            spin.setValue(spin.minimum())
        elif um_per_px > spin.maximum():
            raise GuiToolError(f"um_per_px must be <= {spin.maximum()}")
        else:
            spin.setValue(float(um_per_px))
    cfg = window.config
    return {
        "scale_um_per_px": cfg.um_per_px,
        "auto_detect_um_per_px": bool(getattr(cfg, "auto_detect_um_per_px", False)),
    }


def _tool_set_gui_state(window, state: dict) -> dict:
    if not isinstance(state, dict):
        raise GuiToolError("state must be an object; call get_state to see the keys.")
    if _raw_run_state(window) == "running":
        raise GuiToolError("A run is in progress; pause or cancel it before changing settings.")
    known = set(window.get_gui_state())
    unknown = sorted(set(state) - known)
    if unknown:
        raise GuiToolError(f"Unknown gui_state keys: {unknown}. Known keys: {sorted(known)}")
    window.apply_gui_state(dict(state))
    return _tool_get_state(window)


def _tool_list_output_types(window) -> dict:
    from identify_features.views.csv_export import MEASUREMENT_GROUPS

    from TRACE.pipeline import INTERMEDIATE_OUTPUTS, OUTPUT_TYPES

    return {
        "final_outputs": {k: OUTPUT_TYPES.get(k, k) for k in window.output_checks},
        "intermediate_outputs": {k: OUTPUT_TYPES.get(k, k) for k in sorted(INTERMEDIATE_OUTPUTS)},
        "csv_measurement_groups": dict(MEASUREMENT_GROUPS),
        "how_to_select": (
            "set_gui_state({'output_types': {key: bool}, 'intermediate_outputs': {key: bool}, "
            "'csv_measurement_groups': {key: bool}})"
        ),
    }


def _tool_get_pipeline_config(window) -> dict:
    from TRACE.config_io import config_to_dict

    return config_to_dict(window.config)


def _apply_config(window, new_config) -> None:
    # Mirrors the tail of TraceWindow._apply_settings_dialog_result.
    window.config = new_config
    window.inline_general_panel.refresh_from_state()
    if hasattr(window, "auto_detect_um_per_px_chk"):
        window.auto_detect_um_per_px_chk.blockSignals(True)
        window.auto_detect_um_per_px_chk.setChecked(bool(getattr(new_config, "auto_detect_um_per_px", False)))
        window.auto_detect_um_per_px_chk.blockSignals(False)


def _tool_update_pipeline_config(window, changes: dict) -> dict:
    from TRACE.config_io import config_from_dict, config_to_dict

    if not isinstance(changes, dict) or not changes:
        raise GuiToolError("changes must be a non-empty object of PipelineConfig fields.")
    if _dialog_open(window):
        raise GuiToolError("Advanced Settings is open in the GUI; close it first or its OK would overwrite this edit.")
    if _raw_run_state(window) == "running":
        raise GuiToolError("A run is in progress; pause or cancel it before changing settings.")
    current = config_to_dict(window.config)
    unknown = sorted(set(changes) - set(current))
    if unknown:
        raise GuiToolError(f"Unknown PipelineConfig fields: {unknown}")
    merged = {**current, **changes}
    try:
        new_config = config_from_dict(merged)
    except (TypeError, ValueError) as exc:
        raise GuiToolError(f"Invalid config value: {exc}") from exc
    _apply_config(window, new_config)
    return config_to_dict(window.config)


def _tool_apply_preset(window, name: str) -> dict:
    from TRACE.config_io import config_to_dict
    from TRACE.presets_loader import load_presets

    presets = load_presets()
    if name not in presets:
        raise GuiToolError(f"Unknown preset {name!r}. Available: {sorted(presets)}")
    if _dialog_open(window):
        raise GuiToolError("Advanced Settings is open in the GUI; close it first or its OK would overwrite this edit.")
    if _raw_run_state(window) == "running":
        raise GuiToolError("A run is in progress; pause or cancel it before changing settings.")
    try:
        new_config = dataclasses.replace(window.config, **presets[name])
    except TypeError as exc:
        raise GuiToolError(f"Preset {name!r} could not be applied: {exc}") from exc
    _apply_config(window, new_config)
    return {"preset": name, "config": config_to_dict(window.config)}


def _tool_list_presets(window) -> dict:
    from TRACE.presets_loader import load_presets

    return {name: sorted(fields) for name, fields in load_presets().items()}


def _tool_set_skipped(window, basenames: list[str], skipped: bool) -> dict:
    if _raw_run_state(window) == "running":
        raise GuiToolError("A run is in progress; skips can only be changed between runs.")
    rows = getattr(window, "_basename_to_row", {})
    unknown = [b for b in basenames if b not in rows]
    if unknown:
        raise GuiToolError(f"Not in the current image list: {unknown}")
    target = Qt.Unchecked if skipped else Qt.Checked
    for b in basenames:
        item = window.image_list.item(rows[b])
        if item is not None and item.checkState() != target:
            # Goes through the itemChanged → _on_image_check_toggled path so
            # _user_skip_set, the row glyph and QSettings all update.
            item.setCheckState(target)
    return _tool_list_images(window)


def _tool_start_run(window) -> dict:
    raw = _raw_run_state(window)
    if raw == "running":
        return {"started": False, "reason": "A run is already in progress."}
    if raw == "paused":
        return {"started": False, "reason": "A run is paused; call resume_run (or cancel_run) instead."}
    if _dialog_open(window):
        return {"started": False, "reason": "Advanced Settings is open in the GUI; close it first."}
    # Same entry the Run button uses (minus its "settings dialog still
    # open" confirm, which the check above replaces). Validation failures
    # inside show a modal QMessageBox — reported as awaiting_user_prompt.
    window._last_progress = None
    window._run_pipeline()
    status = _tool_get_run_status(window)
    status["started"] = status["run_state"] == "running"
    return status


def _tool_pause_run(window) -> dict:
    if _raw_run_state(window) != "running":
        raise GuiToolError("No run in progress.")
    # Body of _on_run_button_clicked's pause branch without _confirm_pause().
    window.worker.pause()
    msg = "Pause requested — finishing the current image first."
    window._log(msg)
    window.transient_status_label.setText(msg)
    window.transient_status_label.show()
    window.btn_run.setEnabled(False)
    return {"pause_requested": True, "note": "The in-flight image finishes first; poll get_run_status for 'paused'."}


def _tool_resume_run(window) -> dict:
    if _raw_run_state(window) != "paused":
        raise GuiToolError("No paused run to resume.")
    if _dialog_open(window):
        return {"resumed": False, "reason": "Advanced Settings is open in the GUI; close it first."}
    window._resume_paused_run()
    status = _tool_get_run_status(window)
    status["resumed"] = status["run_state"] == "running"
    return status


def _tool_cancel_run(window) -> dict:
    if _raw_run_state(window) == "idle":
        raise GuiToolError("No run in progress.")
    # Keeps the GUI's Yes/No confirmation (a cancel discards the run), so
    # this normally returns awaiting_user_prompt until a human clicks.
    window._cancel_pipeline()
    return _tool_get_run_status(window)


def _tool_get_log(window, tail_lines: int = 200) -> dict:
    n = max(1, min(int(tail_lines), 5000))
    lines = window.log_text.toPlainText().splitlines()
    return {"total_lines": len(lines), "lines": lines[-n:]}


def _tool_run_folders(window) -> dict:
    """Main-thread half of get_results: just the two paths."""
    return {
        "output_folder": window.output_edit.text(),
        "run_folder": str(window._run_folder) if getattr(window, "_run_folder", None) else None,
    }


def collect_results(output_folder: str, run_folder: Optional[str]) -> dict:
    """Filesystem half of get_results — safe to run off the GUI thread."""
    from TRACE.run_state import load_manifest

    if not run_folder:
        raise GuiToolError("No run folder yet — nothing has been run in this session (pass run_folder explicitly).")
    rf = Path(run_folder)
    if not rf.is_dir():
        raise GuiToolError(f"Run folder does not exist: {rf}")
    out_dir = Path(output_folder) if output_folder else rf.parent
    manifest = load_manifest(rf)
    failed = []
    if manifest is not None:
        for b in manifest.failed_preproc_images:
            failed.append({"basename": b, "stage": "preprocessing", "message": manifest.failure_messages.get(b, "")})
        for b in manifest.analysis_failed_images:
            failed.append({"basename": b, "stage": "analysis", "message": manifest.failure_messages.get(b, "")})
    csv_files = sorted(str(p) for p in out_dir.glob("*.csv")) if out_dir.is_dir() else []
    subdirs = {}
    if out_dir.is_dir():
        for d in sorted(p for p in out_dir.iterdir() if p.is_dir()):
            try:
                subdirs[d.name] = sum(1 for f in d.iterdir() if f.is_file())
            except OSError:
                subdirs[d.name] = None
    return {
        "output_folder": str(out_dir),
        "run_folder": str(rf),
        "run_folder_files": sorted(p.name for p in rf.iterdir()) if rf.is_dir() else [],
        "csv_files": csv_files,
        "csv_for_this_run": [c for c in csv_files if rf.name in Path(c).name],
        "output_subdirs": subdirs,
        "manifest": dataclasses.asdict(manifest) if manifest is not None else None,
        "failed": failed,
    }


# ---------------------------------------------------------------------------
# Service: owns the server thread, started/stopped from the GUI
# ---------------------------------------------------------------------------
class _CaptureErrors(logging.Handler):
    """Grabs uvicorn's startup error text so the status line can show it."""

    def __init__(self):
        super().__init__(level=logging.ERROR)
        self.messages: list[str] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.messages.append(record.getMessage())


class McpService:
    """Lifecycle of the in-process server. Construct on the main thread."""

    def __init__(self, window):
        self.window = window
        self.bridge = GuiBridge()
        self._server = None  # uvicorn.Server
        self._thread: Optional[threading.Thread] = None
        self._port: Optional[int] = None
        self._status = "Not running"

    # -- public --------------------------------------------------------
    @property
    def running(self) -> bool:
        return (
            self._server is not None
            and self._thread is not None
            and self._thread.is_alive()
            and bool(getattr(self._server, "started", False))
            and not getattr(self._server, "should_exit", False)
        )

    @property
    def port(self) -> Optional[int]:
        return self._port

    @property
    def url(self) -> Optional[str]:
        return server_url(self._port) if self.running and self._port else None

    @property
    def status_text(self) -> str:
        return self._status

    def start(self, port: int) -> str:
        """Start serving on ``port``. Returns the new status text (never raises)."""
        if self.running:
            if self._port == port:
                return self._status
            self.stop()
        port = clamp_port(port)
        try:
            import uvicorn

            mcp = build_server(self)
            app = mcp.streamable_http_app(host=HOST)
            config = uvicorn.Config(app, host=HOST, port=port, log_level="warning", lifespan="on")
            server = uvicorn.Server(config)
        except ImportError as exc:
            self._status = f'Error: mcp package not installed ({exc.name}). Run: pip install "mcp>=2,<3"'
            logger.warning("mcp_server: %s", self._status)
            return self._status
        except Exception as exc:  # noqa: BLE001
            self._status = f"Error: could not build server: {type(exc).__name__}: {exc}"
            logger.exception("mcp_server: build failed")
            return self._status

        capture = _CaptureErrors()
        uv_log = logging.getLogger("uvicorn.error")
        uv_log.addHandler(capture)
        thread = threading.Thread(target=server.run, name="trace-mcp-server", daemon=True)
        self._server, self._thread, self._port = server, thread, port
        thread.start()
        deadline = time.monotonic() + STARTUP_WAIT_S
        while time.monotonic() < deadline:
            if getattr(server, "started", False) or not thread.is_alive():
                break
            time.sleep(0.05)
        uv_log.removeHandler(capture)
        if getattr(server, "started", False) and thread.is_alive():
            self._status = f"Running at {server_url(port)}"
            logger.info("mcp_server: %s", self._status)
        else:
            detail = capture.messages[-1] if capture.messages else "server did not start (port in use?)"
            self._status = f"Error: {detail}"
            logger.warning("mcp_server: %s", self._status)
            self._server, self._thread, self._port = None, None, None
        return self._status

    def stop(self) -> None:
        server, thread = self._server, self._thread
        self._server, self._thread, self._port = None, None, None
        if server is None:
            self._status = "Not running"
            return
        server.should_exit = True
        if thread is not None:
            thread.join(STOP_JOIN_S)
            if thread.is_alive():
                logger.warning("mcp_server: server thread still alive after %.0fs", STOP_JOIN_S)
        self._status = "Not running"

    def apply(self, enabled: bool, port: int) -> str:
        """Reconcile the running state with the (enabled, port) preference."""
        if not enabled:
            self.stop()
            return self._status
        return self.start(clamp_port(port))

    # -- helpers used by the registered tools ---------------------------
    def call(self, fn: Callable[..., Any], *args: Any, timeout: float = CALL_TIMEOUT_S) -> Any:
        try:
            return self.bridge.call(lambda: fn(self.window, *args), timeout)
        except FutureTimeoutError:
            raise GuiToolError(
                f"The GUI did not respond within {timeout:.0f}s — it is probably showing a dialog "
                "that needs a human click. Check get_run_status / the screen."
            ) from None

    def launch(self, fn: Callable[..., Any]) -> dict[str, Any]:
        """For tools that may open a modal prompt: short wait, then report."""
        fut = self.bridge.submit(lambda: fn(self.window))
        try:
            return fut.result(LAUNCH_TIMEOUT_S)
        except FutureTimeoutError:
            return {
                "started": None,
                "awaiting_user_prompt": True,
                "note": (
                    "TRACE opened a dialog (missing scale/model, JPEG warning, resume or confirm prompt) "
                    "that needs a human click. Poll get_run_status."
                ),
            }


def build_server(service: McpService):
    """Create the MCPServer and register every tool. Requires the ``mcp`` package."""
    from mcp.server import MCPServer
    from mcp.server.mcpserver.exceptions import ToolError

    from TRACE import __version__

    mcp = MCPServer(
        "TRACE",
        instructions=(
            "Controls the running TRACE wing-analysis GUI. Typical flow: get_state → set_input_folder → "
            "set_output_folder → set_scale (or auto_detect) → set_gui_state for outputs/models → start_run → "
            "wait_for_run → get_results. Everything you do is visible live in the GUI; if a tool reports "
            "awaiting_user_prompt, a dialog in TRACE needs a human click."
        ),
        version=__version__,
        log_level="WARNING",
    )

    def guarded(fn: Callable[..., Any], *args: Any, timeout: float = CALL_TIMEOUT_S) -> Any:
        try:
            return service.call(fn, *args, timeout=timeout)
        except GuiToolError as exc:
            raise ToolError(str(exc)) from None

    @mcp.tool()
    def get_state() -> dict[str, Any]:
        """Snapshot of the TRACE window: folders, scale, models, selected outputs, image count and run status."""
        return guarded(_tool_get_state)

    @mcp.tool()
    def set_input_folder(path: str, recursive: Optional[bool] = None) -> dict[str, Any]:
        """Point TRACE at a folder of wing images (optionally including subfolders) and return the image list."""
        return guarded(_tool_set_input_folder, path, recursive)

    @mcp.tool()
    def set_output_folder(path: str) -> dict[str, Any]:
        """Set (and create if needed) the folder TRACE writes overlays, CSV and run_<stamp>/ folders into."""
        return guarded(_tool_set_output_folder, path)

    @mcp.tool()
    def set_scale(um_per_px: Optional[float] = None, auto_detect: Optional[bool] = None) -> dict[str, Any]:
        """Set the manual microns-per-pixel scale (0 clears it) and/or 'detect scale from image metadata'."""
        return guarded(_tool_set_scale, um_per_px, auto_detect)

    @mcp.tool()
    def set_gui_state(state: dict[str, Any]) -> dict[str, Any]:
        """Apply GUI-level settings by key: landmark_model_path, segmentation_model_path, output_types {key: bool},
        intermediate_outputs {key: bool}, csv_measurement_groups {key: bool}, max_workers, do_rotation,
        wing_isolation_enabled and the overlay flags. Unknown keys are rejected; see get_state for current values
        and list_output_types for valid output keys."""
        return guarded(_tool_set_gui_state, state)

    @mcp.tool()
    def list_output_types() -> dict[str, Any]:
        """Valid keys for output_types, intermediate_outputs and csv_measurement_groups, with their GUI labels."""
        return guarded(_tool_list_output_types)

    @mcp.tool()
    def get_pipeline_config() -> dict[str, Any]:
        """The full PipelineConfig (Advanced Settings) as a JSON object."""
        return guarded(_tool_get_pipeline_config)

    @mcp.tool()
    def update_pipeline_config(changes: dict[str, Any]) -> dict[str, Any]:
        """Merge field changes into the PipelineConfig (same fields as get_pipeline_config). Refused while the
        Advanced Settings dialog is open or a run is in progress."""
        return guarded(_tool_update_pipeline_config, changes)

    @mcp.tool()
    def list_presets() -> dict[str, Any]:
        """Named settings presets from TRACE/presets/*.json and the fields each one sets."""
        return guarded(_tool_list_presets)

    @mcp.tool()
    def apply_preset(name: str) -> dict[str, Any]:
        """Apply a named preset (see list_presets) to the PipelineConfig."""
        return guarded(_tool_apply_preset, name)

    @mcp.tool()
    def list_images() -> dict[str, Any]:
        """Every image in the input folder with its status (pending/in_progress/succeeded/failed/skipped), skip flag and error."""
        return guarded(_tool_list_images)

    @mcp.tool()
    def set_skipped(basenames: list[str], skipped: bool = True) -> dict[str, Any]:
        """Tick (skipped=False) or untick (skipped=True) images in the list so they are excluded from the next run."""
        return guarded(_tool_set_skipped, basenames, skipped)

    @mcp.tool()
    def start_run() -> dict[str, Any]:
        """Click Run. Returns immediately; a batch takes minutes, so follow with wait_for_run or poll get_run_status.
        If TRACE needs a human (missing scale, JPEG warning, resume prompt) the reply says awaiting_user_prompt."""
        try:
            return service.launch(_tool_start_run)
        except GuiToolError as exc:
            raise ToolError(str(exc)) from None

    @mcp.tool()
    def pause_run() -> dict[str, Any]:
        """Pause after the current image finishes (same as the Pause button, without the confirm box)."""
        return guarded(_tool_pause_run)

    @mcp.tool()
    def resume_run() -> dict[str, Any]:
        """Resume a paused run from where it stopped."""
        try:
            return service.launch(_tool_resume_run)
        except GuiToolError as exc:
            raise ToolError(str(exc)) from None

    @mcp.tool()
    def cancel_run() -> dict[str, Any]:
        """Cancel the run (discards it). TRACE asks the user to confirm, so this usually reports awaiting_user_prompt."""
        try:
            return service.launch(_tool_cancel_run)
        except GuiToolError as exc:
            raise ToolError(str(exc)) from None

    @mcp.tool()
    def get_run_status() -> dict[str, Any]:
        """Run state (idle/running/paused/awaiting_user_prompt), counts, current image and stage, progress %, ETA."""
        return guarded(_tool_get_run_status)

    @mcp.tool()
    def wait_for_run(timeout_s: float = 600) -> dict[str, Any]:
        """Block until the run finishes, pauses or needs a human, or timeout_s elapses (max 3600). Returns get_run_status
        plus timed_out."""
        deadline = time.monotonic() + max(1.0, min(float(timeout_s), 3600.0))
        while True:
            status = guarded(_tool_get_run_status)
            if status["run_state"] != "running" or status["awaiting_user_prompt"]:
                status["timed_out"] = False
                return status
            if time.monotonic() >= deadline:
                status["timed_out"] = True
                return status
            time.sleep(1.0)

    @mcp.tool()
    def get_log(tail_lines: int = 200) -> dict[str, Any]:
        """The last N lines of the TRACE log pane."""
        return guarded(_tool_get_log, tail_lines)

    @mcp.tool()
    def get_results(run_folder: Optional[str] = None) -> dict[str, Any]:
        """Where the outputs are: CSV files, overlay subfolders, the run_<stamp>/ folder, its manifest and failed images.
        Defaults to the most recent run in this session."""
        folders = guarded(_tool_run_folders)
        try:
            return collect_results(folders["output_folder"], run_folder or folders["run_folder"])
        except GuiToolError as exc:
            raise ToolError(str(exc)) from None

    return mcp
