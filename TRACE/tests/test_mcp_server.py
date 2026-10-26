"""Tests for the in-GUI MCP server (TRACE/mcp_server.py).

Three layers, cheapest first:

1. ``GuiBridge`` marshals a callable from a worker thread onto the Qt thread
   and forwards exceptions — the one mechanism that keeps the server from
   touching widgets off-thread.
2. The ``_tool_*`` bodies against a duck-typed stub window — no TraceWindow,
   no models, no napari.
3. ``build_server`` + ``McpService`` with the real ``mcp``/``uvicorn`` stack
   (skipped when the package is missing): tools are registered, a call goes
   through ``MCPServer.call_tool``, and the HTTP server starts and stops on a
   free port.
"""

from __future__ import annotations

import asyncio
import http.client
import socket
import sys
import threading
import time
from enum import Enum
from pathlib import Path
from types import SimpleNamespace

import pytest

# Sibling modules export bare package names that don't match their directory,
# so importing TRACE.* needs the same sys.path set run_cli.py builds.
_ROOT = Path(__file__).resolve().parents[2]
for _p in (
    _ROOT,
    _ROOT / "HingeChopper",
    _ROOT / "modelTOjson",
    _ROOT / "identifyFeatures",
    _ROOT / "wingRotator",
    _ROOT / "measurementMaker",
    _ROOT / "scaleEstimator",
    _ROOT / "LandmarkLocator",
):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from PyQt5.QtCore import Qt  # noqa: E402
from PyQt5.QtWidgets import QApplication, QListWidget  # noqa: E402

from TRACE import mcp_server as ms  # noqa: E402


@pytest.fixture(scope="module")
def qapp():
    app = QApplication.instance() or QApplication([])
    yield app


def _pump(app, until, timeout=5.0):
    deadline = time.monotonic() + timeout
    while not until() and time.monotonic() < deadline:
        app.processEvents()
        time.sleep(0.01)
    assert until(), "condition not met before timeout"


# ---------------------------------------------------------------------------
# 1. GuiBridge
# ---------------------------------------------------------------------------
def test_bridge_runs_on_main_thread_and_forwards_exceptions(qapp):
    bridge = ms.GuiBridge()
    main_ident = threading.get_ident()
    out: dict = {}

    def worker():
        out["ident"] = bridge.call(threading.get_ident, timeout=5)
        try:
            bridge.call(lambda: (_ for _ in ()).throw(ValueError("boom")), timeout=5)
        except ValueError as exc:
            out["exc"] = str(exc)

    t = threading.Thread(target=worker, daemon=True)
    t.start()
    _pump(qapp, lambda: "exc" in out)
    t.join(1)
    assert out["ident"] == main_ident
    assert out["exc"] == "boom"


# ---------------------------------------------------------------------------
# 2. Tool bodies against a stub window
# ---------------------------------------------------------------------------
class _Status(Enum):
    PENDING = "pending"
    FAILED = "failed"


class _Edit:
    def __init__(self, text=""):
        self._t = text

    def text(self):
        return self._t

    def setText(self, t):  # noqa: N802 — Qt API shape
        self._t = t


class _Chk:
    def __init__(self, on=False):
        self._on = on

    def isChecked(self):  # noqa: N802
        return self._on

    def setChecked(self, on):  # noqa: N802
        self._on = bool(on)


def _stub_window(tmp_path, qapp):
    img_dir = tmp_path / "imgs"
    img_dir.mkdir()
    paths = [img_dir / "a.tif", img_dir / "b.tif"]
    for p in paths:
        p.write_bytes(b"")
    lst = QListWidget()
    for p in paths:
        lst.addItem(p.name)
        item = lst.item(lst.count() - 1)
        item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
        item.setCheckState(Qt.Checked)
    w = SimpleNamespace(
        worker=None,
        _is_paused=False,
        _settings_dialog=None,
        input_edit=_Edit(str(img_dir)),
        output_edit=_Edit(str(tmp_path / "out")),
        recursive_chk=_Chk(False),
        _image_paths=list(paths),
        _image_status={"b.tif": _Status.FAILED},
        _user_skip_set=set(),
        _image_error_text={"b.tif": "gate failed"},
        _basename_to_row={p.name: i for i, p in enumerate(paths)},
        image_list=lst,
        refreshed=0,
    )
    w._refresh_image_list = lambda: setattr(w, "refreshed", w.refreshed + 1)
    return w


def test_set_input_folder_rejects_non_dir_and_refreshes(tmp_path, qapp):
    w = _stub_window(tmp_path, qapp)
    with pytest.raises(ms.GuiToolError):
        ms._tool_set_input_folder(w, str(tmp_path / "nope"))
    res = ms._tool_set_input_folder(w, str(tmp_path / "imgs"), recursive=True)
    assert w.refreshed == 1
    assert w.recursive_chk.isChecked() is True
    assert res["count"] == 2
    assert res["images"][1] == {
        "basename": "b.tif",
        "path": str(tmp_path / "imgs" / "b.tif"),
        "status": "failed",
        "user_skipped": False,
        "error": "gate failed",
    }
    assert res["images"][0]["status"] == "pending"


def test_set_input_folder_refused_while_running(tmp_path, qapp):
    w = _stub_window(tmp_path, qapp)
    w.worker = object()
    with pytest.raises(ms.GuiToolError, match="in progress"):
        ms._tool_set_input_folder(w, str(tmp_path / "imgs"))


def test_set_output_folder_creates(tmp_path, qapp):
    w = _stub_window(tmp_path, qapp)
    target = tmp_path / "deep" / "out"
    assert ms._tool_set_output_folder(w, str(target)) == {"output_folder": str(target)}
    assert target.is_dir()
    assert w.output_edit.text() == str(target)


def test_set_skipped_toggles_rows_and_rejects_unknown(tmp_path, qapp):
    w = _stub_window(tmp_path, qapp)
    with pytest.raises(ms.GuiToolError, match="Not in the current image list"):
        ms._tool_set_skipped(w, ["zzz.tif"], True)
    ms._tool_set_skipped(w, ["a.tif"], True)
    assert w.image_list.item(0).checkState() == Qt.Unchecked
    assert w.image_list.item(1).checkState() == Qt.Checked
    ms._tool_set_skipped(w, ["a.tif"], False)
    assert w.image_list.item(0).checkState() == Qt.Checked


def test_run_status_state_mapping(tmp_path, qapp):
    w = _stub_window(tmp_path, qapp)
    assert ms._tool_get_run_status(w)["state"] == "idle"
    w._is_paused = True
    assert ms._tool_get_run_status(w)["state"] == "paused"
    w.worker = object()
    st = ms._tool_get_run_status(w)
    assert st["state"] == "running" and st["total"] == 2 and st["progress_percent"] is None
    assert ms._tool_start_run(w) == {"started": False, "reason": "A run is already in progress."}
    with pytest.raises(ms.GuiToolError):
        ms._tool_resume_run(w)


def test_update_pipeline_config_round_trips_and_rejects(tmp_path, qapp):
    from identify_features.config import PipelineConfig

    w = _stub_window(tmp_path, qapp)
    w.config = PipelineConfig()
    w.inline_general_panel = SimpleNamespace(refresh_from_state=lambda: None)
    before = ms._tool_get_pipeline_config(w)
    assert "um_per_px" in before
    with pytest.raises(ms.GuiToolError, match="Unknown PipelineConfig fields: \\['not_a_field'\\]"):
        ms._tool_update_pipeline_config(w, {"not_a_field": 1})
    after = ms._tool_update_pipeline_config(w, {"um_per_px": 0.5})
    assert after["um_per_px"] == 0.5 and w.config.um_per_px == 0.5
    # Everything else unchanged.
    assert {k: v for k, v in after.items() if k != "um_per_px"} == {k: v for k, v in before.items() if k != "um_per_px"}
    w._settings_dialog = SimpleNamespace(isVisible=lambda: True)
    with pytest.raises(ms.GuiToolError, match="Advanced Settings is open"):
        ms._tool_update_pipeline_config(w, {"um_per_px": 0.6})


def test_collect_results_reads_manifest(tmp_path):
    from TRACE.run_state import new_manifest, save_manifest

    out = tmp_path / "out"
    run = out / "run_20260101-000000"
    run.mkdir(parents=True)
    (out / "measurements_run_20260101-000000.csv").write_text("a,b\n")
    (out / "other.csv").write_text("x\n")
    (out / "vein_overlay").mkdir()
    (out / "vein_overlay" / "a.png").write_bytes(b"")
    m = new_manifest(
        input_dir=tmp_path, recursive=False, outputs_selected=["csv"], csv_measurement_groups=[], total_images=2
    )
    m.mark_completed("a.tif")
    m.failed_preproc_images.append("b.tif")
    m.failure_messages["b.tif"] = "gate"
    save_manifest(run, m)

    res = ms.collect_results(str(out), str(run))
    assert res["csv_for_this_run"] == [str(out / "measurements_run_20260101-000000.csv")]
    assert len(res["csv_files"]) == 2
    assert res["output_subdirs"]["vein_overlay"] == 1
    assert res["manifest"]["completed_images"] == ["a.tif"]
    assert res["failed"] == [{"basename": "b.tif", "stage": "preprocessing", "message": "gate"}]
    with pytest.raises(ms.GuiToolError):
        ms.collect_results(str(out), None)


# ---------------------------------------------------------------------------
# 3. Real mcp stack
# ---------------------------------------------------------------------------
mcp_pkg = pytest.importorskip("mcp", reason="mcp package not installed")


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def test_build_server_registers_tools_and_call_tool(tmp_path, qapp):
    w = _stub_window(tmp_path, qapp)
    w.log_text = SimpleNamespace(toPlainText=lambda: "one\ntwo\nthree")
    service = ms.McpService(w)
    server = ms.build_server(service)

    tools = asyncio.run(server.list_tools())
    names = {t.name for t in tools}
    assert {"get_state", "set_input_folder", "start_run", "wait_for_run", "get_results", "get_log"} <= names

    # call_tool runs the sync handler on a worker thread → bridge → the Qt
    # thread. Qt event processing must stay on the thread that owns the
    # QObjects (this one), so run the *call* on a side thread and pump here.
    result_box: dict = {}

    def run_call():
        result_box["r"] = asyncio.run(server.call_tool("get_log", {"tail_lines": 2}))

    t = threading.Thread(target=run_call, daemon=True)
    t.start()
    _pump(qapp, lambda: "r" in result_box, timeout=10)
    res = result_box["r"]
    assert not res.is_error, res
    assert res.structured_content["lines"] == ["two", "three"]


def test_service_start_stop_on_free_port(tmp_path, qapp):
    w = _stub_window(tmp_path, qapp)
    service = ms.McpService(w)
    port = _free_port()
    status = service.start(port)
    assert status == f"Running at http://127.0.0.1:{port}/mcp", status
    assert service.running and service.url.endswith(f":{port}/mcp")
    # Any HTTP answer at all proves the thread is serving; the MCP endpoint
    # rejects a bare GET without the right Accept header, which is fine.
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=5)
    conn.request("GET", "/mcp")
    assert conn.getresponse().status in (200, 400, 405, 406)
    conn.close()
    # Second start on the same port is a no-op; a different port restarts.
    assert service.start(port) == status
    service.stop()
    assert not service.running and service.status_text == "Not running"
    # Port is free again → can rebind.
    assert service.start(port).startswith("Running at")
    service.stop()


def test_service_reports_port_in_use(tmp_path, qapp):
    w = _stub_window(tmp_path, qapp)
    blocker = socket.socket()
    blocker.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    blocker.bind(("127.0.0.1", 0))
    blocker.listen(1)
    port = blocker.getsockname()[1]
    try:
        service = ms.McpService(w)
        status = service.start(port)
        assert status.startswith("Error:"), status
        assert not service.running
    finally:
        blocker.close()


def test_apply_respects_disabled(tmp_path, qapp):
    w = _stub_window(tmp_path, qapp)
    service = ms.McpService(w)
    assert service.apply(False, 8765) == "Not running"
    assert not service.running
