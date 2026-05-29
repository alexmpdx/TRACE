"""Report-a-bug dialog: POSTs a text-only bug report to the TRACE
bug-report Cloudflare Worker, which creates an issue in alexmpdx/TRACE.

Users don't need a GitHub account. The Worker holds a server-side
GitHub PAT as a Cloudflare secret; this client only knows the public
Worker URL.

Zip-bundle expansion (screenshot, failing images, log/settings/manifest
packaging, path scrubbing) is intentionally deferred — see
deferred_bug_report_zip_bundle.md in the project memory.
"""

from __future__ import annotations

import base64
import json
import os
import platform
import sys
import urllib.error
import urllib.request
from pathlib import Path
from typing import Optional

from PyQt5.QtCore import (
    PYQT_VERSION_STR,
    QT_VERSION_STR,
    QBuffer,
    QThread,
    QUrl,
    Qt,
    pyqtSignal,
)
from PyQt5.QtGui import QColor, QDesktopServices, QPainter, QPixmap
from PyQt5.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPlainTextEdit,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

_WORKER_URL = "https://trace-bug-reporter.alexmpdx.workers.dev"
_MIN_DESCRIPTION_CHARS = 20
_MAX_DESCRIPTION_CHARS = 10000
_REQUEST_TIMEOUT_SEC = 30

# Per-artifact size caps (chars). Total ~44KB + ~5KB header fits well under
# GitHub's 64KB issue-body limit. The Worker also enforces these as a sanity
# check.
_MAX_ARTIFACT_CHARS = {
    "run_log": 20000,
    "settings_yaml": 8000,
    "manifest_json": 8000,
    "startup_log": 8000,
}


class _BugReportThread(QThread):
    """POSTs the bug-report payload to the Worker off the GUI thread.

    Emits a single ``result`` signal with either:
      ``{"ok": True, "issue_url": str, "issue_number": int}``
      ``{"ok": False, "error": str}``
    """

    result = pyqtSignal(dict)

    def __init__(self, payload: dict, worker_url: str, parent=None) -> None:
        super().__init__(parent)
        self._payload = payload
        self._worker_url = worker_url

    def run(self) -> None:
        try:
            # Use TRACE's certifi-backed SSL context — the system trust store
            # can be out of date (especially on PyInstaller bundles and conda
            # envs), causing "certificate has expired" failures against
            # Cloudflare even when the cert is in fact valid. Same pattern
            # the auto-update check uses.
            from TRACE.fetch_assets import make_ssl_context

            data = json.dumps(self._payload).encode("utf-8")
            req = urllib.request.Request(
                self._worker_url,
                data=data,
                method="POST",
                headers={
                    "Content-Type": "application/json",
                    "User-Agent": "TRACE-bug-reporter",
                },
            )
            with urllib.request.urlopen(
                req, timeout=_REQUEST_TIMEOUT_SEC, context=make_ssl_context()
            ) as resp:
                body = json.load(resp)
        except urllib.error.HTTPError as exc:
            # The Worker returns JSON error bodies with the user-facing message;
            # fall back to the status code if parsing fails.
            msg = f"Server returned HTTP {exc.code}."
            try:
                err_body = json.loads(exc.read().decode("utf-8", errors="replace"))
                if isinstance(err_body, dict) and err_body.get("error"):
                    msg = str(err_body["error"])
            except Exception:
                pass
            self.result.emit({"ok": False, "error": msg})
            return
        except urllib.error.URLError as exc:
            self.result.emit(
                {
                    "ok": False,
                    "error": (
                        "Could not reach the bug-report server. "
                        "Check your internet connection and try again. "
                        f"(Details: {exc.reason})"
                    ),
                }
            )
            return
        except Exception as exc:  # noqa: BLE001
            self.result.emit({"ok": False, "error": f"Unexpected error: {exc}"})
            return

        if not isinstance(body, dict) or not body.get("ok"):
            err_text = (
                str(body.get("error"))
                if isinstance(body, dict) and body.get("error")
                else "Server returned an unexpected response."
            )
            self.result.emit({"ok": False, "error": err_text})
            return

        self.result.emit(
            {
                "ok": True,
                "issue_url": str(body.get("issue_url") or ""),
                "issue_number": body.get("issue_number"),
            }
        )


def _gather_system_info() -> str:
    try:
        from TRACE import __version__ as trace_version
    except Exception:
        trace_version = "unknown"
    return (
        f"TRACE version: {trace_version}\n"
        f"OS:            {platform.platform()}\n"
        f"Python:        {sys.version.splitlines()[0]}\n"
        f"Qt:            {QT_VERSION_STR}\n"
        f"PyQt:          {PYQT_VERSION_STR}\n"
        f"Frozen exe:    {getattr(sys, 'frozen', False)}"
    )


def _scrub_paths(text: str) -> str:
    """Replace the user's home directory with ~ in arbitrary text.

    Case-insensitive on Windows (C:\\Users\\Alice and C:\\users\\alice both
    occur in practice). Only the home dir is touched — non-home paths (lab
    shares, etc.) stay intact because they're typically diagnostically
    valuable.
    """
    home = str(Path.home())
    if not text or not home:
        return text
    if os.name == "nt":
        import re as _re
        return _re.sub(_re.escape(home), "~", text, flags=_re.IGNORECASE)
    return text.replace(home, "~")


def _read_capped(path: Optional[Path], max_chars: int) -> str:
    """Read a text file, keep only the last ``max_chars``, scrub paths.

    Missing or unreadable files return an empty string — callers should
    treat that as "skip this artifact."
    """
    if path is None or not path.is_file():
        return ""
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ""
    if len(text) > max_chars:
        text = f"... [truncated; full file is at {path}]\n" + text[-max_chars:]
    return _scrub_paths(text)


def _find_latest_run_folder(output_folder: Optional[Path]) -> Optional[Path]:
    """Find the most recently-modified ``run_<stamp>`` subfolder under
    ``output_folder``. Returns None if there isn't one (e.g. user hasn't
    run a pipeline at this output folder yet).
    """
    if output_folder is None or not output_folder.is_dir():
        return None
    try:
        candidates = [
            p for p in output_folder.iterdir()
            if p.is_dir() and p.name.startswith("run_")
        ]
    except OSError:
        return None
    if not candidates:
        return None
    return max(candidates, key=lambda p: p.stat().st_mtime)


# Dotted attribute paths from the TraceWindow root to widgets that show
# filesystem paths or other identifying text and should be blacked out in
# any captured screenshot. ``isVisible()`` is checked at capture time so
# widgets on inactive tabs are naturally skipped.
_REDACT_TARGETS: tuple = (
    ("input_edit",),                 # left panel, always visible
    ("output_edit",),                # left panel, always visible
    ("log_text",),                   # inside Main tab
    # Custom Measurements tab: the sample-image / landmarks paths shown
    # by the embedded picker (measurementMaker/embedded_picker.py).
    ("inline_custom_distances_panel", "_picker", "_image_edit"),
    ("inline_custom_distances_panel", "_picker", "_lm_edit"),
)


def _resolve_widget(root, attr_path):
    """Walk a dotted attribute path from ``root``. Return None if any
    step is missing — callers should treat that as 'skip this target.'"""
    obj = root
    for attr in attr_path:
        obj = getattr(obj, attr, None)
        if obj is None:
            return None
    return obj


def _capture_redacted_screenshot(window, tab_index: Optional[int] = None) -> QPixmap:
    """Grab the main window and paint black over identifying widgets.

    If ``tab_index`` is given and ``window.right_tabs`` exists, switches to
    that tab before grabbing and restores the previous tab afterwards.
    Without this the screenshot always shows whichever tab the Report-a-Bug
    button was clicked from — usually the Help tab, which is useless for
    diagnosing bugs that live in the Main tab or anywhere else.

    Redaction targets are declared in ``_REDACT_TARGETS`` as dotted paths
    from the window root. Widgets that aren't visible on the captured tab
    are skipped via ``isVisible()`` — so e.g. the log_text widget (inside
    the Main tab) and the Custom Measurements path widgets (inside their
    tab) are only redacted when their tab is actually being captured.

    Does NOT redact image_list — image basenames are usually diagnostically
    valuable, and the user is explicitly opting in to share the screenshot.
    """
    tabs = getattr(window, "right_tabs", None)
    original_index: Optional[int] = None
    if tabs is not None and tab_index is not None:
        try:
            original_index = tabs.currentIndex()
            if 0 <= tab_index < tabs.count() and tab_index != original_index:
                tabs.setCurrentIndex(tab_index)
                # Let Qt paint the new tab before we grab the pixmap.
                QApplication.processEvents()
        except Exception:
            original_index = None

    try:
        pix = window.grab()
        # Critical: redact WHILE the captured tab is still active. If we
        # restored the original tab first, widget.isVisible() would return
        # False for the captured tab's widgets and they'd never get blacked
        # out (this was a real bug — Custom Measurements path bars were
        # being missed because the redaction ran after the tab was restored
        # to Help).
        painter = QPainter(pix)
        painter.setBrush(QColor("#000000"))
        painter.setPen(Qt.NoPen)
        for attr_path in _REDACT_TARGETS:
            widget = _resolve_widget(window, attr_path)
            if widget is None or not widget.isVisible():
                continue
            # widget.mapTo(window, ...) gives coordinates relative to the
            # captured pixmap (which is the window's own paint surface).
            top_left = widget.mapTo(window, widget.rect().topLeft())
            rect = widget.rect()
            painter.drawRect(top_left.x(), top_left.y(), rect.width(), rect.height())
        painter.end()
    finally:
        if original_index is not None:
            try:
                tabs.setCurrentIndex(original_index)
            except Exception:
                pass

    return pix


def _encode_pixmap_b64(pix: QPixmap) -> str:
    """Save QPixmap as PNG bytes and return base64-encoded ASCII string."""
    buf = QBuffer()
    buf.open(QBuffer.WriteOnly)
    pix.save(buf, "PNG")
    return base64.b64encode(bytes(buf.data())).decode("ascii")


def _gather_log_artifacts(window, include: set) -> dict:
    """Collect the requested log artifacts from disk.

    ``include`` is a subset of ``{"run_log", "settings_yaml",
    "manifest_json", "startup_log"}`` naming which artifacts the user
    asked for. Anything not in the set is returned as an empty string —
    the Worker treats empty values as "skip this section."

    Reads the *current* output folder rather than the in-memory
    ``window._run_folder`` (which clears on cancel), so the right
    artifacts go out even after a cancel or after pointing TRACE at a
    previous output folder.

    Each gathered artifact is capped per ``_MAX_ARTIFACT_CHARS`` and
    path-scrubbed.
    """
    artifacts = {"run_log": "", "settings_yaml": "", "manifest_json": "", "startup_log": ""}

    # Only touch the filesystem for items the user actually asked for.
    needs_output_folder = bool(include & {"run_log", "settings_yaml", "manifest_json"})
    if needs_output_folder:
        out_text = ""
        try:
            out_text = window.output_edit.text().strip()
        except Exception:
            pass
        out_folder = Path(out_text) if out_text else None

        if out_folder is not None and out_folder.is_dir():
            if "manifest_json" in include:
                artifacts["manifest_json"] = _read_capped(
                    out_folder / "_run_state.json",
                    _MAX_ARTIFACT_CHARS["manifest_json"],
                )
            if include & {"run_log", "settings_yaml"}:
                run_folder = _find_latest_run_folder(out_folder)
                if run_folder is not None:
                    if "run_log" in include:
                        artifacts["run_log"] = _read_capped(
                            run_folder / "run.log",
                            _MAX_ARTIFACT_CHARS["run_log"],
                        )
                    if "settings_yaml" in include:
                        artifacts["settings_yaml"] = _read_capped(
                            run_folder / "settings.yaml",
                            _MAX_ARTIFACT_CHARS["settings_yaml"],
                        )

    if "startup_log" in include:
        try:
            from TRACE.startup_log import LOG_PATH as _STARTUP_LOG
            artifacts["startup_log"] = _read_capped(
                _STARTUP_LOG, _MAX_ARTIFACT_CHARS["startup_log"]
            )
        except Exception:
            pass

    return artifacts


class ReportBugDialog(QDialog):
    """Modal dialog: description + optional system info → POST to the Worker."""

    def __init__(self, window: QWidget) -> None:
        super().__init__(window)
        self._window = window
        self._submit_thread: Optional[_BugReportThread] = None
        self.setWindowTitle("Report a bug")
        self.setMinimumWidth(520)
        self._build_ui()

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)

        prompt = QLabel(
            "<b>Describe the bug</b> — what were you doing, what happened, "
            "what did you expect? (Required, minimum "
            f"{_MIN_DESCRIPTION_CHARS} characters.)"
        )
        prompt.setWordWrap(True)
        layout.addWidget(prompt)

        self.txt_description = QPlainTextEdit()
        self.txt_description.setPlaceholderText(
            "e.g. Pipeline crashed during Stage 2 on image 04. "
            "I had hinge removal turned off."
        )
        self.txt_description.setMinimumHeight(140)
        layout.addWidget(self.txt_description)

        attach_box = QGroupBox("Include in report")
        attach_layout = QVBoxLayout(attach_box)

        self.chk_sysinfo = QCheckBox("System info (OS, TRACE / Python / Qt versions)")
        self.chk_sysinfo.setChecked(True)
        self.chk_sysinfo.setToolTip(
            "A small block with version info. No personal data — just "
            "versions and OS name."
        )
        attach_layout.addWidget(self.chk_sysinfo)

        self.chk_run_log = QCheckBox("Run log (run.log from your most recent run)")
        self.chk_run_log.setChecked(True)
        self.chk_run_log.setToolTip(
            "Text log of your most recent run — which images processed, "
            "errors, per-image diagnostics. Capped at ~20KB (tail kept). "
            "Home directory replaced with ~."
        )
        attach_layout.addWidget(self.chk_run_log)

        self.chk_settings = QCheckBox("Settings (settings.yaml from your most recent run)")
        self.chk_settings.setChecked(True)
        self.chk_settings.setToolTip(
            "Pipeline configuration for the most recent run (model paths, "
            "gate thresholds, opacities, etc.). Home directory replaced with ~."
        )
        attach_layout.addWidget(self.chk_settings)

        self.chk_manifest = QCheckBox("Manifest (_run_state.json — completed / failed images)")
        self.chk_manifest.setChecked(True)
        self.chk_manifest.setToolTip(
            "Which images completed and which failed (basenames only). "
            "Small file. Home directory replaced with ~."
        )
        attach_layout.addWidget(self.chk_manifest)

        self.chk_startup_log = QCheckBox("Startup log (TRACE launch + import diagnostics)")
        self.chk_startup_log.setChecked(True)
        self.chk_startup_log.setToolTip(
            "Errors that fired during TRACE startup (import failures, "
            "missing models, etc.). Useful when TRACE crashes before you "
            "can run a pipeline. Home directory replaced with ~."
        )
        attach_layout.addWidget(self.chk_startup_log)

        # Screenshot is off by default — more identifying than the other
        # artifacts, so we make the user explicitly opt in.
        self.chk_screenshot = QCheckBox("Screenshot of the TRACE window")
        self.chk_screenshot.setChecked(False)
        self.chk_screenshot.setToolTip(
            "Captures the current TRACE window. The input/output path bars "
            "and the on-screen run log are blacked out automatically. The "
            "image list (filenames in the left pane) remains visible. You'll "
            "see a preview before it's sent."
        )
        attach_layout.addWidget(self.chk_screenshot)

        self.lbl_screenshot_warn = QLabel(
            "<span style='color:#cc8;'>⚠ The image list (filenames) is NOT "
            "redacted — check the preview before sending.</span>"
        )
        self.lbl_screenshot_warn.setWordWrap(True)
        # Word-wrapped QLabels in a QVBoxLayout often get short-changed
        # vertically because the layout uses the unwrapped sizeHint. Pinning
        # a height-for-width size policy makes Qt compute the right height.
        _sp = QSizePolicy(QSizePolicy.Preferred, QSizePolicy.MinimumExpanding)
        _sp.setHeightForWidth(True)
        self.lbl_screenshot_warn.setSizePolicy(_sp)
        self.lbl_screenshot_warn.setVisible(False)
        self.chk_screenshot.toggled.connect(self.lbl_screenshot_warn.setVisible)
        attach_layout.addWidget(self.lbl_screenshot_warn)

        # Tab selector — defaults to Main since that's where most bugs live.
        # Only shown while the screenshot checkbox is ticked.
        self._screenshot_tab_row = QWidget()
        tab_row_layout = QHBoxLayout(self._screenshot_tab_row)
        tab_row_layout.setContentsMargins(0, 0, 0, 0)
        tab_row_layout.addWidget(QLabel("Tab to capture:"))
        self.cmb_screenshot_tab = QComboBox()
        tabs = getattr(self._window, "right_tabs", None)
        if tabs is not None:
            for i in range(tabs.count()):
                self.cmb_screenshot_tab.addItem(tabs.tabText(i), i)
            # Default to Main (index 0 in the canonical layout).
            main_idx = 0
            for i in range(tabs.count()):
                if tabs.tabText(i).strip().lower() == "main":
                    main_idx = i
                    break
            self.cmb_screenshot_tab.setCurrentIndex(main_idx)
        else:
            # Stub windows / tests: no tab widget. Provide a placeholder
            # entry so the combo isn't disabled-looking.
            self.cmb_screenshot_tab.addItem("(current view)", None)
        tab_row_layout.addWidget(self.cmb_screenshot_tab)
        tab_row_layout.addStretch(1)
        self._screenshot_tab_row.setVisible(False)
        self.chk_screenshot.toggled.connect(self._screenshot_tab_row.setVisible)
        attach_layout.addWidget(self._screenshot_tab_row)

        layout.addWidget(attach_box)

        info = QLabel(
            "<span style='color:#aaa;'>No GitHub account required. The report "
            "goes to a maintainer-controlled server that files it as a GitHub "
            "issue on your behalf.</span>"
        )
        info.setWordWrap(True)
        layout.addWidget(info)

        self._status_label = QLabel("")
        self._status_label.setWordWrap(True)
        self._status_label.setOpenExternalLinks(True)
        self._status_label.setTextInteractionFlags(Qt.TextBrowserInteraction)
        layout.addWidget(self._status_label)

        buttons = QDialogButtonBox(self)
        self.btn_submit = buttons.addButton("Submit", QDialogButtonBox.AcceptRole)
        self.btn_cancel = buttons.addButton(QDialogButtonBox.Cancel)
        buttons.accepted.connect(self._on_submit)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def _on_submit(self) -> None:
        # Guard against rapid double-clicks while a request is in flight.
        if self._submit_thread is not None and self._submit_thread.isRunning():
            return

        description = self.txt_description.toPlainText().strip()
        if len(description) < _MIN_DESCRIPTION_CHARS:
            self._show_inline_error(
                f"Please provide at least {_MIN_DESCRIPTION_CHARS} characters."
            )
            return
        if len(description) > _MAX_DESCRIPTION_CHARS:
            self._show_inline_error(
                f"Description is too long ({len(description)} chars; "
                f"maximum {_MAX_DESCRIPTION_CHARS})."
            )
            return

        payload: dict = {
            "description": description,
            "sysinfo": _gather_system_info() if self.chk_sysinfo.isChecked() else "",
        }
        try:
            from TRACE import __version__ as trace_version
            payload["trace_version"] = trace_version
        except Exception:
            payload["trace_version"] = "unknown"

        include = set()
        if self.chk_run_log.isChecked():
            include.add("run_log")
        if self.chk_settings.isChecked():
            include.add("settings_yaml")
        if self.chk_manifest.isChecked():
            include.add("manifest_json")
        if self.chk_startup_log.isChecked():
            include.add("startup_log")
        if include:
            payload["artifacts"] = _gather_log_artifacts(self._window, include)

        # Screenshot capture + preview happens here (modal) so the user can
        # bail before anything network-y starts. If they cancel at preview,
        # the dialog stays open with the checkbox still ticked so they can
        # try again (or untick and submit without a screenshot).
        if self.chk_screenshot.isChecked():
            tab_index = self.cmb_screenshot_tab.currentData()
            pix = _capture_redacted_screenshot(self._window, tab_index=tab_index)
            if not self._confirm_screenshot_preview(pix):
                return
            payload["screenshot"] = {
                "content_base64": _encode_pixmap_b64(pix),
                "mime_type": "image/png",
            }

        self._set_busy_state(True)
        self._show_inline_status("Submitting…")

        self._submit_thread = _BugReportThread(payload, _WORKER_URL, parent=self)
        self._submit_thread.result.connect(self._on_result)
        self._submit_thread.finished.connect(self._submit_thread.deleteLater)
        self._submit_thread.start()

    def _on_result(self, payload: dict) -> None:
        self._set_busy_state(False)
        if payload.get("ok"):
            url = payload.get("issue_url") or ""
            num = payload.get("issue_number")
            self._status_label.setText(
                f"<span style='color: #6c6;'>✓ Submitted as issue #{num}.</span><br>"
                f"<a href='{url}' style='color: #4aa3ff;'>{url}</a>"
            )
            confirm = QMessageBox(self)
            confirm.setWindowTitle("Thanks!")
            confirm.setIcon(QMessageBox.Information)
            confirm.setText(f"Bug report submitted as issue #{num}.")
            confirm.setInformativeText(
                f"You can view it at:<br><a href='{url}'>{url}</a>"
            )
            confirm.setTextFormat(Qt.RichText)
            view_btn = confirm.addButton("View on GitHub", QMessageBox.ActionRole)
            confirm.addButton(QMessageBox.Ok)
            confirm.exec()
            if confirm.clickedButton() is view_btn and url:
                QDesktopServices.openUrl(QUrl(url))
            self.accept()
        else:
            err = payload.get("error") or "Submission failed."
            self._show_inline_error(err)

    def _confirm_screenshot_preview(self, pix: QPixmap) -> bool:
        """Show a scaled preview of the redacted screenshot. Returns True if
        the user accepts, False if they cancel.
        """
        preview = QDialog(self)
        preview.setWindowTitle("Screenshot preview")
        v = QVBoxLayout(preview)
        msg = QLabel(
            "Black bars cover the input/output path bars and the run log. "
            "The image list (filenames) is NOT redacted. OK to include this "
            "in the report?"
        )
        msg.setWordWrap(True)
        v.addWidget(msg)
        lbl = QLabel()
        lbl.setPixmap(pix.scaledToWidth(720, Qt.SmoothTransformation))
        v.addWidget(lbl)
        btns = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel, preview)
        btns.accepted.connect(preview.accept)
        btns.rejected.connect(preview.reject)
        v.addWidget(btns)
        return preview.exec() == QDialog.Accepted

    def _set_busy_state(self, busy: bool) -> None:
        self.btn_submit.setEnabled(not busy)
        self.btn_cancel.setEnabled(not busy)
        self.txt_description.setEnabled(not busy)
        self.chk_sysinfo.setEnabled(not busy)
        self.btn_submit.setText("Submitting…" if busy else "Submit")

    def _show_inline_status(self, text: str) -> None:
        self._status_label.setText(f"<span style='color: #888;'>{text}</span>")

    def _show_inline_error(self, text: str) -> None:
        self._status_label.setText(f"<span style='color: #f88;'>{text}</span>")
