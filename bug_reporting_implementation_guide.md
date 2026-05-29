# In-app bug reporting: implementation guide

A self-contained recipe for adding a "Report a bug" feature to a desktop application, modeled after the implementation built for TRACE (a PyQt5 desktop tool for Drosophila wing analysis). Users without GitHub accounts can file rich bug reports — text + logs + screenshots — directly into the maintainer's GitHub Issues from inside the app.

Audience: another Claude session (or a developer) implementing this for a different desktop application. The architecture is framework-agnostic; the example code is PyQt5 but the patterns translate straightforwardly to Electron, Tauri, Tkinter, etc.

---

## TL;DR architecture

```
┌──────────────┐    HTTPS POST    ┌──────────────────┐    GitHub API     ┌─────────────┐
│  desktop app │────────────────▶│ Cloudflare       │──────────────────▶│  GitHub     │
│  (dialog)    │   { description, │ Worker           │  Issues:write     │  Issues +   │
│              │     logs,        │ (~150 lines JS)  │  Contents:write   │  orphan     │
│              │     screenshot } │                  │                    │  branch     │
└──────────────┘                  │  holds GitHub    │                    └─────────────┘
                                  │  PAT as secret   │
                                  └──────────────────┘
```

- **User-facing**: app dialog with a description field and opt-in checkboxes (sysinfo, run log, screenshot, etc.). No GitHub account required.
- **Backend**: a tiny Cloudflare Worker (free tier, ~$0/year) that holds a server-side GitHub fine-grained PAT as a secret and creates issues on behalf of the app.
- **Attachments**: an orphan `bug-attachments` branch in the target GitHub repo. Screenshots upload via the Contents API; raw URLs are embedded in the issue body for inline rendering.

---

## Why this design (decisions and the alternatives we rejected)

The driving constraint: lab biologists using a scientific tool typically don't have GitHub accounts, so a pre-filled GitHub Issues URL excludes most of the audience. Email-based options were rejected by the maintainer who didn't want to monitor an inbox. The maintainer wanted issues to land directly in GitHub for tracking alongside code.

**Rejected: gitreports.com** (third-party form-to-issue service). Eliminates the GitHub-account barrier but: text-only (no attachments), spotty reliability historically, hands UX off to a third-party form. Acceptable for an MVP but limits the ceiling.

**Rejected: embedded PAT in the app binary**. A PAT in a distributed binary is a security disaster — anyone can decompile and spam the maintainer's repo. Even with fine-grained scopes, abuse vectors are real.

**Rejected: mailto: + clipboard fallback**. Requires the maintainer to publish + monitor an email address. Maintainer explicitly wanted issues in GitHub, not an inbox.

**Rejected: Cloudflare R2 for screenshot storage**. Cleaner from a technology standpoint, but historically required credit-card validation to enable. Also adds a second piece of infra to monitor.

**Chosen: Cloudflare Worker → GitHub Issues + Contents API**. The Worker holds the PAT server-side (never in the binary), proxies the user's submission to GitHub, and gets the user a clickable issue link in seconds. Free tier covers low-volume bug reporting forever. Screenshots land on an orphan `bug-attachments` branch via the Contents API and render inline in the issue body via `raw.githubusercontent.com` URLs.

Tradeoffs honestly: ~30 minutes of one-time infrastructure setup (Cloudflare account + Worker + GitHub PAT + orphan branch), one PAT to rotate every 1–2 years. Worth it for the clean UX.

---

## Setup, in order

### Step 1 — Cloudflare account

1. Sign up at https://dash.cloudflare.com/sign-up (free, no credit card needed for Workers).
2. Skip the "add a site" prompt — Workers don't need a domain.
3. Pick a `*.workers.dev` subdomain (e.g., `alex-murphy.workers.dev`). This is permanent.

### Step 2 — GitHub fine-grained PAT

At https://github.com/settings/personal-access-tokens/new:

- Token name: `<App> bug reporter (Cloudflare Worker)`
- Expiration: 1 year recommended (set a calendar reminder to rotate)
- Repository access: **Only select repositories** → the repo where bugs should land
- Repository permissions:
  - **Issues**: Read and write
  - **Contents**: Read and write (needed only if you'll add screenshots; can skip initially)

Copy the `github_pat_...` value. Editable in place if scopes need changing later (Fine-grained PATs *can* have their scopes edited — they're not strictly immutable as widely believed).

### Step 3 — Create the orphan branch (only needed for screenshots / attachments)

The orphan branch is a separate-history branch in the same repo that holds binary attachments. Devs cloning your repo only get main by default, so screenshots don't bloat the working tree.

Create it via the GitHub API (replace `<owner>/<repo>` and `<token>`):

```bash
TOKEN=<your-PAT>
OWNER_REPO=<owner>/<repo>
README='# Bug attachment storage\n\nThis orphan branch holds image attachments from the in-app bug reporter.'

# 1. Create a blob for README.md
BLOB_SHA=$(curl -sS -X POST -H "Authorization: Bearer $TOKEN" \
  -H "Accept: application/vnd.github+json" \
  https://api.github.com/repos/$OWNER_REPO/git/blobs \
  -d "$(python3 -c "import json,sys; print(json.dumps({'content': sys.argv[1], 'encoding':'utf-8'}))" "$README")" \
  | python3 -c "import json,sys; print(json.load(sys.stdin)['sha'])")

# 2. Create a tree containing just that blob
TREE_SHA=$(curl -sS -X POST -H "Authorization: Bearer $TOKEN" \
  -H "Accept: application/vnd.github+json" \
  https://api.github.com/repos/$OWNER_REPO/git/trees \
  -d "{\"tree\":[{\"path\":\"README.md\",\"mode\":\"100644\",\"type\":\"blob\",\"sha\":\"$BLOB_SHA\"}]}" \
  | python3 -c "import json,sys; print(json.load(sys.stdin)['sha'])")

# 3. Create a commit with NO parents — this is what makes it orphan
COMMIT_SHA=$(curl -sS -X POST -H "Authorization: Bearer $TOKEN" \
  -H "Accept: application/vnd.github+json" \
  https://api.github.com/repos/$OWNER_REPO/git/commits \
  -d "{\"message\":\"Initialize bug-attachments orphan branch\",\"tree\":\"$TREE_SHA\",\"parents\":[]}" \
  | python3 -c "import json,sys; print(json.load(sys.stdin)['sha'])")

# 4. Create the ref
curl -sS -X POST -H "Authorization: Bearer $TOKEN" \
  -H "Accept: application/vnd.github+json" \
  https://api.github.com/repos/$OWNER_REPO/git/refs \
  -d "{\"ref\":\"refs/heads/bug-attachments\",\"sha\":\"$COMMIT_SHA\"}"
```

Verify: `git ls-remote origin bug-attachments` should now return the commit SHA. Visit `https://github.com/<owner>/<repo>/tree/bug-attachments` to see the branch with just `README.md`.

### Step 4 — Deploy the Worker

Create a directory anywhere convenient (we use `cloudflare-worker-bug-reporter/` as a sibling of the app source). Inside it:

**`wrangler.jsonc`** — Worker config:

```jsonc
{
  "$schema": "node_modules/wrangler/config-schema.json",
  "name": "your-app-bug-reporter",
  "main": "src/index.js",
  "compatibility_date": "2025-05-01",
  "observability": { "enabled": true }
}
```

**`src/index.js`** — Worker code. Replace `<owner>/<repo>` with the target repo:

```javascript
const ARTIFACT_SECTIONS = [
  { key: "run_log",       label: "run.log",                     lang: "",     max: 20000 },
  { key: "settings_yaml", label: "settings.yaml",               lang: "yaml", max: 8000 },
  { key: "manifest_json", label: "manifest.json",               lang: "json", max: 8000 },
  { key: "startup_log",   label: "startup.log",                 lang: "",     max: 8000 },
];
const MAX_BODY_CHARS = 60000;                  // GitHub issue body cap is 65,536
const MAX_SCREENSHOT_B64_CHARS = 4 * 1024 * 1024;  // ~3MB binary
const ATTACHMENTS_BRANCH = "bug-attachments";
const REPO = "<owner>/<repo>";

export default {
  async fetch(request, env) {
    if (request.method === "OPTIONS") return new Response(null, { headers: corsHeaders() });
    if (request.method !== "POST") return jsonError("Method not allowed.", 405);

    let payload;
    try { payload = await request.json(); } catch { return jsonError("Invalid JSON.", 400); }

    const description = (payload.description || "").toString().trim();
    const sysinfo = (payload.sysinfo || "").toString().trim();
    const appVersion = (payload.app_version || "unknown").toString().trim();
    const artifacts = (payload.artifacts && typeof payload.artifacts === "object") ? payload.artifacts : {};
    const screenshot = (payload.screenshot && typeof payload.screenshot === "object") ? payload.screenshot : null;

    if (description.length < 20) return jsonError("Description must be at least 20 characters.", 400);
    if (description.length > 10000) return jsonError("Description must be at most 10000 characters.", 400);
    if (screenshot && typeof screenshot.content_base64 === "string" &&
        screenshot.content_base64.length > MAX_SCREENSHOT_B64_CHARS) {
      return jsonError("Screenshot exceeds size cap (~3 MB binary).", 400);
    }

    // Upload screenshot first so we can inline its URL in the body. Upload
    // failure does NOT block issue creation — the body gets a transparency note.
    let screenshotUrl = null;
    let screenshotError = null;
    if (screenshot && typeof screenshot.content_base64 === "string" && screenshot.content_base64.length > 0) {
      const result = await uploadScreenshot(env, screenshot);
      if (result.url) screenshotUrl = result.url;
      else { screenshotError = result.error; console.error(`Screenshot upload failed: ${screenshotError}`); }
    }

    const title = `[bug] ${description.split("\n")[0].slice(0, 80)}`;
    const bodyLines = ["**Description**", description, "", `**App version:** ${appVersion}`];
    if (screenshotUrl) bodyLines.push("", "**Screenshot**", "", `![Screenshot](${screenshotUrl})`);
    else if (screenshotError) bodyLines.push("", `_(Screenshot upload failed: ${screenshotError}.)_`);
    if (sysinfo) bodyLines.push("", "**System info**", "```", sysinfo, "```");

    for (const { key, label, lang, max } of ARTIFACT_SECTIONS) {
      let content = (artifacts[key] || "").toString();
      if (!content) continue;
      if (content.length > max) content = `... [truncated to last ${max} chars]\n` + content.slice(-max);
      bodyLines.push("", `<details><summary>${label}</summary>`, "", "```" + lang, content, "```", "</details>");
    }
    bodyLines.push("", "_Submitted via in-app bug reporter._");

    let body = bodyLines.join("\n");
    if (body.length > MAX_BODY_CHARS) body = body.slice(0, MAX_BODY_CHARS) + "\n\n... [body truncated]";

    const ghResp = await fetch(`https://api.github.com/repos/${REPO}/issues`, {
      method: "POST",
      headers: githubHeaders(env),
      body: JSON.stringify({ title, body }),
    });
    if (!ghResp.ok) {
      const text = await ghResp.text();
      console.error(`GitHub error ${ghResp.status}: ${text.slice(0, 200)}`);
      return jsonError(`GitHub returned ${ghResp.status}.`, 502);
    }
    const issue = await ghResp.json();
    return new Response(JSON.stringify({
      ok: true, issue_url: issue.html_url, issue_number: issue.number,
      screenshot_url: screenshotUrl, screenshot_error: screenshotError,
    }), { status: 200, headers: jsonHeaders() });
  },
};

async function uploadScreenshot(env, screenshot) {
  const filename = `${crypto.randomUUID()}.png`;
  const resp = await fetch(`https://api.github.com/repos/${REPO}/contents/${filename}`, {
    method: "PUT",
    headers: githubHeaders(env),
    body: JSON.stringify({
      message: `Upload screenshot ${filename}`,
      content: screenshot.content_base64,
      branch: ATTACHMENTS_BRANCH,
    }),
  });
  if (!resp.ok) {
    const text = await resp.text();
    return { error: `HTTP ${resp.status}: ${text.slice(0, 200)}` };
  }
  return { url: `https://raw.githubusercontent.com/${REPO}/${ATTACHMENTS_BRANCH}/${filename}` };
}

function githubHeaders(env) {
  return {
    "Authorization": `Bearer ${env.GITHUB_PAT}`,
    "Accept": "application/vnd.github+json",
    "X-GitHub-Api-Version": "2022-11-28",
    "User-Agent": "bug-reporter-worker",
    "Content-Type": "application/json",
  };
}
function jsonError(message, status) { return new Response(JSON.stringify({ ok: false, error: message }), { status, headers: jsonHeaders() }); }
function jsonHeaders() { return { "Content-Type": "application/json", ...corsHeaders() }; }
function corsHeaders() {
  return {
    "Access-Control-Allow-Origin": "*",
    "Access-Control-Allow-Methods": "POST, OPTIONS",
    "Access-Control-Allow-Headers": "Content-Type",
  };
}
```

Deploy using a temporary `.env` file containing the Cloudflare API token (generated at https://dash.cloudflare.com/profile/api-tokens with the "Edit Cloudflare Workers" template):

```bash
echo "CLOUDFLARE_API_TOKEN=<your-cf-token>" > .env
echo "GITHUB_PAT_FOR_WORKER=<your-github-pat>" >> .env

set -a && source .env && set +a
npx --yes wrangler@latest deploy

# Set the GitHub PAT as a Worker secret (read from env to avoid command-history exposure)
printf '%s' "$GITHUB_PAT_FOR_WORKER" | npx --yes wrangler@latest secret put GITHUB_PAT
```

Add `.env` and `.wrangler/` to `.gitignore`.

The Worker URL will be `https://your-app-bug-reporter.<your-subdomain>.workers.dev`. Test with curl:

```bash
curl -X POST https://your-app-bug-reporter.<subdomain>.workers.dev \
  -H "Content-Type: application/json" \
  -H "User-Agent: bug-reporter-client" \
  -d '{"description":"Smoke test from curl. Please close this issue.","app_version":"smoke-test"}'
```

Expect `{"ok":true,"issue_url":"https://github.com/.../issues/N","issue_number":N,...}` and a real issue in the target repo.

---

## App-side integration

### MVP dialog (text-only, no attachments)

The minimum viable client is a modal dialog with a description field that POSTs to the Worker. PyQt5 reference (~80 lines including imports — adapt to your framework):

```python
import json, platform, sys, urllib.error, urllib.request
from PyQt5.QtCore import QThread, Qt, pyqtSignal
from PyQt5.QtWidgets import (
    QApplication, QDialog, QDialogButtonBox, QLabel, QMessageBox,
    QPlainTextEdit, QVBoxLayout,
)

WORKER_URL = "https://your-app-bug-reporter.<subdomain>.workers.dev"

class _BugReportThread(QThread):
    result = pyqtSignal(dict)
    def __init__(self, payload, parent=None):
        super().__init__(parent); self._payload = payload
    def run(self):
        try:
            req = urllib.request.Request(
                WORKER_URL,
                data=json.dumps(self._payload).encode("utf-8"),
                method="POST",
                headers={"Content-Type": "application/json", "User-Agent": "bug-reporter-client"},
            )
            with urllib.request.urlopen(req, timeout=30) as resp:
                body = json.load(resp)
            if body.get("ok"):
                self.result.emit({"ok": True, "issue_url": body.get("issue_url", ""),
                                  "issue_number": body.get("issue_number")})
            else:
                self.result.emit({"ok": False, "error": str(body.get("error") or "Unknown server error")})
        except urllib.error.HTTPError as exc:
            msg = f"HTTP {exc.code}"
            try:
                err = json.loads(exc.read().decode("utf-8", errors="replace"))
                if isinstance(err, dict) and err.get("error"): msg = str(err["error"])
            except Exception: pass
            self.result.emit({"ok": False, "error": msg})
        except urllib.error.URLError as exc:
            self.result.emit({"ok": False, "error": f"Network error: {exc.reason}"})
        except Exception as exc:
            self.result.emit({"ok": False, "error": f"Unexpected: {exc}"})


class ReportBugDialog(QDialog):
    def __init__(self, parent_window):
        super().__init__(parent_window)
        self._submit_thread = None
        self.setWindowTitle("Report a bug")
        self.setMinimumWidth(520)
        layout = QVBoxLayout(self)
        layout.addWidget(QLabel("<b>Describe the bug</b> (minimum 20 characters):"))
        self.txt = QPlainTextEdit()
        self.txt.setPlaceholderText("What were you doing, what happened, what did you expect?")
        self.txt.setMinimumHeight(140)
        layout.addWidget(self.txt)
        self._status = QLabel(""); self._status.setWordWrap(True); layout.addWidget(self._status)
        btns = QDialogButtonBox(self)
        self.btn_submit = btns.addButton("Submit", QDialogButtonBox.AcceptRole)
        btns.addButton(QDialogButtonBox.Cancel)
        btns.accepted.connect(self._on_submit)
        btns.rejected.connect(self.reject)
        layout.addWidget(btns)

    def _on_submit(self):
        if self._submit_thread and self._submit_thread.isRunning(): return
        desc = self.txt.toPlainText().strip()
        if len(desc) < 20:
            self._status.setText("<span style='color:#f88;'>At least 20 characters please.</span>"); return
        payload = {"description": desc, "app_version": "1.0.0",
                   "sysinfo": f"{platform.platform()} / Python {sys.version.splitlines()[0]}"}
        self.btn_submit.setEnabled(False); self.btn_submit.setText("Submitting…")
        self.txt.setEnabled(False)
        self._status.setText("<span style='color:#888;'>Submitting…</span>")
        self._submit_thread = _BugReportThread(payload, parent=self)
        self._submit_thread.result.connect(self._on_result)
        self._submit_thread.finished.connect(self._submit_thread.deleteLater)
        self._submit_thread.start()

    def _on_result(self, payload):
        self.btn_submit.setEnabled(True); self.btn_submit.setText("Submit")
        self.txt.setEnabled(True)
        if payload.get("ok"):
            url = payload["issue_url"]; num = payload["issue_number"]
            QMessageBox.information(self, "Thanks!",
                f"Submitted as issue #{num}.\n\n{url}")
            self.accept()
        else:
            self._status.setText(f"<span style='color:#f88;'>{payload.get('error')}</span>")
```

That's the MVP. Hook it to a "Report a bug…" button in your help/about menu.

### Critical app-side details

**Always set a custom `User-Agent` header on the request.** Cloudflare blocks the bare `Python-urllib/<version>` UA as a casual-abuse filter. Set anything specific to your app (e.g., `User-Agent: my-app-bug-reporter`). Curl, Electron's `fetch`, and most browsers send acceptable UAs by default; the gotcha is Python's stdlib.

**Use a background thread** for the network call. A 30-second timeout that freezes the GUI is unacceptable. QThread for PyQt5, `worker_threads` or `fetch` for Electron, `tokio` for Tauri.

**Validate client-side** (length checks) to mirror the Worker's validation. The Worker is the source of truth for caps but client-side validation gives instant feedback.

---

## Upgrade 1: System info

Add a "Include system info" checkbox (default ON — cheap, no PII). When checked, append a small block to the payload:

```python
def _gather_system_info():
    try: from your_app import __version__ as app_version
    except Exception: app_version = "unknown"
    from PyQt5.QtCore import QT_VERSION_STR, PYQT_VERSION_STR
    return (f"App version: {app_version}\n"
            f"OS:          {platform.platform()}\n"
            f"Python:      {sys.version.splitlines()[0]}\n"
            f"Qt:          {QT_VERSION_STR}\n"
            f"PyQt:        {PYQT_VERSION_STR}\n"
            f"Frozen exe:  {getattr(sys, 'frozen', False)}")
```

Pass as `payload["sysinfo"]` when the checkbox is ticked. The Worker wraps it in a code fence in the issue body.

---

## Upgrade 2: Text artifacts (logs / settings / manifest)

Add per-artifact opt-in checkboxes (default ON). When ticked, read each artifact from disk, scrub identifying paths, apply size caps, and send as `payload["artifacts"]`.

**Path scrubbing** — replace the user's home directory with `~`. Cheap, preserves diagnostic value, hides the most personally-identifying piece:

```python
import os, re
from pathlib import Path

def _scrub_paths(text: str) -> str:
    home = str(Path.home())
    if not text or not home: return text
    if os.name == "nt":
        return re.sub(re.escape(home), "~", text, flags=re.IGNORECASE)
    return text.replace(home, "~")
```

**Size caps** — keep each artifact small enough that GitHub's 64KB issue-body limit isn't a concern after they're all combined:

```python
_MAX_ARTIFACT_CHARS = {"run_log": 20000, "settings_yaml": 8000, "manifest_json": 8000, "startup_log": 8000}

def _read_capped(path, max_chars):
    if path is None or not path.is_file(): return ""
    try: text = path.read_text(encoding="utf-8", errors="replace")
    except OSError: return ""
    if len(text) > max_chars:
        text = f"... [truncated; full file at {path}]\n" + text[-max_chars:]
    return _scrub_paths(text)

def _gather_log_artifacts(include: set) -> dict:
    """Only touch the filesystem for items the user actually requested."""
    artifacts = {"run_log": "", "settings_yaml": "", "manifest_json": "", "startup_log": ""}
    # ... read from disk based on `include` set membership; return dict
    return artifacts
```

The Worker turns each non-empty artifact into a collapsed `<details>` block in the issue body — no further app-side work.

**Don't scrub the live UI log.** Only scrub the bundled-for-submission copy. Users debugging their own runs want to see real paths; silently rewriting their on-screen log to use `~` breaks copy-paste workflows on Windows (where `~` doesn't resolve in Explorer).

---

## Upgrade 3: Screenshots with automatic redaction

Requires the orphan branch (Step 3) and the Contents:Write permission on the PAT. The flow:

1. App captures the main window as a QPixmap.
2. Paints black rectangles over widgets containing identifying information (path bars, log windows).
3. Shows a preview dialog so the user can sanity-check before submission.
4. Base64-encodes the PNG bytes.
5. Sends as `payload["screenshot"]`. Worker uploads to the orphan branch via Contents API and embeds the public raw URL in the issue body.

```python
import base64
from PyQt5.QtCore import QBuffer, Qt
from PyQt5.QtGui import QColor, QPainter, QPixmap

# Dotted attribute paths from your main window root to widgets containing
# identifying info. isVisible() filters out widgets on inactive tabs.
_REDACT_TARGETS = (
    ("input_edit",),                                          # left panel
    ("output_edit",),                                         # left panel
    ("log_text",),                                            # main tab
    ("custom_panel", "_picker", "_image_edit"),               # nested under a tab
)

def _resolve_widget(root, attr_path):
    obj = root
    for attr in attr_path:
        obj = getattr(obj, attr, None)
        if obj is None: return None
    return obj

def _capture_redacted_screenshot(window, tab_index=None):
    """Optionally switch tabs before capture; restore the tab afterwards.

    CRITICAL: do the redaction WHILE the target tab is still active.
    If you restore the tab first, widget.isVisible() on the captured tab's
    widgets returns False (because they just got hidden) and they never get
    blacked out. This bit us in development.
    """
    tabs = getattr(window, "right_tabs", None)
    original_index = None
    if tabs is not None and tab_index is not None:
        try:
            original_index = tabs.currentIndex()
            if 0 <= tab_index < tabs.count() and tab_index != original_index:
                tabs.setCurrentIndex(tab_index)
                QApplication.processEvents()  # let Qt repaint
        except Exception:
            original_index = None
    try:
        pix = window.grab()
        # Redact inside try: tab is still on tab_index here.
        painter = QPainter(pix)
        painter.setBrush(QColor("#000000"))
        painter.setPen(Qt.NoPen)
        for attr_path in _REDACT_TARGETS:
            widget = _resolve_widget(window, attr_path)
            if widget is None or not widget.isVisible(): continue
            top_left = widget.mapTo(window, widget.rect().topLeft())
            rect = widget.rect()
            painter.drawRect(top_left.x(), top_left.y(), rect.width(), rect.height())
        painter.end()
    finally:
        if original_index is not None:
            try: tabs.setCurrentIndex(original_index)
            except Exception: pass
    return pix

def _encode_pixmap_b64(pix):
    buf = QBuffer(); buf.open(QBuffer.WriteOnly)
    pix.save(buf, "PNG")
    return base64.b64encode(bytes(buf.data())).decode("ascii")
```

Add to the dialog:
- A "Screenshot of the app window" checkbox (default OFF — most identifying artifact, opt in deliberately).
- A warning label that becomes visible when ticked, explaining which widgets are NOT redacted (anything not in `_REDACT_TARGETS` is fair game and visible).
- A "Tab to capture" QComboBox (for multi-tab apps) — defaults to the most diagnostically useful tab.
- A preview QDialog showing the captured pixmap scaled to a reasonable size, with OK / Cancel. Cancel returns to the form with the checkbox still ticked so the user can re-capture after fixing whatever was wrong.

In `_on_submit`, before starting the background thread:

```python
if self.chk_screenshot.isChecked():
    tab_index = self.cmb_screenshot_tab.currentData()
    pix = _capture_redacted_screenshot(self._window, tab_index=tab_index)
    if not self._confirm_screenshot_preview(pix):
        return  # user cancelled at preview — keep form open
    payload["screenshot"] = {"content_base64": _encode_pixmap_b64(pix), "mime_type": "image/png"}
```

The preview dialog is a few-line QDialog wrapping a QLabel with the scaled pixmap:

```python
def _confirm_screenshot_preview(self, pix):
    preview = QDialog(self)
    preview.setWindowTitle("Screenshot preview")
    v = QVBoxLayout(preview)
    msg = QLabel("Black bars cover the path widgets and the log. OK to include?")
    msg.setWordWrap(True); v.addWidget(msg)
    lbl = QLabel(); lbl.setPixmap(pix.scaledToWidth(720, Qt.SmoothTransformation))
    v.addWidget(lbl)
    btns = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel, preview)
    btns.accepted.connect(preview.accept); btns.rejected.connect(preview.reject)
    v.addWidget(btns)
    return preview.exec() == QDialog.Accepted
```

---

## Pitfalls we hit (and you'll probably hit too)

1. **`Python-urllib` is bot-blocked by Cloudflare.** Always set an explicit `User-Agent` on the request from the app side.

2. **Tab-restore race in screenshot redaction.** If your app has tabs and the screenshot can capture a tab other than the currently-active one, do the redaction *before* you restore the original tab. Otherwise `widget.isVisible()` on the captured tab's widgets returns False and they don't get blacked out. Symptom: redaction works fine on widgets that are always visible (path bars on a fixed left panel), silently fails on widgets that live inside the tab page.

3. **Word-wrapped QLabel sizing in QVBoxLayout/QGroupBox.** Wrapped labels often render with clipped bottom lines because the layout uses the unwrapped sizeHint. Fix: `QSizePolicy(Preferred, MinimumExpanding)` with `setHeightForWidth(True)`, or shorten the text.

4. **GitHub issue body cap is 65,536 chars.** Cap each artifact aggressively (we use 20KB for run.log + 8KB for the others; total ~44KB + ~5KB header fits comfortably). Have the Worker apply a final body cap as a sanity net.

5. **Fine-grained PATs are editable in place.** Common misconception they're immutable — they're not. If you need to expand scopes (e.g., adding Contents:Write later for screenshot uploads), edit the existing PAT rather than generating a new one and rotating.

6. **Public repo required for raw.githubusercontent.com.** The orphan branch must be in a public repo for the inline screenshot URL to render without authentication. If the repo is private, the screenshot URLs will be inaccessible and the issue will show a broken image.

7. **GitHub strips `data:` URIs from issue bodies.** Don't try to inline base64 PNGs directly — they get stripped for security reasons. You need a real HTTPS URL, which is why the orphan branch / Contents API approach exists.

8. **The `data:` URI ban also means base64 in markdown won't work.** Same root cause.

9. **Don't embed any PAT in the binary.** Anyone who decompiles can spam your repo. The whole point of the Worker is to keep the PAT server-side.

10. **Rate limit awareness.** GitHub allows 5000 API requests/hour for an authenticated PAT. Cloudflare Workers free tier is 100k requests/day. Both far exceed any realistic bug-report volume, but worth knowing the numbers.

---

## Verification checklist

After deploying:

- [ ] `curl` smoke test against the Worker creates a real GitHub issue.
- [ ] App dialog opens cleanly; description-length validation works.
- [ ] Submitting from the app creates an issue with the user's description.
- [ ] System info appears in the issue body when checkbox is ticked.
- [ ] Text artifacts appear as `<details>` blocks when their checkboxes are ticked.
- [ ] Paths in bundled text are scrubbed (home dir → `~`); UI log is unchanged.
- [ ] Screenshot capture switches to the selected tab, captures, and restores.
- [ ] Screenshot preview shows the redaction correctly before submission.
- [ ] Screenshot URL renders inline in the GitHub issue.
- [ ] Network failures show a user-friendly inline error; dialog stays open for retry.
- [ ] Browser open failures (if you use the open-issue-in-browser pattern) have a graceful fallback.
- [ ] After clicking Submit, the GUI doesn't freeze (background thread doing its job).

---

## Maintenance

- **PAT rotation**: Generate a new fine-grained PAT, run `npx wrangler secret put GITHUB_PAT` to update the Worker secret, revoke the old one. ~3 minutes total.
- **Cleaning up old screenshots**: The orphan branch accumulates PNGs over time. Easy to clean periodically via `gh api -X DELETE /repos/<owner>/<repo>/contents/<filename>?branch=bug-attachments` or by force-pushing a new empty commit.
- **Worker URL changes**: If you ever need to migrate (e.g. consolidate Workers), the Worker URL is hard-coded as a constant in the app. Bump the version, ship a new release.
- **Repo name changes**: The Worker hard-codes the target repo. Same: bump and ship.

---

## What this is NOT

- A crash reporter. The dialog needs the GUI to be alive to be opened. For uncaught-exception crashes, install a `sys.excepthook` that writes a diagnostic file to disk; users can attach it manually after relaunch. (We considered + deferred auto-prompt-on-crash because the GUI is often dead by then.)
- A telemetry pipeline. Bug reports are explicit user actions, not background events.
- A support ticketing system. Issues land in your GitHub Issues; triage and reply via normal GitHub workflows.

For richer telemetry / crash reporting, Sentry is the right call. This system complements it; doesn't replace it.

---

## Total cost

- **One-time**: ~30 minutes setup (Cloudflare + GitHub PAT + orphan branch + Worker deploy + initial app integration).
- **Ongoing**: ~5 minutes/year to rotate the PAT. Cloudflare free tier covers ~10,000 bug reports/month. GitHub Issues API covers 5,000 requests/hour. Both far above any realistic volume.
- **$0/year** in actual money.
