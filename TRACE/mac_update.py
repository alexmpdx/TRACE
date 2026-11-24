"""In-place update of the frozen macOS app from a downloaded disk image.

The Windows updater hands off to an Inno Setup installer. A Mac has no
installer: the release asset is a ``.dmg`` holding ``TRACE.app``, and
"updating" means replacing the installed bundle with that one. Two phases:

1. :func:`stage_update` (while TRACE is still running) mounts the disk
   image and copies the new bundle next to the installed one, so every
   failure that can be detected up front — bad image, no write permission,
   full disk — is reported in a normal dialog with the old app untouched.
2. :func:`launch_swap_helper` starts a detached shell script that waits
   for this process to exit, renames the new bundle into place and reopens
   it. Both bundles sit in the same folder, so the swap is two renames.

Stdlib-only, and nothing here imports Qt, so it can be exercised with a
fake bundle and a throwaway disk image.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Callable, Optional, Tuple

DMG_APP_NAME = "TRACE.app"
_STAGED_NAME = ".TRACE-update-new"

# The helper inherits nothing from the PyInstaller process: its bootloader
# exports variables describing *this* bundle, which must not leak into the
# relaunched one.
_CLEAN_ENV = {"PATH": "/usr/bin:/bin:/usr/sbin:/sbin", "HOME": str(Path.home())}

_SWAP_SCRIPT = r"""#!/bin/bash
# TRACE self-update, phase 2. Args: <pid to wait for> <installed app> <staged app>
PID="$1"; APP="$2"; NEW="$3"
echo "--- $(date) swapping $NEW -> $APP"
# Wait (up to 60 s) for the running TRACE to exit.
for _ in $(seq 1 300); do
  kill -0 "$PID" 2>/dev/null || break
  sleep 0.2
done
OLD="$(dirname "$APP")/.TRACE-update-old-$$"
ok=0
if mv "$APP" "$OLD"; then
  if mv "$NEW" "$APP"; then
    ok=1
    rm -rf "$OLD"
  else
    mv "$OLD" "$APP"
  fi
fi
if [ "$ok" != 1 ]; then
  echo "swap FAILED; keeping the installed version"
  rm -rf "$NEW"
  osascript -e 'display alert "TRACE update failed" message "The new version could not be put in place, so your current TRACE was left as it was. Download the latest TRACE-macOS-arm64.dmg from the Releases page to update manually."' >/dev/null 2>&1
fi
open "$APP"
echo "done (ok=$ok)"
rm -f "$0"
"""


class UpdateError(Exception):
    """The update could not be staged; the installed app is untouched."""


def bundle_path() -> Optional[Path]:
    """The ``.app`` bundle this process is running from, or None."""
    for parent in Path(sys.executable).resolve().parents:
        if parent.suffix == ".app":
            return parent
    return None


def _run(cmd: list, tick: Optional[Callable[[], None]]) -> None:
    """Run ``cmd`` to completion, calling ``tick`` while waiting (keeps a GUI alive)."""
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, env=_CLEAN_ENV)
    while proc.poll() is None:
        if tick is not None:
            tick()
        time.sleep(0.05)
    if proc.returncode != 0:
        output = (proc.stdout.read() if proc.stdout else b"").decode("utf-8", "replace").strip()
        raise UpdateError(f"{Path(cmd[0]).name} failed: {output[-400:] or f'exit code {proc.returncode}'}")


def stage_update(dmg: Path, app: Optional[Path] = None, tick: Optional[Callable[[], None]] = None) -> Tuple[Path, Path]:
    """Copy the new bundle out of ``dmg`` to sit beside the installed one.

    Returns ``(installed_app, staged_app)`` for :func:`launch_swap_helper`.
    Raises :class:`UpdateError` (nothing changed) if it can't.
    """
    app = app or bundle_path()
    if app is None:
        raise UpdateError("TRACE is not running from an app bundle.")
    if "/AppTranslocation/" in str(app):
        # Gatekeeper runs a quarantined app that was never moved by the user
        # from a randomized read-only mount; the real location is unknowable.
        raise UpdateError("TRACE is running from a temporary location. Move TRACE into Applications first.")
    if not os.access(app.parent, os.W_OK) or not os.access(app, os.W_OK):
        raise UpdateError(f"no permission to change {app}.")

    staged = app.parent / _STAGED_NAME
    shutil.rmtree(staged, ignore_errors=True)
    mount = Path(tempfile.mkdtemp(prefix="TRACE-update-mnt-"))
    try:
        _run(
            [
                "/usr/bin/hdiutil",
                "attach",
                str(dmg),
                "-nobrowse",
                "-readonly",
                "-noautoopen",
                "-mountpoint",
                str(mount),
            ],
            tick,
        )
        try:
            src = mount / DMG_APP_NAME
            if not (src / "Contents" / "MacOS").is_dir():
                raise UpdateError(f"the downloaded disk image does not contain {DMG_APP_NAME}.")
            # ditto keeps the bundle's symlinks and code signature intact.
            _run(["/usr/bin/ditto", str(src), str(staged)], tick)
        finally:
            subprocess.run(["/usr/bin/hdiutil", "detach", str(mount), "-force"], capture_output=True, env=_CLEAN_ENV)
    except BaseException:
        shutil.rmtree(staged, ignore_errors=True)
        raise
    finally:
        try:
            mount.rmdir()
        except OSError:
            pass
    # Belt and braces: a quarantine flag would make Gatekeeper block the
    # un-notarized app again on relaunch.
    subprocess.run(["/usr/bin/xattr", "-dr", "com.apple.quarantine", str(staged)], capture_output=True, env=_CLEAN_ENV)
    return app, staged


def launch_swap_helper(app: Path, staged: Path, log_path: Optional[Path] = None, pid: Optional[int] = None) -> None:
    """Start the detached script that swaps the bundles once this process exits.

    The caller must exit promptly afterwards.
    """
    if log_path is None:
        from TRACE._paths import log_dir

        log_path = log_dir() / "update.log"
    fd, script = tempfile.mkstemp(prefix="TRACE-update-", suffix=".sh")
    with os.fdopen(fd, "w") as fh:
        fh.write(_SWAP_SCRIPT)
    try:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log = open(log_path, "ab")
    except OSError:
        log = open(os.devnull, "wb")
    with log:
        subprocess.Popen(
            ["/bin/bash", script, str(pid or os.getpid()), str(app), str(staged)],
            stdin=subprocess.DEVNULL,
            stdout=log,
            stderr=log,
            start_new_session=True,  # survive this process exiting
            close_fds=True,
            env=_CLEAN_ENV,
            cwd="/",
        )


def open_disk_image(dmg: Path) -> None:
    """Fallback: show the disk image in Finder so the user can drag the app over."""
    subprocess.Popen(["/usr/bin/open", str(dmg)], env=_CLEAN_ENV)
