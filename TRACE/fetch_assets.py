"""Download bundled DL model weights from the GitHub Releases page on demand.

Models (landmarks, vein/intervein segmentation, wing isolation) live as a
single zip attached to a GitHub Release (tag ``v1.0-assets``) rather than
in git itself — they're ~1.6 GB combined and would either reject from
GitHub (per-file 100 MB cap) or require Git LFS.

Call `ensure_assets()` before launching anything that needs the models.
The first call downloads the zip, verifies its SHA-256 against the
hardcoded value below, and unpacks it into ``TRACE/models/``. Subsequent
calls are no-ops once the folder exists.

Set ``TRACE_ASSETS_URL`` in the environment to override the download URL
(useful for development against a mirror or a private fork).
"""

from __future__ import annotations

import hashlib
import os
import shutil
import sys
import tempfile
import urllib.request
import zipfile
from pathlib import Path
from typing import Optional

# GitHub Release asset URL. Bound to a specific tag so the SHA-256 lock
# below remains meaningful — re-publishing requires a new tag + URL + SHA.
_DEFAULT_ASSET_URL = "https://github.com/alexmpdx/TRACE/releases/download/v1.0-assets/trace_models.zip"
# SHA-256 of the zip. Computed at release time; download is rejected if
# the actual file hashes to anything else (guards against truncated /
# corrupted downloads and against silent re-publish of the asset).
_EXPECTED_SHA256 = "0d05ebc0ef566fe5d83362f6b38ae874525b18c41b5bb7d75708a9265cb19070"

_TRACE_DIR = Path(__file__).resolve().parent
_MODELS_DIR = _TRACE_DIR / "models"


def _asset_url() -> str:
    return os.environ.get("TRACE_ASSETS_URL") or _DEFAULT_ASSET_URL


def _has_models() -> bool:
    """True when models/ exists and contains at least one .pt checkpoint."""
    if not _MODELS_DIR.is_dir():
        return False
    return any(_MODELS_DIR.rglob("*.pt"))


def _sha256(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            buf = f.read(chunk)
            if not buf:
                break
            h.update(buf)
    return h.hexdigest()


def _download(url: str, dest: Path, *, progress: bool = True) -> None:
    """Stream a URL to disk with an optional one-line progress indicator.

    Uses urllib so we don't add a third-party dependency. The progress
    indicator writes to stderr so it doesn't pollute stdout if a caller
    is parsing it.
    """
    req = urllib.request.Request(url, headers={"User-Agent": "TRACE-fetch_assets"})
    with urllib.request.urlopen(req) as resp:
        total = int(resp.headers.get("Content-Length") or 0)
        with open(dest, "wb") as out:
            downloaded = 0
            chunk = 1 << 20  # 1 MB
            last_pct = -1
            while True:
                buf = resp.read(chunk)
                if not buf:
                    break
                out.write(buf)
                downloaded += len(buf)
                if progress and total:
                    pct = int(downloaded * 100 / total)
                    if pct != last_pct:
                        sys.stderr.write(
                            f"\rDownloading models: {pct:3d}%  ({downloaded // (1024 * 1024)} / "
                            f"{total // (1024 * 1024)} MB)"
                        )
                        sys.stderr.flush()
                        last_pct = pct
            if progress and total:
                sys.stderr.write("\n")


def ensure_assets(progress: bool = True, url: Optional[str] = None) -> None:
    """Ensure ``TRACE/models/`` is populated; download + extract if not.

    No-op when the folder already contains a ``.pt`` checkpoint. Otherwise
    downloads the release zip, verifies its SHA-256, and unpacks it.
    Network or hash failures raise so the caller surfaces a clean error
    instead of launching with broken models.
    """
    if _has_models():
        return

    src_url = url or _asset_url()
    _MODELS_DIR.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory() as td:
        tmp_zip = Path(td) / "trace_models.zip"
        sys.stderr.write(f"TRACE models not found at {_MODELS_DIR}. Fetching from {src_url}\n")
        _download(src_url, tmp_zip, progress=progress)
        actual = _sha256(tmp_zip)
        if actual != _EXPECTED_SHA256:
            raise RuntimeError(
                f"Downloaded models.zip SHA-256 mismatch.\n"
                f"  expected: {_EXPECTED_SHA256}\n"
                f"  got:      {actual}\n"
                f"Refusing to install. Re-run after the network is stable, "
                f"or set TRACE_ASSETS_URL to a mirror."
            )
        sys.stderr.write("SHA-256 verified. Extracting...\n")
        # Extract into a staging dir first; only move into place on success
        # so a partial extraction can't leave models/ half-populated.
        stage = Path(td) / "stage"
        stage.mkdir()
        with zipfile.ZipFile(tmp_zip) as zf:
            zf.extractall(stage)
        # The zip may contain either `models/...` or the folder contents
        # directly. Detect and move accordingly.
        extracted_models = stage / "models" if (stage / "models").is_dir() else stage
        for child in extracted_models.iterdir():
            target = _MODELS_DIR / child.name
            if target.exists():
                shutil.rmtree(target) if target.is_dir() else target.unlink()
            shutil.move(str(child), str(target))
        sys.stderr.write(f"Models installed under {_MODELS_DIR}\n")


if __name__ == "__main__":
    ensure_assets()
