"""Pause/resume manifest for TRACE runs.

A long batch run (200+ wings on a laptop) can take hours. Users want to
close the lid and pick up where they left off — both intentionally (Pause
button) and accidentally (crash, power loss, OS reboot during install).

This module is the persistence layer: a single ``_run_state.json`` file in
the run's output folder tracks per-image completion + the run's status.
Phase 1 ships the data structures; Phase 2 wires the GUI Pause button +
the on-launch resume prompt to use them.

Truth model: the on-disk per-image artifacts are authoritative. The
manifest is a hint — "we got this far" — used to:

  - skip already-done images on resume without re-scanning every output
    file (fast path),
  - decide whether to surface the resume prompt at all (only if a
    non-completed manifest is present),
  - diff the saved settings against the current ones and ask the user
    which to use.

The user can manually delete an output file to force re-processing of
that image; on the next resume, the manifest entry is still there but
the artifact-presence check will return False and the image gets re-run.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

_MANIFEST_FILENAME = "_run_state.json"
_MANIFEST_VERSION = 1

# Status values. "running" means the worker thread is still going; "paused"
# means the user (or a crash) interrupted between images; "completed" means
# every image in the batch finished (success or per-image error). Only
# "running" / "paused" manifests trigger the resume prompt — "completed"
# manifests are kept on disk for provenance but ignored on launch.
STATUS_RUNNING = "running"
STATUS_PAUSED = "paused"
STATUS_COMPLETED = "completed"
STATUS_CANCELLED = "cancelled"


@dataclass
class RunManifest:
    """Per-output-folder snapshot of run progress.

    Image identifiers are stored as basenames (no directory parts) so the
    manifest stays portable if the user moves the input folder around.
    Recursive runs aren't supported in Phase 1 — they'd need full relative
    paths to disambiguate same-name files in nested folders.
    """

    version: int = _MANIFEST_VERSION
    started_at: str = ""
    updated_at: str = ""
    status: str = STATUS_RUNNING
    input_dir: str = ""
    recursive: bool = False
    outputs_selected: list[str] = field(default_factory=list)
    csv_measurement_groups: list[str] = field(default_factory=list)
    # Relative path (inside output_dir) to the settings YAML written at
    # run start. Read on resume so the user can compare against current
    # GUI state and choose original-vs-new.
    settings_snapshot_path: str = ""
    total_images: int = 0
    completed_images: list[str] = field(default_factory=list)
    # Images whose Stage 1 (preprocessing — landmarks / hinge / segmentation)
    # errored, including confidence-gate failures. Tracked separately from
    # completed_images because on resume we want to skip them — but only if
    # settings are unchanged. A gate-threshold change could unblock them,
    # so a resume that picks "Continue with current settings" clears these
    # from the skip set to give them another shot.
    failed_preproc_images: list[str] = field(default_factory=list)
    # Images whose Stage 2 (identifyFeatures analysis) errored. Persisted so
    # the post-run "Review failed images" / "Reload previous session" flows
    # can resurface them without re-running the pipeline. Disjoint from
    # failed_preproc_images by construction — an image either fails Stage 1
    # or fails Stage 2, never both in the same run.
    analysis_failed_images: list[str] = field(default_factory=list)
    # Per-image error text (basename → message). Populated alongside the
    # failed_* lists; used by the restore-previous-session flow to repaint
    # the image-list tooltips that show *why* each image failed. Empty for
    # any image whose error message wasn't captured (e.g. older manifests).
    failure_messages: dict[str, str] = field(default_factory=dict)

    def mark_completed(self, image_basename: str) -> None:
        """Record an image as Stage-2-done. Idempotent; updates the timestamp.

        Also removes the basename from failed_preproc_images if present —
        the image was previously failing preprocessing and has now
        succeeded, so it no longer belongs on the failure list.
        """
        if image_basename not in self.completed_images:
            self.completed_images.append(image_basename)
        if image_basename in self.failed_preproc_images:
            self.failed_preproc_images.remove(image_basename)
        self.updated_at = _now_iso()

    def mark_failed_preproc(self, image_basename: str, error_text: str = "") -> None:
        """Record an image as Stage-1-failed. Idempotent; updates the timestamp.

        ``error_text`` is stored in ``failure_messages`` so the post-run
        restore flow can replay the per-image tooltip. Pass an empty
        string when no message is available; older callers stay
        backwards-compatible because the default is empty.
        """
        if image_basename not in self.failed_preproc_images:
            self.failed_preproc_images.append(image_basename)
        if error_text:
            self.failure_messages[image_basename] = error_text
        self.updated_at = _now_iso()

    def mark_failed_analysis(self, image_basename: str, error_text: str = "") -> None:
        """Record an image as Stage-2-failed. Idempotent; updates the timestamp.

        Mirrors mark_failed_preproc but for identifyFeatures errors. The
        same image never lands in both lists in the same run — Stage 1
        success is a precondition for Stage 2.
        """
        if image_basename not in self.analysis_failed_images:
            self.analysis_failed_images.append(image_basename)
        if error_text:
            self.failure_messages[image_basename] = error_text
        self.updated_at = _now_iso()

    def completed_set(self) -> set[str]:
        """Set view of the completed-image basenames for O(1) lookup."""
        return set(self.completed_images)

    def failed_preproc_set(self) -> set[str]:
        """Set view of the preproc-failed basenames for O(1) lookup."""
        return set(self.failed_preproc_images)

    def analysis_failed_set(self) -> set[str]:
        """Set view of the analysis-failed basenames for O(1) lookup."""
        return set(self.analysis_failed_images)

    def has_unreviewed_failures(self) -> bool:
        """True when there's at least one image in either failure list.

        Used by find_completed_manifest() to skip completed runs that
        have nothing left to look at.
        """
        return bool(self.failed_preproc_images or self.analysis_failed_images)

    def is_in_progress(self) -> bool:
        """True when this run can be resumed (running or paused, not done)."""
        return self.status in (STATUS_RUNNING, STATUS_PAUSED)


def _now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


def manifest_path(output_dir: Path) -> Path:
    """Resolve the canonical manifest path for an output folder."""
    return Path(output_dir) / _MANIFEST_FILENAME


def load_manifest(output_dir: Path) -> Optional[RunManifest]:
    """Read the manifest if one exists in ``output_dir``.

    Returns None when:
      - the file is absent (typical fresh run);
      - the JSON is unreadable or its version doesn't match (forwards-
        incompatible — newer manifests on an older TRACE won't be touched);
      - the file is empty / malformed.

    Errors are logged but not raised — callers proceed as if no manifest
    existed, which gives "fresh run" semantics rather than a hard fail.
    """
    path = manifest_path(output_dir)
    if not path.is_file():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning("run_state: cannot read %s: %s", path, exc)
        return None
    if not isinstance(data, dict):
        logger.warning("run_state: %s is not a JSON object", path)
        return None
    if int(data.get("version", 0)) != _MANIFEST_VERSION:
        logger.warning(
            "run_state: %s has version %s, expected %s — ignoring",
            path,
            data.get("version"),
            _MANIFEST_VERSION,
        )
        return None
    try:
        raw_failure_messages = data.get("failure_messages", {}) or {}
        if not isinstance(raw_failure_messages, dict):
            raw_failure_messages = {}
        return RunManifest(
            version=int(data.get("version", _MANIFEST_VERSION)),
            started_at=str(data.get("started_at", "")),
            updated_at=str(data.get("updated_at", "")),
            status=str(data.get("status", STATUS_RUNNING)),
            input_dir=str(data.get("input_dir", "")),
            recursive=bool(data.get("recursive", False)),
            outputs_selected=list(data.get("outputs_selected", []) or []),
            csv_measurement_groups=list(data.get("csv_measurement_groups", []) or []),
            settings_snapshot_path=str(data.get("settings_snapshot_path", "")),
            total_images=int(data.get("total_images", 0)),
            completed_images=list(data.get("completed_images", []) or []),
            failed_preproc_images=list(data.get("failed_preproc_images", []) or []),
            # Older manifests (pre-restore-feature) don't have these keys;
            # defaulting to empty preserves backwards compatibility.
            analysis_failed_images=list(data.get("analysis_failed_images", []) or []),
            failure_messages={str(k): str(v) for k, v in raw_failure_messages.items()},
        )
    except (TypeError, ValueError) as exc:
        logger.warning("run_state: %s has bad fields: %s", path, exc)
        return None


def save_manifest(output_dir: Path, manifest: RunManifest) -> None:
    """Write the manifest atomically (write-then-rename).

    Atomic write so a crash mid-write can't leave a half-written JSON file
    that load_manifest would then reject — better to have either the new
    state or the previous state, never a torn one.
    """
    if not manifest.started_at:
        manifest.started_at = _now_iso()
    manifest.updated_at = _now_iso()
    target = manifest_path(output_dir)
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_suffix(target.suffix + ".tmp")
    try:
        tmp.write_text(json.dumps(asdict(manifest), indent=2), encoding="utf-8")
        tmp.replace(target)
    except OSError as exc:
        logger.warning("run_state: cannot write %s: %s", target, exc)


def find_completed_manifest(output_dir: Path) -> Optional[tuple["RunManifest", Path]]:
    """Look for the most-recent COMPLETED manifest that still has failures.

    Sibling of :func:`find_resumable_manifest`. The post-run restore flow
    uses this to surface "the last run finished but you didn't get to
    review N failed images — open them now?". Skips:

      - non-completed runs (those go through find_resumable_manifest),
      - completed runs whose failure lists are both empty (nothing to
        review — no point offering to restore).

    Returns ``(manifest, run_folder)`` for the most-recent qualifying
    run, or ``None`` if there's nothing to offer. "Most recent" follows
    ``manifest.started_at`` so renaming / moving folders doesn't shuffle
    the priority. Same legacy ``output_dir/_run_state.json`` fallback as
    find_resumable_manifest, for users coming from a pre-``run_<N>/``
    layout.
    """
    candidates: list[tuple[RunManifest, Path]] = []
    for run_dir in Path(output_dir).glob("run_*"):
        if not run_dir.is_dir():
            continue
        m = load_manifest(run_dir)
        if m is not None and m.status == STATUS_COMPLETED and m.has_unreviewed_failures():
            candidates.append((m, run_dir))
    legacy_path = manifest_path(output_dir)
    if legacy_path.is_file():
        m = load_manifest(output_dir)
        if m is not None and m.status == STATUS_COMPLETED and m.has_unreviewed_failures():
            candidates.append((m, Path(output_dir)))
    if not candidates:
        return None
    candidates.sort(key=lambda mr: mr[0].started_at, reverse=True)
    return candidates[0]


def find_resumable_manifest(output_dir: Path) -> Optional[tuple["RunManifest", Path]]:
    """Look in ``output_dir/run_*/`` for an in-progress manifest.

    Returns ``(manifest, run_folder)`` for the most-recent resumable run,
    or ``None`` if there's nothing to resume. Falls back to the legacy
    top-level ``output_dir/_run_state.json`` location so users on v0.1.28
    or earlier (where the manifest lived at the top of the output folder)
    can still pick up an unfinished run after upgrading.

    "Most recent" is determined by manifest.started_at, not by folder
    mtime, so the timestamps remain meaningful even after a user has
    copied the output folder around.
    """
    candidates: list[tuple[RunManifest, Path]] = []
    for run_dir in Path(output_dir).glob("run_*"):
        if not run_dir.is_dir():
            continue
        m = load_manifest(run_dir)
        if m is not None and m.is_in_progress():
            candidates.append((m, run_dir))
    # Legacy fallback: older versions wrote the manifest at the top level.
    legacy_path = manifest_path(output_dir)
    if legacy_path.is_file():
        m = load_manifest(output_dir)
        if m is not None and m.is_in_progress():
            candidates.append((m, Path(output_dir)))
    if not candidates:
        return None
    candidates.sort(key=lambda mr: mr[0].started_at, reverse=True)
    return candidates[0]


# Encodings tried in order when reading CSVs for merge. UTF-8 with BOM
# handles anything TRACE ≥ v0.2.25 wrote (utf-8-sig with the BOM stripped
# by the reader), plain UTF-8 handles ≥ v0.2.22 fast-path CSVs written on
# macOS/Linux, cp1252 handles anything TRACE ≤ v0.2.24 wrote on Windows
# (default locale before the explicit-encoding fix), and latin-1 is a
# never-fails safety net that at worst mis-renders 0x80-0xFF bytes but
# preserves every row so the merge succeeds.
_CSV_READ_ENCODINGS: tuple[str, ...] = ("utf-8-sig", "utf-8", "cp1252", "latin-1")


def _read_csv_with_fallback(path: Path) -> tuple[list[dict[str, str]], str]:
    """Read a CSV, falling back through _CSV_READ_ENCODINGS on UnicodeDecodeError.

    Returns (rows, encoding_used). Raises the last encoding's error only if
    every attempt failed, which can't happen for latin-1 (it decodes any
    byte sequence). Caller can treat this as "guaranteed to return rows
    given a readable file".
    """
    import csv

    last_exc: Exception | None = None
    for enc in _CSV_READ_ENCODINGS:
        try:
            with open(path, newline="", encoding=enc) as fh:
                return list(csv.DictReader(fh)), enc
        except UnicodeDecodeError as exc:
            last_exc = exc
            continue
    # Should be unreachable — latin-1 never raises. Defensive re-raise
    # anyway so callers see a real error rather than a silent success
    # on an empty list.
    raise last_exc  # type: ignore[misc]


def count_csv_rows(path: Path) -> int:
    """Return the number of data rows (excluding header) in a CSV.

    Uses the same encoding-fallback ladder as merge_resume_csv, so a file
    that TRACE ≤ v0.2.24 wrote in cp1252 (or a corrupt file) still gets a
    real count instead of 0. Used by the pipeline's append-source unlink
    guard to distinguish "source was empty, safe to delete" from "source
    had rows but the merge dropped them, PRESERVE the file".

    Returns 0 for a nonexistent file. Never raises — silent failures fall
    back to 0 so a broken count doesn't itself cause data loss upstream.
    """
    if not path.is_file():
        return 0
    try:
        rows, _ = _read_csv_with_fallback(path)
    except Exception as exc:
        logger.warning("run_state: cannot count rows in %s: %s", path, exc)
        return 0
    return len(rows)


def merge_resume_csv(new_csv: Path, append_source: Path) -> int:
    """Append rows from ``append_source`` whose specimen isn't already in ``new_csv``.

    Used by the pause/resume CSV-append flow: before the new slice's
    ``export_csv_batch`` writes, the existing measurements.csv is moved
    aside to a ``.append_source`` sibling. After the new write completes,
    this helper folds the un-re-processed rows back in so the consolidated
    CSV reflects the union of all completed images across slices.

    Matching is by the "specimen" column (the image stem, not basename).
    Returns the number of rows appended, or 0 if nothing to merge.

    Read encoding is fallback-tolerant (utf-8-sig → utf-8 → cp1252 →
    latin-1). Before the ladder was in place (TRACE ≤ v0.2.24), a
    Windows-authored CSV containing bytes like 0xba (º in cp1252, from
    specimen names like "29ºC") would silently fail to decode as UTF-8
    here, the merge would return 0, and the caller's unconditional
    unlink() would delete the source — losing every un-re-processed row.

    Errors are logged and swallowed — never raises. The new CSV stays as
    written by export_csv_batch even if the merge fails; the worst case
    is a resumed run's CSV missing rows from a prior slice, recoverable
    by re-running the affected images (or via tools/recover_landmark_csv.py
    for landmarks-only fast-path CSVs).
    """
    import csv

    if not append_source.is_file() or not new_csv.is_file():
        return 0
    try:
        new_rows, new_enc = _read_csv_with_fallback(new_csv)
        old_rows, old_enc = _read_csv_with_fallback(append_source)
    except Exception as exc:
        logger.warning("run_state: cannot read CSVs for merge: %s", exc)
        return 0
    if new_enc != "utf-8-sig" or old_enc != "utf-8-sig":
        logger.debug(
            "run_state: merge read encodings — new=%s, append_source=%s",
            new_enc,
            old_enc,
        )
    if not old_rows:
        return 0
    new_specimens = {row.get("specimen", "") for row in new_rows}
    to_append = [row for row in old_rows if row.get("specimen", "") not in new_specimens]
    if not to_append:
        return 0
    # Column order follows the NEW csv (it reflects the user's current
    # output-group selections; the old one might have stale columns).
    fieldnames = list(new_rows[0].keys()) if new_rows else list(old_rows[0].keys())
    try:
        # Plain utf-8 (not utf-8-sig) for append — the BOM should sit at
        # the file start only, written by the initial CSV writer. Adding
        # utf-8-sig here would embed a BOM in the middle of the file at
        # every merge, breaking Excel and any UTF-8-aware reader.
        with open(new_csv, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            for row in to_append:
                writer.writerow({k: row.get(k, "") for k in fieldnames})
    except OSError as exc:
        logger.warning("run_state: cannot append merged rows to %s: %s", new_csv, exc)
        return 0
    return len(to_append)


def new_manifest(
    *,
    input_dir: Path,
    recursive: bool,
    outputs_selected: set[str] | list[str],
    csv_measurement_groups: set[str] | list[str],
    total_images: int,
    settings_snapshot_path: str = "",
) -> RunManifest:
    """Build a fresh manifest at the start of a run."""
    return RunManifest(
        started_at=_now_iso(),
        updated_at=_now_iso(),
        status=STATUS_RUNNING,
        input_dir=str(input_dir),
        recursive=bool(recursive),
        outputs_selected=sorted(outputs_selected),
        csv_measurement_groups=sorted(csv_measurement_groups),
        settings_snapshot_path=settings_snapshot_path,
        total_images=int(total_images),
        completed_images=[],
    )
