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

    def mark_failed_preproc(self, image_basename: str) -> None:
        """Record an image as Stage-1-failed. Idempotent; updates the timestamp."""
        if image_basename not in self.failed_preproc_images:
            self.failed_preproc_images.append(image_basename)
        self.updated_at = _now_iso()

    def completed_set(self) -> set[str]:
        """Set view of the completed-image basenames for O(1) lookup."""
        return set(self.completed_images)

    def failed_preproc_set(self) -> set[str]:
        """Set view of the preproc-failed basenames for O(1) lookup."""
        return set(self.failed_preproc_images)

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


def merge_resume_csv(new_csv: Path, append_source: Path) -> int:
    """Append rows from ``append_source`` whose specimen isn't already in ``new_csv``.

    Used by the pause/resume CSV-append flow: before the new slice's
    ``export_csv_batch`` writes, the existing measurements.csv is moved
    aside to a ``.append_source`` sibling. After the new write completes,
    this helper folds the un-re-processed rows back in so the consolidated
    CSV reflects the union of all completed images across slices.

    Matching is by the "specimen" column (the image stem, not basename).
    Returns the number of rows appended, or 0 if nothing to merge.

    Errors are logged and swallowed — never raises. The new CSV stays as
    written by export_csv_batch even if the merge fails; the worst case
    is a resumed run's CSV missing rows from a prior slice, recoverable
    by re-running the affected images.
    """
    import csv

    if not append_source.is_file() or not new_csv.is_file():
        return 0
    try:
        with open(new_csv, newline="", encoding="utf-8") as f:
            new_rows = list(csv.DictReader(f))
        with open(append_source, newline="", encoding="utf-8") as f:
            old_rows = list(csv.DictReader(f))
    except Exception as exc:
        logger.warning("run_state: cannot read CSVs for merge: %s", exc)
        return 0
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
