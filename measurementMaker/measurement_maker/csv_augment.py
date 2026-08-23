"""Append user-defined distance columns to a TRACE measurements CSV.

The CSV is written by identifyFeatures' export_csv_batch with one row per
wing and a `specimen_id` column. We read it back, look up the landmark
GeoJSON for each specimen, compute the configured pair distances, and write
the file out with the new columns appended.

Done as a post-processing step (rather than threaded through identifyFeatures)
so that user-defined distances are TRACE-only and don't add config-shape
coupling to the analysis library.
"""

from __future__ import annotations

import csv
import logging
import math
from pathlib import Path
from typing import Optional

from measurement_maker.distance import compute_pair_distance_px, load_landmarks_from_geojson
from measurement_maker.types import LandmarkPair, safe_label

logger = logging.getLogger(__name__)

# Measurement groups the landmarks-only fast path knows how to emit without
# invoking identifyFeatures. wing_area / wing_shape need wing_outline (which
# requires the segmentation Stage-1 sub-stage), so they don't qualify.
LANDMARK_ONLY_MEASUREMENT_GROUPS = frozenset({"cv_ratio"})


def _column_names(pair: LandmarkPair, has_scale: bool) -> tuple[str, Optional[str]]:
    """Return (px_col, um_col_or_None) for a pair, deduped at the caller."""
    suffix = safe_label(pair.label)
    px = f"custom_{suffix}_px"
    um = f"custom_{suffix}_um" if has_scale else None
    return px, um


def _dedupe_suffix(base: str, taken: set[str]) -> str:
    """Append _2, _3, ... until `base` is not in `taken`."""
    if base not in taken:
        return base
    i = 2
    while f"{base}_{i}" in taken:
        i += 1
    return f"{base}_{i}"


def augment_csv_with_user_distances(
    csv_path: Path,
    specimen_landmarks: dict[str, Path],
    pairs: list[LandmarkPair],
    um_per_px: Optional[float],
) -> None:
    """Append user-distance columns to `csv_path` in place.

    Args:
        csv_path: TRACE measurements.csv (must exist; one row per specimen).
        specimen_landmarks: {specimen_id (matches CSV column): path to *_landmarks.geojson}.
        pairs: User-configured pairs; empty list is a no-op.
        um_per_px: Scale; when None only `_px` columns are written.

    Missing landmarks on a particular wing produce a blank cell for that
    pair on that wing (logged at WARNING).
    """
    if not pairs:
        return
    if not csv_path.exists():
        logger.warning("augment_csv: %s does not exist; skipping user-distance columns", csv_path)
        return

    # utf-8-sig on read tolerates both TRACE ≥ v0.2.25 writes (BOM-prefixed
    # UTF-8) and any legacy fallback path — the "-sig" variant transparently
    # strips a leading BOM if present, so plain-UTF-8 files also work.
    with open(csv_path, newline="", encoding="utf-8-sig") as fh:
        reader = csv.DictReader(fh)
        existing_fieldnames = list(reader.fieldnames or [])
        rows = list(reader)

    if not rows:
        logger.info("augment_csv: %s has no rows; nothing to augment", csv_path)
        return

    has_scale = um_per_px is not None and um_per_px > 0
    taken = set(existing_fieldnames)
    new_columns: list[tuple[LandmarkPair, str, Optional[str]]] = []
    for pair in pairs:
        px_col, um_col = _column_names(pair, has_scale)
        px_col = _dedupe_suffix(px_col, taken)
        taken.add(px_col)
        if um_col is not None:
            um_col = _dedupe_suffix(um_col, taken)
            taken.add(um_col)
        new_columns.append((pair, px_col, um_col))

    landmark_cache: dict[str, dict[str, tuple[float, float]]] = {}

    def _landmarks_for(specimen_id: str) -> dict[str, tuple[float, float]]:
        if specimen_id in landmark_cache:
            return landmark_cache[specimen_id]
        path = specimen_landmarks.get(specimen_id)
        loaded: dict[str, tuple[float, float]] = {}
        if path is not None and path.exists():
            loaded = load_landmarks_from_geojson(path)
        landmark_cache[specimen_id] = loaded
        return loaded

    specimen_key = "specimen_id" if "specimen_id" in existing_fieldnames else existing_fieldnames[0]

    for row in rows:
        specimen_id = row.get(specimen_key, "")
        landmarks = _landmarks_for(specimen_id)
        for pair, px_col, um_col in new_columns:
            dist_px = compute_pair_distance_px(landmarks, pair.name_a, pair.name_b)
            if dist_px is None:
                row[px_col] = ""
                if um_col is not None:
                    row[um_col] = ""
                logger.warning(
                    "augment_csv: %s missing landmark(s) %r/%r for pair %r",
                    specimen_id,
                    pair.name_a,
                    pair.name_b,
                    pair.label,
                )
                continue
            row[px_col] = f"{dist_px:.1f}"
            if um_col is not None:
                row[um_col] = f"{dist_px * um_per_px:.1f}"

    augmented_fieldnames = list(existing_fieldnames)
    for _pair, px_col, um_col in new_columns:
        augmented_fieldnames.append(px_col)
        if um_col is not None:
            augmented_fieldnames.append(um_col)

    # utf-8-sig on write so Excel on Windows auto-detects UTF-8 when the
    # user double-clicks the CSV. Without the BOM, Excel treats the file
    # as cp1252 and any non-ASCII in specimen names (e.g. "29ºC") gets
    # mangled on display. See v0.2.25 CSV-encoding fix.
    with open(csv_path, "w", newline="", encoding="utf-8-sig") as fh:
        writer = csv.DictWriter(fh, fieldnames=augmented_fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    logger.info(
        "augment_csv: added %d user-distance pair(s) to %s",
        len(new_columns),
        csv_path.name,
    )


def write_distances_csv(
    csv_path: Path,
    specimen_landmarks: dict[str, Path],
    pairs: list[LandmarkPair],
    um_per_px: Optional[float],
) -> None:
    """Write a fresh CSV with just user-distance columns — no identifyFeatures CSV needed.

    Used by TRACE's fast path: when the only requested Stage 2 output is the
    batch CSV and the user has configured custom distance pairs, we skip
    identifyFeatures entirely and produce the CSV directly from each wing's
    landmark GeoJSON.

    Args:
        csv_path: Output path; parent dirs are created if missing.
        specimen_landmarks: {specimen_id: path to *_landmarks.geojson}.
            One CSV row is written per entry, in insertion order.
        pairs: User-configured distance pairs. Empty list is a no-op.
        um_per_px: Scale; when None only `_px` columns are written.

    Missing landmarks on a particular wing produce a blank cell for that
    pair on that wing (logged at WARNING).
    """
    if not pairs:
        logger.info("write_distances_csv: no pairs configured; skipping write")
        return
    if not specimen_landmarks:
        logger.info("write_distances_csv: no specimens; skipping write")
        return

    has_scale = um_per_px is not None and um_per_px > 0
    fieldnames: list[str] = ["specimen_id"]
    taken: set[str] = set(fieldnames)
    pair_cols: list[tuple[LandmarkPair, str, Optional[str]]] = []
    for pair in pairs:
        px_col, um_col = _column_names(pair, has_scale)
        px_col = _dedupe_suffix(px_col, taken)
        taken.add(px_col)
        fieldnames.append(px_col)
        if um_col is not None:
            um_col = _dedupe_suffix(um_col, taken)
            taken.add(um_col)
            fieldnames.append(um_col)
        pair_cols.append((pair, px_col, um_col))

    rows: list[dict[str, str]] = []
    for specimen_id, lm_path in specimen_landmarks.items():
        landmarks: dict[str, tuple[float, float]] = {}
        if lm_path is not None and lm_path.exists():
            landmarks = load_landmarks_from_geojson(lm_path)
        row: dict[str, str] = {"specimen_id": specimen_id}
        for pair, px_col, um_col in pair_cols:
            dist_px = compute_pair_distance_px(landmarks, pair.name_a, pair.name_b)
            if dist_px is None:
                row[px_col] = ""
                if um_col is not None:
                    row[um_col] = ""
                logger.warning(
                    "write_distances_csv: %s missing landmark(s) %r/%r for pair %r",
                    specimen_id,
                    pair.name_a,
                    pair.name_b,
                    pair.label,
                )
                continue
            row[px_col] = f"{dist_px:.1f}"
            if um_col is not None:
                row[um_col] = f"{dist_px * um_per_px:.1f}"
        rows.append(row)

    csv_path.parent.mkdir(parents=True, exist_ok=True)
    # utf-8-sig — see augment_csv_with_user_distances for rationale.
    with open(csv_path, "w", newline="", encoding="utf-8-sig") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    logger.info(
        "write_distances_csv: wrote %s with %d pair(s) × %d wing(s)",
        csv_path.name,
        len(pair_cols),
        len(rows),
    )


def _cv_ratio_row(landmarks: dict[str, tuple[float, float]], um_per_px: Optional[float]) -> dict[str, str]:
    """Compute cv_ratio group values from a landmarks dict.

    Mirrors identify_features.views.csv_export._wing_measurements' cv_ratio
    block (wing length L1-Rs→DTip, crossvein distance ACV.p→PCV.a, ratio).
    Column names match identify_features' wide-format output.
    """
    has_scale = um_per_px is not None and um_per_px > 0
    vals: dict[str, str] = {
        "wing length_px": "",
        "wing length_um": "",
        "crossvein distance_px": "",
        "crossvein distance_um": "",
        "CV ratio": "",
    }
    l1rs = landmarks.get("L1-Rs")
    dtip = landmarks.get("DTip")
    wl_px: Optional[float] = None
    if l1rs is not None and dtip is not None:
        wl_px = math.hypot(dtip[0] - l1rs[0], dtip[1] - l1rs[1])
        vals["wing length_px"] = f"{wl_px:.1f}"
        if has_scale:
            vals["wing length_um"] = f"{wl_px * um_per_px:.1f}"
    acvp = landmarks.get("ACV.p")
    pcva = landmarks.get("PCV.a")
    cv_px: Optional[float] = None
    if acvp is not None and pcva is not None:
        cv_px = math.hypot(pcva[0] - acvp[0], pcva[1] - acvp[1])
        vals["crossvein distance_px"] = f"{cv_px:.1f}"
        if has_scale:
            vals["crossvein distance_um"] = f"{cv_px * um_per_px:.1f}"
    if wl_px is not None and cv_px is not None and wl_px > 0:
        vals["CV ratio"] = f"{cv_px / wl_px:.4f}"
    return vals


def write_landmark_csv_batch(
    csv_path: Path,
    specimen_landmarks: dict[str, Path],
    pairs: list[LandmarkPair],
    measurement_groups: Optional[set[str]] = None,
    um_per_px: Optional[float] = None,
    um_per_px_by_specimen: Optional[dict[str, float]] = None,
) -> None:
    """Write a landmarks-only CSV: user distances + cv_ratio measurement group.

    Extends write_distances_csv's landmarks-only fast path to also emit CSV
    columns for the cv_ratio measurement group when it's the only selected
    group (wing_area / wing_shape need the wing outline and don't qualify).

    The output columns and specimen key match identify_features' wide-format
    export (``export_csv_batch``) so a fast-path CSV is drop-in comparable to
    a full-pipeline CSV filtered to the same groups.

    Args:
        csv_path: Output path; parent dirs are created if missing.
        specimen_landmarks: {specimen_id: path to *_landmarks.geojson}. One
            CSV row is written per entry, in insertion order.
        pairs: User-configured distance pairs. May be empty when only
            cv_ratio was requested.
        measurement_groups: Subset of LANDMARK_ONLY_MEASUREMENT_GROUPS to
            include as columns. None or empty = no landmark-only groups
            (behaves like write_distances_csv).
        um_per_px: Default µm/px scale used for every specimen that
            doesn't have a per-specimen override or a persisted
            effective_um_per_px in its landmark geojson. When None AND
            no per-specimen scale resolves for a given wing, only the
            `_px` columns are populated for that wing.
        um_per_px_by_specimen: Optional explicit per-specimen scale
            overrides (e.g. re-detected from image metadata via
            tools/recover_landmark_csv.py --images). Takes precedence
            over the geojson's persisted effective_um_per_px, which in
            turn takes precedence over ``um_per_px``.

    Scale precedence (per specimen):
        um_per_px_by_specimen[stem] > geojson fc props effective_um_per_px > um_per_px

    Missing landmarks on a particular wing produce blank cells for the
    affected columns on that wing (logged at WARNING). The _um columns
    are ALWAYS written when any specimen has a resolvable scale — wings
    without a scale get blank _um cells rather than dropping the column
    (a mixed batch shouldn't hide µm data for the wings that have it).
    """
    from measurement_maker.distance import load_landmark_geojson_fc_props

    groups = set(measurement_groups) if measurement_groups else set()
    include_cv_ratio = "cv_ratio" in groups
    if not pairs and not include_cv_ratio:
        logger.info("write_landmark_csv_batch: nothing to emit; skipping write")
        return
    if not specimen_landmarks:
        logger.info("write_landmark_csv_batch: no specimens; skipping write")
        return

    um_by_specimen = dict(um_per_px_by_specimen or {})
    default_scale = float(um_per_px) if um_per_px is not None and um_per_px > 0 else None

    # First pass: resolve per-specimen scale from (a) explicit override,
    # (b) geojson-persisted effective_um_per_px, (c) default_scale.
    per_specimen_scale: dict[str, Optional[float]] = {}
    scale_source_counts = {"override": 0, "geojson": 0, "default": 0, "none": 0}
    for specimen_id, lm_path in specimen_landmarks.items():
        s: Optional[float] = None
        if specimen_id in um_by_specimen and um_by_specimen[specimen_id] and um_by_specimen[specimen_id] > 0:
            s = float(um_by_specimen[specimen_id])
            scale_source_counts["override"] += 1
        else:
            fc_props = {}
            if lm_path is not None and lm_path.exists():
                fc_props = load_landmark_geojson_fc_props(lm_path)
            fc_scale = fc_props.get("effective_um_per_px")
            if fc_scale is not None and float(fc_scale) > 0:
                s = float(fc_scale)
                scale_source_counts["geojson"] += 1
            elif default_scale is not None:
                s = default_scale
                scale_source_counts["default"] += 1
            else:
                scale_source_counts["none"] += 1
        per_specimen_scale[specimen_id] = s

    has_any_scale = any(s is not None for s in per_specimen_scale.values())

    fieldnames: list[str] = ["specimen"]
    if include_cv_ratio:
        fieldnames.append("wing length_px")
        if has_any_scale:
            fieldnames.append("wing length_um")
        fieldnames.append("crossvein distance_px")
        if has_any_scale:
            fieldnames.append("crossvein distance_um")
        fieldnames.append("CV ratio")

    taken: set[str] = set(fieldnames)
    pair_cols: list[tuple[LandmarkPair, str, Optional[str]]] = []
    for pair in pairs:
        px_col, um_col = _column_names(pair, has_any_scale)
        px_col = _dedupe_suffix(px_col, taken)
        taken.add(px_col)
        fieldnames.append(px_col)
        if um_col is not None:
            um_col = _dedupe_suffix(um_col, taken)
            taken.add(um_col)
            fieldnames.append(um_col)
        pair_cols.append((pair, px_col, um_col))

    rows: list[dict[str, str]] = []
    for specimen_id, lm_path in specimen_landmarks.items():
        landmarks: dict[str, tuple[float, float]] = {}
        if lm_path is not None and lm_path.exists():
            landmarks = load_landmarks_from_geojson(lm_path)
        scale_for_this = per_specimen_scale[specimen_id]
        row: dict[str, str] = {"specimen": specimen_id}
        if include_cv_ratio:
            cv_vals = _cv_ratio_row(landmarks, scale_for_this)
            for col in ("wing length_px", "wing length_um", "crossvein distance_px", "crossvein distance_um", "CV ratio"):
                if col in fieldnames:
                    row[col] = cv_vals[col]
            if not cv_vals["CV ratio"]:
                logger.warning(
                    "write_landmark_csv_batch: %s missing landmark(s) needed for cv_ratio", specimen_id
                )
        for pair, px_col, um_col in pair_cols:
            dist_px = compute_pair_distance_px(landmarks, pair.name_a, pair.name_b)
            if dist_px is None:
                row[px_col] = ""
                if um_col is not None:
                    row[um_col] = ""
                logger.warning(
                    "write_landmark_csv_batch: %s missing landmark(s) %r/%r for pair %r",
                    specimen_id,
                    pair.name_a,
                    pair.name_b,
                    pair.label,
                )
                continue
            row[px_col] = f"{dist_px:.1f}"
            if um_col is not None:
                row[um_col] = f"{dist_px * scale_for_this:.1f}" if scale_for_this else ""
        rows.append(row)

    csv_path.parent.mkdir(parents=True, exist_ok=True)
    # utf-8-sig — see augment_csv_with_user_distances for rationale.
    with open(csv_path, "w", newline="", encoding="utf-8-sig") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    logger.info(
        "write_landmark_csv_batch: wrote %s (%d wing(s); groups=%s; %d pair(s); "
        "scale sources: override=%d, geojson=%d, default=%d, none=%d)",
        csv_path.name,
        len(rows),
        sorted(groups & LANDMARK_ONLY_MEASUREMENT_GROUPS),
        len(pair_cols),
        scale_source_counts["override"],
        scale_source_counts["geojson"],
        scale_source_counts["default"],
        scale_source_counts["none"],
    )
