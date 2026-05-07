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
from pathlib import Path
from typing import Optional

from measurement_maker.distance import compute_pair_distance_px, load_landmarks_from_geojson
from measurement_maker.types import LandmarkPair, safe_label

logger = logging.getLogger(__name__)


def _column_names(pair: LandmarkPair, has_scale: bool) -> tuple[str, Optional[str]]:
    """Return (px_col, um_col_or_None) for a pair, deduped at the caller."""
    suffix = safe_label(pair.label)
    px = f"user_distance_{suffix}_px"
    um = f"user_distance_{suffix}_um" if has_scale else None
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

    with open(csv_path, newline="") as fh:
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

    with open(csv_path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=augmented_fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    logger.info(
        "augment_csv: added %d user-distance pair(s) to %s",
        len(new_columns),
        csv_path.name,
    )
