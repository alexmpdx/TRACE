"""Data types for user-defined landmark distance measurements."""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class LandmarkPair:
    """One user-defined distance: straight line between landmarks `name_a` and `name_b`.

    `name_a` / `name_b` are the raw GeoJSON `properties.classification.name`
    values (e.g. "DTip", "L1-Rs", "ACV.p", "alula notch") — i.e. the keys
    used by identifyFeatures' csv_export when looking up landmarks.

    `label` is the user-supplied identifier used as the CSV column suffix
    (e.g. "wing_span"). It is sanitized by `safe_label` before use.
    """

    name_a: str
    name_b: str
    label: str


_SAFE_LABEL_RE = re.compile(r"[^A-Za-z0-9]+")


def safe_label(label: str) -> str:
    """Normalize a user label for use as a CSV column suffix.

    Collapses runs of non-alphanumerics into single underscores, strips
    leading/trailing underscores. Empty input returns "pair".
    """
    cleaned = _SAFE_LABEL_RE.sub("_", label).strip("_")
    return cleaned or "pair"


def pairs_to_dicts(pairs: list[LandmarkPair]) -> list[dict]:
    """Convert pairs to a list of plain dicts for JSON serialization."""
    return [asdict(p) for p in pairs]


def pairs_from_dicts(data: list[dict] | None) -> list[LandmarkPair]:
    """Inverse of pairs_to_dicts. Skips malformed entries."""
    if not data:
        return []
    out: list[LandmarkPair] = []
    for entry in data:
        try:
            out.append(
                LandmarkPair(
                    name_a=str(entry["name_a"]),
                    name_b=str(entry["name_b"]),
                    label=str(entry["label"]),
                )
            )
        except (KeyError, TypeError):
            continue
    return out
