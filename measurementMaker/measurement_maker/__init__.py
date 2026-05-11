"""User-defined landmark distance measurements.

Lets a user pick two landmarks on a sample wing (via an embedded napari
widget in the TRACE settings dialog) and have the straight-line distance
between them computed for every wing in a TRACE batch run, with the result
added as a column in the consolidated CSV.

Public API:
- LandmarkPair: dataclass describing one user-defined pair
- pairs_to_dicts / pairs_from_dicts: JSON-serialization helpers
- load_landmarks_from_geojson: read points from a *_landmarks.geojson
- compute_pair_distance_px: straight-line distance between two named landmarks
- augment_csv_with_user_distances: post-process a TRACE CSV to add user columns
- LandmarkPickerWidget: embeddable QWidget hosting the napari canvas + pair list
"""

from measurement_maker.csv_augment import augment_csv_with_user_distances, write_distances_csv
from measurement_maker.distance import (
    compute_pair_distance_px,
    load_landmarks_from_geojson,
)
from measurement_maker.types import LandmarkPair, pairs_from_dicts, pairs_to_dicts

__all__ = [
    "LandmarkPair",
    "pairs_to_dicts",
    "pairs_from_dicts",
    "load_landmarks_from_geojson",
    "compute_pair_distance_px",
    "augment_csv_with_user_distances",
    "write_distances_csv",
    "LandmarkPickerWidget",
]


def __getattr__(name):
    """Lazy-load the napari-dependent widget so headless callers don't pay napari's import cost."""
    if name == "LandmarkPickerWidget":
        from measurement_maker.embedded_picker import LandmarkPickerWidget

        return LandmarkPickerWidget
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
