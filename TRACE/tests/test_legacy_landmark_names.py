"""Every landmark reader renames the legacy "DTip" landmark to "L3.d" on read.

Landmark GeoJSONs, manual overrides and saved measurement pairs written before
the rename still say "DTip"; none of them are rewritten on disk. Each module
that reads them applies the alias itself, and the standalone modules
(HingeChopper, wingRotator, identifyFeatures, measurementMaker) each keep a
local copy of the alias table, so these tests also guard against those copies
drifting from LandmarkLocator's.
"""

import json
import sys
from pathlib import Path

# Sibling modules export bare package names that don't match their directory.
_ROOT = Path(__file__).resolve().parents[2]
for _p in (
    _ROOT,
    _ROOT / "HingeChopper",
    _ROOT / "modelTOjson",
    _ROOT / "identifyFeatures",
    _ROOT / "wingRotator",
    _ROOT / "measurementMaker",
    _ROOT / "scaleEstimator",
    _ROOT / "LandmarkLocator",
):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import hinge_chopper  # noqa: E402
import wing_rotator  # noqa: E402
from identify_features.models import geojson_io  # noqa: E402
from landmark_locator.names import LEGACY_LANDMARK_NAMES  # noqa: E402
from measurement_maker import landmark_names  # noqa: E402
from measurement_maker.distance import load_landmarks_from_geojson, resolve_landmark_key  # noqa: E402

from preprocessing.pipeline import load_landmarks_override  # noqa: E402
from TRACE.landmark_inspector_dialog import _parse_landmarks_geojson  # noqa: E402


def _write_legacy_landmarks(path: Path) -> Path:
    """A landmarks GeoJSON as written before the rename."""
    points = {"DTip": (5153.0, 1968.0), "alula notch": (300.0, 1500.0), "L1-Rs": (753.0, 1285.0)}
    features = [
        {
            "type": "Feature",
            "geometry": {"type": "Point", "coordinates": [x, y]},
            "properties": {"classification": {"name": name}, "reliable": True, "confidence": 1.0},
        }
        for name, (x, y) in points.items()
    ]
    path.write_text(json.dumps({"type": "FeatureCollection", "features": features}), encoding="utf-8")
    return path


def test_local_alias_tables_match_landmark_locator():
    assert hinge_chopper._LEGACY_LANDMARK_NAMES == LEGACY_LANDMARK_NAMES
    assert wing_rotator._LEGACY_LANDMARK_NAMES == LEGACY_LANDMARK_NAMES
    assert geojson_io._LEGACY_LANDMARK_NAMES == LEGACY_LANDMARK_NAMES
    assert landmark_names.LEGACY_LANDMARK_NAMES == LEGACY_LANDMARK_NAMES


def test_identify_features_reader(tmp_path):
    landmarks = geojson_io.load_landmarks_geojson(_write_legacy_landmarks(tmp_path / "w_landmarks.geojson"))
    assert "L3.d" in landmarks and "DTip" not in landmarks
    assert landmarks["L3.d"].name == "L3.d"


def test_hinge_chopper_reader(tmp_path):
    landmarks = hinge_chopper.load_landmarks(_write_legacy_landmarks(tmp_path / "w_landmarks.geojson"))
    assert landmarks["L3.d"] == (5153.0, 1968.0) and "DTip" not in landmarks
    # L3.d is still the preferred distal reference for the hinge side test.
    assert hinge_chopper._pick_distal_reference(landmarks) == (5153.0, 1968.0)


def test_wing_rotator_reader_uses_the_legacy_point_in_the_fit(tmp_path):
    _, canonical, _, names = wing_rotator._load_landmarks_for_fit(
        _write_legacy_landmarks(tmp_path / "w_landmarks.geojson"), soft_reliability=False
    )
    assert "L3.d" in names and "DTip" not in names
    assert tuple(canonical[names.index("L3.d")]) == wing_rotator.CANONICAL_LANDMARKS["L3.d"]


def test_preprocessing_override_reader(tmp_path):
    landmarks, metadata = load_landmarks_override(_write_legacy_landmarks(tmp_path / "w_landmarks_override.geojson"))
    assert "L3.d" in landmarks and "L3.d" in metadata and "DTip" not in landmarks


def test_inspector_reader(tmp_path):
    parsed = _parse_landmarks_geojson(_write_legacy_landmarks(tmp_path / "w_landmarks.geojson"))
    assert parsed["L3.d"] == (5153.0, 1968.0) and "DTip" not in parsed


def test_measurement_maker_reader_and_saved_pairs(tmp_path):
    landmarks = load_landmarks_from_geojson(_write_legacy_landmarks(tmp_path / "w_landmarks.geojson"))
    assert "L3.d" in landmarks and "DTip" not in landmarks
    # A custom measurement pair saved before the rename still names "DTip".
    assert resolve_landmark_key(landmarks, "DTip") == "L3.d"
    assert resolve_landmark_key(landmarks, "L3.d") == "L3.d"
