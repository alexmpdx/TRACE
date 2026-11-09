"""The landmark inspector shows the landmark set a run would see, not the raw model output."""

from __future__ import annotations

import json
import sys
from pathlib import Path

# Sibling modules export bare package names that don't match their directory.
ROOT = Path(__file__).resolve().parents[2]
for _p in (
    ROOT,
    ROOT / "HingeChopper",
    ROOT / "modelTOjson",
    ROOT / "identifyFeatures",
    ROOT / "wingRotator",
    ROOT / "measurementMaker",
    ROOT / "scaleEstimator",
    ROOT / "LandmarkLocator",
):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from TRACE.landmark_inspector_dialog import (  # noqa: E402
    _internal_landmark_name,
    _parse_failed_gate_landmarks,
    _parse_landmarks_geojson,
    _parse_landmarks_geojson_meta,
    _partition_landmarks,
)


def _feature(name, xy, reliable=True, reason=""):
    return {
        "type": "Feature",
        "geometry": {"type": "Point", "coordinates": list(xy)},
        "properties": {"classification": {"name": name}, "reliable": reliable, "gate_reason": reason},
    }


META = {
    "L3.d": {"xy": (10.0, 10.0), "reliable": True, "gate_reason": ""},
    "subcostal break": {
        "xy": (20.0, 20.0),
        "reliable": False,
        "gate_reason": "peak=0.293<0.449; second_peak=0.50>0.26",
    },
    "ACV.a": {"xy": (30.0, 30.0), "reliable": False, "gate_reason": "peak=0.05<0.193; sharpness=1.10<1.31"},
    "ACV.p": {"xy": (40.0, 40.0), "reliable": False, "gate_reason": "peak=0.03<0.04; sharpness=1.05<1.28"},
}


def test_internal_name_matches_gate_config_keys():
    assert _internal_landmark_name("subcostal break") == "subcostal_break"
    assert _internal_landmark_name("L3.d") == "l3_d"
    assert _internal_landmark_name("L1-Rs") == "l1_rs"
    assert _internal_landmark_name("DTip") == "l3_d"  # legacy name resolves first


def test_non_core_drops_are_hidden_and_listed():
    shown, red, hidden = _partition_landmarks(META, include_unreliable=False, core_internal=set(), failed_names=set())
    assert list(shown) == ["L3.d"]
    assert red == set()
    assert set(hidden) == {"subcostal break", "ACV.a", "ACV.p"}
    assert hidden["ACV.a"].startswith("peak=0.05<0.193")


def test_core_failure_is_shown_in_red():
    shown, red, hidden = _partition_landmarks(
        META, include_unreliable=False, core_internal={"subcostal_break", "l3_d"}, failed_names=set()
    )
    assert list(shown) == ["L3.d", "subcostal break"]
    assert red == {"subcostal break"}
    assert set(hidden) == {"ACV.a", "ACV.p"}


def test_recorded_failure_surfaces_the_dropped_landmark():
    failed = _parse_failed_gate_landmarks(
        "Missing required landmark: subcostal break (dropped by the confidence gate: peak=0.293<0.449; "
        "second_peak=0.50>0.26; it is not in the core set, so the gate did not abort the image)"
    )
    assert failed == {"subcostal break"}
    shown, red, hidden = _partition_landmarks(META, include_unreliable=False, core_internal=set(), failed_names=failed)
    assert "subcostal break" in shown and red == {"subcostal break"}
    assert set(hidden) == {"ACV.a", "ACV.p"}


def test_include_unreliable_shows_everything_red():
    shown, red, hidden = _partition_landmarks(META, include_unreliable=True, core_internal=set(), failed_names=set())
    assert list(shown) == list(META)
    assert red == {"subcostal break", "ACV.a", "ACV.p"}
    assert hidden == {}


def test_core_gate_message_still_parses():
    msg = "Core landmarks failed confidence gate: l3_d (peak=0.029<0.134), subcostal break (peak=0.293<0.405; second_peak=0.50>0.41)"
    assert _parse_failed_gate_landmarks(msg) == {"l3_d", "subcostal break"}
    assert _parse_failed_gate_landmarks("Aborted by quality gate (solidity): 0.91 below 0.94") == set()
    assert _parse_failed_gate_landmarks("") == set()


def test_meta_parser_reads_flags_and_inverse_rescales(tmp_path):
    path = tmp_path / "w_landmarks.geojson"
    path.write_text(
        json.dumps(
            {
                "type": "FeatureCollection",
                "properties": {"rescale_factor": 2.0},
                "features": [
                    _feature("L3.d", (100.0, 50.0)),
                    _feature("ACV.a", (20.0, 10.0), reliable=False, reason="peak=0.05<0.193"),
                    # minimal override schema: no flags → reliable
                    {
                        "type": "Feature",
                        "geometry": {"type": "Point", "coordinates": [4.0, 6.0]},
                        "properties": {"classification": {"name": "alula notch"}},
                    },
                ],
            }
        )
    )
    meta = _parse_landmarks_geojson_meta(path)
    assert meta["L3.d"] == {"xy": (50.0, 25.0), "reliable": True, "gate_reason": ""}
    assert meta["ACV.a"] == {"xy": (10.0, 5.0), "reliable": False, "gate_reason": "peak=0.05<0.193"}
    assert meta["alula notch"]["reliable"] is True
    # coordinate-only wrapper is unchanged for existing callers
    assert _parse_landmarks_geojson(path) == {n: e["xy"] for n, e in meta.items()}
