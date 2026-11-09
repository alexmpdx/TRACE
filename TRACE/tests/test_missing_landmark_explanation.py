"""A gate-dropped landmark that a later stage needs is reported with its gate verdict."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

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

from preprocessing.pipeline import explain_missing_landmark  # noqa: E402

META = {
    "subcostal break": {
        "reliable": False,
        "gate_reason": "peak=0.293<0.449; second_peak=0.50>0.26",
        "confidence": 0.293,
    },
    "alula notch": {"reliable": True, "gate_reason": "", "confidence": 0.8},
}
MSG = "Missing required landmark: subcostal break"


def test_gate_dropped_landmark_gets_the_verdict_appended():
    out = explain_missing_landmark(MSG, META)
    assert out.startswith(MSG)
    assert "dropped by the confidence gate: peak=0.293<0.449; second_peak=0.50>0.26" in out
    assert "not in the core set" in out


def test_reliable_or_unknown_landmarks_are_left_alone():
    assert explain_missing_landmark("Missing required landmark: alula notch", META) == (
        "Missing required landmark: alula notch"
    )
    assert explain_missing_landmark("Missing required landmark: l2_d", META) == "Missing required landmark: l2_d"
    assert explain_missing_landmark(MSG, None) == MSG
    assert explain_missing_landmark(MSG, {}) == MSG


def test_other_messages_pass_through_unchanged():
    assert explain_missing_landmark("something else went wrong", META) == "something else went wrong"


def test_hinge_chop_error_is_explained_end_to_end(tmp_path):
    """The real hinge-chop ValueError, wrapped the way Stage 4 wraps it."""
    from hinge_chopper import build_hinge_line

    landmarks = {"alula notch": (10.0, 20.0), "L1-Rs": (30.0, 40.0)}  # subcostal break dropped
    with pytest.raises(ValueError) as info:
        build_hinge_line(landmarks)
    explained = explain_missing_landmark(str(info.value), META)
    assert explained != str(info.value)
    assert "peak=0.293<0.449" in explained
