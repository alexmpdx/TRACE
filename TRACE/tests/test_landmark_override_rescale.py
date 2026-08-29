"""Regression tests: landmark overrides must be rescaled before downstream use.

Background (the bug):

The landmark inspector's *Landmarks* tab edits points over the ORIGINAL image,
so a manual override is written to disk in original-input pixel space. But when
Stage 1 (resolution adjust) rescales the image, every downstream consumer —
Stage 4 hinge chop, Stage 5 segmentation, and the Stage-2 analysis, which reads
the landmark GeoJSON from disk and runs on the rescaled image — works in
*rescaled* pixel space. The segmentation-override branch already converts for
this (`_scale_geojson_coords(fc, rescale_factor)`), but the landmark-override
branch did not: it fed original-space coordinates straight into the rescaled
pipeline, so a hand-corrected landmark landed off by ``rescale_factor`` on every
rescaling run — silently corrupting exactly the images the user corrected.

The invariant: a landmark override placed at an original-image pixel must reach
the rescaled pipeline at ``pixel * rescale_factor`` — the same transform the
segmentation override uses — so the two stay in the same coordinate frame.
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

from preprocessing.pipeline import (  # noqa: E402
    _scale_geojson_coords,
    _scale_landmarks,
    landmarks_to_geojson,
    load_landmarks_override,
)
from TRACE.landmark_inspector_dialog import _parse_landmarks_geojson  # noqa: E402


def _write_override(path: Path, coords: dict) -> None:
    """Write a minimal landmark-override GeoJSON (inspector schema) in the
    coordinate space the caller supplies (original-input pixel space)."""
    features = [
        {
            "type": "Feature",
            "geometry": {"type": "Point", "coordinates": [x, y]},
            "properties": {"classification": {"name": name}, "reliable": True, "confidence": 1.0},
        }
        for name, (x, y) in coords.items()
    ]
    path.write_text(json.dumps({"type": "FeatureCollection", "features": features}), encoding="utf-8")


def test_scale_landmarks_noop_at_unity():
    lms = {"DTip": (100.0, 200.0), "L2-L3": (50.5, 60.5)}
    # rescale_factor 1.0 (no Stage-1 rescale) must leave coordinates untouched.
    assert _scale_landmarks(lms, 1.0) == lms


def test_scale_landmarks_multiplies_by_factor():
    lms = {"DTip": (100.0, 200.0), "L4-L5": (10.0, 20.0)}
    out = _scale_landmarks(lms, 0.5)
    assert out == {"DTip": (50.0, 100.0), "L4-L5": (5.0, 10.0)}


def test_override_roundtrip_lands_in_rescaled_space(tmp_path):
    """The core regression: a landmark placed at an original-image pixel reaches
    the pipeline (which now scales by rescale_factor) at the rescaled pixel."""
    rescale_factor = 0.4  # Stage 1 downscaled the image to 40% linear size.
    original_coords = {"DTip": (1000.0, 800.0), "subcostal break": (120.0, 640.0)}

    override_path = tmp_path / "wing_0001_landmarks_override.geojson"
    _write_override(override_path, original_coords)

    # Exactly what preprocessing.pipeline now does in the override branch.
    landmarks, _meta = load_landmarks_override(override_path)
    landmarks = _scale_landmarks(landmarks, rescale_factor)

    assert landmarks == {
        "DTip": (400.0, 320.0),
        "subcostal break": (48.0, 256.0),
    }


def test_landmark_and_segmentation_overrides_use_same_direction(tmp_path):
    """A landmark and a polygon vertex at the same original pixel must map to the
    same rescaled pixel — otherwise a corrected landmark and a corrected mask
    would disagree. Guards against the two override paths drifting apart."""
    rescale_factor = 0.4
    pt = (1000.0, 800.0)

    landmarks = _scale_landmarks({"DTip": pt}, rescale_factor)

    fc = {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "geometry": {"type": "Polygon", "coordinates": [[[pt[0], pt[1]], [1.0, 1.0], [2.0, 2.0]]]},
                "properties": {"class": "vein"},
            }
        ],
    }
    _scale_geojson_coords(fc, rescale_factor)
    scaled_vertex = fc["features"][0]["geometry"]["coordinates"][0][0]

    assert list(landmarks["DTip"]) == list(scaled_vertex) == [400.0, 320.0]


# --- Bug B: the inspector displays landmarks over the ORIGINAL image, so it must
# map rescaled-space automated outputs back to original space via the stored
# rescale_factor. Overrides / on-demand geojsons carry no factor and pass through.


def _write_fc(path: Path, coords: dict, fc_props=None) -> None:
    fc = landmarks_to_geojson({n: (x, y) for n, (x, y) in coords.items()}, fc_props=fc_props)
    path.write_text(json.dumps(fc), encoding="utf-8")


def test_parser_inverse_scales_rescaled_output(tmp_path):
    """An automated `_landmarks.geojson` stored in rescaled space, tagged with
    its rescale_factor, is mapped back to original space for display."""
    rescale_factor = 0.4
    # Coordinates as written to disk by the pipeline: rescaled-pixel space.
    rescaled_coords = {"DTip": (400.0, 320.0), "L4-L5": (48.0, 256.0)}
    path = tmp_path / "wing_0001_landmarks.geojson"
    _write_fc(path, rescaled_coords, fc_props={"rescale_factor": rescale_factor})

    parsed = _parse_landmarks_geojson(path)

    # Back in original-image pixel space, aligned with the original image shown.
    assert parsed == {"DTip": (1000.0, 800.0), "L4-L5": (120.0, 640.0)}


def test_parser_passes_through_when_no_rescale_factor(tmp_path):
    """Overrides and on-demand regenerations are already in original space and
    carry no rescale_factor — the parser must not touch their coordinates."""
    coords = {"DTip": (1000.0, 800.0)}
    path = tmp_path / "wing_0001_landmarks_override.geojson"
    _write_fc(path, coords, fc_props=None)  # no rescale_factor

    assert _parse_landmarks_geojson(path) == coords


def test_parser_noop_at_unity_factor(tmp_path):
    coords = {"DTip": (1000.0, 800.0)}
    path = tmp_path / "wing_0001_landmarks.geojson"
    _write_fc(path, coords, fc_props={"rescale_factor": 1.0})

    assert _parse_landmarks_geojson(path) == coords


def test_display_roundtrip_survives_rescale(tmp_path):
    """End-to-end: a landmark truly at an original pixel, put through the
    pipeline's write (rescaled coords + stored factor), comes back to the same
    original pixel in the inspector — so the predicted point lands on the wing."""
    original_pixel = {"DTip": (1000.0, 800.0)}
    rescale_factor = 0.4

    # What the pipeline writes: coords scaled into rescaled space + the factor.
    rescaled = _scale_landmarks(original_pixel, rescale_factor)
    path = tmp_path / "wing_0001_landmarks.geojson"
    _write_fc(path, rescaled, fc_props={"rescale_factor": rescale_factor})

    # What the inspector reads back for display over the original image.
    assert _parse_landmarks_geojson(path) == original_pixel
