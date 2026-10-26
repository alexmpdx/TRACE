"""Tests for the boundary-polygon garbage filter.

Spec: documentation/spec_boundary_polygon_filter.md

The eight wing-margin landmarks are joined into a ring; the filter rejects when two
non-adjacent edges intersect. Helper tests build ``Landmark`` dicts by hand; the
integration tests run ``identify_wing`` on a tiny synthetic detection GeoJSON to prove
the filter fires before the image read. Real-data tests use the untracked landmark
sets under ``identifyFeatures/`` and skip when those folders are absent.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest
from identify_features.config import PipelineConfig
from identify_features.garbage_detector import (
    BOUNDARY_LANDMARK_ORDER,
    FilterContext,
    FilterStage,
    GarbageRejection,
    boundary_polygon_crossings,
    filter_label,
    run_stage_filters,
)
from identify_features.garbage_detector.filters.boundary_polygon import BoundaryPolygonFilter
from identify_features.models.datatypes import Landmark
from shapely.geometry import Point

HERE = Path(__file__).resolve().parent
IF_ROOT = HERE.parent  # identifyFeatures/

# A clean, distal-left wing (real coordinates from a passing LandmarkLocator output).
CLEAN_XY: dict[str, tuple[float, float]] = {
    "subcostal break": (3847.1, 1098.5),
    "L2.d": (975.9, 1193.4),
    "L3.d": (265.3, 2009.1),
    "L4.d": (446.0, 2456.5),
    "L5.d": (2305.4, 3037.8),
    "alula notch": (4676.2, 1843.2),
    "L4-L5": (4791.8, 1512.8),
    "L1-Rs": (4590.3, 1337.6),
    # Non-boundary landmarks — must be ignored by the filter.
    "ACV.a": (3430.9, 1585.8),
    "PCV.p": (2838.5, 2392.8),
    "L2-L3": (4175.9, 1420.1),
}


def _lm(xy: dict[str, tuple[float, float]], reliable: bool = True) -> dict[str, Landmark]:
    return {n: Landmark(name=n, point=Point(x, y), reliable=reliable) for n, (x, y) in xy.items()}


def _transform(xy, *, mirror=False, angle_deg=0.0, scale=1.0):
    c, s = math.cos(math.radians(angle_deg)), math.sin(math.radians(angle_deg))
    out = {}
    for n, (x, y) in xy.items():
        if mirror:
            x = -x
        out[n] = (scale * (c * x - s * y) + 123.4, scale * (s * x + c * y) - 56.7)
    return out


# ---------------------------------------------------------------------------
# boundary_polygon_crossings helper
# ---------------------------------------------------------------------------


def test_clean_wing_is_simple():
    assert boundary_polygon_crossings(_lm(CLEAN_XY)) == []


def test_mirrored_wing_is_simple():
    assert boundary_polygon_crossings(_lm(_transform(CLEAN_XY, mirror=True))) == []


def test_rotated_and_scaled_wing_is_simple():
    assert boundary_polygon_crossings(_lm(_transform(CLEAN_XY, angle_deg=137.0, scale=0.3))) == []


def test_swapped_l4d_l5d_crosses():
    xy = dict(CLEAN_XY)
    xy["L4.d"], xy["L5.d"] = CLEAN_XY["L5.d"], CLEAN_XY["L4.d"]
    crossings = boundary_polygon_crossings(_lm(xy))
    assert crossings
    assert crossings[0][0] == "L3.d→L4.d"


def test_l3d_on_hinge_crosses():
    xy = dict(CLEAN_XY)
    xy["L3.d"] = (4900.0, 2000.0)  # proximal of the alula notch, i.e. on the hinge side
    assert boundary_polygon_crossings(_lm(xy))


def test_fewer_than_four_vertices_is_undefined():
    xy = {n: CLEAN_XY[n] for n in ("subcostal break", "L2.d", "L3.d")}
    assert boundary_polygon_crossings(_lm(xy)) is None


def test_four_vertex_quad_vs_bowtie():
    quad = {"subcostal break": (0, 0), "L2.d": (10, 0), "L3.d": (10, 10), "L4.d": (0, 10)}
    assert boundary_polygon_crossings(_lm(quad)) == []
    bowtie = {"subcostal break": (0, 0), "L2.d": (10, 10), "L3.d": (10, 0), "L4.d": (0, 10)}
    assert boundary_polygon_crossings(_lm(bowtie)) == [("subcostal break→L2.d", "L3.d→L4.d")]


def test_non_boundary_landmarks_are_ignored():
    xy = dict(CLEAN_XY)
    xy["ACV.a"] = (-5000.0, -5000.0)  # absurd, but not on the ring
    assert boundary_polygon_crossings(_lm(xy)) == []
    assert "ACV.a" not in BOUNDARY_LANDMARK_ORDER


def test_coincident_non_adjacent_vertices_are_rejected():
    xy = dict(CLEAN_XY)
    xy["L5.d"] = CLEAN_XY["L2.d"]  # L5.d stacked on L2.d (non-adjacent on the ring)
    assert boundary_polygon_crossings(_lm(xy))


# ---------------------------------------------------------------------------
# Filter via run_stage_filters
# ---------------------------------------------------------------------------


def _ctx(landmarks, **cfg):
    config = PipelineConfig(**cfg)
    # Isolate this filter from the shape filters that share the WING_OUTLINE hook.
    config.solidity_filter_enabled = False
    config.fragmentation_filter_enabled = False
    return FilterContext(config=config, specimen_id="t", landmarks=landmarks)


def test_filter_passes_clean_wing_and_records_scratch():
    ctx = _ctx(_lm(CLEAN_XY))
    run_stage_filters(FilterStage.WING_OUTLINE, ctx)
    assert ctx.scratch["boundary_polygon_crossings"] == 0


def test_filter_rejects_crossing_wing():
    xy = dict(CLEAN_XY)
    xy["L4.d"], xy["L5.d"] = CLEAN_XY["L5.d"], CLEAN_XY["L4.d"]
    ctx = _ctx(_lm(xy))
    with pytest.raises(GarbageRejection) as exc:
        run_stage_filters(FilterStage.WING_OUTLINE, ctx)
    assert exc.value.verdict.filter_name == "boundary_polygon"
    assert str(exc.value).startswith("boundary landmark polygon self-intersects:")
    assert "L3.d" in str(exc.value)


def test_filter_disabled_does_not_run():
    xy = dict(CLEAN_XY)
    xy["L4.d"], xy["L5.d"] = CLEAN_XY["L5.d"], CLEAN_XY["L4.d"]
    ctx = _ctx(_lm(xy), boundary_polygon_filter_enabled=False)
    run_stage_filters(FilterStage.WING_OUTLINE, ctx)
    assert "boundary_polygon_crossings" not in ctx.scratch


def test_filter_passes_when_landmarks_missing_or_none():
    ctx = _ctx(None)
    run_stage_filters(FilterStage.WING_OUTLINE, ctx)
    assert "boundary_polygon_crossings" not in ctx.scratch
    ctx = _ctx(_lm({n: CLEAN_XY[n] for n in ("subcostal break", "L2.d")}))
    run_stage_filters(FilterStage.WING_OUTLINE, ctx)


def test_filter_label_and_stage():
    assert filter_label("boundary_polygon") == "crossed boundary landmarks"
    assert BoundaryPolygonFilter.stage is FilterStage.WING_OUTLINE


# ---------------------------------------------------------------------------
# Integration: identify_wing fires the filter before the image read
# ---------------------------------------------------------------------------


def _write_inputs(tmp_path: Path, xy: dict[str, tuple[float, float]]) -> tuple[Path, Path]:
    def poly(coords, cls):
        return {
            "type": "Feature",
            "properties": {"class": cls},
            "geometry": {"type": "Polygon", "coordinates": [coords]},
        }

    det = {
        "type": "FeatureCollection",
        "features": [
            poly([[0, 0], [1000, 0], [1000, 100], [0, 100], [0, 0]], "vein"),
            poly([[0, 100], [1000, 100], [1000, 1000], [0, 1000], [0, 100]], "intervein"),
        ],
    }
    lms = {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "properties": {"classification": {"name": n}, "reliable": True},
                "geometry": {"type": "Point", "coordinates": [x, y]},
            }
            for n, (x, y) in xy.items()
        ],
    }
    det_p, lm_p = tmp_path / "w_detections.geojson", tmp_path / "w_landmarks.geojson"
    det_p.write_text(json.dumps(det))
    lm_p.write_text(json.dumps(lms))
    return det_p, lm_p


@pytest.mark.parametrize("skip_vein_tracing", [False, True])
@pytest.mark.parametrize("l3d_key", ["L3.d", "DTip"])  # canonical and legacy GeoJSON key
def test_identify_wing_rejects_before_image_read(tmp_path, skip_vein_tracing, l3d_key):
    from identify_features.controllers.pipeline import identify_wing

    xy = dict(CLEAN_XY)
    xy["L4.d"], xy["L5.d"] = CLEAN_XY["L5.d"], CLEAN_XY["L4.d"]
    xy[l3d_key] = xy.pop("L3.d")
    det_p, lm_p = _write_inputs(tmp_path, xy)
    config = PipelineConfig()
    config.solidity_filter_enabled = False
    config.fragmentation_filter_enabled = False
    config.skip_vein_tracing = skip_vein_tracing
    # A nonexistent image: if the filter did NOT run first we'd get FileNotFoundError.
    with pytest.raises(GarbageRejection) as exc:
        identify_wing(det_p, lm_p, tmp_path / "does_not_exist.tif", config=config, specimen_id="w")
    assert exc.value.verdict.filter_name == "boundary_polygon"


# ---------------------------------------------------------------------------
# Real data (untracked folders; skipped when absent)
# ---------------------------------------------------------------------------

_PROBLEM_STEMS = (
    "20241207_en_PknRNAi_108870_0009",
    "20241207_en_PknRNAi_108870_0011",
    "20241207_en_PknRNAi_108870_0013",
    "20241211_en_PknRNAi_108870_0004",
)


def _load_real(path: Path):
    from identify_features.models.geojson_io import load_landmarks_geojson

    return load_landmarks_geojson(path)


@pytest.mark.parametrize("stem", _PROBLEM_STEMS)
def test_known_problem_wings_are_rejected(stem):
    p = IF_ROOT / "problemWings" / f"{stem}_landmarks.geojson"
    if not p.exists():
        pytest.skip(f"real-data fixture absent: {p}")
    assert boundary_polygon_crossings(_load_real(p))


def test_clean_locator_outputs_all_pass():
    folder = IF_ROOT / "LandmarkLocator_output"
    files = sorted(folder.glob("*_landmarks.geojson"))
    if not files:
        pytest.skip(f"real-data fixture folder absent: {folder}")
    bad = [f.name for f in files if boundary_polygon_crossings(_load_real(f))]
    assert bad == []
