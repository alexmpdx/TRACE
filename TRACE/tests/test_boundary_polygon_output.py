"""Tests for the ``boundary_polygon_overlay`` Stage-2 output.

The overlay draws the wing-margin landmark ring the boundary-polygon quality filter
tests, in green (simple) / red (self-intersecting) / amber (too few landmarks). It is
drawn straight from the Stage-1 landmarks GeoJSON rather than from a ``WingResult`` so
that it is written for wings a quality gate aborted — the case it exists for — and on the
landmarks-only fast paths where ``identify_wing`` never runs.

The end-to-end tests drive ``trace_folder`` with Stage 1 and ``identify_wing``
monkeypatched out, so no model, GPU or real segmentation is involved.
"""

import json
import sys
from pathlib import Path

import numpy as np
import pytest

# Sibling modules export bare package names that don't match their directory,
# so importing TRACE.pipeline needs the same sys.path set run_cli.py builds.
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

from identify_features.views.overlay import _BOUNDARY_FAIL_BGR, _BOUNDARY_OK_BGR  # noqa: E402

from TRACE.pipeline import (  # noqa: E402
    _OUTPUT_STAGE_REQUIREMENTS,
    _STAGE2_ANALYSIS_OUTPUTS,
    INTERMEDIATE_OUTPUTS,
    OUTPUT_TOOLTIPS,
    OUTPUT_TYPES,
    _render_boundary_polygon_overlay,
    _required_stages,
    resolve_skip_vein_tracing,
    trace_folder,
)

KEY = "boundary_polygon_overlay"

# Clean teardrop on a 1000 × 700 canvas (original-pixel space).
CLEAN_XY = {
    "subcostal break": (770.0, 220.0),
    "L2.d": (195.0, 240.0),
    "L3.d": (55.0, 400.0),
    "L4.d": (90.0, 490.0),
    "L5.d": (460.0, 610.0),
    "alula notch": (935.0, 370.0),
    "L4-L5": (960.0, 300.0),
    "L1-Rs": (920.0, 270.0),
}


def _crossed():
    xy = dict(CLEAN_XY)
    xy["L4.d"], xy["L5.d"] = CLEAN_XY["L5.d"], CLEAN_XY["L4.d"]
    return xy


def _write_landmarks(path: Path, points: dict) -> None:
    path.write_text(
        json.dumps(
            {
                "type": "FeatureCollection",
                "features": [
                    {
                        "type": "Feature",
                        "geometry": {"type": "Point", "coordinates": [x, y]},
                        "properties": {"classification": {"name": name}, "reliable": True},
                    }
                    for name, (x, y) in points.items()
                ],
            }
        )
    )


def _count(img, bgr):
    return int(np.all(img == np.array(bgr, dtype=np.uint8), axis=-1).sum())


def _grey(h=700, w=1000):
    return np.full((h, w, 3), 128, dtype=np.uint8)


# ---------------------------------------------------------------------------
# Registry consistency
# ---------------------------------------------------------------------------


def test_output_key_is_registered_everywhere():
    assert KEY in OUTPUT_TYPES
    assert KEY in OUTPUT_TOOLTIPS
    assert KEY in INTERMEDIATE_OUTPUTS
    # Drawn from the Stage-1 landmarks only: no hinge chop, no segmentation, no Stage 2.
    assert _OUTPUT_STAGE_REQUIREMENTS[KEY] == (True, False, False)
    assert _required_stages({KEY}) == (True, False, False)
    assert KEY not in _STAGE2_ANALYSIS_OUTPUTS
    assert resolve_skip_vein_tracing({KEY}, set()) is True


def test_tooltip_image_is_bundled():
    from TRACE.output_tooltips import _GUI_IMAGES_DIR, _OUTPUT_IMAGES

    assert KEY in _OUTPUT_IMAGES
    assert (_GUI_IMAGES_DIR / _OUTPUT_IMAGES[KEY]).is_file()


# ---------------------------------------------------------------------------
# Module-level renderer: GeoJSON → PNG, with the rescale inverse applied
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("l3d_key", ["L3.d", "DTip"])  # canonical and legacy GeoJSON key
def test_render_from_geojson_applies_inverse_scale(tmp_path, l3d_key):
    """Stage-1 GeoJSON in rescaled space (×0.5) must land on the original-resolution base."""
    import cv2

    xy = {("DTip" if (n == "L3.d" and l3d_key == "DTip") else n): (x * 0.5, y * 0.5) for n, (x, y) in CLEAN_XY.items()}
    lm = tmp_path / "w_landmarks.geojson"
    _write_landmarks(lm, xy)
    out = tmp_path / "w_boundary_polygon_overlay.png"
    assert _render_boundary_polygon_overlay(_grey(), lm, out, inverse_scale=2.0, show_landmark_labels=False)
    img = cv2.imread(str(out))
    assert _count(img, _BOUNDARY_OK_BGR) > 500
    # Without the inverse the ring would sit in the top-left quadrant only; with it,
    # the L4-L5 vertex (960, 300) is painted in the far right of the canvas.
    assert _count(img[:, 900:], _BOUNDARY_OK_BGR) > 0


def test_render_from_missing_geojson_returns_false(tmp_path):
    assert _render_boundary_polygon_overlay(_grey(), tmp_path / "nope.geojson", tmp_path / "o.png") is False
    assert not (tmp_path / "o.png").exists()


# ---------------------------------------------------------------------------
# End-to-end through trace_folder with Stage 1 + identify_wing mocked
# ---------------------------------------------------------------------------


def _fake_stage1(monkeypatch, image: Path, points: dict, rescale_factor: float = 1.0):
    """Replace ``process_folder`` with one that writes a landmarks GeoJSON + segmentation
    stub into the preproc dir and returns a single successful PipelineResult."""
    import preprocessing.pipeline as pp

    def fake_process_folder(input_dir, output_dir, **_kw):
        lm = output_dir / f"{image.stem}_landmarks.geojson"
        _write_landmarks(lm, points)
        seg = output_dir / f"{image.stem}.geojson"
        seg.write_text('{"type": "FeatureCollection", "features": []}')
        if rescale_factor != 1.0:
            import cv2

            # The preprocessed image lives in the preproc dir at rescaled resolution.
            cv2.imwrite(str(output_dir / image.name), _grey(int(700 * rescale_factor), int(1000 * rescale_factor)))
        return [
            pp.PipelineResult(
                image_path=image,
                landmarks_geojson_path=lm,
                segmentation_geojson_path=seg,
                rescale_factor=rescale_factor,
            )
        ]

    monkeypatch.setattr(pp, "process_folder", fake_process_folder)


def _fake_rejection(monkeypatch):
    import identify_features.controllers.pipeline as ifp
    from identify_features.garbage_detector import FilterStage, GarbageRejection, GarbageVerdict

    def fake_identify_wing(*_a, **_k):
        raise GarbageRejection(GarbageVerdict.reject("boundary_polygon", FilterStage.WING_OUTLINE, "ring crosses"))

    monkeypatch.setattr(ifp, "identify_wing", fake_identify_wing)


@pytest.fixture
def wing_folder(tmp_path):
    import cv2

    input_dir = tmp_path / "in"
    input_dir.mkdir()
    image = input_dir / "wing_0001.tif"
    cv2.imwrite(str(image), _grey())
    return input_dir, image, tmp_path / "out"


def _trace(input_dir, output_dir, outputs, tmp_path, **kw):
    return trace_folder(
        input_dir,
        output_dir,
        landmark_checkpoint=tmp_path / "lm.pt",
        segmentation_model_dir=tmp_path / "seg",
        outputs=outputs,
        max_workers=1,
        **kw,
    )


def test_rejected_wing_still_gets_red_overlay(wing_folder, tmp_path, monkeypatch):
    """A quality-gate abort skips every other Stage-2 output but still writes this one."""
    import cv2

    input_dir, image, output_dir = wing_folder
    _fake_stage1(monkeypatch, image, _crossed())
    _fake_rejection(monkeypatch)

    [res] = _trace(input_dir, output_dir, {"vein_overlay", "landmarks_overlay", KEY}, tmp_path)

    assert res.error_stage == "crossed boundary landmarks"
    assert res.error and res.error.startswith("Aborted by quality gate")
    bp = output_dir / "wing_0001_boundary_polygon_overlay.png"
    assert res.boundary_polygon_overlay_path == bp and bp.is_file()
    assert _count(cv2.imread(str(bp)), _BOUNDARY_FAIL_BGR) > 500
    # The other overlays are not produced for an aborted wing.
    assert not (output_dir / "wing_0001_overlay.png").exists()
    assert res.landmarks_overlay_path is None


def test_rejected_wing_without_the_output_selected_writes_nothing(wing_folder, tmp_path, monkeypatch):
    input_dir, image, output_dir = wing_folder
    _fake_stage1(monkeypatch, image, _crossed())
    _fake_rejection(monkeypatch)

    [res] = _trace(input_dir, output_dir, {"vein_overlay"}, tmp_path)

    assert res.error_stage == "crossed boundary landmarks"
    assert res.boundary_polygon_overlay_path is None
    assert not list(output_dir.glob("*_boundary_polygon_overlay.png"))


def test_landmarks_only_run_writes_green_overlay_with_rescale(wing_folder, tmp_path, monkeypatch):
    """Only this output selected → identify_wing never runs (no Stage 2 analysis), the
    ring is still drawn, and rescaled-space landmarks are mapped back onto the
    original-resolution base."""
    import cv2
    import identify_features.controllers.pipeline as ifp

    input_dir, image, output_dir = wing_folder
    half = {n: (x * 0.5, y * 0.5) for n, (x, y) in CLEAN_XY.items()}
    _fake_stage1(monkeypatch, image, half, rescale_factor=0.5)
    monkeypatch.setattr(ifp, "identify_wing", lambda *a, **k: pytest.fail("identify_wing must not run"))

    [res] = _trace(input_dir, output_dir, {KEY}, tmp_path)

    assert res.error is None
    bp = output_dir / "wing_0001_boundary_polygon_overlay.png"
    assert res.boundary_polygon_overlay_path == bp and bp.is_file()
    img = cv2.imread(str(bp))
    assert img.shape[:2] == (700, 1000)  # inverse-resized back to original resolution
    assert _count(img, _BOUNDARY_OK_BGR) > 500
    assert _count(img[:, 900:], _BOUNDARY_OK_BGR) > 0  # L4-L5 at x≈960 after the inverse
