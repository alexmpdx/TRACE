"""Regression tests for bug #37 — the batch CSV came out entirely blank.

Background:

``_run`` processes images in chunks of ``_INTERMEDIATE_CHUNK_SIZE`` and calls
``_wipe_preproc_dir`` at the end of EVERY chunk, including the last one, to
bound peak intermediate disk usage (the 13ea8fb0 disk-full fix). The batch CSV
block runs AFTER the chunk loop and resolves ``user_dist_landmark_paths`` —
paths that point into the just-wiped ``preproc_dir``.

Both CSV writers (``write_landmark_csv_batch`` on the landmarks-only fast path
and ``augment_csv_with_user_distances`` on the identifyFeatures path) guard the
read with ``lm_path.exists()`` and fall through to blank cells when it's False.
A deleted file is therefore indistinguishable from a wing whose landmarks were
never detected: every measurement column blank, one WARNING per wing per pair,
no error — exactly what bug #37 reported for a 97-image (single-chunk) run.

The invariant: whatever else the wipe removes, a landmark GeoJSON written
during a chunk must still be readable when the CSV block runs.
"""

import sys
from pathlib import Path

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

from TRACE.pipeline import _wipe_preproc_dir  # noqa: E402


def test_wipe_preserves_landmark_geojsons(tmp_path):
    """The files the post-chunk CSV block reads survive the wipe."""
    preproc = tmp_path / "intermediates"
    preproc.mkdir()
    stage3 = preproc / "wing_0001_landmarks.geojson"
    stage3.write_text("{}")
    # Stage 6 rebinds PipelineResult.landmarks_geojson_path to this name.
    rotated = preproc / "wing_0001_rotated_landmarks.geojson"
    rotated.write_text("{}")

    _wipe_preproc_dir(preproc)

    assert stage3.exists()
    assert rotated.exists()


def test_wipe_still_removes_the_bulk_intermediates(tmp_path):
    """The disk-full fix the wipe exists for is unaffected by the exemption."""
    preproc = tmp_path / "intermediates"
    preproc.mkdir()
    heavy = [
        preproc / "wing_0001_resampled.ome.tif",
        preproc / "wing_0001_chopped.ome.tif",
        preproc / "wing_0001_isolated.ome.tif",
        preproc / "wing_0001.geojson",  # segmentation
        preproc / "wing_0001_output.geojson",
    ]
    for f in heavy:
        f.write_text("x")
    subdir = preproc / "scratch"
    subdir.mkdir()
    (subdir / "nested.tif").write_text("x")

    _wipe_preproc_dir(preproc)

    assert not any(f.exists() for f in heavy)
    assert not subdir.exists()
    assert preproc.is_dir()  # the dir itself is reused by the next chunk


def test_landmark_csv_is_populated_after_a_wipe(tmp_path):
    """End-to-end on the reported shape: wipe, then write the CSV.

    This is the assertion bug #37 would have failed — before the fix the CSV
    had a header row plus one all-empty row per wing.
    """
    import csv

    from measurement_maker import pairs_from_dicts, write_landmark_csv_batch

    preproc = tmp_path / "intermediates"
    preproc.mkdir()
    (preproc / "wing_0001_resampled.ome.tif").write_text("x")  # gets wiped

    lm_path = preproc / "wing_0001_landmarks.geojson"
    _write_landmarks(
        lm_path,
        {
            "L1-Rs": (753.0, 1285.0),
            "DTip": (5153.0, 1968.0),
            "ACV.p": (1839.0, 1678.0),
            "PCV.a": (2655.0, 1990.0),
            "L2.d": (4100.0, 900.0),
        },
    )

    # Stage 2 records the path; the chunk then wipes; the CSV block reads it.
    specimen_landmarks = {"wing_0001": lm_path}
    _wipe_preproc_dir(preproc)

    csv_path = tmp_path / "measurements.csv"
    write_landmark_csv_batch(
        csv_path,
        specimen_landmarks,
        pairs_from_dicts([{"name_a": "L2.d", "name_b": "DTip", "label": "L2d_dtip"}]),
        measurement_groups={"cv_ratio"},
        um_per_px=4.1828,
    )

    row = next(iter(csv.DictReader(csv_path.open(encoding="utf-8-sig"))))
    assert row["specimen"] == "wing_0001"
    assert row["wing length_px"]
    assert row["crossvein distance_px"]
    assert row["CV ratio"]
    assert row["custom_L2d_dtip_px"]
    assert row["custom_L2d_dtip_um"]


def _write_landmarks(path: Path, points: dict) -> None:
    import json

    path.write_text(
        json.dumps(
            {
                "type": "FeatureCollection",
                "properties": {"effective_um_per_px": 4.1828},
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


# ---------------------------------------------------------------------------
# The wipe must never reach the run's final outputs.
#
# Structurally it can't today: `preproc_dir` is assigned in exactly one place
# (TRACE/pipeline.py process_folder) as either `output_dir / "intermediates"`
# — a CHILD of the output dir — or a `TemporaryDirectory` outside it entirely,
# and the wipe only iterates direct children, never walking upward. Every final
# artifact (CSV, overlays, exported GeoJSONs) is written to `output_dir`
# itself, and the GUI's run bookkeeping sits beside them. These tests pin that
# separation so a future refactor of either dir can't quietly break it.
# ---------------------------------------------------------------------------


def test_wipe_leaves_the_output_dir_untouched(tmp_path):
    """The keep_intermediates layout: preproc_dir is a child of output_dir."""
    output_dir = tmp_path / "run_20260823-183057"
    preproc = output_dir / "intermediates"
    preproc.mkdir(parents=True)

    finals = [
        output_dir / "measurements_run_20260823-183057.csv",
        output_dir / "wing_0001_cv_ratio_overlay.png",
        output_dir / "wing_0001_output.geojson",
        output_dir / "run.log",
        output_dir / "settings.yaml",
        output_dir / "_run_state.json",
    ]
    for f in finals:
        f.write_text("final")
    (preproc / "wing_0001_resampled.ome.tif").write_text("intermediate")

    _wipe_preproc_dir(preproc, output_dir)

    assert all(f.exists() for f in finals)
    assert not (preproc / "wing_0001_resampled.ome.tif").exists()


def test_wipe_refuses_when_preproc_dir_is_the_output_dir(tmp_path):
    """Containment check: never wipe a dir that IS (or contains) the outputs."""
    output_dir = tmp_path / "run_20260823-183057"
    output_dir.mkdir()
    csv = output_dir / "measurements.csv"
    csv.write_text("specimen,CV ratio\n")
    overlay = output_dir / "wing_0001_cv_ratio_overlay.png"
    overlay.write_text("png")

    # preproc_dir == output_dir — the refactor accident this guards against.
    _wipe_preproc_dir(output_dir, output_dir)

    assert csv.exists()
    assert overlay.exists()


def test_wipe_refuses_when_preproc_dir_is_a_parent_of_the_output_dir(tmp_path):
    parent = tmp_path / "work"
    output_dir = parent / "run_20260823-183057"
    output_dir.mkdir(parents=True)
    (output_dir / "measurements.csv").write_text("specimen\n")
    sibling = parent / "someone_elses_file.tif"
    sibling.write_text("x")

    _wipe_preproc_dir(parent, output_dir)

    assert (output_dir / "measurements.csv").exists()
    assert sibling.exists()


def test_wipe_skips_final_output_filenames_even_inside_preproc_dir(tmp_path):
    """Second net: name-based skip, for anything the containment check misses."""
    preproc = tmp_path / "intermediates"
    preproc.mkdir()
    protected = [
        preproc / "measurements.csv",
        preproc / "measurements.csv.append_source",
        preproc / "run.log",
        preproc / "settings.yaml",
        preproc / "_run_state.json",
    ]
    for f in protected:
        f.write_text("x")
    doomed = preproc / "wing_0001_chopped.ome.tif"
    doomed.write_text("x")

    _wipe_preproc_dir(preproc, tmp_path / "output")

    assert all(f.exists() for f in protected)
    assert not doomed.exists()
