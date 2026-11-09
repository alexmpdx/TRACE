"""Batch landmark gate: the fences a run resolves are recorded so the inspector can reuse them."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

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

from preprocessing.pipeline import PipelineResult  # noqa: E402
from TRACE.landmark_inspector_dialog import _core_internal_from_config, _run_resolved_gate_config  # noqa: E402
from TRACE.pipeline import group_batch_gate_configs  # noqa: E402
from TRACE.run_state import RunManifest, load_manifest, manifest_path, save_manifest  # noqa: E402

CFG_A = {
    "peak": {"per_landmark": {"l3_d": 0.5}},
    "core_landmarks": ["l3_d", "subcostal_break"],
    "batch": {"mode": "batch_mad"},
}
CFG_B = {"peak": {"per_landmark": {"l3_d": 0.4}}, "core_landmarks": [], "batch": {"mode": "batch_mad"}}


def test_pipeline_result_defaults_to_no_resolved_config():
    assert PipelineResult(image_path=Path("a.tif")).gate_config_resolved is None


def test_group_by_shared_config_identity():
    results = [
        SimpleNamespace(image_path=Path("/in/a.tif"), gate_config_resolved=CFG_A),
        SimpleNamespace(image_path=Path("/in/b.tif"), gate_config_resolved=CFG_A),
        SimpleNamespace(image_path=Path("/in/fixed.tif"), gate_config_resolved=None),
        SimpleNamespace(image_path=Path("/in/c.tif"), gate_config_resolved=CFG_B),
    ]
    groups = group_batch_gate_configs(results)
    assert groups == [(["a.tif", "b.tif"], CFG_A), (["c.tif"], CFG_B)]
    assert groups[0][1] is CFG_A
    assert group_batch_gate_configs([]) == []


def test_manifest_records_and_looks_up_latest_chunk(tmp_path):
    m = RunManifest()
    m.record_batch_gate(["a.tif", "b.tif"], CFG_A)
    m.record_batch_gate(["b.tif"], CFG_B)  # a later chunk re-gated b
    assert m.gate_config_for("a.tif") == CFG_A
    assert m.gate_config_for("b.tif") == CFG_B
    assert m.gate_config_for("zzz.tif") is None

    save_manifest(tmp_path, m)
    loaded = load_manifest(tmp_path)
    assert loaded is not None
    assert loaded.batch_gate_configs == m.batch_gate_configs
    assert loaded.gate_config_for("a.tif") == CFG_A


def test_manifest_without_the_field_loads_empty(tmp_path):
    save_manifest(tmp_path, RunManifest())
    data = json.loads(manifest_path(tmp_path).read_text())
    del data["batch_gate_configs"]
    data["failure_messages"] = {}
    manifest_path(tmp_path).write_text(json.dumps(data))
    loaded = load_manifest(tmp_path)
    assert loaded is not None and loaded.batch_gate_configs == []
    # malformed entries are dropped rather than crashing the load
    data["batch_gate_configs"] = [{"images": ["x"], "gate_config": "nope"}, {"images": ["y"], "gate_config": CFG_B}]
    manifest_path(tmp_path).write_text(json.dumps(data))
    loaded = load_manifest(tmp_path)
    assert loaded is not None and [e["images"] for e in loaded.batch_gate_configs] == [["y"]]


def test_inspector_reads_the_run_config_through_the_window():
    assert _core_internal_from_config(CFG_A) == {"l3_d", "subcostal_break"}
    assert _core_internal_from_config(None) == set()
    window = SimpleNamespace(resolved_gate_config_for=lambda name: CFG_A if name == "a.tif" else None)
    assert _run_resolved_gate_config(window, "a.tif") == CFG_A
    assert _run_resolved_gate_config(window, "other.tif") is None
    assert _run_resolved_gate_config(SimpleNamespace(), "a.tif") is None  # host without the hook
