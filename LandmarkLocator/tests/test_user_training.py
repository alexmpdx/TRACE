"""Training from explicit (annotation, image) pairs: scratch, then fine-tune from that model."""

import json
from pathlib import Path

import cv2
import numpy as np
import pytest
import yaml

from landmark_locator.training import user_training
from landmark_locator.training.train import TrainingCancelled
from landmark_locator.training.user_training import make_splits, summarize_samples, train_from_samples

NAMES = ["L3.d", "L1-Rs", "alula notch"]


def _write_sample(folder: Path, idx: int, names=NAMES) -> tuple[Path, Path]:
    """One small synthetic image plus an inspector-style annotation stored apart from it."""
    rng = np.random.default_rng(idx)
    image = rng.integers(0, 255, (88, 128, 3), dtype=np.uint8)
    image_path = folder / f"wing_{idx}.png"
    cv2.imwrite(str(image_path), image)
    features = [
        {
            "type": "Feature",
            "geometry": {"type": "Point", "coordinates": [20.0 + 30 * k + idx, 30.0 + idx]},
            "properties": {"classification": {"name": name}, "reliable": True, "confidence": 1.0},
        }
        for k, name in enumerate(names)
    ]
    ann_dir = folder / "manual_overrides"
    ann_dir.mkdir(exist_ok=True)
    ann_path = ann_dir / f"wing_{idx}_landmarks_override.geojson"
    ann_path.write_text(json.dumps({"type": "FeatureCollection", "features": features}))
    return ann_path, image_path


@pytest.fixture()
def tiny_config(monkeypatch):
    """Small input and no ImageNet download so a fold trains in about a second on CPU."""
    cfg = json.loads(json.dumps(user_training.FALLBACK_CONFIG))
    cfg["input"] = {"width": 128, "height": 96}
    cfg["model"]["pretrained"] = False
    cfg["training"]["batch_size"] = 2
    monkeypatch.setattr(user_training, "_scratch_template", lambda: json.loads(json.dumps(cfg)))


def test_make_splits_cover_every_sample_once():
    splits = make_splits(7, 3)
    assert len(splits) == 3
    assert sorted(i for _, val in splits for i in val) == list(range(7))
    for train, val in splits:
        assert not set(train) & set(val) and len(train) + len(val) == 7
    ((train, val),) = make_splits(10, 1)
    assert len(val) == 2 and len(train) == 8
    assert len(make_splits(3, 5)) == 3  # capped at the sample count
    with pytest.raises(ValueError):
        make_splits(1, 5)


def test_scratch_then_finetune(tmp_path, tiny_config):
    samples = [_write_sample(tmp_path, i) for i in range(4)]

    scratch_dir = tmp_path / "scratch_model"
    info = train_from_samples(samples, scratch_dir, n_folds=2, epochs=1, device="cpu")
    assert info["mode"] == "scratch"
    assert info["landmark_order"] == ["alula_notch", "l1_rs", "l3_d"]
    assert sorted(p.name for p in scratch_dir.glob("*.pt")) == ["best_fold0.pt", "best_fold1.pt"]
    (scratch_dir / "gate_config.yaml").write_text("confidence:\n  peak:\n    global: 0.2\n")

    # Fine-tune: an extra landmark the base model has no channel for is ignored, one
    # sample carrying only that landmark is skipped, and the channel order is inherited.
    extra = _write_sample(tmp_path, 9, names=["mystery point"])
    samples.append(_write_sample(tmp_path, 5, names=NAMES + ["mystery point"]))
    events = []
    tuned_dir = tmp_path / "tuned_model"
    info = train_from_samples(
        samples + [extra],
        tuned_dir,
        base_model=scratch_dir,
        n_folds=2,
        epochs=2,
        device="cpu",
        epoch_callback=events.append,
    )
    assert info["mode"] == "fine-tune" and info["n_images"] == 5
    assert info["landmark_order"] == ["alula_notch", "l1_rs", "l3_d"]
    assert info["ignored_landmarks"] == {"mystery point": 2}
    assert {e["fold"] for e in events} == {0, 1} and events[0]["n_folds"] == 2
    # The starting model is scored first and kept unless training beats it.
    for fold in info["folds"]:
        assert fold["baseline_pixel_error"] is not None
        assert fold["best_pixel_error"] <= fold["baseline_pixel_error"]
    # Too few held-out predictions to calibrate anything: nothing the base sidecar set is lost.
    assert info["gate"]["calibrated"] == {} and info["gate"]["global_only"]
    tuned_gate = yaml.safe_load((tuned_dir / "gate_config.yaml").read_text())["confidence"]
    assert tuned_gate["peak"]["global"] == 0.2 and set(tuned_gate["tiers"]) == {"permissive", "standard", "strict"}
    assert info["folds"][0]["val_images"] and info["folds"][0]["val_annotations"]
    assert json.loads((tuned_dir / "training_info.json").read_text())["n_images"] == 5

    # The result loads through the same factory TRACE uses.
    from landmark_locator.inference.predict import make_predictor

    predictor = make_predictor(tuned_dir, device="cpu")
    assert predictor.landmark_order == ["alula_notch", "l1_rs", "l3_d"]
    assert predictor.gate_config["peak"]["global"] == 0.2


def test_build_gate_config_calibrates_where_there_is_enough_data():
    """Enough held-out predictions → own percentiles; too few → the base model's tier value."""
    from landmark_locator.training.user_training import build_gate_config

    rng = np.random.default_rng(0)
    collected = {
        "l3_d": {
            "peak": rng.uniform(0.5, 0.9, 40).tolist(),
            "sharpness": rng.uniform(1.5, 2.0, 40).tolist(),
            "second_peak_ratio": rng.uniform(0.1, 0.4, 40).tolist(),
        },
        "l1_rs": {"peak": [0.9] * 5, "sharpness": [2.0] * 5, "second_peak_ratio": [0.1] * 5},
    }
    base = {
        "confidence": {
            "peak": {"global": 0.02, "per_landmark": {"dtip": 0.13, "l1_rs": 0.03, "gone": 0.5}},
            "sharpness": {"global": 1.18, "per_landmark": {"l1_rs": 1.26}},
            "second_peak_ratio": {"global": 0.95, "per_landmark": {"l1_rs": 0.83}},
            "core_landmarks": ["dtip", "l1_rs", "gone"],
            "batch": {"mode": "batch_mad", "k": 5.0, "min_batch_size": 25},
            "tiers": {"strict": {"peak": {"global": 0.1, "per_landmark": {"l1_rs": 0.2}}}},
        }
    }
    doc, report = build_gate_config(collected, ["alula_notch", "l1_rs", "l3_d"], base, min_samples=20)
    gate = doc["confidence"]

    assert report["calibrated"] == {"l3_d": 40}
    assert report["inherited_from_base"] == {"l1_rs": 5} and report["global_only"] == {"alula_notch": 0}
    # Calibrated landmark: its own percentile, not the base value (stored under the legacy name).
    assert 0.5 <= gate["peak"]["per_landmark"]["l3_d"] <= 0.52
    assert gate["tiers"]["strict"]["peak"]["per_landmark"]["l3_d"] > gate["peak"]["per_landmark"]["l3_d"]
    assert gate["tiers"]["strict"]["second_peak_ratio"]["per_landmark"]["l3_d"] < 0.4
    # Under-sampled landmark: the base value for that tier, falling back to the base active gate.
    assert gate["peak"]["per_landmark"]["l1_rs"] == 0.03
    assert gate["tiers"]["strict"]["peak"]["per_landmark"]["l1_rs"] == 0.2
    assert gate["tiers"]["strict"]["sharpness"]["per_landmark"]["l1_rs"] == 1.26
    assert "alula_notch" not in gate["peak"]["per_landmark"] and "gone" not in gate["peak"]["per_landmark"]
    # The model-wide value is the loosest per-landmark one; other base settings carry over.
    assert gate["peak"]["global"] == 0.03 and gate["second_peak_ratio"]["global"] == 0.83
    assert gate["core_landmarks"] == ["l1_rs", "l3_d"] and gate["batch"]["mode"] == "batch_mad"

    # No base model: under-sampled landmarks simply have no per-landmark entry.
    doc, report = build_gate_config(collected, ["l1_rs", "l3_d"], None, min_samples=20)
    assert list(doc["confidence"]["peak"]["per_landmark"]) == ["l3_d"] and report["global_only"] == {"l1_rs": 5}
    assert doc["confidence"]["core_landmarks"] == []


def test_scratch_model_gets_a_loadable_gate_config(tmp_path, tiny_config, monkeypatch):
    """With the sample floor lowered, a from-scratch run writes thresholds the predictor accepts."""
    monkeypatch.setattr(user_training, "MIN_CALIBRATION_SAMPLES", 1)
    monkeypatch.setattr(user_training, "GATE_TOLERANCE_PX", 1e9)  # an untrained toy model is nowhere near
    samples = [_write_sample(tmp_path, i) for i in range(4)]
    out = tmp_path / "m"
    info = train_from_samples(samples, out, n_folds=2, epochs=1, device="cpu")
    assert set(info["gate"]["calibrated"]) == {"alula_notch", "l1_rs", "l3_d"}
    text = (out / "gate_config.yaml").read_text()
    assert text.startswith("# Quality-gate thresholds") and "calibrated on this model: alula_notch (4)" in text

    from landmark_locator.inference.predict import make_predictor

    predictor = make_predictor(out, device="cpu")
    assert set(predictor.gate_config["peak"]["per_landmark"]) == {"alula_notch", "l1_rs", "l3_d"}
    assert predictor.gate_config["tiers"]["strict"]["peak"]["per_landmark"]


def test_stop_request_cancels(tmp_path, tiny_config):
    samples = [_write_sample(tmp_path, i) for i in range(4)]
    with pytest.raises(TrainingCancelled):
        train_from_samples(samples, tmp_path / "m", n_folds=1, epochs=3, device="cpu", should_stop=lambda: True)


def test_summarize_counts_and_skips(tmp_path):
    good = _write_sample(tmp_path, 0)
    legacy = _write_sample(tmp_path, 1, names=["DTip"])  # legacy name for L3.d
    empty = _write_sample(tmp_path, 2, names=[])
    summary = summarize_samples([good, legacy, empty], ["alula_notch", "l1_rs", "l3_d"])
    assert summary["counts"] == {"l3_d": 2, "l1_rs": 1, "alula_notch": 1}
    assert [p for p, _ in summary["skipped"]] == [empty[0]]
    assert len(summary["usable"]) == 2
