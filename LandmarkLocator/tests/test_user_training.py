"""Training from explicit (annotation, image) pairs: scratch, then fine-tune from that model."""

import json
from pathlib import Path

import cv2
import numpy as np
import pytest

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
    assert (tuned_dir / "gate_config.yaml").read_text() == (scratch_dir / "gate_config.yaml").read_text()
    assert json.loads((tuned_dir / "training_info.json").read_text())["n_images"] == 5

    # The result loads through the same factory TRACE uses.
    from landmark_locator.inference.predict import make_predictor

    predictor = make_predictor(tuned_dir, device="cpu")
    assert predictor.landmark_order == ["alula_notch", "l1_rs", "l3_d"]
    assert predictor.gate_config["peak"]["global"] == 0.2


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
