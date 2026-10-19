"""A checkpoint and gate sidecar written before the DTip -> L3.d rename still load correctly."""

import torch
import yaml
from landmark_locator.inference.predict import LandmarkPredictor
from landmark_locator.models.unet import LandmarkUNet


def _write_legacy_model(model_dir):
    """A tiny 3-landmark checkpoint + sidecar that use the old "dtip" name."""
    order = ["alula_notch", "dtip", "l1_rs"]
    model = LandmarkUNet(num_landmarks=len(order), pretrained=False)
    cfg = {
        "input": {"width": 64, "height": 64},
        "heatmap": {
            "landmark_order": order,
            "num_landmarks": len(order),
            "geojson_to_landmark": {"alula notch": "alula_notch", "DTip": "dtip", "L1-Rs": "l1_rs"},
        },
    }
    ckpt = model_dir / "best_fold0.pt"
    torch.save({"config": cfg, "model_state_dict": model.state_dict()}, ckpt)
    sidecar = {
        "confidence": {
            "peak": {"global": 0.1, "per_landmark": {"alula_notch": 0.2, "dtip": 0.7, "l1_rs": 0.3}},
            "core_landmarks": ["alula_notch", "dtip"],
            "tiers": {"strict": {"peak": {"per_landmark": {"dtip": 0.9}}}},
        }
    }
    (model_dir / "gate_config.yaml").write_text(yaml.safe_dump(sidecar))
    return ckpt


def test_channel_order_is_renamed_in_place(tmp_path):
    predictor = LandmarkPredictor(_write_legacy_model(tmp_path), device="cpu")
    # Channel 1 was "dtip"; it must stay channel 1, now named "l3_d" (no re-sort).
    assert predictor.landmark_order == ["alula_notch", "l3_d", "l1_rs"]
    assert predictor.geojson_to_landmark == {"alula notch": "alula_notch", "L3.d": "l3_d", "L1-Rs": "l1_rs"}


def test_legacy_sidecar_thresholds_follow_the_renamed_landmark(tmp_path):
    predictor = LandmarkPredictor(_write_legacy_model(tmp_path), device="cpu")
    gate = predictor.gate_config
    assert gate["peak"]["per_landmark"] == {"alula_notch": 0.2, "l3_d": 0.7, "l1_rs": 0.3}
    assert gate["core_landmarks"] == ["alula_notch", "l3_d"]
    assert gate["tiers"]["strict"]["peak"]["per_landmark"] == {"l3_d": 0.9}


def test_legacy_runtime_override_is_renamed(tmp_path):
    predictor = LandmarkPredictor(_write_legacy_model(tmp_path), device="cpu")
    predictor.update_gate_config({"peak": {"per_landmark": {"dtip": 0.05}}})
    assert predictor.gate_config["peak"]["per_landmark"]["l3_d"] == 0.05
    assert "dtip" not in predictor.gate_config["peak"]["per_landmark"]
