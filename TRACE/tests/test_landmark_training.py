"""User-trained landmark models: finding the manual edits, and the dialog end to end."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pytest

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

from preprocessing.pipeline import landmarks_override_path  # noqa: E402
from TRACE.landmark_training import (  # noqa: E402
    find_override_samples,
    list_user_models,
    sanitize_model_name,
)


def _write_override(path: Path, names=("L3.d", "L1-Rs")) -> None:
    features = [
        {
            "type": "Feature",
            "geometry": {"type": "Point", "coordinates": [20.0 + 25 * k, 30.0]},
            "properties": {"classification": {"name": name}, "reliable": True, "confidence": 1.0},
        }
        for k, name in enumerate(names)
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"type": "FeatureCollection", "features": features}), encoding="utf-8")


def _write_image(path: Path, seed: int = 0) -> Path:
    import cv2
    import numpy as np

    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), np.random.default_rng(seed).integers(0, 255, (88, 128, 3), dtype=np.uint8))
    return path


def test_find_override_samples(tmp_path):
    a = _write_image(tmp_path / "a.png")
    b = _write_image(tmp_path / "b.ome.tif")
    c = _write_image(tmp_path / "sub" / "c.png")
    _write_image(tmp_path / "unedited.png")
    # Written through the inspector's own path helper, so the two sides cannot drift.
    for image in (a, b, c):
        _write_override(landmarks_override_path(image))
    # Legacy loose override next to its image; an override whose image is gone.
    d = _write_image(tmp_path / "d.png")
    _write_override(tmp_path / "d_landmarks_override.geojson")
    _write_override(tmp_path / "manual_overrides" / "deleted_landmarks_override.geojson")
    # A segmentation override must never be mistaken for a landmark one.
    _write_override(tmp_path / "manual_overrides" / "a_segmentation_override.geojson")

    flat = find_override_samples([tmp_path])
    assert [image.name for _, image in flat] == ["a.png", "b.ome.tif", "d.png"]
    assert flat[0][0] == landmarks_override_path(a)
    assert flat[2][0] == tmp_path / "d_landmarks_override.geojson"

    deep = find_override_samples([tmp_path, tmp_path / "sub"], recursive=True)  # overlapping folders
    assert sorted(image.name for _, image in deep) == ["a.png", "b.ome.tif", "c.png", "d.png"]
    assert d in [image for _, image in deep]


def test_model_helpers(tmp_path):
    assert sanitize_model_name("  my model: v2/final. ") == "my model_ v2_final"
    assert sanitize_model_name("///") == "_"
    (tmp_path / "m1").mkdir()
    (tmp_path / "m1" / "best_fold0.pt").write_bytes(b"")
    (tmp_path / "not_a_model").mkdir()
    assert list_user_models([tmp_path, tmp_path, tmp_path / "missing"]) == [(tmp_path / "m1").resolve()]


def test_dialog_trains_and_switches_model(tmp_path, monkeypatch):
    """Scratch-train through the real dialog + thread on tiny images, then adopt the model."""
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    pytest.importorskip("albumentations")
    from PyQt5.QtCore import QEventLoop, QTimer
    from PyQt5.QtWidgets import QApplication, QCheckBox, QLineEdit, QWidget

    from landmark_locator.training import user_training
    from TRACE.landmark_training_dialog import LandmarkTrainingDialog

    cfg = json.loads(json.dumps(user_training.FALLBACK_CONFIG))
    cfg["input"] = {"width": 128, "height": 96}
    cfg["model"]["pretrained"] = False
    cfg["training"]["batch_size"] = 2
    monkeypatch.setattr(user_training, "_scratch_template", lambda: json.loads(json.dumps(cfg)))

    images = tmp_path / "images"
    for i in range(4):
        _write_override(landmarks_override_path(_write_image(images / f"wing_{i}.png", seed=i)))

    app = QApplication.instance() or QApplication([])

    class _Window(QWidget):
        """The slice of TraceWindow the dialog touches."""

        def __init__(self):
            super().__init__()
            self.input_edit = QLineEdit(str(images))
            self.recursive_chk = QCheckBox()
            self._landmark_model_path = ""  # no current model → scratch is the only mode
            self.settings = None
            self.used = None

        def use_landmark_model(self, model_dir):
            self.used = Path(model_dir)

    window = _Window()
    dlg = LandmarkTrainingDialog(window)
    assert len(dlg._samples) == 4 and dlg.btn_start.isEnabled()
    assert dlg.cmb_mode.currentData() == "scratch"

    dlg.edit_save_dir.setText(str(tmp_path / "models"))
    dlg.edit_name.setText("test model")
    dlg.spin_epochs.setValue(1)
    dlg.spin_folds.setValue(2)
    dlg._on_start()
    assert not dlg.btn_start.isEnabled() and dlg.btn_stop.isEnabled()

    loop = QEventLoop()
    poll = QTimer()
    poll.timeout.connect(lambda: dlg._thread is None and loop.quit())
    poll.start(50)
    QTimer.singleShot(120_000, loop.quit)
    loop.exec_()
    poll.stop()

    out = tmp_path / "models" / "test model"
    assert dlg._thread is None, dlg.txt_log.toPlainText()
    assert sorted(p.name for p in out.glob("*.pt")) == ["best_fold0.pt", "best_fold1.pt"], dlg.txt_log.toPlainText()
    assert (out / "training_info.json").is_file() and (out / "training.log").is_file()
    assert dlg.btn_use.isEnabled() and dlg.btn_start.isEnabled()

    dlg._on_use_model()
    assert window.used == out and not dlg.btn_use.isEnabled()
    dlg.reject()
    app.processEvents()


def test_mcp_tools_drive_the_dialog(tmp_path, monkeypatch):
    """The MCP tool bodies open the real dialog, train, report status, switch the model and close."""
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    pytest.importorskip("albumentations")
    from PyQt5.QtCore import QEventLoop, QTimer
    from PyQt5.QtWidgets import QApplication, QCheckBox, QLineEdit, QWidget

    from landmark_locator.training import user_training
    from TRACE import mcp_server as ms
    from TRACE.landmark_training_dialog import LandmarkTrainingDialog

    cfg = json.loads(json.dumps(user_training.FALLBACK_CONFIG))
    cfg["input"] = {"width": 128, "height": 96}
    cfg["model"]["pretrained"] = False
    cfg["training"]["batch_size"] = 2
    monkeypatch.setattr(user_training, "_scratch_template", lambda: json.loads(json.dumps(cfg)))

    images = tmp_path / "images"
    for i in range(4):
        _write_override(landmarks_override_path(_write_image(images / f"wing_{i}.png", seed=i)))

    app = QApplication.instance() or QApplication([])

    class _Window(QWidget):
        def __init__(self):
            super().__init__()
            self.worker = None
            self._is_paused = False
            self.input_edit = QLineEdit(str(images))
            self.recursive_chk = QCheckBox()
            self._landmark_model_path = ""
            self.settings = None
            self._landmark_training_dialog = None

        def _open_landmark_training_dialog(self, block=True):
            # Same contract as TraceWindow's: one dialog, remembered until it closes.
            if self._landmark_training_dialog is None:
                dlg = LandmarkTrainingDialog(self)
                self._landmark_training_dialog = dlg
                dlg.finished.connect(lambda _r: setattr(self, "_landmark_training_dialog", None))
                dlg.open()
            return self._landmark_training_dialog

        def use_landmark_model(self, model_dir):
            self._landmark_model_path = str(model_dir)

    window = _Window()
    assert ms._tool_list_landmark_edits(window)["count"] == 4
    with pytest.raises(ms.GuiToolError):
        ms._tool_list_landmark_edits(window, folders=[str(tmp_path / "nope")])
    assert ms._training_status(window) == {"dialog_open": False, "state": "idle"}
    with pytest.raises(ms.GuiToolError):
        ms._tool_use_trained_landmark_model(window)

    # Fine-tuning is refused without a current model, and the refusal is an error the
    # agent can read rather than a message box that would hang the call.
    with pytest.raises(ms.GuiToolError, match="no landmark model"):
        ms._tool_start_landmark_training(window, mode="finetune")
    with pytest.raises(ms.GuiToolError, match="Not enough"):
        ms._tool_start_landmark_training(window, folders=[str(tmp_path)], mode="scratch")

    started = ms._tool_start_landmark_training(
        window,
        folders=[str(images)],
        mode="scratch",
        name="mcp model",
        save_dir=str(tmp_path / "models"),
        epochs=1,
        n_models=2,
    )
    assert started["started"] and started["state"] == "running" and started["n_edited_images"] == 4
    assert ms._tool_start_run(window)["started"] is False  # no pipeline run while the dialog is open
    with pytest.raises(ms.GuiToolError):
        ms._tool_close_landmark_training(window)

    loop = QEventLoop()
    poll = QTimer()
    poll.timeout.connect(lambda: ms._training_status(window)["state"] != "running" and loop.quit())
    poll.start(50)
    QTimer.singleShot(120_000, loop.quit)
    loop.exec_()
    poll.stop()

    status = ms._training_status(window)
    out = tmp_path / "models" / "mcp model"
    assert status["state"] == "finished", status
    assert status["trained_model"] == str(out) and status["final_error_px"] is not None and not status["in_use"]
    assert len(status["folds"]) == 2 and status["log_tail"]

    assert ms._tool_use_trained_landmark_model(window) == {"landmark_model": str(out), "in_use": True}
    assert ms._tool_close_landmark_training(window) == {"closed": True}
    app.processEvents()
    assert window._landmark_training_dialog is None
