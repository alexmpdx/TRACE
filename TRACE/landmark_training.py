"""Find the training data for a user-trained landmark model (no Qt in here).

The landmark inspector saves each corrected image's landmarks as
``<image_dir>/manual_overrides/<stem>_landmarks_override.geojson``, in
original-image pixel coordinates. Those sidecars, paired with the images they
sit next to, are exactly the ``(annotation, image)`` pairs
``landmark_locator.training.user_training.train_from_samples`` takes.

The dialog lives in ``TRACE/landmark_training_dialog.py``.
"""

from __future__ import annotations

import os
import re
from datetime import datetime
from pathlib import Path
from typing import Iterable, Optional

from preprocessing.pipeline import IMAGE_EXTENSIONS, OVERRIDE_SUBDIR

OVERRIDE_SUFFIX = "_landmarks_override.geojson"


def user_models_dir() -> Path:
    """Default parent folder for models the user trains.

    Deliberately outside the TRACE install: the app folder is replaced on update and
    removed on uninstall, and a macOS ``.app`` bundle must not be written into.
    """
    return Path.home() / "TRACE" / "landmark_models"


def default_model_name() -> str:
    """Suggested name for a new model, e.g. ``my_landmarks_2026-11-12``."""
    return f"my_landmarks_{datetime.now():%Y-%m-%d}"


def sanitize_model_name(name: str) -> str:
    """Reduce a typed model name to something safe as a folder name on every platform."""
    cleaned = re.sub(r"[^\w.\- ]+", "_", name.strip()).strip(". ")
    return cleaned[:80]


def _image_for_stem(folder: Path, stem: str) -> Optional[Path]:
    """The image in `folder` whose stem is `stem` (first by name if several formats exist)."""
    try:
        candidates = sorted(
            f
            for f in folder.iterdir()
            if f.stem == stem and f.suffix.lower() in IMAGE_EXTENSIONS and not f.name.startswith(".") and f.is_file()
        )
    except OSError:
        return None
    return candidates[0] if candidates else None


def _samples_in_folder(folder: Path) -> list[tuple[Path, Path]]:
    """Landmark overrides belonging to images directly inside `folder`."""
    found: dict[Path, Path] = {}  # image → override
    # manual_overrides/ first: it wins over a legacy loose file for the same image.
    for override_dir in (folder / OVERRIDE_SUBDIR, folder):
        try:
            overrides = sorted(override_dir.glob(f"*{OVERRIDE_SUFFIX}"))
        except OSError:
            continue
        for override in overrides:
            if override.name.startswith("."):
                continue
            image = _image_for_stem(folder, override.name[: -len(OVERRIDE_SUFFIX)])
            if image is not None:
                found.setdefault(image, override)
    return [(override, image) for image, override in sorted(found.items())]


def find_override_samples(folders: Iterable[Path], recursive: bool = False) -> list[tuple[Path, Path]]:
    """``(override_geojson, image)`` pairs for every manually-edited image under `folders`.

    An override whose image is gone is left out. With `recursive`, subfolders are
    searched too (hidden ones and ``manual_overrides/`` itself excepted). Each image
    appears once even when the folders overlap.
    """
    seen: set[Path] = set()
    samples: list[tuple[Path, Path]] = []
    for root in folders:
        root = Path(root)
        if not root.is_dir():
            continue
        dirs = [root]
        if recursive:
            for dirpath, dirnames, _ in os.walk(root, onerror=lambda _e: None):
                dirnames[:] = sorted(d for d in dirnames if not d.startswith(".") and d != OVERRIDE_SUBDIR)
                dirs.extend(Path(dirpath) / d for d in dirnames)
        for folder in dirs:
            for override, image in _samples_in_folder(folder):
                key = image.resolve()
                if key not in seen:
                    seen.add(key)
                    samples.append((override, image))
    return samples


def is_landmark_model_dir(path: Path) -> bool:
    """True when `path` is a folder TRACE can load as a landmark model."""
    path = Path(path)
    return path.is_dir() and any(path.glob("best_fold*.pt"))


def list_user_models(parents: Iterable[Path]) -> list[Path]:
    """Landmark-model folders directly inside any of `parents`, newest first."""
    models: dict[Path, float] = {}
    for parent in parents:
        parent = Path(parent)
        try:
            children = [c for c in parent.iterdir() if is_landmark_model_dir(c)]
        except OSError:
            continue
        for child in children:
            try:
                models.setdefault(child.resolve(), child.stat().st_mtime)
            except OSError:
                continue
    return sorted(models, key=lambda p: models[p], reverse=True)
