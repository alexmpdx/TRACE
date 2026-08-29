"""Regression tests for retiring ``SkeletonMethod.BOUNDARY_SMOOTH``.

Boundary smoothing was historically a ``SkeletonMethod`` enum member even
though it is mask preprocessing that composes with a skeletonizer rather than
an alternative to one (the dispatch in ``skeleton._build_skeleton_core``
applied it as a separate step, then still picked a skeletonizer). It now lives
on ``PipelineConfig`` as the ``enable_boundary_smooth`` flag.

The hazard is stored configs: exported settings JSON, saved QSettings, and
preset files written before the change still carry ``"boundary-smooth"`` in
``skeleton_methods``, where ``SkeletonMethod("boundary-smooth")`` now raises
ValueError. Both JSON entry points — ``config_io.config_from_dict`` and
``presets_loader._convert_enum_lists`` — must translate it to the flag instead
of blowing up, or a user's saved settings become unloadable on upgrade.
"""

import sys
from pathlib import Path

# Sibling modules export bare package names that don't match their directory,
# so importing TRACE.config_io needs the same sys.path set run_cli.py builds.
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

import pytest  # noqa: E402
from identify_features.config import PipelineConfig  # noqa: E402
from identify_features.models.datatypes import SkeletonMethod  # noqa: E402

from TRACE.config_io import config_from_dict, config_to_dict  # noqa: E402
from TRACE.presets_loader import _convert_enum_lists  # noqa: E402


def test_boundary_smooth_is_not_a_skeleton_method():
    """It's a modifier, not a choice among skeletonizers."""
    assert "boundary-smooth" not in {m.value for m in SkeletonMethod}
    with pytest.raises(ValueError):
        SkeletonMethod("boundary-smooth")


def test_flag_defaults_off():
    """Default config skeletonizes the raw mask — unchanged from before."""
    assert PipelineConfig().enable_boundary_smooth is False


def test_legacy_config_migrates_to_flag():
    """A pre-change settings JSON loads, with the method translated."""
    cfg = config_from_dict({"skeleton_methods": ["ridge", "boundary-smooth"]})
    assert cfg.enable_boundary_smooth is True
    assert cfg.skeleton_methods == [SkeletonMethod.RIDGE]


def test_legacy_preset_migrates_to_flag():
    """The preset loader converts enums on its own path and needs the shim too."""
    out = _convert_enum_lists({"skeleton_methods": ["boundary-smooth", "medial-axis"]})
    assert out["enable_boundary_smooth"] is True
    assert out["skeleton_methods"] == [SkeletonMethod.MEDIAL_AXIS]


def test_explicit_flag_wins_over_legacy_method():
    """A current-build config that somehow carries both keeps its own value."""
    cfg = config_from_dict({"skeleton_methods": ["ridge", "boundary-smooth"], "enable_boundary_smooth": False})
    assert cfg.enable_boundary_smooth is False


def test_roundtrip_is_stable():
    """Export -> import preserves the flag without reintroducing the enum value."""
    cfg = PipelineConfig(enable_boundary_smooth=True)
    data = config_to_dict(cfg)
    assert "boundary-smooth" not in data["skeleton_methods"]
    assert config_from_dict(data).enable_boundary_smooth is True
