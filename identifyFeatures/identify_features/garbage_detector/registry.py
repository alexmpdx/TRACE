"""Filter registry — the single place that knows which filters run at which stage.

Filters self-register with the :func:`register_filter` decorator at import time. The
``filters`` sub-package imports every filter module so those side-effects fire (see
``filters/__init__.py``).
"""

from __future__ import annotations

from typing import Callable, TypeVar

from identify_features.garbage_detector.base import FilterStage, GarbageFilter

# stage -> filters registered for that stage, in registration order.
REGISTRY: dict[FilterStage, list[GarbageFilter]] = {stage: [] for stage in FilterStage}

_T = TypeVar("_T", bound=type)


def register_filter(cls: _T) -> _T:
    """Class decorator: instantiate the filter and add it to the registry.

    The decorated class must expose ``name`` and ``stage`` (per the ``GarbageFilter``
    protocol) and be constructible with no arguments.
    """
    instance = cls()  # type: ignore[call-arg]
    REGISTRY.setdefault(instance.stage, []).append(instance)
    return cls
