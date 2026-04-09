"""Wing-level proximal/distal axis derivation from landmarks."""

from __future__ import annotations

import logging
import math
from typing import Optional

from identify_features.models.datatypes import Landmark, WingAxis

logger = logging.getLogger(__name__)


def compute_wing_axis(landmarks: dict[str, Landmark]) -> Optional[WingAxis]:
    """Build the PD axis from alula notch → DTip.

    Returns None if either anchor landmark is missing or the axis length is
    zero. Callers should degrade gracefully (fall back to the merge behavior
    used when no axis is available).
    """
    alula = landmarks.get("alula notch")
    dtip = landmarks.get("DTip")
    if alula is None or dtip is None:
        logger.info("Wing axis not computed: missing alula notch or DTip")
        return None

    dx = dtip.x - alula.x
    dy = dtip.y - alula.y
    length = math.hypot(dx, dy)
    if length <= 0:
        logger.warning("Wing axis length is zero")
        return None

    unit = (dx / length, dy / length)
    axis = WingAxis(
        proximal_point=alula.point,
        distal_point=dtip.point,
        unit_vector=unit,
        length=length,
    )
    logger.info(
        "Wing PD axis: alula (%.0f, %.0f) → DTip (%.0f, %.0f), length %.0fpx",
        alula.x,
        alula.y,
        dtip.x,
        dtip.y,
        length,
    )
    return axis
