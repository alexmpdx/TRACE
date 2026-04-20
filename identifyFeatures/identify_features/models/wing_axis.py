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

    # Orient the AP vector using an anterior reference landmark so it always
    # points posterior regardless of wing chirality. The raw 90° rotation
    # (-dy, dx) only lands on "posterior" for one orientation; flipping
    # when the anterior landmark projects positively handles the mirror case.
    candidate = (-unit[1], unit[0])
    ap_unit: Optional[tuple[float, float]] = None
    for ref_name in ("subcostal break", "L1-Rs"):
        ref = landmarks.get(ref_name)
        if ref is None:
            continue
        rx = ref.x - alula.x
        ry = ref.y - alula.y
        proj = rx * candidate[0] + ry * candidate[1]
        if proj > 0:
            ap_unit = (-candidate[0], -candidate[1])
        elif proj < 0:
            ap_unit = candidate
        if ap_unit is not None:
            logger.info(
                "Wing AP axis oriented via %s (proj=%.0f): ap_vector=(%.2f, %.2f)",
                ref_name,
                proj,
                ap_unit[0],
                ap_unit[1],
            )
            break

    axis = WingAxis(
        proximal_point=alula.point,
        distal_point=dtip.point,
        unit_vector=unit,
        length=length,
        ap_unit_vector=ap_unit,
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
