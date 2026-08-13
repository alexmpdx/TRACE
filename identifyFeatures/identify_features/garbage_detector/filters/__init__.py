"""Concrete garbage-detector filters.

Importing this package imports every filter module so each ``@register_filter`` side
effect fires and the registry is populated.
"""

from identify_features.garbage_detector.filters import fragmentation  # noqa: F401
from identify_features.garbage_detector.filters import intervein_association  # noqa: F401
from identify_features.garbage_detector.filters import solidity  # noqa: F401
from identify_features.garbage_detector.filters import vein_association  # noqa: F401
from identify_features.garbage_detector.filters import vein_presence  # noqa: F401

__all__ = ["solidity", "fragmentation", "vein_association", "intervein_association", "vein_presence"]
