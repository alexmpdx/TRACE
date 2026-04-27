"""Backward-compat shim. Real code lives in wingIsolator.pipeline / .cli.

Older callers may import this module directly via `wing_isolator.py`. New
callers should import from the package::

    from wingIsolator import isolate_main_wing, IsolationResult, isolate_in_memory
"""

import sys
from pathlib import Path

_pkg_parent = str(Path(__file__).resolve().parent.parent)
if _pkg_parent not in sys.path:
    sys.path.insert(0, _pkg_parent)

from wingIsolator.cli import main  # noqa: E402,F401
from wingIsolator.pipeline import (  # noqa: E402,F401
    DEFAULT_OUTPUT_SUFFIX,
    SUPPORTED_IMAGE_EXTS,
    WING_CLASS_NAMES,
    IsolationResult,
    apply_mask_to_image,
    build_geojson,
    discover_images,
    find_geojson_for_image,
    isolate_folder,
    isolate_in_memory,
    isolate_main_wing,
    load_image,
    load_wing_polygons,
    rasterize_polygon,
    select_main_label,
    select_main_polygon,
    split_merged_wing,
    vectorize_mask,
    write_masked_image,
)

if __name__ == "__main__":
    sys.exit(main())
