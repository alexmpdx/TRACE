"""wingIsolator — pick the centered wing from a multi-wing detection.

Public API::

    from wingIsolator import (
        IsolationResult,
        isolate_main_wing,    # file-based, returns IsolationResult
        isolate_folder,       # batch over two directories
        isolate_in_memory,    # numpy + shapely in, mask + polygon out
    )
"""

from wingIsolator.pipeline import (
    IsolationResult,
    discover_images,
    find_geojson_for_image,
    isolate_folder,
    isolate_in_memory,
    isolate_main_wing,
    load_image,
    load_wing_polygons,
)

__all__ = [
    "IsolationResult",
    "discover_images",
    "find_geojson_for_image",
    "isolate_folder",
    "isolate_in_memory",
    "isolate_main_wing",
    "load_image",
    "load_wing_polygons",
]
