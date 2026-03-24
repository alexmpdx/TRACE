#!/usr/bin/env python3
"""Add a wing annotation to geojson files by computing the geometric union of all polygons."""

import glob
import json
import os
import sys

from shapely.geometry import mapping, shape
from shapely.ops import unary_union


def add_wing(filepath):
    with open(filepath) as f:
        data = json.load(f)

    # Skip if wing already exists
    for feat in data["features"]:
        if feat.get("properties", {}).get("class") == "wing":
            print(f"Skipped (wing exists): {filepath}")
            return

    # Build shapely geometries from all features
    polys = []
    for feat in data["features"]:
        geom = feat.get("geometry")
        if geom:
            polys.append(shape(geom))

    if not polys:
        print(f"Skipped (no polygons): {filepath}")
        return

    union = unary_union(polys)

    wing_feature = {
        "type": "Feature",
        "geometry": mapping(union),
        "properties": {
            "class": "wing",
            "class_index": 1,
            "color": "#00FF00",
        },
    }

    data["features"].insert(0, wing_feature)

    with open(filepath, "w") as f:
        json.dump(data, f, indent=2)

    print(f"Added wing {union.geom_type} (from {len(polys)} polygons): {filepath}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python add_wing.py <file_or_folder> [file_or_folder ...]")
        sys.exit(1)

    for path in sys.argv[1:]:
        if os.path.isdir(path):
            for f in sorted(glob.glob(os.path.join(path, "*.geojson"))):
                add_wing(f)
        elif os.path.isfile(path):
            add_wing(path)
        else:
            print(f"Not found: {path}")
