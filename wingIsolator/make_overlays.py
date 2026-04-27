"""Generate side-by-side QA overlays: input polygons (red) + chosen main wing (green)."""

import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from shapely.geometry import MultiPolygon, shape  # noqa: E402

from wingIsolator import load_image, load_wing_polygons  # noqa: E402


def _polys_to_xy(poly):
    polys = poly.geoms if isinstance(poly, MultiPolygon) else [poly]
    return [list(p.exterior.coords) for p in polys if not p.is_empty]


def make_overlay(image_path, det_geojson, main_geojson, out_path, input_color=(220, 60, 60), main_color=(60, 220, 60)):
    img = load_image(image_path)
    pil = Image.fromarray(img).convert("RGBA")
    overlay = Image.new("RGBA", pil.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    input_polys = load_wing_polygons(det_geojson)
    for p in input_polys:
        for ring in _polys_to_xy(p):
            draw.line([(x, y) for x, y in ring], fill=input_color + (200,), width=8)

    with open(main_geojson) as f:
        fc = json.load(f)
    for feat in fc["features"]:
        shp = shape(feat["geometry"])
        for ring in _polys_to_xy(shp):
            draw.line([(x, y) for x, y in ring], fill=main_color + (220,), width=10)

    h, w = img.shape[:2]
    cx, cy = w / 2, h / 2
    r = 30
    draw.ellipse([cx - r, cy - r, cx + r, cy + r], outline=(255, 255, 0, 255), width=6)

    composite = Image.alpha_composite(pil, overlay).convert("RGB")
    composite.thumbnail((1600, 1600))
    composite.save(out_path, quality=85)


if __name__ == "__main__":
    base = Path(__file__).resolve().parent
    img_dir = base / "testpics"
    det_dir = base / "testpics_geojsons"
    iso_dir = base / "testpics_isolated"
    out_dir = base / "testpics_overlays"
    out_dir.mkdir(exist_ok=True)

    for img_path in sorted(img_dir.iterdir()):
        if img_path.suffix.lower() not in {".tif", ".tiff", ".bmp", ".png", ".jpg"}:
            continue
        stem = img_path.stem
        det = det_dir / f"{stem}_detections.geojson"
        main = iso_dir / f"{stem}_main_wing.geojson"
        if not det.exists() or not main.exists():
            print(f"skip {stem}: missing files")
            continue
        out = out_dir / f"{stem}_qa.jpg"
        make_overlay(img_path, det, main, out)
        print(f"  wrote {out.name}")
