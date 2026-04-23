"""File discovery: scan folders for TIFF+GeoJSON pairs and detect GT files."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


@dataclass
class FilePair:
    """A matched image + annotation pair with optional ground-truth files."""

    image_path: Path
    geojson_path: Path
    gt_intervein_path: Optional[Path] = None
    gt_skeleton_path: Optional[Path] = None

    @property
    def display_name(self) -> str:
        """Short name for display in the UI."""
        return self.image_path.stem


def discover_file_pairs(folder: Path) -> list[FilePair]:
    """Scan a folder (recursively) for TIFF+GeoJSON pairs.

    Matching strategy:
    1. Direct match: same stem (e.g., wing1.tif + wing1.geojson)
    2. Subfolder match: folder contains exactly one .tif and one .geojson
    3. Ground-truth detection: *_expected_intervein_overlay.geojson and
       *_expected_skeleton_overlay.geojson

    Files named *_expected_*_overlay.geojson are excluded from primary pairs.
    """
    pairs: list[FilePair] = []

    # Collect all image and geojson files
    image_exts = {".tif", ".tiff", ".psd", ".psb"}
    images: list[Path] = []
    geojsons: list[Path] = []

    for p in sorted(folder.rglob("*")):
        if not p.is_file():
            continue
        if p.suffix.lower() in image_exts:
            images.append(p)
        elif p.suffix.lower() == ".geojson":
            # Skip ground-truth overlay and landmark files from primary matching
            if "_expected_" in p.stem and "_overlay" in p.stem:
                continue
            if p.stem.endswith("_landmarks"):
                continue
            geojsons.append(p)

    # Build a lookup for geojsons by directory
    geojson_by_dir: dict[Path, list[Path]] = {}
    for gj in geojsons:
        geojson_by_dir.setdefault(gj.parent, []).append(gj)

    matched_geojsons: set[Path] = set()

    for img in images:
        img_dir = img.parent
        img_stem = img.stem

        # Strategy 1: exact stem match in same directory
        matched_gj: Optional[Path] = None
        for gj in geojson_by_dir.get(img_dir, []):
            if gj.stem == img_stem:
                matched_gj = gj
                break

        # Strategy 2: only one geojson in the directory
        if matched_gj is None:
            dir_gjs = [gj for gj in geojson_by_dir.get(img_dir, []) if gj not in matched_geojsons]
            if len(dir_gjs) == 1:
                matched_gj = dir_gjs[0]

        if matched_gj is None:
            continue

        matched_geojsons.add(matched_gj)

        # Detect ground-truth files
        gt_intervein = _find_gt_file(matched_gj, "intervein")
        gt_skeleton = _find_gt_file(matched_gj, "skeleton")

        pairs.append(
            FilePair(
                image_path=img,
                geojson_path=matched_gj,
                gt_intervein_path=gt_intervein,
                gt_skeleton_path=gt_skeleton,
            )
        )

    return pairs


def _find_gt_file(geojson_path: Path, kind: str) -> Optional[Path]:
    """Look for a ground-truth overlay file next to the geojson."""
    stem = geojson_path.stem
    gt_name = f"{stem}_expected_{kind}_overlay.geojson"
    gt_path = geojson_path.parent / gt_name
    return gt_path if gt_path.exists() else None
