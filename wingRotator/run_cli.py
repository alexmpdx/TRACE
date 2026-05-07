#!/usr/bin/env python3
"""Standalone CLI for wingRotator.

Rotates a single image (or all images in a folder) plus its associated landmarks
GeoJSON (and optional extra GeoJSONs — wing-isolation masks, segmentation
outputs, ground-truth overlays) so the wing is in canonical orientation.
"""

import argparse
import logging
import sys
from pathlib import Path

# Add sibling dirs to sys.path so this can be invoked directly.
_project_root = str(Path(__file__).resolve().parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

_self_dir = str(Path(__file__).resolve().parent)
if _self_dir not in sys.path:
    sys.path.insert(0, _self_dir)

from wing_rotator import rotate_from_landmarks  # noqa: E402

logger = logging.getLogger("wing_rotator")


_IMAGE_EXTS = {".tif", ".tiff", ".png", ".jpg", ".jpeg", ".bmp"}


def _discover_pairs(image: Path, landmarks: Path) -> list[tuple[Path, Path]]:
    """Resolve --image / --landmarks into (image, landmarks) pairs.

    Both file → single pair. Both directories → match by stem with `_landmarks`
    suffix. One file + one dir → look up the matching partner by stem.
    """
    if image.is_file() and landmarks.is_file():
        return [(image, landmarks)]

    if image.is_dir() and landmarks.is_dir():
        pairs: list[tuple[Path, Path]] = []
        for img in sorted(image.iterdir()):
            if img.suffix.lower() not in _IMAGE_EXTS:
                continue
            stem = img.stem
            cand = landmarks / f"{stem}_landmarks.geojson"
            if cand.exists():
                pairs.append((img, cand))
            else:
                logger.warning("no landmarks geojson for %s (looked for %s)", img.name, cand)
        return pairs

    if image.is_file() and landmarks.is_dir():
        cand = landmarks / f"{image.stem}_landmarks.geojson"
        if cand.exists():
            return [(image, cand)]
        raise FileNotFoundError(f"no landmarks geojson for {image} in {landmarks}")

    if image.is_dir() and landmarks.is_file():
        # Ambiguous — pick the one matching the landmarks stem.
        target_stem = landmarks.stem.replace("_landmarks", "")
        for img in image.iterdir():
            if img.suffix.lower() in _IMAGE_EXTS and img.stem == target_stem:
                return [(img, landmarks)]
        raise FileNotFoundError(f"no image in {image} matching {landmarks.stem}")

    raise FileNotFoundError("--image and --landmarks must be valid files or directories")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Rotate wing images + GeoJSONs to canonical orientation using " "landmark Procrustes alignment."
    )
    parser.add_argument("--image", required=True, type=Path, help="Image file or directory")
    parser.add_argument(
        "--landmarks",
        required=True,
        type=Path,
        help="Landmarks GeoJSON file or directory of *_landmarks.geojson files",
    )
    parser.add_argument("--output-dir", "-o", required=True, type=Path, help="Output directory")
    parser.add_argument(
        "--extra-geojson",
        action="append",
        default=[],
        type=Path,
        help="Additional GeoJSON to rotate with the same affine. Repeat for several. "
        "When --image is a directory, each path is treated as a sibling per-image "
        "GeoJSON: '<image_stem>_<extra_stem>.geojson' is searched in the directory of "
        "each extra path.",
    )
    parser.add_argument(
        "--soft-reliability",
        action="store_true",
        help="Include landmarks flagged reliable=false at reduced weight (default: drop them).",
    )
    parser.add_argument(
        "--min-landmarks",
        type=int,
        default=2,
        help="Minimum usable landmarks required to fit (default: 2).",
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    pairs = _discover_pairs(args.image, args.landmarks)
    if not pairs:
        logger.error("no image+landmarks pairs found")
        return 1

    args.output_dir.mkdir(parents=True, exist_ok=True)
    n_ok = 0
    n_skipped = 0
    for img_path, lm_path in pairs:
        # Resolve per-image extras: if --extra-geojson points at a file in a dir
        # alongside other per-image GeoJSONs, find the one whose stem starts with
        # this image's stem.
        per_image_extras: list[Path] = []
        for ex in args.extra_geojson:
            if ex.is_file() and len(pairs) == 1:
                per_image_extras.append(ex)
            elif ex.is_dir():
                for cand in ex.iterdir():
                    if cand.suffix.lower() == ".geojson" and cand.stem.startswith(img_path.stem):
                        per_image_extras.append(cand)
            else:
                # Treat as a literal path; user knows what they're doing.
                if ex.exists():
                    per_image_extras.append(ex)

        try:
            result = rotate_from_landmarks(
                image_path=img_path,
                landmarks_geojson_path=lm_path,
                output_dir=args.output_dir,
                extra_geojsons=per_image_extras,
                soft_reliability=args.soft_reliability,
                min_landmarks=args.min_landmarks,
            )
        except Exception as exc:  # noqa: BLE001
            logger.exception("failed to rotate %s: %s", img_path.name, exc)
            continue

        if result is None:
            logger.warning("skipped %s (insufficient reliable landmarks)", img_path.name)
            n_skipped += 1
            continue

        logger.info(
            "%s: angle=%.2f° n_landmarks=%d residual=%.1f mirror=%s -> %s",
            img_path.name,
            result.angle_deg,
            result.n_landmarks_used,
            result.rms_residual,
            result.mirrored_detected,
            result.rotated_image_path.name,
        )
        n_ok += 1

    logger.info("done: %d rotated, %d skipped", n_ok, n_skipped)
    return 0 if n_ok > 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
