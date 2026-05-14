#!/usr/bin/env python3
"""Standalone CLI for resolutionAdjust.

Rescales one image or a folder of images toward a target µm/px. Useful for
quickly inspecting how resolutionAdjust would treat a batch before wiring it
into TRACE.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

_project_root = str(Path(__file__).resolve().parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

_self_dir = str(Path(__file__).resolve().parent)
if _self_dir not in sys.path:
    sys.path.insert(0, _self_dir)

from resolution_adjust import adjust_resolution  # noqa: E402

logger = logging.getLogger("resolutionAdjust")


_IMAGE_EXTS = {".tif", ".tiff", ".png", ".jpg", ".jpeg", ".bmp"}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Rescale images toward a target µm/px when the input is outside a tolerance band."
    )
    parser.add_argument("input", type=Path, help="Image file or folder of images.")
    parser.add_argument("--output-dir", "-o", required=True, type=Path, help="Output directory.")
    parser.add_argument(
        "--input-um-per-px",
        required=True,
        type=float,
        help="µm/px of the input image(s). Single value applied to every image.",
    )
    parser.add_argument(
        "--target-um-per-px",
        required=True,
        type=float,
        help="Target µm/px to rescale toward.",
    )
    parser.add_argument(
        "--tolerance-low",
        type=float,
        default=0.85,
        help="Lower bound of the pass-through ratio band (default 0.85).",
    )
    parser.add_argument(
        "--tolerance-high",
        type=float,
        default=1.15,
        help="Upper bound of the pass-through ratio band (default 1.15).",
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    if args.input.is_file():
        images = [args.input]
    elif args.input.is_dir():
        images = sorted(p for p in args.input.iterdir() if p.is_file() and p.suffix.lower() in _IMAGE_EXTS)
    else:
        logger.error("input not found: %s", args.input)
        return 1

    if not images:
        logger.error("no images to process")
        return 1

    args.output_dir.mkdir(parents=True, exist_ok=True)
    n_rescaled = 0
    n_passthrough = 0
    for img in images:
        try:
            res = adjust_resolution(
                image_path=img,
                input_um_per_px=args.input_um_per_px,
                target_um_per_px=args.target_um_per_px,
                output_dir=args.output_dir,
                tolerance_low=args.tolerance_low,
                tolerance_high=args.tolerance_high,
            )
        except Exception as exc:  # noqa: BLE001
            logger.exception("failed on %s: %s", img.name, exc)
            continue
        if res.rescaled:
            n_rescaled += 1
        else:
            n_passthrough += 1

    logger.info("done: %d rescaled, %d pass-through", n_rescaled, n_passthrough)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
