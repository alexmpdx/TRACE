"""Regenerate ``trace_icon.icns`` (the macOS app icon) from LogoThick_light.svg.

macOS app icons are not full-bleed like Windows .ico files: the artwork sits
on a rounded-square plate that fills ~80% of a 1024 px canvas (Apple's icon
grid), leaving a transparent margin so the icon lines up with its neighbours
in the Dock and Finder. The TRACE logo is a transparent-background circle,
so it is centred on a white plate.

Run from anywhere:  python TRACE/build/make_macos_icon.py
Requires cairosvg (+ the system Cairo library) and Pillow.
"""

import io
from pathlib import Path

import cairosvg
from PIL import Image, ImageDraw

_HERE = Path(__file__).resolve().parent
_SVG = _HERE.parent / "GUI_images" / "logo" / "LogoThick_light.svg"
_OUT = _HERE / "trace_icon.icns"

_CANVAS = 1024
_PLATE = 824  # Apple icon grid: 824 px plate on a 1024 px canvas
_RADIUS = 185
_LOGO = 660
_SUPERSAMPLE = 4  # draw the plate oversized, then downsample for smooth corners


def _master() -> Image.Image:
    big = _CANVAS * _SUPERSAMPLE
    plate = Image.new("RGBA", (big, big), (0, 0, 0, 0))
    inset = (_CANVAS - _PLATE) // 2 * _SUPERSAMPLE
    ImageDraw.Draw(plate).rounded_rectangle(
        (inset, inset, big - inset, big - inset),
        radius=_RADIUS * _SUPERSAMPLE,
        fill=(255, 255, 255, 255),
    )
    plate = plate.resize((_CANVAS, _CANVAS), Image.LANCZOS)
    logo_png = cairosvg.svg2png(url=str(_SVG), output_width=_LOGO, output_height=_LOGO)
    logo = Image.open(io.BytesIO(logo_png)).convert("RGBA")
    offset = (_CANVAS - _LOGO) // 2
    plate.alpha_composite(logo, (offset, offset))
    return plate


def main() -> None:
    # Pillow's ICNS writer downsamples the 1024 px master to every standard
    # size (16 … 512, plus @2x variants).
    _master().save(_OUT, format="ICNS")
    print(f"wrote {_OUT} ({_OUT.stat().st_size:,} bytes)")


if __name__ == "__main__":
    main()
