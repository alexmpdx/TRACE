"""Regenerate TRACE/build/trace_icon.ico from TRACE/GUI_images/logo/LogoThick_light.svg.

Run after any edit to the logo SVG. Inno Setup reads the .ico via the
``SetupIconFile=`` directive in installer.iss, so the .ico has to be
committed alongside the iss file for the GitHub Actions build to find it.

Sourced from the black-circle, thick-line LogoThick_light.svg variant.
Two reasons for that choice:
  - Light variant: the installer title bar / Explorer thumbnail / ARP
    entry are rendered in the OS chrome, which is typically light on
    Windows. The white-circle logo_dark would wash out there.
  - Thick variant: at the 16-48 px sizes that dominate the .ico's
    real-world rendering, the thin logo_*.svg loses too much line detail
    to remain readable.

Sizes are the standard Windows shell icon set: 16/24/32 (taskbar +
Explorer), 48/64 (medium icons), 128/256 (Start Menu, file-properties).
Windows picks the entry that matches the user's display DPI.
"""

import io
from pathlib import Path

import cairosvg
from PIL import Image

_HERE = Path(__file__).resolve().parent
_SVG = _HERE.parent / "GUI_images" / "logo" / "LogoThick_light.svg"
_ICO = _HERE / "trace_icon.ico"
_SIZES = [(16, 16), (24, 24), (32, 32), (48, 48), (64, 64), (128, 128), (256, 256)]


def main() -> None:
    png_bytes = cairosvg.svg2png(url=str(_SVG), output_width=256, output_height=256)
    img = Image.open(io.BytesIO(png_bytes)).convert("RGBA")
    img.save(_ICO, format="ICO", sizes=_SIZES)
    print(f"wrote {_ICO} ({_ICO.stat().st_size:,} bytes)")


if __name__ == "__main__":
    main()
