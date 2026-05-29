"""Regenerate the installer's two .ico files from the LogoThick SVGs.

Outputs:
  - trace_icon_light.ico  — from LogoThick_light.svg (black background)
  - trace_icon_dark.ico   — from LogoThick_dark.svg  (white-on-dark)
  - trace_icon.ico        — copy of trace_icon_light.ico (kept so the
                            existing SetupIconFile= reference in
                            installer.iss still resolves; the installer
                            wizard chrome uses this one).

The installer's [Tasks]/[Icons] sections present a light/dark choice
under the "Create desktop icon" option; whichever the user picks
becomes IconFilename: on the shortcut.

Why per-size vector render: PIL's ICO writer downsamples a single
source image with BICUBIC, which blurs the thin logo strokes badly
at 16/24/32 px (the sizes Windows actually uses on the desktop and
taskbar). Rendering the SVG separately at each target size gives a
crisp result at every shell DPI.

Why thick variant: at small sizes the regular logo_*.svg loses too
much line detail to remain readable.

Sizes match the standard Windows shell set: 16/24/32 (taskbar +
Explorer), 48/64 (medium icons), 128/256 (Start Menu, file-properties).
"""

import io
import shutil
from pathlib import Path

import cairosvg
from PIL import Image

_HERE = Path(__file__).resolve().parent
_SVG_DIR = _HERE.parent / "GUI_images" / "logo"
_SIZES = [(16, 16), (24, 24), (32, 32), (48, 48), (64, 64), (128, 128), (256, 256)]


def _render_ico(svg_path: Path, ico_path: Path) -> None:
    """Render ``svg_path`` at each shell size and emit a multi-frame ICO."""
    frames = []
    for w, h in _SIZES:
        png_bytes = cairosvg.svg2png(url=str(svg_path), output_width=w, output_height=h)
        frames.append(Image.open(io.BytesIO(png_bytes)).convert("RGBA"))
    # Pillow's ICO writer skips any requested size larger than the base
    # image, so pass the largest frame as the base and the rest via
    # append_images. That way every entry in ``sizes`` finds a pre-
    # rendered match (no BICUBIC downsampling step at all).
    frames_largest_first = list(reversed(frames))
    base = frames_largest_first[0]
    base.save(
        ico_path,
        format="ICO",
        sizes=_SIZES,
        append_images=frames_largest_first[1:],
    )
    print(f"wrote {ico_path} ({ico_path.stat().st_size:,} bytes)")


def main() -> None:
    light_ico = _HERE / "trace_icon_light.ico"
    dark_ico = _HERE / "trace_icon_dark.ico"
    _render_ico(_SVG_DIR / "LogoThick_light.svg", light_ico)
    _render_ico(_SVG_DIR / "LogoThick_dark.svg", dark_ico)
    # Backwards-compat alias for installer.iss SetupIconFile=.
    legacy_ico = _HERE / "trace_icon.ico"
    shutil.copyfile(light_ico, legacy_ico)
    print(f"wrote {legacy_ico} ({legacy_ico.stat().st_size:,} bytes) — copy of light variant")


if __name__ == "__main__":
    main()
