"""cv2.imread-compatible loader with extended-format support.

Handles, in addition to formats cv2 reads natively:
  - .psd / .psb  (psd-tools, flattened composite)
  - .heic / .heif (pillow-heif)
  - .raw / camera RAW (.dng .nef .cr2 .cr3 .arw .raf .orf .pef .rw2 .srw) via rawpy
  - .svg (cairosvg, requires the system Cairo library)
  - microscopy formats — .czi (czifile), .nd2 (nd2), .lif (readlif), .lsm
    (tifffile). Multi-dimensional stacks are reduced to 2D YX or YXC by:
      • picking T=0 and the first scene/series
      • max-projecting along Z
      • picking the first 3 channels as RGB (or grayscale → 3-channel duplicate)
    Use ``convert_microscopy_to_ome_tiff`` to write a proper OME-TIFF instead
    of decoding to a flat ndarray.

Returns a BGR / BGRA / Gray ndarray matching the cv2 flag. 16-bit / CMYK / Lab /
indexed modes are coerced to 8-bit on load.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Union

import cv2
import numpy as np

_PSD_EXTS = {".psd", ".psb"}
_HEIC_EXTS = {".heic", ".heif"}
_SVG_EXTS = {".svg"}
_PDF_EXTS = {".pdf"}
_RAW_EXTS = {
    ".raw",
    ".dng",
    ".nef",
    ".cr2",
    ".cr3",
    ".arw",
    ".raf",
    ".orf",
    ".pef",
    ".rw2",
    ".srw",
}
_MICROSCOPY_EXTS = {".czi", ".nd2", ".lif", ".lsm"}


def imread_any(path: Union[str, Path], flag: int = cv2.IMREAD_COLOR) -> Optional[np.ndarray]:
    p = str(path)
    ext = Path(p).suffix.lower()
    if ext in _PSD_EXTS:
        return _imread_psd(p, flag)
    if ext in _HEIC_EXTS:
        return _imread_heic(p, flag)
    if ext in _SVG_EXTS:
        return _imread_svg(p, flag)
    if ext in _PDF_EXTS:
        return _imread_pdf(p, flag)
    if ext in _RAW_EXTS:
        return _imread_raw(p, flag)
    if ext in _MICROSCOPY_EXTS:
        return _imread_microscopy(p, flag)
    return cv2.imread(p, flag)


def _pil_to_cv(pil, flag: int) -> np.ndarray:
    """Convert a PIL Image into a cv2-shaped ndarray for the requested flag."""
    if flag == cv2.IMREAD_GRAYSCALE:
        return np.array(pil.convert("L"))
    if flag == cv2.IMREAD_UNCHANGED:
        mode = pil.mode
        if mode in ("RGBA", "LA") or (mode == "P" and "transparency" in pil.info):
            arr = np.array(pil.convert("RGBA"))
            return cv2.cvtColor(arr, cv2.COLOR_RGBA2BGRA)
        if mode == "L":
            return np.array(pil)
        arr = np.array(pil.convert("RGB"))
        return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
    arr = np.array(pil.convert("RGB"))
    return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)


def _imread_psd(path: str, flag: int) -> Optional[np.ndarray]:
    try:
        from psd_tools import PSDImage
    except ImportError as exc:  # pragma: no cover
        raise ImportError("Reading .psd files requires psd-tools. Install with: pip install psd-tools") from exc

    try:
        psd = PSDImage.open(path)
    except Exception:
        return None
    pil = psd.composite()
    if pil is None:
        return None
    return _pil_to_cv(pil, flag)


def _imread_heic(path: str, flag: int) -> Optional[np.ndarray]:
    try:
        import pillow_heif
        from PIL import Image
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "Reading .heic/.heif files requires pillow-heif. Install with: pip install pillow-heif"
        ) from exc

    pillow_heif.register_heif_opener()
    try:
        pil = Image.open(path)
    except Exception:
        return None
    return _pil_to_cv(pil, flag)


def _imread_svg(path: str, flag: int) -> Optional[np.ndarray]:
    """Rasterize an SVG to a BGR ndarray. Default DPI 300, no resizing."""
    try:
        import cairosvg
    except (ImportError, OSError) as exc:  # pragma: no cover
        raise ImportError(
            "Reading .svg files requires cairosvg + the system Cairo library. "
            "On macOS: `brew install cairo && pip install cairosvg`. "
            f"(Underlying error: {exc})"
        ) from exc

    from io import BytesIO

    from PIL import Image

    try:
        png_bytes = cairosvg.svg2png(url=path, dpi=300)
    except Exception:
        return None
    pil = Image.open(BytesIO(png_bytes))
    return _pil_to_cv(pil, flag)


# Render DPI for PDF → image conversion. 200 is a sweet spot: high enough to
# preserve fine vein detail for the landmark picker, low enough that a typical
# A4/letter page fits comfortably in memory (~2000×2600 px).
_PDF_RENDER_DPI = 200
# Hard ceiling on the rendered pixel count. If the page at _PDF_RENDER_DPI
# would exceed this, the DPI is scaled down proportionally. Prevents
# pathologically-sized pages (custom MediaBox, scanned 600+ DPI bitmap PDFs)
# from triggering a multi-gigabyte allocation.
_PDF_MAX_RENDER_PIXELS = 50_000_000  # ≈ 7100 × 7100 px


def _imread_pdf(path: str, flag: int) -> Optional[np.ndarray]:
    """Render the first page of a PDF as a cv2-shaped ndarray via PyMuPDF.

    Multi-page PDFs are reduced to the first page — the picker only needs
    one sample image and wing micrograph PDFs are typically single-page.

    The render goes through PNG bytes (``Pixmap.tobytes("png")``) and
    ``cv2.imdecode`` rather than manual ``np.frombuffer`` + ``reshape`` on
    the raw pixel buffer; pymupdf's Pixmap layout varies subtly with
    colorspace + alpha and the manual path is easy to misread (which
    triggers garbage allocations downstream in cv2 / napari).
    """
    try:
        import pymupdf  # PyMuPDF — module renamed from 'fitz' in 1.24+
    except ImportError:
        try:
            import fitz as pymupdf  # legacy module name
        except ImportError as exc:  # pragma: no cover
            raise ImportError("Reading .pdf files requires PyMuPDF. Install with: pip install pymupdf") from exc

    try:
        doc = pymupdf.open(path)
    except Exception:
        return None
    if doc.page_count < 1:
        doc.close()
        return None
    try:
        page = doc.load_page(0)
        # Compute the zoom needed for the target DPI, then dial it back if
        # the resulting pixel count would exceed _PDF_MAX_RENDER_PIXELS.
        zoom = _PDF_RENDER_DPI / 72.0
        rect = page.rect
        projected_pixels = (rect.width * zoom) * (rect.height * zoom)
        if projected_pixels > _PDF_MAX_RENDER_PIXELS:
            zoom *= (_PDF_MAX_RENDER_PIXELS / projected_pixels) ** 0.5
        matrix = pymupdf.Matrix(zoom, zoom)
        pix = page.get_pixmap(matrix=matrix, alpha=False)
        png_bytes = pix.tobytes("png")
    finally:
        doc.close()

    decoded = cv2.imdecode(np.frombuffer(png_bytes, dtype=np.uint8), flag)
    return decoded


def _imread_raw(path: str, flag: int) -> Optional[np.ndarray]:
    """Decode a camera RAW via rawpy.postprocess() (auto white balance, sRGB)."""
    try:
        import rawpy
    except ImportError as exc:  # pragma: no cover
        raise ImportError("Reading camera RAW files requires rawpy. Install with: pip install rawpy") from exc

    try:
        with rawpy.imread(path) as raw:
            rgb = raw.postprocess(use_camera_wb=True, no_auto_bright=False, output_bps=8)
    except Exception:
        return None
    if rgb is None:
        return None
    if flag == cv2.IMREAD_GRAYSCALE:
        return cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    if flag == cv2.IMREAD_UNCHANGED and rgb.ndim == 2:
        return rgb
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


# ---------------------------------------------------------------------------
# Microscopy formats (CZI / ND2 / LIF / LSM) — reduced to YX / YXC, optionally
# round-tripped through OME-TIFF for downstream cv2 consumers.
# ---------------------------------------------------------------------------
def _read_microscopy_array(path: str) -> Optional[np.ndarray]:
    """Decode a microscopy file into a numpy array. Shape varies by format."""
    ext = Path(path).suffix.lower()
    if ext == ".czi":
        try:
            import czifile
        except ImportError as exc:  # pragma: no cover
            raise ImportError("Reading .czi files requires czifile. Install with: pip install czifile") from exc
        try:
            return czifile.imread(path)
        except Exception:
            return None
    if ext == ".nd2":
        try:
            import nd2
        except ImportError as exc:  # pragma: no cover
            raise ImportError("Reading .nd2 files requires nd2. Install with: pip install nd2") from exc
        try:
            return nd2.imread(path)
        except Exception:
            return None
    if ext == ".lif":
        try:
            from readlif.reader import LifFile
        except ImportError as exc:  # pragma: no cover
            raise ImportError("Reading .lif files requires readlif. Install with: pip install readlif") from exc
        try:
            lif = LifFile(path)
            entries = list(lif.get_iter_image())
            if not entries:
                return None
            entry = entries[0]
            n_z = max(1, getattr(entry.dims, "z", 1) or 1)
            n_t = max(1, getattr(entry.dims, "t", 1) or 1)
            n_c = max(1, getattr(entry.channels, "n", 1) or getattr(entry, "channels", 1) or 1)
            # Build (C, Z, Y, X) for t=0; we reduce later.
            stacks = []
            for c in range(n_c):
                z_planes = []
                for z in range(n_z):
                    pil = entry.get_frame(z=z, t=0, c=c)
                    z_planes.append(np.array(pil))
                stacks.append(np.stack(z_planes, axis=0))
            return np.stack(stacks, axis=0)
        except Exception:
            return None
    if ext == ".lsm":
        try:
            import tifffile
        except ImportError as exc:  # pragma: no cover
            raise ImportError("Reading .lsm files requires tifffile.") from exc
        try:
            return tifffile.imread(path)
        except Exception:
            return None
    return None


def _reduce_microscopy_to_2d_or_rgb(arr: np.ndarray) -> Optional[np.ndarray]:
    """Squeeze + reduce a microscopy ndarray to (Y, X) or (Y, X, 3).

    Heuristic: drop singleton axes, treat the smallest remaining non-spatial
    axis as the channel axis (≤4 elements), max-project anything else (Z/T/Scene).
    The two largest axes are assumed to be Y and X.
    """
    a = np.squeeze(arr)
    if a.ndim == 2:
        return a

    # Identify the two largest axes as Y and X.
    yx_axes = sorted(np.argsort(a.shape)[-2:].tolist())
    yx_set = set(yx_axes)
    other_axes = [ax for ax in range(a.ndim) if ax not in yx_set]
    if not other_axes:
        return a  # already YX

    # Pick a channel axis: smallest "other" axis with size ≤ 4.
    channel_axis = None
    for ax in sorted(other_axes, key=lambda x: a.shape[x]):
        if 1 <= a.shape[ax] <= 4:
            channel_axis = ax
            break

    # Reduce all non-Y/X/channel axes via max projection (Z, T, scene, …).
    reduce_axes = [ax for ax in other_axes if ax != channel_axis]
    if reduce_axes:
        # Project from the largest axis index downwards so positions stay valid.
        for ax in sorted(reduce_axes, reverse=True):
            a = np.max(a, axis=ax)
            if channel_axis is not None and ax < channel_axis:
                channel_axis -= 1
            yx_axes = [ax_y - (1 if ax < ax_y else 0) for ax_y in yx_axes]

    if channel_axis is None:
        # No channel axis — emit grayscale.
        return a if a.ndim == 2 else np.squeeze(a)

    # Move channel axis to last position to get (Y, X, C).
    a = np.moveaxis(a, channel_axis, -1)
    if a.shape[-1] == 1:
        return a[..., 0]
    if a.shape[-1] >= 3:
        return a[..., :3]
    if a.shape[-1] == 2:
        # 2 channels — pad to 3 so callers see RGB.
        z = np.zeros_like(a[..., :1])
        return np.concatenate([a, z], axis=-1)
    return a


def _to_uint8(arr: np.ndarray) -> np.ndarray:
    """Coerce a numeric ndarray to uint8 with sane scaling for >8-bit microscopy data."""
    if arr.dtype == np.uint8:
        return arr
    a = arr.astype(np.float32)
    lo = float(a.min())
    hi = float(a.max())
    if hi <= lo:
        return np.zeros_like(a, dtype=np.uint8)
    return ((a - lo) / (hi - lo) * 255.0 + 0.5).astype(np.uint8)


def _imread_microscopy(path: str, flag: int) -> Optional[np.ndarray]:
    """Read a microscopy file and reduce it to a 2D / RGB cv2-shaped ndarray."""
    raw = _read_microscopy_array(path)
    if raw is None:
        return None
    reduced = _reduce_microscopy_to_2d_or_rgb(raw)
    if reduced is None:
        return None
    arr = _to_uint8(reduced)
    if arr.ndim == 2:
        if flag == cv2.IMREAD_GRAYSCALE:
            return arr
        bgr = cv2.cvtColor(arr, cv2.COLOR_GRAY2BGR)
        return bgr
    # arr is (Y, X, C) — czifile / nd2 / readlif return RGB-like channel order.
    if flag == cv2.IMREAD_GRAYSCALE:
        return cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
    return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)


def imwrite_ome_tiff(path: Union[str, Path], image: np.ndarray) -> Path:
    """Write a cv2-shaped ndarray (Gray / BGR / BGRA, uint8 or wider) as OME-TIFF.

    BGR/BGRA arrays are converted to RGB/RGBA before write so the OME-XML
    `photometric="rgb"` claim is correct on disk. cv2.imread reads the result
    back as BGR, which matches the round-trip every other intermediate uses.
    Path may be given without the ``.ome.tif`` suffix; the function ensures
    the final extension is ``.ome.tif`` for clarity. Returns the actual path.
    """
    import tifffile

    out = Path(path)
    if not str(out).lower().endswith((".ome.tif", ".ome.tiff")):
        if out.suffix.lower() in (".tif", ".tiff"):
            out = out.with_suffix("")  # strip .tif / .tiff
        out = out.with_name(out.name + ".ome.tif")
    out.parent.mkdir(parents=True, exist_ok=True)

    if image.ndim == 2:
        tifffile.imwrite(str(out), image, ome=True, photometric="minisblack")
    elif image.ndim == 3 and image.shape[-1] == 3:
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        tifffile.imwrite(str(out), rgb, ome=True, photometric="rgb")
    elif image.ndim == 3 and image.shape[-1] == 4:
        rgba = cv2.cvtColor(image, cv2.COLOR_BGRA2RGBA)
        tifffile.imwrite(str(out), rgba, ome=True, photometric="rgb")
    else:
        # Unusual shape (e.g. 16-bit multi-band stack) — write as-is.
        tifffile.imwrite(str(out), image, ome=True)
    return out


def convert_microscopy_to_ome_tiff(src: Union[str, Path], dest_dir: Union[str, Path]) -> Path:
    """Decode a microscopy file (CZI/ND2/LIF/LSM) and write it as OME-TIFF.

    The output is reduced to (Y, X) or (Y, X, 3) — same reduction `imread_any`
    applies to microscopy inputs — but written via tifffile with `ome=True`
    so OME-XML is embedded. Returns the path of the new `<stem>.ome.tif`.
    """
    src = Path(src)
    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)

    raw = _read_microscopy_array(str(src))
    if raw is None:
        raise IOError(f"Could not decode microscopy file: {src}")
    reduced = _reduce_microscopy_to_2d_or_rgb(raw)
    if reduced is None:
        raise IOError(f"Could not reduce microscopy stack to 2D/RGB: {src}")
    arr = _to_uint8(reduced)

    import tifffile

    out_path = dest_dir / f"{src.stem}.ome.tif"
    if arr.ndim == 2:
        tifffile.imwrite(str(out_path), arr, ome=True, photometric="minisblack")
    else:
        # (Y, X, 3) — tifffile expects channels first when writing OME with photometric=rgb.
        tifffile.imwrite(str(out_path), arr, ome=True, photometric="rgb")
    return out_path
