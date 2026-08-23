"""Recover a landmarks-only measurements CSV from a TRACE run folder.

Ships with v0.2.25 to unblock users whose rerun-with-append run silently
destroyed their prior CSV data (bug report #36 — the `.append_source`
unlink was firing unconditionally, and the merge silently failed on
Windows-cp1252 vs UTF-8 encoding mismatch). Also useful for any run
whose landmarks-only fast-path CSV write was aborted (app close, kill,
disk error) mid-flight — as long as the per-image ``*_landmarks.geojson``
files are on disk, the CSV is fully reconstructible.

The recovery uses the SAME code path the pipeline uses
(``measurement_maker.write_landmark_csv_batch``), so the recovered CSV is
byte-identical to what the pipeline would have written on the same input.

Usage
-----

Run this against the folder that contains your per-image
``*_landmarks.geojson`` files. On a normal TRACE run that folder is
``<output_folder>/run_YYYYMMDD-HHMMSS`` — the "run folder", the one that
also contains ``settings.yaml`` and ``manifest.json``.

    # Auto-discover settings + write to <run_folder>/measurements_recovered.csv
    python tools/recover_landmark_csv.py "C:/Users/Alex/Work/Biggerlist_cvareaish/run_20260820-235936"

    # Explicit output path
    python tools/recover_landmark_csv.py <run_folder> --out C:/tmp/recovered.csv

    # Recurse into subfolders (useful if landmarks_geojson was written per-batch)
    python tools/recover_landmark_csv.py <run_folder> --recursive

Settings resolution
-------------------

Reads ``<run_folder>/settings.yaml`` if present to pick up:

- ``pipeline_config.um_per_px``     — µm/px scale for the ``_um`` columns
- ``gui_state.csv_measurement_groups`` — which measurement groups were
  selected (only ``cv_ratio`` is landmarks-only; ``wing_area`` /
  ``wing_shape`` need the wing outline and CANNOT be recovered from
  landmarks alone)
- ``gui_state.user_landmark_distances`` — custom landmark-pair distance
  definitions

Any of these can be overridden on the command line (see ``--help``).

Frozen-Windows-install layout
-----------------------------

If you're running the shipped Windows installer, the ``tools/`` folder
is bundled alongside ``TRACE.exe``. Open a PowerShell / Command Prompt
in that folder and run::

    TRACE.exe __python__ tools\recover_landmark_csv.py <run_folder>

(The ``__python__`` sentinel is the standard way to invoke arbitrary
Python inside the frozen bundle.)
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Optional


def _bootstrap_sys_path() -> None:
    """Add sibling package directories to sys.path so recovery imports work.

    Needs measurement_maker (for write_landmark_csv_batch) and resolutionAdjust
    (for the optional --images TIFF-metadata re-read). Neither pulls in torch,
    napari, or identify_features — this stays a light-touch script.

    Two import shapes to accommodate (matches TRACE/run_gui.py):
      - measurement_maker: package inside measurementMaker/ — add
        ``measurementMaker/`` to sys.path so ``import measurement_maker`` works.
      - resolutionAdjust: package at the top level (its __init__.py imports
        siblings as ``resolutionAdjust.auto_detect``) — add the PARENT of
        resolutionAdjust/ to sys.path so ``import resolutionAdjust`` works.
    """
    here = Path(__file__).resolve().parent
    # Handle both source layout (repo_root/tools/) and bundled layout
    # (bundle_root/tools/ alongside the sibling packages).
    for candidate in (here.parent, here.parent.parent):
        if (candidate / "measurementMaker").is_dir():
            _p = str(candidate / "measurementMaker")
            if _p not in sys.path:
                sys.path.insert(0, _p)
            # resolutionAdjust needs its PARENT on sys.path.
            if (candidate / "resolutionAdjust").is_dir() and str(candidate) not in sys.path:
                sys.path.insert(0, str(candidate))
            return
    # Frozen-Windows layout.
    bundle_root = Path(sys.executable).resolve().parent
    if (bundle_root / "measurementMaker").is_dir():
        _p = str(bundle_root / "measurementMaker")
        if _p not in sys.path:
            sys.path.insert(0, _p)
    if (bundle_root / "resolutionAdjust").is_dir() and str(bundle_root) not in sys.path:
        sys.path.insert(0, str(bundle_root))


_bootstrap_sys_path()


def _load_settings_yaml(run_folder: Path) -> dict:
    """Return the settings.yaml dict, or {} if missing / unreadable.

    YAML is optional — the script also works with fully explicit CLI args.
    """
    p = run_folder / "settings.yaml"
    if not p.is_file():
        return {}
    try:
        import yaml
    except ImportError:
        print("[warn] PyYAML not installed; skipping settings.yaml auto-discovery", file=sys.stderr)
        return {}
    try:
        with open(p, encoding="utf-8") as fh:
            return yaml.safe_load(fh) or {}
    except Exception as exc:  # noqa: BLE001
        print(f"[warn] could not parse {p}: {exc}", file=sys.stderr)
        return {}


def _find_landmark_files(root: Path, recursive: bool) -> dict[str, Path]:
    """Return {specimen_stem: path} for every ``*_landmarks.geojson`` under root.

    Specimen stem strips the "_landmarks" suffix to match what the pipeline's
    write_landmark_csv_batch expects (the CSV's "specimen" column).
    """
    pattern = "*_landmarks.geojson"
    iterator = root.rglob(pattern) if recursive else root.glob(pattern)
    out: dict[str, Path] = {}
    for path in iterator:
        stem = path.stem
        if stem.endswith("_landmarks"):
            stem = stem[: -len("_landmarks")]
        out[stem] = path
    return out


# Supported image extensions for the --images backfill. Kept aligned with
# preprocessing.pipeline.discover_images so we scan for the same file set.
_IMAGE_EXTS = (".tif", ".tiff", ".ome.tif", ".ome.tiff", ".psd", ".bmp", ".png", ".jpg", ".jpeg")


def _subpath_target_stem(image_path: Path, images_root: Path) -> str:
    """Mirror preprocessing.pipeline._subpath_target_name, minus the extension.

    ``images_root=/dir1, image_path=/dir1/folderA/sub/img.tif`` →
    ``folderA_sub_img``.
    """
    parts = image_path.relative_to(images_root).parts
    if not parts:
        return image_path.stem
    # Reconstruct the flattened basename, then strip its extension.
    flat_name = "_".join(parts)
    return Path(flat_name).stem


def _index_images_by_flat_stem(images_root: Path) -> dict[str, Path]:
    """Build a {flat_stem: image_path} map by recursively walking images_root.

    Uses the same flattening as preprocessing so a landmark whose stem was
    derived from ``folderA_sub_img_resampled`` matches the original image at
    ``folderA/sub/img.tif`` (the ``_resampled`` suffix is stripped by the
    caller when looking up).
    """
    out: dict[str, Path] = {}
    for path in images_root.rglob("*"):
        if not path.is_file():
            continue
        # rglob("*") is case-sensitive on some platforms; do a lowercase suffix check.
        low = path.name.lower()
        if not any(low.endswith(ext) for ext in _IMAGE_EXTS):
            continue
        stem = _subpath_target_stem(path, images_root)
        # Later occurrences overwrite earlier ones for same flat_stem —
        # collisions are unlikely (that's the whole point of flattening
        # in the first place) but if they happen we can't do better than
        # picking one.
        out[stem] = path
    return out


def _landmark_stem_to_image_stem(landmark_stem: str) -> list[str]:
    """Candidate flattened image stems for a given landmark stem.

    The pipeline can produce landmarks from either the original image or its
    resolutionAdjust-rescaled sibling (with a `_resampled` suffix). Return
    both candidates (with-suffix first, without) so the caller can try each
    against the pre-built flat_stem index.
    """
    candidates = [landmark_stem]
    if landmark_stem.endswith("_resampled"):
        candidates.append(landmark_stem[: -len("_resampled")])
    return candidates


def _read_image_um_per_px(path: Path) -> Optional[float]:
    """Best-effort read of µm/px from a single image file.

    Uses resolutionAdjust.auto_detect._read_um_per_px_from_tiff for TIFF-family
    files (returns None for anything else — PSD/BMP/etc. don't carry
    resolution metadata in a form we support).
    """
    low = path.name.lower()
    if not any(low.endswith(ext) for ext in (".tif", ".tiff", ".ome.tif", ".ome.tiff")):
        return None
    try:
        from resolutionAdjust.auto_detect import _read_um_per_px_from_tiff
    except ImportError:
        return None
    try:
        return _read_um_per_px_from_tiff(path)
    except Exception as exc:  # noqa: BLE001
        print(f"[warn] TIFF metadata read failed for {path}: {exc}", file=sys.stderr)
        return None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Recover a landmarks-only measurements CSV from a TRACE run folder.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("run_folder", type=Path, help="Path to the TRACE run folder to recover from.")
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output CSV path. Default: <run_folder>/measurements_recovered.csv",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Recurse into subfolders when hunting for *_landmarks.geojson.",
    )
    parser.add_argument(
        "--um-per-px",
        type=float,
        default=None,
        help="Override the DEFAULT µm/px scale from settings.yaml. Set to 0 to emit px-only "
        "columns. Per-specimen scale from --images / geojson fc_props still wins over this.",
    )
    parser.add_argument(
        "--images",
        type=Path,
        default=None,
        help="Path to the ORIGINAL input images folder. Enables per-image µm/px re-detection "
        "from TIFF metadata — matches auto_detect_um_per_px=True runs where every image can "
        "have a different scale. Landmark geojsons written by TRACE >= v0.2.27 already carry "
        "their effective_um_per_px, so this arg is only needed for older runs.",
    )
    parser.add_argument(
        "--include-cv-ratio",
        dest="include_cv_ratio",
        action="store_true",
        default=None,
        help="Force cv_ratio columns on (default: read from settings.yaml).",
    )
    parser.add_argument(
        "--no-cv-ratio",
        dest="include_cv_ratio",
        action="store_false",
        help="Force cv_ratio columns off.",
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable INFO logging.")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(levelname)s %(name)s: %(message)s",
    )

    run_folder = args.run_folder.expanduser().resolve()
    if not run_folder.is_dir():
        print(f"error: {run_folder} is not a directory", file=sys.stderr)
        return 2

    settings = _load_settings_yaml(run_folder)
    gui_state = settings.get("gui_state", {}) or {}
    pipeline_config = settings.get("pipeline_config", {}) or {}

    # Scale precedence: CLI > settings.yaml > None (px only).
    if args.um_per_px is not None:
        um_per_px = args.um_per_px if args.um_per_px > 0 else None
    else:
        raw = pipeline_config.get("um_per_px")
        um_per_px = float(raw) if raw is not None and raw > 0 else None

    # Measurement groups: only cv_ratio is landmarks-only; ignore the rest.
    if args.include_cv_ratio is None:
        groups_dict = gui_state.get("csv_measurement_groups", {}) or {}
        include_cv_ratio = bool(groups_dict.get("cv_ratio"))
    else:
        include_cv_ratio = bool(args.include_cv_ratio)
    measurement_groups = {"cv_ratio"} if include_cv_ratio else set()

    # User-defined landmark pairs (list of dicts with name_a / name_b / label).
    from measurement_maker import pairs_from_dicts, write_landmark_csv_batch

    raw_pairs = gui_state.get("user_landmark_distances") or []
    pairs = pairs_from_dicts(raw_pairs) if raw_pairs else []

    if not pairs and not include_cv_ratio:
        print(
            "error: nothing to write — settings.yaml has neither user_landmark_distances "
            "nor cv_ratio selected. Use --include-cv-ratio to force cv_ratio, or pass an "
            "explicit --um-per-px if you're recovering a landmarks-only distance CSV.",
            file=sys.stderr,
        )
        return 2

    specimens = _find_landmark_files(run_folder, recursive=args.recursive)
    if not specimens:
        recursed = " (recursively)" if args.recursive else ""
        print(
            f"error: no *_landmarks.geojson files found under {run_folder}{recursed}",
            file=sys.stderr,
        )
        return 2

    out_path = args.out.expanduser().resolve() if args.out else (run_folder / "measurements_recovered.csv")

    # Optional per-specimen scale backfill from TIFF metadata. Only used
    # for landmark geojsons that don't already carry effective_um_per_px
    # (pre-v0.2.27 runs), and only for image extensions that carry the
    # metadata (TIFF family).
    um_per_px_by_specimen: dict[str, float] = {}
    if args.images is not None:
        images_root = args.images.expanduser().resolve()
        if not images_root.is_dir():
            print(f"error: --images {images_root} is not a directory", file=sys.stderr)
            return 2
        print(f"Indexing images under {images_root} for per-image µm/px re-detection…")
        image_index = _index_images_by_flat_stem(images_root)
        print(f"  indexed {len(image_index)} image file(s)")

        hits = misses = 0
        for stem in specimens:
            found_scale: Optional[float] = None
            for candidate in _landmark_stem_to_image_stem(stem):
                img_path = image_index.get(candidate)
                if img_path is None:
                    continue
                found_scale = _read_image_um_per_px(img_path)
                if found_scale is not None:
                    break
            if found_scale is not None:
                um_per_px_by_specimen[stem] = found_scale
                hits += 1
            else:
                misses += 1
        print(f"  per-image µm/px resolved for {hits}/{len(specimens)} specimen(s) via image metadata")
        if misses:
            print(f"  ({misses} fell back to geojson fc_props / --um-per-px default)")

    print(f"Recovering CSV from {len(specimens)} landmark geojson(s) under {run_folder}")
    print(f"  default scale (µm/px): {um_per_px if um_per_px else '(none — px only)'}")
    print(f"  include cv_ratio:      {include_cv_ratio}")
    print(f"  user distance pairs:   {len(pairs)}")
    print(f"  --images backfill:     {len(um_per_px_by_specimen)} specimen(s)")
    print(f"  output:                {out_path}")

    write_landmark_csv_batch(
        out_path,
        specimens,
        pairs,
        measurement_groups=measurement_groups,
        um_per_px=um_per_px,
        um_per_px_by_specimen=um_per_px_by_specimen or None,
    )
    if not out_path.is_file():
        print("error: write_landmark_csv_batch did not produce an output file", file=sys.stderr)
        return 1
    print(f"Wrote {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
