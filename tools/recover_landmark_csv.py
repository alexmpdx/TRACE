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


def _bootstrap_sys_path() -> None:
    """Add the sibling package directories to sys.path so measurement_maker imports.

    Mirrors the minimal subset of TRACE/run_gui.py's path setup. Only
    measurement_maker is needed (no torch, no napari, no identify_features).
    """
    here = Path(__file__).resolve().parent
    # Handle both source layout (repo_root/tools/) and bundled layout
    # (bundle_root/tools/ alongside bundle_root/measurementMaker/).
    for candidate in (here.parent, here.parent.parent):
        mm_dir = candidate / "measurementMaker"
        if mm_dir.is_dir():
            for name in ("measurementMaker",):
                p = str(candidate / name)
                if p not in sys.path:
                    sys.path.insert(0, p)
            return
    # Frozen-Windows layout: measurementMaker sits next to TRACE.exe / the tools folder.
    bundle_root = Path(sys.executable).resolve().parent
    mm_dir = bundle_root / "measurementMaker"
    if mm_dir.is_dir():
        p = str(mm_dir)
        if p not in sys.path:
            sys.path.insert(0, p)


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
        help="Override the µm/px scale from settings.yaml. Set to 0 to emit px-only columns.",
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

    print(f"Recovering CSV from {len(specimens)} landmark geojson(s) under {run_folder}")
    print(f"  scale (µm/px):        {um_per_px if um_per_px else '(none — px only)'}")
    print(f"  include cv_ratio:     {include_cv_ratio}")
    print(f"  user distance pairs:  {len(pairs)}")
    print(f"  output:               {out_path}")

    write_landmark_csv_batch(
        out_path,
        specimens,
        pairs,
        measurement_groups=measurement_groups,
        um_per_px=um_per_px,
    )
    if not out_path.is_file():
        print("error: write_landmark_csv_batch did not produce an output file", file=sys.stderr)
        return 1
    print(f"Wrote {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
