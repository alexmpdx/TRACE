"""Calibrate TRACE Stage 2 (identifyFeatures) memory and recommend --workers.

Runs preprocessing (Stage 1) on a single user-supplied image to produce the
detection + landmarks GeoJSONs identifyFeatures needs, then measures Stage 2
peak RSS in a subprocess via CPU_RAM_tester/recommend_workers.

Used both by the TRACE CLI (`--calibrate-workers PATH`) and the GUI Settings
dialog ("Calibrate" button on the General tab).
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path
from typing import Callable, Optional

# Make CPU_RAM_tester/recommend_workers importable.
_REPO_ROOT = Path(__file__).resolve().parent.parent
_CPU_RAM_TESTER = _REPO_ROOT / "CPU_RAM_tester"
if str(_CPU_RAM_TESTER) not in sys.path:
    sys.path.insert(0, str(_CPU_RAM_TESTER))

import recommend_workers  # noqa: E402

IMAGE_EXTS = (".tif", ".tiff", ".bmp", ".png", ".jpg", ".jpeg", ".psd", ".psb")


def pick_calibration_image(path: Path) -> Path:
    """Resolve a folder-or-file argument to a single image path."""
    p = Path(path).expanduser().resolve()
    if p.is_file():
        if p.suffix.lower() not in IMAGE_EXTS:
            raise ValueError(f"Not a supported image: {p.name}")
        return p
    if p.is_dir():
        for f in sorted(p.iterdir()):
            if f.name.startswith(".") or f.name.startswith("._"):
                continue
            if f.suffix.lower() in IMAGE_EXTS:
                return f
        raise FileNotFoundError(f"No supported images in folder: {p}")
    raise FileNotFoundError(f"Path not found: {p}")


def calibrate_for_trace(
    image_or_folder: Path,
    landmark_checkpoint: Path,
    segmentation_model_dir: Path,
    device=None,
    progress_callback: Optional[Callable[[str, str], None]] = None,
) -> dict:
    """Run Stage 1 on one image, measure Stage 2 peak RSS, return recommendation.

    Returns a dict with keys: system (SystemInfo), calibration (CalibrationStats),
    recommendation (dict), image_path (Path).

    `progress_callback(stage, detail)` is called with stage in
    {"preprocessing", "calibration"} when each phase begins. May be None.
    """
    from preprocessing.pipeline import process_single_image

    image_path = pick_calibration_image(Path(image_or_folder))

    landmark_checkpoint = Path(landmark_checkpoint).resolve()
    segmentation_model_dir = Path(segmentation_model_dir).resolve()
    if not landmark_checkpoint.exists():
        raise FileNotFoundError(f"Landmark model not found: {landmark_checkpoint}")
    if not segmentation_model_dir.is_dir():
        raise FileNotFoundError(f"Segmentation model dir not found: {segmentation_model_dir}")

    sysinfo = recommend_workers.probe_system()

    with tempfile.TemporaryDirectory(prefix="trace_calib_") as tmp_str:
        tmp_dir = Path(tmp_str)

        if progress_callback:
            progress_callback("preprocessing", f"Running Stage 1 on {image_path.name}")
        # Calibration cares about timing + peak RAM, not landmark accuracy.
        # Without disabling the gate, a borderline wing whose core landmarks
        # fail confidence checks aborts with LowConfidenceLandmarkError and
        # the user can't calibrate against their own images. Match the live-
        # preview pattern: include unreliable landmarks AND empty the core-
        # landmark set so the hard abort can't fire. The production pipeline
        # is untouched.
        preproc = process_single_image(
            image_path=image_path,
            output_dir=tmp_dir,
            landmark_checkpoint=landmark_checkpoint,
            segmentation_model_dir=segmentation_model_dir,
            stages=(True, True, True),
            device=device,
            include_unreliable_landmarks=True,
            gate_override={"core_landmarks": []},
        )
        if preproc.error is not None:
            raise RuntimeError(f"Stage 1 failed: {preproc.error}")
        if preproc.segmentation_geojson_path is None or preproc.landmarks_geojson_path is None:
            raise RuntimeError("Stage 1 did not produce required GeoJSON outputs.")

        if progress_callback:
            progress_callback("calibration", "Measuring Stage 2 peak memory")
        spec = (
            preproc.image_path.stem,
            preproc.segmentation_geojson_path,
            preproc.landmarks_geojson_path,
            preproc.image_path,
        )
        stat = recommend_workers.calibrate(spec, cli_extra=[])

    if not stat.success:
        # Stage 2 subprocess crashed — surface the actual reason so the
        # GUI can show it in the "Calibration failed" dialog instead of
        # silently returning a fallback recommendation of --workers 1
        # (which hides configuration / environment problems from the
        # user). stderr_tail travels back on CalibrationStats even when
        # sys.stderr is None (frozen Windows build; see #33).
        tail = stat.stderr_tail.strip() or f"CLI returned {stat.returncode}"
        raise RuntimeError(f"Stage 2 calibration subprocess failed:\n{tail}")

    rec = recommend_workers.recommend(stat, sysinfo)
    return {
        "system": sysinfo,
        "calibration": stat,
        "recommendation": rec,
        "image_path": image_path,
    }


def format_report(result: dict) -> str:
    """Multi-line human-readable report for printing or showing in a dialog."""
    sysinfo = result["system"]
    stat = result["calibration"]
    rec = result["recommendation"]
    lines = [
        "System:",
        sysinfo.pretty(),
        "",
        f"Calibration image : {result['image_path'].name}",
        f"  wall            : {stat.wall_s:.1f} s",
        f"  peak RSS        : {stat.peak_rss_gb:.2f} GiB",
        f"  status          : {'OK' if stat.success else f'rc={stat.returncode}'}",
        "",
        "Recommendation:",
    ]
    for k, v in rec.items():
        lines.append(f"  {k:>22}: {v}")
    lines.append("")
    lines.append(f"→ Suggested --workers : {rec['recommended_workers']}")
    return "\n".join(lines)


def main(argv=None) -> int:
    import argparse

    parser = argparse.ArgumentParser(
        description="Calibrate TRACE Stage 2 memory and recommend --workers.",
    )
    parser.add_argument(
        "path",
        type=Path,
        help="Folder containing wing images, or a single image.",
    )
    parser.add_argument("--landmark-model", required=True, type=Path)
    parser.add_argument("--segmentation-model", required=True, type=Path)
    parser.add_argument("--device", default=None, choices=["cpu", "cuda", "mps"])
    args = parser.parse_args(argv)

    device = None
    if args.device:
        import torch

        device = torch.device(args.device)

    def _progress(stage, detail):
        print(f"[{stage}] {detail}")

    try:
        result = calibrate_for_trace(
            image_or_folder=args.path,
            landmark_checkpoint=args.landmark_model,
            segmentation_model_dir=args.segmentation_model,
            device=device,
            progress_callback=_progress,
        )
    except Exception as e:
        print(f"Calibration failed: {e}", file=sys.stderr)
        return 1

    print()
    print(format_report(result))
    return 0


if __name__ == "__main__":
    sys.exit(main())
