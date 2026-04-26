#!/usr/bin/env python3
"""Recommend a safe `--workers` count for identify-features.

v1 strategy: probe the system, run the CLI on a single specimen as a subprocess,
sample its peak RSS, then compute:

    workers = min(physical_cores, floor((available_ram - reserve) / (peak * safety)))

Fast (~one specimen), good enough for a sane recommendation.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
import threading
import time
from dataclasses import asdict, dataclass
from pathlib import Path

try:
    import psutil
except ImportError:
    sys.stderr.write("This script requires psutil. Install with: pip install psutil\n")
    sys.exit(2)


REPO_ROOT = Path(__file__).resolve().parents[1]
IDENTIFY_FEATURES_SRC = REPO_ROOT / "identifyFeatures"
DEFAULT_DET = IDENTIFY_FEATURES_SRC / "geojsons"
DEFAULT_LM = IDENTIFY_FEATURES_SRC / "LandmarkLocator_output"
DEFAULT_IMG = IDENTIFY_FEATURES_SRC / "OGpics"

GIB = 1024**3
SAMPLE_INTERVAL = 0.1  # seconds between RSS samples
RAM_RESERVE_GB = 2.0  # held back for OS / other apps
SAFETY_FACTOR = 1.3  # multiplier on observed peak for headroom


# --------------------------------------------------------------------------- #
# System probe
# --------------------------------------------------------------------------- #


@dataclass
class SystemInfo:
    physical_cores: int
    logical_cores: int
    total_ram_gb: float
    available_ram_gb: float
    platform: str

    def pretty(self) -> str:
        return (
            f"  Physical cores : {self.physical_cores}\n"
            f"  Logical cores  : {self.logical_cores}\n"
            f"  Total RAM      : {self.total_ram_gb:.1f} GiB\n"
            f"  Available RAM  : {self.available_ram_gb:.1f} GiB\n"
            f"  Platform       : {self.platform}"
        )


def probe_system() -> SystemInfo:
    vm = psutil.virtual_memory()
    return SystemInfo(
        physical_cores=psutil.cpu_count(logical=False) or 1,
        logical_cores=psutil.cpu_count(logical=True) or 1,
        total_ram_gb=vm.total / GIB,
        available_ram_gb=vm.available / GIB,
        platform=sys.platform,
    )


# --------------------------------------------------------------------------- #
# Specimen discovery
# --------------------------------------------------------------------------- #


def find_first_specimen(
    det_dir: Path, lm_dir: Path, img_dir: Path | None
) -> tuple[str, Path, Path, Path | None] | None:
    """First detection that has matching landmarks (and image, if dir given)."""
    for det in sorted(det_dir.glob("*_detections.geojson")):
        stem = det.name.replace("_detections.geojson", "")
        lm_path = lm_dir / f"{stem}_landmarks.geojson"
        if not lm_path.exists():
            lm_path = lm_dir / f"{stem} _landmarks.geojson"
        if not lm_path.exists():
            continue
        img_path: Path | None = None
        if img_dir is not None and img_dir.is_dir():
            for ext in (".tif", ".tiff", ".bmp", ".png", ".jpg", ".jpeg", ".psd", ".psb"):
                p = img_dir / f"{stem}{ext}"
                if p.exists():
                    img_path = p
                    break
        return stem, det, lm_path, img_path
    return None


# --------------------------------------------------------------------------- #
# Subprocess monitor
# --------------------------------------------------------------------------- #


@dataclass
class CalibrationStats:
    specimen: str
    wall_s: float
    peak_rss_gb: float
    success: bool
    returncode: int


class _Monitor(threading.Thread):
    def __init__(self, root_pid: int):
        super().__init__(daemon=True)
        self.root_pid = root_pid
        self.peak_rss = 0
        self.stop_event = threading.Event()

    def run(self) -> None:
        try:
            root = psutil.Process(self.root_pid)
        except psutil.NoSuchProcess:
            return
        while not self.stop_event.is_set():
            try:
                with root.oneshot():
                    procs = [root] + root.children(recursive=True)
                total = 0
                for p in procs:
                    try:
                        total += p.memory_info().rss
                    except (psutil.NoSuchProcess, psutil.AccessDenied):
                        continue
                if total > self.peak_rss:
                    self.peak_rss = total
            except psutil.NoSuchProcess:
                break
            time.sleep(SAMPLE_INTERVAL)


def calibrate(spec: tuple[str, Path, Path, Path | None], cli_extra: list[str]) -> CalibrationStats:
    stem, det, lm, img = spec
    with tempfile.TemporaryDirectory(prefix="cpu_ram_tester_") as out_str:
        out_dir = Path(out_str)
        cmd = [
            sys.executable,
            "-m",
            "identify_features.cli",
            str(det),
            str(lm),
        ]
        if img is not None:
            cmd.append(str(img))
        cmd += ["--output-dir", str(out_dir), *cli_extra]

        env = os.environ.copy()
        existing_pp = env.get("PYTHONPATH", "")
        env["PYTHONPATH"] = (
            f"{IDENTIFY_FEATURES_SRC}{os.pathsep}{existing_pp}" if existing_pp else str(IDENTIFY_FEATURES_SRC)
        )

        t0 = time.time()
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            env=env,
        )
        monitor = _Monitor(proc.pid)
        monitor.start()
        stdout, _ = proc.communicate()
        monitor.stop_event.set()
        monitor.join(timeout=2.0)
        wall = time.time() - t0

        success = proc.returncode == 0
        if not success:
            tail = "\n".join(stdout.splitlines()[-20:])
            sys.stderr.write(f"\nCLI exited {proc.returncode}; tail:\n{tail}\n")

        return CalibrationStats(
            specimen=stem,
            wall_s=wall,
            peak_rss_gb=monitor.peak_rss / GIB,
            success=success,
            returncode=proc.returncode,
        )


# --------------------------------------------------------------------------- #
# Recommendation
# --------------------------------------------------------------------------- #


def recommend(stat: CalibrationStats, sysinfo: SystemInfo) -> dict:
    if not stat.success or stat.peak_rss_gb <= 0:
        return {
            "recommended_workers": 1,
            "reason": "Calibration run failed; defaulting to --workers 1.",
        }

    usable_ram = max(0.0, sysinfo.available_ram_gb - RAM_RESERVE_GB)
    per_worker = stat.peak_rss_gb * SAFETY_FACTOR
    ram_cap = max(1, int(usable_ram // per_worker)) if per_worker > 0 else sysinfo.physical_cores
    cpu_cap = sysinfo.physical_cores
    rec = max(1, min(cpu_cap, ram_cap))

    binding = "CPU (physical cores)" if rec == cpu_cap and cpu_cap <= ram_cap else "RAM (per-worker peak)"
    return {
        "recommended_workers": rec,
        "binding_constraint": binding,
        "cpu_cap": cpu_cap,
        "ram_cap": ram_cap,
        "peak_rss_gb": round(stat.peak_rss_gb, 2),
        "per_worker_budget_gb": round(per_worker, 2),
        "ram_reserve_gb": RAM_RESERVE_GB,
        "safety_factor": SAFETY_FACTOR,
        "calibration_wall_s": round(stat.wall_s, 1),
    }


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Calibrate identify-features memory on one specimen and recommend --workers.",
    )
    parser.add_argument("--detections-dir", type=Path, default=DEFAULT_DET)
    parser.add_argument("--landmarks-dir", type=Path, default=DEFAULT_LM)
    parser.add_argument("--image-dir", type=Path, default=DEFAULT_IMG)
    parser.add_argument(
        "--detection",
        type=Path,
        default=None,
        help="Calibrate against this specific detection file instead of auto-pick.",
    )
    parser.add_argument(
        "--landmarks",
        type=Path,
        default=None,
        help="Landmarks file paired with --detection.",
    )
    parser.add_argument(
        "--image",
        type=Path,
        default=None,
        help="Optional image file paired with --detection.",
    )
    parser.add_argument("--json", type=Path, default=None, help="Write JSON report to this path.")
    parser.add_argument(
        "--cli-extra",
        type=str,
        default="",
        help="Extra args passed verbatim to identify-features (e.g. '--preset fast').",
    )
    args = parser.parse_args()

    sysinfo = probe_system()
    print("System:")
    print(sysinfo.pretty())

    if args.detection and args.landmarks:
        stem = args.detection.name.replace("_detections.geojson", "")
        spec = (stem, args.detection, args.landmarks, args.image)
    else:
        if not args.detections_dir.is_dir():
            sys.stderr.write(f"Detections dir not found: {args.detections_dir}\n")
            return 2
        img_dir = args.image_dir if args.image_dir.is_dir() else None
        found = find_first_specimen(args.detections_dir, args.landmarks_dir, img_dir)
        if found is None:
            sys.stderr.write("No matched specimen found; pass --detection / --landmarks explicitly.\n")
            return 2
        spec = found

    print(f"\nCalibrating against: {spec[0]}")
    cli_extra = args.cli_extra.split() if args.cli_extra else []
    stat = calibrate(spec, cli_extra)
    print(
        f"  wall={stat.wall_s:.1f}s  peak_rss={stat.peak_rss_gb:.2f} GiB  "
        f"status={'OK' if stat.success else f'rc={stat.returncode}'}"
    )

    rec = recommend(stat, sysinfo)
    print("\nRecommendation:")
    for k, v in rec.items():
        print(f"  {k:>22}: {v}")
    print(f"\n→ Run identify-features with: --workers {rec['recommended_workers']}")

    if args.json:
        args.json.write_text(
            json.dumps(
                {
                    "system": asdict(sysinfo),
                    "calibration": asdict(stat),
                    "recommendation": rec,
                },
                indent=2,
            )
        )
        print(f"JSON written to {args.json}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
