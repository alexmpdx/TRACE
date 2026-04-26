# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Purpose

Standalone calibration tool that recommends a safe `--workers` count for `identify-features` (TRACE Stage 2) on the current machine. Measures peak RSS during a one-specimen run, divides available RAM by that, and caps by physical core count.

## Running

```bash
# Auto-pick the first matched specimen from sibling identifyFeatures dirs:
python recommend_workers.py

# Calibrate against a specific specimen:
python recommend_workers.py --detection PATH --landmarks PATH --image PATH

# Persist the report:
python recommend_workers.py --json calib.json

# Forward extra flags into the identify-features subprocess used for calibration:
python recommend_workers.py --cli-extra "--preset fast"
```

Default specimen sources (overridable via `--detections-dir`, `--landmarks-dir`, `--image-dir`):
`../identifyFeatures/{geojsons, LandmarkLocator_output, OGpics}`.

Requires `psutil`. Does **not** require `identify-features` to be `pip install`-ed — the script prepends `../identifyFeatures` to `PYTHONPATH` for the calibration subprocess, so a stale or broken editable install is OK.

## Calibration model

```
recommended = min(physical_cores, floor((available_ram - RAM_RESERVE_GB) / (peak_rss * SAFETY_FACTOR)))
```

Constants live at the top of `recommend_workers.py`:
- `RAM_RESERVE_GB = 2.0` — held back for OS / other apps
- `SAFETY_FACTOR = 1.3` — multiplier on observed peak
- `SAMPLE_INTERVAL = 0.1` — RSS poll interval (seconds)

The recommendation dict reports `binding_constraint` ("CPU (physical cores)" vs "RAM (per-worker peak)") so callers can see which side was the bottleneck.

## Module + integration

`recommend_workers.py` is dual-purpose: a CLI **and** an importable module. Public surface:
- `probe_system() -> SystemInfo`
- `find_first_specimen(det_dir, lm_dir, img_dir) -> tuple | None`
- `calibrate(spec, cli_extra) -> CalibrationStats`  (spec = `(stem, det_path, lm_path, img_path)`)
- `recommend(stat, sysinfo) -> dict`

`TRACE/calibrate_workers.py` imports these to compose Stage 1 (preprocessing) with Stage 2 calibration; the TRACE CLI exposes the result as `--calibrate-workers PATH` and the GUI exposes it as a "Calibrate" button on the Settings dialog's General tab. When changing this module's API, update those two consumers.

## Subprocess + monitoring

Calibration spawns `python -m identify_features.cli <det> <lm> <img>` (output to a `TemporaryDirectory`) and runs `_Monitor` in a background thread that walks `parent + recursive children` and sums `memory_info().rss` every `SAMPLE_INTERVAL` seconds. The peak across all samples is the per-worker estimate. The threaded sampler is necessary because identify-features itself uses a `ProcessPoolExecutor` — children must be enumerated to capture true peak.

Note: TRACE Stage 2 actually uses `ThreadPoolExecutor`, not processes, so the per-worker peak measured here slightly **overestimates** memory pressure when extrapolated to TRACE workers (baseline interpreter state isn't replicated across threads). That's the right side to err on for a "safe parallelism" recommendation.
