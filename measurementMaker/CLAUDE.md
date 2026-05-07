# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Purpose

User-defined landmark distance measurements bolted onto the TRACE pipeline. The user picks two landmarks once on a sample wing (via a napari canvas embedded in the TRACE settings dialog) and the straight-line distance between them is computed for every wing in a TRACE batch run, appended as `user_distance_<label>_{px,um}` columns in the consolidated `measurements.csv`.

## Relationship to other modules

- Consumed by **TRACE only** — preprocessing/identifyFeatures know nothing about it.
- TRACE/run_gui.py and TRACE/run_cli.py add `measurementMaker/` to `sys.path` so `import measurement_maker` works.
- `TRACE/settings_dialog.py` exposes the "Custom Distances" tab (right after "General") whose body **is** a `measurement_maker.LandmarkPickerWidget`.
- `TRACE/pipeline.py` calls `measurement_maker.augment_csv_with_user_distances` after `export_csv_batch` finishes.

## Layout

```
measurementMaker/                # outer dir (PascalCase, sibling-module convention)
├── pyproject.toml               # declares napari[pyqt5] dep
└── measurement_maker/           # actual import name (snake_case)
    ├── __init__.py              # public API + lazy `open_pair_picker` entry point
    ├── types.py                 # LandmarkPair, safe_label, pairs_{to,from}_dicts
    ├── distance.py              # load_landmarks_from_geojson, compute_pair_distance_px
    ├── csv_augment.py           # augment_csv_with_user_distances
    └── embedded_picker.py       # LandmarkPickerWidget (heavy import — napari)
```

Install with `pip install -e measurementMaker` from the project root — pulls napari (~100MB+ first time).

## Architecture

**Lazy napari import.** napari is heavy and only needed for the interactive picker. `__init__.py` resolves `LandmarkPickerWidget` via `__getattr__` so headless callers (e.g. `TRACE/pipeline.py` running on a server) don't pay the import cost. Pure-logic modules (`types`, `distance`, `csv_augment`) never touch napari.

**Embedded canvas model.** `LandmarkPickerWidget` is a plain `QWidget` that lazy-creates a `napari.Viewer(show=False)` on first wing load, then re-parents `viewer.window.qt_viewer` (napari's QtViewer canvas) into a placeholder slot inside the widget. Subsequent loads call `viewer.layers.clear()` and add fresh layers — the canvas itself stays mounted in the tab. The widget emits `pairs_changed(list[LandmarkPair])` whenever the configured pair list mutates, so the hosting dialog can mirror state without polling.

**TRACE state lives in TRACE.** Configured pairs are stored as `self._user_landmark_distances: list[dict]` on `TraceWindow` (TRACE/gui.py), persisted via QSettings (`user_landmark_distances_json` key), passed to the settings dialog as a constructor parameter and read back via `get_user_landmark_distances()`, then handed to `trace_folder(user_landmark_distances=...)`. **Do not** add this field to `identifyFeatures.PipelineConfig` — that library stays focused on analysis, not user-facing pipeline config.

**CSV augmentation is post-hoc.** After `export_csv_batch` writes `measurements.csv`, `TRACE/pipeline.py` calls `augment_csv_with_user_distances(csv_path, specimen_landmarks, pairs, um_per_px)` to read the CSV back, append columns, and rewrite. This avoids pushing TRACE-specific shape into identifyFeatures' export path. Per-wing landmark GeoJSON paths are collected in `_analyze_one` into `user_dist_landmark_paths: dict[stem, Path]` for the augmenter to consume.

## Landmark name conventions

Pairs reference landmarks by **raw GeoJSON `properties.classification.name`** strings — `"DTip"`, `"L1-Rs"`, `"ACV.a"`, `"alula notch"`, etc. — **not** the snake_case internal names that LandmarkLocator's `dataset.py` uses (`dtip`, `l1_rs`, `acv_a`, `alula_notch`). This matches how `identifyFeatures.views.csv_export` looks landmarks up (`landmarks.get("L1-Rs")`). `load_landmarks_from_geojson` returns the dict keyed by raw names.

## Edge cases handled

- **Missing landmark on a particular wing** (e.g. low-confidence drop by LandmarkLocator) → blank cell for that pair on that wing, logged at WARNING. Other wings still get values.
- **No scale (`um_per_px is None`)** → only `_px` columns, no `_um`.
- **Duplicate user labels** → `_dedupe_suffix` appends `_2`, `_3`, ... so columns never collide.
- **Pair label with spaces/punctuation** → `safe_label` collapses non-alphanumerics into `_` (so `"wing span!"` becomes `wing_span` for the column suffix).

## Code style

Matches the rest of the project: black + isort + flake8 at 120 chars. No new pre-commit hooks; the top-level config covers this directory.
