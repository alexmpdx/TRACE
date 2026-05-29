# Live Settings — Embedded Vein-Tuning Preview (TRACE TODO #14)

> **Handoff spec for the implementing session.** This session plans; you implement.
> Produced 2026-05-29; **architecture revised 2026-05-29** to embed the live preview
> directly in the Advanced Settings dialog (was: standalone window). No app code written yet.

---

## 1. Primer (read this first)

Add a **live vein-overlay preview embedded in the Advanced Settings dialog**. As the user
edits **Wing Graph** and **Tracing** parameters, a shared preview pane shows the
`identifyFeatures` vein overlay updating in **near-real-time**, instead of running a full
batch to evaluate one tweak.

**This replaces the earlier "standalone window" plan.** Embedding in the dialog is strictly
better because:

1. **No widget extraction.** The param widgets already live in the dialog
   (`TRACE/settings_dialog.py`). A standalone window would have required pulling them into a
   shared builder so two UIs could share them — pure overhead. Gone.
2. **No "Apply to Settings" plumbing.** The dialog *is* the settings. Whatever you tune is
   already the dialog's config; clicking **OK** returns it via the existing `get_config()`.
   No push-back, no drift between two UIs.
3. **The tier dependency maps onto tab order.** Wing Graph → Tracing → Intervein **is**
   Tier A → B → C (see §2). On the Tracing tab the skeleton is already built from the current
   Wing Graph values, so tuning tracing re-runs only the cheap downstream stages. This is the
   "tracing/intervein need a set wing graph" problem the user raised — solved structurally.

### Shape of it

- The dialog becomes **two-column**: the existing `QTabWidget` on the left, **one shared
  `LivePreviewPane`** on the right that persists across tab switches. *One* preview, *one*
  cache, *one* worker — **not** a separate canvas per tab (that would duplicate image + compute).
- The preview is **opt-in / collapsible**: default collapsed behind a "Show live preview ▸"
  toggle so users who just want to set params don't pay for a sample load + skeleton build.
  Expanding prompts for a sample (raw image *or* existing GeoJSONs — both supported).
- **Per-tab trigger policy** on the same pane:
  - **Wing Graph (Tier A, seconds):** recompute on **commit** (`editingFinished`/`sliderReleased`)
    or ~500ms debounce; show "Rebuilding skeleton…" + dim preview.
  - **Tracing (Tier B, sub-second):** **live**, ~200ms debounce.
  - **Intervein (Tier C, slow):** **manual "Refresh preview" button only** — never on drag.
    Honors the user's slowness concern while still letting them see intervein on demand.
- **Save as preset…** remains useful (write JSON to `TRACE/presets/`); **Apply-to-Settings is
  unnecessary** (OK already commits the config).

### v1 scope

Veins live (Wing Graph + Tracing). Intervein tab is **wired to the same pane with a manual
Refresh button** (Tier C on-demand), not the live debounce loop.

---

## 2. The cache-tier model (unchanged — this is what makes it responsive)

`identify_wing()` is a linear stage sequence
(`identifyFeatures/identify_features/controllers/pipeline.py:121-236`). A changed parameter
invalidates **its tier and everything downstream**; upstream stays cached.

| Tier | Stages | Tab | Cost | Trigger |
|------|--------|-----|------|---------|
| **A — Skeleton** | `build_skeleton_graph` | Wing Graph | **seconds** | on-commit / 500ms |
| **B — Trace** | `anchor_landmarks` → `compute_wing_axis` → `trace_veins_from_landmarks` → `assign_vein_tissue_polygons` | Tracing | sub-second–~1s | live, 200ms |
| **C — Intervein** | `split_merged_intervein_polygons` → `name_intervein_regions` | Intervein | slow | **manual Refresh** |
| **D — Render** | `render_overlay` | Appearance group | ms | instant |

```
S1 parse      load_detection_geojson / load_landmarks_geojson / _compute_wing_outline   (l.152-157)
              + image read → image_shape                                                  (l.159-172)
S2 skeleton   skel = build_skeleton_graph(vein_polys, image_shape, config)               (l.176)   ── TIER A
S3 anchor     anchor_landmarks(skel, landmarks, config)   ← MUTATES skel & landmarks      (l.180)   ┐
S4 axis       wing_axis = compute_wing_axis(landmarks)                                    (l.183)   │ TIER B
S5 trace      veins = trace_veins_from_landmarks(skel, landmarks, wing_outline,                     │
                                                 config, wing_axis=wing_axis)            (l.187)   │
S6.3 tissue   assign_vein_tissue_polygons(veins, skel.median_vein_width_px, config,                 │
                                          wing_outline)   ← MUTATES veins                 (l.194)   ┘
─ skip_intervein_regions gate ─                                                          (l.196)
S6.1 split    split_merged_intervein_polygons(...)                                       (l.205)   ┐ TIER C
S6.2 name     regions = name_intervein_regions(...)                                      (l.216)   ┘
```

> ⚠️ **Critical correctness trap.** `anchor_landmarks` (S3) **mutates `skel` and `landmarks`
> in place**; `assign_vein_tissue_polygons` mutates `veins` in place. **Tier B must always
> start from a pristine deepcopy** of the cached Tier-A skeleton and of the raw landmarks —
> never anchor the cached skeleton directly, or repeated re-tracing drifts. Idempotence test
> in §7.

> **Invalidation is driven by `FIELD_TIER` (§5), not by the active tab.** The tab only sets
> the *trigger policy* (how eagerly to recompute). A field that lives on one tab but feeds an
> earlier tier (e.g. `um_per_px` feeds every µm→px conversion → Tier A) must still bust the
> correct tier. Always diff the changed field through `FIELD_TIER`.

---

## 3. Verified hookpoints (absolute-from-repo-root)

Repo root: `/Users/alexmurphy/claude_scripts/mapThemVeins`

### 3.1 Stage functions — importable directly from submodules (no `__init__` re-exports)

| Function | Import | Signature (verified) |
|----------|--------|----------------------|
| `build_skeleton_graph` | `from identify_features.models.skeleton import build_skeleton_graph` | `(vein_polygons, image_shape, config=None, debug_dir=None) -> SkeletonGraph` — `skeleton.py:113` |
| `anchor_landmarks` | `from identify_features.models.landmark_anchor import anchor_landmarks` | `(skel_graph, landmarks, config=None) -> dict` (mutates args) — `landmark_anchor.py:36` |
| `compute_wing_axis` | `from identify_features.models.wing_axis import compute_wing_axis` | `(landmarks) -> Optional[WingAxis]` — `wing_axis.py:14` |
| `trace_veins_from_landmarks` | `from identify_features.models.vein_tracer import trace_veins_from_landmarks` | `(skel_graph, landmarks, wing_outline=None, config=None, wing_axis=None, debug_dir=None) -> list[VeinIdentification]` — `vein_tracer.py:125` |
| `assign_vein_tissue_polygons` | `from identify_features.models.intervein_splitter import assign_vein_tissue_polygons` | `(veins, median_vein_width_px, config, wing_outline=None) -> None` — `intervein_splitter.py:325` |
| `split_merged_intervein_polygons` | `from identify_features.models.intervein_splitter import split_merged_intervein_polygons` | `(intervein_polys, veins, wing_outline, image_shape, median_vein_width_px, config, debug_out=None, debug_base_image=None) -> list[Polygon]` — `intervein_splitter.py:38` |
| `name_intervein_regions` | `from identify_features.models.intervein_namer import name_intervein_regions` | `(intervein_polys, veins, landmarks, config, median_vein_width_px=0.0, wing_outline=None, wing_axis=None) -> list[InterveinRegion]` — `intervein_namer.py:64` |
| `load_detection_geojson` / `load_landmarks_geojson` / `_compute_wing_outline` | `from identify_features.models.geojson_io import ...` | parse inputs (S1) |
| `render_overlay` | `from identify_features.views.overlay import render_overlay` | `(base_image, veins, regions, show_vein_tissue=False, show_veins=True, show_regions=True, vein_color_overrides=None, region_color_overrides=None, vein_opacity=1.0, intervein_opacity=0.2) -> np.ndarray` — `overlay.py:125` |
| `imread_any` | `from identify_features.utils.psd_loader import imread_any` | image read (BGR ndarray) |

`SkeletonGraph` / `PipelineConfig` from `identify_features.models.datatypes` / `identify_features.config` (config dataclass `config.py:11-202`).

### 3.2 The dialog — where to embed (this is the main integration surface)

`TRACE/settings_dialog.py`, class `PipelineConfigDialog`:
- UI build / outer layout: `226-429`; bottom button box: ~`375-429`.
- **Tabs: Wing Graph `1024-1115`, Tracing `1117-1164`, Intervein `1166-1186`.**
- Widget factories (`_add_float`, `_add_opt_float`, `_add_enum_list`, `_add_bool`,
  `_add_float_list`): `1191-1304`. **Dispatch table `self._widgets`: `276`** (KIND consts `229-237`).
  ← this table is the key: it holds **every param widget**, so you can connect them all to one
  "param changed" slot in a loop (see §4.2).
- `get_config() -> PipelineConfig`: `304-350` ← the preview calls this on each change.
- `_load_from_config(config)`: `1309-1350` (used by preset apply; preview doesn't need it).
- Preset apply / combo: `1400-1423` / `375`.

`TRACE/config_io.py`: `config_to_dict`/`config_from_dict` (`27-118`), `_ENUM_FIELDS` (`21-24`).
`TRACE/presets_loader.py:37-53` loads `*.json`. Existing presets in `TRACE/presets/` are partial
override dicts with enum-lists as value strings — match that format for Save-as-preset.

> The dialog is **modal**. A worker `QThread` inside it is fine; ensure it's stopped on
> `accept()/reject()/closeEvent`. Make the preview **optional** so existing callers/flows that
> just open Settings are unaffected (default collapsed, dialog stays compact).

### 3.3 Input — both paths (raw image OR existing GeoJSONs)

`preprocessing/pipeline.py`:
- `process_single_image(image_path, output_dir, landmark_checkpoint=None, segmentation_model_dir=None, stages=(True,True,True), ...) -> PipelineResult` (`578-911`); `PipelineResult` (`539-564`)
  carries `landmarks_geojson_path` + `segmentation_geojson_path` — exactly what `identify_wing` consumes.
- DL stages `run_landmarks` (`243`) / `run_segmentation` (`495`) are the expensive ones; they
  cache via `predictor_cache` / `model_cache` dicts — **reuse the same dicts so models load once.**
  Device select `_auto_device()` (`567`). Model paths come from the dialog's `gui_state`.

### 3.4 Picker pattern to mirror

`measurementMaker/measurement_maker/embedded_picker.py:45` `LandmarkPickerWidget` (file pickers +
optional `landmarks_generator` callback); embedded by TRACE at `inline_panels.py:1327`. Reuse the
*pattern* for the sample chooser. Note it only generates *landmarks*; the live preview also needs
the *detection/segmentation* GeoJSON, so the raw-image path calls `process_single_image`, not the
picker's landmark-only generator. A zoomable `QGraphicsView`/`QLabel` is enough for the preview.

---

## 4. Implementation

### 4.1 Module layout (`liveSettings/`) — no standalone window

```
liveSettings/
  live_tune/
    __init__.py
    session.py       # LiveTuneSession (headless orchestrator) + FIELD_TIER + RenderResult
    worker.py        # QThread recompute worker (no Qt objects built off-GUI-thread)
    preview_pane.py  # LivePreviewPane(QWidget): sample picker + zoomable preview + status
                     #   + Refresh(intervein) + Save-as-preset buttons. Hosted by the dialog.
    input_loader.py  # resolve raw-image vs existing-geojson; preprocess-once + cache
  IMPLEMENTATION_SPEC.md
```

**Import plumbing** (per repo `CLAUDE.md`): whatever imports these must add to `sys.path`:
`<root>`, `<root>/HingeChopper`, `<root>/modelTOjson`, `<root>/identifyFeatures`,
`<root>/preprocessing`, `<root>/measurementMaker`. The TRACE GUI already sets most of these up in
`TRACE/run_gui.py`; verify `identifyFeatures` + `preprocessing` are present when the dialog imports
`live_tune` (import lazily inside the "Show live preview" handler to avoid loading heavy deps when
the preview is never opened).

### 4.2 Wiring the dialog to the pane

1. Add a collapsible right column hosting one `LivePreviewPane`. Toggle "Show live preview ▸"
   instantiates it lazily and asks for a sample.
2. **One change-signal for all params.** Iterate `self._widgets` (dispatch table at `276`) and
   connect each widget's change signal to a single slot:
   - FLOAT/INT spin → `valueChanged`; for **Tier A** fields prefer `editingFinished`/
     `sliderReleased` (commit-only) to avoid rebuilding the skeleton mid-drag.
   - BOOL check → `toggled`; ENUM_LIST → `itemChanged`; FLOAT_LIST line edit → `editingFinished`;
     OPT_* → both the checkbox `toggled` and the spin signal.
   - Slot body: `self.live_pane.on_config_changed(self.get_config(), changed_field=name)`.
3. The pane owns a `LiveTuneSession`; on `on_config_changed` it looks up `FIELD_TIER[changed_field]`,
   applies the active-tab debounce policy, and dispatches a recompute to the worker.
4. Tab-aware policy: connect the dialog's `QTabWidget.currentChanged` to the pane so it knows the
   current trigger policy (A=commit, B=live, C=manual). The Intervein tab shows a "Refresh preview"
   button that calls `session.compute_intervein(config)` and re-renders.
5. **OK already commits the tuned config** (existing `get_config()`); no apply step. Save-as-preset
   button → `config_io.config_to_dict(get_config())` → JSON in `TRACE/presets/<name>.json`.

### 4.3 `LiveTuneSession` (session.py) — headless, no Qt, unit-testable

```python
class LiveTuneSession:
    def set_input(self, base_image_bgr, vein_polys, intervein_polys,
                  landmarks_raw, wing_outline, image_shape, um_per_px) -> None:
        """Store pristine S1 results. Clears all tier caches."""

    def update(self, config) -> RenderResult:
        """Diff config vs last; recompute from the lowest invalidated tier among A/B/D
        (NOT C). Returns overlay ndarray + tier-that-ran + timings."""

    def compute_intervein(self, config) -> list:   # InterveinRegion
        """On-demand Tier C from cached veins; used only by the Intervein Refresh button."""
```

Caches: `_pristine_skel` (Tier A; **never anchored**), `_veins`, `_anchored_landmarks`,
`_wing_axis`, `_last_config`, plus pristine S1 inputs. Recompute discipline:

```python
# Tier A:
self._pristine_skel = build_skeleton_graph(deepcopy(self._vein_polys), self._image_shape, config)
# Tier B (ALWAYS from pristine copies — anchor mutates!):
skel = deepcopy(self._pristine_skel); lms = deepcopy(self._landmarks_raw)
anchor_landmarks(skel, lms, config)
axis  = compute_wing_axis(lms)
veins = trace_veins_from_landmarks(skel, lms, self._wing_outline, config, wing_axis=axis)
assign_vein_tissue_polygons(veins, skel.median_vein_width_px, config, self._wing_outline)
self._veins, self._anchored_landmarks, self._wing_axis = veins, lms, axis
# Tier D:
overlay = render_overlay(self._base_image, self._veins, regions_or_[], **appearance)
```

> Measure `deepcopy(SkeletonGraph)` (networkx graph + numpy arrays); expected low-tens-of-ms.
> If the read-only `distance_map`/`skeleton`/`vein_mask` arrays make it slow, deep-copy the
> graph but shallow-copy those arrays (Tier B never writes them).

### 4.4 `FIELD_TIER` — central mapping (build explicitly; completeness-tested in §7)

Recompute from `min(tier of each changed field)`. Unknown changed field → default **Tier A** (safe).

- **TIER A:** `skeleton_methods`, `smooth_sigma`, `enable_basic_prune`,
  `enable_small_fragment_removal`, `min_component_edge_fraction`, `prune_methods`,
  `prune_min_length_um`, `prune_min_length_vein_widths`, `final_stub_vein_widths`,
  `junction_merge_vein_widths`, `prune_radius_ratio_threshold`, `prune_scale_sigmas`,
  `prune_single_scale_sigma`, `collinear_min_angle`, all `bridge_*`/`bridge2_*`/`bridge3_*`,
  `um_per_px`.
- **TIER B:** `snap_radius_um`, `snap_radius_vw`, `departure_sample_um`, `departure_sample_vw`,
  `tangent_continuity_max_angle`, `merge_max_gap_um`, `distal_landmark_search_vw`,
  `soft_landmark_reach_metric`, `costa_min_in_band_fraction`, `costa_propagation_max_distance_vw`,
  `crossvein_min_angle`, `crossvein_max_length_frac`, `crossvein_min_length_vw`,
  `crossvein_max_length_vw`, `synthesize_missing_crossveins`, `ectopic_min_length_um`,
  `ectopic_min_length_vw`.
- **TIER C (manual refresh only):** `skip_intervein_regions`, `vein_buffer_vw`,
  `adjacency_min_length_vw`, `max_merge_size`, `intervein_split_h_vw`,
  `intervein_split_reseed_min_area_um2`, `intervein_split_vein_barrier_vw`,
  `intervein_split_wing_buffer_vw`. *(`vein_buffer_vw` also feeds `assign_vein_tissue_polygons`
  in Tier B; v1 is centerline-focused so keeping it Tier-C-only is fine — document the choice.)*
- **TIER D (render):** `vein_opacity`, `intervein_opacity`, `vein_colors`, `region_colors`, +
  UI-only `show_veins`/`show_regions`/`show_vein_tissue`.

### 4.5 Threading / debounce (worker.py)

- All recompute off the GUI thread in one worker `QThread`; stage fns are pure
  python/numpy/shapely/networkx — safe off-thread (build no Qt objects there). Return the overlay
  ndarray; convert to `QPixmap` on the GUI thread.
- Stage fns aren't cancellable → if a new request arrives mid-run, keep only the **latest** pending
  config and run it after the current finishes (drop intermediates).
- Per-tier status label ("rebuilding skeleton…", "re-tracing…", "intervein…").

### 4.6 Input resolution (input_loader.py)

If detection + landmarks GeoJSONs are supplied/pointed-at → use directly (no models). Else run
`process_single_image` once into a temp dir with shared `predictor_cache`/`model_cache`; cache the
two output paths keyed by image. Show progress for this one slow step; **never** re-run it on param
changes.

---

## 5. Test checklist

- [ ] `FIELD_TIER` covers **every** `PipelineConfig` field:
      `set(FIELD_TIER) == {f.name for f in fields(PipelineConfig)} - APPEARANCE_FIELDS` (catches new params).
- [ ] `vein_opacity` change → **Tier D only** (skeleton + trace counters unchanged).
- [ ] `smooth_sigma` → **Tier A** rebuild.
- [ ] `snap_radius_um` → **Tier B**, skeleton **not** rebuilt, overlay changes.
- [ ] `departure_sample_um` → **Tier B**.
- [ ] **Idempotence / no-mutation-leak:** A→B→A round trip yields byte-identical overlay
      (guards the `anchor_landmarks` in-place-mutation trap).
- [ ] Intervein params never trigger live recompute; only the Refresh button runs Tier C, from cached veins.
- [ ] Raw-image mode: across many param changes, DL models load **exactly once**; existing-geojson mode loads none.
- [ ] No-scale image (`um_per_px=None`): runs via `_vw` fallbacks; banner shown.
- [ ] Pathological param combo that raises inside a stage → worker catches, last good overlay stays, error banner, no crash.
- [ ] **Dialog regression:** opening Advanced Settings with the preview collapsed behaves exactly as before; OK still returns the edited config; existing callers unaffected.
- [ ] Worker thread is stopped/joined on dialog accept/reject/close (no dangling QThread).
- [ ] Save preset → load preset reproduces the same config (enum-lists round-trip as value strings).

---

## 6. Implementation phases

1. **Headless `LiveTuneSession`** + `FIELD_TIER` + caching/invalidation + deepcopy discipline.
   Unit-test against a sample's *existing* GeoJSONs (fast, no models) — covers most of §5.
2. **input_loader**: existing-geojson path + raw-image `process_single_image` (cached, models-once).
3. **`LivePreviewPane`** widget: sample picker, zoomable preview, status, worker thread,
   per-tab debounce, Intervein Refresh button, Save-as-preset.
4. **Embed in `PipelineConfigDialog`**: collapsible right column, connect all `self._widgets` to one
   change slot, wire `currentChanged` for trigger policy. Verify dialog regression tests pass.

---

## 7. Edge cases

- Preprocessing failure / low-confidence landmark gate → surface error in the pane; allow retry or
  manual GeoJSON selection; never crash the dialog.
- Switching the sample image mid-session → reset all tier caches, re-run S1.
- `prune_scale_sigmas` is a comma-separated float list (`_KIND_FLOAT_LIST`) — validate parse before recompute.
- `skeleton_methods` / `prune_methods` enum-lists currently allow one value each — preserve that.
- OPT_FLOAT/OPT_INT fields: `None` = "auto"; round-trip correctly through Save-as-preset.
- Large images: compute on full-res `image_shape` (skeleton is sized to it), display a scaled
  preview; consider an optional "preview at reduced res" toggle if Tier A feels too slow.
- Preview opened with no sample yet → pane shows a "Load sample to preview" placeholder, params still editable.
