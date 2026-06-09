# liveSettings — build status

Live vein-overlay preview embedded in the TRACE Advanced Settings dialog
(TODO #14). Built and tested in-session 2026-05-29. Design rationale is in
`IMPLEMENTATION_SPEC.md`; this file records what actually shipped and the
measured performance.

## What's built

```
liveSettings/live_tune/
  session.py             LiveTuneSession — headless tiered-cache orchestrator + FIELD_TIER
                         + _effective() (preview-scale µm/px adjustment)
  worker.py              LiveTuneWorker  — serialized QThread, coalescing (latest-wins),
                         caches full-res bundle for cheap rescale
  input_loader.py        raw-image preprocessing → InputBundle (load_from_raw_image,
                         forwards wing-iso/rotation/expand/rescale knobs);
                         scale_bundle() (downscale image + geometry for preview).
                         load_from_geojsons remains as the internal S1 parser + a
                         test entry point, but is no longer a user-facing input mode.
  preview_pane.py        LivePreviewPane — raw-image picker + Preview-res combo +
                         zoomable overlay + Refresh/Save; preprocessing re-runs
                         (debounced) when a preproc setting changes
  dialog_integration.py  attach_live_preview(dialog) — layout surgery + widget wiring
                         + _build_preproc_getter (reads dialog + main-window preproc state)
  __init__.py            public exports
liveSettings/tests/
  test_session.py            13 unit tests (tier selection, idempotence, intervein)
  test_pane_smoke.py         offscreen Qt: load → tier B → tier D → intervein
  test_dialog_integration.py offscreen Qt against the REAL PipelineConfigDialog
  test_preview_scale.py      7 unit tests for the reduced-resolution preview
  test_preproc_wiring.py     6 tests: kwarg-bind vs real process_single_image,
                             forwarding, rescale-factor µm/px, preproc-getter
```

## 2026-05-29 rework: raw-image-only + preprocessing affects the image

Per user: preprocessing steps (wing isolation, rotation, rescale) change the
*image*, so a pre-made GeoJSON can't reflect a preproc-setting change. The
"From GeoJSONs" input mode was **removed** — the pane is raw-image-only.

- `load_from_raw_image` now forwards `wing_model_dir`, `wing_expand_fraction`,
  `do_rotation`, `rotation_mirror_correct`, `target_um_per_px` to
  `process_single_image` (names bind-checked against the real signature in
  `test_preproc_wiring.py` — this is the bug class that bit us before).
- Preproc settings reachable in the dialog (wing model, wing-expand) trigger a
  **debounced preprocessing re-run** (700 ms; the DL pass can't be reduced-res'd).
  Models stay cached (`predictor_cache`/`model_cache`), so only the forward
  passes re-run.
- Rotation / isolation-enable / rescale target are read from the **main window
  at re-run time** (per user's "read main-window values at open" choice), via
  `_build_preproc_getter`, defensively (missing attr → safe default).
- All Qt + unit tests updated/passing (smoke + dialog tests inject a GeoJSON
  loader directly into the worker, since the DL path can't run headlessly).

Integration: `TRACE/settings_dialog.py` `PipelineConfigDialog.__init__` calls
`attach_live_preview(self)` inside a guarded try/except — if anything fails the
dialog works exactly as before, without a preview. The pane is collapsed behind
a "Show live preview ▸" toggle, so opening Settings is unchanged until the user
opts in.

## Test results (all passing, real specimen 0003 data)

- `test_session.py`: **13/13** (≈240 s — each test runs the real pipeline).
- `test_pane_smoke.py`: load=12 veins, tier B `B_trace` only (no skeleton), tier D
  render-only, intervein tier C succeeds, BGR→QPixmap 5440×3648 ✓.
- `test_dialog_integration.py`: pane mounts, `dialog.get_config()` still works,
  editing the real `snap_radius_um` widget → tier B, `smooth_sigma` → tier A,
  worker stops cleanly on dialog close ✓.

The two correctness guarantees from the spec are verified:
- **FIELD_TIER covers every PipelineConfig field** (asserted at import + test).
- **Tier B is idempotent** — A→B→A round-trips to a byte-identical vein set, and
  the cached pristine skeleton's node count is unchanged after a trace. This is
  the `anchor_landmarks` in-place-mutation trap; the pristine-deepcopy discipline
  holds.

## Measured performance (specimen 0003, 5440×3648 — a LARGE wing)

| Tier | Stage | Cost | Live? |
|------|-------|------|-------|
| A | build_skeleton_graph | ~4.0 s | on-commit / 500 ms debounce |
| B | anchor 7ms · axis 0ms · **trace 5.1 s** · tissue 174ms | **~5.3 s** | 200 ms debounce |
| C | split + name (intervein) | ~2 s | manual "Refresh intervein" button |
| D | render_overlay | ~30 ms | live |
| — | deepcopy(skeleton) | 29 ms | (negligible — graph is ~16 nodes) |

### The honest headline

- **Appearance edits (opacity, colours, show/hide) are genuinely live** (~30 ms).
- **Tracing and skeleton edits are NOT per-keystroke live on a large wing** — each
  recompute is seconds. The debounce makes this "one slow recompute after you stop
  dragging," not stutter-per-tick. It is still dramatically better than the old
  workflow (full batch run per tweak), and caching the skeleton means a Tier-B
  edit costs ~5 s instead of ~9 s.
- The spec originally assumed Tier B was "sub-second." That was wrong: the deepcopy
  is cheap, but `trace_veins_from_landmarks` does full-resolution raster work
  internally (~5 s here). Corrected in the spec/docstrings.

### The lever (BUILT 2026-05-29): reduced-resolution preview

Both Tier A and Tier B scale with image area, so the preview runs the stages at a
downscaled resolution. A "Preview res" combo in the pane offers Full / Half /
Quarter (default **Half**); the real batch run always uses full resolution, so
this only trades preview sharpness for speed.

How it works:
- `input_loader.scale_bundle(bundle, s)` downscales the base image (`cv2.resize`
  INTER_AREA), all vein/intervein polygons + the wing outline + landmark points
  (`shapely.affinity.scale`), and records `preview_scale=s`.
- `LiveTuneSession._effective(config)` divides `um_per_px` by `s` so micron
  thresholds (px = µm / um_per_px) shrink by the same factor as the image.
  Vein-width-relative thresholds need no adjustment — they auto-scale with the
  smaller `median_vein_width_px` measured from the downscaled skeleton. When
  `um_per_px is None`, the vw fallbacks carry everything and `_effective` is a
  no-op.
- The worker caches the FULL-resolution bundle, so changing the resolution combo
  (`request_rescale`) re-scales from that cache — no reload / no re-preprocess.

Measured (specimen 0003): Tier B `anchor+trace+tissue` drops from **~5.3 s
(full)** to **~1.0 s (half)** — a ~5× speedup (≈4× from area + smaller graph),
matching the prediction. Quarter-res is ~0.3 s. Verified end-to-end in
`test_dialog_integration.py` (real dialog widget edit → tier B at 1026 ms).

## View modes (skeleton / traced / final)

A "View" selector chooses which pipeline product the preview shows:
- **Wing graph (skeleton)** — end of skeletonization (Tier A). Graph edges +
  degree-colored nodes. **Needs no tracing**, so Wing-Graph tuning skips the
  ~1–5s Tier B entirely; Tracing/Intervein changes are no-ops in this view
  (the trace is *deferred*, run lazily when a tracing view is next shown).
- **Traced veins + landmarks** — end of vein tracing (Tier B). Labeled vein
  centerlines (reusing `render_overlay`, veins only) + snapped landmarks. The
  landmark marker uses the anchored graph node (`lm.snapped_node`), not the
  raw `lm.point`, with a thin tie-line when they differ.
- **Final output** — the original full overlay (veins + intervein regions).

Implemented as a per-view recompute cap in `LiveTuneSession.update(view=...)`
(`_VIEW_MAX_TIER`); a `_veins_dirty` flag defers Tier-B work the active view
doesn't need. Renderers live in `live_tune/preview_render.py` (drawing patterns
lifted from skeleton.py `_DebugDumper` / vein_tracer.py `_TracerDumper`). The
worker carries the active view (`set_view`); the pane gates the display
checkboxes + intervein refresh to the final view only. Tests:
`test_views.py` (7, headless) + a view-switch segment in `test_pane_smoke.py`.

## Not done / deferred

- Smaller test wings would make `test_session.py` much faster; left on 0003 for
  realism.
- Hands-on GUI verification by the user (offscreen Qt tests pass; no human has
  driven the real window yet).
- Not committed — no git changes were requested.

## Files touched outside liveSettings/

- `TRACE/settings_dialog.py` — added a guarded `attach_live_preview(self)` call at
  the end of `PipelineConfigDialog.__init__` (≈18 lines). This is the only edit to
  the parallel session's territory; everything else is self-contained in
  `liveSettings/`.
