# ETA fix — spec

The "~Xh Ym until pipeline finishes" label under the progress bar is wildly wrong on large batches. Observed 13h 18m at 23% for a 2573-image run that's actually running at ~4h 20m pace (200 images in 21 min). ETA keeps climbing as the run progresses.

Two independent bugs are stacking. **Bug #1 is the dominant one; fix it first.** Bug #2 is a smaller but real drift that will still bite chunked runs after #1 is fixed.

## Reproduction

- `trace_version=0.2.19` (should also repro on ≥0.2.16 — anything after the chunking refactor).
- 2000+ image batch with `outputs=[csv]` and `csv_measurement_groups=[cv_ratio]`. Wing isolation on (it doesn't matter for the bug, but matches the reported case).
- ETA displays 3–5× the true remaining time and grows across chunk boundaries instead of shrinking.

## Bug #1 — GUI never sees the runtime `skip_intervein_regions` flip

**Impact:** Stage 2 wall-time weight is 83% when it should be 13% for `csv`-only-with-cv-ratio runs. That inflates `total_predicted` in the throughput branch by ~3×.

**Trace:**
- `TRACE/gui.py:4005` calls `compute_progress_weights(..., skip_intervein_regions=getattr(self.config, "skip_intervein_regions", False))` at run start. `self.config.skip_intervein_regions` is `False` (the dataclass default).
- `TRACE/pipeline.py:633` flips `config.skip_intervein_regions = True` inside `_run()` when no output actually consumes intervein regions (i.e. `outputs={csv}` with `csv_measurement_groups={cv_ratio}`). The GUI's copy of the flag is never updated.
- Correct weights when `skip_intervein_regions=True` in `compute_progress_weights` (pipeline.py:296): `s1=17, s2=13 → normalized (0.567, 0.433)` instead of the buggy `(0.17, 0.83)`.

**Numeric proof at t=6 min after chunk 1 preprocessing ends** (100 completions, throughput 0.278 img/s, `_stage_total=2573`):
- Buggy `stage1_share=0.17`: `full_stage1 = 2573/0.278 = 154 min`, `total_predicted = 154/0.17 = 906 min ≈ 15 h`.
- Correct `stage1_share=0.567`: `total_predicted = 154/0.567 ≈ 272 min ≈ 4 h 32 m`. **Matches the observed rate.**

**Fix design:**
- Extract the intervein-need logic (currently inline in `pipeline.py:625-633`) into a shared helper in `TRACE/pipeline.py`. Suggested signature:
  ```python
  def resolve_effective_config(config, outputs, csv_measurement_groups) -> PipelineConfig:
      """Return a copy of `config` with `skip_intervein_regions` set to the
      value the pipeline will actually use at run time."""
  ```
  It should reproduce the exact logic at pipeline.py:625-633:
  ```python
  csv_needs_intervein = "csv" in outputs and "intervein_areas" in csv_measurement_groups
  _, gj_writes_regions = _geojson_content_wanted(outputs, csv_measurement_groups)
  geojson_needs_intervein = "geojson" in outputs and gj_writes_regions
  always_intervein = outputs & (_INTERVEIN_DEPENDENT_OUTPUTS - {"csv", "geojson"})
  skip = not (always_intervein or csv_needs_intervein or geojson_needs_intervein)
  ```
  Return a `dataclasses.replace(config, skip_intervein_regions=skip)`. Keep the assignment in `_run()` too but derive it from the same helper so the two paths can't diverge.
- At `TRACE/gui.py:4004`, call `resolve_effective_config` first with the currently selected outputs + csv measurement groups, then pass `resolved.skip_intervein_regions` into `compute_progress_weights`. Also store the resolved config so the worker path uses the same value (either overwrite `self.config` or pass the resolved copy explicitly into `_run_pipeline`).
- Add a one-line comment at pipeline.py:633 noting the flip now goes through `resolve_effective_config` and must stay in sync with what the GUI predicted.

## Bug #2 — Chunked pipeline: `stage_elapsed` bleeds cross-stage time into throughput

**Impact:** Once the run has cycled through a few chunks, the "preprocessing throughput" divides preprocessing completions by wall-clock time that includes every analysis phase in between. Ratio deflates. Over 26 chunks this compounds — ETA keeps growing instead of converging.

**Trace:**
- `TRACE/gui.py:5079-5088` sets `_stage_first_event_time` **once per stage per run**, never per chunk (this is deliberate — see the comment at 5073-5078, which fixed a prior bug where chunk boundaries reset the throughput tracker).
- So `stage_elapsed = monotonic() - _stage_first_event_time` for `_current_stage=="preprocessing"` = time since chunk 1's first preprocessing event, which includes every subsequent analysis phase's wall-clock.
- `throughput = _stage_completions / stage_elapsed` therefore underestimates the true rate.
- Bonus: `_stage_completions` is a single shared variable (gui.py:970) reused between preprocessing and analysis. It happens to hold the right value at chunk boundaries because `idx` is batch-wide and `max()` clamps upward, but the values it holds mid-chunk are stale from the other stage.

**Fix design (Option A — recommended):**
- Split state per stage:
  ```python
  self._stage_first_event_time: dict[str, float] = {}
  self._stage_completions: dict[str, int] = {}
  self._stage_active_seconds: dict[str, float] = {}
  self._last_stage_switch_time: Optional[float] = None
  ```
- Update `_on_progress` (gui.py:5079+) to track "active time" per stage:
  - When `stage != self._current_stage`:
    - If `_current_stage is not None` and `_last_stage_switch_time is not None`, add `(now - _last_stage_switch_time)` to `_stage_active_seconds[self._current_stage]`.
    - Set `_last_stage_switch_time = now`, `self._current_stage = stage`.
  - First-time-seeing-stage bookkeeping stays (it initializes the per-stage dicts).
- Update `_stage_completions[stage] = max(_stage_completions.get(stage, 0), idx + 1)` on "done" events, keyed by stage.
- Rewrite the throughput branch (gui.py:5195-5214):
  ```python
  active_seconds = self._stage_active_seconds.get(current, 0.0) + (time.monotonic() - self._last_stage_switch_time)
  throughput = self._stage_completions[current] / max(active_seconds, 1e-6)
  ```
  Everything else in the branch (Stage 1 extrapolation via `_progress_stage1_share`, Stage 2 remaining) stays.
- Reset the dicts in `_run_pipeline` around gui.py:3988: `_stage_first_event_time.clear()`, `_stage_completions.clear()`, `_stage_active_seconds.clear()`, `_last_stage_switch_time = None`.

**Alternative (Option B):** signal per-image active seconds from the pipeline in the progress event. Cleaner numbers but requires touching the progress signal shape and every emit site. Only worth it if Option A's stage-switch-boundary imprecision proves too noisy in testing — start with Option A.

## Non-bug: CV-ratio CSV group runs full `identify_wing`

Not part of this fix. The v0.2.16 fast path is for the `cv_ratio_overlay` output only; the `cv_ratio` measurement group in the CSV still runs full analysis because `csv` is in `_STAGE2_ANALYSIS_OUTPUTS`. Extending the landmarks-only fast path to that measurement group is a **separate** performance task — do not scope-creep into it here.

## Implementation checklist

1. **New helper** `resolve_effective_config` in `TRACE/pipeline.py`:
   - Takes `(config, outputs, csv_measurement_groups)`.
   - Returns a `PipelineConfig` copy with `skip_intervein_regions` set correctly.
   - Delete the inline computation at pipeline.py:625-633 and call the helper instead.
2. **GUI applies the helper before computing weights** at `TRACE/gui.py:4004`:
   - Call `resolve_effective_config` with the currently selected outputs + csv measurement groups.
   - Store the resolved config so the worker gets the same value.
   - Pass `resolved.skip_intervein_regions` into `compute_progress_weights`.
3. **Split stage state** in `TRACE/gui.py` (around lines 964-971 and 5079-5102):
   - Convert `_stage_first_event_time`, `_stage_completions` to `dict[str, ...]`.
   - Add `_stage_active_seconds: dict[str, float]` and `_last_stage_switch_time: Optional[float]`.
   - Update `_on_progress` to accumulate active seconds on stage switches.
4. **Rewrite the throughput branch** at `TRACE/gui.py:5195-5214`:
   - Use `_stage_active_seconds[current] + (now - _last_stage_switch_time)` for `stage_elapsed`.
   - Use `_stage_completions[current]` for the count.
5. **Reset the new state** in `_run_pipeline` at `TRACE/gui.py:3988+`.

## Verification

- **Regression:** on a full-outputs run (all overlays + all measurement groups), the ETA behavior should be essentially unchanged — `skip_intervein_regions` stays `False` and `stage1_share` stays 0.17. Bug #2's correction is a small delta because chunk analysis phases are relatively short.
- **Target case:** re-run a smaller subset (~200 images) of the reported config (`csv` + `cv_ratio` group, wing iso on). Expect the ETA at 5-min intervals to stay within ±30% of `elapsed × total_images / done_images − elapsed`.
- **Chunk-boundary sanity:** log `throughput`, `active_seconds`, `_stage_completions[stage]` on every ETA update at DEBUG level (behind a flag). Confirm throughput doesn't step-drop each chunk boundary.
- **Monotonicity:** in steady state the ETA should be non-increasing (small bumps OK from EMA + parallelism). If it climbs across chunks, Bug #2 still bites.

## Files touched

- `TRACE/pipeline.py` — new `resolve_effective_config` helper; call it from `_run`.
- `TRACE/gui.py` — call the helper at run start; split stage state; rewrite throughput branch; add resets.

---

# Addendum — post-v0.2.21 regression (2026-08-21)

The v0.2.21 fix landed but broke the progress bar + ETA entirely: on a 4527-image run the progress bar sits at 0% and the label stays "Estimating time until pipeline finishes…" even after 500+ images are done. Log/image-list updates continue normally.

## Root cause

The refactor converted `_stage_total`, `_stage_completions`, and `_stage_first_event_time` from scalars to `dict[str, ...]` in `_on_progress` / `_run_pipeline` (and their inits), but did **not** update `_smoothed_within_stage_fraction` at `TRACE/gui.py:5161-5181`, which still treats them as scalars:

```python
# gui.py:5169   dict <= 0            → TypeError
if self._stage_total <= 0 or self._stage_first_event_time is None:
# gui.py:5180   dict + float          → TypeError
estimated = min(float(self._stage_total), self._stage_completions + fractional)
# gui.py:5181   float / dict          → TypeError
return estimated / self._stage_total
```

Any of those raises `TypeError` on the first tick. The exception propagates out of `_smoothed_within_stage_fraction`, kills `_refresh_progress` before it reaches `_update_eta()` at line 5199, and the QTimer slot exits early. Net effect: progress bar stuck at 0%, ETA stuck at "Estimating…". The image-list column still updates because that's a separate signal path.

## Fix

Rewrite `_smoothed_within_stage_fraction` (whole function replacement — the shape is fine, it just needs to key by `_current_stage`):

```python
def _smoothed_within_stage_fraction(self) -> float:
    """Return 0.0..1.0 reflecting smoothed progress within the current stage.

    Between completion events the value advances linearly toward the next
    expected completion based on the locked-in average time per image,
    so the bar doesn't sit frozen for tens of seconds when parallel
    workers complete in clustered bursts.
    """
    current = self._current_stage
    if current is None:
        return 0.0
    stage_total = self._stage_total.get(current, 0)
    if stage_total <= 0 or current not in self._stage_first_event_time:
        return 0.0
    if self._last_completion_time is None or self._avg_time_per_image is None:
        # No completions yet in this stage — bar stays at the previous
        # stage's final position (or 0 for Stage 1). Honest: we don't
        # have a per-image time to extrapolate from yet.
        return 0.0
    time_since_last = max(0.0, time.monotonic() - self._last_completion_time)
    # Fractional advance toward the next completion (0 → 1 over avg).
    # Cap at 1.0 so we never appear to have completed an image we haven't.
    fractional = min(1.0, time_since_last / max(self._avg_time_per_image, 0.01))
    stage_completions = self._stage_completions.get(current, 0)
    estimated = min(float(stage_total), stage_completions + fractional)
    return estimated / stage_total
```

`_refresh_progress` (line 5183+) already reads through `self._current_stage`; no changes needed there.

## Also worth grepping for

Look for any other reader that assumes the three dicts are scalars — v0.2.21's own diff should be a good starting point. Candidates to spot-check:

- `_avg_time_per_image` update at gui.py:5140-5147 — appears to already use the dict form (`self._stage_completions[stage]`), but double-check the surrounding block.
- Any `self._stage_completions >= N` or `self._stage_first_event_time is not None` comparisons — those are all now dict operations that either mean the wrong thing or raise.

Suggested one-liner:
```
grep -n "_stage_completions\|_stage_first_event_time\|_stage_total" TRACE/gui.py
```
and eyeball every hit that isn't a dict-shaped read (`.get`, `[stage]`, `in`, `.clear`, `= {}`, assignment).

## Verification

1. Cancel the currently-broken run so its dead ETA loop stops firing.
2. Apply the patch; launch on a small (~50 image) batch.
3. Progress bar should climb; the ETA label should replace "Estimating…" within a few seconds of the first "done" event.
4. On the reported 4527-image config, ETA at ~500 images done should be within ±30% of `elapsed × (total / done) − elapsed`.
5. Also re-check the original 2573-image cv_ratio-only case still tracks reality post-fix (this was the target of v0.2.21).

## Files touched

- `TRACE/gui.py` — one function body: `_smoothed_within_stage_fraction`. Plus any other scalar-vs-dict readers grep turns up.
