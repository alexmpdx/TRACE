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

---

# Addendum #2 — ETA still drifts up each chunk (2026-08-23)

Post-v0.2.21 the ETA is in the right ballpark and no longer goes to 13h+ on a 4h-real run. But the reported new symptom: it **still climbs across chunk boundaries** on all-outputs runs (all six measurement groups selected, wing iso on, ~4500 images).

## Root cause

`_update_eta` at `TRACE/gui.py:5294-5303` extrapolates the entire run from preprocessing throughput alone, dividing by the fixed prior `_progress_stage1_share`:

```python
if current == "preprocessing":
    if self._progress_stage1_share > 0:
        full_stage1 = current_total / max(throughput, 1e-6)
        total_predicted = full_stage1 / self._progress_stage1_share   # <-- fixed prior
        throughput_eta = max(0.0, total_predicted - elapsed)
```

`_progress_stage1_share` comes from `compute_progress_weights` (pipeline.py:378-402), which uses the constants `_PROGRESS_STAGE1_TOTAL_SHARE=17.0` and `_PROGRESS_STAGE2_TOTAL_SHARE=83.0` calibrated on a 133-image / 47-min reference. For an all-outputs run those give `stage1_share=0.17` — i.e. "Stage 2 is ~5× Stage 1".

When the actual run's Stage 2 is faster or slower than that ratio, the extrapolation is off by a matching factor. Each chunk boundary re-samples preprocessing throughput slightly differently (small warm-up drift), and the fixed-prior amplifier turns a small throughput drift into a large ETA drift. It also means the ETA is systematically wrong from the start on any workload that doesn't match the reference — the fix in v0.2.21 only removed the drift caused by `stage_elapsed` bleeding, not the drift caused by the wrong prior.

The Stage 2 branch (line 5304-5306) has the mirror-image bug: it computes `stage_remaining_seconds = analysis_remaining / analysis_throughput` — but during chunked runs there are ALSO more preprocessing chunks to come after this analysis chunk. Sum-of-two-stages is closer to reality than either alone.

## Fix design — per-stage empirical extrapolation, no prior

Once BOTH stages have completions, extrapolate each stage independently from its own observed active-time throughput and sum:

```python
# In _update_eta, after computing `current` and `active_seconds` for the current stage:
prep_completions = self._stage_completions.get("preprocessing", 0)
prep_active      = self._stage_active_seconds.get("preprocessing", 0.0)
anal_completions = self._stage_completions.get("analysis", 0)
anal_active      = self._stage_active_seconds.get("analysis", 0.0)
# Include time-since-last-switch on the CURRENT stage so mid-chunk math accounts
# for wall-clock elapsed since we last banked.
if self._last_stage_switch_time is not None:
    delta = time.monotonic() - self._last_stage_switch_time
    if current == "preprocessing":
        prep_active += delta
    elif current == "analysis":
        anal_active += delta

grand_total = <the batch-wide image count — see below>

# Per-image active-time cost for each stage.
prep_per_img = prep_active / prep_completions if prep_completions >= 2 else None
anal_per_img = anal_active / anal_completions if anal_completions >= 2 else None

remaining_prep = max(0, grand_total - prep_completions)
remaining_anal = max(0, grand_total - anal_completions)

if prep_per_img is not None and (anal_per_img is not None or remaining_anal == 0):
    # Both stages sampled (or no analysis needed) → sum of remaining active time
    # per stage. Chunk-boundary invariant because per-stage active-time cost
    # stabilizes across chunks — no fixed prior amplifying throughput noise.
    throughput_eta = (
        remaining_prep * prep_per_img
        + remaining_anal * (anal_per_img or 0.0)
    )
elif prep_per_img is not None:
    # Analysis hasn't been sampled yet (very first chunk still preprocessing).
    # Fall back to the current fixed-prior extrapolation so we have SOME number
    # to show. It'll get replaced by the empirical path once chunk 1 analysis
    # produces its first samples.
    full_stage1 = grand_total * prep_per_img
    if self._progress_stage1_share > 0:
        throughput_eta = max(0.0, full_stage1 / self._progress_stage1_share - elapsed)
    else:
        throughput_eta = full_stage1 - prep_active
else:
    # No usable samples yet — leave eta_raw as percent_eta.
    ... (unchanged path)
```

`grand_total` should be `max(self._stage_total.get(s, 0) for s in ("preprocessing", "analysis"))`, or just `self._stage_total.get(current, 0)` since both stages carry the same batch-wide count.

The blend with `percent_eta` (lines 5308-5313) can stay — it damps early-run noise. But once analysis has ≥2 samples the empirical path is authoritative; consider dropping the blend once both stages are fully sampled.

## Numeric sanity check (on the reported 4527-image run)

At start of chunk 3 preprocessing (t≈19min elapsed, chunks 1+2 fully done):
- `prep_completions=200`, `prep_active≈780s` → prep_per_img ≈ 3.9s
- `anal_completions=200`, `anal_active≈360s` → anal_per_img ≈ 1.8s
- `remaining_prep = remaining_anal = 4527 - 200 = 4327`
- `throughput_eta = 4327 × 3.9 + 4327 × 1.8 = 16875 + 7789 = 24664s ≈ 6h51m`
- Real remaining: ~4h × (4327/4327) minus 19 min elapsed ≈ (10min/100img × 43 chunks) − 19 min ≈ 4h11m.
- Ballpark, not exact — the reference numbers above are guesses. Reality: same math re-run at chunks 4, 5, 6 should give **similar** results (small monotonic decrease), not steadily-climbing values.

vs current v0.2.21 same moment:
- `full_stage1 = 4527 × 3.9 = 17,655s ≈ 294 min`
- `total_predicted = 294/0.17 = 1731 min ≈ 28h51m`
- `throughput_eta = 28h32m` — matches "still counting up between batches" symptom.

## Verification

- Steady-state monotonicity: on a 200-image test run (2 chunks), log `throughput_eta` and `_stage_active_seconds` at every ETA tick. Confirm `throughput_eta` at the START of chunk 2 preprocessing is **within 5%** of the value at the END of chunk 1 preprocessing. It should NOT step up.
- Full-outputs regression: run the same all-groups + wing-iso config on ~500 images and confirm ETA at 50% mark is within ±25% of `elapsed × (total/done) − elapsed`.
- CV-ratio-only regression: re-run the original 2573-image cv_ratio-only config (target of v0.2.21). Should still track reality — the empirical path handles it too.
- First-chunk behavior: for the very first chunk preprocessing, before any analysis samples exist, the fallback prior path is active. ETA can be wrong here — that's OK. Log a debug line noting "fallback: no analysis samples yet" so it's diagnosable.

## Files touched

- `TRACE/gui.py` — rewrite the throughput branch in `_update_eta` (roughly lines 5276-5313). No pipeline.py changes needed. `compute_progress_weights` and the 17/83 constants stay put — they're used only as the first-chunk fallback now, not as the primary extrapolator.

## What NOT to do

- **Don't** touch `_PROGRESS_STAGE1_TOTAL_SHARE`/`_PROGRESS_STAGE2_TOTAL_SHARE` values. They're the reference-calibration prior; the fix is to stop relying on them as the primary path once empirical data is available.
- **Don't** re-introduce cross-stage `stage_elapsed`. The banked-active-time model from v0.2.21 is correct — this fix builds on top of it.
- **Don't** try to detect "current chunk boundaries" and reset — that goes back to the pre-v0.2.16 chunk-reset bug the original comment warned against.

---

# Addendum #3 — progress bar locks to 99% on first analysis event (2026-08-23)

Independent from the ETA drift, the progress **bar** is stuck at 99% at ~140/4527 on a run where the v0.2.22 Tier-2 fast path is active. The ETA reads a plausible ~7h58m, so the throughput branch is behaving. Only the bar is broken.

## Root cause

`_refresh_progress` at `TRACE/gui.py:5225-5229`:

```python
if self._current_stage == "analysis":
    pct_float = (self._progress_stage1_share + self._progress_stage2_share * within_stage) * 100.0
else:
    pct_float = self._progress_stage1_share * within_stage * 100.0
pct = min(99, int(pct_float))
```

The Tier-2 fast path in v0.2.22 makes `compute_progress_weights` return `(1.0, 0.0)` — the whole run is Stage 1 wall time; Stage 2 is a rounding error. But `_analyze_one` still runs and still emits `_emit_progress(i, stem, "starting")` for each image (pipeline.py:1175), which flips `_current_stage` to `"analysis"`.

At that moment:
- `s1=1.0`, `s2=0.0`.
- `pct_float = (1.0 + 0.0 × within_stage) × 100 = 100`.
- `pct = min(99, 100) = 99`.
- `_progress_pct_high` monotonic-maxes to 99 and never comes down.

The rest of the run alternates between "preprocessing" and "analysis" stages, but the bar can only go up, so it stays at 99%.

The Tier-1 (`cv_ratio_overlay`) fast path from v0.2.16 has the same shape — anywhere weights are `(1.0, 0.0)` and analysis events still fire, this bug bites.

## Fix

Guard the analysis-stage branch on `stage2_share > 0`. When there's no real Stage-2 wall time reserved, drive the bar purely from preprocessing completions — the "analysis" events during a fast-path run represent negligible work and shouldn't advance the bar independently.

Replacement for `_refresh_progress` lines ~5225-5232:

```python
if self._progress_stage2_share > 0 and self._current_stage == "analysis":
    pct_float = (self._progress_stage1_share + self._progress_stage2_share * within_stage) * 100.0
elif self._progress_stage2_share <= 0:
    # Fast-path runs: weights are (1.0, 0.0) — analysis events fire but do
    # negligible work. Drive the bar purely from preprocessing completions
    # so it climbs proportionally across the run instead of jumping to 99%
    # on the first _analyze_one "starting" event.
    prep_total = self._stage_total.get("preprocessing", 0)
    prep_completions = self._stage_completions.get("preprocessing", 0)
    within_prep = prep_completions / prep_total if prep_total > 0 else 0.0
    pct_float = within_prep * 100.0
else:
    # Normal preprocessing branch during a run that does have real Stage 2 work.
    pct_float = self._progress_stage1_share * within_stage * 100.0
```

`within_stage` (from `_smoothed_within_stage_fraction`) is still used when Stage 2 has real work — leave that path untouched.

## Why this is safe for the ETA path

`_update_eta` (line 5255+) reads `_stage_completions` and `_stage_active_seconds` directly, not `_progress_pct_high`. It won't be affected by this bar-only fix. The addendum #2 empirical-throughput rewrite for ETA is orthogonal — do that separately.

## Verification

- Fast-path smoke test (biggest fix target): run `outputs={csv}` + `csv_measurement_groups={cv_ratio}` (or `{wing_area}`, or `{wing_shape}` — anything in `LANDMARK_ONLY_MEASUREMENT_GROUPS`) on ~200 images. The bar should climb linearly from 0% to ~99% as preprocessing progresses. It should NOT jump to 99% on the first analysis event.
- Cv-ratio-overlay-only smoke test (Tier-1 fast path): outputs={cv_ratio_overlay}, no CSV. Same shape as above — bar climbs with preprocessing.
- Full-run regression: outputs=all, all csv groups. Weights should be (0.17, 0.83). Preprocessing bar climbs to ~17%, then jumps to ~17% + analysis progress. No regression from this fix.

## Files touched

- `TRACE/gui.py` — `_refresh_progress` only. No changes to `_on_progress`, `_smoothed_within_stage_fraction`, `_update_eta`, or pipeline.py.
