# Investigation Plan: BandReflMax/Min/Median/AmpBandRefl bug on h12v02

> **Outcome (2026-09-20 note):** the root cause below was acted on. The zenith value clamp was removed for Bands 1-7 and added for Band_31 (2026-09-17), and the result is validated on both tiles in `Bands17_Fix_Validation_Results.md`, which is the current state. This plan and its status sections are historical. The whole chain is summarised in `README_START_HERE.md`.

**Status as of 2026-09-16, later same day: ROOT CAUSE FOUND, superseding
everything below.** The "zenith-gap fix" (`VALUE_CLAMP_ZENITH_CUTOFF=70`)
was itself the dominant remaining problem, not a completed, correct
partial fix. See "2026-09-16 root-cause update (supersedes the status
update below)" immediately following this line. The "2026-09-16 status
update" section further down is kept as historical record of what was
believed at the time, but its conclusion ("keep the zenith-gap fix, it's
correct") is now known to be wrong for Band_1-7.

## 2026-09-16 root-cause update (supersedes the status update below)

A single h12v02 pixel (row 50, col 2264) was traced end to end - both
pipelines, all 23 raw days, all 8 windows - against the real persisted
C intermediates and the raw archive granules (not simulated/assumed
data). Full trace: `Single_Pixel_Trace_h12v02_row50_col2264.txt`.
Finding, cross-checked byte-for-byte against the real
`h12v02.2019.metrics.bin` row: **C's `fix_bands.c:210` zenith>70
reflectance-value clamp never actually executes for `Band_1`-`Band_7`**,
because it reads `arrayz` (solar zenith) before that array has been
populated for the current invocation - `Band_1`-`Band_7` are `b=0..6`,
processed before `b=7` (`solar_zenith`) in the same band loop. This was
already flagged as a real risk in
`C_vs_Python_Code_Differences.md`'s Finding 3 ("likely a latent bug,
though its practical impact hasn't been isolated") - this session's
trace is that isolation, and the impact turned out to be large, not
latent.

Consequence: the "zenith-gap fix" added earlier today
(`BandDayFile.VALUE_CLAMP_ZENITH_CUTOFF = 70`) makes Python correctly
clamp `Band_1`-`Band_7` values at real `solz > 70`, using real solar
zenith - which is *more correct by the literal code* than what C
actually does, since C's equivalent line is dead for these bands. The
fix was validated only against h09v05, where a direct tile-wide check
(`Zenith_Value_Clamp_Root_Cause_Full_Picture.md`) shows the clamp fires
on at most 0.003% of pixels on any single day - i.e. the validation
tile essentially never exercises this code path, so "h09v05: complete
success" was not actually testing whether the fix matches C's real
behavior. On h12v02, where whole months of raw days can exceed 70°
zenith tile-wide, the same fix inflates any 3-day fallback window
containing such a day, which is the dominant driver of the pixel-level
mismatch traced (`BandReflMax` 5954 real (C) vs. 16000 sentinel
(Python) on the traced pixel, purely from this mechanism on 2 of the 3
divergent windows).

A second, smaller, separate mechanism (the already-known `solz<=0`
edge case) explains the remainder on this same pixel, and a third,
opposite-direction discrepancy was found for `Band_31` specifically:
C's clamp *does* fire correctly for `Band_31` (`b=9`, after `b=7`), but
Python's `CompositeDayFile` explicitly sets `applyQa=False` for
`Band_31`, skipping all zenith-based value handling for that band -
under-clamping where Bands 1-7 over-clamp.

**Not yet decided**: what the actual code fix should be (remove the
`>70`/`<=0` value clamp for Bands 1-7 entirely vs. some narrower
change; whether to add the missing real clamp to `Band_31`). This is
the next step, not done here.

## 2026-09-16 status update - read this first on resume

Executed steps 1-2 (`Reports/zenith_gap_diagnostic.py`) and got a
decisive result: the "two zenith thresholds" hypothesis in section 3
below was **confirmed** - h12v02 had 29.8M (pixel, day) samples in the
65-70 solz gap where Python clamped to 16000 and C did not (mean +11,961
inflation each), h09v05 had only 28,263 such samples explaining why it
was clean. User chose fix option (a): added
`BandDayFile.VALUE_CLAMP_ZENITH_CUTOFF = 70` (separate from the existing
`ZENITH_CUTOFF = 65`, which stays as-is for the QA-mask role shared with
`CompositeDayFile.py` - see the fix's own code comment and
`execution_log.md`'s ~15:10 entry for why this had to be a new constant,
not a straight edit of the old one). Re-ran the diagnostic: confirmed
Python's clamp rate for 65-70 pixels dropped from 100% to match C's
0.08% almost exactly on both tiles.

**Then ran the real validation** (`Metrics/<tile>/Python_zenithfix/`,
full `makeMetrics.py` + `compare_c_python_metrics_zenithfix.py`, both
tiles in parallel) and got a mixed result:

- **h09v05: fix is a complete success.** `BandReflMax`/`Min`/`AmpBandRefl`
  now bias=0.000/corr=1.0000/exact%=100.00% on every band - exact C
  parity. `BandReflMaxGreenness`/`BandReflMaxTemp` recovered to
  97.67-99.87%.
- **h12v02: the fix is real but NOT the dominant cause of this tile's
  problem.** `BandReflMax` still shows 0 valid pixels on every band
  (unchanged). `BandReflMin`/`AmpBandRefl` bias essentially unchanged
  (e.g. `AmpBandRefl Band_1`: 8172.068 before -> 8173.237 after - no real
  movement). `BandReflMedian` improved partially (~26% bias reduction,
  e.g. `Band_1`: 7060.930 -> 5228.878) - the only metric where the fix
  clearly helped, but still far from resolved.

**Conclusion**: keep the zenith-gap fix (it's correct, and helped h09v05
fully + h12v02's Median partially), but **h12v02's `BandReflMax`/`Min`/
`AmpBandRefl` catastrophe has a separate, still-unidentified cause.**
Full numbers: `Metrics/h12v02/compare/comparison.h12v02.zenithfix.after.txt`
vs. `comparison.h12v02.after.txt` (pre-fix baseline).

**Where to pick this up tomorrow**: the original sections 2-4 below (what
was ruled out, the now-confirmed-but-insufficient zenith hypothesis, and
the secondary `np.array_equal` bug) are still accurate background, but a
NEW hypothesis is needed for what else is inflating h12v02's `BandReflMax`/
`Min`/`AmpBandRefl` specifically. Worth considering: since `BandReflMedian`
improved partially but `Max`/`Min`/`Amp` did not at all, whatever remains
may affect the *extremes* (max/min) specifically rather than the whole
distribution - e.g. a different sentinel/fallback value that dominates
just the tails, not the typical case. The `np.array_equal` bug (section
4) has still not been fixed and remains a fine, independent, low-risk
cleanup item regardless of the main investigation's outcome.

## 1. Scope and evidence

After reverting `Metrics.py` to the official pre-Finding-6 version (see
`execution_log.md`, 2026-09-16 ~13:00 entry), h12v02 still shows a large,
systematic divergence from the C baseline on four related metrics:
`BandReflMax` (0 valid pixels, bands 1-7), `BandReflMin` (bias 271-1718),
`BandReflMedian` (bias 5698-7061), `AmpBandRefl` (bias 7285-12910). All
biases are **positive** (Python > C). h09v05 does not show this problem.

Confirmed this is **not** a Finding 6 / windowing artifact: the same
magnitudes appear nearly identically in `comparison.h12v02.after.txt`
(Findings-1-2-only, pre-windowing, from Phase 5) and
`comparison.h12v02.finding6.after.txt` (post-windowing) - e.g.
`BandReflMedian Band_1` bias 7060.930 vs. 7061.024. This bug predates
Finding 6 and predates this session's work entirely; it must live in
code common to both configurations.

## 2. Ruled out this session (with evidence)

- **Comparison-script band/index mapping.** Traced `compare_c_python_metrics.py`'s
  `K_TO_NAME` (k=7→NDVI, k=8→Band_31) against the actual C pipeline:
  `MOD44B3.c:333` opens `fpin[i]` from files named `Band_<i+1>.fix.out`
  for i=0..8; `fix_bands.c:228` writes the NDVI output using
  `bands2[8]` = `"Band_8"`, and `MOD44B3.c:101-102`'s `bands2[]` array
  confirms slot 8 = `"Band_8"` (NDVI) and slot 9 = `"Band_9"` (the next
  physical band written, i.e. Band_31 by elimination - not yet
  double-checked directly, see open item below). This matches the
  script's k=7→NDVI, k=8→Band_31 mapping. Not the cause.
- **NaN propagation in `np.max`/`.min()`/`np.median()`.** These are not
  NaN-aware (unlike `metricAmpBandRefl`, which correctly uses
  `np.nanmin`/`np.nanmax`), which looked like a plausible cause at
  first. But `CompositeDayFile._computeComposite()`
  (`CompositeDayFile.py:262-272`) always returns a real `int16` value via
  its Case-1/Case-2 fallback (`comp = np.where(goodComp.mask, allComp,
  goodComp).astype(np.int16)`) - the cube `getBandCube()` builds never
  actually contains NaN in practice. Not the cause.

## 3. Leading hypothesis: asymmetric zenith threshold between C's QA flag and C's value clamp

Traced two **separate** zenith-angle thresholds in the C pipeline that
Findings 1-2 only partially aligned:

- `fix_bands.c`'s `zenith_cutoff` (extern, read from the PCF at
  `MOD44B3.c:161`, confirmed **65** in this run's PCF -
  `build_context/test_pcf/generated/h12v02/MOD_PR44B3.h12v02.pcf:537`,
  `244195|ZENITH_CUTOFF|65`) - used only to set the **QA validity flag**
  arrays (`fix_bands.c:189,201,213`, feeding `array[9]`/`array[10]` in
  `make_metrics.c`, i.e. which raw days count as "good" for the Case-1
  average vs. triggering the Case-2 fallback). This matches Python's
  `ZENITH_CUTOFF = 65` (`BandDayFile.py:17`).
- A **second, hardcoded `70`** at `fix_bands.c:221`:
  `if (arrayz[j][...] > 70) arrays[j][...] = 16000;` - this is the branch
  that actually **substitutes the reflectance value itself** with the
  16000 sentinel, for every band except the QA/cloud/snow flag arrays
  (the `else` branch covers b = Band_1-7 and Band_31). This threshold was
  **not** touched by Findings 1-2 - only the aerosol bit shift at
  `fix_bands.c:187` and the (different) `zenith_cutoff` PCF value were
  addressed.

Python's `BandDayFile.py:119`
(`np.where(solz > self.ZENITH_CUTOFF, self.ZENITH_CLAMP, outBand)`) uses
the **same** `ZENITH_CUTOFF=65` for both roles - there is no second,
looser threshold for the value substitution the way C has.

**Consequence, if confirmed:** for any raw day where solar zenith falls
in **65°-70°**, C keeps the true reflectance value (only excludes it from
the QA-good average, vía `zenith_cutoff`) while still allowing it to
contribute a *real* number to the Case-2 fallback average. Python instead
overwrites that same day's value with the 16000 sentinel before it ever
reaches compositing, so *Python's* Case-2 fallback average - and any
period selected as the max - gets pulled toward 16000, while C's does
not. This would produce exactly the observed pattern: systematic
**positive** bias, worst on `BandReflMax` (picks the single most-inflated
period), still present but smaller on `BandReflMin`/`Median` (only some
periods affected), and worse on h12v02 than h09v05 if h12v02 has more
raw-day/pixel combinations with zenith in the 65-70° gap (plausible given
its known heavier degradation, but not yet confirmed - see step 1 below).

This is a genuinely new finding from this session, not yet verified
against real pixel data - treat it as the leading hypothesis, not a
confirmed root cause.

## 4. Secondary, likely-minor item to fix alongside

`metricAmpBandRefl` (`Metrics.py:970`, reverted/official version) and the
already-flagged `metricAmpGreenestBandRefl` both call
`np.array_equal(minBand, maxBand, equal_nan=True)` inside `np.where(...)`
- `np.array_equal` returns a single scalar for whole-array equality, not
an elementwise per-pixel mask, so the intended "set amplitude to 0 where
min==max" branch almost never fires (arrays are essentially never
identical). In practice this makes the `np.where` degenerate to always
`maxBand - minBand`, which is very likely the intended default anyway -
so this is probably not a significant contributor to the observed bias
magnitude, but it is a real latent bug and cheap to fix once the primary
issue is understood.

## 5. Proposed investigation steps (in order; none executed yet)

1. **Quantify the zenith-gap hypothesis directly.** Pick a handful of
   h12v02 pixels already known to have degraded/case-2-fallback periods
   (reuse pixels identified by `Reports/check_state_masking.py` /
   `check_c_window_masking.py` from the Finding 6 work). For those
   pixels, pull the raw per-day solar zenith angle (`state`/`solz` inputs
   already available under `Metrics/h12v02/...`) across the 23 raw days
   and count how many fall in the 65-70° band. Compare against a
   similar sample from h09v05 to confirm the tile-level difference in
   how often this gap is hit (grounds why h09v05 is clean and h12v02
   isn't, before changing any code).
2. **Directly compare one raw day's Python vs. C reflectance value** for
   a pixel/day known to have solz in 65-70°. Python's `BandDayFile`
   output for that day should show `16000` (`ZENITH_CLAMP`); the C
   `Band_1.<date>.fix.out` (or equivalent) raw file for the same
   pixel/day should show a real reflectance number, not 16000. This is
   the most direct falsification test of the hypothesis - if C's value is
   *also* 16000 here, the hypothesis is wrong and the search moves
   elsewhere.
3. **If confirmed**, decide the correct fix jointly for both pipelines,
   not just Python - since this project's stated goal is Python/C parity,
   not "whichever one seems more scientifically correct." Options to
   weigh: (a) loosen Python's value-clamp to match C's hardcoded 70
   (leaving the QA-flag threshold at 65, mirroring C's two-threshold
   design), (b) tighten C's hardcoded 70 to 65 to match Python (would
   require a C rebuild and revalidation against the existing C baseline
   used throughout this session - a bigger, riskier change), or (c)
   confirm with the project owner (rlgill) which threshold was actually
   intended, since a hardcoded, unlabeled `70` sitting alongside a
   correctly-parameterized `zenith_cutoff` looks like it may itself be an
   oversight in the original C code rather than an intentional two-tier
   design.
4. **If the hypothesis is wrong or only partially explains the bias**,
   fall back to a systematic per-raw-day audit: dump Python's and C's
   per-period composite value (not just the final max/min/median) for a
   handful of h12v02 pixels across all 8 periods, and diff period-by-
   period to localize exactly which period(s) diverge and by how much,
   before looking for the mechanism.
5. **Fix the `np.array_equal` broadcasting bug** in `metricAmpBandRefl`
   (and `metricAmpGreenestBandRefl`) alongside whatever the primary fix
   turns out to be - low-risk, independent of the investigation above.
6. **Re-run `compare_c_python_metrics.py`** for both tiles after any fix
   to confirm the bias/corr/exact% numbers move as expected and h09v05
   does not regress, following the same pattern already used this
   session (`Reports/run_compare_finding6_both.sh` as a template).

## 6. Explicitly out of scope for this plan

- The Finding 6 windowing redesign itself - already reverted; revisiting
  it is a separate decision for later, not blocked by this bug.
- `BandReflMaxGreenness`/`BandReflMaxTemp`/`TempMeanWarmest3` - these
  were validated as healthy (or already resolved by the windowing
  revert) and are not part of this investigation's scope.
