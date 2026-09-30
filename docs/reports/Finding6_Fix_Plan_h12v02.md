# Finding 6 Fix Plan — h12v02 Exemplar

**SUPERSEDED (2026-09-16 end of day): this plan WAS executed
(`Python_finding6/` output for both tiles) but the windowing redesign was
then REVERTED.** After comparing `comparison.h12v02.finding6.after.txt`
against the pre-Finding-6 baseline, the catastrophic `BandReflMax`/`Min`/
`Median`/`AmpBandRefl` numbers turned out to be nearly identical in both
(a pre-existing bug, not caused by this windowing work), while this
redesign measurably regressed `BandReflMaxGreenness`/`BandReflMaxTemp`
exact% (95-99% -> 73-75%, since windowed period means shift which period
ranks "greenest"/"warmest"). `Metrics.py` was reverted to the official
pre-Finding-6 version (see `execution_log.md`'s 2026-09-16 ~13:00 entry) -
Findings 1-2 were kept, only this windowing change was undone. The actual
h12v02 root-cause investigation moved to
`Reports/BandReflMax_Bug_Investigation_Plan.md`. This document is kept
below as historical record only - do not re-execute it without first
re-reading the revert rationale and the investigation plan's status.

**Original status: planning only, not executed.** No code has been edited and no
commands have been run for this plan — this is a design document to review
before any of it is acted on. Builds on top of the current sync workspace
copy (`AI_code_sync_09152026/Python_code/modis_vcf`), which already has the
Findings 1-2 fix applied — this plan is additive, not a fresh copy.

## Root cause (grounded in code, not just the docs)

`C_vs_Python_Code_Differences.md`'s Finding 6 section already diagnoses this
accurately; this section adds the exact call-site inventory needed to scope
a real fix, found by reading `Metrics.py` directly.

`getBandCube()` (`Metrics.py:193-225`) builds an 8-period cube per band from
`CompositeDayFile`, one fixed calendar day each. Any period `fix_bands.c`
(C side) / `BandDayFile.py:116,119` (Python side) considers unusable —
solar zenith too high, or invalid geometry (`solz <= 0`) — is written into
that composite as the literal value `BandDayFile.ZENITH_CLAMP = 16000`,
mixed directly into the same array as real reflectance/thermal values. The
line that looks like it should strip this back out is disabled:
`# cube = self._applyThreshold(cube)` (`Metrics.py:225`), and even if
re-enabled, `_applyThreshold` checks for `NaN` (never present here) against
a threshold (`_nanThreshold=9999`) that can never trigger on an 8-period
cube — it does not and cannot filter the sentinel.

**The codebase is in a half-migrated state, not a from-scratch problem.**
Some newer metric functions already assume `getBandCube()` returns `NaN`
for masked periods and use NaN-aware reducers; others still use plain
reducers that treat `16000` as a real, very-high value:

| Function | Line(s) | Reducer used | NaN-aware already? |
|---|---|---|---|
| `metricBandReflMax` | 578 | `np.max` | No |
| `metricBandReflMedian` | 600 | `np.median` | No |
| `metricBandReflMin` | 622 | `cube.min()` | No |
| `metricBandReflMaxGreenness` | 646-647 | `_sortByNDVI` + index | Partial (see below) |
| `metricBandReflMedianGreenness` | 689-690 | `_sortByNDVI` + `np.median` | Partial |
| `metricBandReflMinGreenness` | 714-715 | `_sortByNDVI` + index | Partial |
| `metricBandReflMaxTemp` | 739-740 | `_sortByThermal` + index | Partial |
| `metricBandReflMedianTemp` | 782-783 | `_sortByThermal` + `np.median` | Partial |
| `metricBandReflMinTemp` | 807-808 | `_sortByThermal` + index | Partial |
| `_lowestMeanBandRefl` (3/6/8) | 833-835 | `np.sort` + `np.mean` | No |
| `_greenestMeanBandRefl` (3/6/8) | 878-880 | `_sortByNDVI` + `np.mean` | Partial, and **not** nan-aware on the final mean (compare to `_warmestMeanBandRefl` below, which is) |
| `_warmestMeanBandRefl` (3/6/8) | 923-925 | `_sortByThermal` + `np.nanmean` | **Yes** — already correct pattern |
| `metricAmpBandRefl` | 967-972 | `np.nanmin`/`np.nanmax` | **Yes** — already correct pattern, *but see separate bug below* |
| `getNdvi` | 399 | plain arithmetic | Would self-correct once inputs are NaN (NaN propagates through `+`/`-`/`/` automatically) |

`_sortByNDVI`/`_sortByThermal` (`Metrics.py:428-466`) are themselves
**already written assuming NaN**: both do
`np.where(np.isnan(ndvi), Metrics.VERY_LOW_SORT_VALUE, ndvi)` before
`argsort`. This is exactly the missing half of the migration — the ranking
key is ready for NaN, but nothing currently supplies it NaN, because
`getBandCube()`/`getNdvi()` never convert the sentinel.

**Why fixing it once at the source is enough:** solar zenith is a per-day
viewing-geometry property, not band-specific, so if one band is
zenith/geometry-masked for a given day/pixel, every band is masked for that
same day/pixel (same `solz` check applied uniformly in `BandDayFile.py`).
A single sentinel→NaN conversion inside `getBandCube()` therefore correctly
propagates to every downstream consumer without needing per-metric special
cases beyond making each reducer NaN-aware.

## Scope check against real h12v02 data (added after re-review, still no execution)

Re-read `fix_bands.c` in full (not just the QA-flag excerpt from the
original Finding 6 writeup) to confirm what C's "exclude invalid, only
average valid" design actually guarantees, then measured it directly
against h12v02's real, already-saved `state.2019.h12v02.fix.out` from this
session's Phase 3b run (read-only — no new runs, no code edits) to ground
the plan's scope in this tile's actual data rather than theory:

- **100% of h12v02 pixels** have at least 3 of Python's 8 fixed calendar
  days masked (validity flag `!=1`); **0%** have fewer than 3 masked.
  Modal case: **7 of 8 masked (41% of pixels)**.
- **24.0% of pixels have all 8 of Python's periods masked** — meaning even
  a fully-correct Finding-6 fix (exclude masked periods, NaN-aware reduce)
  will still legitimately output no-data for about a quarter of this
  tile's pixels, because there's nothing left to reduce over.
- Replicating C's actual window construction (`make_metrics.c`'s 8
  windows of up to 3 raw sub-periods, `m=0,3,...,21`) against the *same*
  state data: only **4.42% of pixels have all 8 windows fully masked** —
  C's fallback-to-raw-average-anyway-and-flag-QA mechanism plus the
  3-day-window design rescues the large majority of pixels that would
  otherwise be starved, because a window only needs *one* clean day out of
  three, not the *one specific* day Python looks at.

**Conclusion: the planned NaN/exclude fix is correct and faithful to C's
intent for the "some good data" case** (this is most of each pixel's 8
periods, most of the time) — re-reading `fix_bands.c` confirms C never
lets a masked raw value pollute an average it actually uses; it only
falls back to averaging regardless of validity when a window is fully
dead, and even then flags it. **But the fix alone will not close the gap
to C's own no-data rate on a tile this degraded** — expect roughly a
~20-percentage-point higher no-data rate from Python than from C on
h12v02's max/min/median-type metrics post-fix, purely from the
single-day-vs-3-day-window structural difference, not from anything left
unfixed in Finding 6 itself. This should be anticipated and reported as
an expected side effect (likely showing up as a materially smaller `n` in
the compare script's output for h12v02 specifically, once Python starts
correctly emitting 32767 no-data instead of silently computing a garbage
number) — not mistaken for a failed or incomplete fix.

**Decision (per user direction 2026-09-15): fix the structural difference
too, not just the sentinel bug.** Both are now planned in full below —
"Proposed fix" covers the sentinel-pollution bug (needed regardless), and
the new "Structural fix" section below covers the windowing redesign.
**Neither has been executed.**

This is not scope creep: `Metrics.py:125-137` already contains the
evidence that 3-day windowing was the *original intent*, just never
finished. There are two hardcoded `_daysSought` variants, one active, one
commented out:
```python
# Two sixteen-day composites
# self._daysSought = \
#     [(self._year,  65), (self._year,  97), (self._year, 129), ...]  # 12 entries, 32-day spacing

# Three sixteen-day composites
self._daysSought = \
    [(self._year,  65), (self._year, 113), (self._year, 161), ...]    # 8 entries, 48-day spacing (ACTIVE)
```
The active variant is explicitly labeled **"Three sixteen-day
composites"** — the author's own words say each of the 8 periods is meant
to represent three 16-day composites. But the code only stores the *first*
day of each trio (`65`, not `65, 81, 97`) and never reads or averages the
other two. The label already matches C's actual 3-day window design
exactly (confirmed by direct comparison: window 0's three days
`65, 81, 97` are exactly `_daysSought`'s anchor day plus the next two raw
composites, 16 days apart each). **The structural fix below completes
what this comment already says should exist**, rather than introducing new
scope.

## Proposed fix (not yet applied)

1. **`getBandCube()`** (`Metrics.py:193-225`): replace the disabled
   `# cube = self._applyThreshold(cube)` with an explicit sentinel-to-NaN
   conversion: `cube = np.where(cube == BandDayFile.ZENITH_CLAMP, np.nan, cube)`
   (needs `BandDayFile` imported — currently only referenced in a comment,
   `Metrics.py:403`). This one change is the actual fix for the
   Finding-6-as-documented mechanism; everything else below is making the
   rest of the file consistent with it.
2. **Switch the non-NaN-aware reducers** in the table above to their
   NaN-aware equivalents: `np.max`→`np.nanmax`, `np.median`→`np.nanmedian`,
   `cube.min()`→`np.nanmin(cube, ...)`, `np.sort`→needs care (see below),
   `_greenestMeanBandRefl`'s final `np.mean`→`np.nanmean` (matching
   `_warmestMeanBandRefl`'s already-correct sibling at line 925).
3. **`np.sort`/`np.argsort` on a cube that can now contain NaN**
   (`_lowestMeanBandRefl:833`, `_sortByNDVI`/`_sortByThermal`'s own
   `argsort` at 442/464 already handle this for the *ranking key* via the
   `VERY_LOW_SORT_VALUE` substitution — but `_lowestMeanBandRefl` sorts the
   band's own raw cube directly, not through a ranking key, and numpy's
   default `np.sort` puts `NaN` at the end (treated as greatest), which
   would corrupt a "lowest N" pick. Needs the same low-value substitution
   pattern applied here, or a NaN-aware manual partition.
4. **New: an explicit NaN→no-data cast before writing.** `writeMetric()`
   (`Metrics.py:494-527`) calls `gdBand.WriteArray(metric.value)` on an
   `Int16` GDAL band with no NaN handling at all — feeding it a float array
   containing `NaN` casts to an undefined/garbage `Int16` bit pattern, not a
   clean no-data value. **This does not exist anywhere in the codebase
   today** (verified: no `isnan(...) → sentinel` pattern before any
   `astype`/`WriteArray` call) — this fix would introduce a new failure
   mode (silent garbage) if not paired with an explicit cast, e.g.
   `np.where(np.isnan(value), 32767, value)` before `Metric(...)` is built,
   matching the `32767` no-data convention `compare_c_python_metrics.py`
   already assumes for NDVI/Band_31. This is the single most important
   detail to get right — a fix that silences the 16000-sentinel bug but
   introduces a NaN-to-garbage-int16 bug would look like progress on
   `exact%`/`corr` while actually being wrong in a new way.
5. **Do not touch `getNdvi()`'s NDVI-arithmetic branch itself** — once its
   `b1`/`b2` inputs are NaN-ified by fix #1, `ndviRaw = np.where(b1 + b2 !=
   0, ((b2-b1)/(b2+b1)), -1.001)` already produces `NaN` correctly via
   ordinary float NaN propagation (`NaN != 0` is `True`, so it takes the
   division branch, which yields `NaN`). No code change needed there beyond
   #1 — but this should be verified, not just assumed, in step 3 below.
6. **Watch the on-disk NDVI cache**: `getNdvi()` reads a cached
   `<ProductType>-NDVI.bin` from `self._metricsDir` if one already exists
   (`Metrics.py:361-369`), bypassing recomputation entirely. A stale cache
   from a prior (pre-fix) run would silently defeat this fix. Using a fresh
   output directory (see below) sidesteps this automatically — call this
   out as *why* the fresh-directory choice matters, not just for tidy
   comparisons.

## Structural fix: redesign `getBandCube()` to average C's actual 3-day windows

**Goal**: make each of Python's 8 periods a mean of the same 3 (or 2, for
the last) raw 16-day composites C's window uses for that period, instead
of a single fixed day — closing the ~20-point no-data gap quantified
above, not just fixing the sentinel.

### Exact windows (grounded in `fix_bands.c`'s `days[23]` array, not re-derived by arithmetic)

Hardcoding the explicit `(year, day)` list per window, the same way
`_daysSought` already hardcodes its 8 anchors (safer than computing offsets
and re-deriving year-rollover/leap-year edge cases):

| Period | Raw days (matches C's window exactly) |
|---|---|
| 0 | `(year,65), (year,81), (year,97)` |
| 1 | `(year,113), (year,129), (year,145)` |
| 2 | `(year,161), (year,177), (year,193)` |
| 3 | `(year,209), (year,225), (year,241)` |
| 4 | `(year,257), (year,273), (year,289)` |
| 5 | `(year,305), (year,321), (year,337)` |
| 6 | `(year,353), (year+1,1), (year+1,17)` |
| 7 | `(year+1,33), (year+1,49)` — **only 2**, matching C's last window exactly |

Verified: period 0's anchor (`65`) plus the next two raw composites 16
days apart (`81`, `97`) exactly reproduces C's window 0. Same pattern holds
for all 8 — this is not a new list invented for this plan, it's the list
`_daysSought`'s "Three sixteen-day composites" comment already implies but
never implemented.

### Correction after further review: use the existing QA mask, not sentinel-detection

The original draft of this section (above, and the narrow "Proposed fix")
assumed the only available validity signal was detecting `cube == 16000`
in the composite value itself. **That's no longer the design** — checked
`CompositeDayFile.py:118-205` and found Python already computes and caches
exactly the signal needed, as a proper 0/1 mask, not implied by a sentinel
value:

- `CompositeDayFile.getQaMask()` — calls `ProductType.createQaMask(state,
  solz, zenithCutOff)` (`ProductTypeMod44.py:68-87`, the same function
  Findings 1-2 already fixed), returning a `(ROWS, COLS)` array that is
  **the exact same combined check as C's `array[9]`/"state" flag** —
  `cloud==0 & shadow==0 & aerosol!=3 & adjacency==0 & 0<solz<zenithCutOff`.
  The code's own comment confirms it: "Like pge61's qa and cloud masks, 0
  is bad and 1 is good" (`CompositeDayFile.py:168`).
- `CompositeDayFile.getCloudMask()` — the cloud-only counterpart, matching
  C's `array[10]`/"cloud" flag exactly the same way.
- Both are computed once per raw composite day and cached to disk
  (`<name>-qa.npy`, `<name>-cloud.npy`) as a side effect of building that
  day's raster.

This is a better foundation than sentinel-detection: it's the authoritative
QA decision already computed by the existing pipeline, not an inference
from whether a value happens to equal a magic number — and it means the
redesigned `getBandCube()` doesn't need to guess at validity at all, it
just needs to *use* what's already there.

### Code changes (not yet applied)

1. **Add a new full 23-entry raw-day list** (e.g. `_rawDaysSought`) built
   from the table above, alongside the existing `_daysSought` (which stays
   — still needed as the per-period anchor/xref for anything that keys off
   "which period is this," e.g. `Metrics.py:336`).
2. **Rewrite `getBandCube()`** (`Metrics.py:193-225`) to, per raw day:
   a. Read the `CompositeDayFile` raster **and** its `getQaMask()` (23 of
      each instead of 8 raster reads today — see Efficiency section below
      for why this is cheaper than it sounds).
   b. Build two parallel 23-slot arrays per band: raw values, and the 0/1
      QA validity flag (no sentinel-to-NaN conversion needed — validity
      comes from the QA array, not from inspecting the value).
   c. **Reduce into 8 windows, replicating `make_metrics.c:65-106`
      exactly**: for each window, sum the raw values of sub-periods where
      `qa==1` and divide by that count. **Decision (per user direction
      2026-09-15): when a window has zero valid sub-periods, replicate
      C's fallback exactly** — average the raw values of all sub-periods
      in that window *regardless* of QA, and mark the period degraded
      (see QA-bitmask output below) rather than emitting a no-data value.
      This is a straight port of `make_metrics.c:82-95`'s `indexx2==0`
      branch, using the same window definitions. No `NaN`/no-data value is
      produced for a "no good data" window — every window always yields a
      number, exactly matching C never nulling a period.
   d. Result: same `(8, ROWS, COLS)` shape `getBandCube()` already returns
      today — every downstream caller (`metricBandReflMax`, `_sortByNDVI`,
      etc.) keeps working against an 8-period cube unchanged in shape.
3. **Since every window always yields a real number now** (matching C),
   the NaN-aware-reducer audit from the narrow "Proposed fix" section is
   **no longer needed for this path** — there's no `NaN` for `np.max`/
   `np.median`/etc. to choke on. The narrow fix's reducer changes were
   only needed under the "leave it NaN" fallback option, which this plan
   no longer uses. (`writeMetric()`'s NaN-to-32767 cast is still cheap
   insurance to keep, in case any future code path produces one, but it's
   not load-bearing for this design.)

### Persistent QA output — new 8-bit degraded-period field, matching C's `*QA` convention

C's `make_metrics()` sets one bit per period in a single per-pixel byte
(`*QA |= (1 << n)` for `n` in 0-7, `make_metrics.c:85`) whenever that
period's window had zero valid sub-periods. This bit currently only lives
inside `gen_metrics`'s process memory and is discarded — the original
Reduced-Scope plan's Phase 3.5 already proposed exporting it from the C
side for exactly this reason.

**Python should produce the equivalent, as a new persistent output, not
just an internal flag**, per user direction:

- Build an 8-bit-per-pixel array (`np.uint8`, values 0-255) during the
  windowing step in 2c above: bit `n` set wherever period `n`'s window hit
  the zero-valid-subperiods fallback for that pixel. This falls out
  directly from the same per-window reduction loop — no extra pass needed.
- **This is a genuinely per-pixel, per-tile field, not per-band or
  per-metric** — confirmed from `make_metrics.c` itself: the `*QA` bit-set
  is driven entirely by the shared validity flag (`array[9][m]`, C's
  `state`), not by any band-specific value (the `k` band loop just
  triggers the same-valued write redundantly for each band). Python's
  `getQaMask()` is likewise QA-mask, not reflectance-value, per raw day —
  identical across all bands for a given day/pixel (same underlying
  cloud/zenith/aerosol geometry). So this bitmask should be **computed
  once per tile** (using any one band's QA masks — they're identical
  across bands — or a dedicated pass that doesn't need band values at
  all, just the QA masks), not recomputed 9 times.
- **Persist it as a new single-band GeoTIFF**, written the same way
  `writeMetric()` already writes everything else (same driver,
  `gdal.GDT_Byte` instead of `GDT_Int16` since it's 0-255), e.g.
  `QA_DegradedPeriods.tif`, alongside the existing 24 metric `.tif`s in
  `3-Metrics/`. One band, one file, matches C's single-byte-per-pixel
  design.
- Worth also persisting the cloud-only equivalent (`getCloudMask()`) the
  same way, matching C's separate `*cloud` output — cheap to add since
  the mask is already being read for the QA computation anyway, and gives
  the same pixel-exact "why was this degraded" diagnostic Phase 3.5
  proposed adding to the C side. Not strictly required by this plan's
  goal (closing the no-data gap), but low-cost given the data's already
  in hand — flagging as an easy add-on, not mandatory.

## Efficiency: the current per-metric-function design re-reads everything, repeatedly

The user asked whether there's a more efficient way to do this than the
current code — yes, and the windowing redesign makes fixing it more
important, not just nice-to-have.

**Confirmed the actual execution model** (`makeMetrics.py:97-113`): one
`Metrics` instance is constructed per tile, then every requested metric
function is called sequentially against that *same* instance
(`for metricName in metricsToRun: mm.getMetric(metricName)`). This means
an instance-level cache is safe and correctly scoped — nothing tears down
or goes stale mid-run.

**Confirmed the actual redundancy** (`grep` of all `getBandCube()` call
sites, `Metrics.py`): ~20 metric functions each call `self.getBandCube(bandName)`
independently, with **zero caching** — no memoization, and (unlike
`getNdvi()`, which at least has an on-disk cache file it writes after its
first computation) no fallback dedup either. The same band's data gets
re-read from scratch by every metric function that needs it — e.g. Band_1
alone is pulled by `metricBandReflMax`, `Median`, `Min`,
`MaxGreenness`, `MedianGreenness`, `MinGreenness`, `MaxTemp`,
`MedianTemp`, `MinTemp`, and more — on the order of 10+ redundant
re-reads of the *identical* data per band, per tile, today. With the
windowing redesign roughly tripling the per-call read cost (23 raster +
23 QA-mask reads instead of 8 raster reads), this redundancy would
multiply into a real runtime problem if left as-is — potentially pushing
Phase 4's already-long ~45-50 min/tile runtime up by much more than the
"true" 3x the windowing itself requires.

**Proposed fix**: add a simple instance-level cache to `Metrics.__init__`
(e.g. `self._bandCubeCache: dict[str, tuple[np.ndarray, dict]] = {}`),
and have `getBandCube()` check it first, populate it on first call per
band name, and return the cached result on every subsequent call within
that tile's run. This drops the true minimum work to exactly once per
band (9 bands × 23 raw reads = 207, not 10+ redundant passes over the same
9 bands) — the caching pays for itself well beyond covering the 3x
windowing increase. The QA-bitmask computation (previous section) should
piggyback on this same cache, computed once alongside the first band's
cube build, not recomputed per band.

**Caveat found while sizing this for a possible two-tile-parallel run**:
`getBandCube()`'s cube is currently built via `np.full((n, ROWS, COLS),
np.nan)`, which defaults to `float64`. A 23-slot cube at `float64` is
~4.2GB per band; if the cache above ends up holding all 9 bands
simultaneously (worst case, depends on metric call order), that's ~38GB
for one tile's run. **Use `float32` for the cache instead** (`NaN` is
representable in `float32` too, and the metrics are ultimately cast to
`Int16` on write anyway, so the extra `float64` precision buys nothing
here) — halves the worst-case footprint to ~19GB/tile, giving comfortable
headroom against forest203's actual 104GB available (checked directly,
2026-09-15: 12 CPUs, 112GB total / 104GB available / near-idle) even if
h09v05 and h12v02 are run in parallel (~38GB combined worst case, vs 104GB
available). Parallel execution of both tiles is otherwise safe — already
tested this session (`compare_c_before_after.py` ran both tiles
concurrently on this same node without issue) and there's no path
collision risk (`Metrics.__init__` nests output under `<tile>/<year>/`
even within a shared `-o` root, and this plan already uses distinct `-o`
roots per tile besides).

### Impact / cost of this option vs. the narrow fix

- **~3x the raw composite reads** per band per tile during cube
  construction (23 vs 8), each now also reading a cached QA-mask file
  alongside the raster — with the instance-level cache above, this cost
  is paid exactly once per band (9 × 23 = 207 raster + 207 QA-mask reads
  per tile), replacing what's currently 10+ redundant re-reads of 8 raw
  days per band. Net effect on wall-clock runtime is genuinely uncertain
  without measuring — more total files touched, but far fewer redundant
  ones — should be measured on the first real run, not assumed either
  direction.
- **Changes the actual numeric values of every metric**, not just the
  ones affected by masking — even a fully clean, unmasked pixel's period 0
  changes from "day 65's raw value" to "mean of days 65, 81, 97" — this
  is a bigger behavioral change than Finding 6 alone. **This is exactly
  why both h09v05 and h12v02 need to be rerun** (see Validation plan
  below, updated per user direction) — h09v05 was the "clean" tile for
  the narrow Findings 1-2 fix precisely because nothing else was touching
  it, but this redesign changes every pixel's period values on every
  tile, clean or not, so h09v05 can no longer be assumed unaffected.
- **Bigger blast radius for review**: this changes `getBandCube()`'s core
  semantics, touching all ~20 downstream metric functions' actual output,
  not just the subset Finding 6's narrow fix touches.
- **No more NaN-fallback ambiguity**: because the zero-valid-subperiods
  case now always produces a real averaged number (matching C, per the
  decision above) instead of `NaN`, the earlier concern about auditing
  ~10 reducers for NaN-awareness is moot for this path — every window
  always yields a number, so plain `np.max`/`np.median`/etc. work
  correctly without modification. One less category of risk than the
  narrow fix alone carried.

## A separate, related bug — do not conflate with Finding 6

`metricAmpBandRefl` (`Metrics.py:970`):
```python
value = np.where(np.array_equal(minBand, maxBand, equal_nan=True), 0, maxBand - minBand)
```
`np.array_equal` returns **one scalar boolean for the entire array**, not
an element-wise comparison — almost certainly meant to be
`minBand == maxBand`. As written, `np.where` receives a single `True`/`False`
condition broadcast across the whole tile: either the *entire* tile's
`AmpBandRefl` becomes 0, or *every* pixel uses `maxBand - minBand`, never a
genuine per-pixel mix. This is independent of Finding 6 (it's a
broadcasting mistake, not a masking-sentinel one) and is a plausible
additional contributor to `AmpBandRefl`'s especially bad h12v02 numbers
already seen in this session's before/after comparison (bias in the
thousands). **Flagging only — not part of this plan's scope**, but
interpreting `AmpBandRefl` results after a Finding 6 fix should account for
this separately, or the two bugs' effects will get attributed to the wrong
fix.

## Validation plan — both h09v05 and h12v02 (updated per user direction)

**Originally scoped to h12v02 alone**, since that's where Finding 6 was
discovered. **Now runs both tiles**, per explicit user direction: the
windowing redesign changes every pixel's period values on every tile, not
just masked ones (see Impact/cost above) — h09v05 can no longer be assumed
unaffected/still-clean the way it could for the narrow Findings 1-2 fix,
where nothing about that fix touched pixels outside the QA-adjacent bit
logic. Both tiles need a real before/after comparison to know whether the
redesign is actually correct, not just whether it "looks fixed" on the one
tile that was broken.

This session already has matching Findings-1-2-fixed C baselines for
*both* tiles from Phase 3c (`Metrics/h09v05/C/...`, `Metrics/h12v02/C/...`)
— **no C-side rebuild needed for either tile**, both the sentinel fix and
the structural windowing fix are Python-only.

Per the decision above, `Python_finding6` below means **both fixes
together** (sentinel exclusion + 3-day windowing + exact-C fallback +
QA-bitmask output), not the sentinel fix alone — implemented and validated
as one combined change to `getBandCube()`.

**Save results distinctly — do not overwrite anything from the Findings
1-2 pass — for both tiles:**

```
Metrics/<tile>/Python_finding6/          <- new makeMetrics.py -o target
                                             (parallel to existing Python/),
                                             <tile> in {h09v05, h12v02}
Metrics/<tile>/compare/comparison.<tile>.finding6.after.txt
                                          <- new compare_c_python_metrics.py
                                             output (parallel to the
                                             existing .after.txt)
Metrics/<tile>/compare/py_before_after.<tile>.finding6.txt
                                          <- new: Python-vs-Python own
                                             output, old (Findings-1-2-only)
                                             vs new (+Finding 6), mirroring
                                             the C-vs-C script already
                                             written this session
                                             (Reports/compare_c_before_after.py)
                                             but comparing the two Python
                                             .tif directories instead of
                                             two .bin files
```

Steps (in order, once approved to execute; repeat the per-tile steps for
both h09v05 and h12v02):

1. Apply the code changes above to
   `Python_code/modis_vcf/model/Metrics.py` (the same copy already carrying
   the Findings 1-2 fix — cumulative, not a fresh checkout). Applies once,
   not per tile.
2. For each tile, re-run `makeMetrics.py -o Metrics/<tile>/Python_finding6
   -y 2019 -t <tile>` via the same `singularity exec` pattern already
   working this session (Phase 4). **Estimated ~70-150 min/tile** (not
   measured, a real range not a firm number) — traced why Phase 4 took
   ~45-50 min/tile and found it's dominated by first-time composite
   building from raw HDF for 8 days × 9 bands = 72 builds (later metric
   calls mostly hit the on-disk composite cache, not raw HDF again); the
   new design needs 23 days × 9 bands = 207 builds, ~2.875x more, only
   partly offset by the `getBandCube()` cache (which mainly helps the
   already-cheap re-read path, not the expensive first build). **Both
   tiles can run in parallel** on forest203 without conflict — checked
   directly (2026-09-15): 12 CPUs, 104GB available, near-idle; no path
   collisions (each tile's `-o` is already distinct, and `Metrics.__init__`
   nests by `<tile>/<year>/` regardless); already proven safe this session
   (`compare_c_before_after.py` ran both tiles concurrently on this same
   node). With the `float32` cache fix above, worst-case combined memory
   for both tiles in parallel is ~38GB against 104GB available — running
   parallel cuts wall-clock roughly in half versus sequential.
3. Sanity-check the QA-bitmask output and the exact-C fallback landed
   correctly before trusting any metric numbers: spot-check a handful of
   pixels known (from the `check_state_masking.py`/`check_c_window_masking.py`
   diagnostics already run this session) to have fully-masked windows on
   h12v02, and confirm (a) the corresponding metric value is a real
   averaged number, not a sentinel or garbage, and (b) `QA_DegradedPeriods.tif`
   has the expected bit set for that pixel/period.
4. Run `compare_c_python_metrics.py <tile>` (already repointed to this
   workspace's own `Metrics/<tile>/C|Python/` from the earlier Phase 5
   work) against `Metrics/<tile>/Python_finding6` instead of
   `Metrics/<tile>/Python`, output to
   `comparison.<tile>.finding6.after.txt` — **this needs `PY_METRICS_DIR`
   parameterized or a second copy of the script**, since it's currently a
   module-level constant built from `TILE` alone; simplest is a copy or an
   added CLI arg for the Python subdirectory name.
5. Run the new Python-vs-Python before/after script (write it analogous to
   `compare_c_before_after.py`, swapping the two `.bin`-reading `load()`
   calls for two `.tif`-reading passes over
   `Metrics/<tile>/Python/<tile>/2019/3-Metrics/` vs.
   `Metrics/<tile>/Python_finding6/<tile>/2019/3-Metrics/`) to isolate
   Finding 6's own impact on Python's output, the same way
   `compare_c_before_after.py` isolated Findings 1-2's impact on C's
   output. Run for both tiles.
6. Compare each tile's `comparison.<tile>.finding6.after.txt` against its
   existing `comparison.<tile>.after.txt` (Findings-1-2-only) and against
   the original pre-fix baseline
   (`C_vs_Python_VCF_Metrics_Comparison_<tile>.md` §3) — three-way, not
   just before/after, for both tiles.
   **What "fixed" looks like**:
   - h12v02: `BandReflMax`/`Min`/`Median`/`AmpBandRefl` (the rows currently
     at ~0% exact / huge bias due to Finding 6) should move toward the same
     high-corr, high-exact% territory `BandReflMaxGreenness`/`BandReflMaxTemp`
     already showed after just the Findings 1-2 fix — if they don't, the
     fix is incomplete or wrong somewhere above. `n` (comparable pixel
     count) should also come back to something close to the full tile —
     the previous no-data floor (24% for the single-day design) should
     mostly disappear now that windows always produce a real value.
   - h09v05: was already good after Findings 1-2 alone — check this
     redesign doesn't *regress* it. Since h09v05's masking rate is much
     lower than h12v02's, expect a smaller but still real shift in every
     metric's numbers (single-day value → 3-day windowed mean), and
     correlation/exact% against C should stay high or improve, not drop.

## Open questions to resolve before executing (not blocking the plan itself)

**Resolved by the exact-C-fallback decision** (no longer open): the
all-NaN-slice/`np.nanmax` warning-spam question and the
`_lowestMeanBandRefl` NaN-substitution question both only applied under
the "leave fully-masked windows as NaN" option, which this plan no longer
uses — every window now always produces a real averaged number, so plain
(non-NaN-aware) reducers are correct as-is throughout.

Still open:

- **`_computeComposite()`'s own internal logic** (`CompositeDayFile.py`,
  not yet read in this pass) — each of the 23 "raw" composites Python
  reads is itself already built from finer-grained daily acquisitions
  within that ~16-day window, with its own QA-driven combination logic.
  This plan treats `CompositeDayFile.getQaMask()`'s output as the
  authoritative per-raw-period validity signal without auditing exactly
  how that composite (and its QA mask) gets built internally — reasonable
  to trust, since it's the same existing, already-used mechanism the
  current 8-single-day design already relies on, but not independently
  verified in this pass.
- Whether any other `.py` files construct/consume a raw band cube outside
  `Metrics.py` (not checked in this pass — this plan only audited
  `Metrics.py` itself, since that's where the finding was documented and
  reproduced).
- Exact on-disk format/GDAL type for the new `QA_DegradedPeriods.tif`
  (and optional cloud-bit companion) — `gdal.GDT_Byte` proposed above but
  not cross-checked against how other tools in this pipeline expect to
  consume a QA band, if any do.
