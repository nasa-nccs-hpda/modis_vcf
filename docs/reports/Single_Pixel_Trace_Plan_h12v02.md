# Single-pixel C-vs-Python trace plan — h12v02, BandReflMax(Band_1)

Status: **executed 2026-09-16** (results: `Single_Pixel_Trace_h12v02_row50_col2264.txt` and `Zenith_Value_Clamp_Root_Cause_Full_Picture.md`); originally written as a plan.

## Goal

Pick one (row, col) pixel on h12v02 where Python's `BandReflMax`-`Band_1`
is wrong (>=16000, the "bad" sentinel) but C's is a real, plausible value,
then walk that one pixel through every step of both derivations (per the
already-written `BandReflMax_Band1_C_vs_Python_Derivation.txt`) so the
exact raw day / exact window where the two pipelines' numbers first
diverge can be pointed to directly, instead of reasoned about statistically.

## What's already available, no fresh archive reads needed

Both pipelines already have on-disk intermediates for the current code
state (Findings 1-2 + the zenith-gap value-clamp fix, Finding-6 windowing
reverted) sitting in this sync workspace right now:

- **C**: `build_context/runtime/intermediate/h12v02/Band_1.2019.h12v02.fix.out`
  and `state.2019.h12v02.fix.out` (23 raw-day values / 23 binary QA flags
  per pixel, row-major, seekable by `(row*4800+col)*23*2` bytes — same
  trick already used this session in `sample_c_bin.py` [script no longer present]), and
  `Metrics/h12v02/C/h12v02.2019.metrics.bin` (the final per-pixel
  `metrics[324]` array, sparse/row-major, non-water pixels only).
- **Python**: `Metrics/h12v02/Python_zenithfix/h12v02/2019/1-Days/MOD44-h12v02-<yyyyddd>-Band_1.bin`
  (23 raw-day cached values — already both value-clamps applied, i.e.
  exactly what `BandDayFile._getRaster()` produced), 
  `2-Composites/MOD44-h12v02-<anchor-day>-Band_1.bin` (the 8 window
  composite values) plus each window's `-qa.npy`/`-cloud.npy` (Case 1
  vs. Case 2 flag per window, already computed and saved), and
  `3-Metrics/BandReflMax.tif` (the final metric actually compared).

Neither pipeline persists the raw per-day solar-zenith value anywhere
(C never writes `arrayz` to disk; Python's `getSolz()` reads it fresh
every time, uncached) — that's the one thing both sides are missing and
that must come from the original archive granules directly (see below).
Since both C and Python ultimately read the same underlying MOD44 CQ/CH
granules for a given tile/day, one shared read of solar-zenith (and, for
full transparency, the raw `state` bits) gives common ground truth both
derivations can be checked against.

Archive location (from `Python_code/modis_vcf/view/makeMetrics.py:91`):
`/css/modis/Collection6.1/L3/MOD44B-VCF/dev/<year>/`, granule naming per
`ProductType.findFile()`: `MOD44<CQ|CH>.A<year><doy>.<tile>.*.hdf`.

## Step-by-step plan

**Step 0 — pick the pixel.** Open `Metrics/h12v02/C/h12v02.2019.metrics.bin`
(pure Python, no container, same approach as `sample_c_bin.py`) and
`Metrics/h12v02/Python_zenithfix/h12v02/2019/3-Metrics/BandReflMax.tif`
(needs GDAL → container) side by side, scan for a (row, col) where
Python's Band_1 value is >=16000 and C's is comfortably below it (e.g.
<10000, a large, unambiguous gap). Record that one (row, col).

**Step 1 — pull C's persisted per-pixel trace** (pure Python, no
container): seek into `Band_1.fix.out` and `state.fix.out` at that
pixel's byte offset to get the 23 raw (already zenith>70-clamped) Band_1
values and the 23 binary QA-valid flags; seek/scan `h12v02.2019.metrics.bin`
for this row/col to pull the actual persisted `metrics[]` array (unsorted
8 periods at idx `k*8..k*8+7` for k=0, sorted 8 at idx 126-133).

**Step 2 — pull Python's persisted per-pixel trace** (container, for
`.npy`/`.tif` reads): read the 23 `1-Days/*Band_1.bin` values, the 8
`2-Composites/*Band_1.bin` window values, and the 8 `*-Band_1-qa.npy`
Case-1/Case-2 flags, all at the same (row, col); read `BandReflMax.tif`
at this pixel to confirm it's the max of those 8 window values.

**Step 3 — pull shared ground truth** (container, one-time read of the
23 raw archive granules for h12v02/2019 at this same pixel): raw
(unclamped) Band_1 reflectance, raw solar_zenith, and raw state bits,
for all 23 days. This is the one genuinely new read neither pipeline's
saved intermediates already contain.

**Step 4 — reconstruct both derivations by hand from Step 3's ground
truth**, and cross-check each reconstruction against Steps 1/2's
persisted values before trusting it:
  - C side: recompute the b==10 QA-valid bit (cloud/shadow/adjacency/
    aerosol/zenith<65) and the value clamp (zenith>70→16000) per raw
    day; confirm this matches Step 1's `state.fix.out`/`Band_1.fix.out`
    exactly. Then run `make_metrics.c`'s window logic (Case 1 mean-of-
    valid, Case 2 fallback-mean-of-all-in-window) by hand for the 8
    windows and confirm the result matches Step 1's `metrics.bin` row.
  - Python side: recompute `BandDayFile`'s two value clamps (solz<=0→
    16000, solz>70→16000) and `createQaMask()`'s bit logic (same as C's
    plus the extra `solz>0` requirement) per raw day; confirm this
    matches Step 2's `1-Days` cache exactly. Then run
    `_computeComposite()`'s Case 1/Case 2 logic by hand for the 8
    windows and confirm it matches Step 2's `2-Composites` cache and
    `-qa.npy` flags.

**Step 5 — line up one table**: 23 raw days x (C's raw-clamped value,
C's valid flag, Python's raw-clamped value, Python's valid flag,
ground-truth solar zenith), then 8 windows x (C's period value, Python's
period value, Python's Case 1/2 flag), then the two Max values. Read
across it to find the first row/window where the two pipelines' numbers
part ways, and check specifically whether that point coincides with a
Case-2 (fallback) window on the Python side that contains a raw day
with ground-truth solz<=0 — the mechanism flagged as the leading
hypothesis in `BandReflMax_Band1_C_vs_Python_Derivation.txt`.

## Mechanics / tooling notes for execution

- Steps 0/2/3 need GDAL/HDF4, so they run via the same
  `singularity exec ... modis-vcf-latest ...` pattern already used all
  session (read-only — no `-o` output, no pipeline mutation, just
  opening files and printing values for one pixel).
- Steps 1 and the fix.out/metrics.bin parts of Step 4 need no container
  at all — plain `python3` + `struct`, as already proven working this
  session.
- Everything reads existing, already-produced files; nothing reruns
  `makeMetrics.py`, `gen_metrics`, or any part of the B1/B3 pipeline.
- Estimated cost: one short container invocation (single-pixel reads
  from ~25 small files plus 23 archive granule opens — seconds, not
  minutes) plus local struct-based reads: no meaningful compute or time
  risk.

## Not yet done

No pixel has been selected and no values have been read. This document
is the plan only, per instruction, awaiting approval to execute Steps
0-5 above.
