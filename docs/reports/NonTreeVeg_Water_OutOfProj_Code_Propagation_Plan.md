# VcfPredict output range: two fixes, revised

**Status: implemented and validated, 2026-09-27; existing 63-tile prediction files not regenerated
(not yet needed - see "Validation" below).** Both fixes are applied to
`Git_delivery/modis_vcf/model/VcfPredict.py` and `test_VcfPredict.py`, committed 2026-09-30 on local branch `mcarrol2/c-python-parity-fixes` (see `Git_Delivery_Clone_modis_vcf.md`).
**Note (2026-09-30):** the aerosol QA runs of 2026-09-29/30 (`aerosol_qa_variants_20260929`, `qaC_full63_retrain_20260929`) used a code copy made from the 63-tile run, which does not have this fix. Their XGBoost predictions are unclamped, for B and C alike, so the B-vs-C comparisons are unaffected. Originally written the same day, combining README open items 4 (the
"no 200 in non-tree vegetation over water" part) and 11 (which turned out to already be implemented
for the other two layers - see `Valid_Mask_Disagreement_MOD44W_Root_Cause.md`), plus a second,
related defect found while checking whether predictions are clamped to 0-100 (they are not - XGBoost
genuinely produces out-of-range values in real output; Random Forest/Bagging never do). **Revised the
same day** at the project lead's direction: fix 1 now makes tree cover, non-vegetated and non-tree
vegetation a true partition of 100 (reduces non-vegetated, never tree cover, to absorb any overshoot)
instead of only patching the derived layer. Both fixes touch the same method
(`_runPrediction`/`runTileForYear` in `VcfPredict.py`) and are planned together.

## What was checked first

Per the project lead's instruction, checked the full Python code for an existing MOD44W-based
mechanism before proposing anything new. Found one, already correct:

- `VcfPredict.py` defines `MASK_WATER = 1`, `MASK_OUT_OF_PROJ = 250`, `PRED_WATER = 200`,
  `PRED_OUT_OF_PROJ = 250`.
- `_maskPrediction(tid, year, prediction)` reads the MOD44W water mask (`_getMask`, cached per
  tile/year) and applies both rules in sequence: `mask == 1` -> 200, then `mask == 250` -> 250.
- `_runPrediction` calls `_maskPrediction` for **both** `Percent_Tree_Cover` and
  `Percent_NonVegetated` (`runTileForYear` calls `_runPrediction` once per layer).
- This is unit-tested directly (`test_VcfPredict.py::testMaskPrediction`) and confirmed against the
  real 63-tile-run output: `Percent_Tree_Cover` and `Percent_NonVegetated` both contain exactly 250 at
  exactly the MOD44W-fill pixel counts on h12v01, h13v01, h21v01 and h11v02 (see the root-cause
  document). **Nothing to build for the water/out-of-projection codes on these two layers themselves.**

Separately checked whether predictions are clamped to a sane 0-100 range before writing: **no
`np.clip` or equivalent exists anywhere in the Python code.**

## The current, unmasked-overshoot behavior

`runTileForYear` writes `Percent_Tree_Cover` and `Percent_NonVegetated` as soon as each is predicted,
then derives the third layer:

```python
pctTc: np.ndarray = self._runPrediction(tid, year, self._treeCoverRf, self._treeCoverTopN)
self._write(tid, year, 'Percent_Tree_Cover', pctTc)

pctNonveg: np.ndarray = self._runPrediction(tid, year, self._nonvegRf, self._nonvegTopN)
self._write(tid, year, 'Percent_NonVegetated', pctNonveg)

zeros = np.zeros_like(pctTc)                                    # unused - dead code
pctNontreeVeg: np.ndarray = np.maximum(0, 100 - pctTc - pctNonveg)
self._write(tid, year, 'Percent_NonTree_Vegetation', pctNontreeVeg)
```

Two problems with this, found in the course of the same investigation:

1. **The three layers are not required to sum to 100.** When `pctTc + pctNonveg > 100`,
   `pctNontreeVeg` floors to 0, but `Percent_NonVegetated` is already written and never revisited - so
   the three layers on disk can sum to well over 100 for a real land pixel. Confirmed against real
   output: masked (water/out-of-projection) pixels alone account for this on the four high-latitude
   tiles (`Percent_Tree_Cover == Percent_NonVegetated == 200` or `250`, `Percent_NonTree_Vegetation`
   incorrectly 0 there - see the root-cause document), and the same formula has no way to distinguish
   that case from an ordinary land pixel whose two model predictions simply overshoot 100 together.
2. **No clamping to 0-100 anywhere.** Harmless for the baseline and Random Forest (both average
   decision-tree leaf means, so a prediction is mathematically bounded within the training labels'
   range) but XGBoost tunedB is additive/boosted, not bounded, and genuinely produces out-of-range
   values in the real 63-tile output: min -22, ~1.4% of all pixels (20,977,212) outside `0-100`.
   These don't corrupt any metric reported so far only because the `0-100` valid-range convention used
   throughout this project's comparison scripts already excludes them everywhere they're read - a
   fragile coincidence, not something the raster guarantees on its own.

## The fix, revised

**Design (the project lead's direction): tree cover, non-vegetated and non-tree vegetation should
always sum to exactly 100 on an ordinary land pixel. Any overshoot between tree cover and
non-vegetated is absorbed by reducing non-vegetated - never tree cover - down to what's left after
tree cover, so non-tree vegetation lands on exactly 0 by arithmetic, not by a separate floor. Water
and out-of-projection pixels are untouched by this and keep their existing 200/200/200 or
250/250/250.**

```python
pctTc: np.ndarray = self._runPrediction(tid, year, self._treeCoverRf, self._treeCoverTopN)
pctNonvegRaw: np.ndarray = self._runPrediction(tid, year, self._nonvegRf, self._nonvegTopN)

mask: np.ndarray = self._getMask(tid, year)                     # already cached, no new I/O
okMask = (mask != VcfPredict.MASK_WATER) & (mask != VcfPredict.MASK_OUT_OF_PROJ)

# Reduce non-vegetated, never tree cover, so the two never sum past 100 on an ordinary land
# pixel; leave water/out-of-projection pixels (already 200/200 or 250/250) untouched.
pctNonveg: np.ndarray = np.where(okMask,
                                 np.minimum(pctNonvegRaw, 100 - pctTc),
                                 pctNonvegRaw)

pctNontreeVeg: np.ndarray = 100 - pctTc - pctNonveg              # exactly 0 on an absorbed pixel
pctNontreeVeg = self._maskPrediction(tid, year, pctNontreeVeg)  # write 200/250 on masked pixels

self._write(tid, year, 'Percent_Tree_Cover', pctTc)
self._write(tid, year, 'Percent_NonVegetated', pctNonveg)
self._write(tid, year, 'Percent_NonTree_Vegetation', pctNontreeVeg)
```

Why `np.minimum` needs no explicit `if`: on a pixel that doesn't overshoot,
`pctNonvegRaw <= 100 - pctTc` already, so `np.minimum` returns `pctNonvegRaw` unchanged. On a pixel
that does overshoot, it returns `100 - pctTc` exactly, which makes `pctNontreeVeg` land on 0 by
arithmetic. **The old `np.maximum(0, ...)` floor and the unused `zeros` variable both become
unnecessary** and can be dropped - for `okMask` pixels, `pctNontreeVeg` is non-negative by
construction once `pctTc` is clamped to `[0,100]` (fix below); for masked pixels,
`_maskPrediction` overwrites whatever the raw arithmetic produced anyway.

**This depends on fix 2 (clamping) running first**, and only applies inside `okMask`:

- `min(pctNonvegRaw, 100 - pctTc)` is only guaranteed non-negative if `pctTc <= 100` - true after the
  clamp below, not guaranteed before it (an unclamped XGBoost `pctTc` above 100 would make
  `100 - pctTc` negative, and the `min` would then force a negative `pctNonveg`).
- Applying this on a masked pixel would corrupt it: e.g. on water, `pctTc = pctNonveg = 200`, so
  `min(200, 100-200) = min(200,-100) = -100` - clearly wrong. `okMask` exists specifically to prevent
  this; masked pixels pass through this step with `pctNonveg` unchanged (still 200 or 250 from
  `_maskPrediction` inside `_runPrediction`), then get their `Percent_NonTree_Vegetation` code from the
  explicit `_maskPrediction` call on the last line, the same way `Percent_Tree_Cover` and
  `Percent_NonVegetated` already do.

## Fix 2: clamp raw predictions to [0, 100], before masking

```python
prediction: np.ndarray = np.concatenate(parts).astype(np.int16).reshape(...)
prediction = np.clip(prediction, 0, 100)
...
maskedPred: np.ndarray = self._maskPrediction(tid, year, prediction)
```

One line in `_runPrediction`, ahead of the existing `_maskPrediction` call (clamping must happen
*before* masking - 200 and 250 are legitimate values outside 0-100 that the clamp must not touch).
Applies uniformly to both `Percent_Tree_Cover` and `Percent_NonVegetated` (both go through
`_runPrediction`), which in turn makes fix 1's `100 - pctTc` term always land in `[0,100]`.

## Where the change goes

The run root's code copy (`$RUN/code/python_code/...`) is not to be edited - it is a frozen,
already-compared artifact. `Git_delivery/modis_vcf/model/VcfPredict.py` is the right place: it is
byte-for-byte identical to the run's `VcfPredict.py` (diffed, confirmed), already on the staged
`mcarrol2/c-python-parity-fixes` branch set up for exactly this kind of source change (see
`Git_Delivery_Clone_modis_vcf.md`), and already holds other uncommitted fixes. Nothing gets committed
or pushed without the repository owner's access, same as everything else on that branch.

## Test changes made

`test_VcfPredict.py::testRunTileForYear` (previously asserting only
`(pctTree + pctNonVeg + pctNonTreeVeg < 100).all()` and `!= -10001`) now calls a shared
`_checkPartition` helper, mirroring `testMaskPrediction`'s `okIndexes` pattern:

- On `okMask` pixels: `pctTree + pctNonVeg + pctNonTreeVeg == 100` **exactly**, and
  `pctNonVeg <= 100 - pctTree` (tree cover is never reduced).
- On masked pixels: `pctTree == pctNonVeg == pctNonTreeVeg` equal to `PRED_WATER` or
  `PRED_OUT_OF_PROJ`, matching what `testMaskPrediction` already checks for the first two layers, now
  extended to the third.
- A new `testRunPredictionClamps`, using a minimal fake regressor (`predict()` returning a fixed
  out-of-range array) rather than relying on a real model happening to overshoot, confirms
  `_runPrediction` clips to `[0,100]` on `okMask` pixels.

(These tests were not run in this environment - their fixtures live under
`/explore/nobackup/people/rlgill/SystemTesting/modis-vcf/UnitTests/VcfPredict`, the repository owner's
system-testing area, not something to write into. Correctness was instead validated directly against
real data, below.)

## Validation: done, against real data

Rather than the repository owner's test fixtures, validated by reconstructing exactly what the new
logic produces from the **already-existing** 63-tile prediction rasters (Random Forest and XGBoost
tunedB) plus the real MOD44W mask - no retraining or model reload needed, since both fixes are in the
prediction/writing path, not the model itself.

**Single-tile check, h12v01** (large water and out-of-projection area, both models): every "ok" pixel
sums to exactly 100 after the fix (737,198 pixels, min=max=100); every masked pixel has all three
layers consistently at 200 (4,344,996 water pixels) or 250 (17,957,806 out-of-projection pixels), for
both Random Forest and XGBoost tunedB.

**Full 63-tile frequency check** (how common each problem actually is, out of 1,110,288,083 ok
pixels):

| | Random Forest | XGBoost tunedB |
|---|---|---|
| ok-pixel overshoot (fix 1) | 6,993,110 (0.630%) | 5,206,778 (0.469%) |
| out-of-range (fix 2) | 0 (0.000%) | 20,977,212 (1.889%) |

Confirms the theory precisely: overshoot is model-independent (both algorithms hit it, since it's just
two separate predictions summing past 100), concentrated heavily in a few tiles (`h16v01`, `h20v06`
account for most of it in both models - both near-zero-tree-cover edge tiles); the clamp only ever
fires for XGBoost, never Random Forest or the baseline.

Re-predicting the actual 63-tile files with the fixed code was **not done** - not required by this
plan (see "Scope and blast radius" below) and not yet needed by anything downstream.

## Scope and blast radius

- **Fix 1 (revised) now changes `Percent_NonVegetated` itself**, not just the derived layer, wherever
  tree cover and non-vegetated previously overshot 100 together - a larger scope than the original
  version of this plan. `Percent_Tree_Cover` is never changed by fix 1.
- **Fix 2** changes `Percent_Tree_Cover` and `Percent_NonVegetated` only at the ~1.4% of XGBoost
  tunedB pixels that currently fall outside `0-100`; zero effect on Random Forest or the baseline,
  which never produce such values.
- No retraining, no change to either model itself - purely a change to how raw model output is
  clamped, reconciled against each other, and written.
- Nothing needs to be regenerated urgently: no comparison run so far in this project used
  `Percent_NonVegetated`'s exact values or `Percent_NonTree_Vegetation` at all, and the fix-2 pixels
  were already excluded from every reported metric by the existing `0-100` valid-range convention.
  Re-predicting existing tiles is only needed once something depends on either fixed layer being
  correct at the raster level, not just in the already-published comparison numbers.

## Done, and what's left

Both code changes, the test additions, and the validation above are complete, per the project lead's
instruction to execute this plan. **Left for a separate decision**: whether to re-predict the existing
63-tile Random Forest and XGBoost tunedB output with the fixed code. Not urgent (nothing downstream
depends on `Percent_NonVegetated`'s exact values or `Percent_NonTree_Vegetation` yet), but the
frequency numbers above are now available to inform that decision whenever it comes up.
