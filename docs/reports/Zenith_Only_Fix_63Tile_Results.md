# Zenith-only NDVI-default fix: 31-tile x 3-model validation results

**Status (2026-09-28): diagnostic complete. Not adopted as a general fix; the project lead's direction (2026-09-28) is to scope it to Greenland (and possibly Iceland) tiles: `Greenland_Iceland_Scoped_Fix_Plan.md`, blocked on MOD44C data. No code changed; the fix is not in any shipped script or on the 2026-09-30 git branch.** This is Stage 3 of `Zenith_Only_Fix_63Tile_Plan.md` - the cross-tile
comparison the plan said had "not been generated yet." All three detached node jobs (forest203/204/205)
finished between 18:20 and 18:29 today; this document concatenates their output
(`stages123_results_forest203/204/205.jsonl`, plus the earlier h16v01 smoke test) into one table and
reports what it shows.

**PROJECT LEAD'S DIRECTION (2026-09-28, after the cutoff-70 retest): scope the fix to Greenland, and consider Iceland.**
h16v01 is a Greenland tile. The project lead's view is that applying this zenith-only NDVI-default fix
only to Greenland tiles is likely to be the solution. Iceland tiles should be considered for inclusion
as well. This has not been designed or tested yet. Points to check when it is:
- **None of the 10 tiles the fix hurt is in Greenland or Iceland** (h11v02, h11v03, h18v03, h18v04,
  h20v02, h20v03, h21v02, h23v02, h23v03, h24v03 are northern Canada, Europe and Russia). That is
  consistent with the direction, but it is not a test of it.
- **The C code already scopes a rule this way.** `MOD44B3.c` lines 411-421 (the "Greenland" block, MSZ
  4/30/2026) sets non-vegetated to 100 when some metrics are zero, but only for `vtile < 3 && htile < 18`.
  That box also contains h11v02, one of the hurt tiles, so a Python scope copied from C's box would
  include a tile the fix damages. The list of Greenland (and Iceland) tiles needs to be explicit.
- **The tile list is still to be written.** Iceland is about 63-67 N and 13-24 W, which should put it in
  h16v02 and possibly the edge of h15v02; confirm against the tile grid. Check which of these tiles are
  among the 63 run tiles and the 31 in the 65-degree test.

**Headline: the fix is not the narrow, h16v01-only correction the two-tile test suggested.** It resolves
h16v01's known over-prediction, but it measurably **degrades agreement with real training labels on at
least 10 of the other 30 tiles**, consistently across all three models (baseline, Random Forest, XGBoost
tunedB). This is new information; the plan's two open questions ("does the fix touch any other training
tile besides h16v01?" and "does that answer change across models?") are both answered, and the second
answer is "no, the degradation is consistent across models," not "no effect."

## Coverage

31 of 63 tiles (those Stage 0 flagged with any 16-day period having more than 0.01% of pixels zeroed by
the zenith cutoff) x 3 models = 93 rows, all `ok: true`, no failures. Full table:
`stage3_consolidated_31tiles_3models.csv` (also in this directory), columns as written by the driver:
`nCommonValid`, `meanReal`/`meanFixed` (whole-tile prediction means), `diffMean`/`diffStd`
(fixed-minus-real), `fracAbsDiffLe5`/`fracAbsDiffGt20` (fraction of common-valid pixels differing by at
most 5 points / more than 20 points), `nLabels`/`meanLabel` (the tile's real training samples from
`Mark_training/VCF_training_adjusted/tile_adjustment/v5.0.3samp/<tile>.samp.bin`), `maeReal`/`maeFixed`
and `corrReal`/`corrFixed` (accuracy of each prediction against those same real labels), `predictSeconds`.

## Per-model aggregate (mean over 31 tiles)

| Model | mean diffMean | mean diffStd | mean fracAbsDiffLe5 | mean fracAbsDiffGt20 | mean maeReal | mean maeFixed |
|---|---|---|---|---|---|---|
| baseline | -0.30 | 3.29 | 89.46% | 0.761% | 6.06 | 8.29 |
| rf | -0.26 | 2.34 | 91.84% | 0.071% | 6.09 | 8.15 |
| xgboost_tunedB | -0.24 | 2.11 | 92.82% | 0.104% | 10.63 | 11.03 |

(`corrReal`/`corrFixed` are `NaN` for the several tiles below whose training labels are uniformly 0 -
correlation is undefined with zero label variance - so they are left out of any correlation average
here; see the per-tile table instead.)

These averages by themselves look mild and would support adopting the fix. They are misleading: the
per-tile detail below shows the average is a mix of a few tiles with a large, real accuracy loss and many
tiles that are essentially untouched.

## Tiles where the fix clearly helps (map-level shift plus label-accuracy check both improve or hold)

- **h16v01**: the tile this fix was designed for. `diffMean` -4.19/-2.74/-3.03 (baseline/RF/XGBoost),
  `fracAbsDiffGt20` 6.6% - by far the largest map-level shift of the 31 tiles. `maeReal`/`maeFixed` are
  nearly identical (3.17 -> 3.19) and `corrReal`/`corrFixed` are both `NaN` because every training label
  on this tile is 0 (it is barren/tundra with no real tree cover) - the fix changes the *map* substantially
  without moving it away from the (all-zero) ground truth, consistent with the known over-prediction
  finding in `H16v01_TreeCover_OverPrediction_Investigation.md`.

## Tiles the fix leaves essentially untouched

h08v04, h09v04, h10v04, h12v12, h13v12, h19v12, h21v04, h27v04, h29v12, h30v12, and the three all-zero-
label tiles h12v01/h13v01/h24v04: `diffMean` within about +-0.1, `maeFixed` within a few hundredths of
`maeReal`, `corrFixed` within about 0.01 of `corrReal` where defined. These are tiles where the flagged
zenith periods are a small enough fraction of pixels that the NDVI-default substitution rarely changes
anything the model sees.

## Tiles where the fix clearly hurts accuracy against real training labels

Ten tiles show `maeFixed` roughly 1.3x to 2.9x `maeReal` and a correlation drop of 0.1 to 0.3, and the
pattern holds for all three models (not just baseline):

| Tile | Model | maeReal | maeFixed | corrReal | corrFixed |
|---|---|---|---|---|---|
| h18v04 | baseline | 5.50 | 15.43 | 0.965 | 0.878 |
| h18v04 | rf | 5.54 | 13.57 | 0.966 | 0.904 |
| h18v04 | xgboost_tunedB | 11.52 | 14.67 | 0.876 | 0.863 |
| h20v03 | baseline | 6.71 | 15.93 | 0.867 | 0.589 |
| h20v03 | rf | 6.82 | 13.77 | 0.867 | 0.712 |
| h20v03 | xgboost_tunedB | 13.78 | 16.16 | 0.681 | 0.586 |
| h20v02 | baseline | 6.91 | 13.02 | 0.908 | 0.758 |
| h20v02 | rf | 6.97 | 13.58 | 0.909 | 0.757 |
| h20v02 | xgboost_tunedB | 14.30 | 14.91 | 0.645 | 0.635 |
| h21v02 | baseline | 7.68 | 14.79 | 0.844 | 0.701 |
| h21v02 | rf | 7.78 | 15.33 | 0.843 | 0.697 |
| h21v02 | xgboost_tunedB | 15.73 | 16.99 | 0.661 | 0.636 |
| h23v02 | baseline | 6.51 | 11.96 | 0.953 | 0.873 |
| h23v03 | baseline | 7.57 | 13.04 | 0.863 | 0.750 |
| h24v03 | baseline | 6.12 | 10.15 | 0.872 | 0.797 |
| h18v03 | baseline | 9.50 | 17.82 | 0.520 | 0.272 |
| h11v02 | baseline | 6.04 | 8.67 | 0.649 | 0.573 |
| h11v03 | baseline | 12.46 | 15.98 | 0.092 | -0.011 |

(rf/xgboost_tunedB rows for h23v02, h23v03, h24v03, h18v03, h11v02, h11v03 show the same direction and
similar magnitude; omitted here for length, present in the full CSV.)

These are ordinary vegetated tiles with real, non-degenerate training labels (`meanLabel` 26 to 55,
tens of thousands to hundreds of thousands of samples) - unlike h16v01, they are not the structural,
near-zero-cover failure case the fix targeted. Substituting a fixed NDVI default on their flagged
zenith-marginal days appears to override otherwise-valid observations, moving the prediction away from
the tile's real measured tree cover rather than toward it.

## What this means for the open question

The plan asked two questions before this fix could be considered a real candidate: does it touch any
tile besides h16v01, and is the answer stable across models. Both are now answered, and neither supports
adopting the fix as designed:

1. **Yes, it touches many more tiles than h16v01** - 21 of the 31 flagged tiles show some map-level
   shift, and at least 10 show a real, substantial loss of accuracy against ground truth.
2. **The degradation is consistent across all three models**, so it is not an artifact of one model's
   sensitivity - it is a property of the NDVI-default substitution itself on these tiles.

This is still a diagnostic result, not a recommendation to change code (per standing instructions, no
code has changed and nothing is adopted without the project lead's decision). But it changes the
conclusion the two-tile test suggested: a **zenith-only, tile-blind NDVI default is not a safe general
fix** - it trades a large, correct improvement on the one tile it was built for against a real accuracy
loss on roughly a third of the tiles it was tested on. Any path forward would need to scope the
substitution more narrowly than "any zenith-marginal period on any flagged tile" - for example, limiting
it to pixels/tiles that look like h16v01's failure mode (near-total, tile-wide unfiltered blackout at
extreme latitude with no real vegetation) rather than applying it wherever Stage 0's cheap per-period
threshold was crossed.

## Zenith cutoff used, and a codebase discrepancy found while checking it

The scan and the fix both used `ZENITH_CUTOFF = 65` (a pixel/day is "good" if `0 < solz < 65`, raw HDF
degrees - the diagnostic reads `solz` directly with no scale factor, and confirmed
`ProductType.solarZenithScaleFactor` defaults to `1.0` and is never overridden, so comparing raw HDF
values to 65 is unit-consistent, not a bug). This value was carried over unchanged from the original
2-tile test; it was not re-derived or re-checked against the shipped code before the 31-tile run.

**Checking it now against `Git_delivery/modis_vcf` turned up three different zenith-cutoff constants in
the codebase, for three different purposes, which is worth flagging to the repository owner regardless of
this diagnostic's outcome:**

- **`BandDayFile.ZENITH_CUTOFF = 65`** (`model/BandDayFile.py:17`) - this is the one that actually matters
  here. It is passed into `createQaMask`/`createCloudMask` (`model/ProductTypeMod44.py`) from
  `CompositeDayFile.py:171,176`, and is production's real "is this day usable for compositing" test - the
  same test the diagnostic's `solz_pass()` replicates. **The diagnostic's 65 matches production exactly**;
  it is not a mismatch.
- **`BandDayFile.VALUE_CLAMP_ZENITH_CUTOFF = 70`** (`model/BandDayFile.py:28`) - a different, unrelated
  purpose: clamping Band_31's *raw reflectance value* to a fill code when `solz > 70`, mirroring C's
  `fix_bands.c` (see that file's own comment, lines 21-27). It has nothing to do with which days are
  usable for compositing NDVI - it is not a looser or stricter version of the QA cutoff, it is a value
  clamp on a single band (`BandDayFile.py`'s own comment above the constant, lines 21-27, explains it
  mirrors C's `fix_bands.c` clamp, and that the clamp is dead code for Bands 1-7 and only actually fires
  for Band_31).
- **`view/stateDump.py:206`, `zenithCutOff = 72`** - a standalone debug/inspection script that
  re-implements the exact same QA-mask formula as `createQaMask` (cloud/shadow/aerosol/adjacency/zenith)
  but with its own hardcoded `72`, not a reference to `BandDayFile.ZENITH_CUTOFF`. This looks like
  independent drift: if the production cutoff ever changes, this debug tool silently stops matching it.
  Not touched by this diagnostic, but a real inconsistency worth a note to the repository owner.

**So there is no error in this validation's use of 65** - it is the one the shipped compositing/QA logic
actually uses. But per the user's request, `70` (the value-clamp constant, applied here as an alternate,
looser "day is good" threshold rather than its original Band_31 clamp role) is being tried as a sensitivity
check: does raising the "day is good" bar from 65 to 70 degrees reduce the false-positive damage seen on
the 10 tiles above, while still fixing h16v01? See `Zenith_Only_Fix_Cutoff70_Retest_Plan.md` for the plan
(not yet executed).

## Where things are

Raw per-node JSONL results, driver, and launch scripts: same scratchpad as the rest of this
investigation (see `Zenith_Only_Fix_63Tile_Plan.md`, "Progress and resume info", for the path). This
document's consolidated CSV is also copied into this `Reports/` directory
(`stage3_consolidated_31tiles_3models.csv`) so it does not depend on the scratchpad's lifetime.
