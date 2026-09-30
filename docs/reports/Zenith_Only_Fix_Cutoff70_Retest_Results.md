# Results: zenith-only NDVI-default fix retested with a 70-degree cutoff

**Status: finished 2026-09-28 21:09. Diagnostic only; no code changed, nothing adopted.** Plan:
`Zenith_Only_Fix_Cutoff70_Retest_Plan.md`. The 65-degree run this is compared against:
`Zenith_Only_Fix_63Tile_Results.md` (numbers from `stage3_consolidated_31tiles_3models.csv`).

## Bottom line

Loosening the "day is usable" zenith bar from 65 to 70 degrees **keeps the fix working on h16v01 and
reduces, but does not remove, the damage on h20v03**. This is the third outcome the plan anticipated:
the cutoff trades fix strength against collateral damage, and neither 65 nor 70 is a clean answer. The
underlying problem, a single tile-blind rule applied whether or not a tile's failure looks like h16v01's
structural blackout, is not addressed by changing the cutoff. The recommendation from the 65-degree
results (scope the substitution by failure pattern rather than by a global zenith threshold) still holds.

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

## Scope

Scoped down from the plan's ~11 tiles to two, at the project lead's direction:

- **h16v01**, the tile the fix was built for. Earlier work (`H16v01_TreeCover_OverPrediction_Investigation.md`,
  "Checked: does the earlier 65-vs-70 zenith-cutoff decision explain this?") had already shown that 70
  degrees does not change h16v01's QA-pass rate, so this run mainly confirms the full pipeline agrees.
- **h20v03**, one of the worst-hit of the 10 tiles the fix hurt at 65 degrees (MAE against labels roughly
  doubled).

The other nine hurt tiles (h11v02, h11v03, h18v03, h18v04, h20v02, h21v02, h23v02, h23v03, h24v03) were not
rerun, so this result does not say how they would behave at 70.

Same pipeline, models and label comparison as the 65-degree run; only `ZENITH_CUTOFF` changed.

## Cheap check first: how much of h20v03 gets zeroed

Percentage of h20v03 pixels whose NDVI is zeroed per compositing period (no raw day with `0 < solz < cutoff`):

| Period | 65 deg | 70 deg |
|---|---|---|
| Day_2019065 - Day_2019257 (5 periods) | 0.00% | 0.00% |
| Day_2019305 | 96.34% | 80.30% |
| Day_2019353 | 100.00% | 95.78% |
| Day_2020033 | 20.29% | 5.99% |

So 70 degrees shrinks the zeroed area on h20v03 but leaves most of the two winter periods zeroed. This
predicted the partial recovery seen below.

## h16v01: fix still works

`meanLabel` is 0 at h16v01, so correlation is undefined; the useful checks are the drop in mean tree cover
and MAE.

| Model | Mean real | Mean fixed, 65 | Mean fixed, 70 | MAE real | MAE fixed, 65 | MAE fixed, 70 |
|---|---|---|---|---|---|---|
| baseline | 5.66 | 1.47 | 1.48 | 3.169 | 3.195 | 3.179 |
| rf | 4.44 | 1.70 | 1.69 | 3.164 | 3.180 | 3.169 |
| xgboost_tunedB | 5.05 | 2.07 | 2.06 | 3.256 | 3.311 | 3.261 |

The over-prediction is removed just as at 65 (mean within 0.01), and MAE is marginally closer to the real
prediction.

## h20v03: damage reduced, not removed

Against 167,914 training-label pixels (`meanLabel` 45.6):

| Model | MAE real | MAE fixed, 65 | MAE fixed, 70 | Corr real | Corr fixed, 65 | Corr fixed, 70 |
|---|---|---|---|---|---|---|
| baseline | 6.71 | 15.93 | 13.77 | 0.867 | 0.589 | 0.675 |
| rf | 6.82 | 13.77 | 11.95 | 0.867 | 0.712 | 0.759 |
| xgboost_tunedB | 13.78 | 16.16 | 15.22 | 0.681 | 0.586 | 0.630 |

70 degrees recovers roughly a fifth to two-fifths of the lost accuracy, consistently across all three
models. For baseline and RF, MAE is still about double the real prediction's. Mean tree cover moved
slightly further from the real prediction at 70 (baseline 29.48 -> 30.34, real 28.61), so the recovery
is in pixel-level agreement, not in the tile mean.

## Run mechanics and an incident

Scripts are `_cutoff70` copies of the 65-degree run's scripts, with only the constant changed, in the
scratchpad `/panfs/ccds02/nobackup/people/mcarrol2/.nccstmp/claude-267298826/-panfs-ccds02-nobackup-projects-ilab-projects-MODIS-VCF-pge61-08312026-pge61/6bad0349-fa57-4839-9b0c-2ab4012c7938/scratchpad`
(`stages123_cutoff70_driver.py`, `run_stages123_cutoff70_node.sh`). Results: `stages123_cutoff70_results_forest203.jsonl`
(h16v01) and `stages123_cutoff70_results_forest204.jsonl` (h20v03). The 65-degree run's files are untouched.

- h16v01 on forest203: 20:21-20:42, one clean run, rc=0.
- h20v03 on forest204: the 20:21 launch command started forest203 first, and that `ssh` call did not return
  for about 20 minutes, so the forest204 launch in the same command stayed queued. The original session
  appeared disconnected, and a second session relaunched h20v03 at 20:31. At 20:42 the original
  command's `ssh` finally returned and its queued forest204 launch started a **duplicate** h20v03 job
  writing to the same paths. The original session detected this, killed both, deleted their partial
  outputs (the old status file was kept as `status/stages123_cutoff70_forest204.status.stale_duplicate_launch`),
  and started one clean run at 20:46:51, which finished 21:09:14 with rc=0. The reported h20v03 numbers
  come only from that clean run; the error in `expC/logs/predict_rf_h20v03.log` from 20:46 belongs to the
  killed duplicates, and that log was overwritten by the clean run.

Lesson for launches: one `ssh` per node, each wrapped in `timeout`, and before relaunching anything after a
disconnect, check whether the earlier session is actually still alive (its transcript under
`~/.claude/projects/` still being written) so two sessions do not manage the same run.
