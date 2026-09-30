# Plan: retest the zenith-only NDVI-default fix with a 70-degree cutoff

**Status: finished 2026-09-28 21:09 as a two-tile run (h16v01, h20v03); results in `Zenith_Only_Fix_Cutoff70_Retest_Results.md`. No code in the shared repos changed.** Written 2026-09-28 following the
31-tile x 3-model validation in `Zenith_Only_Fix_63Tile_Results.md`.

## Why

`Zenith_Only_Fix_63Tile_Results.md` found the fix (zero a pixel/period's NDVI if no raw day had
`0 < solz < 65`, replacing it with a default) fixes h16v01 but degrades label accuracy on at least 10 of
the other 30 flagged tiles, consistently across all three models. The `65` came from the two-tile test and
was never re-derived. Checking the codebase while answering where `65` came from found it exactly matches
`BandDayFile.ZENITH_CUTOFF`, the constant production's real `createQaMask`/`createCloudMask` use to decide
whether a day is usable at all - so the diagnostic is not stricter or looser than production's own "day is
good" test as it stands. But a different constant, `BandDayFile.VALUE_CLAMP_ZENITH_CUTOFF = 70`, exists
in the same file for a different purpose (clamping Band_31's raw reflectance, not deciding day usability).
The user asked to try `70` as the "day is good" bar instead of `65`, as a sensitivity check: does loosening
the bar from 65 to 70 degrees (accepting more raw days as usable, so fewer periods get zeroed) reduce the
false-positive damage on the 10 hurt tiles while still resolving h16v01's structural failure (which was 5
of 8 periods **100%** unfiltered tile-wide - a near-total blackout, not a marginal one)?

This is explicitly a sensitivity check on an already-negative result, not a rescue attempt: if 70 still
hurts the same tiles, or stops fixing h16v01, that closes the question; if it removes the damage while
keeping h16v01's fix, that is a real, reportable finding worth its own writeup.

## Method

Reuse the existing scratchpad infrastructure from the 65-cutoff run unchanged except for one constant:

- `stage0_scan_zenith_63tiles.py`, `stages123_driver.py`, and the two single-tile smoke-test scripts
  (`h16v01_make_ndvi_zenith_only.py`, `h12v09_compare_zenith.py`'s companion) all define
  `ZENITH_CUTOFF = 65` as a top-level constant read by `solz_pass()`. Copy each script (do not edit the
  originals - they document the 65-cutoff run and should stay as-is for the record) to a `_cutoff70`
  variant with only that one line changed to `ZENITH_CUTOFF = 70`. All other logic (symlinking the real
  `2-Composites` dir, leaving `3-Metrics` for fresh computation, the three models, the per-tile-per-model
  compare against `<tile>.samp.bin`) stays identical, so results are directly comparable to the 65-cutoff
  table.
- Output goes to new files/dirs (`*_cutoff70` suffix or a sibling directory) - never overwrite the
  65-cutoff run's `stages123_results_*.jsonl`, `stage0_zenith_scan_63tiles.csv`, or
  `stage3_consolidated_31tiles_3models.csv`. Both runs need to exist side by side to compare.

## Stages

**Stage 0 (cheap check first, per this project's standing practice of checking impact cheaply before an
expensive run):** two things, in order, before committing to any 31-tile-scale compute:

1. Re-run the fast, I/O-bound 63-tile scan (`stage0_scan_zenith_63tiles.py`, ~5-10 min total last time)
   with `ZENITH_CUTOFF = 70`. This alone answers the most important question cheaply: **does h16v01 still
   get any periods zeroed at 70 degrees?** If loosening the bar to 70 means h16v01's periods no longer
   zero out (i.e., h16v01's failing days all had `solz` between 65 and 70), the fix stops fixing the tile
   it was built for, and there is no reason to run Stages 1-3 at all. If h16v01 still zeroes out (expected,
   since its documented failure is a near-total blackout, not a marginal 65-70 degree band), also check
   whether the 31-tile flagged list shrinks - some of the 10 "hurt" tiles may drop out of the flagged set
   entirely at 70, in which case they need no Stage 1-3 rerun (their real, unmodified prediction is
   untouched by construction).
2. **Single-tile smoke test at 70** on h16v01 (reusing the existing pattern from the 65-cutoff run):
   confirm the fixed prediction still differs meaningfully from the real one and still moves toward the
   labels (or at least does not move away, given `meanLabel` is 0 there) before scaling up.

**Stages 1-3, only if Stage 0 still flags h16v01 and at least one of the 10 hurt tiles:** run the same
per-tile pipeline as before (symlink `2-Composites`, let `3-Metrics` recompute from the cutoff-70 NDVI,
predict with all three models, compare against real prediction and against `<tile>.samp.bin`) - but
**scope it to the tiles that matter for the question being asked**, not necessarily all 31:
- h16v01 (must still be fixed for this to be worth anything).
- The 10 tiles that got worse at 65: h11v02, h11v03, h18v03, h18v04, h20v02, h20v03, h21v02, h23v02,
  h23v03, h24v03 (does loosening to 70 remove or reduce their damage?).
- Any tile Stage 0 newly flags at 70 that wasn't flagged at 65 (unexpected, but Stage 0's job is to check
  rather than assume - loosening the "day is good" bar should only ever shrink or hold the flagged set,
  never grow it, since fewer periods get zeroed; a tile appearing that wasn't there before would mean a
  bug in the cutoff-70 scripts, not a real result).
- The remaining ~20 tiles that were untouched at 65 do not need to be rerun: since 70 is a *looser* bar
  than 65 (accepts everything 65 accepted, plus more), any tile untouched at 65 is guaranteed untouched at
  70 too - rerunning them would just reproduce `diffMean == 0`. This cuts the compute from 31 to about 11
  tiles x 3 models, roughly a third of the previous run's cost.

## Estimate

Based on the 65-cutoff run's actual timings: Stage 0 ~5-10 minutes (cheap, all 63 tiles). Stage 1
(baseline, first model call per tile, computes metrics fresh) ~3-14 minutes/tile depending on whether the
driver bug fix (real `2-Composites` symlink only) is reused correctly; Stage 2 (RF, XGBoost tunedB, cached
metrics) ~1-6 minutes/tile. For ~11 tiles x 3 models on 3 nodes (forest203-205, same concurrency budget as
before - baseline/RF at 3 concurrent jobs/node for the ~14 GB models, XGBoost tunedB at higher
concurrency): **roughly 30-45 minutes wall clock**, well under a third of the original 1-2 hour run, since
scope is about a third the tiles and the expensive per-tile metric-recompute step is unchanged in kind.

Re-check node load immediately before launching (per environment notes in `README_START_HERE.md` - other
users share these nodes and load changes day to day).

## What would change the conclusion

- If h16v01 stops being fixed at 70: the cutoff can't be loosened without losing the one confirmed benefit
  - the plan ends here, and the 65-cutoff conclusion (not a safe general fix) stands as the final word on
  this fix design.
- If h16v01 stays fixed and the 10 hurt tiles' `maeFixed`/`corrFixed` return close to their `maeReal`/
  `corrReal` values: this is a materially better result and worth its own results document and a decision
  point on whether to propose adoption at 70 instead of 65 - still requiring the project lead's sign-off
  and the repository owner's involvement per the standing sync-hold condition.
- If h16v01 stays fixed but the 10 tiles are still damaged, just less severely: report the gradient
  honestly: loosening the cutoff trades off strength of the fix against collateral damage, and neither 65
  nor 70 are then a clean answer - the deeper issue (a tile-blind rule applied regardless of whether the
  tile's failure mode resembles h16v01's structural blackout) remains unaddressed, and the recommendation
  from the 65-cutoff results doc (scope the substitution by failure pattern, not by a single global zenith
  threshold) still holds.

## Standing conditions (unchanged, restated for this plan)

Detached launch (`setsid nohup ... & disown`, verified parent PID 1) if Stages 1-3 run, so it survives
disconnect; status/done markers per node; nothing outside the sync workspace and its scratchpad is
overwritten; no code in the shared repos is touched; nothing is adopted without the project lead speaking
to the repository owner first.

## Awaiting approval

Per this project's standing practice (plans written and approved before anything is executed), this plan
itself does not launch anything. Confirm before Stage 0's scan is run.

## Progress and resume info (added 2026-09-28 20:35)

The plan was approved and the run was scoped down to two tiles: h16v01 (the tile the fix was built for)
and h20v03 (one of the 10 tiles the fix hurt at 65; a quick single-tile check, `check_h20v03_cutoff70.py`,
was run first). The scripts are `_cutoff70` copies with only `ZENITH_CUTOFF = 70` changed. They live in the
scratchpad `/panfs/ccds02/nobackup/people/mcarrol2/.nccstmp/claude-267298826/-panfs-ccds02-nobackup-projects-ilab-projects-MODIS-VCF-pge61-08312026-pge61/6bad0349-fa57-4839-9b0c-2ab4012c7938/scratchpad`
(`stages123_cutoff70_driver.py`, `run_stages123_cutoff70_node.sh`). The originals from the 65-cutoff run
are untouched.

- **h16v01 on forest203**: launched 20:21, finished 20:42 (rc=0).
- **h20v03 on forest204**: the 20:21 launch was held up behind a hung forest203 `ssh`; a second session
  relaunched it at 20:31, then the original command's queued launch fired at 20:42 and created a
  duplicate. Both were killed and one clean run started at 20:46:51, finished 21:09:14 (rc=0). Full
  account in the results doc.

Status: `status/stages123_cutoff70_<host>.status` and `.done`/`.failed` markers. Results:
`stages123_cutoff70_results_<host>.jsonl`. Logs: `expC/logs/`.
Actual: about 21 minutes (h16v01) and 22 minutes (h20v03) per tile for all three models.

Note: when launching detached jobs over `ssh` here, wrap each call in `timeout 30 ssh -n ...` and use
one call per node. A hung `ssh` stops everything that comes after it in the same command.
