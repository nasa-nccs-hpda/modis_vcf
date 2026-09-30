# Zenith-only NDVI fix, 63-tile check across three models: plan

**Status: finished 2026-09-28 18:29 (all three node jobs done); results in `Zenith_Only_Fix_63Tile_Results.md`, 70-degree follow-up in `Zenith_Only_Fix_Cutoff70_Retest_Results.md`.** (Originally: Stages 1-3 running 2026-09-28 16:30, detached, on forest203/204/205.) Written 2026-09-28,
after the h16v01 (over-prediction) and h12v09 (validation, PASS) two-tile test of the zenith-only
NDVI-default fix (`H16v01_TreeCover_OverPrediction_Investigation.md`, section "Zenith-only NDVI-default
variant"). Stage 0 (below) found 31 of 63 tiles meaningfully affected.

## Progress and resume info (read this first if resuming)

- **Stage 0 done**: 31 of 63 tiles flagged (any period > 0.01% zeroed); full per-tile-per-period table in
  `stage0_zenith_scan_63tiles.csv` (scratchpad, see path below).
- **Stages 1-3, driver bug found and fixed 2026-09-28 16:08-16:25**: the first design pre-symlinked all of
  a tile's other real, cached metric files to avoid recomputing them - but NDVI-*derived* metrics
  (`CGreenMean13-NDVI`, `CUnsMean13-NDVI`, etc.) are cached in their own files and are never recomputed
  once present, so this silently produced output identical to the real, unmodified prediction (caught by
  a smoke test on h16v01 before scaling up - `meanFixed == meanReal` exactly, diff 0.0 everywhere).
  **Fixed**: only the tile's real `2-Composites` directory is symlinked (a read-only input, safe to
  reuse); every metric file under `3-Metrics` is left for `predict_tree_cover_model.py` to compute fresh
  from the modified NDVI - expensive (~13-14 min) only for each tile's *first* model call (always
  `baseline`), after which `rf` and `xgboost_tunedB` reuse that now-correctly-fixed cache (fast, ~1-6
  min). Re-validated on h16v01 after the fix: baseline mean 5.66 -> 1.47 (matches the original two-tile
  test), RF 4.44 -> 1.70, XGBoost tunedB 5.05 -> 2.07 - all three show the same pattern, training-label
  accuracy essentially unchanged. A second bug (XGBoost's saved model has no `estimators_` attribute,
  crashing a logging line in `predict_tree_cover_model.py`) was worked around with a patched copy of that
  script in the scratchpad (`predict_tree_cover_model_patched.py`) rather than editing the shared
  original in the run root.
- **Launched 2026-09-28 16:30**, fully detached (`setsid nohup ... & disown`, verified parent PID 1 on
  all three nodes), 30 remaining tiles (h16v01 already done in the smoke test) split 10/10/10 across
  forest203/204/205, 3 concurrent tile-workers per node (9-way total), `nJobs=4` per predict call.

**Where things are** (all under the same scratchpad as the two-tile test):
`/panfs/ccds02/nobackup/people/mcarrol2/.nccstmp/claude-267298826/-panfs-ccds02-nobackup-projects-ilab-projects-MODIS-VCF-pge61-08312026-pge61/cfb2496e-dede-49f4-9d80-ded423d651cb/scratchpad/`
- Driver: `stages123_driver.py`; per-node launcher: `run_stages123_node.sh <nWorkers> <tile...>`.
- Status: `status/stages123_<hostname>.status` / `.done` / `.failed` (one set per node).
- Per-tile logs: `expB/logs/`. Per-tile-per-model results, one JSON object per line, written only after
  all 3 models finish for a tile: `stages123_results_<hostname>.jsonl` (plus the smoke test's own h16v01
  result, saved separately as `stages123_results_forest203_smoketest_h16v01.jsonl` so the real
  forest203 batch didn't overwrite it).
- **Stage 3 done 2026-09-28 (after all three `.done` markers appeared 18:20-18:29):** the three
  `stages123_results_*.jsonl` files plus the h16v01 smoke test were concatenated (93 rows, 31 tiles x 3
  models, all `ok: true`) and summarized. Result: the fix resolves h16v01 but degrades label accuracy on
  at least 10 other tiles, consistently across all three models - not the narrow effect the two-tile test
  suggested. Full writeup: `Zenith_Only_Fix_63Tile_Results.md`.

## Goal

The two-tile test showed the fix resolves most of h16v01's over-prediction and provably does not touch
h12v09. Before this could be considered a real candidate (still not decided - no code has changed), two
questions remain: does the fix touch **any other** training tile besides h16v01, and does that answer
change across the three models that share an identical 64-predictor list - baseline
(`BaggingRegressor(DecisionTreeRegressor)`), Random Forest, and XGBoost tunedB (confirmed identical
`pcttreeTopN.bin`, byte-for-byte, across all three; Monte Carlo models are out of scope - different,
smaller-overlap predictor sets, not requested this round).

## Design: staged, to avoid unnecessary large compute

### Stage 0 - cheap structural scan, all 63 tiles, no model, no prediction

For every tile in `manifests/tiles.txt` (63 tiles), compute the same zenith-only zero-mask used on
h16v01/h12v09 (zero NDVI per pixel/period where no archived observation in that tile/period had
`0 < solz < 65`) directly against the raw MOD44CQ archive, per period, and report the percentage of
pixels affected. The 8-period day-of-year window structure (`Day_2019065` ... `Day_2020033`) is
tile-independent - confirmed from `ProductType.yearOneDays`/`yearTwoDays` and
`CompositeDayFile._getDaysToFind()` - so this generalizes from the two tiles' hand-written window lists
by globbing the archive for each tile/day instead (`MOD44CQ.A<year><doy>.<tile>.*.hdf`), handling a
missing file as `None` exactly as today's scripts already do.

No model is loaded and no metric is computed in this stage - purely archive reads and a boolean mask,
the read-only half of `h16v01_make_ndvi_zenith_only.py`. Output: one table, tile x period x percent
pixels affected, plus a tile-wide total. This is what actually decides which tiles matter to Stages 1-3
- it replaces the assumption that only extreme-high-latitude tiles are touched with a real per-tile
number (h12v09, a normal-latitude forest tile, already measured at exactly 0.00% in every period -
Stage 0 puts every other tile on the same footing before any prediction is run).

### Stage 1 - build the fix + baseline predict, only for tiles Stage 0 flags (>0% anywhere)

For each flagged tile: write the modified `MOD44-NDVI.bin` (same method as h16v01/h12v09), then instead
of today's approach (bind `2-Composites` read-only and let the predict script recompute all ~21 metric
files from scratch, which cost about 13 minutes for h12v09 alone) - **symlink the tile's other real,
already-precomputed `3-Metrics` files** (`Python/metrics/<tile>/<tile>/2019/3-Metrics/`, confirmed
present and complete for all 63 tiles, 21 files/tile, about 7-8 GB) read-only into a fresh per-tile
scratch `metricsDir`, and only the NDVI file is the modified one. `Metrics.getMetric()` only computes a
metric if its file is missing (`Metrics.py:262`, `if not metricFileName.exists()`) - with every other
file already present via symlink, nothing gets recomputed, only NDVI differs. Then run the baseline
model's prediction (`predict_tree_cover_model.py`, unmodified) against that `metricsDir`; this also
leaves it fully populated with real cached metrics for Stage 2 to reuse.

### Stage 2 - Random Forest and XGBoost tunedB, same flagged tiles, reusing Stage 1's `metricsDir`

Same predict script, only `--model`/`--topN` changed to
`regtree_alt_models_20260927/Python/models/{rf,xgboost_tunedB}/pcttree.bin` /
`pcttreeTopN.bin`. Since Stage 1 already populated the tile's `metricsDir` with every metric file
(NDVI fixed, everything else real), these runs read cache only - no recomputation, no archive access
needed at all.

**Model-size caveat, affects concurrency, not correctness:** baseline is 13.4 GB, Random Forest 14.6 GB,
XGBoost tunedB only 10 MB (joblib-pickled tree ensembles are far smaller than bagging/RF's fully
materialized trees). Concurrency below is sized around the two large ones.

### Stage 3 - compare, every flagged tile x 3 models

Same method as `h12v09_compare_zenith.py`, generalized across tiles: whole-tile diff stats (mean,
median, std, fraction `|diff|<=5`, fraction `|diff|>20`) against each model's real, unmodified
prediction, and accuracy against that tile's real training labels - every one of the 63 tiles has a
`<tile>.samp.bin` file under
`Mark_training/VCF_training_adjusted/tile_adjustment/v5.0.3samp/` (confirmed, 63 files present) - MAE,
bias, correlation, real vs. fixed, by label bin. One consolidated table across tiles x models, plus the
per-tile detail logs Stage 1/2 already produce.

## Resource check (done 2026-09-28 15:36, read-only - no jobs started)

| Node | Load avg (1 min) | Memory | Notes |
|---|---|---|---|
| forest201 | 9.30 | 107 GB reported free | Another user (another user) has a job pushing load near saturation on 12 cores - **avoid**. |
| forest202 | 3.85, falling from a recent peak | 40 GB free / 105 GB available | another user running an `mv` (large file copy) recently; **use cautiously, re-check before launch**. |
| forest203 | 0.01-0.81 | 104 GB available | Idle - where today's h16v01/h12v09 work ran; safe to reuse. |
| forest204 | 0.16-0.71 | 99 GB available | Idle - safe. |
| forest205 | 0.01-0.13 | 104 GB available | Idle - safe. |

**Recommendation: use forest203, forest204, forest205.** Other users share this cluster (already noted
in `README_START_HERE.md`'s Environment section) - re-check load immediately before actually launching
anything, since this snapshot will be stale by execution time.

## Concurrency and time estimate

- **Stage 0** (scan, no model, I/O-bound not memory-bound): can run with real concurrency, e.g. 4-6
  tiles at once per node across the 3 idle nodes (~12-18 concurrent). All 63 tiles are cheap enough to
  just run outright rather than sample. Estimated 1-2 minutes/tile -> **roughly 5-10 minutes total**.
- **Stage 1** (baseline, flagged tiles only, symlinked metrics not recomputed): budget about 20 GB/job
  including working memory around the 13.4 GB model, so **3 concurrent jobs/node** (~60 GB) on
  forest203/204/205 = 9 concurrent. Estimated 3-5 minutes/tile (archive read + symlink + one model's
  predict, no metric recomputation - versus today's 13-14 minutes/tile with recomputation).
- **Stage 2** (RF, same concurrency budget as Stage 1 since it's also ~14.6 GB; XGBoost tunedB can run
  at much higher concurrency, 6+/node, since it's 10 MB): metrics already cached, so this should be
  faster than Stage 1 per tile, roughly 1-2 minutes/tile.
- **Stage 3** (compare): numpy-only, CPU-light, a few minutes total regardless of tile count.

**Total, contingent on how many tiles Stage 0 flags** (the actual driver of cost - not yet known,
Stage 0's job is to measure it rather than assume it): a rough **1-2 hours wall clock** if, as expected
from the mechanism (structural zenith blackout at extreme latitude), only a handful of high-latitude
tiles are affected; more if the effect turns out broader than the two tiles checked so far. Stage 0's
result will be reported before Stages 1-3 (where essentially all the compute cost lives) are launched.

## What this does not do

- No changes to the real pipeline anywhere - everything reads real, unmodified composites/metrics/
  models read-only, except each tile's own fresh scratch copy of the modified NDVI file.
- Does not extend to the Monte Carlo top-64/top-20 models (out of scope this round, per instruction).
- Does not extend to years other than 2019, or to the non-vegetated/bare model.
- Does not propose adopting the fix in the real pipeline - still a diagnostic/validation exercise, same
  as the two-tile test.

## Execution

Not started. If approved, Stage 0 runs first (all 63 tiles, fully detached per the project's standing
convention - survives an SSH/VS Code disconnect) and its results are reported back before Stages 1-3 are
launched, since Stage 0's flagged-tile count is what the time estimate above actually depends on.
