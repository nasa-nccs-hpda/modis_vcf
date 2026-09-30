# Plan: feature importance from the 63-tile model, and a stand-alone Monte Carlo predictor selection

Written 2026-09-20, revised the same day with the user's answers and one validity requirement. **Status (updated 2026-09-20 20:50): every step in this plan has been run, steps 0 to 9; the results are in `Feature_Importance_Part_A_Results.md` and `Feature_Importance_and_Monte_Carlo_Results.md`.** The sections below are the plan as written, with each step's result added under it.

`$RUN` = `/explore/nobackup/people/mcarrol2/MOD44B_VCF_Development/FullRuns/full63_c_vs_py_20260919-0806`. The code referred to is the run-root copy, `$RUN/code/python_code/modis_vcf/`.

## Decisions recorded (2026-09-20)

1. **Tree cover only.** No bare models in this plan.
2. **Full sample.** Part A scores permutation importance on the whole held-out set (all 2,288,197 `xTest` rows), not a subsample.
3. **Monte Carlo subsample size, `minVarUsage` and caps: set after the timing pilot.**
4. **Fit and save the Monte Carlo final model.** It is saved in a new folder, never over the existing models.
5. **Split the work across the other compute nodes where possible** (section "Compute nodes").
6. **The test is valid only if it uses all the metric updates learned from the 63-tile run** (next section). This is a requirement on the data, and it changes the plan: see Part 0.
7. **Candidate pool confirmed: all 324 metrics** (72 unsorted plus 252 derived).
8. **Keep the extra metric rasters for now** (space is available); the plan has no delete step.
9. **The final model uses 64 predictors**, kept for a like-for-like comparison with the existing 64-predictor model; the 20-predictor result is reported as well.
10. **The five canary tiles are regenerated only if the comparison shows a difference.**
11. **h12v02 is added to the fresh-regeneration check** next to h09v05.
12. **The table-against-raster comparison is run on all 63 tiles.**
13. **For the canary tiles, a difference of "plus or minus a few points" is acceptable.** Read as: every cell within **3 native raster units** of the existing value, except the rank metrics (`CSortedRank*`), which must be exact because they are small integers where 3 units would hide real differences. The tolerance is a command-line option (`--tol`, default 0). **Confirmed by the user, 2026-09-20.** The fresh regeneration of h09v05 and h12v02 stays exact, since decision 13 concerns the canary tiles only.

## The two questions, answered

1. **Feature importance from the existing 63-tile model: yes.** `$RUN/Python/models/pcttree.bin` (13.5 GB) is a scikit-learn `BaggingRegressor` of 30 unpruned decision trees (joblib) on 64 predictors; the predictor
   list is saved next to it (`pcttreeTopN.bin`) and equals `$RUN/manifests/predictors_pcttree.txt`. A held-out set exists (`$RUN/Python/training/pcttree/xTest.parq`, 2,288,197 rows). Limit: it ranks only
   the 64 predictors C chose, and it describes the Python model, not C's Weka models.
2. **A stand-alone Monte Carlo: yes**, since `view/monteCarloSim.py` needs none of `makeMetrics`, `buildTraining`, the model fit or predict. But (a) the shipped code has defects, (b) a full-data run is far too
   heavy per trial, and (c) **the existing training table does not contain all the metrics**, so it cannot support a valid selection as it stands (next section).

## Validity requirement: the metrics must carry every update from the 63-tile run

**What "the updates" are.** The Python metrics used for the 63-tile run include these changes, each validated in `Reports/`:
- zenith cutoff 65 in `BandDayFile.py`; Bands 1-7 value clamp removed and Band_31 clamp added (`Bands17_Fix_Validation_Results.md`);
- QA bit reads: adjacency bit 6, aerosol bits 7-8 (`ProductTypeMod44.py`);
- the unsorted-band layout, band-major, verified against C's `metrics.bin`;
- **the C-mirror derived metrics** (`CAttributes.py`, `CMirrorMetrics.py`), which reproduce **all 252 of C's derived metrics exactly** from the 72 unsorted values (`OptionC_ImplementationB_Validation_Results.md`);
- the `v5.0.3samp` tree labels used for training (`FullRun_Launch_and_Status.md`).
The metric-read cache added to `Metrics.py` is a speed change with no intended numerical effect. The NDVI difference from C (finding 4) is a known, accepted difference, not an update.

**What was checked (read-only):**
- **The code did not change during the production metrics run.** No file under `$RUN/code/python_code` is newer than the production start marker (2026-09-19 12:42:45). Last modification times of the files that matter: `BandDayFile.py` 09-17,
  `ProductTypeMod44.py` 09-15, `CAttributes.py` 09-19 08:28, `CMirrorMetrics.py` 09-19 11:27, `Metrics.py` and `BuildTraining.py` 09-19 11:59, `rfFitFromPge61.py` 09-19 12:00.
- **58 of the 63 tiles' metrics were produced by that final code** (production run from 12:43). **Five tiles (h10v05, h11v04, h12v01, h18v04, h19v04) still hold metric files from the earlier canary phase, written 10:39 to 11:38,**
  before the 11:59 edit to `Metrics.py`. `CMirrorMetrics.py` (11:27) predates them, so their values should be the same, but that is inferred from the edit times, not proven.
- **The training table has only 116 predictor columns, and 208 of C's 252 derived metrics are missing.** The 116 are all 72 unsorted monthly band values plus 44 derived columns, which are exactly the derived metrics that
  C's tree list (64) or bare list (21) uses. The other **208** derived metrics that C computes are not in it. The saved predictor list matches the manifest, but its order does not match the column order of `X.parq`, so any
  ranking must map by name, never by position.

**Evidence about the existing 116 columns (checked 2026-09-20, read-only).** The 63-tile run did not run the Monte Carlo, so the 116 columns were not selected by it. They are what the run's predictor lists and the training-table
build put there. What has and has not been shown:
- *Shown, by direct comparison:* the 116 are 72 unsorted monthly band values and 44 derived metrics, and **all 44 derived columns are in C's tree list (40) or bare list (6); none is outside them.** The 39 columns in neither list are all unsorted values.
- *Shown, by direct comparison, for one tile:* on **h09v05, all 116 columns of the training table (268,039 rows, 31,092,524 cells) equal the existing metric rasters cell for cell**, read through the pipeline's own `Metrics` reader, with zero mismatches.
  So the table is exactly the existing rasters for that tile. The other 62 tiles were not compared.
- *Supporting, but not proof:* no code file changed after the production start, and 58 of 63 tiles' metrics were made by the final code.
- *Not shown:* that regenerating the metrics **reproduces** the existing rasters. That is what step 2 tests. An earlier draft of this plan said the regeneration "confirms" the 116 columns are identical; that was an expectation, not a result.
- *A flaw in the earlier draft, now corrected:* regenerating into the existing folders would reuse the cached composites, so the comparison could not detect a problem in the composite stage (the stage the zenith and QA-bit fixes act on). The check must use a **fresh output folder with no cached composites**, so the composites are recomputed from the daily inputs.

**Consequence.** A Monte Carlo on the existing 116 columns could only re-rank raw band values and the metrics C already picked. It could not show whether any of the other 208 metrics would rank higher, which is what
selecting "metrics based on feature importance" is for. So for the test to be valid the candidate pool must be **the complete corrected metric set: 72 unsorted plus 252 derived = 324 metrics, produced by the final code.**
This is my reading of "include all of the updates"; the plan's decisions section asks you to confirm it.

## Part 0. Rebuild the tree-cover training table with all 324 metrics

The plan cannot proceed to Part B without it. Part A (importance of the existing model) does not need it, because that model was trained on the 116-column table.

**Inputs already in place:** the per-tile composites are kept (`$RUN/Python/metrics/<tile>/<tile>/2019/{1-Days,2-Composites,3-Metrics}`), so `makeMetrics` reuses them and only has to compute the missing metrics; the labels and
the row selection come from the same `buildTraining` code; the run-root code copy is frozen.

**Steps:**
1. **Provenance checks first (cheap):**
   - (a) record a SHA-256 for every file in the run-root code copy now, and confirm none is newer than the production start marker (already true by modification time; the hashes make it checkable later);
   - (b) regenerate one of the five canary tiles (for example h10v05) from the final code into a scratch folder and compare the metrics it shares with the existing files, value for value. If they are identical, the five tiles are accepted; if not, those five tiles are regenerated;
   - (c) **regenerate h09v05 and h12v02 completely in fresh folders, with no cached composites** (the production run took about 31 and 35 minutes for these tiles with 20 metrics; timing for all 324 is measured here), producing all 324 metrics, and confirm that the 116 columns already in the training table are identical, cell for cell, to the new values for the same rows. h12v02 is included because it is the tile where the metric fixes mattered most. This is the regression check that proves the new table matches the model's table before anything else is built on it;
   - (d) extend the direct table-against-raster comparison that was done for h09v05 to the other 62 tiles (about 3 minutes per tile, run in parallel across the nodes).
2. **Generate all 324 metrics per tile** with `makeMetrics` (no metric subset, so all derived names), from the final code, for the 63 tiles, into a new folder under `$RUN/Python/` (not into `metrics/`). Disk: about 46 MB per metric raster, so the 208 extra metrics are about 9.6 GB per tile,
   **about 600 GB for 63 tiles if all are kept**. **They are kept for now (decision 8);** free space is to be confirmed just before the run.
3. **Build the training table** with `buildTraining` for tree cover: same labels, same rows. Expected 11,440,984 rows, 324 predictor columns, plus the label.
4. **Split it exactly as the model's table was split.** Verified 2026-09-20 (read-only): the existing split files were built by `$RUN/code/validation/build_xy_from_training.py` (`train_test_split(test_size=0.2, random_state=0)`) over the per-tile files **in the order `MasterTraining` lists them, which is an unsorted directory listing** (the first file is `h08v05`, before `h08v04`). Their index (`__index_level_0__`) is the **integer row position, not a tile-x-y key**, so membership cannot be matched by key. The split depends only on the row count and the seed, so **the new table is concatenated in the recovered file order and split with the same seed**, and the result is checked against the old files. The recovered order was checked: 63 files, 11,440,984 rows in total, and at all 10 file boundaries tested the first row equals `X.parq` at the cumulative offset (`p0_recover_file_order.py` repeats the check for every file before it writes the order).
5. **Acceptance test for the new table:** 324 columns, no missing names, the 116 shared columns identical to the existing table, row order and labels identical.

Time: the earlier 63-tile Python metrics run took about 2.6 hours on three nodes with 4 tiles per node for 20 metrics; computing the extra 208 derived metrics from cached composites is expected to be much cheaper than that,
but **I cannot give a reliable figure; the h09v05 check in step 1(c) measures it.**

## What the Monte Carlo does (from reading `model/MonteCarloSim.py`)

Each trial draws 10 predictors at random from all the training predictors, fits a 30-tree `BaggingRegressor` using only those 10, and measures each predictor's **permutation importance** (how much the score drops
when that column is shuffled). Trials repeat until every predictor has been used at least `minVarUsage` times (default 10), enough predictors have non-zero importance, and the top N by average importance are valid
(default N = 20). It then fits a final model on those N and writes `SortedMetrics-<type><timestamp>.txt` (all predictors, ranked), `trials/Trial-*.bin` (one per trial; a rerun resumes from them), and `<type>.bin` and
`<type>TopN.bin`. This is a different measure from the model's own importance: it is measured on random 10-predictor subsets, so it reflects a predictor's value in small combinations, not its role in the 64-predictor model.

## Defects in the Monte Carlo code

All are present in rlgill's original (`model/MonteCarloSim.py`, `view/monteCarloSim.py`); the sync and run-root copies only changed the model class to `BaggingRegressor`. None has been changed. Listed so nobody
is surprised, not as criticism: the class looks mid-refactor, and its unit test expects an older interface.

| # | Where | Problem | Consequence |
|---|---|---|---|
| 1 | `view/monteCarloSim.py` lines 78-79 | Calls `mcs.saveFinalModel(finalModel)`, but `saveFinalModel(self)` takes no argument and `run()` returns `None` | `TypeError` after `run()` ends. Trials and the sorted list are already written, but the final model is not saved. |
| 2 | `run()` | `topnMet` is read before it is assigned when the top-N check fails; `allConditionsMet = True` runs whether or not the top-N was valid | An `UnboundLocalError` on the first failed check; otherwise an empty final model. Trials are kept. |
| 3 | `_runTrials()` | `numCompleted = 0` is reset every batch | The `maxTrials` guard (default 20000) never stops the loop. There is no cap on run time. |
| 4 | `_runOneTrial()` | Permutation importance is computed on all of `X` (train and test rows together) | Importance is partly measured on data the trees were fit to. |
| 5 | `_initXY()` | The non-default branch uses `self._masterTraining`, which does not exist | Unused with the default setting. |
| 6 | `tests/test_MonteCarloSim.py` | Expects `run()` to return a model and `saveFinalModel()` to return a path | Tests are out of date; not run. |

## Cost of a full-data Monte Carlo, with the 324-metric pool

A trial fits 30 unpruned trees on 9.15 million rows and then predicts 11.4 million rows about 50 times (10 predictors x 5 repeats). With **324 candidates** and 10 per trial, using each at least 10 times needs at least
324 trials and, because the draws are random, about 690 to 780 (see "Trial count"). **I cannot give a per-trial time without measuring it**; the pilot does that, and subsampling and the node split (below) are how the total is brought down.

## Step 2 result (2026-09-20, 10:10 to 11:03)

**Every check passed.** Runs were detached on forest201 to 205 (parent PID 1). Outputs: `$RUN/Python/metrics_324_fresh/<tile>/`, `$RUN/Compare/p0_regen/<tile>/`, `$RUN/Compare/p0_lineage/`; manifests `metric_names_all324.txt`, `columns_all324.txt`, `pcttree_file_order.txt`.

| Check | Result |
|---|---|
| Metric names | 29 names, 324 columns (72 unsorted + 252 derived), all distinct. |
| Row order of the existing split | Recovered and verified: 63 files, 11,440,984 rows, the first row of **every** file equals `X.parq` at its offset. |
| **Table against existing rasters, all 63 tiles** | **63 of 63 exact: 1,327,154,144 cells (11,440,984 rows x 116 columns), 0 differ,** none unmatched. The row total equals the full table. |
| **Fresh uncached regeneration, h09v05** | **Exact.** Table: 116 columns, 31,092,524 cells. Rasters: 116 bands, 2,672,640,000 cells, 0 differ. |
| **Fresh uncached regeneration, h12v02** (the polar tile) | **Exact.** Table: 116 columns, 35,720,924 cells. Rasters: 116 bands, 2,672,640,000 cells, 0 differ. |
| **Canary h10v05** (tolerance 3, ranks exact) | **Exact.** Table 116 columns over 352,807 rows; rasters 116 bands, 0 differ, largest difference 0; the tolerance was not needed. The script's verdict said DIFFERENT only because the tile's folder holds 23 Python-native rasters outside the 324-metric set (see `NOTE.txt` there); the script was corrected. |
| The other four canary tiles | Accepted under the rule "regenerate only if different": their files were written after the last change to `CMirrorMetrics.py`, and h10v05 matches although its files predate that change. Not directly regenerated. |

**What this establishes:** for three tiles, including the one where the fixes mattered most, the metrics recomputed from the daily inputs with the final code equal the production metrics cell for cell, so the composite and metric stages are reproducible and the training table (all 63 tiles) is exactly those metrics. The 116 columns the current model was trained on are the same values a new 324-metric build will contain.
The existing h10v05 folder holds a subfolder `_superseded_mirror_bug`: an earlier mirror-metric bug was found during the canary phase and corrected; the current files are the corrected ones and match the fresh regeneration.

**Measured timings, for step 3:** the fresh regeneration took about 34 to 37 minutes per tile (daily inputs and composites about 18 minutes; unsorted raster about 7; the 252 derived rasters about 6 at about 10 seconds each), three tiles at once on three nodes. The comparison took about 16 minutes per tile. The lineage check took about 2.9 minutes per tile.
**Step 3 estimate.** Cached mode skips the 18-minute stage, so a tile needs about 13 to 16 minutes; at 2 tiles per node on 5 nodes that is about 6 waves, **about 2 hours for the 60 remaining tiles** (concurrency of 2 per node is unmeasured). Building the tables (1 per node, larger than before) overlaps with it and adds about 30 to 60 minutes; the split and acceptance test about 20 to 30 minutes more. **Total about 3 hours, give or take 1.**
**Disk for step 3:** the 324 metric rasters take about 16.4 GB per tile, so **about 1 TB for the 60 tiles**, kept as decided. The filesystem shows 2.0 PB free (72% used); no per-user quota was visible from here, and the run root's `Python/` already holds 1.8 TB, so check the project quota before starting.

## Step 3 result (2026-09-20, 11:07 to 14:03)

**The 324-metric tree-cover training table is built and passed its acceptance test.**

| Item | Result |
|---|---|
| Metrics | All 63 tiles have all 324 metrics: 60 built from cached composites, 3 (h09v05, h12v02, h10v05) linked from the fresh regenerations of step 2. 28 derived rasters plus the unsorted and NDVI files per tile. `$RUN/Python/metrics_324/`: **1.1 TB** including the 3 links (the fresh regenerations, 106 GB, are in `metrics_324_fresh/`). Kept, as decided. |
| Tables | 63 per-tile tables, each 4 metadata columns plus 324 predictors, in `$RUN/Python/training_324/` (21 GB with the split files). |
| Split files | `$RUN/Python/training_324/pcttree/`: `X.parq` (5.87 GB), `xTrain.parq` (5.01 GB), `xTest.parq` (1.26 GB), `y`, `yTrain`, `yTest`. |
| **Acceptance test: PASS** | Row count equals the old table (11,440,984); train and test **index arrays equal the old ones** (9,152,787 and 2,288,197 rows); **all 116 shared columns equal the old table cell for cell** (0 differ); 324 predictor columns. |
| Time | Metrics 11:07 to 13:27 (2 h 20 min); table builds 11:10 to 13:56; split and acceptance 13:56 to 14:03 (7 minutes, well under the 25 to 30 I estimated). |

**What this establishes:** the Monte Carlo can now draw from the complete corrected metric set, and it will be scored on exactly the same held-out rows as the existing 64-predictor model. The 116 columns that model was trained on are, by the acceptance test, identical in the new table.

**What went wrong, and what it cost (so the next run avoids it):**
- **Memory.** The watchdog killed **9 metrics tiles** (a tile is killed when free memory drops under 8 GB). The cause was overlapping a 35 GB table build and Part A's model (14 GB) with two 20 GB metrics workers on the same node. On these nodes the free memory is well below what the process sizes suggest (probably filesystem cache), so **about 65 to 70 GB per node is usable, not 105**. What worked: two metrics workers per node, or one worker plus one build, with a 60 GB free-memory guard before each build, and nothing else on the node.
- **`noclobber` and retried logs.** The shell's `noclobber` setting made a retry die instantly when its log file already existed, which failed 4 tiles that were not memory failures. The table-build script was fixed (unique log names); the metrics script was not changed because it was running. Any retry of a metrics tile needs its old log removed first.
- **Stale claims.** Two tiles (h12v09, h11v09) were held by claims from workers I had stopped, and were never processed until I noticed and cleared them. Before stopping a runner, list its claims.
- **A build guard that gave up.** The first build script proceeded after 60 minutes of waiting for memory; version 2 waits indefinitely and skips tiles whose metrics are not ready.
- **Label folder.** The build's container needs the `Mark_training` folder bound read-only; version 1 of the script lacked it.
- **Cost:** the failures and the stale claims added roughly 45 to 60 minutes compared with the plan.
- **Nothing was lost or overwritten:** every failed tile was cleared (my own partial output only) and rerun, the production composites were never touched, and no failed marker remains.

## Steps 5 and 6 result: smoke test and timing pilots (2026-09-20, 15:05 to 15:18)

**Smoke test: passed** (`run_mc_smoke.sh`, 3.5 minutes, output `$RUN/Python/monte_carlo_standalone/smoke/`, 20,000 training rows and 20,000 held-out rows). Ten trials were written by two workers; the coordinator stopped on the trial cap and said why (236 predictors still under-used, 88 with non-zero importance), refusing to declare the conditions met; all 324 predictors were ranked; a fresh worker read the 10 existing trials, saw the stop marker and ran none; a 20-predictor final model was fit, saved (`pcttree.bin`, `pcttreeTopN.bin`) and scored against the existing 64-predictor model on the same held-out rows. The scores themselves mean nothing at 10 trials.

**Timing pilots** (`run_mc_pilot.sh`, output `pilot_p250`, `pilot_p500a`, `pilot_p500b`; one trial = a 30-tree fit on 10 random predictors plus permutation importance of those 10 on the evaluation rows):

| Pilot | Node | Time per trial |
|---|---|---|
| 250,000 training and 250,000 evaluation rows, 1 worker | forest203 | **47 s** (50, 48, 46) |
| 500,000 and 500,000 rows, 1 worker | forest201 | **94 s** (98, 96, 91) |
| 500,000 and 500,000 rows, **3 workers at once** | forest202 | **214 s per trial per worker** (load 24 on 12 cores) |

- **Time scales linearly with rows** (47 s to 94 s for double the rows).
- **Three workers per node gain 1.31 times in throughput**, not 3 times: each worker's trial takes 2.3 times longer because every fit already uses all 12 cores (`n_jobs=-1` inside the class, which was not changed). One node: 38.5 trials per hour with 1 worker, 50.4 with 3 workers at 500,000 rows.
- Worker memory is small (1.4 GB each), so memory is not a constraint here. Each 500,000-row subsample folder is about 1 GB.

**Projected time for the ranking phase on 5 nodes** (a trial count is random; 780 and 520 are the 90% figures from "Trial count" for `minVarUsage` 10 and 5):

| Subsample | Workers per node | Trials per hour, 5 nodes | 780 trials (`minVarUsage` 10) | 520 trials (`minVarUsage` 5) |
|---|---|---|---|---|
| 500,000 | 3 | 252 | **3.1 h** | 2.1 h |
| 500,000 | 1 | 193 | 4.1 h | 2.7 h |
| 250,000 | 3 (estimated) | about 500 | 1.6 h | 1.0 h |

**The final model adds about 1.5 hours** and cannot overlap the ranking: a fit on the full 9.15 million training rows with 64 predictors (the earlier full fit of the tree model took 81 minutes with 12 jobs), plus scoring, plus a top-20 fit if wanted (shorter).
**Whole run:** about **4.6 hours** at 500,000 rows and `minVarUsage` 10 (3.6 h at `minVarUsage` 5); about 3.1 hours (2.5 h) at 250,000 rows. These are estimates: the trial count needed is random, and a trial's time was measured on idle nodes.

## Decision rule for continuing to `minVarUsage` 15 (written 2026-09-20 18:25, before the stability numbers were computed)

The user asked for a judgment, after the final models finish, on whether the ranking is conclusive or whether more trials or a higher `minVarUsage` would help, and to continue to `minVarUsage` 15 if the answer is yes. The rule is fixed here first so the result cannot shape it.

**Measured on the trials that existed at the stop (702, not 703 as first written; the results document has the count) (`stability/` in the run folder):**
- *A. Membership uncertainty:* resample the trials with replacement 300 times and recompute the ranking each time. A predictor is **uncertain** if it lands in the top 64 in between 20% and 80% of the resamples.
- *B. Split-half agreement:* split the trials at random into halves 200 times; the median overlap of the two top-64 sets and the Spearman correlation of the two rankings.
- *C. Convergence:* the top-64 overlap between the ranking after 400, 500 and 600 trials (in creation order) and after all 703.

**Conclusive (no more trials) if all three hold:** at most 6 uncertain predictors; median half-against-half top-64 overlap at least 56 of 64; half-against-half Spearman at least 0.90.
**Otherwise continue to `minVarUsage` 15 only if more trials would help:** the uncertainty measure A, recomputed on 350 and 500 trials and extrapolated to about 1,050 trials (what `minVarUsage` 15 needs), must fall by at least a quarter. If it would not, more trials are not the remedy and I say so.
**The final models are read alongside:** a top-64 model that scores well on the held-out rows supports the ranking; a poor one is a finding in itself, not a reason for more trials.

**If continuing:** the existing stop marker, ranking and final models are renamed into a `results_minvar10/` folder first (the final fit saves under fixed names and would overwrite them), workers and coordinator restart with `minVarUsage` 15 on the same trials folder, and the top-64 fit is repeated only if the top-64 set changes.
Trials needed for `minVarUsage` 15 (the same random-draw calculation as in "Trial count"): about **916 for a 50% chance and 1,023 for 90%**, that is 213 to 320 more trials than the 703 already done, or 0.8 to 1.2 hours on five nodes at 265 trials per hour.

## Trial count

**No fixed number is planned.** The run stops on a rule, not a count: every one of the 324 predictors has been used in at least `minVarUsage` trials, **and** at least 64 predictors have non-zero average permutation importance (64 because the final model uses 64). A trial cap and a time cap, set after the timing pilot, stop it earlier and say so in the stop marker.
The expected count comes from the random draw of 10 of 324 predictors per trial (each predictor is in a trial with probability 10/324), calculated 2026-09-20:

| `minVarUsage` | Hard minimum | Trials for a 50% chance that every predictor has enough uses | For a 90% chance |
|---|---|---|---|
| 10 (the code's default) | 324 | about 690 | about 780 |
| 5 (a first look) | 162 | about 440 | about 520 |

The hard minimum is unreachable with random draws; the 50% and 90% columns are the working estimates. An earlier draft said "600 or more"; the calculation gives about 700 to 800 for the default. Recommended: settle `minVarUsage` after the pilot, with 10 as the target and 5 as the fallback if a trial is too slow.
The time per trial is unknown until the pilot, so **the total time is unknown too**; trials are independent, so the node split divides it.

## Part A. Feature importance from the existing 64-predictor model

**Script (new):** `$RUN/code/validation/feature_importance_existing_models.py`; **launcher:** `run_feature_importance.sh`. Read-only on the model and the held-out data; writes only to a new folder
`$RUN/Compare/feature_importance/`, which the script refuses to reuse.

1. **Impurity importance:** load `pcttree.bin`, take `feature_importances_` from each of the 30 trees, report the mean and the standard deviation across trees for each predictor. Cheap once loaded (the 13.5 GB pickle
   needs roughly 15 to 30 GB of memory and a few minutes).
2. **Permutation importance on the full held-out set** (all 2,288,197 `xTest` rows, decision 2), 5 repeats per predictor, default R-squared scoring, mean and standard deviation. Fixed seeds so the result is reproducible.
   **Split across nodes by predictor:** each node loads the model and scores its own share of the 64 predictors on the same rows with the same seeds, and a merge step joins the per-node files (compute nodes section).
3. Map predictor names to columns **by name** (the saved list is not in `X.parq` column order), and check that the model's own feature names, if it stored them, match.
4. Outputs: `importance_pcttree.csv` (rank, predictor, impurity mean and sd, permutation mean and sd), a `summary.txt` (top 15 by each measure, and where they disagree), and the logs.

Reading it: many of these metrics are strongly correlated, and correlated predictors share credit in both measures, so a low rank can mean "redundant", not "useless". The held-out 20% is a random split of pixels,
not of tiles, so neighbouring pixels sit on both sides and the scores are optimistic.

## Part B. Stand-alone Monte Carlo on the 324-metric table

**No edits to rlgill's files or to the run-root copy.** A new driver imports the class and controls the loop itself.
- **Script (new):** `$RUN/code/validation/run_monte_carlo_standalone.py`; launchers per node, started detached (`setsid nohup`, status and done markers).
- **Output folder (new, refuses reuse):** `$RUN/Python/monte_carlo_standalone/tree/`. Nothing is written into `$RUN/Python/models/`.
- **Input:** the Part 0 table (324 predictors). Trials use **fixed-seed row subsamples** of the pre-split files (size set by the pilot) written into the output folder so the class skips its own split. The evaluation subsample
  is taken from `xTest`/`yTest` rows only, aligned, which avoids defect 4.
- **Loop:** the driver seeds the random draw and `PYTHONHASHSEED`, runs batches with the class's own trial method, and stops when every predictor has been used at least `minVarUsage` times, **or** a trial cap or a
  time cap is reached. After each batch it writes the running ranking, so a stopped run still gives a result.
- **Final model (decision 4):** after the ranking is complete, fit the final model on the top N predictors **using the full training rows** (the driver points the class at the full files for that step only) and save it
  with the class's own names (`pcttree.bin`, `pcttreeTopN.bin`) in the new folder, avoiding defect 1. N is 64 (decision 9), and the ranking also gives the top 20 for reference. The driver then reports its held-out score on the same `xTest` rows as the existing model, side by side.
- **Ranking outputs:** `ranking.csv` (predictor, average permutation importance, number of trials it appeared in), the class's `SortedMetrics-*.txt`, and a comparison of the top N against C's 64 predictors and against Part A's ranking.

## Compute nodes

Checked 2026-09-20: `forest201` to `forest205` each have 12 cores and 112 GB memory (about 105 GB available). Load was about 0 on 201, 202, 203 and 205, and about 1 on 204 (not identified).
Earlier runs of this project used 201 and 202 for C and 203 to 205 for Python.

- **Part 0 (metrics):** tiles are independent; use the existing claim-by-`mkdir` queue pattern of `python_metrics_node.sh`, with a memory watchdog, across 201, 202, 203 and 205.
- **Part A (permutation):** the 64 predictors are split into four groups, one per node (201, 202, 203, 205), each loading the model once (13.5 GB) and scoring its own predictors on the same rows and seeds; 204 stays free.
- **Part B (trials):** trials are independent. Each node runs workers that write `Trial-*.bin` files into the shared `trials/` folder, with a per-node random seed. **One coordinator** (a small script on one node)
  reads all trial files every few minutes, checks the usage and importance conditions, and writes a `STOP` marker that the workers check between trials. The trial caps and time caps live in the coordinator, not in the workers.
- **Rules for all of it:** every run is launched detached with status and done markers; nodes are checked for load and free memory immediately before launch; nothing runs on the login nodes (they have no singularity or GDAL).

## Steps, each with a check before the next

| Step | What | Cost |
|---|---|---|
| 0 (**done for the split**, see Part 0 step 4) | The split script, seed and row order are identified and the order checked at the file boundaries; the index is the row position, not a key. Still to confirm from the fit log: that the 63-tile model was fit on `pcttree/xTrain.parq`. | no compute |
| 1 (**done 2026-09-20**, see "Step 1 result") | Write the scripts and review them with you. | no compute |
| 2 (**done 2026-09-20, all passed**, see "Step 2 result") | **Part 0 checks:** file-order recovery; metric names; fresh, uncached regeneration of h09v05, h12v02 and the canary h10v05 with all 324 metrics; comparison with the existing metrics and table; table-against-raster comparison for all 63 tiles. | about 1 hour of wall time |
| 3 (**done 2026-09-20 14:03, acceptance PASS**, see "Step 3 result") | **Part 0 build:** metrics for the other 60 tiles from cached composites, tables, split in the recovered order, acceptance test. | 2 h 56 min (11:07 to 14:03), including the failures below |
| 4 (**done 2026-09-20 12:07**, see `Feature_Importance_Part_A_Results.md`) | **Part A:** impurity importance, then full-sample permutation importance split over four nodes, on the existing 64-predictor model. | 52 minutes on four nodes |
| 5 (**done 2026-09-20 15:09, passed**, see "Steps 5 and 6 result") | **Part B smoke test:** 20,000 rows, `minVarUsage` 1, at most 10 trials; checks the mechanics, output files and resume, and that the defects are avoided. | minutes |
| 6 (**done 2026-09-20 15:18**, see "Steps 5 and 6 result") | **Part B timing pilot:** one trial at the proposed subsample size, measured; set subsample, caps and the number of worker processes per node. | tens of minutes |
| 7 (**done 2026-09-20 20:00**, see `Feature_Importance_and_Monte_Carlo_Results.md`) | **Part B full run** across the nodes, detached and resumable, then the final model fit and save. | workers 15:26 to 18:05 (702 trials), final fits 18:09 to 20:01 |
| 9 (**done 2026-09-20 20:47**, requested after step 8; see "Step 9") | **Map-level check** of the top-64 and top-20 models on h12v02, h09v05 and h12v09 against the historical MOD44B, with a control. | 20 minutes |
| 8 (**done 2026-09-20**) | Read the results; write `Feature_Importance_and_Monte_Carlo_Results.md` in `Reports/`; add both documents to the README map. **Decision on continuing to `minVarUsage` 15: no** (results document, "Decision on more trials"). | no compute |

## Step 9. Map-level check of the Monte Carlo models (requested 2026-09-20 20:20; launched 20:28; **done 20:47**)

**Why:** the held-out scores in `Feature_Importance_and_Monte_Carlo_Results.md` are on a random pixel split and are optimistic, and the top-64 model scored 0.0053 R-squared below the existing model. The question is whether maps made with the new models differ from the 63-tile Python maps, and whether any difference matters against the historical MOD44B. This is "test 4" of the list of further tests offered after the Monte Carlo (the others, a noise floor, a random-set null, a held-out-tile split and a redundancy-aware selection, were **not** done).
**Tiles (chosen by the user):** h12v02, h09v05 and h12v09. h12v02 and h09v05 are the two tiles whose metrics were regenerated fresh in Part 0; h12v09 is an ordinary tile.
**Runs (one per node, detached, 12 threads each, `nice` 10):** three models, each predicting the three tiles through the *same code path as the 63-tile run* (`VcfPredict._runPrediction` and `_write`: metric reads by name, blocks of 1,000,000 rows, truncation to `int16`, water mask):
- `mc_top64`: the new top-64 model (forest202),
- `control_existing`: the **existing** model, read from the 324-metric rasters through the new script (forest203). This is a control: it must reproduce the existing maps in `Python/predictions/` exactly, otherwise a difference between the maps of the other two models and the existing maps could come from the input path and not from the model,
- `mc_top20`: the new top-20 model (forest204).
**Metrics read:** `Python/metrics_324/<tile>/2019/3-Metrics/` (the 324-metric rasters, verified against the earlier metrics in Part 0), mounted read-only, so a missing metric fails instead of being computed and written.
**New scripts (run-root copy, `$RUN/code/validation/`, nothing of rlgill's edited):** `predict_tree_cover_model.py` (bypasses only the `VcfPredict` constructor, which would also load the 13 GB non-vegetated model), `run_predict_model_node.sh`, `compare_model_variants.py`, `run_variant_compare.sh`.
**Outputs (new folders, refuse reuse):** `Compare/mc_model_prediction/<label>/` (map, `predictors.txt`, copied water mask) and `Compare/mc_model_prediction/comparison/`.
**Comparison:** all products (C, the existing maps, the control, top 64, top 20) against the historical MOD44B Collection 6 on the pixels valid in all six products, per tile and pooled: bias, MAE, RMSE, correlation, share within 5 and 10 points, by historical-cover bin, and the share of pixels where the top-64 map is closer to the historical than the existing map.
**Not done here:** the non-vegetated layer, the other 60 tiles, and any figure of the new maps.

**Result (2026-09-20 20:47; full numbers in `Feature_Importance_and_Monte_Carlo_Results.md`, "Map-level check"):**
- **Control passed:** the existing model through the new path reproduces the existing maps exactly (0 differing pixels on each of the three tiles).
- **Against the historical MOD44B on 65.3 million common pixels, mean absolute error:** C 3.67, existing Python 4.98, Monte Carlo top 64 **5.22**, top 20 5.55. The top 64 is worse than the existing model on all three tiles (by 0.11, 0.37 and 0.26); the top 20 is worse on two and slightly better on h09v05.
- **The maps do change:** top 64 against existing differ by a mean absolute 4.60 points pixel by pixel (existing Python against C: 4.39), so the change is large per pixel but not an improvement against the historical product.
- **Not settled:** no run-to-run baseline exists for that 4.60 (same 64 predictors, different seed), so the part due to the predictor set is unknown. Three tiles only, no significance test.
- Run times: model load 1.5 to 2 minutes, 2 to 5 minutes per tile, three nodes in parallel; markers `status/predict_<label>.done` and `status/variant_compare.done`.

## Write safety (audited 2026-09-20)

- **Where the new work writes.** Only new folders and new files under `$RUN`: `Python/metrics_324_fresh`, `Python/metrics_324`, `Python/training_324`, `Python/monte_carlo_standalone`, `Compare/feature_importance`, `Compare/p0_regen`, `Compare/p0_lineage`; new files in `manifests/`, `status/` and `logs/` whose names start `p0_` or are new; new claim folders under `Python/status/`. **None of these paths exists today** (checked), and no `p0_` status file exists.
- **Enforced by the operating system for every step run in the container:** the whole project tree is bound read-only, and only the new folders above are bound writable (the launcher was tightened on 2026-09-20: it had bound all of `Compare/` writable, which would have exposed the existing results there; it now binds only the new sub-folders). The models, the training data, the existing metrics, the existing comparison results, `/css/modis`, the pge61 tree and rlgill's checkouts cannot be written by any containerized step.
- **Enforced by the scripts for the host-side shell code** (which runs as the user and so has no operating-system barrier): the shell scripts use `noclobber` so a redirect onto an existing file fails; the Python scripts open outputs with exclusive-create or refuse an existing output folder. The remaining appends (`>>`) go only to per-node or per-run status files that are new.
- **In cached mode** the node script creates new symlinks under `Python/metrics_324/<tile>/2019/` that point at the production composites; the production folders are bound read-only in the container, so a missing composite would fail instead of being written.
- **Not covered by "new folders only":** I will edit existing documents in the sync workspace (this plan, the README) and add results documents, keeping backups. Those are inside `$SYNC/Reports`.
- **Scope confirmed by the user, 2026-09-20: "the existing workspace" is the sync workspace `$SYNC` plus the run root `$RUN`.** Every write in this plan lands in one of those two, in a new folder or new file, apart from the edits to existing documents in `$SYNC/Reports` noted above. Nothing outside them is written: not the pge61 tree, rlgill's checkouts, `/css/modis`, `VCF_outputs`, or the models. The container runtime itself may keep its own cache or temporary files under the user's home or `/tmp`; those are not project data. The session scratchpad (outside both) holds only the throwaway synthetic test data used to check the scripts.

## Constraints

- Nothing under rlgill's repositories or the pge61 tree is touched; no code copy in the workspace or run root is edited. The new scripts are new files in `$RUN/code/validation/`.
- The existing models, the existing training table and the existing metrics are opened read-only. All outputs go to new folders, which the scripts refuse to reuse.
- Each run step is an approval point.

## Step 1 result: the scripts (2026-09-20)

All are new files in `$RUN/code/validation/`. None edits any existing file, and none has been run on real data yet.

| Script | What it does | Tested |
|---|---|---|
| `p0_make_metric_names.py` | writes the 29 `makeMetrics -m` names and the 324 column names, checking 72 + 252 = 324 and that every C index decodes to one of them | run on the real `CAttributes`: 29 names, 324 columns |
| `p0_makemetrics_all.py` | wraps `view/makeMetrics.py`, re-installing all 28 mirror families for all nine bands (the `wanted` option of `CMirrorMetrics.install`), without editing a file | `makeMetrics -p` on a real tile lists all 28 families, including ones C's models never use (`CUnsMean46`, `CSortedRank3`, `CThermMax`, `CGreenMean79`) |
| `p0_metrics324_node.sh` | per-node tile queue for all 324 metrics, `fresh` (no cache) or `cached` (composites symlinked, production folders read-only); memory guard and watchdog | syntax only |
| `p0_compare_regen.py` | fresh against existing: table columns exact; whole rasters with the tolerance and the exact-rank rule; reads rasters with GDAL directly | synthetic: identical passes; +2 passes at tolerance 3 and fails at 1; a rank off by 1 fails at tolerance 3; a table difference is caught |
| `p0_lineage_check.py`, `p0_lineage_node.sh` | table against existing rasters for a list of tiles, per node | synthetic pass; refuses to overwrite |
| `p0_recover_file_order.py` | recovers and verifies the row order of the old split | synthetic pass; a mismatching `X.parq` is refused |
| `p0_build_table324_node.sh` | runs the run's own `buildTraining.py` per tile for all 324 metrics, waiting for each tile's metrics | syntax only |
| `p0_build_xy324.py` | builds the 324-column X/y split in the recovered order and checks it against the old files: row count, identical train/test index, the 116 shared columns equal cell for cell, 324 columns | synthetic pass; a wrong file order is caught |
| `feature_importance_existing_model.py` | Part A: impurity, permutation split by feature range, merge; resumable | synthetic model: the two true drivers rank first; resume adds no duplicate rows |
| `run_monte_carlo_standalone.py` | Part B: `prep` (subsamples, evaluation from held-out rows only), `worker`, `coordinator` (writes the stop marker and the ranking), `final` (full-row fit of the top 64 and, optionally, top 20; scored against the existing model on the same rows) | synthetic run through the **real `MonteCarloSim` class**: 16 trials by two workers, ranking, stop, final model saved, existing model scored on the same rows |
| `run_node_job.sh` | detached launcher for one step on one node, with status and done markers, and a free-memory check | syntax only |

**The tests found four bugs in my own scripts, all fixed:** the transposed-orientation indexing ran out of bounds on a non-square raster; the coordinator's output-file name had an operator-precedence error; the permutation script's resume would have skipped a feature interrupted part-way; and a wrong assumption in my plan about the split index (above), which the read-only probe found before any script relied on it.

**Not yet exercised on real data:** `p0_metrics324_node.sh`, `p0_build_table324_node.sh`, `run_node_job.sh` and the real-size behavior of every script. Step 2 is the first real run and starts with one tile.

## Open questions

1. ~~Confirm the reading of decision 13.~~ **Confirmed 2026-09-20.**
2. **Approval to start step 2**, the first compute: the `p0_recover_file_order.py` run (metadata only), then the fresh regeneration of h09v05 and h12v02 on two nodes, the canary tile comparison, and the 63-tile table check. Recommended order: the order recovery and the h09v05 metric-name check first, since they are cheap. **Not yet given.**
