# Rerun the Monte Carlo predictor selection with the current metrics and composites (plan, to-do item 9)

**Written 2026-10-02; revised the same day after the project lead's decisions: both layers (tree cover and non-vegetated), RF as the model, no lidar evaluation, delete the old 324-metric rasters, ngaproc201/202 only. Status: plan only. Not approved for execution. Nothing has run.**

**Update 2026-10-03: approved and run 2026-10-02 08:23-19:37 (`FullRuns/mc_rerun_20261002`); all steps done except step 8 (deleting
the old rasters), which waits for the project lead. Results: `Monte_Carlo_Rerun_Results.md`.**

## Why

The only Monte Carlo (MC) predictor selection (`Feature_Importance_and_Monte_Carlo_Results.md`, 2026-09-20) ran on inputs that have since changed:

| What the 2026-09-20 MC used | What the product uses now |
|---|---|
| Composites screened with the bits 7-8 aerosol test, which over-rejects (29-50% of good pixels on odd days) | bit-15 QA (adopted 2026-09-29, `Aerosol_QA_C_Full63_Retrain_Results.md`) |
| No zenith-only NDVI on h16v01 | zenith-only NDVI on the 6 Greenland tiles (commit `13773db`); among the 63 tiles this touches h16v01 |
| Trial model and final model: `BaggingRegressor` (the old baseline) | Random Forest, 30 trees; tree cover = MEDIAN of the trees, non-vegetated = MEAN; values rounded |
| Tree cover only | Tree cover, and non-vegetated (RF on C's 21 bare attributes; never selected by a Monte Carlo) |

Its result was undecided: the top 64 shared only 20 predictors with C's 64 and did not beat them on maps. The run-to-run (seed) baseline needed to judge that was never done. This plan reruns the selection for both layers on the current inputs, with RF, and adds the seed baseline.

## Decisions (project lead, 2026-10-02)

| Decision | Choice |
|---|---|
| Scope | **Phase A, tree cover, and phase B, non-vegetated (bare); both in this run** |
| Model | **Random Forest** for the trials, the final fits and the seed baseline: 30 trees, `max_features='sqrt'`, the delivered hyper-parameters |
| Lidar | **No lidar evaluation** (no NEON, no G-LiHT) |
| Old rasters | **Delete the 2026-09-20 324-metric rasters** after this run (see "Disk") |
| Systems | **ngaproc201 and ngaproc202 only** (see "Systems") |
| Decision on the results | **No automatic rule: the project lead reviews the results** (see "Review of the results") |
| 63-tile maps | **Run them** for one MC candidate per layer, **if the map-check tile maps complete without error** (see "Evaluation") |

**Confirmed by the project lead, 2026-10-02:**
- **Trial settings as on 2026-09-20,** so the old and new rankings can be compared: 10 random predictors per trial; 500,000 training and 500,000 evaluation rows per trial (fixed-seed subsamples); permutation importance. Stop when every predictor has been used in at least 10 trials and at least 64 have non-zero importance; caps of 1,200 trials or 6 hours per layer.
- **Candidate pool:** the same 324 metrics (72 unsorted monthly band values plus 28 families × 9 bands), rebuilt on the bit-15 composites.
- **Final candidate sets:**
  - tree cover: MC top 64 (the delivered size) and top 20;
  - non-vegetated: MC top 21 (the size of C's set) and top 64.
- **Seed baseline:** the delivered model refit with seeds 1 and 2, for each layer.
- **Map-check tiles:**
  - tree cover: h12v09, h12v02 and h09v05 (the 2026-09-20 tiles);
  - non-vegetated: h18v07 and h20v06 (desert; both have held-out bare labels), plus h12v02 and h09v05.

## Systems: ngaproc201 and ngaproc202 only

- **Hardware:** each node has 24 CPUs and 225 GB of memory; both were idle on 2026-10-02.
- **Access:** ssh needs `-o CheckHostIP=no`, and jobs need `module load singularity`. Container: `/explore/nobackup/projects/ilab/containers/modis-vcf-latest`. All jobs are launched fully detached (`setsid nohup`), with status and done markers.
- **Memory budget:** at most about 180 GB committed per node, leaving headroom. This morning's rounding run lost 4 tiles at 3 × 42 GB plus extra arrays.
- **Compared with also using forest201-205:**
  - ngaproc has 48 CPUs in total, against 60 on the five forest nodes, but 4× the memory per node. So metric building fits about 5 workers per node, against 2 on a forest node.
  - The MC trial rate is CPU-bound, so ngaproc alone is about 20% slower than September's 5 forest nodes, unless RF trials are faster than bagging trials.
- **Speed check:** RF trials (3 of the 10 predictors tried per split) should be faster than bagging trials (all 10), but this is unmeasured. Step 0 includes a 20-trial pilot to measure it. If the projected MC time per layer is over 6 hours, the plan says so before step 4 starts, and adding forest nodes becomes your call.
- **Risks:**
  - Both nodes carry the whole run. If one goes down, the work continues on the other at half speed; claims and markers let a restart resume.
  - Other users could start jobs on ngaproc. Free memory is checked before each new task, and tasks wait rather than start below the budget.

## Method

Run root: `FullRuns/mc_rerun_<date>/`, new. Code: a frozen copy of the delivery branch at `16d6474` or later, plus the stand-alone MC driver and table, split and fit scripts from 2026-09-20 (`full63_c_vs_py_20260919-0806/code/validation/`), copied into the run root. Changes to the copies: RF instead of `BaggingRegressor`; layer and label options for non-vegetated; new paths. The shipped `MonteCarloSim` still has the 6 defects listed in open item 9, so it is not used.

| Step | What | Workers | Wall time |
|---|---|---|---|
| **0. Preflight** | Check that the bit-15 composites (`qaC_full63_retrain_20260929/Python/metrics_all/<tile>/2019/2-Composites`) are complete for all 63 tiles; check free disk (about 1.1 TB needed); freeze the code; write the manifests. **Pilot:** 20 RF trials on the existing 2026-09-20 table, to measure the trial rate on ngaproc. | 1 node | about 30 min |
| **1. Gate (h08v05)** | Build the 324 metrics on h08v05 from its bit-15 composites ("cached mode"; NDVI recomputed by the current code). Every metric the delivered tree and bare models use must equal `qaC_full63_retrain_20260929/Python/metrics_all/h08v05` cell for cell. A failure stops the run (`status/gate.failed`). | 1 | about 20 min |
| **2. 324 metrics, other 62 tiles** | Cached mode, as on 2026-09-20 (13-16 min per tile there). h16v01 gets the zenith-only NDVI. | 5 per node, 10 at once, about 20 GB each | **about 1.75 to 2.5 h** (7 waves; the range allows for disk contention with 10 writers) |
| **3. Tables, splits, acceptance** | Per-tile tables of 324 predictors for the tree labels (63 tiles) and the bare labels (20 tiles), built as each tile's metrics finish. The same keyed 80/20 split as the delivered models. **Acceptance:** row counts and train/test keys equal the delivered splits; the delivered models' columns (64 tree, 21 bare) equal their training tables cell for cell. | alongside step 2, 2 per node | about 30-45 min after step 2 ends |
| **4. MC phase A, tree cover** | About 700 trials to the stop rule. | 12 (6 per node) | **about 3 to 4.5 h** (the pilot will narrow this) |
| **4s. Seed baselines, both layers** (alongside step 4) | Refit the delivered tree RF and bare RF with seeds 1 and 2 (tree fit about 8 min, bare about 4.5 min, measured); map the map-check tiles with each. | 2 slots kept free of MC workers | about 1.5 h, hidden behind step 4 |
| **5. MC phase B, non-vegetated** | The same trial settings, on the bare table. | 12 | **about 3 to 4.5 h** |
| **5f. Phase A final fits and evaluation** (alongside step 5) | Fit the MC top 64 and top 20 for tree cover on the full training rows. Stability analysis (bootstrap membership, half-against-half overlap, convergence). Evaluation as below. | 2 kept slots | about 2 to 2.5 h, hidden behind step 5 |
| **6. Phase B final fits and evaluation** | Fit the MC top 21 and top 64 for non-vegetated; stability; evaluation. | up to 4 | about 1.5 to 2 h |
| **7. Report** | `Reports/Monte_Carlo_Rerun_Results.md` (both layers), and the README is updated. | | about 1.5 h |
| **8. Delete the old rasters** | See "Disk". Done only after the report, and only if every acceptance test in this run passed. | 1 | about 15 min |

**Evaluation for each layer** (no lidar):
- **Held-out labels:** the MC sets against the delivered set and the seed refits, on the same held-out rows. Pooled, by cover bin (0-20 / 21-40 / 41-60 / 61+), and by continent and latitude band.
- **Maps:** the map-check tiles, against C6, C6.1 and the C code, with the delivered settings (tree = MEDIAN, non-vegetated = MEAN, rounded, sum cap). Map differences are compared with the seed baseline's map differences.
- **Stable set:** pixels whose C6 SD over 2000-2020 is ≤ 5 (tree cover; non-vegetated where the stable-set table has it).
- **63-tile maps (project lead, 2026-10-02):** run automatically for each layer **if that layer's map-check tile maps all completed without error**. Otherwise they are skipped and the report says why.
  - **What is mapped:** one MC candidate per layer, the one with the lowest held-out MAE: tree top 64 or top 20; bare top 21 or top 64. This choice is only for mapping and implies no adoption.
  - **Compared with:** the delivered 63-tile maps, against C6, C6.1 and the C code, by the same summaries as the rounding run.
  - **Time:** about 1 to 1.5 h per layer on ngaproc (3 per node, about 42 GB each).

## Time estimate (ngaproc201/202 only)

| Phase | Wall time |
|---|---|
| Preflight, pilot and gate | about 0.75 to 1 h |
| 324 metrics, tables, splits and acceptance | about 2.5 to 3.25 h |
| MC phase A (seed baselines alongside) | about 3 to 4.5 h |
| MC phase B (phase A's fits and evaluation alongside) | about 3 to 4.5 h |
| Phase B fits and evaluation | about 1.5 to 2 h |
| 63-tile maps, both layers (if the tile maps completed without error; phase A's partly alongside phase B) | about 1.5 to 2.5 h |
| Report and deletion | about 1.75 h |
| **Total** | **about 14 to 19.5 hours**: an overnight run plus most of the next day |

**For comparison:** with ngaproc and forest201-205 together, about 11 to 15 hours. The forest nodes would mainly speed up the two MC phases. The pilot in step 0 decides whether the difference matters.

## Disk

| Item | Size | Action |
|---|---|---|
| New 324-metric rasters (`Python/metrics_324/`, 63 tiles) | about 1.1 TB | kept until you decide, after the report |
| New tables and splits (tree and bare) | about 25 GB | kept |
| New models: 2 seed refits and 2 MC sets for each layer, about 15-17 GB each | about 130 GB | kept |
| **Old 2026-09-20 rasters:** `full63_c_vs_py_20260919-0806/Python/metrics_324/` (1.1 TB) and `Python/metrics_324_fresh/` (106 GB), built with the old bits 7-8 QA | about 1.2 TB | **deleted in step 8** (project lead, 2026-10-02) |

**How the old rasters are deleted:**
- In `metrics_324/`, 60 tiles' `2019/` folders contain symlinks to the production composites. The deletion removes only regular files and the links themselves, never what a link points to, so the production composites are untouched.
- The full list of files is written to `status/delete_old_metrics324.list` before anything is removed, with a byte total.
- Kept: the old Monte Carlo results, trials and models (`Python/monte_carlo_standalone/`), and the old 324-column tables (`Python/training_324/`, 21 GB). They document the 2026-09-20 result.

Net change on disk: about +1.25 TB new, −1.2 TB old.

## Review of the results (project lead, 2026-10-02)

There is no automatic adoption rule. The report puts the evidence in front of the project lead, who decides for each layer whether to adopt a new predictor set or keep C's. For each layer the report gives:
- the held-out MAE, bias and RMSE of each MC candidate, the delivered set and the seed refits (pooled, by cover bin, by region);
- how far each candidate's maps differ from the delivered maps, next to how far the seed refits' maps differ from each other (the run-to-run yardstick);
- each set's scores against C6, C6.1, the C code and the stable set, on the map-check tiles and, if they ran, on all 63 tiles;
- the stability of the ranking (bootstrap membership, half-against-half overlap) and its overlap with C's set and with the 2026-09-20 ranking.

The standing tie rule (about 1 point MAE or less) is stated next to each comparison as context, not as a decision.

## Caveats

- **Held-out scores are optimistic:** the held-out rows are a random 20% of pixels, neighbours of training pixels. A split by tile is a follow-up, not part of this plan.
- **Comparisons against the historical MOD44B carry the aerosol caveat:** the historical product used no working aerosol test.
- **Non-vegetated has fewer data:** labels on only 20 tiles, so its held-out set is smaller and more regional than the tree set. Its ranking may be less stable.
- **Effect on the additional-years plan:** if either layer's predictor set changes, the years plan (`Additional_Years_Plan.md`) must use the new set. That plan's suggested order (after this decision) still applies.
