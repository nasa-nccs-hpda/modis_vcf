# Monte Carlo predictor selection rerun on the current inputs (results, to-do item 9)

**Status 2026-10-03: the run is finished (`FullRuns/mc_rerun_20261002`, launched 2026-10-02 08:23 on ngaproc201/202, `status/ALL.done`
19:37, about 11.25 h against the plan's 14 to 19.5 h). Every gate and acceptance test passed; no task failed. Step 8 (deleting the
old 2026-09-20 324-metric rasters) was done on 2026-10-03 after the project lead's go-ahead (see "Disk"). No adoption decision has
been made: the project lead reviews these results.** Plan: `Monte_Carlo_Rerun_Plan.md`.

## Summary

- **Neither layer's Monte Carlo (MC) set beats the delivered predictors on the held-out labels.** The seed refits of the delivered
  models set the run-to-run yardstick at about 0.01 to 0.02 MAE.
  - **Tree cover:** MC top 64 is +0.28 MAE worse (better on 9 of 63 tiles; every interval is above zero); MC top 20 is +1.49 worse.
  - **Non-vegetated:** MC top 64 is +0.11 worse (better on 4 of 20 tiles; the per-tile interval spans zero, the block interval
    does not); MC top 21 (the size of C's set) is +2.15 worse.
- **Under the standing tie rule (about 1 point MAE or less), tree top 64 and bare top 64 are ties with the delivered sets.** The
  smaller sets (tree 20, bare 21) are not.
- **Maps, 63 tiles:**
  - Tree top 64 agrees less with every reference than the delivered map does (C code +0.30, C6 +0.35, C6.1 +0.25 MAE).
  - Bare top 64 is mixed (C code −0.08, C6 +0.07, C6.1 +0.18). This is not consistent across references, so by the project's rule it
    is not a reason to act.
- **The ranking reproduces:** the new tree ranking matches the 2026-09-20 ranking (Spearman 0.97, 56 of the top 64 shared), even
  with bit-15 composites and RF instead of bagging. As in September, it shares few predictors with C's set: 19 of 64 for tree cover,
  and 1 of 21 for non-vegetated.
- **Reading:** the MC finds a different but slightly weaker predictor set for both layers. The evidence favours keeping C's
  predictor sets. That is for the project lead to decide.

## What ran

| Step | Result | Finished (2026-10-02) |
|---|---|---|
| 0. Preflight and pilot | 24 RF pilot trials, median 194 s per trial; projected 3.16 h per MC phase on 12 workers (under the 6 h cap, so no forest nodes added) | 08:23 |
| 1. Gate (h08v05) | **PASS:** 116 bands the delivered models use compared cell for cell with `qaC_full63_retrain_20260929`; 0 differ; NDVI identical | 08:39 |
| Pipeline check | Delivered h12v09 map rebuilt by this run's code equals the stored delivered map (tree, non-vegetated, non-tree vegetation: 0 differing) | |
| 2-3. 324 metrics (63 tiles), tables, splits | **PASS:** held-out keys and labels equal the delivered splits row for row; delivered columns equal cell for cell on 9.0 M train / 2.3 M test tree rows and 8.2 M / 2.1 M bare rows (h16v01 excluded from the cell check, since its NDVI changed on purpose) | |
| 4. MC phase A, tree cover | 682 trials to the stop rule (every predictor in at least 10 trials, 64+ with non-zero importance) | 13:56 |
| 4s. Seed baselines | Delivered tree RF and bare RF refit with seeds 1 and 2, mapped on the map-check tiles | |
| 5. MC phase B, non-vegetated | 777 trials to the stop rule | 15:38 |
| 5f/6. Final fits, evaluation | Tree MC top 64 and top 20; bare MC top 21 and top 64; held-out scores, map-check tiles, stable set, stability | tree 15:00, bare 17:23 |
| 63-tile maps | Run for both layers (map-check maps all completed). Candidate by lowest held-out MAE: **tree MC top 64, bare MC top 64**. 63/63 maps and comparisons OK each | tree 16:56, bare 19:37 |

Settings as planned: RF with 30 trees, `max_features='sqrt'`; 10 random predictors per trial; 500,000 training and 500,000 evaluation
rows per trial (seed 0); permutation importance; the same keyed 80/20 split as the delivered models. Tree = MEDIAN of the trees,
non-vegetated = MEAN, values rounded. No lidar.

## Tree cover

### Held-out labels (2,289,990 pixels, 63 tiles)

| Model | Predictors | MAE | Bias | RMSE | MAE vs delivered | Tiles better | Block interval |
|---|---|---|---|---|---|---|---|
| Delivered (C's set) | 64 | 6.71 | −0.17 | 11.49 | | | |
| Seed 1 | 64 | 6.70 | −0.17 | 11.47 | −0.009 | 37 / 63 | −0.014 to −0.004 |
| Seed 2 | 64 | 6.70 | −0.17 | 11.48 | −0.002 | 30 / 63 | −0.008 to +0.003 |
| MC top 64 | 64 | 6.99 | −0.21 | 11.89 | **+0.28** | 9 / 63 | +0.27 to +0.30 |
| MC top 20 | 20 | 8.20 | −0.36 | 13.67 | **+1.49** | 1 / 63 | +1.45 to +1.54 |

- **By cover bin:** MC top 64 is worse in every bin: 0-20 +0.17, 21-40 +0.39, 41-60 +0.32, 61+ +0.44. The seeds move each bin by 0.02
  or less.
- **By region:** MC top 64 is worse in all 15 continent/latitude strata, from +0.01 (Africa 10-30N) to +0.68 (South America 10N-10S).
  The largest losses are in the tropics (South America 10N-10S +0.68, Africa 10N-10S +0.54, South America 10-40S +0.49, Asia 10-30N
  +0.49).

### Maps

**Map-check tiles (h12v09, h12v02, h09v05; 65.3 M pixels), MAE against each reference:**

| Product | C code | C6 | C6.1 | Distance from the delivered map |
|---|---|---|---|---|
| Delivered | 5.12 | 4.62 | 7.05 | |
| Seed 1 | 5.15 | 4.61 | 7.01 | 2.71 |
| Seed 2 | 5.15 | 4.62 | 7.02 | 2.72 |
| MC top 64 | 5.55 | 5.02 | 6.95 | **3.72** |
| MC top 20 | 5.80 | 5.51 | 7.72 | 4.52 |

Seed 1 against seed 2: 2.70 MAE. That is the pixel-level run-to-run yardstick: MC top 64 moves the map about 1 point MAE further than
a seed change does.

**All 63 tiles (1.11 billion pixels), MC top 64:**

| Product | C code | C6 | C6.1 |
|---|---|---|---|
| Delivered | 5.36 | 4.88 | 5.92 |
| MC top 64 | 5.66 (+0.30) | 5.23 (+0.35) | 6.17 (+0.25) |

MC top 64 against the delivered map: 3.82 MAE, bias +0.03. MC top 64 agrees less with all three references, consistently.

**Stable set** (C6 SD over 2000-2020 ≤ 5; only 3 of the 63 tiles have stable-set pixels, 7,261 pixels): delivered 5.06 MAE, MC top 64
5.35 (+0.29), worse in every bin. The stable set has no pixels on the map-check tiles.

## Non-vegetated

### Held-out labels (2,160,056 pixels, 20 tiles with bare labels)

| Model | Predictors | MAE | Bias | RMSE | MAE vs delivered | Tiles better | Per-tile interval | Block interval |
|---|---|---|---|---|---|---|---|---|
| Delivered (C's set) | 21 | 4.21 | +0.10 | 7.14 | | | | |
| Seed 1 | 21 | 4.19 | +0.07 | 7.11 | −0.019 | 9 / 20 | −0.068 to +0.003 | −0.032 to −0.009 |
| Seed 2 | 21 | 4.19 | +0.07 | 7.11 | −0.021 | 12 / 20 | −0.077 to +0.001 | −0.035 to −0.010 |
| MC top 21 | 21 | 6.36 | +0.10 | 10.24 | **+2.15** | 1 / 20 | +1.28 to +3.29 | +2.01 to +2.30 |
| MC top 64 | 64 | 4.32 | +0.08 | 7.28 | **+0.11** | 4 / 20 | −0.03 to +0.24 | +0.09 to +0.13 |

- **By cover bin, MC top 64:** 0-20 +0.07, 21-40 +0.16, 41-60 +0.12, 61+ +0.16.
- **By region, MC top 64:** better in 2 of 11 strata: North America 50-80N (−0.23) and Asia 50-80N (−0.07). It is worse elsewhere,
  most in Africa 10N-10S (+1.00, only 3,282 pixels). MC top 21 is worse everywhere, by up to +12.7 (Africa 10N-10S) and +7.5 (Africa
  10-40S).
- **Seeds against delivered:** both seed refits score about 0.02 better than the delivered bare model. The block interval excludes
  zero, but the per-tile interval does not, and 9 and 12 of 20 tiles are better. This is the size of the run-to-run noise, not a
  finding.

### Maps

**Map-check tiles (h18v07, h20v06, h12v02, h09v05; 88.8 M pixels), MAE against each reference:**

| Product | C code | C6 | C6.1 | Distance from the delivered map |
|---|---|---|---|---|
| Delivered | 14.93 | 8.12 | 21.23 | |
| Seed 1 | 14.97 | 8.19 | 21.31 | 2.45 |
| Seed 2 | 15.06 | 8.23 | 21.31 | 2.52 |
| MC top 21 | 15.15 | 8.67 | 21.49 | 5.25 |
| MC top 64 | 15.17 | 8.58 | 21.41 | **3.78** |

Seed 1 against seed 2: 2.48 MAE. The large MAE against C6.1 (bias about −18) is the C6.1 high-latitude zeroing and the known
non-vegetated differences, not a property of these candidate sets.

**All 63 tiles, MC top 64:**

| Product | C code | C6 | C6.1 |
|---|---|---|---|
| Delivered | 12.39 | 8.85 | 15.22 |
| MC top 64 | 12.31 (−0.08) | 8.92 (+0.07) | 15.40 (+0.18) |

MC top 64 against the delivered map: 3.93 MAE, bias −0.43. The directions are mixed and every change is small: not a consistent
difference.

## Ranking and stability

**Top of the rankings:**
- **Tree:** CGreenMean16-Band_1, CGreenMean19-NDVI, CGreenMean16-NDVI, CGreenMean46-NDVI, CThermMean16-NDVI, CThermMean19-NDVI,
  CThermMean16-Band_1, CGreenMean13-Band_1, CThermMean19-Band_1, CThermMean16-Band_7. These are greenest- and warmest-period means,
  the same families that led on 2026-09-20.
- **Non-vegetated:** CSortedRank1-NDVI, CUnsMean46-NDVI, CUnsMean19-NDVI, CThermMean16-NDVI, CGreenMean16-NDVI, CGreenMean13-NDVI,
  CThermMean13-NDVI, CUnsMean16-NDVI, CGreenMean19-NDVI, CSortedRank0-NDVI. The MC picks NDVI almost exclusively. C's 21 bare
  attributes are mostly single monthly bands.

**Overlap with C's sets and with 2026-09-20:**

| | Shared |
|---|---|
| Tree MC top 64 ∩ C's 64 | 19 (2026-09-20: 20) |
| Tree MC top 20 ∩ C's 64 | 7 |
| Tree MC top 64 ∩ 2026-09-20 top 64 | 56 |
| Tree MC top 20 ∩ 2026-09-20 top 20 | 17 |
| Tree ranking against 2026-09-20 ranking, all 324 | Spearman 0.97 |
| Bare MC top 21 ∩ C's 21 | 1 |
| Bare MC top 64 ∩ C's 21 | 6 |

- C's tree predictors rank at a median of 125 of 324 in the new tree ranking. 27 of them are in the top 100, and the lowest is at 297.
- C's bare attributes rank from 5 to 277 in the new bare ranking; only one is in the top 21.
- The new composites and the change from bagging to RF moved the tree ranking very little. So the disagreement with C's set is not
  caused by the old aerosol screening.

**Stability (bootstrap of trials, half against half, convergence):**

| Set | Trials | Uncertain members (20-80% of resamples) | Half-vs-half overlap (median) | Spearman (median) | Convergence |
|---|---|---|---|---|---|
| Tree top 64 | 682 | 18 | 54 / 64 | 0.956 | 62-63 / 64 from 600 trials |
| Tree top 20 | 682 | 7 | 16 / 20 | 0.956 | 19-20 / 20 from 500 trials |
| Bare top 64 | 777 | 21 | 54 / 64 | 0.936 | 62 / 64 from 500 trials |
| Bare top 21 | 777 | 5 | 19 / 21 | 0.936 | 20 / 21 from 300 trials |

- The rankings have converged: overlap with the full ranking is flat from about 500 trials. The extrapolation says more trials would
  cut the uncertain members by less than a quarter.
- The membership of the last few places in each top N stays uncertain (about 9-10 expected flips in each top 64). This is expected,
  because neighbouring predictors are highly correlated.
- **Defect in the stability script's verdict line:** the "conclusive" rule applies the top-64 overlap threshold (≥ 56) to the top-20
  and top-21 sets too, so those two sets print "NOT CONCLUSIVE" automatically.
  - With the threshold scaled to the set size (56/64 = 87.5%, i.e. ≥ 18.4 of 21), **bare top 21 would pass:** 5 uncertain members,
    19 of 21 overlap, Spearman 0.94.
  - Tree top 20 fails the uncertain-member test anyway (7 > 6).
  - The numbers above are correct; only the printed verdict for the small sets is wrong. Script:
    `FullRuns/mc_rerun_20261002/code/analyze_mc_stability.py`.

## Points for the project lead's review

1. **Tree cover:** MC top 64 is a tie under the 1-point rule. Against that, it is worse on the held-out labels in every bin and every
   region, and agrees less with the C code, C6 and C6.1 on all 63 tiles. The differences are consistent across references, and they
   all point against the MC set.
2. **Non-vegetated:** MC top 64 is a tie under the rule. It has 3× the predictors of C's set (64 against 21), and the held-out labels
   and maps show no consistent gain. MC top 21, the like-for-like set size, is clearly worse (+2.15).
3. **Effect on other plans:** if C's sets are kept for both layers, `Additional_Years_Plan.md` needs no change to its predictor sets.
4. **The old shipped `MonteCarloSim` defects (open item 9 in the decisions list) are unaffected:** this run used the stand-alone
   driver.

## Caveats

- **Held-out scores are optimistic:** the held-out rows are a random 20% of pixels, neighbours of training pixels.
- **References are comparisons, not truth:** C6, C6.1 and the C code share the same cloud and historical-aerosol issues, and C6.1
  zeroes high-latitude pixels. A map moving away from them is not by itself an error.
- **Non-vegetated has fewer data:** labels on 20 tiles, and some strata are small (Africa 10N-10S: 3,282 pixels).
- **The stable set is small:** 7,261 pixels on 3 tiles, so it only adds weight to the tree result.

## Disk

| Item | Size | State |
|---|---|---|
| New 324-metric rasters (`FullRuns/mc_rerun_20261002/Python/metrics_324/`) | 1.2 TB | kept until the project lead decides |
| New models (seed refits and MC sets, both layers) | 129 GB | kept |
| Old 2026-09-20 rasters (`full63_c_vs_py_20260919-0806/Python/metrics_324/` and `metrics_324_fresh/`) | 1,076.8 GB | **Deleted 2026-10-03** (step 8, approved by the project lead): 3,141 entries (123 links, removed as links only; their targets in the old run's `Python/metrics/` are untouched). List and byte total: `FullRuns/mc_rerun_20261002/status/delete_old_metrics324.list`. Kept: `monte_carlo_standalone/` and `training_324/` |

## Where the results are

All under `FullRuns/mc_rerun_20261002/`:
- `out/heldout/<tree|bare>/`: `overall.csv`, `differences.csv`, `by_bin.csv`, `by_region.csv`, `per_tile.csv`, `predictions.parq`.
- `out/summary_mapcheck/<layer>/` and `out/summary63/<layer>/`: `summary.txt`, `pooled.csv`, `pooled_by_bin.csv`, `per_tile.csv`.
- `out/stable63/`, `out/stable_mapcheck/`; `out/gate_h08v05.json`.
- `Python/mc/<layer>/`: `ranking.csv`, `top*.txt`, `trials/`, `stability_top*/summary.txt`.
- `manifests/`: code origin (delivery branch `16d6474`), the delivered predictor lists, tiles and checksums.
- `logs/`, `status/`.
