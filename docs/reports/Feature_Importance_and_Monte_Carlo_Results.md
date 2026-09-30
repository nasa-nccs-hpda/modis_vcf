# Results: stand-alone Monte Carlo predictor selection on all 324 metrics (tree cover), 2026-09-20

Plan: `Feature_Importance_and_Monte_Carlo_Plan.md`. Part A (the existing 64-predictor model) is in `Feature_Importance_Part_A_Results.md`. This document covers Part B, the Monte Carlo (MC), its final models, and the decision on running further.

`$RUN` = `/explore/nobackup/people/mcarrol2/MOD44B_VCF_Development/FullRuns/full63_c_vs_py_20260919-0806`; MC outputs are in `$RUN/Python/monte_carlo_standalone/tree/` (`$MC` below).

## Short answers

1. **The MC ran to its own stop rule.** 702 trials, every one of the 324 candidate metrics used in 10 to 35 trials (the rule was at least 10), and all 324 have a positive average permutation importance. Workers started 15:26 and the stop marker was written at 18:05, about 2 hours 40 minutes on five nodes.
2. **The MC's top 64 is a very different set from the 64 the existing model uses.** Only **20 of 64 are shared**. The existing model's 64 have a median MC rank of 137.5 of 324 (range 1 to 306; 27 of them are in the MC's top 100).
3. **The new top-64 model is slightly worse, not better, on the same held-out rows:** R-squared 0.8484 against 0.8537, mean absolute error 7.63 against 7.46 (tree-cover points), correlation 0.9213 against 0.9242. The difference is small. The top-20 model is clearly worse (0.7922, 9.22).
4. **The ranking is not "conclusive" by the rule fixed beforehand, but more trials would not fix that.** Decision: **do not continue to `minVarUsage` 15.** Reasons are in "Decision on more trials".
5. **On maps of three tiles (h12v02, h09v05, h12v09) the same ordering holds.** Against the historical MOD44B on the same 65.3 million pixels, mean absolute error is **4.98 for the existing model, 5.22 for the new top 64 and 5.55 for the top 20** (C: 3.67). The top-64 map is worse than the existing map on all three tiles, by 0.11 to 0.37 points, which is about a fifth of the gap between Python and C. **The results change but they do not improve:** pixel by pixel the top-64 and existing maps differ by a mean absolute 4.60 points, about as much as the existing Python map differs from C (4.39). See "Map-level check".
6. **What this does not show:** which set is better for the final product in general. The held-out rows are a random 20% of pixels (neighbours of training pixels), so all held-out scores are optimistic; the map check covers only three tiles, uses the historical product (not ground truth) as the reference, and has no run-to-run baseline, so how much of the 4.60-point difference between the two Python maps is the predictor set and how much is the random tree-building is not known.

## What ran

| Item | Value |
|---|---|
| Candidate pool | 324 metrics: 72 unsorted monthly band values plus 252 derived (28 families x 9 bands), from the corrected metric code; table checked against the rasters for all 63 tiles (`Feature_Importance_and_Monte_Carlo_Plan.md`, Step 3) |
| Data | 500,000 training rows and 500,000 evaluation rows per trial, fixed-seed subsamples of the 80% / 20% split (the split is the same as the existing model's: same file order, seed 0; acceptance test PASS) |
| Trial | 10 randomly drawn predictors, 30-tree unpruned bagging regressor (same as the Python model), permutation importance on the evaluation rows |
| Stop rule | every predictor used in at least `minVarUsage` = 10 trials **and** at least 64 predictors with non-zero average importance; caps of 1,200 trials or 6 hours (not reached) |
| Trials | **702** at the stop marker (18:05:09); the coordinator's ranking and the stability analysis both use these 702. 15 more trial files, from trials the workers had already started, were written after the marker; **717 files exist** and the 15 are in neither the ranking nor the analysis |
| Compute | 5 nodes (forest201 to 205), 3 workers each, 15:26 to 18:05 (the first trial file is at 16:45, after the data preparation step); about 265 trials per hour |
| Final fits | started 18:09 (after a 3-minute wait for the workers to finish their current trial); top 64 saved 19:28:56, top 20 at 20:00:14, evaluation written 20:00:53 |

An earlier draft of the plan said 703 trials at the stop; the count of files not newer than the marker, and the sum of the ranking's usage counts, are both 702.

## The ranking

Full list: `$MC/ranking.csv` (324 rows: rank, predictor, average permutation importance, trials used in). With bootstrap membership frequencies: `$MC/stability/predictor_stability.csv`.

Top 10 (average permutation importance, a drop in R-squared when the predictor is shuffled, measured in 10-predictor models):

| Rank | Predictor | Importance | Trials used in |
|---|---|---|---|
| 1 | CThermMean19-NDVI | 0.552 | 20 |
| 2 | CGreenMean16-Band_1 | 0.539 | 19 |
| 3 | CGreenMean16-NDVI | 0.494 | 24 |
| 4 | CThermMean16-Band_1 | 0.468 | 16 |
| 5 | CGreenMean19-NDVI | 0.466 | 21 |
| 6 | CThermMean16-NDVI | 0.462 | 24 |
| 7 | CThermMean16-Band_7 | 0.398 | 12 |
| 8 | CThermMean19-Band_1 | 0.395 | 26 |
| 9 | CGreenMean13-Band_1 | 0.395 | 24 |
| 10 | CGreenMean46-NDVI | 0.394 | 19 |

Importance at rank 20 is 0.317, at rank 64 0.176, at rank 65 0.173, at rank 100 0.139 and at rank 324 0.012. So the boundary of the top 64 lies in a stretch where neighbouring predictors differ by about 0.003.

**What the top 64 consists of**
- **63 derived, 1 raw monthly value** (`UnsortedMonthlyBands-NDVI-Day_2019065`). The existing model's 64 has 24 raw monthly values and 40 derived.
- By family: the green-season means (`CGreenMean`, 12), thermal-season means (`CThermMean`, 20), unsorted-period means (`CUnsMean`, 10), sorted-rank metrics (`CSortedRank0` to `6`, 19), and `CGreenMax` and `CThermMax` (1 each). The largest single groups are `CThermMean16` (7), `CThermMean19` (5) and `CThermMean13` (5).
- By band: NDVI 20, Band_1 17, Band_7 11, Band_4 7, Band_31 6, Band_6 2, Band_3 1.

## Comparison with the existing model (Part A) and with C's 64

- **Shared with the existing model's 64: 20.** They are `CThermMean19-NDVI`, `CGreenMean16-Band_1`, `CGreenMean16-NDVI`, `CThermMean16-NDVI`, `CGreenMean13-Band_1`, `CThermMean19-Band_7`, `CGreenMean13-NDVI`, `CUnsMean19-Band_7`, `CThermMean13-Band_1`, `CThermMean13-NDVI`, `CThermMean19-Band_31`, `CThermMean13-Band_7`, `CUnsMean19-Band_1`, `CUnsMean16-NDVI`, `CUnsMean16-Band_1`, `CThermMean16-Band_31`, `CUnsMean16-Band_7`, `UnsortedMonthlyBands-NDVI-Day_2019065`, `CThermMean13-Band_6`, `CUnsMean46-NDVI`.
- The existing model's 64 predictors are C's list of 64 (the list the Python model was given for the 63-tile run). Their MC ranks: median 137.5, minimum 1, maximum 306; 27 are in the MC's top 100.
- **Why they can disagree** (this is my reading of how the MC works, not something tested): each trial fits a model on only 10 random predictors, so a predictor's score is how much it adds to a model that has few others. Many metrics carry overlapping information (for example several band and period combinations of the same season mean), so a metric can score highly in every 10-predictor trial and still add little once its neighbours are also in the model. The ranking therefore favours predictors that are informative alone, and it does not measure redundancy. A 64-predictor set built from the top of this list contains many near-duplicates (seven `CThermMean16` bands, for example). This would explain a top-64 set that scores slightly below a set chosen differently, but it has not been tested (for example by selecting for low correlation, or by a greedy forward selection).
- Part A's ranking (permutation importance inside the 64-predictor model) and this ranking measure different things: Part A's top predictors, `CGreenMean13-Band_1` and `CGreenMean16-Band_1`, are 9th and 2nd here, so the two agree on the leaders but not on the order below them.

## The final models, on the same 2,288,197 held-out rows

| Model | Predictors | R-squared | Mean absolute error | Correlation |
|---|---|---|---|---|
| Existing (63-tile run) | C's 64 | **0.8537** | **7.46** | **0.9242** |
| New, MC top 64 | 64 (63 derived) | 0.8484 | 7.63 | 0.9213 |
| New, MC top 20 | 20 | 0.7922 | 9.22 | 0.8902 |

- Top 64 against existing: R-squared lower by 0.0053, mean absolute error higher by 0.17 points. The two models share only 20 predictors yet score within 0.005 of each other, so the score is **not sensitive to which of the marginal predictors are in the set**. That matters for the decision below.
- Going from 64 to 20 predictors costs 0.056 of R-squared and 1.6 points of mean absolute error. The top-20 model is a reference, not a proposal.
- Both new models are fitted on the full training rows (about 9.1 million; the 80% split), like the existing model, so the comparison is like for like. The saved models are `$MC/pcttree.bin` and `pcttreeTopN.bin` (top 64, 13.6 GB) and `$MC/final_top20/` (14.7 GB). Nothing was written into `$RUN/Python/models/`.
- **The scores are optimistic for the reason given at the top:** a random pixel split, so neighbouring pixels are on both sides. The comparison between the models is affected the same way, but a difference this small should not be read as a ranking of the two selections.
- The models were then run through the tile prediction for three tiles; see "Map-level check" below.

## How stable is the ranking?

The rule was written before these numbers were computed (`Feature_Importance_and_Monte_Carlo_Plan.md`, "Decision rule for continuing"). Results, from `$MC/stability/summary.txt`:

| Measure | Result | Needed for "conclusive" |
|---|---|---|
| A. Predictors whose top-64 membership is uncertain (in the top 64 in 20 to 80% of 300 bootstrap resamples of the trials) | **25** (about 11 expected membership flips) | 6 or fewer |
| B. Median top-64 overlap between two random halves of the trials (200 splits, about 351 trials each) | **53 of 64** (5th percentile 50) | 56 or more |
| B. Median top-20 overlap between halves | 15 of 20 | (not part of the rule) |
| B. Median Spearman correlation between halves | **0.953** (5th percentile 0.944) | 0.90 or more |
| C. Convergence: first 300 / 400 / 500 / 600 / 650 trials against all 702 | overlap 56 / 60 / 62 / 62 / 63 of 64; Spearman 0.982 to 0.997 | (context) |
| D. Uncertain predictors at 350 / 500 / 702 trials, and extrapolated to 1,050 | 32.8 / 28.2 / 27.0, and **23.7** at 1,050 (a fall of 12%) | fall of at least 25% to say more trials help |

Reading: the ranking as a whole is stable (Spearman 0.95 between halves), the leaders are firmly the leaders, and the disagreement is at the cutoff, where about 25 predictors are almost tied. The convergence numbers in C compare the first n trials with the set they are part of, so they flatter the result; measures A, B and D are the honest ones.

## Decision on more trials or a higher `minVarUsage`

**Decision: do not continue. Stop at `minVarUsage` 10 with 702 trials.** Nothing further was launched.

1. **By the pre-registered rule:** the ranking is not conclusive (25 uncertain predictors against 6 allowed; half-against-half overlap 53 against 56), but the uncertainty would fall only about 12% at 1,050 trials (what `minVarUsage` 15 needs), against the quarter required to say more trials would help.
2. **The uncertainty is in near-ties, which more trials do not resolve.** Predictors ranked around 64 differ in importance by about 0.003; each extra trial adds a use to only 10 of 324 predictors, so the numbers there move slowly, and the differences may simply be smaller than the trial-to-trial noise.
3. **The final model does not depend on those ties.** The top-64 set and the existing set share only 20 predictors and their scores differ by 0.005 R-squared. Swapping a handful of marginal predictors in or out will not visibly change the model.
4. **Cost is not the reason.** `minVarUsage` 15 needs about 916 trials for a 50% chance and 1,023 for 90%, that is 214 to 321 more trials, roughly 0.8 to 1.2 hours on five nodes. If you want it done anyway (for example so that the ranking rests on a round number), it is cheap and the plan for doing it, including moving the minvar-10 results aside first, is in the plan document. It would not be expected to change the conclusions here.

What would change the picture is not more of the same trials. The map-level check below did not change the decision, and shows the top-64 model is not better than the existing one. What is still unknown is listed at the end of "Map-level check"; none of it was run.

## Map-level check on three tiles (test 4; predictions 20:28 to 20:47, comparison 20:47, 2026-09-20)

**What was done.** The three saved models each predicted `Percent_Tree_Cover` for h12v09, h12v02 and h09v05 (2019) through the same code path as the 63-tile run (`VcfPredict._runPrediction` and `_write`), reading the 324-metric rasters read-only. Nothing in `Python/predictions/` or the earlier `Compare/` folders was touched. Maps and tables are in `$RUN/Compare/mc_model_prediction/` (`mc_top64/`, `mc_top20/`, `control_existing/`, `comparison/`); scripts in `$RUN/code/validation/` (`predict_tree_cover_model.py`, `run_predict_model_node.sh`, `compare_model_variants.py`, `run_variant_compare.sh`). Each tile took 2 to 5 minutes; the whole set ran in parallel on three nodes.

**Control (passed).** The existing model, run through the new script and the 324-metric rasters, reproduces the existing maps **exactly: 0 of about 23 million pixels differ on each of the three tiles.** So the new input path adds nothing, and every difference below comes from the model.

**Against the historical MOD44B Collection 6, pooled over the three tiles** (65,265,789 pixels valid in the historical product, C and all four Python maps, so every product is scored on the same pixels; bias is product minus historical):

| Product | MAE | RMSE | Bias | Correlation | Within 5 points | Within 10 |
|---|---|---|---|---|---|---|
| C (B3) | **3.67** | 6.07 | -0.48 | 0.979 | 77.1% | 91.8% |
| Python, existing model | 4.98 | 7.67 | +0.64 | 0.967 | 67.5% | 85.3% |
| Python, Monte Carlo top 64 | 5.22 | 8.04 | +1.26 | 0.965 | 66.1% | 84.0% |
| Python, Monte Carlo top 20 | 5.55 | 8.51 | +0.64 | 0.959 | 64.2% | 82.3% |

Mean absolute error per tile:

| Tile | C | Existing | Top 64 | Top 20 |
|---|---|---|---|---|
| h12v09 (mean cover about 64%) | 5.55 | 7.04 | 7.15 | 7.40 |
| h12v02 (about 16%) | 4.16 | 5.13 | 5.50 | 6.70 |
| h09v05 (about 8%) | 1.41 | 2.85 | 3.10 | 2.73 |

- **The top 64 is worse than the existing model on all three tiles** (by 0.11, 0.37 and 0.26 points) and pushes the map higher on average (bias +1.26 against +0.64). The held-out numbers point the same way, in the same size: mean absolute error 7.63 against 7.46 on held-out pixels, 5.22 against 4.98 on these maps.
- **The top 20 is worse on two tiles and slightly better on h09v05** (2.73 against 2.85), the tile with the least tree cover; pooled it is worse by 0.57. Its bias equals the existing model's (+0.64) by coincidence of averaging: it is lower on some tiles and higher on others.
- **C stays clearly closest** on every tile. On these three tiles the existing Python model is 1.31 points above C; the top 64 adds 0.24 to that, so the predictor set is a small part of the difference between Python and C.
- **By historical cover** (`by_cover_bin.csv`, no pixels above 90% on these tiles): the top 64 is worse than the existing model in every bin from 1 to 70 percent (for example 8.74 against 7.94 in the 11 to 20 bin) and better only at 71 to 80 percent (5.60 against 6.11; bias -1.74 against -3.31). It reduces the under-prediction of dense cover and increases the over-prediction of sparse cover.
- **The maps differ a lot pixel by pixel.** Top 64 against existing, same pixels: mean absolute difference 4.60 points (6.37 on h12v09, 4.49 on h12v02, 2.97 on h09v05), within 5 points on 68.6% of pixels, a 90th percentile difference of 12. That is close to the existing Python map against C (4.39, within 5 on 70.6%), and much larger than the 0.24 difference in their errors against the historical product. Pixel by pixel neither map is consistently closer to the historical product: the top-64 map is closer on 29 to 45% of pixels per tile and the existing map on 33 to 46% (the rest are equal, mostly zero-cover pixels), `closer.csv`.

**Reading.** Changing the predictor set changes individual pixels by a lot, and it does not improve the agreement with the historical product; the Monte Carlo selection is slightly worse here. This is the same answer as on the held-out rows. It does not say the ranking is wrong (see "The final models" for why a redundant top 64 is plausible), and it does not say the existing 64 is the best set.

**Limits.**
- Three tiles, chosen by the user, not a sample of the 63; the existing Python model's error on them (4.98) is close to its 63-tile figure (5.00), so they are not unusual, but only one tile has dense cover.
- No significance test; neighbouring pixels are strongly correlated, so 65 million pixels are not 65 million independent observations. A 0.24-point difference in mean absolute error should be read as small and consistent across the three tiles, not as proven.
- The historical product is a reference with different inputs, not ground truth (`Tree_Cover_vs_Historical_MOD44B_Results.md`).
- **There is no run-to-run baseline.** How much of the 4.60-point pixel difference between the top-64 and existing maps would appear between two models with the same 64 predictors but a different random seed is unknown. That baseline (refit the existing 64 with another seed and compare its map with the existing map) was not part of this test.

**Not done, and would tell more:** the run-to-run baseline above; a random 64-predictor set as a null (does the selection matter at all); a split of the held-out data by tile; a redundancy-aware selection. Map figures of the new maps were not made. All of these need your go-ahead.

## Things to know

- **The MC as shipped is not what was run.** The shipped `MonteCarloSim` has defects (final-model arguments, an unassigned variable, a counter reset, permutation over all rows including training rows, out-of-date tests); the stand-alone driver `run_monte_carlo_standalone.py` avoids them and does not edit rlgill's files. The list is in the plan ("Defects in the Monte Carlo code"). This ranking is therefore what the shipped class's method would give with those defects removed, not what the shipped pipeline produces.
- Another user's processes (about 17 processes, 48.6 GB) shared forest205 for part of the run; they had gone by 19:30 and did not affect any result, only timing.
- All MC runs used detached processes with status and done markers (`$RUN/status/mc_full_*`).

## Where things are

| What | Path |
|---|---|
| Ranking, 324 predictors | `$MC/ranking.csv`, `$MC/SortedMetrics-*.txt` |
| Stop marker | `$MC/STOP` ("conditions met") |
| Trials | `$MC/trials/` (717 files; 702 used) |
| Stability analysis | `$MC/stability/` (`predictor_stability.csv`, `convergence.csv`, `extrapolation.csv`, `summary.txt`) |
| Final top-64 model | `$MC/pcttree.bin`, `$MC/pcttreeTopN.bin` (13.6 GB) |
| Final top-20 model | `$MC/final_top20/` (14.7 GB) |
| Scores | `$MC/final_evaluation.txt` |
| Pilots and smoke test (about 1 GB each) | `$MC/smoke/`, `$MC/pilot_p250/`, `$MC/pilot_p500a/`, `$MC/pilot_p500b/` |
| Part A (existing model) | `$RUN/Compare/feature_importance/` |
| Scripts | `$RUN/code/validation/run_monte_carlo_standalone.py`, `analyze_mc_stability.py`, `run_mc_final_chain.sh` |
| Map-level check (three tiles) | `$RUN/Compare/mc_model_prediction/` (maps for `mc_top64`, `mc_top20`, `control_existing`; tables in `comparison/`); scripts `predict_tree_cover_model.py`, `run_predict_model_node.sh`, `compare_model_variants.py`, `run_variant_compare.sh` |
