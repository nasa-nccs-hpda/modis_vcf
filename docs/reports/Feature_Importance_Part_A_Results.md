# Feature importance of the existing 64-predictor tree-cover model (Part A)

Run 2026-09-20, 11:15 to 12:07. Plan: `Feature_Importance_and_Monte_Carlo_Plan.md`. Files are in `$RUN/Compare/feature_importance/`, where `$RUN` = `/explore/nobackup/people/mcarrol2/MOD44B_VCF_Development/FullRuns/full63_c_vs_py_20260919-0806`.
The model is `$RUN/Python/models/pcttree.bin`, a scikit-learn `BaggingRegressor` of 30 unpruned trees fit on 64 predictors chosen by C's tree-cover attribute list (the 63-tile run, `FullRun_Results_C_vs_Python_63tiles.md`). Nothing about the model or the data was changed.

## What was measured

- **Impurity importance:** the mean over the 30 trees of each tree's `feature_importances_` (how much a predictor reduces squared error where it is used to split). The values sum to 1.
- **Permutation importance:** the fall in R-squared when one predictor's column is shuffled, scored on **all 2,288,197 held-out rows** (the 20% of the 11,440,984 training rows that the model never saw), 5 repeats per predictor with fixed seeds, on four nodes. The unshuffled model scores **R-squared 0.8537** on those rows.
- Predictors are named from the model's saved predictor list, not from column order in the parquet files. Both rankings cover all 64 predictors.

## Result

| Rank | By impurity | Impurity | By permutation | R-squared drop |
|---|---|---|---|---|
| 1 | CGreenMean13-Band_1 | 0.508 | CGreenMean13-Band_1 | 0.123 |
| 2 | CGreenMean16-Band_1 | 0.104 | CGreenMean16-Band_1 | 0.095 |
| 3 | UnsortedMonthlyBands-NDVI-Day_2019065 | 0.027 | CThermMean16-Band_31 | 0.054 |
| 4 | CThermMean13-NDVI | 0.018 | UnsortedMonthlyBands-NDVI-Day_2019065 | 0.051 |
| 5 | UnsortedMonthlyBands-Band_1-Day_2019257 | 0.014 | CThermMean19-Band_31 | 0.039 |
| 6 | CThermMean16-Band_31 | 0.011 | UnsortedMonthlyBands-NDVI-Day_2019353 | 0.036 |
| 7 | UnsortedMonthlyBands-Band_2-Day_2019113 | 0.010 | CUnsMean19-Band_7 | 0.036 |
| 8 | UnsortedMonthlyBands-NDVI-Day_2020033 | 0.009 | CThermMean19-NDVI | 0.030 |
| 9 | CThermMean19-Band_31 | 0.009 | CThermMean13-NDVI | 0.027 |
| 10 | CUnsMean19-Band_7 | 0.008 | UnsortedMonthlyBands-NDVI-Day_2020033 | 0.027 |

The full ranking for all 64 is `importance_pcttree.csv`; `summary.txt` has the top 15 of each and the largest disagreements.

- **The two rankings agree at the top and diverge below it.** Both put `CGreenMean13-Band_1` and `CGreenMean16-Band_1` first and second. Spearman rank correlation between the two rankings over all 64 is **0.751**.
- **Impurity importance is very concentrated:** the first predictor carries 50.8%, the first two 61.3%, the first five 67.1% and the first ten 71.8%. Permutation importance is much flatter: the first ten predictors account for a summed drop of 0.518 and the first twenty for 0.717.
- **No predictor is dead weight.** The smallest permutation drop of any of the 64 is 0.0037, none is negative, and none is under 0.001. The permutation spread over the 5 repeats is negligible next to the mean (2.29 million rows), so the ranking order is stable at this level.
- **Derived metrics against raw band values:** of the 64 predictors, 40 are C-derived metrics and 24 are raw unsorted monthly values. Among the top 10 by permutation, 7 are derived and 3 raw; among the top 20, 13 and 7; among the top 32, 18 and 14. The 24 raw values together account for the largest single family total drop (0.363), because there are many of them.
- **Largest rank disagreements** (impurity rank, permutation rank): `CUnsMean16-Band_1` (60, 11), `CThermMean19-NDVI` (57, 8), `CGreenMean13-Band_31` (31, 56). A predictor can be rarely chosen for a split yet cost a lot when shuffled, and the reverse.

## How to read it

- **Correlated predictors share credit.** Many of the 64 are strongly related (the same statistic for neighbouring bands or periods), so a low rank means "not needed once its relatives are present", not "uninformative". The permutation drops do not add up to the model's R-squared (they sum to 1.10 against 0.854) for this reason.
- **The held-out rows are a random 20% of pixels, not of tiles.** Neighbouring pixels sit on both sides of the split, so the R-squared of 0.854 and the drops are optimistic against a tile-level held-out test.
- **This ranks only the 64 predictors C chose.** It says nothing about the other 260 of the 324 metrics. That question is what the Monte Carlo (Part B, done; `Feature_Importance_and_Monte_Carlo_Results.md`) is for; it measures importance in random 10-predictor subsets, a different measure, so it is not expected to reproduce this ranking.
- **It describes the Python model**, not C's Weka models.

## Timings

Baseline scoring of 2.29 million rows took 47 seconds and each predictor-repeat about 35 seconds on a busy node (6 threads). Four nodes ran 16 predictors each in about 52 minutes; impurity importance took under 2 minutes. The four permutation jobs shared their nodes with the metric jobs of step 3.
