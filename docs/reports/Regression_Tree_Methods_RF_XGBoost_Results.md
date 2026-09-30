# Regression tree alternatives (Random Forest, XGBoost): results

**Status: done.** Plan: `Regression_Tree_Methods_RF_XGBoost_Plan.md`. Run root
`$RUN2 = /explore/nobackup/people/mcarrol2/MOD44B_VCF_Development/FullRuns/regtree_alt_models_20260927/`.
Baseline run `$RUN = FullRuns/full63_c_vs_py_20260919-0806/` (the 63-tile C-vs-Python run) is untouched;
everything here reused its training data, predictor list and tile list read-only. Written 2026-09-27.

## Bottom line

**Random Forest and XGBoost (tunedB: 2000 rounds, depth 6, learning rate 0.02) both beat the
existing Python baseline (BaggingRegressor+DecisionTreeRegressor) at full 63-tile scale**, confirmed
against the historical MOD44B, and both fit far faster than the baseline (81 minutes):

| Model | Fit time | 63-tile corr vs historical | 63-tile MAE vs historical |
|---|---|---|---|
| Baseline (Bagging+DecisionTree) | 81 min | 0.9522 | 5.06 |
| **Random Forest** (30 trees) | **10.1 min** | **0.9627** | **4.48** |
| **XGBoost tunedB** (2000/depth6/lr0.02) | **29.9 min** | **0.9685** | **4.02** |
| C (B3), for reference | - | 0.9714 | 3.54 |

Both are adopted as this experiment's outcome. XGBoost tunedB closes more of the gap to C than
Random Forest does, at roughly 3x the fit cost; Random Forest is essentially free by comparison and
still a clear win over the baseline. Three other configurations were tried and set aside along the
way (untuned XGBoost, XGBoost tunedA, sklearn's HistGradientBoostingRegressor) - each answered a
specific question and is recorded below, but none is carried forward.

## What was tried, and why

The baseline's fitting method (`BaggingRegressor(DecisionTreeRegressor())`, mirroring C's Weka
Bagging) is a deliberate choice for the C-vs-Python comparison, not a claim that it is the best
available method. This experiment tried two genuinely different tree-ensemble families on the same
64 predictors and the same training data:

| Model | Algorithm | Hyperparameters | Fit time |
|---|---|---|---|
| Baseline | `BaggingRegressor(DecisionTreeRegressor)` | 30 members, bootstrap, all features/split | 81 min |
| Random Forest | `sklearn.ensemble.RandomForestRegressor` | 30 trees, `max_features='sqrt'` | 10.1 min |
| XGBoost (untuned) | `xgboost.XGBRegressor` | 500 rounds, depth 6, lr 0.05, subsample/colsample 0.8 | 7.2 min |
| XGBoost tunedA | same | 1500 rounds, depth 8, lr 0.03 | 27.4 min |
| **XGBoost tunedB** | same | **2000 rounds, depth 6, lr 0.02** | **29.9 min** |
| HGB (tunedB-equivalent) | `sklearn.ensemble.HistGradientBoostingRegressor` | `max_iter` 2000, depth 6, lr 0.02 (no subsample knob) | 13.3 min |

All fits used the same 9,152,787-row, 64-predictor `xTrain.parq`/`yTrain.parq` from the baseline run,
`random_state=0`, `n_jobs=12`.

## Held-out check (`xTest`/`yTest`, before touching any tile)

| Model | R² | MAE | RMSE | corr | bias |
|---|---|---|---|---|---|
| Baseline | 0.8537 | 7.46 | 11.71 | 0.9242 | +0.10 |
| Random Forest | 0.8542 | 7.52 | 11.69 | 0.9246 | +0.10 |
| XGBoost untuned | 0.8128 | 8.86 | 13.24 | 0.9016 | +0.02 |
| XGBoost tunedA | 0.8425 | 7.94 | 12.15 | 0.9179 | +0.02 |
| XGBoost tunedB | 0.8217 | 8.59 | 12.92 | 0.9066 | +0.02 |
| HGB (tunedB-equivalent) | 0.8133 | 8.84 | 13.22 | 0.9019 | +0.02 |

Random Forest ties the baseline. Every boosting variant - XGBoost at any tuning, and HGB - keeps a
much smaller bias (+0.02) than Bagging/RF (+0.10); that gap tracks the modeling family, not tuning.
Among the XGBoost configs, depth mattered more than round count: tunedA (deeper, fewer rounds) beat
tunedB (shallower, more rounds) on this metric.

## Two-tile map check (h09v05, h12v02), before the expensive 63-tile run

Against C, the baseline, and the historical MOD44B:

| Model | corr vs hist | MAE vs hist | bias vs hist |
|---|---|---|---|
| C (B3) | 0.9522 | 2.74 | +0.31 |
| Baseline | 0.9223 | 4.00 | +1.31 |
| Random Forest | 0.9390 | 3.51 | +1.33 |
| XGBoost untuned | **0.9517** | **2.94** | +0.30 |
| XGBoost tunedA | 0.9429 | 3.16 | +0.50 |
| XGBoost tunedB | 0.9494 | 2.99 | +0.37 |

**A real tension, not a fluke**: the better an XGBoost config fit the held-out labels (tunedA: R²
0.8425, best), the *worse* it matched the historical product on these two tiles (tunedA: MAE 3.16,
worst of the three). The untuned config was the best map match despite the worst label fit; tunedB
sits in between on both axes - the reason it was chosen for the full 63-tile run over tunedA (best
label fit) or the untuned config (best two-tile map match but essentially no tuning behind it): a
deliberate middle ground rather than optimizing for either single metric.

## The HGB detour

Tried once, at the project lead's suggestion, to check whether XGBoost's low-bias-vs-historical
pattern was an XGBoost implementation quirk or a boosting-family trait: sklearn's
`HistGradientBoostingRegressor`, given tunedB's rounds/depth/learning-rate. Held-out bias matched
every XGBoost config almost exactly (+0.017 against XGBoost's +0.017-0.019) - **confirms the
low-bias pattern belongs to gradient boosting generally, not to XGBoost specifically.** Its label-fit
score (R² 0.8133) landed close to XGBoost's *untuned* config, most likely because sklearn's HGB has
no row/column subsampling knob to match XGBoost's `subsample=0.8, colsample_bytree=0.8`
regularization - not a fair comparison of the two libraries' ceilings. Per the project lead's
instruction, not tuned further and not carried to the map or 63-tile stage.

## Full 63-tile result (Random Forest and XGBoost tunedB)

Predicted all 63 tiles (`vcfPredictQueue.py`, unmodified, split across 2 nodes per model for speed:
Random Forest finished in about 1h52-54m wall clock, XGBoost tunedB in about 2h29-30m, both with 0
tile failures), then compared against C, the baseline, and the historical MOD44B on the common-valid
pixels (1,085,891,778 pooled):

| Product | corr | bias | MAE | RMSE | within10 |
|---|---|---|---|---|---|
| C (B3) | 0.9714 | -0.12 | 3.54 | 6.06 | 92.0% |
| Baseline (Bagging) | 0.9522 | +0.27 | 5.06 | 7.83 | 85.1% |
| Random Forest | 0.9627 | +0.24 | 4.48 | 6.92 | 88.2% |
| XGBoost tunedB | 0.9685 | -0.18 | 4.02 | 6.40 | 90.4% |

Both new models track the two-tile test's direction: consistent, meaningful improvements over the
baseline at full scale, XGBoost tunedB closer to C than Random Forest is.

## Data-quality check on these numbers

The historical MOD44B product disguises a real MODIS-sinusoidal-projection out-of-bounds area as
ordinary-looking tree-cover values (found and confirmed while looking at these same high-latitude
tiles - see `Valid_Mask_Disagreement_MOD44W_Root_Cause.md`). Checked whether that contaminates the
63-tile scorecard above: **no, negligibly** (about 206,856 of 1,085,891,778 pooled pixels, 0.019%),
because Random Forest and XGBoost tunedB share the baseline's masking pipeline and are themselves
invalid across nearly all of the same contaminated area, so the multi-product common-valid mask used
here already excludes it. (The *original*, separately published C-vs-historical pairwise comparison
in `Tree_Cover_vs_Historical_MOD44B_Results.md` is a different story - flagged there, not corrected.)

**A second, separate caveat found the same day, checking whether predictions are clamped to 0-100**:
they are not, anywhere in the code. Harmless for the baseline and Random Forest - both are
mathematically bounded to the training range and produced zero out-of-range pixels across all 63
tiles. **XGBoost tunedB is not naturally bounded and genuinely produces out-of-range values**: min
-22, ~1.4% of all pixels (20,977,212) outside `0-100`. These again don't affect the scorecard above -
the same `0-100` valid-range convention used throughout this project already excludes them from every
metric - but they are real, physically-meaningless values sitting in the raster (`Percent_Tree_Cover`
should never be negative). **Fixed and validated the same session** (staged, not yet regenerated into
the existing 63-tile files): `NonTreeVeg_Water_OutOfProj_Code_Propagation_Plan.md`.

## Not done / open

- **Bare/non-veg model** was not tried with either algorithm - scope was tree cover only throughout,
  per the plan.
- **Further XGBoost tuning** could plausibly close more of the remaining gap to C; tunedB was a
  reasoned middle ground between two tried extremes, not a search optimum.
- **HGB tuning** was deliberately not pursued further (see above).
- **Per-pixel ensemble stability** (a standard-deviation-across-trees layer, the way C's own 30-model
  ensemble and Random Forest both naturally support) was discussed but not implemented for either
  model.
- **(Resolved, not "not done")** An explicit 250 ("out of projection") code for Python's prediction
  output was thought to be missing (README open item 11, since superseded); checking the full code
  found it already implemented and correct for `Percent_Tree_Cover`/`Percent_NonVegetated`. The one
  real gap found - `Percent_NonTree_Vegetation` not propagating it - is now fixed and validated, along
  with the clamp above, in `NonTreeVeg_Water_OutOfProj_Code_Propagation_Plan.md` (staged in
  `Git_delivery/modis_vcf/model/VcfPredict.py`, not yet regenerated into the existing 63-tile files).
- Unrelated, already-open items from the 63-tile run (a seed baseline, a random-64-predictor null,
  split-by-tile validation) were not touched by this experiment.

## Where things are

`$RUN2/Python/models/{rf,xgboost_tunedB}/`, `$RUN2/Python/predictions/{rf,xgboost_tunedB}/` (63
tiles each), `$RUN2/Compare/regtree_63tile/` (this comparison's csv/summary files),
`$RUN2/code/validation/` (`fitAltModel.py`, `eval_heldout.py`, `compare_regtree_variants.py`, a copy
of `compare_tree_cover_vs_historical.py`, and the `run_*.sh` launchers). Exploratory outputs for the
untuned/tunedA/HGB configurations were cleaned up after this write-up; the scripts that can
reproduce any of them (and the numbers above) were kept.
