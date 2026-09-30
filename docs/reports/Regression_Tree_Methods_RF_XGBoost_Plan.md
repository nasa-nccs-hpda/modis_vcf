# Regression tree alternatives (Random Forest, XGBoost): plan

**Status: executed 2026-09-27; kept with its estimates.** Results:
`Regression_Tree_Methods_RF_XGBoost_Results.md`.

## Goal

The 63-tile C-vs-Python run (`FullRun_Results_C_vs_Python_63tiles.md`) fit its Python tree-cover
model with `sklearn.ensemble.BaggingRegressor` wrapping a `DecisionTreeRegressor` (30 members,
bootstrap-resampled rows, all features considered at every split). That choice was deliberate: it is
the closest sklearn analogue of C's Weka `Bagging` meta-classifier around a REPTree-like learner, so
the C-vs-Python comparison would be measuring the same modeling family, not a different one. It is
not a claim that Bagging+DecisionTree is the best-performing method available.

This plan treats that pipeline as **version 1 (baseline)** and, as a **separate experiment track**,
trains the same target with two different tree-ensemble methods to see how they compare:

- **Random Forest** (`sklearn.ensemble.RandomForestRegressor`) — true Random Forest, which also
  restricts each split to a random feature subset, unlike Bagging+DecisionTree. (Note: the existing
  baseline script is named `rfFitFromPge61.py` but is not actually Random Forest; see its comment
  block. This experiment adds the real thing.)
- **XGBoost** (`xgboost.XGBRegressor`) — gradient-boosted trees, an additive-boosting ensemble
  rather than a bagging/averaging one.

This is explicitly the direct-C-comparison run's data and predictor set, not the later Monte Carlo
predictor-selection work (324 metrics, different top-64/top-20 subsets). No product code changes
here, and this does not touch the Monte Carlo results or open item 8 about adopting a different
predictor set.

## Scope

- **Target:** `pcttree` (tree cover) only. "The 64 metrics from C" is the existing
  `predictors_pcttree.txt` list built for the 63-tile run. The non-veg/bare model uses a different,
  19-predictor list (`predictors_pctbare.txt`) and is out of scope for this pass — say if you want it
  added.
- **Training data:** reuse, read-only, what the 63-tile run already built:
  `$RUN/Python/training/pcttree/{xTrain,yTrain,xTest,yTest}.parq` and
  `$RUN/manifests/predictors_pcttree.txt` (64 columns), where
  `$RUN = /explore/nobackup/people/mcarrol2/MOD44B_VCF_Development/FullRuns/full63_c_vs_py_20260919-0806`.
  No metrics or training-table rebuild — that took hours in the original run and the inputs are
  identical. Nothing under `$RUN` is written.
- **Tiles:** the same 63, `$RUN/manifests/tiles.txt`, year 2019.

## Where the new work lives

Nothing is added to `$RUN`, the shared `pge61` repo, or the `modis_vcf` repo. A new, separate run
root, following the existing dated-folder convention:

```
$RUN2 = /explore/nobackup/people/mcarrol2/MOD44B_VCF_Development/FullRuns/regtree_alt_models_20260927/
  code/validation/          new scripts only (below): fitAltModel.py, eval_heldout.py,
                            compare_regtree_variants.py, and an unmodified copy of
                            compare_tree_cover_vs_historical.py (imported as a helper module)
  Python/models/{rf,xgboost}/
  Python/predictions/{rf,xgboost}/
  Compare/regtree_2tile/, Compare/regtree_63tile/
  status/, logs/
```

`$RUN2` binds `$RUN/code/python_code` (the fixed, sklearn-using package — unmodified),
`$RUN/Python/training/pcttree/`, and `$RUN/manifests/predictors_pcttree.txt` read-only, the same way
the original run's `SING()` wrapper binds things read-only.

## Why the existing predict/compare code needs no changes

`vcfPredictQueue.py` → `VcfPredict.py` loads a model with `joblib.load`, calls
`.set_params(n_jobs=...)`, then `.predict()`. `RandomForestRegressor` and `XGBRegressor` both support
all three, and both can be `joblib.dump`ed/loaded the same way as the current
`BaggingRegressor`. So the new models only need a new **fitting** script; prediction and the
C-vs-Python compare script (`compare_c_vs_python_all.py`) are reused as-is, pointed at the new model
files.

## Steps

0. **Environment check** (cheap, ~5 min on a compute node): confirm `xgboost` imports inside the
   `modis-vcf-latest` container (`sklearn` is already used there). If missing,
   `pip install --user xgboost` once — installs under `~/.local`, no container rebuild, the same way
   other custom packages have been added on these nodes. Check node load/free memory first (other
   users have shared these nodes before).
1. **Write `fitAltModel.py`** (new, `$RUN2/code/validation/`): same CLI shape as
   `rfFitFromPge61.py` (`-o`, `--xTrainPath`, `--yTrainPath`, `--predictorFile`, `--nJobs`,
   `--randomState`), plus `--algo {rf,xgboost}` and the hyperparameters below. Writes
   `<algo>/pcttree.bin` (joblib) and `<algo>/pcttreeTopN.bin` (pickled predictor list), matching the
   baseline's file layout so the predict script needs no changes.
2. **Write `eval_heldout.py`** (new): loads a model plus `xTest.parq`/`yTest.parq`, reports R², MAE,
   and Pearson correlation. First run it against the **existing** `$RUN/Python/models/pcttree.bin` as
   a sanity check, to confirm the harness reproduces the known baseline numbers before trusting it on
   the new models.
3. **Cheap check before the expensive fit** (per your standing instruction to sanity-check before an
   hour-plus run): fit each new algorithm once at small size — Random Forest with `n_estimators=5`,
   XGBoost with `n_estimators=50` — to confirm the script runs end to end and to get a real per-unit
   timing, before committing to the full-size fit.
4. **Full fits** on `xTrain.parq` (~1.8 GB, same rows/columns as the baseline):
   - **Random Forest:** `n_estimators=30` (matches the ensemble size used for the C comparison, so
     "number of trees" is comparable), `random_state=0`, `n_jobs=12`. Cost estimate: the baseline
     Bagging+DecisionTree fit at this size took about 81 minutes (`xTrain.parq` written 15:52,
     `pcttree.bin` written 17:13 on 2026-09-19); Random Forest does less work per split (feature
     subsampling), so expect the same order of magnitude — step 3's timing will confirm before this
     runs. If held-out R² looks under-converged at 30 trees, raising it (e.g. to 100-300) is a cheap
     follow-up, at roughly proportional wall-clock cost.
   - **XGBoost:** boosting wants many shallow rounds, not 30. First-pass hyperparameters (documented
     so they're easy to override, not a tuned result): `n_estimators=500`, `max_depth=6`,
     `learning_rate=0.05`, `subsample=0.8`, `colsample_bytree=0.8`, `random_state=0`, `n_jobs=12`.
     Tuning this further is optional and out of scope for pass 1 (see below).
5. **Held-out check**: `eval_heldout.py` on `xTest.parq`/`yTest.parq` for baseline, Random Forest, and
   XGBoost side by side. A first, numeric-only sanity check — cheaper and faster than any map step —
   before spending time on real tiles.
6. **Two-tile test (h09v05, h12v02 — the standard pair from the 2026-09-18 two-tile test)**:
   - Predict just these two tiles with the new Random Forest and XGBoost models
     (`vcfPredictQueue.py`, `--tilesFile` restricted to the two tiles, `-o $RUN2/Python/predictions/{rf,xgboost}`).
     Cheap: about 2-5 minutes/tile/model (measured 2026-09-20), so roughly 20 minutes total for both
     tiles and both models.
   - **New script `compare_regtree_variants.py`** (`$RUN2/code/validation/`), adapted from the
     existing `compare_model_variants.py` pattern (copy, not edit — it already does exactly this kind
     of multi-product tile comparison for the Monte Carlo models) and reusing
     `compare_tree_cover_vs_historical.py`'s readers/statistics as `base`. For the two tiles, per your
     request it compares:
     - each new model (RF, XGBoost) **against the existing Python bagged-tree result**
       (`$RUN/Python/predictions`, read-only) — same pixels, corr/bias/MAE/RMSE/within-5/within-10;
     - each new model **against the original historical MOD44B** in `/css/modis` (the same collection
       and period as the existing comparisons: `/css/modis/Collection6/L3/MOD44B-VCF/2019/065`);
     - as a bonus, since it costs nothing extra and the data is already sitting there read-only,
       against **C's B3 output** too (`$RUN/C/production/build_context/runtime/b3_output`), so this
       two-tile table lines up with the existing 63-tile scorecards.
   - Output: a new folder under `$RUN2/Compare/regtree_2tile/` (refuses to overwrite, per the
     existing scripts' convention).
   - **Pause and confirm with you here** before spending the time on the full 63-tile map
     prediction, in case a model looks obviously broken on these two tiles (e.g. XGBoost defaults
     clearly under/overfit) and needs a hyperparameter change first.
7. **Full-tile prediction** (only after step 6 looks sane and you say go): reuse `vcfPredictQueue.py`
   unmodified, pointed at each new model, over all 63 tiles. Cost estimate: about 2-5 minutes/tile/model
   (measured 2026-09-20), so roughly 2-4 hours per model end to end at similar concurrency to the
   original run.
8. **Full compare**: reuse `compare_c_vs_python_all.py` for C vs Random Forest and C vs XGBoost
   (mirrors the existing C vs Bagging comparison), plus the same three-way-vs-historical scorecard
   from step 6, now pooled over all 63 tiles, the way `Tree_Cover_vs_Historical_MOD44B_Results.md`
   did for the baseline.
9. **Write up** `Reports/Regression_Tree_Methods_RF_XGBoost_Results.md` with the held-out, two-tile,
   and full 63-tile numbers for all three methods.

## Standing conditions carried over

- Long runs launched fully detached (`setsid nohup ... & disown`, status/done marker files), so this
  survives an SSH or VS Code disconnect.
- Nothing outside `$RUN2` (and this plan doc) is written; `$RUN`, the shared `pge61`/`modis_vcf`
  repos, and `VCF_outputs` are untouched.
- No commit or push anywhere; this is exploratory and unrelated to the sync-to-main-repos decision,
  which stays on hold.
- Check `uptime`/free memory before claiming a node; usable memory is ~65-70 GB per node under load,
  not the full 105 GB.

## Not in scope for this pass (say if you want any of these added)

- The bare/non-veg (19-predictor) model with the same two algorithms.
- Hyperparameter tuning of either method beyond the first-pass values above.
- A run-to-run seed baseline for the existing Bagging model (that is the unrelated, already-open
  "Seed baseline" option from 2026-09-20's "Resume here" list).

## Outcome

All steps executed, including the step-6 pause: the two-tile test's results were reviewed and
discussed (a real held-out-fit-vs-map-agreement tension for XGBoost tuning) before approving the full
63-tile run. See `Regression_Tree_Methods_RF_XGBoost_Results.md` for what each step actually found.
