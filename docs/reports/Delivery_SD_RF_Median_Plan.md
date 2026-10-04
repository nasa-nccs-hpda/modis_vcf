# Delivery: SD layers, RF for non-vegetated, MEDIAN tree cover (plan)

**Written 2026-10-01. Status: APPROVED 2026-10-01, with non-vegetated = MEAN (provisional; reason: this layer already changes a lot at once — dynamic models instead of static ones, RF instead of bagging, and the tree-cover QA changes; may be revisited).**

Decisions (project lead, 2026-10-01; `RF_Aggregation_Maps_63Tile_Results.md`):
- SD layer approved;
- RF approved for non-vegetated;
- tree cover is the MEDIAN of the 30 RF trees.

**The non-vegetated combination method (MEAN, MID20 or MEDIAN) is still open.** This plan makes it a parameter, so the code does not wait on that decision. Only the 63-tile run (step 4) needs it.

Branch: `mcarrol2/c-python-parity-fixes` in `AI_code_sync_09152026/Git_delivery/modis_vcf`. It is 3 commits ahead of GitHub (the local predictor-list commit and two report snapshots). New commits go there, and the project lead pushes. Nothing in rlgill's checkouts is touched.

## Code changes

1. **`model/VcfPredict.py`: one pass over the members gives the value and the SD.**
   - A new helper, `_memberPredict(model, X)`, does the following:
     - predicts each member (RF: `estimators_`; `BaggingRegressor`: `estimators_` with `estimators_features_`) in chunks of 1 M rows on the existing thread pool;
     - clips each member's prediction to 0-100;
     - returns the combined value and the population SD.
   - The combination is set per layer by new `__init__` arguments, `treeCoverAgg` and `nonvegAgg`. Each takes `mean`, `mid20` (drop the 5 highest and 5 lowest of 30) or `median`. **The default is `mean`,** so existing callers are unchanged.
   - Models without members (XGBoost) keep using `predict()` and get no SD layer; the log says so. Asking for `mid20` or `median` with such a model is an error, not a silent fallback.
   - The value is cast to int16 by truncation, as `predict()` output is today and as in the aggregation run. That keeps it exactly reproducible against the run's maps. **Rounding instead of truncating is a separate question,** not part of this change.
   - The SD is stored as round(100 × SD). Water and out-of-projection are 0. The written files are `Percent_Tree_Cover_SD-<tile>-<year>.tif` and `Percent_NonVegetated_SD-<tile>-<year>.tif`, as in the SD test.
     - These stay Int16 GeoTIFFs for now. uint16 with scale 0.01 comes with the HDF writer (to-do item 9).
     - The SD measures the spread of the members around their mean, whatever combination is chosen for the value. It describes how much the members disagree.
     - A switch, `writeSd`, turns it off; it is on by default.
   - **No other change in `runTileForYear`.** The sum cap stays as it is: non-vegetated is reduced to 100 − tree cover, and non-tree vegetation is the remainder. It now works on MEDIAN tree cover.
2. **`view/vcfPredict.py` and `view/vcfPredictQueue.py`:** new options `--tcAgg` and `--nvAgg` (default `mean`) and `--noSd`.
3. **Unit tests** (`model/tests/test_VcfPredict_members.py`), on a small RF and a small `BaggingRegressor` fitted on synthetic data:
   - `mean` equals `predict()`;
   - `median` equals `np.median` over the members;
   - `mid20` equals the trimmed mean;
   - the SD equals `np.std` over the members;
   - XGBoost with `median` raises an error.
4. **Docs:**
   - The module docstring and the branch README give the recommended fits for both layers: `fitAltModel.py --algo rf --nEstimators 30 --maxFeatures sqrt --randomState 0`, with bit-15 metrics and C's 21 bare predictors for non-vegetated.
   - They also give the prediction options. The `docs/reports` snapshot is refreshed with the same redaction as before.

## Steps

0. **Cheap check (h08v05, ngaproc201, about 10 minutes, then stop for review).** Run the patched code from a copy in a new run root, `FullRuns/delivery_sd_rf_median_<date>/code/`, using the bit-15 tree RF and the non-vegetated RF from `nonveg_bit15_models_20261001`. Then check:
   - tree cover equals the aggregation run's `out/MEDIAN/Percent_Tree_Cover-h08v05-2019.tif` exactly;
   - both SD layers equal `rf_aggregation_maps_20261001/out/SD` exactly;
   - non-vegetated equals `min(MEAN bare, 100 − MEDIAN tree)` from the aggregation run on land, and is 200/250 on water and out-of-projection;
   - the three layers sum to 100 on land, with no negative values;
   - **report how many pixels the cap changes:** tree + bare > 100 is more common with MEDIAN tree cover, which is higher in dense forest;
   - run time and peak memory per worker.
1. **Code on the branch:** the same patch as step 0, with the unit tests run in the container and the docs updated.
2. **Re-run step 0 from the branch code itself,** not the copy, to confirm the committed code gives identical output.
3. **Commit locally** in logical pieces: member pass and aggregation; SD output; CLI options; tests; docs and reports snapshot. **No push.** The project lead pushes, and the pull request into `develop` is still to be opened.
4. **63-tile production run (needs the non-vegetated decision):**
   - all 63 tiles from the branch code, with tree MEDIAN, non-vegetated `<decision>`, and SD on;
   - **ngaproc201/202 only, 3 workers each** (attempt 1 of the aggregation run ran out of memory with 2 workers per forest node);
   - fully detached, with claims, markers and `ALL.done`;
   - checks on every tile: tree cover equals the aggregation run's MEDIAN maps; SDs equal the run's SD; layer sums are 100; water/out-of-projection codes are unchanged.
   - Expected time: about 40 minutes for predictions plus 10 minutes for checks.
5. **Report:** `Reports/Delivery_SD_RF_Median_Results.md`, and the README is updated.

## Expected cost

About 2 hours in total: step 0 takes 10 minutes; code, tests and docs about 1 hour; step 2 about 10 minutes; step 4 about 50 minutes once the non-vegetated choice is made.

## Not in scope

- The non-vegetated method decision itself. The code supports all three choices.
- Rounding instead of truncating the value cast.
- The `Quality`/`Cloud` layers and the HDF product file (to-do item 9, `Quality_Cloud_Layers_and_HDF_Output_Plan.md`). uint16 storage for the SD comes with them.
- The bare post-processing rules from the C code (deferred).
- Retraining. The fitted models from `qaC_full63_retrain_20260929` (tree) and `nonveg_bit15_models_20261001` (non-vegetated) are used as they are. Production fitting follows the documented commands.
