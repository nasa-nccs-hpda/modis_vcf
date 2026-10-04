# Non-vegetated (bare) layer: baseline, RF and XGBoost on the bit-15 metrics (plan)

**Written 2026-10-01. Status: plan only, awaiting approval. Nothing has run.**

## Why

The non-vegetated layer (`Percent_NonVegetated`, the C code's BARE models) is the last layer still on the Python baseline model.
That model (`full63_c_vs_py_20260919-0806/Python/models/pctbare.bin`) was trained on metrics from before the bit-15 aerosol screen. The bit-15 run then applied it to bit-15 metrics without retraining it (`qaC_full63_retrain_20260929`).
Item 3 of the recommendation in `C_to_Python_Conversion_Assessment.md` is to replace it. This plan retrains the baseline on bit-15 metrics and tries RF and XGBoost, using the same approach that settled tree cover.

## What already exists (checked 2026-10-01)

No new metrics are needed.
- **Bit-15 metrics for all 63 tiles:** `qaC_full63_retrain_20260929/Python/metrics_all/`.
  - 60 tiles are in `metrics_C/`.
  - h09v05, h12v09 and h16v01 are linked in from `aerosol_qa_variants_20260929`.
  - All five summary families the bare predictors need (CUnsMean16, CUnsMean79, CSortedRank6, CSortedRank7, CGreenMean16) and `UnsortedMonthlyBands` are present on h09v04 and h16v01.
- **Bare labels, 20 tiles:** `qaC_full63_retrain_20260929/Python/training_labels/VCF_training_pctbare_*.7.0.0.out`. All 20 tiles are among the 63.
- **The 21 predictors:** `full63_c_vs_py_20260919-0806/manifests/predictors_pctbare.txt`. This is the C code's `attributes_bare[]` (`NonVegetated_Model_Predictors_C.md`).
- **Older bare training tables** (pre-bit-15): `full63_c_vs_py_20260919-0806/Python/training/*-pctbare-*.parq`. They are used only to cross-check row counts.
- **The current bare map with bit-15 tree cover:** `qaC_full63_retrain_20260929/Python/predictions_C/rf/`. Here the old `pctbare.bin` was applied to the bit-15 metrics. This is the "current" reference.
- **The C code output:** `full63_c_vs_py_20260919-0806/C/`. This is the static 30 M5P bare models, run with the bits 7-8 shift.
- **The four-way comparison code:** `hist4way_c6_c61_20261001/code/` (masks, `region_bins.py`, cover bins).
- **Fit code:**
  - `rfFitFromPge61.py` (baseline);
  - `regtree_alt_models_20260927/code/validation/fitAltModel.py` (RF and XGBoost). It names its outputs after `--trainingType`, so `pctbare` works unchanged.

## Compute

- **ngaproc201:** table build, split, fits and held-out evaluation.
- **ngaproc202:** predictions, and fits if they run in parallel.
- Both nodes were idle on 2026-10-01 (load ~0, ~217 GB free; 24 CPUs each).
- ssh uses `-o CheckHostIP=no`.
- The chain script runs `module load singularity/4.3.1` before calling singularity.
- Container: `modis-vcf-latest`, with the bit-15 code copy `aerosol_qa_variants_20260929/code_C/python_code` bound read-only. This is the same setup as the bit-15 run.

## Run root

`FullRuns/nonveg_bit15_models_20261001/` (new). Nothing in earlier run roots is written to.
- **Subfolders:** `manifests/`, `scripts/`, `logs/`, `status/`, `Python/{training,training_split,models/{baseline,rf,xgboost_tunedB},eval,predictions/{baseline,rf,xgboost_tunedB}}`, `Compare/`.
- **All output folders are created before launch.** In the bit-15 run, predictions stalled for 5 hours because the output folders did not exist.
- **Launch:** fully detached (`setsid nohup`, with `< /dev/null`). Each step writes `.done` or `.failed` markers in `status/`, and any `.failed` stops the chain at its next step.

## Steps

### 0. Cheap check (one tile, about 10 minutes, then stop for review)

- **Tile:** h20v06. It has bare labels, is outside the high latitudes and isn't flagged for C6.1.
- **Run:** `buildTraining.py --trainingType pctbare` on that tile against the bit-15 metrics. Use the same arguments as stage 2 of the bit-15 run, with `-m` set to the metric families behind the 21 predictors.
- **Pass if:**
  - all 21 predictor columns are present, and no column is more than 1% NaN or fill;
  - the label range is 0-100;
  - the row count is within 5% of the 2026-09-19 `h20v06-2019-pctbare` table.
- **Also record:** the elapsed time (used to size step 1), and the label histogram in the four bins 0-20, 21-40, 41-60 and 61+.
- **Stop here and report before step 1.**

### 1. Bare training tables (20 tiles, ngaproc201)

- 4 tiles at a time. The bit-15 run took about 6 minutes per tile on the forest nodes, so this should take about 30-40 minutes.
- Output: `Python/training/<tile>-2019-pctbare-training+obsForRF.parq`.
- Per-tile check: a non-empty table, and a row count within 5% of the 2026-09-19 table. Anything larger is written to `.failed` and stops the chain.

### 2. 20% held-out pixel split

- **Script:** `split_keyed.py`, copied into `scripts/` and changed to take one variant and `LABEL = 'pctbare'`.
- **Rule:** the same as tree cover, a pixel is held out when the hash of its key (`tid-year`, `x`, `y`) mod 5 is 0. Bare held-out pixels are therefore the same pixels as tree cover held-out pixels wherever the two label sets overlap.
- **Tiles:** all 20 stay in training. No tile is held out.
- **Output:** `Python/training_split/{xTrain,yTrain,xTest,yTest,keysTest}.parq` and `split_summary.json`.

### 3. Fits (three models)

All three use `--trainingType pctbare` and `--predictorFile predictors_pctbare.txt`, with `randomState 0`.

| Model | Code | Settings (same as tree cover) |
|---|---|---|
| baseline | `rfFitFromPge61.py` | BaggingRegressor(DecisionTree), 30 estimators |
| rf | `fitAltModel.py --algo rf` | 30 trees, `max_features='sqrt'` |
| xgboost_tunedB | `fitAltModel.py --algo xgboost` | 2000 rounds, depth 6, learning rate 0.02, subsample 0.8, colsample 0.8 |

- **Size and time:** about 7 M training rows (the 2026-09-19 table had 8.6 M rows before the split) with 21 predictors. Tree cover fits took 8-30 minutes on 9 M rows and 64 predictors, so bare should be faster.
- **Scheduling:** the three fits run at the same time on ngaproc201 with `nJobs 8` each, or are split across both nodes.
- **Pass if:** `pctbare.bin` and `pctbareTopN.bin` are non-empty, and `pctbareTopN.bin` holds exactly the 21 names in the C code's order.

### 4. Held-out evaluation (20 tiles, ngaproc201)

- **What is scored:**
  - the three new models on `xTest`/`yTest`;
  - the current `pctbare.bin` on the same rows;
  - the C code bare map, read at the held-out pixel locations.
- **Measures:** MAE, bias (mean of per-pixel differences), RMSE and R².
- **Breakdowns:**
  - pooled and per tile;
  - by the label's bins 0-20, 21-40, 41-60 and 61+;
  - by continent and latitude band (`region_bins.py`; no land cover).
- **Model differences:** 95% block-bootstrap intervals, by tile, for each pair (new vs current, RF vs baseline, XGBoost vs RF, each vs the C code).
- **Caveat:** the C code's static bare models may have been trained on labels that overlap these legacy labels. Its held-out score is reported but may be optimistic.

### 5. Predictions (63 tiles, ngaproc202, starting once the first fit is done)

- **Script:** `vcfPredictQueue.py`, run once per new bare model.
  - `--tc`/`--tcn`: the bit-15 RF tree cover model (`qaC_full63_retrain_20260929/Python/models/C/rf`), the same for all three runs, so only the bare model differs.
  - `--nv`/`--nvn`: the new bare model.
- **Time:** in the bit-15 run the predictions took about 1 hour per model across nodes. On one ngaproc node with `nJobs 20`, expect about 1-1.5 hours per model, about 3-4 hours for all three. Fits finishing early can be split across both nodes.
- **Layer-sum rule:** the code clamps predictions to 0-100 and lowers non-vegetated so that tree cover plus non-vegetated is at most 100. Non-tree vegetation therefore changes with each bare model and is reported as well.

### 6. Map comparison (63 tiles)

- **Code:** a copy of the four-way comparison, run on the non-vegetated layer.
- **Products compared:** each new bare map and the current one, against C6, C6.1 and the C code.
- **Masks:** the same as the four-way run. The MOD44W out-of-projection area and the value-18 area of the historical product are left out.
- **C6.1 flags:** C6.1 sets non-vegetated to **100** on the 9 flagged tiles. Those tiles are left out of every pooled C6.1 number. 6 of the 20 bare-label tiles are among them (h11v02, h12v01, h12v02, h13v01, h16v01, h21v01), so the 14 others carry the pooled label-side comparison against C6.1.
- **Breakdowns:** per tile; by the reference's non-vegetated value in the bins 0-20, 21-40, 41-60 and 61+; by continent and latitude band.
- **Also report:**
  - the share of pixels where the layer-sum rule lowered non-vegetated;
  - the mean change in non-tree vegetation against the current map.

### 7. Side check: NDVI agreement (cheap, runs alongside, report only)

7 of the 21 bare predictors are NDVI-based, and an earlier check found Python's unsorted NDVI within 1 of the C code's on only 66-83% of pixels.
- **Check:** compare `UnsortedMonthlyBands` NDVI with the C code's metrics on 2 bare-label tiles (h20v06 and one high-latitude tile), for each period.
- **Output:** report the agreement only. Nothing is changed.

## How the result is judged

The project rule, unchanged:
- A model difference counts only if it has the same sign across independent references and its block-bootstrap interval clears zero.
- About 1 point of non-vegetated MAE or less is a tie.

**The references and their limits:**
- **The held-out labels** are the closest thing to an independent reference. They cover only 20 tiles.
- **C6, C6.1 and the C code** all come from the same static M5P bare models, so they are one lineage, not three references. They favour whatever resembles M5P. A Python model can tie with or lose to them without being worse.
- **Lidar:** no lidar reference for non-vegetated is planned. NEON CHM could give a bare-ground fraction, but that would be a separate plan.

**Expected outcomes:**
- **If RF and XGBoost tie,** RF is preferred for the same reason as tree cover: per-tree predictions give the SD layer (`Percent_NonVegetated_SD`).
- **If RF ties or beats the retrained baseline on held-out labels without losing on the map references,** the recommendation is RF for both layers.

## Out of scope

- The SD layer (a separate item).
- Changing the predictor list or the learner settings.
- The second-year and non-training-tile test.
- Any code change in `Git_delivery/modis_vcf`. If RF is chosen, wiring it in as the default bare model gets its own plan.

## Outputs

- **Results report:** `Reports/NonVegetated_Bit15_Models_Results.md`.
- **Tables:** in `eval/` and `Compare/`.
- **README:** `README_START_HERE.md` is updated when the run starts and when it finishes.

## Estimated wall time

About 6-7 hours after the cheap check is approved: tables about 40 minutes, split about 10 minutes, fits about 30 minutes, evaluation about 20 minutes, predictions about 3-4 hours, comparison about 1 hour.
