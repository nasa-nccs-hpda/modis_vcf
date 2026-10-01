# Plan: single-tile (h12v09) test bed for cloud-fallback scenarios

**Written 2026-09-30. Status: awaiting approval. Nothing run.**
Background: `Cloudy_Fallback_Diagnosis_Results.md` (stage A and its addendum). This replaces steps 5 and 6 of `Cloudy_Fallback_Diagnosis_Plan.md` for now (project lead, 2026-09-30: no literature search yet).

## Why a single tile

A scenario that changes compositing needs the metrics rebuilt. For the 63-tile model that means hours; for h12v09 alone it takes minutes. Model fits on one tile take seconds (19,863 training rows). The single-tile model is used only to **rank scenarios**. The best one or two are then confirmed with the 63-tile RF (phase 3, a separate approval).

## Setup

- **Data:** h12v09 rows of `qaC_full63_retrain_20260929/Python/training/h12v09-2019-pcttree-training+obsForRF.parq` (bit-15 build).
- **Split:** the same 20% pixel hold-out rule as the 63-tile run (`split_keyed.py`: hash of tid-year, x, y mod 5 == 0). This gives 19,863 training and 4,997 test pixels. The test pixels are the same ones the 63-tile RF was scored on, so the two compare directly. There is no whole-tile hold-out.
- **Model:** RF exactly as in the 63-tile run (`fitAltModel.py --algo rf`, 30 trees, sklearn defaults, same 64 predictors unless a scenario changes them). Every fit is repeated with seeds 0, 1 and 2 to measure seed noise.
- **Full-tile prediction** from the existing metric rasters (`aerosol_qa_variants_20260929/Python/metrics_C/h12v09/.../3-Metrics`, 21 files), or from rebuilt ones for S3.
- **References:**
  - **S0**, the single-tile RF with nothing changed. Every scenario is compared with S0.
  - **G**, the existing 63-tile RF (bit 15), on the same test pixels and the same map.

## Scoring (every scenario, against S0)

1. **Held-out labels (4,997 px):** MAE and bias overall, and by fallback count (0-2, 3-4, 5+; the 5+ group has about 660 test pixels). Each difference from S0 gets a paired bootstrap 95% CI (1,000 resamples).
2. **Map, dense forest (historical >= 60):**
   - mean bias against the historical product, and the share <= historical - 30;
   - both at the existing drop pixels (103,011) and control pixels (200,000), and by fallback count.
   
   The historical product is not independent here (it has its own fallback), so this is a supporting check only.
3. **Decision rule:** a scenario counts as a gain only if the label MAE improves in the 5+ or 3-4 group by more than the seed noise and the bootstrap CI excludes 0, and the 0-2 group gets no worse by more than 1 point. Label MAE differences of about 1 point or less overall count as a tie.

The labels sit about one bin higher than both maps (73% of h12v09 labels are 80-99). Every model trained on them will predict higher than the historical product. Only differences between scenarios are read, not levels against the historical product.

## Phase 1: cheap scenarios (no metric rebuild)

| ID | Scenario | What changes |
|---|---|---|
| S0 | Single-tile baseline | none (64 predictors) |
| S1 | Fallback count as a predictor | + 1 column: the sum of the 8 `*-Band_1-qa.npy` at the pixel (already on disk) |
| S1b | Per-period fallback flags | + 8 binary columns (`outQa` for each period) |
| S2 | Drop contaminated training rows | exclude training rows with 6 or more fallback periods (about 1,150 of 19,863); test set unchanged |
| S2b | Down-weight them | `sample_weight` = share of clean periods (floor 0.1) |
| S4 | Ranked predictors only | the 25 CGreen, CTherm and CSortedRank predictors of the 64 (their top ranks avoid cloudy periods) |
| S4b | Ranked predictors, wider | all ranked columns available in the 116-column table |

S2 and S2b test the stage A hypothesis: low-label pixels are the most fallback-heavy, so the training data teaches "cloudy signature = low tree cover". The test set is not filtered, so removing rows cannot win by leaving out hard test cases.

## Phase 2: compositing scenarios (h12v09 metrics rebuilt, only if phase 1 leaves the problem unsolved)

| ID | Scenario | What changes |
|---|---|---|
| S3 | Fill fallback periods | per pixel and band, replace a fallback period's composite with the median of the pixel's clean periods (keep it as is if there are none), then recompute all metrics |
| S5 | 2-input periods | 12 periods of 32 days; unsorted predictors from 12 periods; the C-mirror statistics redefined on ranks. Largest change; the predictor set must be reselected, so last |

S3 is done by a script that reads the existing `2-Composites` and `qa.npy` and writes filled composites into a new metrics directory. It then runs the unchanged metric code (`Metrics.py`, `CMirrorMetrics.py`) on them. No pipeline code is edited. Training rows are re-extracted at the label pixels. Estimated time: 30-60 minutes for one tile.

## Phase 3: confirmation (separate approval)

The best one or two scenarios go through the 63-tile setup. RF refits take 10-30 minutes. S1 and S2 need no metric rebuild (the `qa.npy` files exist for all 63 tiles). S3 would need a 63-tile metric rebuild (hours). Score them on the held-out labels for all tiles and by fallback count, on the cloudy tiles (h12v09, h11v08, h19v09, h19v08, h26v06), and on the stable set and NEON, as in earlier comparisons.

## Order, cost, conditions

1. **Cheap check first:** S0 with seeds 0-2. Compare with G on the same 4,997 test pixels. A single-tile model should be at least as good on its own tile (G's MAE there is 7.1). If S0 is clearly worse, report that and stop before the scenarios.
2. Phase 1 (S1-S4b): about 1 hour including full-tile predictions (about 5-10 minutes each, 30 trees, 23M pixels).
3. Report, then phase 2 only if phase 1 leaves the drop unsolved and the project lead agrees.

- **Where:** ngaproc201 or 202, fully detached, with status and done markers.
- **Output:** a new run root `FullRuns/single_tile_h12v09_<date>/`. The scripts refuse to reuse it.
- **Code:** new scripts in the run root only. No product code is edited; nothing in rlgill's repositories or on the pushed branch. The existing training table, metrics and models are read-only.
- **Results:** `Reports/Single_Tile_h12v09_Scenario_Results.md`, with the README updated.

## Decisions for the project lead

- Go-ahead for the cheap check and phase 1.
- Whether the S2 threshold (6 or more fallback periods) is acceptable, or should be 5 or more.

## Amendment (project lead, 2026-09-30 ~16:30): S1 at 63-tile scale first, instead of the single-tile model

Direction: compute the fallback count for all 63 tiles, add it as a column, train a new 63-tile RF, and predict h12v09 as the test case. Also add a **cloudy count**, so that fallback caused by low sun or snow (Greenland, winter tiles) is not confused with cloud. Run root: `FullRuns/rf_fallback_counts_20260930/`.

1. **Counts (all 63 tiles):**
   - `fallback_count` = the sum of the 8 stored `*-Band_1-qa.npy` (1 = no input passed QA).
   - `cloudy_count` = the number of periods where no input has MOD44CQ `state` cloud bits (0-1) = 0 (clear). Solar zenith is ignored; a missing input counts as not clear. Inputs are grouped in threes as in `CompositeDayFile` (2019065 ... 2020049).
   - Check: cloudy <= fallback on every pixel (a cloudy period cannot pass QA).
2. **Split:** rebuilt from the per-tile C training files with the same row order and hash rule as `split_keyed.py`, plus the two columns. Check: the 116 existing columns are identical to the current `training_split/C` (train and test).
3. **Fits** (`fitAltModel.py --algo rf`, 30 trees, `max_features` sqrt as in the qaC run, seed 0):
   - **N**, the 64 predictors plus the 2 counts.
   - **G1**, the unchanged 64 predictors with seed 1, to measure seed noise against the existing model **G** (seed 0).
4. **Evaluation:**
   - Held-out 20% labels for all 63 tiles: per tile, pooled, and by fallback and cloudy count. Every tile where N - G differs by more than 1 point MAE is flagged, with the high-latitude and snowy tiles checked explicitly.
   - h12v09 map for N and G1, predicted from the 3-Metrics rasters, with dense-forest bias against the historical product at the drop and control pixels and by count.
   - The raster feature assembly is checked by re-predicting G on h12v09 and comparing with its saved map.

Fully detached on ngaproc201 and 202, with status markers. No product code is edited.
