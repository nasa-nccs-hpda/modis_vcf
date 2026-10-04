# MODIS-VCF Python product: versions, experiments and validation (work map)

**Written 2026-10-03. Naming set by the project lead on 2026-10-03.** This is the reference for what each name means. Other reports
should use these names.

| Name | Meaning |
|---|---|
| **v1.0.0** | The current product: tree cover, non-vegetated, non-tree vegetation and the two SD layers, as made by delivery-branch commit `16d6474` with the models below. Called **BASE** in the cloudy-pixel reports. |
| **v1.0.1** | **EXPERIMENTAL, not adopted** (named by the project lead on 2026-10-03 for reference). Tree cover = v1.0.0 blended with the retrained V4R model under the main cloud, cover and snow gates (`Gated_V4r_Blend_63Tile_Results.md`), 63 training tiles, 2019. Other layers = v1.0.0's, not recomputed. See section 1b. |
| **v0.0.0** | The bagged-tree version: `BaggingRegressor(DecisionTreeRegressor)`, 30 members, MEAN of the members, on the same bit-15 inputs as v1.0.0. Called **baseline** or **BASELINE** in earlier reports. **In the project lead's view, the closest match to the original intent of the C code** (see section 2). |
| **v0.0.0-pre.2** ("bits 7-8") | Before v0.0.0: our C-parity fixes and bagged trees (MEAN of 30), but with the bits 7-8 aerosol test in place of bit 15. Run-folder copies only (`FullRuns/full63_c_vs_py_20260919-0806`), no git tag. |
| **v0.0.0-pre.1** ("upstream") | rlgill's Python code as it stood when this work started (upstream `develop`, `b83e302`): incomplete, with GPU cuML Random Forest, adjacency bit 13, zenith cutoff 70 and the bits 7-8 aerosol test. |
| **Experiments** | Attempts to improve v1.0.0. Each is v1.0.0 with one change. So far: the cloudy-pixel work and the Monte Carlo rerun. Nothing from them is adopted. |
| **Validation / verification** | Comparisons with independent data: NGA WorldView-2, AlphaEarth, and lidar (NEON, G-LiHT). |
| **Assessment** | Held-out labels, map agreement with C6 / C6.1 / the C code, the stable pixel set and C-vs-Python parity. These are how model work is scored, and they stay with the model work. They are not validation. |

**Pre-release labels (project lead, 2026-10-03).** The two `-pre.N` labels rank below v0.0.0, in time order. They are used only
where reports still cite results from those states:
- **pre.1:** the early 2-tile C-vs-Python comparisons and the six code differences (`C_code/pge61/MOD_PR44B_Metrics/C_vs_Python_Code_Differences.md`).
- **pre.2:**
  - the 2026-09-19 63-tile run (`FullRun_Results_C_vs_Python_63tiles.md`);
  - feature importance and the 2026-09-20 Monte Carlo;
  - tree cover against the historical MOD44B (2026-09-20);
  - the "B" side of the aerosol QA test.

Pilots and other one-off states are described as variants of pre.2, not given numbers. RF and XGBoost fits on the bits 7-8 inputs are
experiments on pre.2, not versions. Results from pre.1 and pre.2 are not results for v0.0.0 or v1.0.0. v0.0.0 and pre.2 differ only
in the QA test (bit 15 against bits 7-8), which is the main reason September numbers do not match later ones.

---

## 1. v1.0.0: the current product

**Code:** delivery branch `mcarrol2/c-python-parity-fixes` in `AI_code_sync_09152026/Git_delivery/modis_vcf`, commit `16d6474`
(2026-10-02), local tag `v1.0.0`.

**Models:**

| Layer | Model file | Fitted in |
|---|---|---|
| Tree cover | `FullRuns/qaC_full63_retrain_20260929/Python/models/C/rf/pcttree.bin` | 63-tile bit-15 retrain, 2026-09-29 |
| Non-vegetated | `FullRuns/nonveg_bit15_models_20261001/Python/models/rf/pctbare.bin` | non-vegetated model run, 2026-10-01 |

**Maps (2019):**
- 62 of the 63 training tiles: `FullRuns/rounding_vs_truncation_20261001` (ROUND).
- h16v01 and the 6 Greenland tiles: `FullRuns/final_round_v00_20261002/out/final`.
- That run used `13773db` plus uncommitted changes. They are the same code changes as `16d6474`, which adds only a unit test file.
- **Check:** the Monte Carlo rerun rebuilt the h12v09 map with `16d6474` and these models, and it equals the stored map in all three
  layers (`mc_rerun_20261002/logs/pipeline_check.log`).

### Rules, in processing order

**1. Inputs and year**
- MOD44C 16-day files:
  - MOD44CQ (250 m): Bands 1-2, `state` and solar zenith;
  - MOD44CH (500 m, read onto the 250 m grid): Bands 3-7 and 31.
- The 2019 processing year is 2019 day 65 to 2020 day 49, in 16-day steps.

**2. Per-day quality test.** A pixel on a 16-day date is good only if all of these hold:
- cloud bits 0-1 = clear;
- shadow bit 2 off;
- adjacency bit 6 off;
- **bit 15 set** (PGE72's overall QC flag);
- 0 < solar zenith < 65.

**3. Compositing**
- **Periods:** 8 per year, each made of 3 consecutive 16-day dates (48 days). They start on 2019 days 065, 113, 161, 209, 257, 305 and
  353, and 2020 day 033.
- **Value:** the mean of the period's good days.
- **Fallback rule:** if no day is good, the value is the mean of all its days, good or not. The number of such periods per pixel
  is its **fallback count** (0-8).
- **Missing day files:** filled with 10000 and counted as not good.
- **Band 31 clamp:** where zenith is over 70, the value is set to 16000. Bands 1-7 get no clamp, matching how the C code actually
  behaves.

**4. NDVI**
- (B2 − B1)/(B2 + B1) × 1000, from the composites; −1001 where B1 + B2 = 0.
- **Greenland rule:** on h15v02, h16v00, h16v01, h16v02, h17v00 and h17v01, NDVI is set to 0 in any period with no day under the
  zenith cutoff. Iceland (h17v02) is excluded.

**5. Predictors**
- These are the C code's attributes, mirrored in Python with the C code's integer arithmetic.
- **Tree cover:** 64; **non-vegetated:** 21.
- They come from 9 bands (B1-7, NDVI, B31) × 8 periods:
  - the raw period values;
  - means and amplitudes of the periods in date order (`CUns*`);
  - the same, ordered by NDVI (`CGreen*`) or by B31 (`CTherm*`);
  - each band sorted on its own (`CSortedRank*`).
- Definitions: `model/CAttributes.py`, `model/CMirrorMetrics.py`.

**6. Training data**
- **Tree labels:** `Mark_training/VCF_training_adjusted/tile_adjustment/v5.0.3samp`, 63 tiles.
- **Non-vegetated labels:** the legacy bare labels from `Mark_training`, 20 tiles.
- Label 255 means no label, and those pixels are dropped.
- **Split:** a pixel is held out when a fixed hash of (tile-year, x, y) is 0 mod 5. That is 20% of pixels, chosen by pixel, not by
  tile. Held-out bare and tree pixels coincide where the two label sets overlap.
- **The product models are fitted on the other 80% only.**

**7. Models**
- Random Forest (scikit-learn): 30 trees, `max_features='sqrt'`, `random_state=0`, other settings at their defaults
  (`fitAltModel.py --algo rf`).
- Both layers use the same settings.

**8. Prediction**
- Each tree's prediction is clipped to 0-100.
- **Tree cover = MEDIAN of the 30 trees; non-vegetated = MEAN.**
- Values are **rounded** (halves up, floor(x + 0.5), as the C code does).
- SD layers: round(100 × SD of the 30 trees), taken before the sum cap.

**9. Masks and layer rules**
- From MOD44W (`/css/modis/Collection6.1/L3/MOD44W-LandWaterMask`):
  - water (1) → **200** in all three layers;
  - outside the projection (250) → **250** in all three layers;
  - SD layers are 0 on both.
- **Sum cap (land only):** non-vegetated is reduced to at most 100 − tree cover. Tree cover is never lowered.
- **High-latitude rule:** on v00 tiles, land is set to tree cover 0, non-vegetated 100.
- **Non-tree vegetation** = 100 − tree − non-vegetated on land, with 200 and 250 kept.
- Output: Int16 GeoTIFFs, one per layer.

### Path from v0.0.0 to v1.0.0 (the changes, in the order adopted)

| Date | Change | Report |
|---|---|---|
| 2026-09-30 | Clip to 0-100; sum cap lowers non-vegetated, never tree cover; water and out-of-projection kept in non-tree vegetation (commit `2a37f9e`) | `NonTreeVeg_Water_OutOfProj_Code_Propagation_Plan.md` |
| 2026-09-30 / 10-01 | Tree cover: Random Forest instead of bagging. RF and XGBoost were treated as equal in accuracy; RF was chosen because it gives the SD layer | `Regression_Tree_Methods_RF_XGBoost_Results.md`, `Model_vs_Historical_DeepDive_20260929.md` |
| 2026-10-01 | Non-vegetated: Random Forest instead of bagging | `NonVegetated_Bit15_Models_Results.md` |
| 2026-10-01 | Tree cover = MEDIAN of the trees; non-vegetated = MEAN (provisional) | `RF_Aggregation_Maps_63Tile_Results.md`, `Delivery_SD_RF_Median_Results.md` |
| 2026-10-01 | SD layers | `SD_Layer_Results.md`, `Delivery_SD_RF_Median_Results.md` |
| 2026-10-02 | Zenith-only NDVI on the 6 Greenland tiles (commit `13773db`) | `Greenland_Iceland_Scoped_Fix_v2_Results.md` |
| 2026-10-02 | Rounding instead of truncation; high-latitude rule on v00 tiles (commit `16d6474`) | `Rounding_vs_Truncation_Results.md`, `Bare_C_Regions_Investigation_Results.md` |

Unchanged between the two versions: inputs, QA, compositing, predictors, labels, the split, and the water and out-of-projection codes
in the tree and non-vegetated layers.

### Open decisions about v1.0.0 itself (not experiments)

- Non-vegetated MEAN is provisional. MEDIAN is better on held-out labels but worse against C6 and C6.1.
- The rest of the C code's high-latitude bare mask, the Sahara stretch and the v6.0.3-vs-C6 conflict
  (`Bare_C_Regions_Investigation_Results.md`).
- The `Quality` and `Cloud` layers and a single HDF output file (`Quality_Cloud_Layers_and_HDF_Output_Plan.md`). The composite's
  per-period cloud output (`outCloud`) is computed but unused, and it is known to be wrong.
- More years (`Additional_Years_Plan.md`).

---

## 1b. v1.0.1: experimental version (named 2026-10-03)

**Status: EXPERIMENTAL, not adopted. v1.0.0 remains the delivered product.** It was named so it can be referred to and reproduced.
Recipe, paths and checksums: `FullRuns/gated_v4r_63_20261003/manifests/VERSION_v1.0.1.md`.

| | v1.0.1 |
|---|---|
| **Tree cover** | v1.0.0 blended with **V4R** (the model retrained on V4r metrics, applied to V4r composites): value = floor((1 − w) × v1.0.0 + w × V4R + 0.5) |
| **w, cloud** | (snow-free, zenith-gated V4r replaced periods) / 8 where that count is ≥ 5, else 0. A period counts when the V4r rule replaced a cloud-caused fallback, some input had 0 < zenith < 65, and the V4r pick is not snow (NDSI ≥ 0.4 and NIR ≥ 0.11 means snow) |
| **w, cover** | ramp on v1.0.0: 0 at 20, 1 at 50 |
| **Other layers** | v1.0.0's non-vegetated, non-tree vegetation and SD layers, **not recomputed**. Where tree cover changed, the sum cap and non-tree vegetation have not been re-derived, so the layers need not sum to 100 there |
| **Coverage** | the 63 training tiles, 2019; maps in `FullRuns/gated_v4r_63_20261003/out/maps/v1.0.1_tree_cover/` |
| **Code** | that run folder only: V4r compositing, counts and gating are not in the product code. No delivery-branch commit or tag |
| **What changes** | about 1% of land on the 63 tiles (cloudy pixels with 5+ periods and v1.0.0 > 20); 9 tiles change measurably |
| **Results** | `Gated_V4r_Blend_63Tile_Results.md`: h12v09 drop pixels +8.6, C6 gap halved, C6.1 closer, C code further; held-out dense labels +0.3 (n.s.); open land and clean dense forest unchanged; figures in `figures_gated_v4r_63_20261003/` |

**Relation to v1.0.0:** identical everywhere except the gated cloudy pixels. Every input of v1.0.0 (QA, compositing, predictors, model,
split, rules) is unchanged.

---

## 2. v0.0.0: the bagged-tree version

**Why v0.0.0 matters (project lead, 2026-10-03):**
- It is the version **closest to the original intent of the C code.**
- rlgill's Python code (v0.0.0-pre.1) was incomplete when this work started. v0.0.0 is the first Python version that follows the C
  design end to end.
- Every later change is a deliberate departure from the C code, made to improve the product.

**Against the C code (MOD_PR44B), from `Ensemble_Prediction_C_vs_Python.md` and `C_vs_Python_Code_Differences.md`:**

| | C code | v0.0.0 | v1.0.0 |
|---|---|---|---|
| Predictors | 64 tree, 21 bare (`attributes[]`, `attributes_bare[]`) | the same, as Python mirrors with the C code's integer arithmetic | the same |
| Ensemble | 30 members, each clamped to 0-100, then the **mean** | 30 members, **mean** | 30 trees; tree cover **median**, non-vegetated mean |
| Member model | Weka M5P model trees (linear models at the leaves, at least 500 rows per leaf), each trained on its own sample file | decision trees on bootstrap resamples of one pool | Random Forest trees (`max_features='sqrt'`) |
| Non-vegetated | 30 **static** M5P models (`BARE/`) | retrained bagged trees | retrained Random Forest |
| Float to integer | round half up | **truncate** (a departure, about 0.5 low on average) | round half up |
| Zenith cutoff, adjacency | 65, bit 6 (as decided 2026-09-15) | the same | the same |
| Aerosol / QA test | bits 7-8 aerosol test (`fix_bands.c` shift corrected from `>> 13` to `>> 7` on 2026-09-15) | **bit 15** (PGE72's overall QC flag), because bits 7-8 overlap other fields and rejected 29-50% of good pixels | bit 15 |
| SD layer | per-pixel SD across the 30 members | none | yes |

So v0.0.0 keeps the C code's structure (predictors, a 30-member mean ensemble, compositing and metrics), and departs from it in:
- the member model;
- the truncation;
- the retrained bare model;
- the bit-15 QA fix to the input screening.


| | v0.0.0 |
|---|---|
| Inputs, QA, compositing, NDVI, predictors, labels, split | **Same as v1.0.0** (bit-15 inputs), except that it has no Greenland NDVI rule |
| Model | `BaggingRegressor(DecisionTreeRegressor())`, 30 members, bootstrap rows, all predictors at every split, `random_state=0` (`rfFitFromPge61.py`) |
| Combination | MEAN of the 30 members |
| Value cast | truncated (`astype(int16)`) |
| SD layers | none |
| Clip / sum | no clip. Non-tree vegetation = max(0, 100 − tree − non-vegetated), and non-vegetated is not lowered. On water and out-of-projection pixels this gives 0 in non-tree vegetation, not 200/250 |
| High-latitude rule | none |
| Tree model | `FullRuns/qaC_full63_retrain_20260929/Python/models/C/baseline/pcttree.bin` (2026-09-29) |
| Non-vegetated model | `FullRuns/nonveg_bit15_models_20261001/Python/models/baseline/pctbare.bin` (2026-10-01) |
| Code | the run-root copy `FullRuns/aerosol_qa_variants_20260929/code_C/python_code` and the container's `rfFitFromPge61.py`. No branch commit or tag holds v0.0.0 exactly |

**Note:** the two v0.0.0 models were fitted in different runs, and **no single map set contains both bagged layers.**
- The tree maps are `qaC_full63_retrain_20260929/Python/predictions_C/baseline`.
- `nonveg_bit15_models_20261001/Python/predictions/baseline` has the bagged non-vegetated layer, paired with the RF tree layer.

**v0.0.0 against v1.0.0, from the reports.** These intermediate comparisons are model against model, before MEDIAN and rounding:
- **Tree, against the historical MOD44B (63 tiles):** pooled MAE: bagging 4.87, RF 4.20 (`Aerosol_QA_C_Full63_Retrain_Results.md`).
- **Tree, held-out labels:** bagging 7.25, RF (MEAN) 7.32. That is a tie. Best tile counts: bagging 48, RF 15, XGBoost 0.
- **Non-vegetated, held-out labels:** bagging 4.18, RF 4.20, also a tie. On maps, RF was 0.95 MAE closer to C6, in the same direction
  against C6.1 and the C code (`NonVegetated_Bit15_Models_Results.md`).

---

## 3. Experiments to improve v1.0.0

Each candidate is v1.0.0 with one change. Assessment uses the held-out labels, the maps against C6 / C6.1 / the C code, and the
stable set where it has pixels. The seed refits of v1.0.0 (seeds 1 and 2) are the run-to-run yardstick. About 1 point MAE or less is
a tie, and a change is acted on only when the references agree.

### 3a. Cloudy pixels (fallback composites)

**Problem:** dense forest with many fallback periods drops sharply. For example, on h12v09 v1.0.0 goes from +3.6 against C6 at 4
fallback periods to −13.1 at 8.

| Item | What it is | Status | Report |
|---|---|---|---|
| Stage A diagnosis | Why the models drop on fallback composites, and how widespread it is | done 2026-09-30 | `Cloudy_Fallback_Diagnosis_Plan.md`, `_Results.md` |
| V4r compositing, h12v09 | Darkest-red guarded pick in all-cloudy periods, first without retraining | done 2026-09-30. Better against C6, worse against the labels, so not adopted | `V4r_Darkest_Red_h12v09_Results.md` |
| **v1.0.0 + counts (BASE-C)** | Fallback and cloudy counts added as predictors (model N, fitted 2026-09-30, scored as MEDIAN and rounded) | run 2026-10-02: −0.03 held-out MAE, better on 51 of 63 tiles (a tie) | `RF_Fallback_Counts_Results.md`, `Cloudy_Pixels_Continuation_Results.md` |
| **v1.0.0 + weights (BASE-W)** | Training rows weighted by share of clear periods, (8 − fallback)/8, floor 0.1 | run 2026-10-02: +0.004, no gain | `Cloudy_Pixels_Continuation_Results.md` |
| **v1.0.0 + V4r composites (V4R)** | V4r compositing on 63 tiles, retrained | run 2026-10-02: +0.13 overall; +0.60 on dense 5+ fallback pixels, but better on 17 of 22 tiles; on h12v09 drop pixels, −18.9 → −2.7 against C6; sharp decreases elsewhere still unexplained | `Cloudy_Pixels_Continuation_Results.md` |
| Single-tile test bed | h12v09 scenario plan | superseded by the continuation plan | `Single_Tile_h12v09_Scenario_Plan.md` |
| **v1.0.0 on V4r composites; cloud-gated blends** | One model on two composite strategies (V1-V4C, BLEND-1) and v1.0.0 + V4R (BLEND-2), gated by replaced-period count | run 2026-10-03; reassessed (zenith-gated count, labels by bin): blends raise cloudy pixels +4 to +5 at 5+ periods; a cover gate removes the open-land raise; correction: the pooled label result was dominated by snowy h10v04 — without it, V4r raises under-predicted dense pixels at 5+ periods on 11-12 of 14 tiles (+1.2 to +1.8), consistent with the maps | `V4r_Cloud_Blend_Plan.md`, `_Results.md` |
| **Gated V4r blend, 63 tiles, FULLY EXPERIMENTAL → main configuration named v1.0.1** | cloud (5+ snow-free) × cover (v1.0.0 ramp 20-50) × snow (pick NDSI) gates, 48 variants, both partners | run 2026-10-03: gates work (open land, seams, snow partly); maps +4 in dense cloudy pixels; labels +0.3 (n.s.); weighted count not better | `Gated_V4r_Blend_63Tile_Plan.md`, `_Results.md` |
| **v1.0.0 with 60 trees and / or cloud counts (2 × 2), FULLY EXPERIMENTAL** | T60, C30 (fallback + zenith-gated cloud count), C60 | run 2026-10-03: T60 −0.12 MAE (57/63 tiles better); counts raise dense cloudy pixels slightly (+0.6) without raising open land; additive | `Trees_and_Counts_2x2_Plan.md`, `_Results.md` |
| QA bits: open vs dense in cloudy pixels (diagnostic) | all state fields and 512 bit combinations | done 2026-10-03: no usable separation; v1.0.0 already ranks them (AUC 0.99) | `QA_Bits_Dense_vs_Open_Check.md` |
| Snow / low sun in the V4r trigger | Check of MOD44C snow bit and solar zenith on 5 northern tiles and h12v09 | done 2026-10-03: northern firing is about half low sun (zenith test removes it, no effect in tropics); snow bit partial | `Snow_vs_Cloud_V4r_Trigger_Check.md` |
| Less-cloudy compositing from MOD09GA/GQ | Cloudiness only, no model | plan, not approved | `Cloud_Composite_Alternatives_Plan.md` |

**Waiting on the project lead:**
1. Which checks to run on V4R's sharp decreases.
2. Whether any candidate is adopted.

The full write-up of the 2026-10-02 run is still to do.

### 3b. Predictor selection (Monte Carlo rerun)

| Item | What it is | Status | Report |
|---|---|---|---|
| **v1.0.0 + MC predictors** | Stand-alone Monte Carlo over 324 metrics on v1.0.0's inputs and RF settings. Tree: top 64 and top 20; non-vegetated: top 21 and top 64; seed refits as the yardstick | run 2026-10-02. No MC set beats v1.0.0: tree top 64 +0.28 MAE, bare top 64 +0.11 (ties under the rule, but worse than v1.0.0; tree top 64 is worse against every reference) | `Monte_Carlo_Rerun_Plan.md`, `Monte_Carlo_Rerun_Results.md` |

**Waiting on the project lead:**
1. Keep the C code's predictor sets, or change them.
2. Go-ahead to delete the old 2026-09-20 324-metric rasters (about 1.2 TB).

---

## 4. Validation and verification (independent data)

| Item | Data | Product version tested | Status | Report |
|---|---|---|---|---|
| NEON lidar | NEON AOP CHM / canopy cover, 19 sites, 50,561 pixels | an intermediate RF tree model (bit 15, MEAN, truncated): **not exactly v1.0.0** | done 2026-09-30. Ties the historical MOD44B (3 m MAE 20.39 against 20.43); every product and the labels read far below lidar in dense forest | `NEON_Lidar_Eval_Plan.md`, `_Results.md` |
| G-LiHT lidar | G-LiHT on h11v02 | bagging, RF and XGBoost (bit 15), before v1.0.0 | done 2026-09-30. Does not separate the models | `Lidar_Reference_Coincidence_20260930.md` |
| NGA WorldView-2 | 30 WV2 scene parts over cloudy-forest tiles and a control tile: ortho TOA, pan-sharpened, a basic canopy assessment (footprint index from `nga_wv02_m1bs_index_2019_20261002`) | none yet | done 2026-10-02. Products good; cloud mask and canopy thresholds not robust, so **no comparison with any product version yet** | `WV2_Reference_Scenes_Provenance.md` |
| AlphaEarth | Embeddings over the WV2 scenes | none (tested against the WV-derived canopy) | done 2026-10-02. Useful as a consistency check; weak transfer across tiles | `WV2_Reference_Scenes_Provenance.md` |

- **Framing (project lead):** these are comparisons, not truth.
- **NGA rule:** NGA imagery and all derivatives stay on ADAPT/Explore.
- **Not yet done:** none of the validation has been run on v1.0.0 exactly. NEON on v1.0.0 is a cheap rerun if wanted.
