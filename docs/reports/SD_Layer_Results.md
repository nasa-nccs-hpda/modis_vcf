# Per-pixel SD layers for the Python VCF (results)

**Run 2026-10-01, 14:38-15:20, forest201-205, fully detached. Plan: `SD_Layer_Plan.md` (approved).** Folder: `FullRuns/sd_layer_test_20261001/`.

## Summary

- **The SD works and costs almost nothing.** It is the population SD of the 30 members (trees for RF, bagged trees for the baseline), stored as round(100 × SD).
  - The map values are untouched: all 5 tiles are identical to the existing maps.
  - The members' mean equals `predict()` exactly.
  - The member pass takes about the same time as `predict()`.
- **The Python SD is a useful uncertainty signal, and a better one than C6.1's.** On held-out pixels its rank correlation with absolute error is 0.87 for tree cover and 0.60 for non-vegetated. C6.1's SD against C6.1's error gives 0.78 and 0.45. Error rises steadily with SD.
- **Levels differ, as expected.** The RF tree SD is about 1.7-3× C6.1's, because RF trees are noisier members than the C code's M5P models. The value is not comparable one-to-one with the historical layer.

## Step 0 and step 1 checks

| Check | Result |
|---|---|
| Members' mean == `rf.predict` (h08v05, 23 M pixels) | exact (largest difference 0.0); int16 equal to the existing map on all 17,991,862 land pixels |
| `_memberSd` == `np.std` over members (1,000 rows) | exact (0.0) for the tree RF, the bare RF and the bare baseline (`BaggingRegressor`) |
| Value layers with the patch == existing maps (5 tiles, both layers) | identical on every tile |
| Time per tile (h08v05, 12 threads) | member pass 33 s against `predict()` 25 s; reading metrics (154 s) dominates |

**Code:** the patched copy is `code/modis_vcf/model/VcfPredict.py` (diff in `manifests/VcfPredict_sd.diff`). It adds a `_memberSd` helper and writes `Percent_Tree_Cover_SD` and `Percent_NonVegetated_SD`. Water and out-of-projection are 0. XGBoost (no members) gets no SD layer.

In this test the value layers still come from `predict()`. The single-pass version, which takes the value from the members' mean, is safe for RF (exact match shown above). For the test files `Utils.writeRaster` writes Int16 (the largest value is 5,000). The product layer should be uint16 with scale factor 0.01, as in the historical files.

## Step 3: does the SD track error? (held-out pixels, 5 tiles)

**Labels:**
- tree cover: the bit-15 RF's 20% held-out split;
- non-vegetated: the new RF's 20% held-out split.

**Caveat:** for C6.1 the same labels are in-sample, because the C code trains its tree models on all label pixels. That makes C6.1's errors look smaller than they would be held out. The SD-error correlation is still a fair like-for-like comparison.

| Layer | Product | Pixels | Mean SD | MAE | Spearman(SD, abs. error) |
|---|---|---|---|---|---|
| Tree cover | Python RF | 224,494 | 4.58 | 2.97 | **0.87** |
| Tree cover | C6.1 | 224,686 | 2.37 | 4.35 | 0.78 |
| Non-vegetated | Python RF | 698,115 | 6.20 | 3.94 | **0.60** |
| Non-vegetated | C6.1 | 549,665 | 6.00 | 9.29 | 0.45 |

C6.1 non-vegetated excludes h12v02, where the C6.1 high-latitude rule applies.

**MAE by SD decile (pooled):**

| Decile | Tree, Python (SD / MAE) | Tree, C6.1 | Non-veg, Python | Non-veg, C6.1 |
|---|---|---|---|---|
| 1 | 0.00 / 0.01 | 0.50 / 0.01 | 0.38 / 0.34 | 0.97 / 1.59 |
| 5 | 0.74 / 0.26 | 0.85 / 1.34 | 5.11 / 3.22 | 4.64 / 8.84 |
| 8 | 7.99 / 5.06 | 3.54 / 8.19 | 8.05 / 4.99 | 7.90 / 14.61 |
| 10 | 17.79 / 12.50 | 8.30 / 13.57 | 15.67 / 10.09 | 15.33 / 13.09 |

All ten deciles are in `out/analysis/sd_vs_error_deciles.csv`.

- **Python's error rises steadily with its SD.** For tree cover, about 40% of held-out pixels have SD below 0.05 and error of about 0 (deserts, closed forest, tundra).
- **C6.1's SD stops tracking error at the top.** Non-vegetated MAE peaks at decile 8 and then falls.
- **The 0.5% floor in the C code's SD** (see the plan) shows in C6.1's lowest deciles.
- **The tree-combining test agrees on all 2.3 M tree-cover and 2.16 M non-vegetated held-out pixels.** In `RF_Tree_Aggregation_Test_Results.md`, MAE rises from about 0 in the lowest SD decile to 18.5 and 11.8 in the highest.

## Step 4: levels against C6.1 and the C code (land pixels, 5 tiles)

| Tile | Layer | Python SD mean / median | C6.1 mean / median | Ratio of medians | Correlation with C6.1 |
|---|---|---|---|---|---|
| h08v05 | Tree cover | 5.12 / 2.31 | 2.18 / 0.86 | 2.69 | 0.82 |
| h12v09 | Tree cover | 16.39 / 16.68 | 5.86 / 5.64 | 2.96 | 0.55 |
| h19v11 | Tree cover | 1.67 / 0.87 | 0.68 / 0.51 | 1.71 | 0.50 |
| h20v06 | Tree cover | 0.99 / 0.90 | 0.55 / 0.51 | 1.76 | 0.74 |
| h12v02 | Tree cover | 12.19 / 14.69 | 5.27 / 5.47 | 2.69 | 0.69 |
| h08v05 | Non-veg | 10.40 / 9.55 | 6.78 / 6.19 | 1.54 | 0.34 |
| h12v09 | Non-veg | 7.36 / 4.43 | 5.20 / 5.09 | 0.87 | 0.31 |
| h19v11 | Non-veg | 17.78 / 18.46 | 9.48 / 8.36 | 2.21 | 0.06 |
| h20v06 | Non-veg | 1.42 / 0.48 | 0.91 / 0.51 | 0.94 | 0.68 |
| h12v02 | Non-veg | 14.93 / 14.63 | 5.83 / 4.91 | 2.98 | 0.12 |

The C code columns are in `out/analysis_v1_invalid_ycolumn/sd_levels.csv`; that step was not affected by the v1 bug.

- **Tree-cover SD has the same spatial pattern as C6.1** (correlation 0.5-0.8) at about 1.7-3× the level.
- **Non-vegetated SD agrees poorly with C6.1** (0.06-0.68). That is expected, because the historical bare SD comes from the static models.

## Run note: v1 analysis error (fixed)

The first analysis run stored the labels in a column named `y`, which overwrote the pixel-column key `y`. Held-out pixels in step 3 were therefore read from the wrong places. That output is kept as `out/analysis_v1_invalid_ycolumn/` and `logs/analysis_v1_invalid_ycolumn.log` and should not be used. Steps 1 and 4 of v1 did not use that table and match the rerun. The script was fixed and rerun at 15:13 on forest204.

## Next (for approval)

1. **Add the SD code to the delivery branch** (`Git_delivery/modis_vcf`, as a local commit; the project lead pushes). Write uint16 with scale factor 0.01, and optionally use the single-pass prediction for RF.
2. **Produce SD for all 63 tiles** with the chosen models through the normal prediction run.
3. **Decide whether the value layer should use the median of the trees** (`RF_Tree_Aggregation_Test_Results.md`). The SD stays the SD of the trees either way.
