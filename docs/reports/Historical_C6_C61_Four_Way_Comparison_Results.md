# Results: tree cover from historical C6, historical C6.1, the C code and Python RF, 63 tiles

Run 2026-10-01, plan in `Historical_C6_C61_Four_Way_Comparison_Plan.md`. Run root `FullRuns/hist4way_c6_c61_20261001/` (generated tables in `summary/tables.md`).

## Conclusion

**The 2026-10-01 conclusion that Python is comparable to the historical MOD44B holds against C6.1 as well as C6.** On the 54 tiles not affected by the C6.1 high-latitude rule:
- Python RF differs from C6.1 by a pooled MAE of 4.17, with bias +0.25 and correlation 0.968. Against C6 on the same pixels the figures are 4.10, +0.19 and 0.970.
- The C code differs from C6.1 by 3.42 and from C6 by 3.44. Python RF is therefore about 0.7 points further from either historical collection than the C code is.
- For scale, the two historical collections differ from each other by 2.64 on these pixels.

On the 9 flagged tiles, C6.1 sets most vegetated pixels to 0% tree cover. There, Python RF and the C code both agree with C6 (bias within ±1.6) and sit 12 to 22 points above C6.1. That is the C6.1 rule, not a model problem.

## What was compared

| Key | Product |
|---|---|
| C6 | Historical MOD44B Collection 6 (.006), 2019 day 065 |
| C6.1 | Historical MOD44B Collection 6.1 (.061), 2019 day 065 |
| C code | Operational pge61 C code (B1/B2/B3), 63-tile run of 2026-09-19 |
| Python RF | Random Forest, bit-15 QA, 63-tile retrain (written 2026-09-30 09:31-10:30, before the cloudy-pixel work; nothing from the h12v09 cloud-composite runs is used) |

**Note on the C code product (accepted by the project lead):** this is the C code product used in every earlier comparison. It was built with the bits 7-8 aerosol shift (`fix_bands.c` `>> 7`), not the operational `>> 13` and not the bit-15 screen Python now uses. It was not rebuilt for this comparison.

**Method:**
- **Pixels:** one valid-pixel mask per tile, covering pixels where all four products are in 0-100.
- **Pairs and statistics:** six pairs, with the same statistics functions as the earlier reports.
- **Pooling:** the pooled numbers are computed exactly from per-tile sums.
- **Control (passed):** on h09v05 and h12v09, the new code with the old Stage 5b mask reproduced the Stage 5b statistics for the C code and Q15_RF exactly, every column.
- **Run:** ngaproc201, 8 tiles in parallel, 09:34-09:40, 63 of 63 tiles OK.

## The C6.1 high-latitude rule and the flagged tiles

**What C6.1 does.** The previous developer could not solve a high-latitude problem, so C6.1 sets those pixels to 0. In the data, the affected pixels have:
- `Percent_Tree_Cover` = 0;
- `Percent_NonTree_Vegetation` = 0;
- `Percent_NonVegetated` = 100 (99.6-100% of zeroed pixels on the tiles checked).

So the pixel becomes fully non-vegetated. The non-vegetated layer is set to 100, not 0. On those pixels C6 has a mean tree cover of 26-29 and non-tree vegetation of 57-61 (h11v02, h20v02).

**Flag rule (fixed before the run):**
- `zeroed`: more than 10% of common-valid pixels have C6 tree cover above 0 and C6.1 tree cover 0;
- `v01`: grid row v01 (the project lead's decision).

Flagged tiles appear in the per-tile tables and are left out of the pooled numbers. 9 tiles were flagged, leaving 54.

| Tile | Flag | Common pixels | Tree cover zeroed | Non-tree veg zeroed | C6.1 vs C6 bias | RF vs C6 bias | RF vs C6.1 bias | C code vs C6 bias | C code vs C6.1 bias |
|---|---|---|---|---|---|---|---|---|---|
| h11v02 | zeroed | 16,132,711 | 76.3% | 80.0% | -20.07 | -0.04 | +20.03 | +0.38 | +20.45 |
| h12v01 | zeroed | 691,688 | 47.8% | 83.1% | -1.59 | -1.33 | +0.26 | +0.02 | +1.61 |
| h12v02 | zeroed | 20,083,446 | 65.8% | 82.6% | -11.73 | +1.62 | +13.35 | +0.84 | +12.57 |
| h13v01 | zeroed | 71,764 | 49.5% | 98.2% | -1.18 | -1.10 | +0.08 | -0.14 | +1.05 |
| h16v01 | v01 | 14,224,884 | 1.5% | 2.5% | -0.06 | +4.71 | +4.77 | +0.03 | +0.09 |
| h20v02 | zeroed | 21,022,268 | 69.8% | 87.0% | -21.35 | +0.37 | +21.72 | -1.25 | +20.10 |
| h21v01 | zeroed | 7,690,243 | 33.8% | 100.0% | -1.34 | -0.31 | +1.03 | +0.68 | +2.01 |
| h21v02 | zeroed | 21,942,212 | 81.4% | 97.2% | -14.92 | +0.06 | +14.98 | -0.34 | +14.58 |
| h23v02 | zeroed | 21,735,561 | 69.2% | 84.1% | -11.93 | -0.06 | +11.87 | -0.42 | +11.51 |

- **v01 tiles are zeroed too, which supports the project lead's view.** h12v01, h13v01 and h21v01 met the zeroing rule on their own (34-50%), so they would have been excluded even without the v01 decision. Their bias is small only because they have little tree cover.
- **The pre-check's "uniform -5 offset" on the v01 tiles came from the out-of-projection area.** That is the MOD44W out-of-projection region (`Valid_Mask_Disagreement_MOD44W_Root_Cause.md`), where C6 and C6.1 carry values but Python does not. There, C6.1 is about 5 below C6 (-4.9 to -5.4 on h11v02, h13v01, h20v02 and h21v01). The four-way mask leaves that area out.
- **h16v01 was flagged only as v01; the zeroing rule did not flag it** (1.5% zeroed). C6 and C6.1 agree there almost exactly, and so does the C code (MAE 0.08-0.09). Python RF is +4.7 against both, which is the known Greenland over-prediction (`H16v01_TreeCover_OverPrediction_Investigation.md`), not a C6.1 effect.
- **Common pixels on h12v01 and h13v01 are few** (0.69 M and 0.07 M), because most of those tiles is out-of-projection area or not valid in Python.

## Pooled, 54 unflagged tiles (headline)

983,125,140 common-valid pixels.

| Pair | Bias | MAE | RMSE | Corr | Within 5 | Within 10 |
|---|---|---|---|---|---|---|
| Python RF vs C6 | +0.19 | 4.10 | 6.39 | 0.9695 | 73.0% | 90.2% |
| Python RF vs C6.1 | +0.25 | 4.17 | 6.49 | 0.9682 | 72.7% | 90.0% |
| C code vs C6 | -0.12 | 3.44 | 5.97 | 0.9734 | 79.3% | 92.3% |
| C code vs C6.1 | -0.07 | 3.42 | 5.83 | 0.9744 | 79.5% | 92.6% |
| C6.1 vs C6 | -0.06 | 2.64 | 4.58 | 0.9844 | 85.9% | 95.8% |
| Python RF vs C code | +0.31 | 4.43 | 6.92 | 0.9634 | 70.8% | 88.5% |

**All 63 tiles, for reference** (1,106,719,917 pixels):
- Python RF vs C6: 4.13. This matches the earlier 4.20 (Stage 5b) closely; the mask differs slightly.
- C code vs C6: 3.49. This matches `Tree_Cover_vs_Historical_MOD44B_Results.md` (3.49).
- Python RF vs C6.1: 5.29, bias +1.76. C code vs C6.1: 4.52, bias +1.38. C6.1 vs C6: 3.83, bias -1.50.

The zeroed tiles inflate every C6.1 number, which is why they are left out of the headline.

## Per tile, 54 unflagged tiles

| Pair | Median MAE | Tiles over 5 | Tiles over 7 | Worst tile |
|---|---|---|---|---|
| Python RF vs C6 | 4.28 | 20 | 2 | 7.45 (h19v08) |
| Python RF vs C6.1 | 4.33 | 18 | 4 | 7.64 (h24v03) |
| C code vs C6 | 3.37 | 10 | 2 | 8.02 (h19v08) |
| C code vs C6.1 | 3.44 | 10 | 1 | 7.51 (h19v08) |
| C6.1 vs C6 | 2.46 | 3 | 0 | 5.67 (h19v08) |
| Python RF vs C code | 4.75 | 25 | 3 | 8.37 (h19v09) |

- **Tiles over 7 against C6.1:** h24v03 7.64, h19v09 7.52, h19v08 7.46 and h21v10 7.07.
  - h19v08 and h19v09 are hard for every product, and the two historical collections differ by 5.2-5.7 there.
  - h24v03 and h21v10 are tiles where C6.1 itself moves away from C6 (C6.1 vs C6 5.41 and 4.53), so RF's larger MAE there comes partly from the reference.
- The full per-tile MAE table for all 63 tiles and all 6 pairs is in `FullRuns/hist4way_c6_c61_20261001/summary/tables.md`, with one row per tile and pair in `summary/per_tile_all.csv`.

## By reference cover in 4 bins (added 2026-10-01 at the project lead's request)

Bias and MAE are more meaningful by cover level, because tile averages hide high readings at low cover and low readings at high cover that cancel. Bins are by the reference's cover. Each cell is bias / MAE (product minus reference). These are the bit-15 Python RF; earlier 63-tile cover-bin tables (`Model_vs_Historical_DeepDive_20260929.md`) were for the bits 7-8 RF.

**54 unflagged tiles**

| Reference cover | Pixels C6 (M) | RF vs C6 | RF vs C6.1 | C code vs C6 | C code vs C6.1 |
|---|---|---|---|---|---|
| 0-20 | 576.0 | +1.03 / 2.73 | +0.97 / 2.96 | +0.68 / 1.86 | +0.57 / 2.06 |
| 21-40 | 141.4 | +1.00 / 6.78 | +1.23 / 7.06 | -0.30 / 6.13 | -0.01 / 6.32 |
| 41-60 | 123.2 | -1.19 / 6.12 | -1.12 / 5.91 | -0.68 / 5.92 | -0.55 / 5.50 |
| 61+ | 142.6 | -2.81 / 5.23 | -2.53 / 4.73 | -2.70 / 4.99 | -2.30 / 4.32 |

**All 63 tiles** (the C6.1 columns include the zeroed tiles, which push C6.1 down in the 0-20 bin)

| Reference cover | Pixels C6 (M) | RF vs C6 | RF vs C6.1 | C code vs C6 | C code vs C6.1 |
|---|---|---|---|---|---|
| 0-20 | 658.5 | +1.08 / 2.81 | +3.24 / 4.91 | +0.74 / 1.95 | +2.76 / 4.01 |
| 21-40 | 164.5 | +0.90 / 6.76 | +1.30 / 7.09 | -0.50 / 6.15 | +0.06 / 6.34 |
| 41-60 | 137.9 | -1.18 / 6.14 | -1.05 / 5.91 | -1.00 / 6.06 | -0.55 / 5.50 |
| 61+ | 145.8 | -2.80 / 5.25 | -2.51 / 4.73 | -2.76 / 5.03 | -2.30 / 4.32 |

- **Above 60% cover, Python RF and the C code read low by the same amount:** -2.8 and -2.7 against C6.
- **Between 21% and 40%, Python RF reads about 1 point high while the C code is close to 0.** This is where RF differs most from the C code.
- **MAE is 5 to 7 in every bin above 20% cover, for both products.** Pixel-level agreement is much looser than the tile averages suggest.
- **Four tropical tiles chosen at random** (seed 20261001: h12v09, h13v10, h19v09 and h21v10) show the same pattern, more strongly.
  - Python RF reads +0.6 to +6.1 at 0-20% and -1.5 to -9.9 above 60%.
  - Per-bin MAE is 4 to 10.
- **Files:** the 63- and 54-tile tables in 10-point bins are in `summary/by_bin_sets.md`, with the numbers in `summary/by_bin_sets.csv`, generated by `code/by_bin_sets.py`.

## By reference cover in 10-point bins, 54 unflagged tiles

| Reference cover | RF vs C6 | RF vs C6.1 | C code vs C6 | C code vs C6.1 |
|---|---|---|---|---|
| 0 | +0.27 | +0.22 | +0.10 | +0.07 |
| 1-10 | +1.17 | +0.91 | +0.98 | +0.65 |
| 11-20 | +1.62 | +1.85 | +0.76 | +0.92 |
| 21-30 | +1.29 | +1.57 | -0.01 | +0.27 |
| 31-40 | +0.63 | +0.77 | -0.68 | -0.39 |
| 41-50 | -0.43 | -0.34 | -0.54 | -0.43 |
| 51-60 | -1.89 | -1.87 | -0.82 | -0.66 |
| 61-70 | -2.53 | -2.40 | -2.02 | -1.66 |
| 71-80 | -2.97 | -2.71 | -2.82 | -2.54 |
| 81-90 | -3.01 | -2.32 | -3.89 | -3.09 |

(Values are bias, product minus reference. The 91-100 bin is left out: it holds 23 pixels.)

- **The same pattern appears against both collections:** too high at low cover, too low at high cover.
- **Dense forest:** at 81-90% cover, Python RF reads low by -3.0 against C6 and -2.3 against C6.1. The C code reads lower still, at -3.9 and -3.1. So the dense-forest shortfall is not specific to Python.
- **Sparse cover:** at 11-30% cover, Python RF reads higher than the C code does.

## By region and reference cover, 54 unflagged tiles (added 2026-10-01)

Tiles are grouped by location only:
- **latitude band** from the grid row: v01-v03 = 50-80N, v04-v05 = 30-50N, v06-v07 = 10-30N, v08-v09 = 10N-10S, v10-v12 = 10-40S;
- **continent** from the tile centre, using a fixed table in `code/region_bins.py`.

Land cover was tried for the grouping and dropped at the project lead's direction, because it is another satellite-derived product and adds uncertainty. Its outputs are not used.

Each cell gives bias vs C6 for Python RF, then for the C code. MAE and the C6.1 numbers are in `summary/region_bins.md` and `summary/region_bins.csv`.

| Region (tiles) | 0-20 | 21-40 | 41-60 | 61+ | Whole region |
|---|---|---|---|---|---|
| North America 50-80N (h11v03, h12v03) | +1.7 · +1.7 | 0.0 · -0.9 | +0.5 · -0.5 | -2.1 · -2.9 | +0.5 · -0.2 |
| North America 30-50N (10 tiles) | +0.7 · +0.6 | +1.9 · -0.1 | -0.9 · -0.1 | -2.5 · -2.3 | +0.3 · +0.2 |
| South America 10N-10S (h11v08, h11v09, h12v09) | +4.9 · +0.8 | +3.4 · -1.1 | +1.4 · +0.3 | -2.4 · -3.2 | -0.8 · -2.3 |
| South America 10-40S (6 tiles) | +1.5 · +0.8 | +0.3 · +0.6 | -2.7 · -0.1 | -3.5 · -1.3 | +0.3 · +0.4 |
| Europe 50-80N (h18v03, h20v03) | +1.4 · +2.1 | -2.9 · +0.8 | -3.6 · -1.0 | -4.1 · -3.5 | -1.1 · +0.6 |
| Europe 30-50N (h18v04, h19v04) | -0.4 · +1.2 | -0.5 · +0.9 | -1.5 · +0.8 | -2.1 · -1.6 | -0.8 · +0.7 |
| Africa 10N-10S (h19v08, h19v09, h20v08, h20v09) | +1.5 · +3.3 | -0.5 · 0.0 | -2.0 · -0.8 | -2.8 · -2.5 | -0.8 · +0.1 |
| Africa 10-40S (6 tiles) | +1.2 · +0.3 | +3.1 · -0.7 | -1.7 · -4.3 | -7.6 · -6.8 (1M px) | +1.2 · -0.2 |
| Asia 50-80N (h22v03, h23v03, h24v03) | +1.8 · +1.1 | +1.9 · -1.0 | -1.0 · -1.5 | -1.4 · -2.9 | +0.7 · -0.5 |
| Asia 10-30N (h21v06, h26v06, h27v06, h27v07) | +1.7 · +0.6 | +1.8 · -1.0 | -1.4 · -0.4 | -4.3 · -3.3 | +0.4 · -0.4 |
| Asia 30-50N, Australia, Africa north of 10N (mostly arid, 11 tiles) | +0.1 to +0.9 · 0.0 to +0.3 | small | small | small | +0.3 to +0.8 · 0.0 to +0.2 |

**Findings:**
1. **Equatorial South America is where Python RF departs most from the C code, in both directions.**
   - It reads high on sparse and mid-cover pixels: +4.9 and +3.4, against +0.8 and -1.1 for the C code. These cells come mostly from h12v09 and h11v08.
   - In the dense forest (48M pixels) it is closer to C6 than the C code is: -2.4 against -3.2.
2. **Europe is the only region where Python RF reads low overall** (-1.1 and -0.8; the C code reads +0.6 and +0.7). This comes mainly from mid cover in the north (-2.9 and -3.6 at 21-60%). It is a regional pattern, the opposite of the general one.
3. **At 21-40% cover, Python RF reads about 2 points high in most regions where the C code is close to 0.** That covers temperate North America, Asia north and south, and southern Africa (+3.1). This is the most consistent difference between the two products.
4. **Southern South America reads low at higher cover** (-2.7 at 41-60%, -3.5 above 60%, against -0.1 and -1.3 for the C code). This is almost entirely h13v11 (-8.2 against -0.4 above 60%) and h13v10.
5. **Equatorial Africa is equally hard for both products.** It has the highest regional MAE (6.6 for Python RF, 7.1 for the C code), and here the C code reads higher on sparse pixels (+3.3 against +1.5).
6. **Above 60% cover, both products read 1.4 to 4.3 low in every region with real dense area.** That points to a shared limit, not one specific to Python.
7. **The arid regions agree within about 1 point** for both products.

**Per-tile standouts** (bins of at least 1M pixels where Python RF's bias is more than 4 from the 54-tile bin bias, or its MAE is more than 2 above the C code's):
- **High at sparse or mid cover:** h12v09 (+5.6 at 0-20 and +5.4 at 21-40, against the C code's +0.9 and +0.1), h11v08, h08v05 (+6.6 at 21-40) and h09v05.
- **Low at dense cover:** h13v11 and h10v05.
- **Bad for both:** h19v08 at 41-60 (MAE 11.0 and 12.3).

Full list: `summary/region_bins.md`.

## Limits

- Neither historical collection is ground truth. This measures agreement, and C6 and C6.1 share the historical training data and the missing aerosol screen.
- The comparison covers 2019 and the 63 training tiles only.
- The C code product uses the bits 7-8 aerosol shift (see above).

## Files

- **Code:** `code/four_way.py` (per tile, plus `--control`), `code/summarize.py`, `code/c61_layers_check.py`, `code/run_all.sh`, `code/sing.sh`, and the pre-check scripts in `code/precheck/`.
- **Outputs:** `per_tile/` (stats, by bin and flag measures per tile), `summary/` (`per_tile_all.csv`, `pooled.csv`, `by_bin_pooled.csv`, `tables.md`), `control/` and `logs/`.
