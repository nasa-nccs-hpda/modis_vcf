# Combining the 30 RF trees: mean, median or trimmed mean (results)

**Run 2026-10-01, 14:58-15:10, forest202, fully detached.** This was a short test approved by the project lead without a separate plan. Folder: `FullRuns/rf_aggregation_test_20261001/` (`scripts/aggregation_test.py`, `out/`).

## Question

RF's prediction is the mean of its 30 trees. Would the median, or a mean with the tails dropped, be better? Nothing was refitted. Each tree's prediction was taken on the held-out pixels and combined in each way:

- **Tree cover:** the bit-15 RF, on the 20% held-out split (2,289,990 pixels, 63 tiles).
- **Non-vegetated:** the new bit-15 RF, on its 20% held-out split (2,160,056 pixels, 20 tiles).
- **Ways of combining:**
  - the mean (current);
  - the median;
  - trimmed means dropping k = 1, 2, 3 or 5 trees at each end;
  - a winsorized mean (k = 3).
- **Scoring:** raw (unrounded) values, as in the earlier held-out evaluations. Intervals come from 2,000 bootstrap replicates over 100 × 100-pixel blocks and over tiles.

## Results

| Layer | Combination | MAE | Bias | RMSE | MAE − mean | 95% CI (blocks) | 95% CI (tiles) | Tiles better than the mean |
|---|---|---|---|---|---|---|---|---|
| Tree cover | mean | 7.32 | +0.07 | 11.43 | - | - | - | - |
| | trim k = 3 | 6.92 | -0.16 | 11.31 | -0.40 | -0.41 to -0.39 | -0.47 to -0.34 | 62/63 |
| | trim k = 5 | 6.81 | -0.23 | 11.31 | -0.51 | -0.53 to -0.50 | -0.61 to -0.43 | 62/63 |
| | **median** | **6.71** | -0.32 | 11.49 | **-0.61** | -0.63 to -0.59 | -0.74 to -0.50 | 61/63 |
| Non-vegetated | mean | 4.20 | +0.06 | 7.11 | - | - | - | - |
| | trim k = 3 | 3.94 | -0.25 | 7.04 | -0.27 | -0.28 to -0.25 | -0.35 to -0.19 | 20/20 |
| | trim k = 5 | 3.88 | -0.34 | 7.05 | -0.33 | -0.34 to -0.32 | -0.44 to -0.23 | 20/20 |
| | **median** | **3.83** | -0.48 | 7.20 | **-0.37** | -0.39 to -0.35 | -0.49 to -0.26 | 20/20 |

The full tables, including trim k = 1 and 2 and the winsorized mean, are in `out/overall.csv` and `out/differences.csv`. The gain grows steadily from trimming 1 tree, through 2, 3 and 5, to the median.

**By label bin** (MAE, with bias in brackets):

| Layer | Combination | 0-20 | 21-40 | 41-60 | 61+ |
|---|---|---|---|---|---|
| Tree cover | mean | 4.95 (+4.1) | 10.24 (+3.0) | 9.44 (-3.1) | 9.72 (-8.5) |
| | trim k = 5 | 4.08 (+2.8) | 11.01 (+2.4) | 10.05 (-2.4) | 8.68 (-6.9) |
| | median | 3.82 (+2.3) | 11.52 (+2.1) | 10.49 (-2.1) | 8.30 (-6.3) |
| Non-vegetated | mean | 3.28 (+1.9) | 6.64 (0.0) | 5.87 (-1.7) | 3.77 (-2.2) |
| | median | 2.55 (+0.4) | 6.96 (-0.6) | 5.91 (-1.5) | 3.47 (-1.4) |

## Reading

- **The median is better than the mean on both layers, consistently.** The difference is −0.61 points for tree cover and −0.37 for non-vegetated. Every interval clears zero, and the median is better on 61/63 and 20/20 tiles.
- **Under the project rule both gains are ties,** because they are under about 1 point. So the median is not required on accuracy alone. Its case rests on how it behaves:
  - **It reduces the range compression.** The 61+ tree-cover bias goes from −8.5 to −6.3 and the 0-20 bias from +4.1 to +2.3. In the 61+ bin MAE improves from 9.7 to 8.3. This is the known dense-forest under-reading.
  - **It costs a little in the middle bins:** 21-40 MAE goes from 10.2 to 11.5, 41-60 from 9.4 to 10.5. RMSE rises slightly, because the median gives up some squared error to reduce absolute error.
- **Why it works:** the trees are fully grown, so each tree's prediction is essentially one training label. Where most trees say 100 and a few say 60, the mean is pulled toward the middle and the median is not.
- **The trees' disagreement tracks error closely.** MAE rises steadily with the per-pixel SD of the 30 trees, from about 0 in the lowest SD decile to 18.5 (tree cover) and 11.8 (non-vegetated) in the highest (`out/by_sd_decile_*.csv`). The median helps in every SD decile. So the SD layer is a meaningful uncertainty signal; that is tested properly in the SD run.

## Options (decision for the project lead)

1. **Keep the mean.** It is a tie by the rule, matches the C code and scikit-learn, and keeps RMSE lowest.
2. **Switch to the median.** It gives the lowest MAE and the least compression at both ends, and costs nothing extra: the per-tree predictions are computed anyway for the SD layer. The trade-off is slightly worse middle bins and RMSE.
3. **A compromise, a trimmed mean with k = 3-5,** which gets most of the median's gain with a smaller middle-bin cost.

Whichever is chosen, the SD layer stays the SD of the 30 trees. This test also does not cover the map-level effect against C6, C6.1 or lidar, which would be the next check before changing the default.

## Addendum: the mean of the middle trees around the median (v2 run, 15:08-15:20, forest205)

The project lead asked about the mean of k trees on either side of the median. With 30 sorted trees, the middle 2k trees is the trimmed mean that drops 15 − k at each end. The v2 run added the middle 18, 16, 14, 12, 10, 8, 6 and 4 trees. The v1 outputs are kept in `out_v1_trim1-5/` and `scripts/aggregation_test_v1.py`.

| Trees kept (middle) | Tree cover MAE | Δ vs mean | RMSE | 21-40 MAE | 61+ bias | Non-veg MAE | Δ vs mean | RMSE |
|---|---|---|---|---|---|---|---|---|
| 30 (mean) | 7.319 | - | 11.43 | 10.24 | −8.48 | 4.204 | - | 7.11 |
| 24 | 6.917 | −0.40 | **11.31** | 10.79 | −7.34 | 3.938 | −0.27 | **7.04** |
| 20 | 6.808 | −0.51 | **11.31** | 11.01 | −6.94 | 3.875 | −0.33 | 7.05 |
| 16 | 6.746 | −0.57 | 11.34 | 11.18 | −6.67 | 3.841 | −0.36 | 7.07 |
| 12 | 6.712 | −0.61 | 11.37 | 11.31 | −6.48 | 3.826 | −0.38 | 7.11 |
| 10 | 6.703 | −0.62 | 11.40 | 11.36 | −6.40 | **3.823** | −0.38 | 7.12 |
| 6 | **6.698** | −0.62 | 11.44 | 11.45 | −6.30 | 3.826 | −0.38 | 7.16 |
| 2 (median) | 6.706 | −0.61 | 11.49 | 11.52 | −6.25 | 3.834 | −0.37 | 7.20 |

All differences are significant (block CIs about ±0.02). Middle-20 and narrower are better than the mean on 61-62/63 tree-cover tiles and on 20/20 non-vegetated tiles.

**Reading:**
- **The MAE curve is flat from about the middle 12 trees down to the median.** The minimum is the middle 6-10 trees, but it is within 0.02 of the median: a tie.
- **Narrowing the band trades in a predictable way.** It lowers MAE and range compression (61+ bias −8.5 → −6.3) and raises the 21-60 errors and RMSE.
- **The middle 20 (drop 5 at each end) is the most balanced point.** It gives about 83% of the median's MAE gain, the lowest RMSE of all the options, and a smaller cost in the middle bins.
- **The middle 10 gives essentially all of the median's gain,** with slightly better RMSE and middle bins than the median.

**Options:** the middle 20 (balanced; the lowest RMSE), the middle 10 (≈ median MAE, a little less middle-bin cost), or the median. Under the 1-point rule all are ties with the mean. The choice is about which error the product should minimise.
