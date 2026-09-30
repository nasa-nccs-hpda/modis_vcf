# Tree cover: C and Python compared with the historical MOD44B (2019, 63 tiles)

Run 2026-09-20. Plan: `Tree_Cover_vs_Historical_MOD44B_Plan.md`. Everything is under `$RUN` = `MOD44B_VCF_Development/FullRuns/full63_c_vs_py_20260919-0806/`.

## Bottom line

Both C and Python track the historical MOD44B tree cover closely (pooled correlation 0.97 and 0.95, bias near zero). C is consistently closer: its MAE is about 30% lower (3.49 against 5.00), it has the lower MAE on 60 of 63 tiles, and Python's error is larger in every historical-cover bin.
The gap is modest in absolute size and consistent in direction.

Read with care:
- The historical product is a Collection 6 reference with different inputs and training data, not ground truth, and no standard for "reasonable" was set for this comparison.
- No significance test was run. With 1.1 billion pixels almost any gap would test as significant and neighbouring pixels are not independent, so the tile-level consistency (60 of 63) is the more meaningful evidence.
- Python is a reimplementation of the C pipeline, so C being closer to the historical product is not evidence that Python is wrong. How closely the historical product's production code matches this C run was not checked.

## What was compared

Layer `Percent_Tree_Cover`, year 2019 (composite `A2019065`), all 63 tiles. Three products:

- **H, historical:** MOD44B Collection 6, `/css/modis/Collection6/L3/MOD44B-VCF/2019/065/` (produced April 2020).
- **C:** the C B3 output of the 63-tile run (Collection 6.1 inputs, models trained on the pooled samples of the 63 tiles).
- **P, Python:** the Python predictions of the same run.

The historical product is a reference, not ground truth: it differs from the new runs in collection, inputs and training data, so the numbers show agreement with it, not error.
"Bias" is the first product minus the second. Python writes integers by truncation, which by itself lowers its bias by about 0.5 against a rounded product.

**Headline mask: common-valid**, meaning pixels with a value 0-100 in all three products, so C and Python are scored on exactly the same 1,106,719,917 pixels. Pairwise-valid results (pixels valid in
both products of each pair) are in `summary.txt` and `per_tile.csv` and differ little, except for C against historical (below).

## Result

| Common-valid, pooled over 63 tiles | Corr | Bias | MAE | RMSE | Within 5 | Within 10 |
|---|---|---|---|---|---|---|
| **C vs historical** | 0.972 | -0.12 | **3.49** | 6.02 | 78.6% | 92.1% |
| **Python vs historical** | 0.953 | +0.25 | **5.00** | 7.77 | 67.0% | 85.3% |
| Python vs C (the earlier comparison) | 0.960 | +0.38 | 4.52 | 7.03 | 70.1% | 88.1% |

- **C is closer to the historical product than Python is.** MAE 3.49 against 5.00. On the pixels where the two differ in distance, C is closer on 50.8% and Python on 24.9% (equally close on 24.3%).
- **By tile, C has the lower MAE against the historical product on 60 of 63 tiles.** Median tile MAE: C 3.42, Python 5.39.
- **Check:** the Python vs C row reproduces the earlier headline exactly (correlation 0.96, MAE 4.5, bias +0.4, about 1.11 billion pixels). A pilot on h09v05 and h12v02 reproduced the earlier per-tile
  rows to a difference of exactly 0 before the full run was started.
- **Overall bias is small for both (-0.12 and +0.25), but it hides opposite errors at low and high cover**, as the next table shows.

## Where the disagreement is: by historical tree-cover bin

Common-valid, pooled; bias, MAE and RMSE against the historical value.

| Historical cover | Pixels | C bias | C MAE | Python bias | Python MAE |
|---|---|---|---|---|---|
| 0 | 196,461,018 | +0.2 | 0.2 | +0.8 | 0.8 |
| 1-10 | 309,769,792 | +1.1 | 2.0 | +1.5 | 3.5 |
| 11-20 | 152,244,367 | +0.9 | 4.1 | +1.9 | 6.6 |
| 21-30 | 91,921,188 | -0.2 | 5.9 | +1.4 | 7.9 |
| 31-40 | 72,625,854 | -0.9 | 6.5 | +0.4 | 8.0 |
| 41-50 | 68,227,831 | -0.9 | 6.5 | -0.9 | 7.6 |
| 51-60 | 69,694,925 | -1.0 | 5.7 | -2.3 | 7.1 |
| 61-70 | 58,326,426 | -2.2 | 5.7 | -3.4 | 7.4 |
| 71-80 | 62,389,988 | -2.8 | 4.7 | -4.0 | 6.4 |
| 81-90 | 25,058,505 | -3.9 | 4.1 | -4.2 | 5.2 |
| 91-100 | 23 | -24.3 | 24.3 | -37.6 | 37.6 |

- Both products are biased **high at low cover and low at high cover**, the usual regression-to-the-mean pattern. Python's error is larger in every bin, by 0.6 to 2.5 MAE points.
- The 91-100 row rests on 23 pixels, so it says nothing.
- Python's high bias at 0% cover (+0.8, MAE 0.8, against C's +0.2) matters because 0% cover is a very large group, with 196 million pixels.

## Tiles

- **Best agreement with the historical product** (C MAE): h21v06 0.0, h20v06 0.0, h16v01 0.1, h19v11 0.1, h29v11 0.3. A MAE this close to 0 presumably means almost every pixel is 0% cover (inferred from the MAE, not checked), where small errors are easy.
- **Worst for both products:** seven tiles are in both products' ten worst by MAE (h11v08, h19v08, h19v09, h20v09, h21v10, h26v06, h27v06); the worst are h19v08 (C 8.0, Python 8.9) and h19v09 (7.5, 8.6). When both
  products miss the historical value on the same tiles, the cause is more likely the reference or the inputs the two share than either implementation, consistent with the caution in the plan. This
  was not investigated.
- **Python beats C on three tiles**, by small margins: h21v01 (0.46 lower MAE), h12v01 (0.28) and h13v01 (0.26).
- **Lowest correlations with the historical product:**
  - C: h13v01 0.30, h12v01 0.35, h21v01 0.60, h16v01 0.62;
  - Python: h16v01 0.005, h12v01 0.05, h13v01 0.10, h21v06 0.45.
  - Two tiles have C correlation below 0.5 and six have Python below 0.5. The lowest ones are mostly v01 tiles, the northernmost tile row.
- **Caution on those tiles:** the common-valid mask is small on h13v01 (71,764 pixels), h12v01 (691,688) and h12v05 (766,661), so their statistics rest on few pixels. The correlation on tiles with almost no variance
  in tree cover (h21v06, h20v06) is also unstable and should not be read as a difference.

## The mask, and what this comparison does not cover

- Choosing common-valid removes 2.97% of the pixels that C and the historical product share (1,140,640,643 pairwise against 1,106,719,917 common); for Python against historical the two masks differ by only
  179,551 pixels (valid in Python and the historical product but not in C). The per-tile mask counts and 3 x 3 cross tables (valid, water, other) for historical-C, historical-Python and
  C-Python are in `mask_agreement.csv`.
- **The difference in valid pixels between Python and C is not analyzed here**, as decided; the counts are recorded so the later investigation can start from them. It is worth noting that three of
  the tiles where Python beats C (h21v01, h12v01, h13v01) are among the tiles named in README open item 2 (the fourth tile in that item, h11v02, is not one of them). That is an observation for that later work, not a conclusion.
- **Caveat added 2026-09-27, not corrected here**: that later investigation (`Valid_Mask_Disagreement_MOD44W_Root_Cause.md`) found that on h12v01, h13v01, h21v01 and h11v02 the historical product
  marks the same MOD44W out-of-projection area (a real sinusoidal-grid edge, not a data defect) with the value **18**, not a documented fill code - it passes the 0-100 valid check used everywhere,
  including here, and is spatially ~93-100% coincident with the area Python leaves unpredicted on those tiles. The `hist_inventory` step above found no fill code because it only checked for values
  outside 0-100/200 - 18 is inside that range. Because Python is also invalid across nearly all of that same area, the **common-valid (three-product) numbers above are only negligibly affected**
  (independently confirmed: ~0.02% of a comparable common-valid pixel pool in a later run). The **C-vs-historical pairwise numbers are the ones to distrust**: pairwise adds back 33.9 million pixels
  over common-valid (1,140,640,643 vs 1,106,719,917, this document's own count above), almost exactly the ~34.0 million value-18 pixels found across these four tiles - i.e. the pairwise C-vs-historical
  comparison is likely built substantially from this disguised fill value on these tiles, not real tree cover. Not recomputed here.
- Only tree cover, only 2019, only the Collection 6 historical product. The Collection 6.1 product in `/css/modis` was deliberately not used. The Quality and Cloud layers were not used.
- Nothing here says which product is right, only which is closer to the historical one.

## Files

`$RUN/Compare/tree_cover_vs_historical_63/`: `summary.txt` (all pooled numbers and the ten best and worst tiles), `per_tile.csv` (every pair and mask per tile, including a 90th-percentile absolute
difference), `by_cover_bin.csv`, `closer_per_tile.csv`, `mask_agreement.csv`. Also `$RUN/Compare/hist_inventory/` (step 0: the historical layer holds only 0-100 and 200, no fill code) and
`$RUN/Compare/tree_cover_vs_historical_pilot/` (the two-tile pilot and its check).

Code: `$RUN/code/validation/compare_tree_cover_vs_historical.py`, `run_hist_compare.sh` (`pilot` or `full`), `hist_value_inventory.py`, `run_hist_inventory.sh`. The full run started 07:40:20 and ended 07:56:04 on
forest203 (15 minutes 44 seconds), detached, rc=0.

To rerun, use a new output folder name (the script refuses to reuse one): edit `OUT` in `run_hist_compare.sh`, then `bash run_hist_compare.sh full` on forest203, started with `setsid nohup`.
