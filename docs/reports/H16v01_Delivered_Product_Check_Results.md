# h16v01 (Greenland snow tile): check of the delivered product (results, to-do item 3)

**Run 2026-10-01, finished 20:43, right after the delivery run. Read-only. Plan: `H16v01_Delivered_Product_Check_Plan.md` (approved).** Folder: `FullRuns/h16v01_delivered_check_20261001/` (`out/summary.md`, `out/stats.csv`).

## Summary

- **MEDIAN tree cover removes the h16v01 tree over-prediction.** The delivered tree cover averages 0.40% (C6 MAE 0.44), close to the C code (0.11 / 0.08). RF MEAN gave 4.78 and the bit-15 baseline 9.20. Only 0.58% of land pixels are above 10% tree cover, against 22.6% (RF MEAN) and 32.4% (baseline).
- **The problem moves to non-vegetated.** The delivered bare averages 89.0 against C6's ~98.7 (MAE 10.2, bias −9.6). The C code gives 99.2 (MAE 0.7). Non-tree vegetation therefore takes up the rest, at 10.6% on average, with 41% of land above 10%. On the Greenland ice sheet that is wrong.
- RF MEAN and the delivered product have identical non-vegetated layers (MEAN for both). The baseline is about 1 point better (MAE 9.3), a tie.
- **So:** the tree half of the h16v01 problem is solved without any fix. The bare half remains, and it is the same signal the zenith-only fix addresses (`Greenland_Iceland_Scoped_Fix_v2_Results.md`: bare MAE 10.2 → 5.5 on this tile). It is also the region where the C code forces bare = 100 (`Bare_C_Regions_Investigation_Results.md`).

## Results (land pixels; C6 is the primary reference)

C6.1 applies the high-latitude rule (tree 0 / bare 100) on this tile. The labels (`v5.0.3samp`) are all 0% tree.

| Layer | Product | Mean | > 10 % | > 20 % | C6 MAE | C6 bias | C6.1 MAE | Labels MAE |
|---|---|---|---|---|---|---|---|---|
| Tree cover | **delivered (RF, MEDIAN tree / MEAN bare)** | **0.40** | 0.58 | 0.10 | **0.44** | +0.33 | 0.39 | 0.0 |
| Tree cover | RF MEAN (aggregation run) | 4.78 | 22.60 | 0.78 | 4.78 | +4.70 | 4.77 | 0.006 |
| Tree cover | bit-15 baseline | 9.20 | 32.41 | 22.03 | 9.18 | +9.11 | 9.10 | 0.006 |
| Tree cover | C code (2026-09-19) | 0.11 | 0.38 | 0.14 | 0.08 | +0.03 | 0.09 | 0.09 |
| Non-vegetated | **delivered** | **88.98** | 99.97 | 99.83 | **10.21** | −9.63 | 10.97 | |
| Non-vegetated | RF MEAN | 88.98 | 99.97 | 99.83 | 10.21 | −9.63 | 10.97 | |
| Non-vegetated | bit-15 baseline | 89.91 | 99.95 | 99.78 | 9.29 | −8.70 | 10.02 | |
| Non-vegetated | C code | 99.17 | 100.00 | 99.99 | 0.70 | +0.53 | 0.79 | |
| Non-tree vegetation | **delivered** | **10.62** | 40.97 | 16.35 | **9.80** | +9.29 | 10.58 | |
| Non-tree vegetation | RF MEAN | 4.03 | 12.48 | 4.92 | 5.53 | +4.96 | 6.22 | |
| Non-tree vegetation | bit-15 baseline | 2.92 | 9.04 | 5.01 | 3.90 | +3.24 | 4.50 | |
| Non-tree vegetation | C code | 0.72 | 1.90 | 1.66 | 0.72 | −0.56 | 0.70 | |

Non-tree vegetation is larger in the delivered product than in RF MEAN because MEDIAN tree cover is lower while bare is unchanged. With MEAN tree cover, the sum cap took up part of the gap (1,014,739 capped pixels, against 7,649 with MEDIAN).

## SD layers on this tile

| | Mean | p95 |
|---|---|---|
| Tree SD | 9.19 | 25.05 |
| Non-vegetated SD | 9.18 | 21.57 |

Tree SD is 23.6 where delivered tree cover is above 10%, against 9.1 elsewhere. The few remaining tree-cover pixels are flagged as uncertain.

## Decision it feeds

The tree half of the h16v01 problem needs no fix. Whether the bare half needs the zenith-only fix, a bare rule like the C code's, or neither is in the Greenland and bare-regions reports.
