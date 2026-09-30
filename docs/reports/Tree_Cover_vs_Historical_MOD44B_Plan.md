# Plan: compare the C and Python tree cover with the historical MOD44B (2019, 63 tiles)

Written 2026-09-20. **Status (2026-09-20): the plan was carried out; results are in `Tree_Cover_vs_Historical_MOD44B_Results.md`.** It began as a plan only. The only things done while writing it were read-only probes of one
tile (h12v02) to learn the file formats; they are listed under "What was checked".

## Goal

Add a third view to the 63-tile C vs Python result (`FullRun_Results_C_vs_Python_63tiles.md`): how close is each of the two new products to the
historical MOD44B for the layer **Percent_Tree_Cover**, and does either one agree with the historical product better than with the other?
This is tree cover only, year 2019 (composite `A2019065`).

The historical product is a reference, not ground truth. Differences can come from the historical processing (different collection, inputs and training data)
as much as from the C or Python runs. The comparison shows agreement, not error.

## What was checked (read-only, 2026-09-20)

| Item | Finding |
|---|---|
| Historical files | `/css/modis/Collection6/L3/MOD44B-VCF/2019/065/`: 291 tile HDFs named `MOD44B.A2019065.<tile>.006.<production stamp>.hdf` (produced April 2020). All 63 of our tiles are present, one file each. |
| Layers in a historical HDF | Percent_Tree_Cover, Percent_NonTree_Vegetation, Percent_NonVegetated, Quality, Percent_Tree_Cover_SD, Percent_NonVegetated_SD, Cloud. Same names and order as C's B3 output. |
| Grid | Historical and C: 4800 x 4800, 8-bit, identical origin (-6671703.118) and pixel size (231.656 m) on h12v02. Python: 4800 x 4800 int16 GeoTIFF **with no georeferencing** (pixel size 1, origin 0), so its alignment with the others rests on row and column identity. The earlier C vs Python comparison (correlation 0.96) already relies on that. |
| Value codes, h12v02 | Historical tree cover: 0-100 valid (20,127,839 pixels), 200 water (2,912,161), no 253. C: 0-100 valid (20,510,996), 200 water (2,517,812), 253 fill (11,192; `_FillValue` 253, valid range 0-100). Python: values up to 200. |
| Consequence | The valid and water masks **differ between the historical product and C already on this one tile** (about 0.4 million more water pixels historically). Any comparison has to choose a mask and report the disagreement. |
| Where things run | The login nodes have no singularity, no GDAL and no numpy build that reads HDF4. Everything runs on **forest203** in the `modis-vcf-latest` container, as the earlier comparison did. |
| Existing tool to reuse | `$RUN/code/validation/compare_c_vs_python_all.py` (`$RUN` = `FullRuns/full63_c_vs_py_20260919-0806`). It already reads C's HDF, reads Python's per-tile GeoTIFFs, computes corr, bias, MAE, RMSE, within 5 and within 10, pools exactly by accumulating sums, and refuses to overwrite its output folder. |

## Inputs

- **Historical:** `/css/modis/Collection6/L3/MOD44B-VCF/2019/065/MOD44B.A2019065.<tile>.006.*.hdf`, subdataset `Percent_Tree_Cover`.
- **C:** `$RUN/C/production/build_context/runtime/b3_output/MOD44B.A2019065.<tile>.061.hdf`, subdataset `Percent_Tree_Cover`.
- **Python:** `$RUN/Python/predictions/Percent_Tree_Cover-<tile>-2019.tif`.
- **Tiles:** the same 63 used for the C vs Python comparison.

All inputs are read only. Nothing is written outside a new output folder.

## Method

1. **One new script**, `$RUN/code/validation/compare_tree_cover_vs_historical.py`, built from `compare_c_vs_python_all.py` (copy, do not edit the original). It takes the historical
   directory as an argument, so the same script can be pointed at another reference later. It writes to a new folder,
   `$RUN/Compare/tree_cover_vs_historical_63/`, and refuses to reuse it.
2. **Per tile, three pairs:** C vs historical, Python vs historical, and C vs Python (the last reproduces the existing result and serves as a check on the new code).
3. **Two masks per pair:**
   - *common-valid* (**the headline**, decided 2026-09-20): pixels with 0-100 in all three products. C and Python are scored on exactly the same pixels;
   - *pairwise-valid* (shown alongside): pixels with a value 0-100 in both products of the pair.
4. **Statistics, per tile and pooled over all tiles (exact pooling from accumulated sums, as the existing script does):** pixel counts, corr, mean of each side, bias (product minus historical),
   MAE, RMSE, share within 5 and within 10, and the 90th percentile of the absolute difference (per tile only).
5. **Mask agreement, per tile (recorded only, not analyzed):** for each of the historical, C and Python tree-cover layers, counts of valid (0-100), water (200), fill and other codes, and the cross table of historical against each product.
   It is written to `mask_agreement.csv` so that the counts are available next to every statistic and for the later valid-pixel investigation (see "Out of scope"). This comparison does not interpret it.
6. **Where the disagreement lives:** MAE and bias per historical tree-cover bin (0, 1-10, 11-20, ... 91-100) for C and for Python, pooled over all tiles. It shows whether either product is
   biased at low or high cover, which a single tile-level number hides.
7. **Which is closer:** on the common-valid pixels, the share where C is closer to the historical value, where Python is closer, and where they are equally close.
8. **Reading the results.** Python writes integers by truncation and C rounds, so truncation by itself lowers Python's bias by about 0.5 against a rounded product; that part is expected, and any bias beyond it is a real difference. The
   tree-cover layer is unaffected by the non-vegetated harmonization rules, so only the raw Python layer is used.

## Steps, with checks before anything expensive

| Step | What | Expected cost |
|---|---|---|
| 0 (**done 2026-09-20**, see "Step 0 result") | Value inventory of the historical `Percent_Tree_Cover` over all 63 tiles (histogram of values above 100: 200, 253, anything else), to fix the valid-value rule for the historical product before it is coded. Read only. | a few minutes |
| 1 (**done 2026-09-20**, see "Step 1 result") | Write the script and review it. | no compute |
| 2 (**done 2026-09-20, passed**, see "Step 2 result") | **Pilot on h09v05 and h12v02 only.** Check that (a) the C vs Python leg reproduces those two rows of `Compare/c_vs_python_63/per_tile.csv` exactly, (b) all three arrays are 4800 x 4800, and (c) the historical-versus-C correlation on the common-valid pixels is high enough to show the grids line up (a value near zero would mean a row or column offset, not a product difference). | about 1 minute |
| 3 (**done 2026-09-20**: 63 of 63 tiles, 07:40-07:56, rc=0) | Full run over 63 tiles, detached on forest203 (`setsid nohup`, status file and a `done` marker, parent PID 1), so a VS Code or SSH disconnect cannot stop it. | about 10-15 minutes (the earlier 63-tile C vs Python comparison took 15 minutes; this reads one layer per product instead of three, plus 63 small historical files) |
| 4 (**done 2026-09-20**) | Read the outputs, write the results document `Tree_Cover_vs_Historical_MOD44B_Results.md` in `Reports/`, add both documents to the map in `README_START_HERE.md`. | no compute |

## Step 0 result (run 2026-09-20 07:21-07:22 on forest203, 1 minute, rc=0)

Script `$RUN/code/validation/hist_value_inventory.py`, launcher `run_hist_inventory.sh` (detached, read-only on inputs); output `$RUN/Compare/hist_inventory/` (`inventory.csv`, `summary.txt`).

- All 63 tiles have exactly one historical file, all 4800 x 4800 uint8, and the layer `Percent_Tree_Cover` is present in each.
- Over the 63 tiles the historical layer holds only two kinds of value: **0-100 on 1,142,937,232 pixels and 200 (water) on 308,582,768 pixels**. The two add up to 63 x 23,040,000 = 1,451,520,000, so every
  pixel is accounted for. There is **no 253 and no other code above 100 in any tile**.
- **Valid-value rule for the historical product: 0-100 is valid, everything else (200) is water.** That is the same 0-100 rule the existing script uses for C and Python, so the three-way common-valid mask
  needs no special case for the historical layer. Unlike C (which writes 253 fill), the historical product has no fill pixels at all.

## Step 1 result (2026-09-20)

Written in `$RUN/code/validation/`: `compare_tree_cover_vs_historical.py` (the comparison) and `run_hist_compare.sh` (`pilot` or `full`, detached on forest203, status and done markers under `$RUN/status/`).
The original `compare_c_vs_python_all.py` was not changed.

- **Tested on synthetic data before any real file was touched** (in the container on forest203; the login nodes have no numpy). The test checked the statistics against brute-force numpy on a
  known case, exact pooling of two tiles against the statistics of their concatenation, the bin edges (0, 1-10, ..., 91-100), the 3 x 3 cross-table counts, and the whole flow with a tile whose
  products are missing (reported under MISSING, no crash). It also checked that the built-in comparison with an "earlier" per-tile file passes when the numbers agree.
- **The test found one real bug**: the cross-table function passed a 2-D array to `bincount`, which fails on a whole tile. Fixed (`.ravel()`) and retested.
- **The full run refuses to start** unless the pilot's `check_vs_existing.txt` says `RESULT: PASS`.
- **Two extra output files** beyond the list below: `closer_per_tile.csv` (per-tile counts of C closer, equal, Python closer) and, in the pilot only, `check_vs_existing.txt`.
- **Not yet exercised on real data:** the GDAL readers (`_read_sub`, `read_py`). Step 2 is the first time they touch the HDFs and GeoTIFFs; the pilot's check against the earlier results covers the C and Python readers.

## Step 2 result: pilot on h09v05 and h12v02 (2026-09-20 07:37-07:38, 47 s, rc=0)

Output `$RUN/Compare/tree_cover_vs_historical_pilot/`. Launched detached with `run_hist_compare.sh pilot`.

- **Check against the earlier C vs Python comparison: PASS.** The Python vs C leg reproduced all 11 fields (pixel counts, correlation, both means, bias, MAE, RMSE, within 5 and 10) of the earlier
  `per_tile.csv` rows for both tiles with a largest difference of exactly 0. The C and Python readers, the masks and the statistics therefore match the earlier, already reported analysis.
- **Grids line up.** C against the historical product on the common-valid pixels: correlation 0.979 (h09v05) and 0.926 (h12v02). A row or column offset would have given a value near zero.
- **Preview only, two tiles, not conclusions.** Common-valid, 43.0 million pixels: MAE against the historical product 2.69 for C and 3.92 for Python; correlation 0.952 and 0.923; C closer on 46.7% of
  pixels, equal on 30.0%, Python closer on 23.3%. Python is biased high at 11-30% historical cover (+4.5 and +3.3 in the 11-20 and 21-30 bins, C +1.8 and -0.3) and both are biased low above 60%,
  Python more (-11.0 at 71-80, C -6.0). h12v02 is worse than h09v05 for both products (MAE 4.16 C, 5.13 Python against 1.41 and 2.85).
- One wording fix made after the pilot: the summary's note about Python's truncation said a bias of about -0.5 "is expected". The pilot shows Python's bias is positive, so the note now says truncation
  lowers it by about 0.5 and any other bias is a real difference. Text only; no calculation changed.

## Outputs

In `$RUN/Compare/tree_cover_vs_historical_63/`:
- `per_tile.csv` (long format: tile, pair, mask, statistics);
- `mask_agreement.csv` (per tile counts and cross tables);
- `by_cover_bin.csv` (pooled bias and MAE per historical tree-cover bin, for C and for Python);
- `summary.txt` (pooled statistics, the ten best and worst tiles per product, the closer-product shares, and any missing or shape-mismatched tiles).

## Risks and things that will limit the conclusions

1. **Collection and date differ.** The historical files are Collection 6, produced in April 2020. Our C and Python runs use Collection 6.1 inputs and a model trained on the pooled samples of the 63
   tiles. Expect differences even for a perfect reproduction. The comparison ranks C and Python against the same reference; it does not certify either.
2. **Water and no-data masks differ**, as h12v02 already shows. Statistics on the common-valid pixels are cleaner but drop pixels near water and no-data, so the mask counts must be reported next to every
   number.
3. **Python has no georeferencing.** Alignment is by row and column, checked in step 2 only through the correlation test. If a tile's Python raster were shifted, the pilot would not show it (only two
   tiles). The full run's per-tile correlation column will show any outlier tile.
4. **The Quality and Cloud layers are not used** in the base plan. The historical Quality layer could flag low-confidence pixels; masking on it is a possible follow-up once its value meanings are
   read from the product documentation.
5. **No ground truth.** If C and Python both differ from the historical in the same direction, the cause is more likely the reference or the shared inputs than either implementation.

## Decisions

1. **Reference product: Collection 6 only (decided 2026-09-20).** The Collection 6.1 MOD44B-VCF product in `/css/modis/Collection6.1/` will not be used. Risk 1 above therefore stands as written.
2. **Headline mask: common-valid (decided 2026-09-20)**, with pairwise-valid alongside and pixel counts next to every number.
3. **Pilot tiles: h09v05 and h12v02 (decided 2026-09-20).**

## Out of scope

**The difference in valid pixels between Python and C is not investigated here** (README open item 2: C valid where Python is not on h12v01, h13v01, h21v01 and h11v02). The user will explore it separately.
This comparison only avoids being distorted by it, by using the common-valid mask as the headline, and records the mask counts so that the later work starts from data. It draws no conclusion about the
cause and does not use the historical product to decide which of C or Python has the better mask.
