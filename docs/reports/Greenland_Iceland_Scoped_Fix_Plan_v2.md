# Greenland/Iceland: zenith-only NDVI fix on the delivered product (plan v2, to-do item 11)

**Written 2026-10-01 evening. Status: plan only, awaiting approval. Nothing has run.**

This revises `Greenland_Iceland_Scoped_Fix_Plan.md` (2026-09-29), which was blocked on input data. That plan was written for bits 7-8 QA and three candidate models. Since then:
- bit-15 QA was adopted;
- RF was chosen;
- the delivered product is tree = MEDIAN, non-vegetated = MEAN, with SD layers (`Delivery_SD_RF_Median_Plan.md`).

This version tests the fix on that product only.

## Input data (checked 2026-10-01, read-only)

The MOD44C files are now in `/css/modis/Collection6.1/L3/MOD44B-VCF/dev/2019` and `/2020`. That is the pipeline's default input (`MOD44_DIR`), so nothing is copied.

| Tile | Area | MOD44CH / CQ for the 2019 window | Missing dates |
|---|---|---|---|
| h15v02 | SW Greenland | 23 / 23 | none |
| h16v00 | N Greenland edge, mostly Ellesmere (Canada) | 17 / 17 | 2019305-2020017 (polar night) |
| h16v01 | NW Greenland + Ellesmere | 21 / 21 | 2019337, 2019353 |
| h16v02 | SE/W-central Greenland, Reykjanes tip | 23 / 23 | none |
| h17v00 | N Greenland (Peary Land) | 17 / 17 | 2019305-2020017 (polar night) |
| h17v01 | E Greenland | 21 / 21 | 2019337, 2019353 |
| h17v02 | Iceland | 23 / 23 | none |
| h15v03 | Cape Farewell (tiny land area) | 0 / 0 | all; **excluded** |

The missing dates are polar-night composites. On h16v00 and h17v00 they remove **whole compositing periods**: the Day_2019305 window (305, 321, 337) and the Day_2019353 window (353, 001, 017) have no input at all. h16v01 ran with 2 missing dates, but a period with no input has not been tested in the Python pipeline. Step 0 tests that tile first.

## Method

Run root: `FullRuns/greenland_zenith_v2_<date>/`, new. Code: the frozen delivery commit (`delivery_sd_rf_median_20261001/code_commit_615557d`). Models: the delivered tree RF and non-vegetated RF.

0. **Automatic gate on h17v00** (polar-night tile, forest node, about 30-40 minutes):
   - `makeMetrics` (bit-15) completes and writes all metric files the two models need, with no NaN columns;
   - `vcfPredict` (`--tcAgg median --nvAgg mean`) completes;
   - the share of land pixels with valid predictions is reported.

   If `makeMetrics` fails on the empty periods, the run stops (`status/step0.failed`) and the cause is reported. It is not patched overnight.
1. **Zenith scan (read-only, minutes per tile):** the Stage 0 scan from `Zenith_Only_Fix_63Tile_Plan.md`, generalized from `h16v01_make_ndvi_zenith_only.py`, gives the share of pixels per period with no raw day where `0 < solz < 65`. A tile with 0% everywhere cannot change and skips steps 3-4.
2. **Real metrics and real prediction** for the other 5 new tiles: h15v02, h16v00, h16v02, h17v01, h17v02.
   - `makeMetrics`, then `vcfPredict` with the delivered settings.
   - h16v01 reuses its existing bit-15 metrics (`qaC_full63_retrain_20260929/Python/metrics_all/h16v01`) and the delivered map.
   - Forest201-205 for `makeMetrics` (1 per node, about 30 minutes per tile); ngaproc for prediction (about 42 GB per process).
3. **Zenith-only fixed prediction** for every tile flagged in step 1, including h16v01:
   - a per-tile `metricsDir` with the real metrics linked read-only;
   - the NDVI composite replaced by the zenith-only version (NDVI = 0 where no day in the period had `0 < solz < 65`);
   - every NDVI-derived metric removed from the links, so `Metrics` recomputes it from the fixed NDVI (the method of the 2026-09-28 driver, with its symlink bug fixed).
   - **Gate:** on h16v01 the fixed tree cover must fall, as in 2026-09-27 (mean 5.66 to 1.47 with the baseline). If it does not move, the run stops.
4. **Compare real against fixed per tile,** for tree cover and non-vegetated:
   - against C6, the primary reference;
   - against C6.1, with a caveat: its high-latitude rule sets non-vegetated = 100 and tree = 0 on v < 3 tiles;
   - against the h16v01 labels (all 0% tree);
   - measures: MAE, bias, mean cover, share of land with tree > 5 / > 10 / > 20, SD levels;
   - Greenland tiles and Iceland (h17v02) reported separately; Ellesmere pixels on h16v00/h16v01 reported separately using a longitude split.
5. **Report:** `Reports/Greenland_Iceland_Scoped_Fix_v2_Results.md`, and the README is updated.

**Success criteria (unchanged from v1):**
- **Greenland:** the fix moves every Greenland tile toward C6 (lower tree cover on ice sheet and tundra, lower MAE) and damages none.
- **Iceland:** adopted only if MAE against C6 does not get worse. This is a weaker test, because Iceland has no labels.

## Expected cost

About 3-4 hours unattended: gate about 40 minutes; 5 `makeMetrics` runs in parallel about 40 minutes; predictions about 30 minutes; fixed variants and comparisons about 1 hour. Forest nodes for metrics, ngaproc for prediction, after the rounding run (item 6) frees them.

## Still open for the project lead (from v1)

1. Tile-level scope (Ellesmere pixels in h16v00/h16v01 included) or a pixel-level Greenland mask. The results report Ellesmere separately to inform this.
2. Whether to include h16v00 (mostly Canadian). h15v03 is out (no data).
3. **Link to item 5:** all 7 tiles are inside the C code's v7 high-latitude box (h < 18, v < 3), so step 2's real predictions also feed the high-latitude part of `Bare_C_Regions_Investigation_Plan.md`.
