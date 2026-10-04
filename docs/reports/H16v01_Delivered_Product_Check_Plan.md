# h16v01 (Greenland snow tile): check of the delivered product (plan, to-do item 7)

**Written 2026-10-01 evening. Status: plan only, awaiting approval. Nothing has run.**

## Why

h16v01 is the known problem tile.
- Five of its 8 compositing periods are 100% unfiltered across the whole tile at 69-80 N, because of the zenith cutoff (`H16v01_TreeCover_OverPrediction_Investigation.md`).
- When bit-15 QA was adopted, its baseline model was +3.8 MAE against historical, worse than the other QA variants. That was flagged as a check before production.

The delivered product (RF, tree = MEDIAN, non-vegetated = MEAN, sum cap) has not been looked at on this tile.

## Method (read-only; no prediction needed)

The delivered h16v01 maps come from the 63-tile delivery run (`delivery_sd_rf_median_20261001/out/full63`, done marker `status/ALL.done`). This check starts automatically once that marker exists.

For tree cover, non-vegetated and non-tree vegetation:
1. **Products compared:**
   - delivered (RF, MEDIAN tree / MEAN bare);
   - RF MEAN (`rf_aggregation_maps_20261001/out/MEAN`);
   - the bit-15 baseline (`qaC_full63_retrain_20260929/Python/predictions_C/baseline`), the source of the +3.8;
   - the C code (2026-09-19 run).
2. **References:**
   - historical C6 (primary);
   - C6.1, with a caveat: the C6.1 high-latitude rule sets non-vegetated = 100 and tree = 0 across the tile;
   - the training labels for h16v01 (`v5.0.3samp`, all 0% tree);
   - the held-out 20% pixels where the tile has them.
3. **Measures:**
   - MAE, bias, mean cover;
   - share of land pixels with tree cover > 0, > 5, > 10 and > 20;
   - the sum-cap count;
   - SD levels (tree and non-vegetated SD mean / p95), and whether high SD marks the over-predicted pixels;
   - a map of tree cover > 10, for the report.
4. **Per compositing period:** the share of the tile with no valid zenith (from the existing Stage 0 scan, `Zenith_Only_Fix_63Tile_Results.md`). This links any over-prediction to the cause found earlier.

**Output:** `FullRuns/h16v01_delivered_check_<date>/`, a short report `Reports/H16v01_Delivered_Product_Check_Results.md`, and the README is updated.

**Decision it feeds:** whether h16v01 needs the Greenland-scoped zenith fix (to-do item 11, which runs it on h16v01 too) before production, or whether MEDIAN/RF already removed the problem.

## Expected cost

About 10 minutes on one forest node, after the delivery run ends.
