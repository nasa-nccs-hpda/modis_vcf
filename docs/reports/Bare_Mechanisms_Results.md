# Non-vegetated where C6 is 100: mechanisms, labels and pixel traces (results; steps 2-3 of `Bare_Where_C6_Is_100_Investigation_Plan.md`)

**Status 2026-10-04: steps 2 and 3 done.** Run `FullRuns/bare_mech_20261004`, 10:13-10:55, on ngaproc202 and forest201-205. Model:
the 60-tree non-vegetated RF (B60S0, `Bare60_Step0_Results.md`). Read-only: no product or model was changed.

C6, C6.1 and the C code are comparisons, not truth. The held-out labels (Landsat 30 m, 20 tiles, 2.16 M pixels) are the
independent check.

## Summary

1. **Where C6 is 100, the labels side with the new model, not with C6.**
   - On the 277,586 held-out pixels where C6 = 100, about a quarter of the labels are below 100.
   - MAE against the labels: 60 trees 1.95, v1.0.0 1.97, C6 6.04, C6.1 6.01, C code 8.12.
   - On labels of exactly 100 there, C6 is perfect (0.00) and the new model is close (0.19).
   - On labels of 61-99 there, C6 is 14.6 off and the new model 5.7.
2. **The project lead's theory (C6 tuned for fully bare pixels) fits every number here,** though it is not proven.
   - **At label = 100:** C6 is right (MAE 1.08 overall, 0.00 where C6 is 100).
   - **Every other label bin:** C6 is 11-18 off overall, where the new model is 3-7 off.
   - **By region:** the new model is closer to the labels than C6 in every region, for example Australia 3.96 against 15.74, and
     Africa 10-40S 4.00 against 18.94.
3. **The 90-99 band where C6 is 100 comes mostly from MEAN aggregation.**
   - In those pixels, about two thirds of the 60 trees predict 100: mean share 0.63 in desert, 0.67 on ice. The mean then lands at
     90-99.
   - Pixels at exactly 100 have 96-98% of trees at 100.
4. **Median beats mean on the held-out labels.**

   | Label bin | Mean (delivered) | Median | 75th percentile | 90th percentile |
   |---|---|---|---|---|
   | All | 4.15 | **3.77** | 4.91 | 7.96 |
   | 0-20 | 3.24 | **2.51** | 4.42 | 8.42 |
   | 21-40 | **6.55** | 6.86 | 8.64 | 12.87 |
   | 41-60 | **5.79** | 5.81 | 6.52 | 9.44 |
   | 61-99 | 5.75 | 5.40 | **5.18** | 6.80 |
   | 100 | 0.62 | 0.30 | 0.09 | **0.03** |

   - **Median:** better overall (−0.38) and at both ends; slightly worse at 21-40 (+0.31), even at 41-60.
   - **Effect on the maps' C6 = 100 pixels** (equal sample per tile): exact 100s go from 14% with the mean to 50% with the median.
     MAE against C6 goes from 14.6 to 12.4, and on Greenland from 5.0 to 2.0, with 81% at exactly 100.
   - **Against C6 where C6 < 95,** the median is worse (MAE 12.0 against 9.9). That matches the earlier result in to-do 10, and in the
     light of point 2 it says more about C6 than about the median.
   - **Upper percentiles** (75th, 90th) fix the 100s but cost too much lower down.
5. **The sum cap plays no part.**
   - Where C6 is 100, the cap cuts bare on 0.01% of pixels or fewer, and predicted tree cover there is about 0.
   - It acts mainly on Asia 10-30N pixels with C6 < 95: cut on 0.12%, by about 6 points.
6. **QA on ice and Arctic pixels: no input passes the pipeline's QA.**
   - On every traced ice and Arctic pixel (including the controls at 100), 0 of 23 inputs pass the QA mask. All 8 periods use the
     all-input fallback mean, cloudy inputs included.
   - **Reasons:**
     - bit 15, which the mask requires to be set, is unset on about 22 of 23 inputs;
     - solar zenith is outside 0-65° on about 14 of 23 inputs;
     - the cloud state is set on 11-23 of 23.
   - **Bit names (corrected 2026-10-04):** the MOD44C state word is repacked by PGE72 (see the `mod44c-state-bits` skill and
     `MOD44C_PGE72_Investigation/Reports/MOD44C_State_Aerosol_Bit_Collision.md`).
     - The trace columns `snow_mod35` (bit 12) and `internal_cloud` (bit 10) are not snow or cloud in MOD44C: bits 9-11 hold
       `day % 16`, and bit 12 is unused. `bit6_aerosol` is **cloud adjacency**.
     - The snow flag is **bit 5**. On the traced ice inputs it is set on 55%, and every one of those has bit 15 = 0. PGE72's QC
       requires "not snow", so ice fails bit 15 by construction. This agrees with the earlier aerosol-QA work (h16v01: under 8%
       pass any variant); it is not new.
   - **Training is consistent with this:** the ice labels (h16v01) are fallback-only too. So it does not explain gap against control
     pixels, but it does mean the ice predictions rest entirely on cloud-including means.
   - **Desert and Australia:** about 22 of 23 inputs pass and there is essentially no fallback, so QA is not the issue there.
7. **The straight edge on h16v02 (row 651/652) is a polar-night input boundary.**
   - In the December input (DOY 2019353), pixels above the edge have solar zenith 0 (no data). Just below, some pixels have a twilight
     observation at solz 93° (band 7 = −28672 fill, band 31 clamped to 16000); the "about 37°" was the mean of 0, 0, 0, 93 and 93 (corrected 2026-10-04).
   - Because every ice period is a fallback mean, those values enter the 2019353 period composite: band 7 is about 1,900 lower and
     band 31 about 2,100 higher below the edge.
   - Below the edge, the share of pixels at least 5 points under C6 jumps from 3% to 11%.
   - This is the known `solz <= 0` unreliable-data edge case, here showing up as a sharp line.
8. **Australia is not a QA or C6 = 100 problem.**
   - C6 is at or near 100 on less than 0.2% of land.
   - The traced low pixels (new map about 27 against C6 about 59) pass QA like their controls.
   - The labels side with the new model: 3.96 against C6's 15.74.
   - The training labels there are mostly 0-20 (987,490 of 1,514,006); only 5,716 are 100.
9. **Pixels below 90 where C6 = 100 are mostly not fully bare by the labels or land cover.**
   - **Arctic islands and North America 50-80N:** 74% are MCD12Q1 grassland. Here C6.1 (81.9) and the C code (42.3) are far from the
     labels; the new model is at 6.0.
   - **Desert < 90:** 84% barren in MCD12Q1.
   - **Small areas:** h11v10 (Atacama) and h19v10/h19v11 (Namib) have only about 9,000-23,000 C6 = 100 pixels each. The equal
     per-tile sample overweights them.
10. **Out of training range:** C6 = 100 pixels are often outside the training data's 1-99 percentile range on at least one
    predictor, but controls at 100 are too (57% against 67% for the 90-99 band; Greenland 75-97%). It does not separate gap pixels
    from controls.

## What this suggests (for the project lead; nothing adopted)

- **MEDIAN for non-vegetated** is the strongest candidate from this work.
  - It is better on held-out labels overall and at 100, and puts about half the C6 = 100 pixels at exactly 100 (81% on Greenland).
  - It is slightly worse at 21-40.
  - It ties into to-do 10: MEAN is provisional.
- **No post-model rule is supported by the labels where they exist.**
  - The C6 = 100 areas the labels cover say C6 is too high for about a quarter of its 100s.
  - Rules would help agreement with C6, not accuracy against the labels.
  - The v00 rule is already in place where the models have no usable input (polar night).
- **The ice-sheet QA**: no QA-passing input on ice, and invalid polar-night values entering fallback means. These are
  input-handling questions to raise separately.
  - **Option:** treat `solz <= 0`, or zenith outside the cutoff, as missing in the fallback mean as well, not only in the QA mean.
  - **Caution:** this would change the inputs, so it needs a retrain to be consistent.
- **The calendar-shift caveat** for the static models (C6) still stands. Nothing here tests it.

## Checks and failures

- **Self-check:** on all 67 non-v00 tiles, the rounded, sum-capped mean of the 60 trees equals the 60-tree map at every sampled
  pixel. The 2 v00 tiles are set to 100 by the high-latitude rule.
- **Held-out:** the mean of the trees reproduces bare60's held-out B60S0 exactly. The orientation check gives 0.9947 (valid map
  pixels).
- **Traces:** the period composite equals the truncated mean of its QA-pass inputs ((636 + 604 + 577) / 3 = 605.7, stored 605).
- **Failures, all fixed, with attempts kept in `status/attempt*`:**
  1. Held-out keys were read as row = y; they are row = x. The script's own check stopped it.
  2. The v1.0.0 split has no training keys, so the MC rerun's identical bare split is used.
  3. `d.map` is the pandas method; now `d['map']`.
  4. Traces tried to cache inputs into the read-only Greenland metrics folder; they are now computed without caching.
  5. The first h16v02 edge search found a coastline; the edge was found from the share of pixels at least 5 points under C6.

## Files

`FullRuns/bare_mech_20261004/out/`:
- `tiles/` (per-tile samples with all 60 tree statistics and 21 predictors);
- `summary/` (`strata_by_region.csv`, `cap.csv`, `trees100.csv`, `igbp_gap.csv`, `range_check.csv`);
- `labels/` (`labels_vs_refs.csv`, `train_labels_by_*.csv`, orientation check);
- `agg_heldout/agg_heldout.csv`;
- `trace/<tile>/{days,periods}.csv` and `trace_summary_*.csv`.
