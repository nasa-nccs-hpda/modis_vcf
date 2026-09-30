# Airborne and spaceborne lidar as a tree cover reference: G-LiHT coverage and ICESat/ICESat-2 coincidence (2026-09-30)

**Goal (project lead):** there are no independent tree cover labels. The idea is to use airborne lidar (G-LiHT) as per-pixel truth, measure the
bias of spaceborne lidar (ICESat-2, ICESat) against it where both exist, and then use spaceborne lidar, with that error/bias
model, for near-global reference coverage. This note covers the first step: what is coincident. No bias model yet.

## G-LiHT on local storage
`/explore/nobackup/projects/ilab/data/GLiHT/`: **Alaska only**, 10 flights (July 2014 and July 2018), 1 m CHM + DTM GeoTIFFs (UTM).
`code/gliht_modis_coverage.py` samples every valid CHM cell on a 3 m grid, projects it to the MODIS sinusoidal grid, and gives per 250 m
pixel the lidar coverage fraction and canopy cover (share of cells with CHM >= 5 m, the MOD44B tree definition).
Run on forest204 (detached, one process per flight, ~16 min). Output `FullRuns/stable_pixels_20260930/gliht/`.

| Flight | MODIS tiles | px >= 50% | px >= 90% | px >= 90% by cover bin (0-10/10-30/30-60/60-100) |
|---|---|---|---|---|
| AK_10Jul2014 | h11v02 | 6124 | 889 | 624/191/59/15 |
| AK_11Jul2014 | h11v02 | 9941 | 1090 | 618/320/131/21 |
| AK_20140712_DoD1 | h11v02 | 1852 | 359 | 184/62/69/44 |
| AK_20140712_DoD2 | h11v02 | 943 | 293 | 12/53/92/136 |
| AK_20140713 | h11v02 | 2928 | 102 | 25/23/24/30 |
| AK_20140714_TIU | h11v02 | 4172 | 17 | 15/1/1/0 |
| AK_20140715 | h11v02 | 3578 | 14 | 7/6/1/0 |
| AK_20180721_FIA | h11v02 | 1417 | 67 | 53/7/7/0 |
| AK_20180728_FIA | h10v02, h11v02 | 1832 | 1 | 0/1/0/0 |
| AK_20180731_JBER_1 | h10v02 | 81 | 34 | 12/8/14/0 |

About 2,900 pixels are >= 90% covered, mostly on h11v02 (a training tile), skewed to low cover. Boreal only.

## ICESat-2 (ATL08 v007, local at /css/icesat-2/ATLAS/ATL08.007)
NASA CMR (public) bounding-box search per flight (`code/cmr_granules.py`): 2,707 ATL08 granules, 2,395 of them local (88%).
`code/is2_gliht_coincidence.py` (8 shards, forest204, ~3 min) keeps 100 m land segments whose centre is in a pixel with G-LiHT
coverage >= 50%. Output `spaceborne/is2_segments_in_gliht.csv`.

| Filter | Segments | Pixels | r(G-LiHT cover, raw photon canopy fraction) | r(G-LiHT cover, h_canopy) |
|---|---|---|---|---|
| all | 175,746 | 28,383 | 0.20 | 0.28 |
| strong beam, snow-free | 29,870 | 17,190 | 0.26 | 0.33 |
| + night | 7,993 | 6,718 | 0.26 | 0.43 |
| strong, snow-free, Jun-Sep | 21,037 | 13,634 | 0.28 | 0.35 |
| + pixel >= 90% G-LiHT | 2,087 | 1,298 | 0.37 | 0.35 |

71% of segments are flagged snow (the Oct-May passes). Strong, snow-free, Jun-Sep segments by G-LiHT cover bin: 12,559 / 4,590 / 2,805 / 1,083.
Dates 2018-10 to 2025-09.

**Why the correlations are low (expected at this stage, not a verdict on ICESat-2):**
1. A segment samples a 100 m x ~13 m strip, but it is compared with the whole 250 m pixel's G-LiHT cover. The next step
   is to compute G-LiHT cover on the segment's own footprint (the 1 m CHM allows it), which is the proper calibration unit.
2. The photon fraction is raw: no canopy/ground reflectance correction and no 5 m height threshold on photons.
3. Time gap: 2014 G-LiHT against 2018-2025 ICESat-2, in interior Alaska where fires are common. The 2018 flights are closest.
4. Geolocation (a few metres) matters at segment scale in patchy boreal cover.

## ICESat (GLAS, GLAH14 v034)
CMR: 6-179 granules per flight box, 2003-02 to 2009-10. **Not on local storage**, so it needs an NSIDC download (Earthdata login).
Footprints are about 65 m every 170 m along widely spaced tracks, so expect far fewer coincident shots than ICESat-2. The time gap
to G-LiHT is 5-11 years, which is usable only on stable pixels.

## What this means for the combined approach
- Coincidence is not the constraint. The constraint is that the local G-LiHT is boreal Alaska only. A bias model from it would
  apply only to open boreal forest. Temperate and tropical coverage needs more G-LiHT (CONUS campaigns: MD, NC, VA, WI, MA, ME, NH,
  NY; plus Puerto Rico and Mexico) from LP DAAC / glihtdata.gsfc.nasa.gov, and NEON AOP CHMs (UNDE, STEI, TREE on h11v04;
  HARV, BART on h12v04).
- Proposed next step: footprint-level G-LiHT cover for each strong, snow-free, summer segment (and night vs day as a factor),
  then fit the ICESat-2 vs G-LiHT relation and its error, first on the 2018 flights.

## Footprint-level comparison, Alaska (2026-09-30 ~08:40)
Code: `code/is2_footprints.py` (step 1: strong-beam, snow-free segments in G-LiHT pixels; five 20 m centres; ATL08 classified
photons counted as ground, canopy >= 5 m above ground, canopy < 5 m), `code/gliht_footprint_cover.py` (step 2: G-LiHT 1 m CHM
sampled every 1 m on each segment's own strip, along the 20 m centres +/- 10 m, 11 m wide), `code/fp_analysis.py` (step 3).
Chain `code/run_footprints.sh`, forest204, detached, 3 minutes. Output `FullRuns/stable_pixels_20260930/footprint/`, log `logs/fp3_analysis.log`.

Kept: 20,681 segments in 12,983 pixels (footprint >= 80% covered by G-LiHT, >= 20 classified photons). G-LiHT footprint cover
bins 0-10/10-30/30-60/60-100: 12,927 / 3,600 / 2,508 / 1,646. ICESat-2 cover = Nv / (Nv + k*Ng), with Nv = canopy photons >= 5 m and
Ng = ground + canopy < 5 m; k (canopy/ground reflectance ratio) fitted on half the granules and scored on the other half.

| Segments (test half) | k | r | bias | MAE | RMSE |
|---|---|---|---|---|---|
| night, raw (k = 1) | - | 0.64 | -5.5 | 10.4 | 18.3 |
| night, corrected | 0.60 | 0.68 | -1.6 | 9.7 | 17.5 |
| night, corrected, 2018 G-LiHT flights only | 0.60 | 0.77 | -2.7 | 9.2 | 16.2 |
| day, corrected | 1.15 | 0.62 | -2.2 | 10.5 | 17.6 |

- **Footprint matching helps a lot**: r goes from 0.26 (segment vs whole pixel) to 0.64-0.77.
- **The error depends strongly on cover**, and a single k cannot remove it. By G-LiHT cover bin, the corrected bias is +4.5 (0-10),
  -3.9 (10-30), -14.9 (30-60) and -32.5 (60-100). ICESat-2 compresses the range: it underestimates dense canopy and overestimates open canopy.
  A cover-dependent calibration curve is needed (a next step, not done).
- Time gap matters: at night, r is 0.77 with the 2018 flights and 0.66 with the 2014 flights.
- Height check (ATL08 h_canopy vs G-LiHT p98 on the same strip): r 0.38, bias +3.1 m. This is weak. Noise photons in h_canopy
  (especially by day) and short boreal canopy are the likely reasons. Not yet resolved.
- **MODIS pixel level:** mean ICESat-2 cover vs the whole-pixel G-LiHT cover has RMSE 17.6 (1 segment), 15.5 (2-3) and 15.1 (4+).
  G-LiHT on the same footprints vs the whole pixel has RMSE 10.8, 8.0 and 6.6. So the ICESat-2 measurement error, not the strip
  sampling, dominates.

**Reading:** after calibration, ICESat-2 could at best be a bias-corrected, per-bin aggregate reference, with about 15-18 points
RMSE per pixel. It cannot be per-pixel truth. Its biases are cover-dependent, so a calibration from boreal Alaska alone should not
be carried to other biomes. Separately, G-LiHT itself gives direct whole-pixel truth for about 2,900 pixels (>= 90% covered) on h11v02 and h10v02.
Those pixels can score the B and C models directly, after excluding training labels plus the 2-pixel buffer, since h11v02 is a training tile.

## G-LiHT as direct truth on h11v02: preliminary product scores (2026-09-30 ~09:20; the final run with C RF is in "Final h11v02 G-LiHT evaluation with C RF" below)
Code `code/gliht_eval.py`; logs `logs/gliht_eval_prelim*.log`; output `gliht_eval/`. Retrained-B h11v02 predictions were made for this
(`code/run_predict_B_h11v02.sh`, B metrics bound read-only). Pixels: G-LiHT coverage >= 90% on h11v02, not a training label
and >= 3 px from any label, valid in all 21 historical years, and valid in every product: **2,663 pixels**, almost all from the July 2014 flights.
**The comparison has a 5-year gap (2014 lidar vs 2019 products)**, which matters in interior Alaska.

**Tree height cutoff.** On the 78 G-LiHT pixels that ARE training labels (`code/gliht_vs_labels.py`), labels vs G-LiHT cover:
5 m cutoff r 0.80, bias +9.0; **3 m r 0.86, bias +0.1**; 2 m r 0.86, bias -4.2. So on this tile the labels behave like ">= 3 m" cover.
3 m is the primary reference below. Labels still read about 14% where lidar sees almost no canopy.

**Scores against 3 m G-LiHT cover (MAE / r):**

| Subset | HIST | B base | B RF | B XGB | C base | C XGB | Bold base | Bold XGB |
|---|---|---|---|---|---|---|---|---|
| all 2,663 | 19.9 / 0.52 | 19.1 / 0.58 | 19.6 / 0.59 | 18.9 / 0.59 | 19.9 / 0.54 | 20.0 / 0.52 | 19.0 / 0.58 | 18.8 / 0.59 |
| stable 379 (hist SD <= 5) | 11.3 / 0.80 | 12.3 / 0.77 | 13.5 / 0.76 | 12.2 / 0.78 | 12.6 / 0.76 | 11.8 / 0.78 | 12.6 / 0.76 | 12.1 / 0.77 |

- Every product, historical included, compresses the range against lidar: bias about +21 at G-LiHT 0-10 and about -27 at 60-100.
  Correlation within a bin is near zero above 10% cover. The products are further from lidar than the labels are (r about 0.55 vs 0.86).
- C minus B MAE (95% block bootstrap): all pixels baseline **+0.77 [+0.39, +1.18]**, XGBoost **+1.09 [+0.71, +1.48]** (C worse);
  stable pixels baseline +0.32 [-0.46, +1.08], XGBoost -0.41 [-0.98, +0.16] (no significant difference).
- Against the held-out training labels on the same tile, C was better (baseline MAE 7.24 vs 7.64). So on h11v02 the independent
  lidar and the labels disagree on the sign of the B-C difference. h11v02 is a boreal snow-season tile, where bit 15 was expected
  to act differently. The 2014-2019 gap and the product range compression both add noise.
- Final numbers with C RF: `gliht_eval/final_3m`, `final_5m`, run automatically by `code/run_gliht_eval_final.sh` (forest204, detached)
  when the bit-15 run's RF prediction of h11v02 appears. Marker `gliht_eval/status/final.done`.

## Fire and definition checks on the h11v02 offset (2026-09-30 ~09:40)
`code/fire_check.py`, log `logs/fire_check.log`. MCD64A1 C6.1 burned area, 2000-2019 (500 m, applied to the 250 m pixels).
- **Fire does not explain it.** Only 32 of 2,663 pixels burned after the lidar flights (2015: 11, 2018: 15, 2019: 6). 236 last burned
  before the flights, and 2,395 did not burn in 2000-2019. Without the 32 pixels, nothing changes: HIST MAE 19.96, B baseline 19.18,
  C baseline 19.93; C-B still +0.74 (baseline) and +1.08 (XGBoost).
- Historical MOD44B rises from 29.9 (2014) to 35.8 (2019) on the unburned pixels, while the lidar (2014) mean is 25.4. Part of the
  +10 overall bias may be real regrowth, or a trend in the historical product, over the 5-year gap.
- **Definition.** The MOD44B user guide defines tree cover as canopy cover: the share of skylight intercepted by trees, so gaps inside
  crowns count as open. Cover from a 1 m CHM (share of cells >= 3 m) is closer to crown cover, because it fills crown interiors. That
  predicts lidar > MODIS in dense stands, which is what we see: at lidar 60-100 (mean 76.4) every product is about 49 (ratio about 0.64,
  plausible crown porosity for boreal conifers). It does NOT explain the low end: at lidar 0-10 (mean 4.4) the products are about 25-26
  (the labels also read about 14 there). That points to short or sparse woody vegetation that the products count and the lidar does not,
  or to regression toward the mean in the models.
- A proper canopy-cover (gap fraction) reference from lidar needs the G-LiHT point clouds (first-return or all-return fraction above
  3-5 m), not the gridded CHM. Only CHM and DTM are on local storage.

## How much the products differ from each other, and what that means for decisions (2026-09-30 ~09:50)
On the 2,663 h11v02 lidar pixels (3 m), HIST and the B and C models all have MAE 18.9-20.0 and bias +9.9 to +10.8 against lidar, with the same
bin pattern (about +21 at 0-10, about -27 at 60-100). Their per-pixel errors correlate at 0.92-0.96. The products differ from each other by 4-6 points
on average. So almost all of the error against lidar is shared: it comes from the labels and tree cover definition, the MODIS inputs and the
2014-2019 gap, not from the model or the QA screen.

**Working principle (agreed with the project lead):** because the shared error cancels in comparisons, choices between models or screens can be
made on their differences, not on absolute accuracy. But a difference is only acted on when it has the same sign across independent references
(held-out labels, lidar pixels, stable historical pixels) and its interval clears zero. For the bit-15 screen the references currently
disagree: on the 63-tile held-out labels C is better, and on the h11v02 lidar (boreal) B is better by 0.7-1.1. Hold the decision until Stage 5b
and, if more G-LiHT is found, a second lidar region.

## Final h11v02 G-LiHT evaluation with C RF (ran automatically, ~10:00)
Logs `logs/gliht_eval_final_3m.log`, `_5m.log`; output `gliht_eval/final_3m`, `final_5m`. C RF: MAE 20.39, r 0.54 (3 m), in line with the others.
C minus B MAE, 3 m, all 2,663 pixels: baseline +0.77 [+0.39, +1.18], **RF +0.76 [+0.39, +1.13]**, XGBoost +1.09 [+0.72, +1.49]. C is worse for all three models.
Stable pixels (379): baseline +0.32 [-0.44, +1.11], RF -0.56 [-1.15, +0.05], XGBoost -0.41 [-0.94, +0.17]. None significant.
The 5 m results are the same in sign and nearly the same in size.

## Same tile, stable-pixel reference instead of lidar (2026-09-30)
`code/stable_ref_eval.py h11v02`, log `logs/stable_ref_eval_h11v02.log`. Reference = historical MOD44B C6 2000-2020 mean, on every h11v02 pool
pixel (no labels, >= 3 px from labels) meeting the option (a) rule: SD <= max(3, 10th percentile of SD in its 5-point sub-bin). 4,186,978 pixels
(79% at 0-10). MAE: hist 2019 1.81; B baseline/RF/XGB 2.97/2.68/2.50; C 2.74/2.55/2.36.
C minus B MAE: baseline -0.23 [-0.24, -0.22], RF -0.13 [-0.14, -0.13], XGBoost -0.14 [-0.15, -0.13]. **C is better** (tight intervals because of the
very large n; ties are common because values are integers).
So on h11v02: stable historical pixels and held-out labels favour C by 0.1-0.5; lidar (all pixels) favours B by 0.8-1.1; lidar on stable pixels shows no
significant difference. Caveat: this reference is the historical product itself (no aerosol screen, stable over time), which by construction
favours whichever product is closest to it.

## Decision (project lead, 2026-09-30)
Differences between models and QA screens of about 1 percentage point of tree cover (MAE) or less are treated as equivalent. All references
put B vs C within that range (labels, stable pixels, lidar), so B and C are considered equally accurate. Per `Aerosol_QA_C_Full63_Retrain_Plan.md`,
a tie favours C (bit 15), because B's bits 7-8 test is known to be wrong: it rejects 29-50% of good pixels due to PGE72's DOY-parity/BRDF bits.
The NEON evaluation (`NEON_Lidar_Eval_Plan.md`) is therefore no longer needed for the B/C choice. Its remaining value is describing absolute accuracy
and bias by biome (range compression, crown vs canopy definition, height cutoff).
