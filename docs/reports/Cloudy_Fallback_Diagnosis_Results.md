# Cloud-fallback diagnosis: stage A results (steps 1, 2 and 4)

**Written 2026-09-30 ~15:30. Stage A of `Cloudy_Fallback_Diagnosis_Plan.md`. Stage B (steps 3, 5 and 6) waits for the project lead.**
Run root: `FullRuns/cloudy_fallback_20260930/` (code in `code/`, logs in `logs/`, tables in `out/prevalence/`). No product code changed, no models trained.

## Answer in brief

1. **The C code has no better fallback rule.** C and Python composite in the same way: the mean of the good days, or, when no day is good, the mean of all days (clouds included). C's per-period QA and cloud bytes are written only as output layers (`Quality`, `Cloud`) and never change a prediction.
2. **The current C code is not less affected.** The 2026 C-code rerun of h12v09 (bits 7-8 screen) is on average 18.5 points below the historical product at the drop pixels, and 17.9% of them are 30 or more points below. On all dense pixels it moves further from the historical product than RF (bit 15) does once a pixel has 3 or more fallback periods (at 8 periods: C code -15.6, RF -8.3).
3. **Only the historical MOD44B (C6) is less affected.** Its own `Quality` byte shows that it fell back less often at the drop pixels: 4.1 of 8 periods in 2019, against 5.1 for Python bit 15 and 6.0 for the 2026 C code. At the control pixels the counts are about the same (2.6 against 2.8). The rest of its advantage is the model: a different model, with different training and C6 inputs. It is unstable there as well: 63-72 across 2017-2020 at the drop pixels, against 73-77 for the controls. The drop set was selected on the Python baseline's own error, so comparing on it favours the historical product.
4. **Fallback is widespread and costs accuracy in dense forest.** Outside the high latitudes (v04 and below), 3.7% of land pixels have 5 or more fallback periods and 10.8% have 3-4. On dense forest (historical >= 60), RF's MAE against the historical product rises from 4.7-5.2 at 0-4 periods to 8.6 at 5 or more (bias -6.8), within the tile on 17 of 20 tiles.

**Recommendation:** skip the full pixel trace in step 3. The rules are the same in both codebases, and the historical product's `Quality` byte already shows where the difference lies. Go on to steps 5 and 6. The fix is Python's own (a fallback rule or a quality layer), not parity with C. Details and the two side findings follow.

## Step 1: Python fallback mechanics

- Each composite is 48 days: three 16-day MOD44C inputs (`CompositeDayFile.py:30`). There are 8 periods per year (2019065 ... 2020033).
- The rule is in `model/CompositeDayFile.py:272`: `comp = np.where(goodComp.mask, allComp, goodComp)`. `outQa` (line 275) = 1 where the composite fell back. A missing input day becomes 10000 with QA 0 and enters the all-day mean. C does the same.
- **`outQa` and `outCloud` are not used anywhere downstream.** They are saved as `*-qa.npy` and `*-cloud.npy` (line 202). Their only readers are `getQaMask()` and `getCloudMask()`, and nothing outside the tests calls those. `Metrics.py:209` reads only `cdf.raster()`. The delivered branch's `CompositeDayFile.py` is identical to `Python_code`.
- **Side finding: `outCloud` is inverted.** The input cloud mask is 1 = clear (`ProductTypeMod44.py:105`), so `cloudTotal == 3` (line 285) flags composites whose three days were **all clear**, not all cloudy as the comment says. C's `Cloud` bit means "no clear day in a fallback period" (`make_metrics.c:97-102`). This has no effect on the product today, because the flag is unused. It must be fixed before the flag can serve as a predictor or a quality layer.
- **Exposure of the 64 RF predictors** (the list is from `models/C/*/pcttreeTopN.bin`). The sorted families rank high to low (`CMirrorMetrics.py:10-24`). Cloud-contaminated periods have low NDVI and a cold Band_31, so they sort to the bottom.

| Group | Count | Exposed when |
|---|---|---|
| Per-period values (`UnsortedMonthlyBands-*`) | 24 | that period falls back |
| `CUns*` (means and amplitudes in calendar order) | 14 | any of their periods falls back |
| `CGreenMax`, `CGreenMean13`, `CThermMean13`, `CSortedRank0/1-Band_31` (top 1-2 ranks) | 13 | only when 7-8 periods fall back (robust) |
| `CGreenMean16/46`, `CGreenAmp15`, `CThermMean16` (ranks 3-4) | 8 | 5 or more periods |
| `CGreenMean19`, `CThermMean19` (ranks 5-6) | 5 | 3 or more periods |

So 38 of the 64 predictors are hit directly by any fallback period, and 13 are robust unless nearly the whole year falls back.

## Step 2: C fallback mechanics

- `MOD_PR44B1/fix_bands.c:181-205`: for each of the 23 16-day inputs, a QA flag is set (cloud = 0, no shadow, aerosol bits 7-8 != 3, no adjacency, zenith < cutoff) and a cloud flag (cloud = 0, zenith < cutoff). A missing input becomes 10000 with QA 0.
- `MOD_PR44B1/make_metrics.c:64-104` (shared by MOD44B1 and MOD44B3): for each band and period (3 inputs; the last period has 2), the value is the mean of the QA-good inputs. If there are none, it takes the mean of all inputs, sets `QA` bit n, and sets `cloud` bit n when no input is clear. This is the same rule as Python.
- `MOD_PR44B3/MOD44B3.c:360, 400, 520-523`: `QA` and `cloud` go only to the `Quality` and `Cloud` output fields. The prediction does not use them.
- The one known compositing difference is NDVI. C averages the per-input NDVI; Python computes NDVI from the averaged bands. This is documented and accepted (`C_vs_Python_Code_Differences.md`, item 4). In a fallback period with bright clouds, Python's ratio of means is pulled further toward cloud NDVI than C's mean of ratios. This affects only the NDVI predictors, and the C code drops anyway (see below), so it does not explain the gap.

### h12v09: C code and the historical product at the drop pixels
`code/c_code_at_drops.py`, `code/hist_quality_at_drops.py`, and their logs. Drop set = the existing mask (103,011 px). Controls = the same 200,000-pixel sample rule as `drop_diagnosis.py`.

| h12v09, 2019 | Historical | C code 2026 (bits 7-8) | Python RF (bit 15) | Python baseline (bit 15) |
|---|---|---|---|---|
| Tree cover at drop pixels | 72.1 | 53.6 | 49.9 | 37.3 |
| Tree cover at controls | 76.7 | 74.2 | 75.8 | 76.7 |
| Fallback periods at drop pixels (of 8) | 4.12 | 6.03 | 5.13 | 5.13 |
| Fallback periods at controls | 2.63 | 4.06 | 2.77 | 2.77 |

Historical fallback at the drop pixels in other years: 4.21 (2017), 5.01 (2018), 3.75 (2020). The historical value drops in the years with more fallback (63.3 in 2018).

All dense pixels (15.9M, historical 60-100), by the C code's own fallback count:

| C fallback periods | 0-2 | 3 | 4 | 5 | 6 | 7 | 8 |
|---|---|---|---|---|---|---|---|
| Pixels | 1.65M | 3.00M | 4.05M | 3.62M | 2.27M | 1.04M | 0.25M |
| C code minus historical | -0.1 to -0.4 | -0.7 | -1.5 | -3.3 | -5.8 | -9.1 | -15.6 |
| RF minus historical | -0.8 to -1.8 | -0.6 | -0.6 | -1.5 | -3.6 | -5.8 | -8.3 |

Why the historical product fell back less often at the drop pixels is not resolved. Its aerosol read rejected nothing, but on h12v09 bit 15 removes only 2.1% of the pixels that pass the other tests (`Aerosol_QA_Variant_Test_Plan.md`), which is too little to explain a whole period. The likely reason is its C6 MOD44C inputs, a different PGE72 version with different state bits and compositing. Checking this would mean reading the C6 MOD44C state for h12v09. That is an optional, narrowed step 3.

## Step 4: prevalence on the 63 tiles, and the link to accuracy

`code/fallback_prevalence.py`, run on ngaproc201 (about 30 s). Tables: `out/prevalence/fallback_per_tile.csv` (per tile: mean count, shares of 0, 1-2, 3-4 and 5+), `rf_error_by_fallback_per_tile.csv`, `rf_error_by_fallback_pooled.csv` and `stratified_summary.txt`. The fallback count is the sum of the 8 `*-Band_1-qa.npy` from the bit-15 build (3 tiles from `aerosol_qa_variants_20260929`). Check: h12v09 drop pixels give 5.13 (the report said 5.1). The test keys were confirmed as x = row, y = col (label MAE 7.1, against 15.0 the other way round).

**Where.** At v01-v03 the fallback is structural (solar zenith cutoff and polar night; h16v01 7.8, h12v01, h13v01 and h21v01 all 5+) and is already covered by the Greenland and Iceland work. Among the other tiles, the highest mean counts are:
- h12v04 2.9
- h11v08 2.4 (16% of pixels at 5+)
- h24v04 2.4
- h09v04 2.2
- h11v04 2.2
- h26v06 2.2 (12% at 5+)
- h19v09 1.9 (14% at 5+)
- h19v08 1.5 (11% at 5+)

h12v09 has a mean of 3.3, with 24% at 5+. The worst tiles are in the humid tropics (Amazon, Congo, Southeast Asia) and at mid-latitudes in winter (v04, where snow and low sun raise the count). Arid tiles are close to 0.

**Accuracy** (RF bit 15, tiles v04 and below; pooled MAE / bias):

| Fallback periods | 0 | 1-2 | 3-4 | 5+ |
|---|---|---|---|---|
| Share of land pixels | 56.2% | 29.3% | 10.8% | 3.7% |
| Vs historical, all land | 3.09 / +0.2 | 4.63 / -0.2 | 5.13 / +0.4 | 7.64 / +2.2 |
| Vs historical, dense (>= 60) | 5.66 / -3.3 | 4.70 / -2.4 | 5.17 / -2.4 | **8.60 / -6.8** |
| Vs held-out 20% labels | 6.06 / -0.3 | 8.19 / -0.3 | 9.50 / -1.3 | 8.91 / -1.1 |
| Label pixels | 1.00M | 0.59M | 0.20M | 0.04M |

Within tiles, 5+ is more than 1 point worse than the better of 0 and 1-2:
- dense forest against the historical product: 17 of 20 tiles
- held-out labels: 11 of 25 tiles
- all land against the historical product: 22 of 44 tiles

**Reading.** The dense-forest signal is strong and consistent: fallback-heavy pixels are biased low by about 7 points. Against the labels, error rises with the count up to 3-4 periods, but this is confounded, because cloudy pixels are also more often in mountains, on coasts and in mosaics where labels are harder. The historical product is not an independent reference here (item 3 above). The pixels at stake are about 14% of land (3 or more periods), concentrated in tropical forest. That is where the gain lies.

## For stage B

- **Step 3:** do not do the full C-vs-Python pixel trace. Optional, narrowed version: compare C6 and C7 MOD44C state and QA pass rates at the h12v09 drop pixels, to see why the historical run fell back less often.
- **Step 5:** search as planned. Note that the container has scikit-learn 1.2.0, and NaN-aware random forest splits need 1.4 or later. "Fallback as missing" therefore needs either a newer container or imputation.
- **Step 6:** candidates to rank:
  - the fallback count (0-8, free: the `*-qa.npy` files already exist);
  - the per-period `outQa` flag;
  - a corrected all-cloudy flag (fix the inversion first);
  - the share of good days per period;
  - a cloud-contamination score;
  - the neighbouring-year value.
  
  A variant of the fallback rule should also be tested: use only the robust sorted predictors, or replace a fallback period with the pixel's median over its good periods.
- **Tiles for option 2:** h12v09, h11v08 (northern South America), h19v09 (Congo), h26v06 or h27v06 (Southeast Asia), plus one v04 winter tile (h12v04) as a check for the mid-latitude case.

## Addendum (2026-09-30 ~16:00): 2-input periods, and the h12v09 training labels

Code: `code/period_length_and_labels_h12v09.py`. Log: `logs/period_length_and_labels_h12v09.log`. Label table: `out/h12v09_training_labels.csv`. The QA is the pipeline's own mask (`ProductTypeMod44.createQaMask`, bit 15, zenith 65) on the 23 MOD44CQ inputs. The recomputed 3-input fallback matches the stored `qa.npy` on 100% of pixels.

**3 inputs (8 periods) against 2 inputs (12 periods; the last is 1 input), h12v09:**

| Pixels | Clear inputs (of 23) | Fallback share, 3 / 2 | Good periods, 3 / 2 | Share with <= 2 good periods, 3 / 2 | Share with >= 4 good periods, 3 / 2 |
|---|---|---|---|---|---|
| All land | 9.0 | 0.41 / 0.48 | 4.7 / 6.2 | 0.094 / 0.060 | - |
| Dense | 9.3 | 0.39 / 0.46 | 4.9 / 6.4 | 0.052 / 0.033 | 0.83 / 0.91 |
| Drop | 5.2 | 0.64 / 0.69 | 2.9 / 3.7 | 0.40 / 0.23 | 0.21 / 0.51 |
| Control | 9.9 | 0.35 / 0.43 | 5.2 / 6.9 | 0.023 / 0.015 | - |

With 2 inputs, a larger share of periods falls back, so the calendar-order predictors see more contaminated periods. It also gives more clean periods, because clear inputs are no longer averaged together: at the drop pixels, the share with 4 or more clean periods goes from 21% to 51%. That helps the ranked predictors (greenest and warmest), which draw on the top periods. The net effect on the model is unknown without a test, and 12 periods would mean redefining the C-mirror metrics and the predictor selection.

**h12v09 training labels:**
- 24,860 pixels (19,863 train and 4,997 test under the split rule); mean 76.8, median 86.
- 0: 0.6%. 1-9: 2.6%. 10-19: 1.7%. 20-29: 1.8%. 30-39: 2.0%. 40-49: 2.8%. 50-59: 4.1%. 60-69: 4.9%. 70-79: 6.4%. **80-89: 55.3%. 90-99: 17.9%.** 100: 0.
- On the tile's land, the historical product and RF put 39% and 38% in 70-79 and only 18% and 16% in 80-89. So the labels sit about one bin higher than both maps, which is the known range compression.
- **Low-label pixels are the most fallback-heavy:** a mean of 5.8 fallback periods at label 0 and 5.3 at 1-9, against 2.4-3.0 at 50-99. The training data therefore pairs "cloud-contaminated signature" with "low tree cover", which may be part of how the models learn to drop cloudy forest. This is a hypothesis, not tested.
- Rows by fallback count (0-8): 640, 3,658, 6,135, 6,674, 4,466, 1,852, 536, 395, 504.
