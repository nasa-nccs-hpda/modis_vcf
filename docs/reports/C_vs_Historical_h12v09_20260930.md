# Bit-15 (C) models vs historical MOD44B on h12v09, by tree cover bin (2026-09-30)

**Question (project lead):** in dense forest, do the C models now predict less tree cover than the historical MOD44B?
**Answer: no meaningful systematic reduction.** On average C is about 1.3-1.8 points lower where the historical value is >= 60. Most of that is
regression to the mean from binning on the historical value: binned on C's own value, the difference in dense forest is 0 to +0.6. Where both are
>= 60 it is -0.2 to -0.8. The tile mean is unchanged (HIST 63.63; C 63.54-63.67).

Code `FullRuns/stable_pixels_20260930/code/c_vs_hist_bins.py`, `c_vs_hist_binby_c.py`; logs `logs/c_vs_hist_h12v09.log`,
`logs/c_vs_hist_binby_c_h12v09.log`; tables `c_vs_hist/`. HIST = C6 2019 (standard folder). C = qaC run Stage 5b predictions. Bold = the earlier
B predictions. 22,254,126 pixels valid in all products; HIST bins 0-10/10-30/30-60/60-100 = 0.11M / 2.55M / 3.72M / 15.88M (71% dense).

## Binned by the historical value (product minus HIST)
| HIST bin | C baseline | C RF | C XGBoost | Bold baseline | Bold RF | Bold XGBoost |
|---|---|---|---|---|---|---|
| 0-10 | +8.7 | +8.2 | +3.2 | +10.2 | +9.5 | +4.9 |
| 10-30 | +7.0 | +5.6 | +2.3 | +6.9 | +5.9 | +3.1 |
| 30-60 | +2.8 | +2.8 | +3.5 | +2.7 | +2.5 | +3.0 |
| 60-100 | **-1.8** | **-1.7** | **-1.3** | -2.5 | -2.3 | -2.3 |

- Dense (HIST 60-100): median -1 for all. Pixels lower by > 10: C 11.0 / 8.4 / 4.5% (Bold 14.4 / 11.4 / 9.8%). Higher by > 5: 11.9 / 8.9 / 7.0%.
- The largest deficit is at HIST 70-80: C -1.7 to -2.7 (Bold -2.7 to -3.6). At 80-90 it is -1.1 to -2.5.
- No saturation: 99th percentile in dense forest is C 85-86 vs HIST 83.
- C is **less** low than the earlier B predictions in dense forest, by about 0.7-1.0 points.

## Binning artifact check (binned by C's value instead)
C minus HIST in C's 60-100 bin: baseline +0.63, RF +0.01, XGBoost +0.03. Where both HIST and C are >= 60: -0.22 / -0.67 / -0.77.
The sign flips with the binning variable, which is what regression to the mean produces. So the dense-forest "deficit" and the low-end "excess"
seen when binning on HIST mostly reflect the two products disagreeing pixel by pixel, not C being biased low in forest.

## Training labels on the tile (in-sample for the models; a check only)
In HIST 60-100, the label pixels (n = 49,820) have mean label 85.0, HIST 79.6, C 81.1-83.1. MAE vs label: HIST 7.5, C 3.9 / 3.9 / 6.4.
So where it can be checked, the historical product is the one that reads low in dense forest here, not C.
Caveat: the historical product had no working aerosol screen, and h12v09 (Amazon) is a tile where high-aerosol pixels are common (5-16% in the MOD44C study).

## Pixel-level range in HIST 60-100 (15,876,458 pixels; `code/c_vs_hist_dense_range.py`, log `logs/c_vs_hist_dense_range_h12v09.log`)
| C minus HIST | min | p1 | p5 | p25 | median | p75 | p95 | p99 | max | pixels <= -30 | pixels >= +20 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| baseline | -73 | -27 | -16 | -5 | -1 | +3 | +9 | +14 | +30 | 103,010 (0.65%) | 17,774 |
| RF | -73 | -23 | -13 | -5 | -1 | +2 | +7 | +12 | +29 | 37,645 (0.24%) | 7,261 |
| XGBoost | -80 | -20 | -10 | -4 | -1 | +2 | +7 | +11 | +29 | 24,472 (0.15%) | 6,349 |
The distribution is skewed: the low tail is longer than the high tail. A small set of dense pixels drops sharply under C (most in the baseline).
Where these pixels are, and why, is not yet examined (candidates: clearing after the historical training period, cloud/aerosol-screened composites, edges).

## Why some dense pixels drop sharply under C (2026-09-30 ~11:00)
Code `code/drop_diagnosis.py`, `code/drop_followup.py`; logs `logs/drop_diag_h12v09.log`, `logs/drop_followup_h12v09.log`; per-period table
`drop_diag/drop_diagnosis_h12v09_periods.csv`; mask `drop_diag/drop_mask_h12v09.npy`. Retrained-B predictions for h12v09 were made for this
(`code/run_predict_B_tile.sh h12v09`). Drop = HIST 60-100 and C baseline <= HIST - 30 (103,011 px). Control = 200,000 random dense pixels with |C - HIST| <= 3.

**Conclusion: not caused by the bit-15 QA change.** They are persistently cloudy pixels, and every Python model (B as well) drops there.
- **B drops too, and more often.** Dense pixels at <= HIST - 30: retrained B vs C = baseline 111,179 vs 103,011; RF 94,101 vs 37,646; XGBoost 54,935 vs 24,473.
  At the C-baseline drop pixels, retrained B is also about 20-22 below HIST (C RF/XGBoost are within 0.4-1.4 of B there). Only C baseline goes further (-13.8 vs B baseline).
- **Cloud fallback.** When no day in a composite passes QA, the code averages all days, cloud included. Drop pixels fall back in 5.1 of 8 periods under C
  (6.0 under B), against 2.8 (4.1) for controls. In 5 of the 8 periods, 70-96% of drop pixels are fallback composites. Their wet-season composites are very bright in red
  (e.g. period 2019353: 1,848 vs 608 for controls) and low in NDVI (0.40 vs 0.74), i.e. cloud-contaminated. **C has fewer fallbacks than B at these pixels**
  (only 0.1% have more under C), and in the dry-season period (2019257) C's fallback share is 43% vs B's 83%, with higher NDVI (0.74 vs 0.69).
- **Where:** eastern Amazon, Pará (block centres about 2.5-4.5 S, 51.5-53.6 W; top 6 blocks = 2.5-4% of each block, 36% of drops in the top 5).
  This is a very cloudy region with active clearing. Not burned in 2019 (0.1%), not at tile edges (0.4%).
- **The historical product is unstable there too:** at the drop pixels HIST was 66.4 (2017), 63.3 (2018), 72.1 (2019), 70.6 (2020), against 73-77 for controls.
- Model dependence: RF and XGBoost drops overlap the baseline drops only 20% / 17%. The bagged-tree baseline produces the most extreme values.

**Open question (not QA):** why the operational C-code product (HIST) is less affected than the Python models on persistently cloudy pixels.
This looks like a compositing or metric difference in cloud-fallback composites between the C code and Python (a parity issue), and would be the next thing to check.

## Area for exploration: better models in persistently cloudy regions (project lead, 2026-09-30)
Cloud-fallback composites are a model-quality opportunity for all variants, not a QA-screen issue. B and C are affected alike, and C slightly less.
Improving how the Python pipeline handles pixels with few or no good-QA days should make the models better overall in cloudy areas
(the Amazon, Congo, Southeast Asia and other humid tropics, and cloudy mountain and coastal regions). Starting points:
- Compare the C-code and Python compositing and metrics on cloud-fallback composites. The operational historical product is less affected here, so
  the difference may show a better fallback rule.
- Alternative fallbacks: drop the fallback composite instead of averaging all days (let metrics use the remaining periods); use the
  least-cloudy or darkest-red day instead of the mean; take the fallback from adjacent periods or years.
- A per-pixel count of good-QA periods as a model input or confidence layer, so the models can learn to discount contaminated composites.
- Test on h12v09 (the Pará blocks above) and at least one other cloudy tropical tile, scored against held-out labels and stable pixels, using the
  equivalence rule (a change of <= ~1 point is a tie).
Not started; to be planned separately.
