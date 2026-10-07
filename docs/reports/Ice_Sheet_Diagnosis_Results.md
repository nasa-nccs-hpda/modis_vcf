# Why v1.1.0 gives the Greenland ice sheet less than 100 bare: diagnosis results

**Status 2026-10-05 late evening: done (read-only).** Run `FullRuns/EXP_icesheet_diag_20261005` (EXPERIMENTAL), 21:12-21:40 on forest202,
forest203 and forest205. Plan: `Ice_Sheet_Diagnosis_Plan.md` (E1 approved). Nothing was refit, and no product file changed.

**Pixel selection.** "Ice" below means MCD12Q1 2019 class 15, largest connected region per tile. MCD12 is a **reference used to select pixels,
not a definitive boundary of the ice sheet**; one finding below is that some of its class-15 pixels are probably not ice.

**Groups.**
- **Clean:** bare = 100 and tree = 0 on the v1.1.0 maps.
- **Low:** bare < 100.
- **Very low:** bare < 90, which is part of low.

## Bottom line

1. **Two different problems:**
   - **The ice margin (h17v01, and the < 90 pixels everywhere).**
     - These pixels are within about 0.3-0.7 km of the edge of the MCD12 region.
     - In June they look like **ground, not ice**: blue reflectance about 850 against about 7,000 for clean ice, warmer band 31, and NDVI
       slightly positive.
     - A third or more of them get clean (QA-passing) inputs in summer.
     - **The model's < 100 there is probably right, and MCD12's class 15 is what is off.**
   - **Interior ice just below 100 (h16v02's 1.9 million pixels, and h16v01's interior low pixels).**
     - These are ice in summer (blue reflectance close to clean ice), far from the edge (median 96 km and 76 km).
     - What sets them apart is mainly the **band-31 clamp**: in mid-summer their thermal composite is the artificial value 16000 (160 K if
       the 0.01 K scale applies), against a real reading of about 25,000-26,000.
2. **The clamp mainly hurts through the training data.**
   - In June almost no training row carries 16000 (79 of 8.6 million), so the value lands with the coldest real June rows, labelled about 82.
   - Without the clamp, low and clean ice have nearly the same June thermal value (h16v02 medians 26,381 and 26,170).
3. **h16v02's ice learns mostly from Canadian Arctic tundra, not from the ice-sheet training rows.**
   - The only ice training rows are on h16v01 (359,766 bare rows, all labelled 100).
   - Of h16v02's low ice pixels' leaf neighbours, 54% come from h12v02 and only 4% from h16v01; even its clean pixels take only 16% from h16v01.
4. **On h16v02's ice, no MOD44C input passes QA in any period, not even in June.**
   - The inputs' recorded solar zenith is about 86° in June, so they look like evening-pass observations under the midnight sun.
   - The QA test (solz < 65) rejects them all, and every composite is a fallback mean of low-sun inputs.
   - This is an upstream (PGE72/MOD44C) question, and it is left for later.
5. **H3 (h16v01 training vs prediction inputs, decision 19.6) does not matter.** 25-32% of the cells differ, but the models give the same
   answer from either.

**Direction:** keep band 31's real values, since it is an emissive band and the solar-zenith clamp has no physical basis there. The plan is
`Band31_Unclamped_Plan.md`. Southern ice-sheet training labels are the second lever, if the unclamped test is not enough.

## Checks

- **Exactness:** on each tile, 4,000 sampled ice pixels re-predicted with the v1.1.0 models (VcfPredict's steps exactly) equal the v1.1.0
  maps for both layers.
- **Thermal rebuild:** rebuilding band 31 with the clamp from the raw MOD44C inputs reproduces the stored composites exactly, in all 8 periods
  on all 3 tiles. The unclamped numbers below come from the same rebuild without the clamp.
- **Training keys:** they align with the model's training rows (20,000 rows equal in both splits), and the orientation is right (x = row; 2,000
  rows of h12v02 equal the metric folder).
- **One wrapper failure:** after `diag_train.py tree` had finished cleanly (RESULT PASS, all outputs written), its wrapper on forest203
  died before writing the done marker. The marker was written by hand with a note. Nothing was lost.

## 1. The groups

| Tile | Ice pixels | Clean | Low (< 100) | Very low (< 90) | Low: median km to the MCD12 edge | Very low: median km |
|---|---|---|---|---|---|---|
| h16v02 | 7,908,136 | 5,986,390 | 1,921,746 | 4,267 | **96.5** | 0.3 |
| h17v01 | 7,967,477 | 7,819,277 | 148,200 | 94,188 | **0.7** | 0.7 |
| h16v01 | 11,352,145 | 11,197,792 | 154,353 | 6,752 | **76.0** | 0.3 |

Clean pixels lie 63-148 km from the edge (median).

## 2. What sets the low pixels apart

### The margin pixels (h17v01 low; very low on every tile) are not ice in summer

| June 2019 (2019161) | Clean ice | h17v01, very low | h16v01, very low |
|---|---|---|---|
| Band 3 (blue) | 6,931 / 7,342 | 843 | 858 |
| Band 31 | 25,268 / 25,030 | 28,829 | 28,743 |
| NDVI, mean of the 4 greenest periods | 0 | +0.06 | +0.05 |
| Composites with a QA-passing input (June) | 0% | 54% | 62% |

They are dark, warm and slightly green. The AUC of summer blue reflectance, low against clean, is 0.02-0.04, so the groups barely overlap.

### The interior low pixels carry the band-31 clamp

**Band-31 composite, stored (clamped) against rebuilt without the clamp:**

| Tile, period | Group | Inputs all clamped | Stored median | Unclamped median |
|---|---|---|---|---|
| h16v02, June (2019161) | clean | 46% | 18,796 | 26,170 |
| h16v02, June | **low** | **77%** | **16,000** | 26,381 |
| h16v02, April (2019113) | clean | 48% | 18,044 | 25,635 |
| h16v02, April | **low** | **68%** | **16,000** | 25,698 |
| h16v01, July (2019209) | clean | 46% | 18,761 | 25,439 |
| h16v01, July | **low** | **71%** | **16,000** | 25,650 |

- **The pattern:** with the clamp, the low pixels sit on 16000; without it, low and clean are within about 200.
- **h16v02's other differences:**
  - Spectrally, summer blue is 5,194 for low against 5,906 for clean (AUC 0.44), so they look alike.
  - The remaining separation is weak and spread over the low-sun autumn and winter periods (mean |AUC − 0.5| 0.05-0.12 by predictor kind).
- **Recorded solar zenith of the inputs on the ice (median):**
  - 86° in April and June on h16v02;
  - 86-87° in July on h16v01 and h17v01;
  - 51-63° only in March, May-June or July, depending on the tile.

  The noon sun at these latitudes in June is about 42-55° from vertical, so many MOD44C inputs here look like observations from the
  evening pass under the midnight sun.

## 3. The training rows

### Snow/ice training rows (MCD12 class 15, selection only)

| Layer | Tile | Rows on class 15 | Label mean | Share < 100 bare / > 0 tree |
|---|---|---|---|---|
| Bare | **h16v01** | **359,766** (of 401,547) | 100.0 | 0.01% |
| Bare | h11v02 | 9,518 | 96.7 | 24.6% |
| Bare | h11v10 | 891 | 67.5 | 99.1% |
| Bare | h12v02 | 772 | 97.4 | 44.0% |
| Tree | **h16v01** | **123,684** (of 126,906) | 0.0 | 0% |
| Tree | h12v02, h10v04, h11v02, h09v04/05 | 41-3,007 each | 0-2.1 | |

h16v01 holds almost all of them, and it lies at 70-80°N.

### Clamped band 31 in the bare training rows

| Period | Rows at 16000 | Their labels | Main tiles |
|---|---|---|---|
| April | 38,100 | mean 82.9; 40% < 100 | h16v01 (20,627, mean 97), h12v02 (80), h21v01 (23), h13v01 (77), h12v01 (46), h11v02 (22) |
| **June** | **79** | mean 94 | almost none; the coldest real June rows (1,955 below 20,000) have mean label 82 |
| July | 94,941 | 100 | h16v01 ice |

- **Share of each tile's rows at 16000 (`out/clamp_share_bare_training.csv`):**
  - 100% in 2020033 on every high-latitude training tile;
  - 2019305/2019353 at 37-100% on h11v02 and h12v02;
  - **29% of h27v04 rows in 2019353**, a mid-latitude tile, so the clamp also fires outside the Arctic.
- **Why it matters:** an RF split is a threshold, so 16000 always goes to the coldest side. What the model learned there depends on which
  training rows carry the cold values in that period. In June those are not ice rows.

### Leaf neighbours: which training rows the ice pixels land with

This is the mean over the 60 trees of each training tile's share of the rows in the pixel's leaf, from 5,000 sampled pixels a group.

| Layer | Ice pixels | From h16v01 | Largest other sources | Neighbour label mean |
|---|---|---|---|---|
| Bare | h16v01, clean | 77.5% | h13v01 12.7% | 97.8 |
| Bare | h16v01, low | 50.8% | h13v01 26.6%, h12v02 9.7% | 91.9 |
| Bare | h17v01, clean | 76.4% | h13v01 9.9% | 97.0 |
| Bare | h17v01, low | 38.6% | h21v01 23.7%, h13v01 19.5% | 82.5 |
| Bare | **h16v02, clean** | **15.6%** | h11v02 37.1%, h12v02 34.1% | 97.1 |
| Bare | **h16v02, low** | **3.6%** | **h12v02 54.3%**, h11v02 25.6%, h13v01 15.9% | 94.7 |
| Tree | h16v02, low | 41.9% | h12v02 45.3% | 0.33 |
| Tree | h17v01, low | 36.6% | h12v02 17.2%, h21v01 12.7% | 2.7 |

The leaves hold every training row that falls in them, not only each tree's bootstrap sample, so these shares approximate the rows behind
each prediction.

### h16v01 training inputs against prediction inputs (H3, decision 19.6)

- **Cells that differ** between the stored training predictors and the v1.1.0 metric folder on h16v01's ice rows: bare 25.2%, tree 32.5%.
- **Model on 20,000 of those rows** (bare): the mean prediction is 99.998 from the stored values and 99.998 from the current ones; the share
  below 100 is 0.015% from both. Tree is 0 from both.
- **H3 is rejected:** the mismatch exists but does not change the predictions.

## 4. The hypotheses

| | Finding |
|---|---|
| **H1** (winter periods) | **Partly.** On h16v02 the separating values are the band-31 clamp in sunlit periods (April, June) plus weak low-sun autumn and winter differences. On h16v01 the clamp falls in July. The polar-night periods themselves separate little. The fill-only test, which changes only the −28672 values in those periods, left the ice unchanged (h16v01 < 100: 154,353 → 161,530; h17v01: 148,200 → 148,995). |
| **H2** (training rows) | **Yes, for h16v02.** Its ice draws mostly on Canadian Arctic tundra rows (h12v02, h11v02), because its predictor profile differs from the only ice rows, on h16v01. For the margin pixels, the neighbours are mixed Arctic land, which fits pixels that are not ice. |
| **H3** (h16v01 mismatch) | **No.** |
| **New** (not in the plan) | **The band-31 clamp**, and MOD44C inputs on the ice carrying a near-horizon solar zenith in summer (all fallback). |

## 5. What follows (each needs its own plan and the project lead's approval)

1. **Band 31 unclamped** (`Band31_Unclamped_Plan.md`): a pipeline correction with no new data source. Rebuild band-31 composites without
   the clamp, refit and evaluate everywhere, not only on the ice, because the clamp also fires at mid-latitudes in winter.
2. **Southern ice-sheet training labels**, only if (1) is not enough. This needs a label source decision: the project's own Landsat-based
   method, not MCD12 or C6.
3. **MOD44C inputs at high latitude in summer** (upstream, PGE72). Why do the inputs carry solar zeniths near 86° in June? This is linked to
   the MOD44C/PGE72 investigation. It is not part of this work.
4. **The margin pixels** need no fix if they are ground. A look on ADAPT at a few of them (summer Landsat or MODIS true colour) would confirm it.

## 6. Files

- Run: `FullRuns/EXP_icesheet_diag_20261005/`. Code in `code/` (`diag_tile.py`, `diag_train.py`, `thermal_check.py`,
  `margin_and_clamp.py`, `ice_counts.py`). Logs in `logs/`.
- Outputs in `out/`:
  - `<tile>_predictors.csv` (contrasts, AUC, period classes);
  - `<tile>_periods.csv` (periods chosen by ranked metrics);
  - `<tile>_groups.json`;
  - `<tile>_thermal.csv` and `_thermal_inputs.parq` (unclamped band 31);
  - `train_ice_rows_{tree,bare}.csv`;
  - `neighbours_summary_{tree,bare}.csv` and `neighbours_*.parq`;
  - `h16v01_stored_vs_current_*.csv`, `train_checks_*.json`;
  - `margin_distance.csv`, `clamp_share_bare_training.csv`.
