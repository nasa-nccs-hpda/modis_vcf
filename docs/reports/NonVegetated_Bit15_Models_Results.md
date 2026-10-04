# Non-vegetated (bare) layer: baseline, RF, XGBoost and the static models on bit-15 metrics (results)

**Run 2026-10-01, 11:18-15:11, ngaproc201/202, fully detached. Plan: `NonVegetated_Bit15_Models_Plan.md` (approved), plus the static-model fourth candidate (approved during the run).** Folder: `FullRuns/nonveg_bit15_models_20261001/`.

## Recommendation

**Use RF for the non-vegetated layer, as for tree cover.**
- **Held-out labels:** RF ties the retrained baseline (4.20 against 4.18).
- **Maps:** RF beats every other candidate against C6, C6.1 and the C code alike. Against C6 it is 0.95 better than the baseline and 0.75 better than the current model, with the same sign across all three references and every interval clear of zero.
- **By the rule** (about 1 point or less is a tie) the gains are ties, but they all point the same way. RF also gives the per-pixel SD layer (`SD_Layer_Results.md`).
- **XGBoost** is worse on held-out labels and on maps.
- **The C code's static models** are the worst on held-out labels by far, because they appear to expect a calendar shifted by half a year (`Bare_Static_Model_Pixel_Trace.md`).

## What was compared

| Name | What it is |
|---|---|
| CURRENT | the old `pctbare.bin` (trained 2026-09-19 on pre-bit-15 metrics), applied to bit-15 metrics. The non-vegetated layer until now |
| BASELINE | BaggingRegressor(DecisionTree), 30 members, retrained on bit-15 metrics |
| RF | RandomForest, 30 trees, `max_features='sqrt'`, retrained on bit-15 metrics |
| XGB | XGBoost tunedB, retrained on bit-15 metrics |
| STATIC | the C code's 30 static M5P models, run in numpy (`Weka_M5P_Bare_Models_to_Python.md`) on the bit-15 metrics |
| CCODE | the C code's map from the 2026-09-19 run (static models on C metrics, bits 7-8 aerosol shift, and the C code's bare rules) |

**Setup:**
- **Training:** the 21 C bare predictors and the legacy bare labels on 20 tiles: 10.8 M pixels, of which 8.6 M train and 2.16 M are a 20% held-out pixel split.
- **Prediction:** all 63 tiles.
- **Tree cover:** the bit-15 RF in every product. All Python products went through the same code copy. That copy sets non-tree vegetation = max(0, 100 − tree − non-vegetated) and does not lower non-vegetated. The plan's wording said otherwise; this is the correct description.

## 1. Held-out labels (2.16 M pixels, 20 tiles)

| Product | MAE | Bias | R² |
|---|---|---|---|
| BASELINE | 4.18 | +0.08 | 0.960 |
| RF | 4.20 | +0.06 | 0.961 |
| CURRENT | 5.01 | +0.40 | 0.948 |
| XGB | 5.40 | 0.00 | 0.942 |
| STATIC | 28.9 | +19.0 | - |
| CCODE | 28.3 | +21.9 | -0.08 |

**Differences** (block bootstrap; tile bootstrap in brackets):
- RF − baseline: +0.02 (tile CI −0.01 to +0.07; RF better on 9/20 tiles). **A tie.**
- XGB − RF: +1.19 (+0.86 to +1.54). XGBoost is worse on 20/20 tiles.
- RF − CURRENT: −0.81 (−1.24 to −0.44), better on 16/20 tiles. The current model may have been trained on some of these pixels, so this gap is if anything understated.

**RF by label bin:** 3.3 (0-20), 6.6 (21-40), 5.8 (41-60), 3.7 (61+).

**C6 at the same pixels:** 11.9 from the labels (bias +6). That makes the C6-label disagreement itself about 12 points. Note that 6 of the 20 label tiles are under the C6.1 high-latitude rule.

## 2. Maps against C6, C6.1 and the C code (54 tiles without the C6.1 high-latitude flag)

Pooled over 959 M pixels on a common valid mask.

| Product | vs C6 MAE (bias) | vs C6.1 MAE (bias) | vs C code MAE (bias) |
|---|---|---|---|
| **RF** | **9.13** (−5.1) | **9.20** (−5.4) | **11.08** (−4.6) |
| CURRENT | 9.88 (−5.4) | 9.96 (−5.7) | 11.70 (−4.9) |
| XGB | 9.97 (−6.9) | 10.07 (−7.2) | 11.63 (−6.5) |
| BASELINE | 10.07 (−5.5) | 10.16 (−5.9) | 11.67 (−5.1) |
| CCODE | 10.32 (−0.5) | 10.39 (−0.8) | - |
| STATIC | 10.75 (−1.5) | 10.82 (−1.8) | 3.52 (−1.0) |

For reference, C6.1 against C6 is 1.98.

**Differences against C6** (95% CI over blocks, then tiles):

| Comparison | MAE difference | Block CI | Tile CI | Tiles where the first is better |
|---|---|---|---|---|
| RF − BASELINE | −0.95 | −0.96 to −0.93 | −1.15 to −0.74 | 49/54 |
| RF − CURRENT | −0.75 | −0.76 to −0.74 | −0.94 to −0.56 | 47/54 |
| XGB − RF | +0.84 | +0.82 to +0.86 | +0.46 to +1.24 | 16/54 |
| RF − STATIC | −1.62 | −1.67 to −1.57 | −2.94 to −0.43 | 28/54 |

Against C6.1 the results are the same within 0.02. Against the C code: RF − BASELINE −0.59, RF − CURRENT −0.62, XGB − RF +0.55, all with the same sign.

**By C6 cover bin (MAE / bias):**

| Product | 0-20 | 21-40 | 41-60 | 61+ |
|---|---|---|---|---|
| RF | 6.7 / −2.6 | 12.9 / −7.6 | 14.2 / −9.8 | 10.4 / −8.7 |
| CURRENT | 7.1 / −2.7 | 14.2 / −8.4 | 16.1 / −10.8 | 10.7 / −8.6 |
| BASELINE | 7.3 / −3.1 | 14.7 / −8.4 | 16.0 / −10.3 | 10.9 / −8.6 |
| XGB | 7.3 / −5.0 | 14.9 / −10.1 | 15.3 / −10.6 | 10.2 / −7.8 |
| STATIC | 7.6 / +1.3 | 13.7 / −1.3 | 15.3 / −2.0 | 16.8 / −13.6 |
| CCODE | 7.7 / +1.8 | 13.8 / −1.0 | 15.6 / −1.2 | 12.7 / −9.0 |

**Reading:**
- **RF is best in the three lower bins and close in 61+.**
- **Every retrained Python model reads about 5 points less bare than C6.** That matches the labels, which read about 6 points less bare than C6 at the held-out pixels. So the retrained models follow their labels, not the historical product.
- **The static models and the C code are nearly unbiased against C6 in the lower bins** (same model lineage) but much worse at 61+.

## 3. Effect on non-tree vegetation (54 tiles)

**Means across the 54 tiles:**

| Product | Mean tree cover | Mean non-vegetated | Mean non-tree vegetation | Share with non-tree vegetation = 0 |
|---|---|---|---|---|
| RF | 24.4 | 22.4 | 53.2 | 1.4% |
| CURRENT | 24.4 | 22.1 | 53.5 | 2.9% |
| BASELINE | 24.4 | 22.0 | 53.7 | 3.0% |
| XGB | 24.4 | 20.6 | 55.1 | 1.6% |
| STATIC | 24.4 | 26.0 | 49.6 | 0.3% |

**What this shows:**
- Switching from CURRENT to RF shifts non-vegetated by +0.3 points on average.
- It also halves the share of pixels where tree cover and non-vegetated reach 100 together, because RF overshoots less.

## Caveats

- **The C code's map has bare rules that the Python products do not.** Its rules are the sum cap, the high-latitude bare = 100 rule and the Sahara stretch (`Bare_Static_Model_Pixel_Trace.md` section 3). Comparisons against CCODE on these 12 tiles mix those rules with model differences:
  - high-latitude rule: h11v02, h12v01, h12v02, h13v01, h16v01;
  - Sahara stretch: h17v05, h18v07, h19v08, h20v06, h20v08, h21v05, h21v06.

  The project lead has deferred the rules until after the model assessment.
- **C6, C6.1 and the C code come from one model lineage, not three independent references.** The held-out labels are the independent check, and they cover only 20 tiles.
- **C6 and the labels disagree by about 12 points.** Absolute agreement with either is limited; the decision rests on relative differences.

## Next

1. If RF is approved, set it as the default non-vegetated model in the delivery branch. It would be the model trained here: `models/rf/pctbare.bin` with the 21 C predictors.
2. Add the SD code to the delivery branch (`SD_Layer_Results.md`) and produce SD for all 63 tiles.
3. Decide between the mean and the median of the trees for both layers (`RF_Tree_Aggregation_Test_Results.md`).
4. After that, consider the C code's bare side rules.
