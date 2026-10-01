# V4r (darkest-red pick in all-cloudy periods): results on h12v09

**Run 2026-09-30, finished ~20:39.** Plan: amendment in `Cloud_Composite_Alternatives_Plan.md`.
Run root `FullRuns/v4r_darkest_red_h12v09_20260930_r2/`. Tables are in `eval/` and the full log is `logs/compare.log`.

## Bottom line

- **Against the historical MOD44B, V4r removes most of the cloud-driven drop.**
  - At the 103,011 drop pixels, the bias of RF (G) moves from -21.7 to -4.9 points.
  - The share of drop pixels 30 or more points below historical falls from 19% to 1.6%.
  - In dense forest with 5 or more fallback periods, the bias of -5 to -16 becomes +1 to +2.
- **Against the held-out training labels, V4r is worse.**
  - On 4,932 h12v09 held-out pixels, MAE rises from 6.84 to 8.70 (+1.9), and the bias moves from -3.8 to -6.2.
  - It is worse in every fallback stratum, including pixels with 0-2 fallback periods.
- **The two references disagree, so by the relative-difference rule this is not a decision to adopt V4r.** The same split showed up in the model choice: the historical references favoured XGBoost, and the held-out labels put it last.
- **The test has a built-in handicap.** The models were trained on the current, cloud-contaminated metrics and were not retrained. A fair test of V4r needs a model trained on V4r metrics (see "Next step").

## What was run

1. **Compose.** For each pixel and 48-day period:
   - When all MOD44C inputs are cloudy or cloud-adjacent, the input with the lowest red replaces the composite, with all 8 bands from that one date.
   - Guards exclude shadow, NDVI < 0, NIR < 1500, invalid and missing inputs.
   - Everything else is copied unchanged.
2. **Metrics.** The unchanged `makeMetrics`, on the V4r composites.
3. **Compare.** Prediction with the existing bit-15 RF (G) and RF + fallback counts (N) from `rf_fallback_counts_20260930`. No retraining. The comparison is against the same models on the current metrics.

**Two fixes were needed before it ran correctly** (details in the README status):
- The first alignment check failed only because it treated the missing 2020065 slot as an input. The stored 2020033 composite is the mean of the 2 inputs that exist; this was verified at 100% of fallback pixels.
- The chain pointed makeMetrics at the wrong directory, so it rebuilt the standard composites. It was stopped, corrected and restarted.

## Integrity checks (all passed)

| Check | Result |
|---|---|
| Alignment: mean of the available inputs equals the stored composite at fallback pixels, every period and band | 1.00000 |
| Shadow/water gate: median NIR of the picks divided by clear dense-forest NIR (fail if < 0.6) | 0.96 |
| Metrics differ where V4r never triggered | 0 pixels |
| Predictions differ where V4r never triggered (G and N) | 0 pixels |
| Current-metrics map identical to the `rf_fallback_counts` run (G and N) | 1.00000 |

## How much V4r changes

- V4r triggers often. 97% of dense-forest pixels had at least one period replaced (only 418,174 of 15.9 M never triggered).
- By period, the share of land pixels triggered ranges from 4% (2019209, dry season) to 82% (2020033, wet season).
- Of the triggered pixels, the median predictor changed value at 59%.
- **The picks look like forest, not cloud.** For example, at dense pixels in 2019065 the pick has red 309 against 849 for the current mean, and NDVI 0.82 against 0.59. Clear dense pixels in the same period are 277 and 0.85.
- **The picks are slightly dark in NIR**: 3,096 against 3,248 for clear dense pixels in 2019065, about 5% lower. The gate ratio, pooled over periods, is 0.96.

## Map against historical MOD44B (mean of prediction minus historical, points)

| Pixel set | n | G current | G V4r | N current | N V4r |
|---|---|---|---|---|---|
| Drop pixels | 103,011 | **-21.7** | **-4.9** | -20.2 | -4.2 |
| Control (dense, baseline within 3 of historical) | 200,000 | -0.35 | -0.29 | -0.16 | -0.14 |
| Dense forest (historical >= 60) | 15.9 M | -1.20 | +0.07 | -0.92 | +0.28 |
| All land | 22.3 M | +0.44 | **+3.02** | +1.02 | +3.47 |

**Dense forest by number of fallback periods (G):**

| Fallback periods | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 |
|---|---|---|---|---|---|---|---|---|---|
| Current | -1.2 | -0.6 | -0.1 | +0.1 | -0.4 | -4.7 | -8.1 | -10.9 | -15.8 |
| V4r | -1.2 | -1.0 | -0.6 | -0.2 | +1.0 | +0.9 | +1.5 | +2.0 | +2.2 |

- The dependence on fallback count is gone: V4r flattens the dense-forest bias to within about 2 points at every count.
- **On all land, though, V4r moves the map 2.6 points above historical.** Most of that comes from outside dense forest, where it is not checked here.

## Held-out training labels on h12v09 (prediction minus label)

| Model | n | MAE | Bias | MAE, 0-2 fallback (2,015) | MAE, 3-4 (2,251) | MAE, 5+ (666) |
|---|---|---|---|---|---|---|
| G current | 4,932 | **6.84** | -3.80 | 7.52 | 5.32 | 9.91 |
| G V4r | 4,932 | 8.70 | -6.23 | 8.68 | 7.95 | 11.29 |
| N current | 4,932 | **6.71** | -3.45 | 7.58 | 5.25 | 9.00 |
| N V4r | 4,932 | 8.41 | -5.70 | 8.67 | 7.62 | 10.34 |

**By label range (G):**

| Label | n | MAE current | MAE V4r | Bias current | Bias V4r | Mean change in prediction |
|---|---|---|---|---|---|---|
| 0-9 | 131 | 9.34 | 9.15 | +9.1 | +8.8 | -0.3 |
| 10-29 | 151 | 10.70 | 11.05 | +5.4 | +4.1 | -1.3 |
| 30-59 | 436 | 14.45 | 14.75 | -4.9 | -5.4 | -0.6 |
| 60-79 | 547 | 14.22 | 14.08 | -8.2 | -9.4 | -1.2 |
| **80-100** | **3,667** | **4.58** | **7.06** | -3.9 | -6.8 | **-3.0** |

- The whole loss is at the dense-forest labels (74% of the held-out pixels). There, V4r *lowers* predictions by 3 points, while the current model is already 3.9 low.
- Across the full dense-forest map, by contrast, V4r *raises* predictions relative to historical.
- So at the label pixels, V4r inputs push the existing model down. A likely reason is the model learning cloud-contaminated relationships. The picks also have slightly lower NIR (about -5%), and the replaced Band_31 comes from a different date than the mean. **Not investigated.**

## Reading

- The composites are clearly cleaner, and the drop artefact against historical is largely removed.
- Whether the tree-cover map is *more accurate* is not shown. The only reference independent of historical MOD44B, the held-out labels, says the existing models do worse with V4r inputs, by about 2 points MAE.
- **A model fit on one kind of input and applied to another is expected to shift.** The labels test therefore says "the current model does not transfer to V4r inputs" more than "V4r is worse".
- The +2.6 shift on all land against historical also needs a look before any adoption.

## Next step (needs approval)

**Retrain on V4r metrics.**
1. Build V4r composites and metrics for the training tiles. V4r is a PGE61/Python-only change, but it needs the MOD44C per-input data for each tile.
2. Refit G with the same predictors, split and seed.
3. Compare against G current on the held-out labels (all 63 tiles, not just h12v09) and against historical.

Adopt V4r only if it is no worse against the labels, by more than about 1 point MAE, and still removes the drop. A cheaper first step is h12v09 plus 2-3 cloudy training tiles.

Also open: where the +2.6 all-land shift comes from (non-forest, by land cover).
