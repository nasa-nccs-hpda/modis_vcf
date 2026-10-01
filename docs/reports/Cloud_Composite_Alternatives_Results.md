# Can a different compositing rule make less cloudy composites? Results on h12v09 (daily MOD09GA/GQ)

**Run 2026-09-30 16:43-17:15, ngaproc201/202.** Plan: `Cloud_Composite_Alternatives_Plan.md` (steps 1-3, before its V4r amendment).
Run root `FullRuns/cloud_composites_h12v09_20260930/`. Tables are in `out/measures_summary.txt` and `out/measures.csv`.
**Scope: cloudiness of the composites only. No metrics and no model.** The effect on the tree-cover model is tested separately for V4r on the real MOD44C (`V4r_Darkest_Red_h12v09_Results.md`).

## Bottom line

- **The two rules that replace cloudy picks with the darkest guarded observation give clearly cleaner composites. Changing PGE72's priority bits does not.**
  - At the drop pixels, the share of cloud-bright composites falls from 51% (current pipeline) to 24% with V4 and 17% with V3/V3m.
  - V1 reaches only 40% and V2 49%.
- **V4 gets most of the gain without daily data.**
  - V4 changes nothing in PGE72: it acts at the PGE61 step on the existing 3 picks.
  - Of V3's improvement over PIPE at the drop pixels, it gets 79% for cloud-bright share and periods with NDVI < 0.6, 62% for temporal SD and 57% for NDVI P10.
  - It leaves every already-clear composite exactly as it was. V3 shifts clear pixels by a median of about 18 in red, because it averages several clear days.
- V3 and V3m (one composite per 48 days straight from daily data) are the cleanest. But they replace MOD44C in the chain, they need thermal from somewhere else, and they rely on daily minimum picks. The project lead set those aside because of the risk of shadow and water.
- This is why V4 was refined into V4r and tested through the model. **That test found that V4r largely removes the drop against historical, but the existing models do about 2 points MAE worse against the held-out labels.** The next step proposed is retraining (see the V4r report).

## Validity of the R0 baseline (accepted, with a known limitation)

R0 re-implements the current PGE72 rule on the public MOD09GA/GQ, so every variant is compared like for like with it.

- **Exact match with the real MOD44CQ** (bands 1 and 2, on land, per 16-day window): minimum 0.853, median 0.881, maximum 0.936, over 23 windows. This misses the plan's threshold (0.90 in the code, 95% in the plan text), so the run reports "R0 VALID: False".
  - **The project lead accepted it for this evaluation (2026-09-30).**
  - The likely cause is that the public files hold only the first observation of each day, while PGE72's internal inputs have more.
- **Fallback flag** against the current pipeline: identical on 95.8% of land pixel-periods, with the same mean of 3.29 fallback periods per land pixel.
- So R0 reproduces the pipeline's cloudiness behaviour closely, but not pixel for pixel. Read the variant differences as relative to R0, not as exact predictions of what PGE72 would produce.

## What each variant does

| ID | Rule |
|---|---|
| PIPE | The current pipeline composites, from the real MOD44C (for reference) |
| R0 | PGE72's priority rule on MOD09GA/GQ, then PGE61's 48-day rule (mean of the good picks, else mean of all 3) |
| V1 | PGE72 changed: in a 16-day window with no clear observation, take the lowest-red guarded observation |
| V2 | PGE72 changed: "not adjacent to cloud" and "no cirrus" rank above the view-angle tiers |
| V3 | No 16-day step: per 48 days, the mean of all clear observations; if there are none, the lowest-red guarded observation |
| V3m | As V3, with the median of the clear observations |
| V4 | R0's 3 picks; where all 3 are cloudy or cloud-adjacent, take the lowest-red guarded pick instead of the mean |

The guards are not shadow, valid, NDVI >= 0 (NIR >= red) and NIR >= 1500.

## How to read the measures

Every pixel gets **8 composites a year**, one per 48-day period. A *pixel-period* is one of these.

| Measure | Meaning | Better |
|---|---|---|
| Cloud-bright share | Fraction of pixel-periods with blue (Band 3) reflectance > 0.10. Clear forest is about 0.02-0.04; cloud is often > 0.2. | Lower |
| Wet-season cloud-bright | The same, for the periods 2019353 and 2020033 only | Lower |
| NDVI P10 | NDVI of the worst 10% of pixel-periods. Clear tropical forest is about 0.8-0.85; cloud pulls it toward 0-0.3. | Higher |
| Periods with NDVI < 0.6 | Per pixel, how many of its 8 periods look cloudy, averaged | Lower |
| NDVI temporal SD | Per pixel, the SD of NDVI over its 8 periods, averaged. Stable forest should be steady. | Lower |
| Fallback per pixel | Periods (out of 8) with no good pick under the PGE61 rule | Lower |
| Clear-pixel shift | Median absolute change in red, relative to R0, where R0 had a good pick | 0 = no side effect |

The blue > 0.10 and NDVI < 0.6 lines are rough "looks cloudy" cuts for tropical forest. They are good for ranking the variants, not as absolute cloud fractions.

## Results

**Drop pixels** (103,011 pixels where the tree-cover prediction drops sharply):

| Variant | Cloud-bright | Wet-season cloud-bright | NDVI median | NDVI P10 | Periods NDVI < 0.6 | NDVI temporal SD | Fallback per pixel | Clear-pixel shift |
|---|---|---|---|---|---|---|---|---|
| PIPE | 51% | 87% | 0.60 | 0.27 | 4.0 | 0.22 | 5.13 | - |
| R0 | 50% | 90% | 0.61 | 0.16 | 3.9 | 0.26 | 5.07 | - |
| **V4** | **24%** | **51%** | 0.83 | 0.45 | 1.6 | 0.17 | 5.07 | **0** |
| V1 | 40% | 80% | 0.71 | 0.20 | 2.9 | 0.23 | 5.07 | 0 |
| V2 | 49% | 89% | 0.63 | 0.16 | 3.8 | 0.26 | 4.70 | 0 |
| V3 | 17% | 37% | 0.84 | 0.58 | 0.9 | 0.13 | 4.70 | 20 |
| V3m | 17% | 37% | 0.85 | 0.58 | 0.9 | 0.13 | 4.70 | 17 |

**Dense forest** (historical >= 60, 15.9 M pixels):

| Variant | Cloud-bright | NDVI P10 | Periods NDVI < 0.6 | NDVI temporal SD | Fallback per pixel |
|---|---|---|---|---|---|
| PIPE | 21% | 0.44 | 1.6 | 0.15 | 3.09 |
| R0 | 24% | 0.24 | 1.9 | 0.22 | 3.09 |
| **V4** | **8%** | 0.70 | 0.5 | 0.09 | 3.09 |
| V1 | 18% | 0.28 | 1.3 | 0.20 | 3.09 |
| V2 | 23% | 0.24 | 1.8 | 0.21 | 2.72 |
| V3 | 6% | 0.74 | 0.3 | 0.07 | 2.72 |
| V3m | 6% | 0.75 | 0.3 | 0.07 | 2.72 |

All land and the 200,000 control pixels rank the variants the same way (`out/measures_summary.txt`).

**Side effects:**
- **None of the darkest-pick rules picked water or shadow in any measurable amount.** NIR < 0.15 is at most 0.4% of drop pixel-periods for every variant, the same as PIPE, and NDVI < 0 is 0 everywhere.
- The guards did their job. Note that the daily-minimum risk the project lead raised is about *daily* data; V3 would rely on these guards alone.

## Notes

- **V2 lowers the fallback count without cleaning the composites.** Fallback per pixel drops from 3.29 to 2.93 on land, but the physical measures barely move. This is consistent with R0 picking cloud-adjacent observations that PGE61's QA then rejects, while V2 ranks "not adjacent" first. V2 changes *which* observation passes QA more than *how clean* it is. Not investigated further.
- V3 and V3m have the same fallback count as V2. Both count a period as fallback only when no observation in the 48 days passes QA.
- R0 is cloudier than PIPE in places (NDVI P10 0.16 against 0.27 at the drop pixels). This fits R0 seeing only one observation per day, so it has fewer chances of a clear one. It affects every variant built on MOD09GA equally.
- Thermal was out of scope. PGE72 ranks thermal validity (bit 14) above clearness; that could not be tested with public data.

## Where this leaves things

| Option | Cleaner? | What it would take |
|---|---|---|
| V4 / V4r | Yes, most of the gain | PGE61/Python only. Model effect tested on h12v09: the drop is fixed against historical, but the existing model is worse against the labels without retraining. |
| V1 | Somewhat | PGE72 change |
| V2 | No | PGE72 change |
| V3 / V3m | Yes, the most | Replaces MOD44C; needs a thermal source; relies on daily minimum picks |

Not done: a second cloudy tile (h19v09, Congo) was offered in the plan as a follow-up if h12v09 showed a gain. It has not been requested.
