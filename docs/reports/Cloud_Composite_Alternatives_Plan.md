# Plan: can a different compositing rule make less cloudy composites? (h12v09, from MOD09GA/GQ)

**Written 2026-09-30. Status: narrowed to V4r (darkest red among the 3 MOD44C inputs when all are cloudy or cloud-adjacent) — see the amendment at the end. Approved and run 2026-09-30.** The steps 1-3 variants are in `Cloud_Composite_Alternatives_Results.md`; V4r is in `V4r_Darkest_Red_h12v09_Results.md`.
**Scope (project lead):** cloudiness only. No model runs. Thermal availability is ignored for now; MOD09GA has no thermal bands, so a composite built here could not feed the model as is. The goal is to learn whether a different rule changes cloud contamination, and by how much.
Background: `Cloudy_Fallback_Diagnosis_Results.md` and `RF_Fallback_Counts_Results.md` (labelling fallback pixels helps only about 1 point; the composites themselves need to be cleaner).

## How MOD44C is made now (PGE72, `pge72-main/MOD_PR44C/sort_values.c`, read only)

For each 250 m pixel, PGE72 goes through every observation in the 16 days, including the extra same-day observations, and **keeps one**: the one with the highest priority score `tlevel`.

**The score's bits, from most to least important** (`MOD44C.h:147-163`, `sort_values.c`):

| Bit | Set when |
|---|---|
| 15 | valid Band 1 and valid state |
| 14 | all reflective bands valid **and thermal bands 20, 31 and 32 non-zero** |
| 13 | not cloudy: cloud state is not cloudy (1) or mixed (2), MOD09 internal-cloud flag clear, and no shadow |
| 12 | solar zenith < 65 |
| 11 | no shadow |
| 10 | aerosol low or average |
| 9, 8, 7 | view-angle and coverage tiers (<= 45, 55, 65 degrees with obscov >= 15, 10, 5; PCF 299201-299206) |
| lowest | cirrus level, then aerosol level |

**Ties:**
- If the kept observation is clear: keep the **smaller view angle**, then the higher NDVI.
- If it is cloudy: keep the **higher NDVI**.

PGE61 then averages the good ones of the 3 MOD44C picks, or all 3 when none is good (the fallback).

**Where this could leave avoidable cloud:**
1. **Periods with no clear observation (the fallback case that hurts us).** Every candidate misses bit 13, so the pick is decided by solar zenith, shadow, aerosol and view angle. NDVI only breaks the last tie. A bright, near-nadir cloud can beat a darker, partly clear observation. PGE61 then **averages three** such picks.
2. **Clear observations that are not really clear** (undetected cloud, cloud edges, haze, smoke):
   - Among flagged-clear observations the tie goes to the smallest view angle, not the darkest or greenest.
   - MOD09's "adjacent to cloud" flag (state bit 13) is not used.
   - Cirrus ranks below view angle.
3. **One observation per 16 days.** Within a 48-day period, the best single observation, or the mean of all clear ones, is never considered directly.
4. **Thermal validity (bit 14) outranks clearness.** A cloudy observation with thermal can beat a clear one without it. This is noted only; it cannot be tested with public data, and thermal is out of scope.

"Mixed" cloud state is treated the same as "cloudy". Snow flags are not used.

## Data

- MOD09GA and MOD09GQ C6.1 for h12v09, 2019065-2020049: 350 days each, all present (`/css/modis/Collection6.1/L2G/`).
- **Limitation:** the public files hold only the first observation per day. PGE72's internal inputs have more.
- So every rule is compared **like for like on the same MOD09GA/GQ inputs**, including a re-implementation of the current rule (R0). The real MOD44C is used only to check R0.

## Steps

**1. R0: re-implement the current PGE72 rule** on MOD09GA/GQ (bit 14 without the thermal part), giving 23 16-day composites. Then apply the PGE61 48-day averaging, giving 8 periods.
- **Check against the real MOD44C:** exact Band_1 and Band_2 match on land should be at least 95%. Stage 3 matched 99.96-100% of h12v09 pixels to a public MOD09GQ day.
- **Check against the current pipeline:** R0's fallback counts should be close to the stored `qa.npy` (h12v09 mean 3.29).
- **Stop and report** if R0 fails either check.

**2. Variants** (same inputs, same 8 periods):

| ID | Change | Targets |
|---|---|---|
| V1 | When no observation in the 16 days is clear, pick the minimum-blue observation (V1b: maximum NDVI) instead of going by the angle and aerosol bits | point 1 |
| V2 | Among clear observations, add "not adjacent to cloud" and "no cirrus" as priority bits above view angle, and break ties by minimum blue instead of view angle | point 2 |
| V3 | One composite per 48-day period straight from all daily observations: the mean of all clear observations if any exist, otherwise the minimum-blue observation | points 1 and 3 |
| V3m | As V3, but the median of the clear observations | point 2 |
| V4 | No daily data: keep MOD44C, but in PGE61 fallback periods take the pick with minimum red (V4b: maximum NDVI) instead of the mean of the 3 | point 1, with the existing inputs |

V4 is the key control. If it gets most of the gain, the daily data is not needed.

**3. Cloudiness measures.** No model. For every variant, per pixel and period, on all land, dense forest (historical >= 60), the 103,011 drop pixels and the 200,000 control pixels:
- **Flag-based:** state of the selected observation (clear, mixed or cloudy; shadow), and the fallback count under the PGE61 rule where it applies.
- **Physical, independent of the flags:**
  - the share of composites with blue (Band 3) > 0.10 (bright, cloud-like);
  - red and NDVI in dense forest (median and 10th percentile);
  - the wet-season periods (2019353, 2020033) shown separately.
- **Stability:** in dense forest, the standard deviation of NDVI across the 8 periods, and the number of periods with NDVI < 0.6 (clear forest is about 0.8).
- **Side effects:**
  - the shadow share and too-dark NIR (minimum-blue rules can pick cloud shadow);
  - the view-angle distribution (maximum-NDVI rules favour off-nadir views);
  - the change in the clear-pixel values themselves (a new rule should barely change pixels that are already clear).

**4. Report:** `Cloud_Composite_Alternatives_Results.md`. It covers:
- whether any variant gives clearly less cloudy composites at the drop pixels without side effects;
- whether that needs the daily data (V1-V3) or not (V4);
- what it would take to use it: V4 is a PGE61/Python-only change; V1 and V2 need PGE72 changes; V3 replaces MOD44C in the chain, and thermal would have to come from elsewhere.

## Cost, conditions

- **Code:** one reader for MOD09GA/GQ (state_1km, sensor and solar zenith, 500 m bands 3-7, 250 m bands 1-2, obscov) and one compositing script.
- **Processing:** one period at a time, holding 48 days in memory (about 8 GB), so each file is read once. Estimated 30-60 minutes on ngaproc, fully detached, with status markers. V4 runs in minutes from the existing composites.
- **Output:** a new run root `FullRuns/cloud_composites_h12v09_<date>/`; the scripts refuse to reuse it.
- **Not touched:** nothing in rlgill's pge72, pge61 or `modis_vcf`, and no product code.

## Decisions for the project lead

- Go-ahead for steps 1-4 on h12v09.
- Whether to add a second cloudy tile afterwards (h19v09, Congo) if h12v09 shows a gain.

## Amendment (project lead, 2026-09-30): test V4 only, refined; darkest = red

**Why:** from daily data, the darkest value is almost always cloud shadow or water, which would create a new problem. So V1-V3 are set aside. V4 is cheap: it needs only the h12v09 48-day composites and metrics remade, and no new model.

**Rule (V4r), per pixel and 48-day period, over the 3 MOD44C inputs:**
1. **At least one input passes the pipeline QA** (bit 15): the mean of the good inputs. Unchanged.
2. **All 3 inputs are cloudy or cloud-adjacent** (MOD44C state cloud bits != 0, or adjacency bit `state & 64` set, on every input): take the input with the **lowest red (Band_1)**.
   - The whole observation is used: all 8 bands, including Band_31, come from that input. There is no per-band minimum, which would mix dates.
   - **Guards.** A candidate is excluded if it:
     - is shadow-flagged (`state & 4`);
     - has NDVI < 0 (water);
     - has NIR (Band_2) below a floor, set from the cheap check (default 1500, 0.15 reflectance);
     - is a missing input (10000).
   - If no candidate survives the guards: the mean of all 3, as now.
3. **All other fallback** (aerosol, zenith, shadow, missing): the mean of all 3, as now.

With this rule, clear pixels and non-cloud fallback are identical to the current composites. The `qa.npy` fallback flag is unchanged; a new per-period flag records where the darkest pick was used.

**Steps:**

1. **Cheap check (read-only, minutes).** From the existing per-input band files (`1-Days`) and the MOD44CQ state, on h12v09:
   - where the rule triggers (share of pixel-periods; mean per pixel at drop, control, dense and land);
   - red, blue, NIR and NDVI of the darkest-red pick against the current all-input mean;
   - how often each guard rejects a candidate, and how often all 3 are rejected;
   - agreement of the darkest-red pick with the darkest-blue pick, as a check on the band choice;
   - the pick's view angle, if available.

   Stop and report if the picks look like shadow or water: NIR much below clear-forest NIR at dense pixels, or NDVI far from the controls in the dry season.
2. **Rebuild (if step 1 is clean).**
   - Copy the Python code into the run root and change only `CompositeDayFile._computeComposite`; the state is read as it already is for the QA mask.
   - Rebuild the h12v09 composites and metrics (about 30-60 minutes).
   - Predict with the existing 63-tile RF (G), using the validated raster-prediction path from `rf_fallback_counts_20260930/code/evaluate.py`.
3. **Compare with the current h12v09 map (G):**
   - bias against the historical product at the drop and control pixels, and in dense forest by fallback count;
   - the share of pixels 30 or more below the historical value;
   - held-out label MAE on the 4,997 h12v09 test pixels, overall and by fallback count;
   - the change on pixels where the rule never triggers, which must be 0.

   **Caveat:** G was trained on mean-based fallback composites. A gain here is a signal. A 63-tile retrain with the rule applied everywhere would be the confirmation step, with separate approval.

**Output:** a new run root `FullRuns/v4r_darkest_red_h12v09_<date>/`. **Results:** `V4r_Darkest_Red_h12v09_Results.md`. Detached on ngaproc; no product code or rlgill repositories touched.
