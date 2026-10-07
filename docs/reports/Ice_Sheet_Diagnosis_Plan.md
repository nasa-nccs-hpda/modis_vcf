# Why v1.1.0 gives the Greenland ice sheet less than 100 bare: diagnosis plan (read-only)

**Results 2026-10-05 late evening: `Ice_Sheet_Diagnosis_Results.md`.** **Status: E1 APPROVED by the project lead 2026-10-05 late evening.** Read-only: nothing is refit or changed by this plan.
**Project lead, on E3 and the counts:** the MCD12-based ice-sheet counts are a **reference, not a definitive boundary of the ice sheet**;
report them as such (pixels selected by MCD12 class 15), never as the ice sheet's true extent or as truth.

## 1. The question

On the v1.1.0 maps, part of the Greenland ice sheet is not 100% non-vegetated and 0% tree cover. The project lead wants this fixed **from the
data and the pipeline**: no external product used as a mask, and no override rule like the v00 rule. MCD12Q1 land cover is used below **only
to select pixels for the diagnosis**. It is a reference, not truth, as C6 is.

Snow and ice are about the most distinctive surface in the inputs: very bright in bands 1-4, dark in bands 6-7, cold in band 31. The
diagnosis asks **what in the chain (composites → metrics → training rows → models) blurs that signal.**

## 2. Baseline (v1.1.0 maps, `FullRuns/v110_sd_20261004`, counted 2026-10-05)

Ice sheet = MCD12Q1 2019 class 15, largest connected region per tile; 31,841,541 pixels (231.656 m) on the 6 Greenland tiles.

| Tile | Ice pixels | Tree > 0 | Tree ≥ 5 | Bare < 100 | Bare < 90 | Mean bare |
|---|---|---|---|---|---|---|
| h15v02 | 2,234,222 | 166 | 6 | 124,269 (5.6%) | 1,648 | 99.88 |
| h16v00 | 328,332 | 0 | 0 | 0 | 0 | 100.00 (v00 rule) |
| h16v01 | 11,352,145 | 3,760 | 159 | 154,353 (1.4%) | 6,752 | 99.96 |
| h16v02 | 7,908,136 | 353 | 53 | **1,921,746 (24.3%)** | 4,267 | 99.61 |
| h17v00 | 2,051,229 | 0 | 0 | 0 | 0 | 100.00 (v00 rule) |
| h17v01 | 7,967,477 | 1,035 | 72 | 148,200 (1.9%) | **94,188** | 99.72 |
| **Total** | 31,841,541 | 5,314 | 290 | 2,348,568 (7.4%) | 106,855 | |

Two different problems may be mixed here:
- **A, h16v02:** many pixels just below 100 (mean 99.61);
- **B, h17v01:** fewer pixels, but far below 100 (94,188 below 90).

Tree cover above 0 is rare: 5,314 pixels, of which 290 are ≥ 5.

## 3. What we already know

- **Every change to the winter composites has moved the ice sheet:**
  - fill72: on h16v01, tree = 0 went 99.7 → 94.1%;
  - the full unusable-input rule: on h16v01 and h17v01, tree = 0 went to 64-69% (h16v02 and h15v02 improved against C6);
  - the fill-only test is running now.
- **On the ice sheet, 3-4 of the 8 periods are polar night or twilight** (2019305, 2019353, 2020033, and partly 2019257). Their composites
  hold fallback means of zeros, −28672 and 10000.
- **Both models read those periods directly:**
  - Tree (64 predictors): 5 raw winter bands (for example `Band_1-Day_2019353`, `Band_2-Day_2020033`, `Band_6-Day_2020033`) and 3 winter NDVI.
  - Bare (21 predictors): **8 of 21 are winter periods**: `Band_2-Day_2019353`, `Band_7-Day_2019305`, NDVI 2019305/353/2020033, and
    band 31 for 2019305/353/2020033.
- **Both models also read period-ranked metrics:** CGreenMean, CThermMean, CSortedRank, CUnsMean/Amp. These rank or average across the
  periods, so a winter period can be chosen, or averaged in, as if it were an observation.
- **Decision 19.6:** h16v01, the main ice-sheet training tile, was trained on the original NDVI but is predicted with the Greenland-corrected NDVI.

## 4. Hypotheses

- **H1, winter periods carry no information and drive the result.** Ice pixels below 100 differ from ice pixels at 100 mainly in predictors
  built from polar-night or twilight periods: raw winter bands, and ranked metrics that pick a winter period.
- **H2, the training rows teach it.** Ice pixels below 100 land, in the forest's leaves, with training rows labelled below 100: non-ice bright
  surfaces, or ice-margin and nunatak rows. Or the snow/ice training rows themselves carry labels below 100.
- **H3, the h16v01 mismatch** (decision 19.6). Ice predictors at prediction time are outside the range the model learned for ice in training.

These can all be true at once. The steps below measure each one.

## 5. Steps (read-only; scratch output only; one node, about 1-2 h)

Tiles: **h16v02** (problem A) and **h17v01** (problem B), with **h16v01** for H3. Maps and models are v1.1.0: T60 and B60S0, MEDIAN.
Metrics come from the v1.1.0 metric folders (`manifests/tasks.txt` of `cand60med_20261004`).

1. **Predictor contrast (H1).**
   - Groups: ice pixels with bare = 100 and tree = 0 ("clean"), against bare < 100 ("low"), with bare < 90 separately on h17v01.
   - For each predictor, compare the groups' distributions: medians, the share of fill-like values (0, 10000, very negative), and a
     separation score.
   - Group the predictors by type: summer raw, winter raw, period-ranked.
   - **H1 holds** if the predictors that separate the groups are mostly winter or ranked ones, and the summer predictors look alike.
2. **Which periods the ranked metrics choose (H1).** For the ranked predictors the models use, count on low against clean pixels how
   often the chosen periods are polar-night or twilight periods. These are periods whose inputs all have solz ≤ 0 or ≥ 65, known from the
   1-Day files, as in the count run.
3. **Leaf neighbours (H2).**
   - For a sample of about 5,000 low and 5,000 clean pixels per tile, run `apply()` on the 60 trees to find each pixel's leaves.
   - List the training rows that share those leaves: their tiles, labels, and whether they lie on snow/ice (MCD12, selection only).
   - **H2 holds** if low pixels share leaves mostly with rows labelled below 100 that come from elsewhere.
4. **Snow/ice training rows (H2, H3).**
   - Count the training rows on snow/ice for each tile, and give their label distribution, tree and bare.
   - For h16v01, compare each training row's stored predictors with the same pixel's predictors in the v1.1.0 metric folder. Report how
     many columns differ and by how much.
   - **H3 holds** if the ice rows' training values sit outside what the maps feed in.
5. **Write-up:** `Reports/Ice_Sheet_Diagnosis_Results.md`, with the findings for each hypothesis and, from them, what a pipeline fix would be.

## 6. What the answer could lead to (not decided here; each one is a later plan)

- **If H1:** a metrics fix. Periods with no real observation get no weight in raw-period and ranked predictors. This differs from the
  composite rules tried so far, which only changed what value such a period holds; the ranking and the models still saw it. It would need
  a retrain.
- **If H2:** a training-data fix, such as the label range on ice and margin rows, or which rows the ice pixels learn from. This would also
  need a retrain.
- **If H3:** rebuild h16v01's training rows from the current metrics (decision 19.6) at the next retrain.

Under the project rules, any of these would be evaluated against v1.1.0: held-out labels, then maps against C6, C6.1 and the C code, with
D9 (b) as the pass rule.

## 7. Decisions for the project lead

- **E1:** approve this read-only diagnosis (about 1-2 h, one ngaproc node, after the fill-only test finishes).
- **E2 (withdrawn 2026-10-05):** the summer-snow ice rule (decision 19.5) is a tested candidate only, not in any product or code; no
  decision is needed, and it stays as it is.
- **E3:** the fill-only test (`EXP_fillonly_20261005`) reports on the same ice-sheet counts. If it makes the ice worse, that is evidence for
  H1, and it is not adopted.

## 8. Files

- Baseline counts: computed by `ice_counts.py` (session scratch); the numbers are in section 2. The script will be copied into the run's `code/`.
- Maps: `FullRuns/v110_sd_20261004/out/maps/v1.1.0`. Models: `trees_counts_2x2_20261003/Python/models/T60`,
  `bare60_20261004/Python/models/B60S0`.
- Splits: tree rows with `keysTrain`/`keysTest` in `rf_fallback_counts_20260930/split/`. Bare: `nonveg_bit15_models_20261001/Python/training_split/`.
