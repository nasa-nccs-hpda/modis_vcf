# Plan: diagnose cloud-fallback composites, and find QA information layers for the RF model (option 1)

**Written 2026-09-30. Status: stage A (steps 1, 2 and 4) done 2026-09-30 ~15:30, results in `Cloudy_Fallback_Diagnosis_Results.md`. Stage B awaits the project lead.**

**Staging and token budget (project lead, 2026-09-30).** Run this plan in a **fresh session** (this one grew long, and every step re-sends it). Two stages, with a checkpoint between them:
- **Stage A: steps 1, 2 and 4** (about 75-150k tokens). Why C is less affected, and how widespread fallback is. Then stop and report to the project lead.
- **Stage B: steps 3, 5 and 6**, only if stage A shows it is worth it and the project lead approves. Stage A may make the pixel trace (step 3) unnecessary and narrow the search (step 5).
- **Keep it lean:**
  - Search the C code with grep for the fallback logic instead of reading whole files.
  - Web search (step 5): at most about 5 targeted searches, summaries only, no full-page reads unless one source is clearly central.
  - Reuse existing outputs (drop mask, `drop_diagnosis_h12v09_periods.csv`, qaC metrics) instead of recomputing.
- Step 7 (report) is written at the end of each stage.
- **Starting prompt for the new session:** "Read Reports/README_START_HERE.md and Reports/Cloudy_Fallback_Diagnosis_Plan.md, then do stage A. Keep token use low."
Working configuration: Random Forest with the bit-15 QA screen (project lead, 2026-09-30).

## Why

On h12v09 (eastern Amazon, Pará) about 103,000 dense-forest pixels drop 30 or more points below the historical MOD44B (`C_vs_Historical_h12v09_20260930.md`). They are persistently cloudy pixels. When no day in a compositing period passes QA, the Python pipeline fills the period with the mean of **all** days, clouds included. Drop pixels fall back in about 5 of 8 periods, against about 3 for control pixels. Their wet-season composites are bright in red and low in NDVI, so the model reads less forest. Every Python model does this; the bit-15 screen falls back less often than bits 7-8 did. **The operational C-code product is less affected, and we do not know why.** This plan finds out why, measures how widespread the problem is, and collects ways to give RF information about data quality. It changes no product code and trains no models; option 2 (a screening test of new rules or layers) is planned afterwards from its results.

## What is known about the Python side

`modis_vcf/model/CompositeDayFile.py`, lines about 225-285 (the delivered branch and `Python_code` are the same):
- Each period combines several daily MOD44C inputs. Case 1: mean of the days whose QA mask is good. Case 2: mean of all days.
- Where no day is good (case 1 empty), the composite takes case 2 (line 272), clouds included.
- It already writes two per-period flags: **`outQa`** (1 = fell back to the all-day mean) and **`outCloud`** (1 = every input was cloudy, `cloudTotal == 3`). These are saved, but as far as is known they are not used as predictors or anywhere downstream. Step 1 confirms this.

## Steps

**Step 1: Python fallback mechanics (code reading, about 30 minutes).**
- Confirm the number of inputs per period and the period length (8 periods per year; `drop_diagnosis.py` calls them 48-day composites).
- Find where the composites go next (`Metrics.py`, `CMirrorMetrics.py`, `BuildTraining.py`, `VcfPredict.py`), and whether `outQa` or `outCloud` is read anywhere.
- Record which of the 64 RF predictors come from fallback-prone periods: all of them, or only some metric types (for example, per-period band values vs annual max/min/median).

**Step 2: C fallback mechanics (code reading, about 1 hour).**
- In the C code (`C_code/pge61/MOD_PR44B1`: `fix_bands.c`, `make_metrics.c`, and the compositing code they call), find what C does for a period with no good day. Possibilities: skip it, keep the last good value, take the minimum red or maximum NDVI, or mark it missing so the annual metrics ignore it.
- Record the rule, with file and line references.

**Step 3: trace drop pixels through both codebases (about 1 hour, one tile).**
- Use the existing drop mask (`stable_pixels_20260930/drop_diag/drop_mask_h12v09.npy`) and 10 drop pixels and 10 control pixels. For each fallback period, compare C's composite and metrics with Python's. The period-by-period values are in `drop_diag/drop_diagnosis_h12v09_periods.csv` and the qaC metrics.
- If C per-pixel metrics for h12v09 are not on disk, generate them for this tile only with the existing `gen_metrics` tool into a new folder (about 30-60 minutes, detached).
- Answer: does C avoid the cloud-contaminated composites, and how?

**Step 4: how widespread is it (about 30 minutes of compute on ngaproc, detached).**
- For each of the 63 tiles, from the bit-15 metric build (`qaC_full63_retrain_20260929/Python/metrics_C/`), count the fallback periods (0-8) per land pixel.
- Report per tile the share of pixels with 0, 1-2, 3-4 and 5 or more fallback periods, and name the worst tiles (expected: humid tropics, cloudy mountains and coasts).
- **Link to accuracy:** RF (bit 15) error by fallback count, against the held-out 20% labels (the training split has pixel positions) and against the historical MOD44B. If error rises with the count, that sizes the gain available beyond h12v09.

**Step 5: literature and web search (about 45 minutes).** How others give a random forest (or similar tree models) information about input data quality alongside the real data. Examples to look for:
- Quality or observation-count layers used directly as predictors (for example, the clear-observation counts used in Landsat and MODIS land cover and tree cover mapping).
- Weighting training samples by data quality (scikit-learn `sample_weight`).
- Treating contaminated composites as missing instead of filling them (recent scikit-learn random forests accept NaN inputs; check the container's version), and how missing-aware splits behave.
- Quality-aware compositing rules (for example, the MODIS vegetation index rule of taking the maximum NDVI, or constrained view angle, as used in MOD13), and temporal gap filling (interpolation from neighbouring periods or years; smoothing methods such as Savitzky-Golay or HANTS).
- Per-pixel confidence or uncertainty outputs derived from QA inputs.

The output is a short summary with sources, and a note on which ideas fit our pipeline and data. Sources are cited; nothing is downloaded or installed.

**Step 6: candidate information layers from the data we already have (design only, about 30 minutes).**
For each candidate: how it is computed, what it costs, whether RF can use it at prediction time on every tile, and the risks. Examples:
- Number of good-QA periods per pixel (0-8), or its complement, the fallback count (`outQa` summed).
- The per-period fallback flag and all-cloudy flag (`outQa`, `outCloud`), as predictors or as a mask that turns a fallback period into missing.
- The share of good days within each period. This needs a count kept in `CompositeDayFile`, which currently keeps only the means.
- A cloud-contamination score per period: brightness in red and blue, or NDVI, relative to the pixel's own clear-sky median across the year.
- The same pixel's value in the previous or next year, as a fallback or a predictor.

Also noted for each candidate: whether it could be written out as a quality layer next to the planned SD layer (README item 8).

**Step 7: report.**
- `Cloudy_Fallback_Diagnosis_Results.md` with the C-vs-Python comparison, the prevalence table, the literature summary and the ranked candidates.
- A proposed design for option 2: the rules or layers to test, the tiles (h12v09 plus the worst tiles from step 4, for example one Congo and one Southeast Asia tile), and how to score them (held-out labels, stable pixels and NEON; a change of about 1 point MAE or less counts as a tie).
- README and summary updated.

## Cost, and what is not touched

- About 4-5 hours in total, mostly reading and writing. Compute is light: step 4 about 30 minutes, step 3 up to about 1 hour if the C metrics have to be generated.
- Detached runs go on ngaproc201/202 (ssh `-o CheckHostIP=no`, `module load singularity/4.3.1` in a login shell) or forest nodes, with status markers.
- New outputs only, in a new folder `FullRuns/cloudy_fallback_20260930/` (the scripts refuse to reuse it) and in `Reports/`.
- No product code changes, no retraining, nothing in rlgill's repositories or on the pushed branch.

## Decisions for the project lead

- Go-ahead for steps 1-7.
- Whether step 3 may generate C metrics for h12v09 if they are not on disk.
