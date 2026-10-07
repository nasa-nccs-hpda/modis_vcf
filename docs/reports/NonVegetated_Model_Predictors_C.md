# The 21 predictors of C's non-vegetated (bare) model, and what the Python code needs

**Written 2026-09-30.** Fix applied the same evening (see "Fix applied" at the end).

## Where C defines them

- `pge61/MOD_PR44B3/MOD44B3.c:98-99`: `jint attributes_bare[N_ATTRIB_BARE]`, with `N_ATTRIB_BARE 21` (`MOD44B1.h:20`), out of `TOTAL_ATTRIB 324`.
  The local uncommitted diff of `MOD44B3.c` touches only the model directory lookup, not this array.
- `static_coeff/PGE61/MOD_PR44B/header.bare.arff` (PCF LUN 244105) lists the same 21 indices as `n014 ... n223`, then `Y`.
- `Predict.java:54-59` copies `mets[attribute_list2[i]]` into the bare Weka instance. A value of -30000 or less is left as missing.
- The models are `BARE/weka.bare.00-29.model`: 30 serialized Weka **M5P** model trees (all 30 checked), fixed coefficient files shipped with the PGE.
  The prediction is their mean. The tree-cover models are also M5P, but they are trained inside the PGE each year (`MOD_PR44B2/MOD44B2.c:129-133`: `M5P -M 500.0 -x 3`), whereas the bare models are never retrained.

## How the 324 metrics are laid out (`MOD_PR44B3/make_metrics.c`)

**Bands** (k): Band_1..Band_7, NDVI (k=7), Band_31 (k=8).
**Periods** (n), 48-day: 2019065, 113, 161, 209, 257, 305, 353, 2020033.

| Index | Block | Defined as |
|---|---|---|
| 0-71 | unsorted | `k*8 + n` |
| 72-125 | calendar-period statistics, 6 blocks of 9 bands | mean of periods 0-1, 0-3, 0-5, 2-3, 4-5; amp = (p0+p1-p2-p3)/2. No sorting happens before these lines. |
| 126-197 | each band sorted on its own, descending | `126 + k*8 + rank`; rank 0 is the maximum, rank 7 the minimum |
| 198-260 | all bands ordered by NDVI, descending | value at the greenest period, then the same 6 statistics over the greenest ranks |
| 261-323 | all bands ordered by Band_31, descending | the same 7 blocks, over the warmest ranks |

## The 21, decoded

| C index | Meaning | Python name (CAttributes) |
|---|---|---|
| 14 | Band_2, 2019353 | UnsortedMonthlyBands-Band_2-Day_2019353 |
| 18 | Band_3, 2019161 | UnsortedMonthlyBands-Band_3-Day_2019161 |
| 53 | Band_7, 2019305 | UnsortedMonthlyBands-Band_7-Day_2019305 |
| 56, 60, 61, 62, 63 | NDVI, 2019065, 257, 305, 353, 2020033 | UnsortedMonthlyBands-NDVI-Day_... |
| 64, 65, 66, 67, 69, 70, 71 | Band_31, every period except 2019257 | UnsortedMonthlyBands-Band_31-Day_... |
| 86 | Band_6, mean of the first 4 calendar periods | CUnsMean16-Band_6 |
| 109 | Band_2, mean of periods 4-5 (2019257, 2019305) | CUnsMean79-Band_2 |
| 111 | Band_4, mean of periods 4-5 | CUnsMean79-Band_4 |
| 140 | Band_2, 7th highest of 8 (second lowest) | CSortedRank6-Band_2 |
| 189 | NDVI, lowest of 8 (minimum) | CSortedRank7-NDVI |
| 223 | NDVI, mean of the 4 greenest periods | CGreenMean16-NDVI |

That is 15 unsorted values and 6 statistics. Band_31 is 7 of the 21, so thermal matters more here than in the tree-cover model.

## State of the Python code

- **Our runs already use the right 21.** The 63-tile non-vegetated model (`FullRuns/full63_c_vs_py_20260919-0806/Python/models/pctbare.bin`, reused by `qaC_full63_retrain_20260929`) has a saved `pctbareTopN.bin` that holds exactly the 21 names above, in C's order. They came from `CAttributes.BARE_ATTRIBUTES` (option C, `Predictor_Subset_C_vs_Python.md`). The C-mirror families they need (CUnsMean16, CUnsMean79, CSortedRank6/7, CGreenMean16) were validated against `metrics.bin` at the time.
- **The code's default is still wrong.** In the delivery branch, `modis_vcf/view/rfFitFromPge61.py` (around line 140) still has the old hard-coded `barePreds`: 18 names translated from C with the wrong (date-major) layout. Examples are `Lowest6MeanBandRefl-Band_4`, `BandReflMedian-NDVI` and `Greenest6MeanBandRefl-Band_31`. Our runs bypassed it only because they passed a predictor file. The `treeCoverPreds` default has the same problem.
- **Fix:** build both defaults from `CAttributes`, for example `predictorNames(BARE_ATTRIBUTES, periodDays(year))`, and keep `--predictorFile` as an override. Then any fit run without a predictor file gets C's set.

## Open points

- NDVI: earlier work found Python's unsorted NDVI agreeing with C's to within 1 on only 66-83% of sampled pixels (Bands 1-7 and 31 are exact). That was not investigated. 7 of the 21 bare predictors are NDVI-based, so this matters more for the non-vegetated layer than for tree cover. Check this before relying on a Python non-vegetated model.
- C's learner for both layers is M5P (a model tree with linear models at the leaves; tree cover with at least 500 instances per leaf). Python's baseline uses plain bagged decision trees for both layers. The project summary already notes that the non-vegetated layer has not been tried with RF or XGBoost.

## Fix applied (2026-09-30, project lead's go-ahead)

`modis_vcf/view/rfFitFromPge61.py` on branch `mcarrol2/c-python-parity-fixes` (`Git_delivery/modis_vcf`), copied to `Python_code/modis_vcf`. **Committed locally as `2778628`, not pushed.** (Pushed since: on GitHub as of 2026-10-05.)

- The hand-written `barePreds` list is replaced by `CAttributes.predictorNames(CAttributes.BARE_ATTRIBUTES, CAttributes.periodDays(args.year))`.
- A new `--year` option (default 2019) sets the period days in those names. `--predictorFile` still overrides the list.
- The tree-cover default (`treeCoverPreds`) is unchanged. It has the same layout error, but our runs always pass `predictors_pcttree.txt`.

Checks:
- The new default is the same 21 names, in the same order, as the 63-tile non-vegetated model's `pctbareTopN.bin`.
- All 21 are columns of that run's bare `xTrain.parq` (8,631,248 rows).
- `--help` parses.
- No model was refit. The existing non-vegetated model already used these predictors, so its predictions do not change.
