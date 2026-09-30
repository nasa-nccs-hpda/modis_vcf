# Do C and Python use the same subset of metrics? And does Python run the Monte Carlo search?

**Later note (2026-09-20):** what follows is true of the 63-tile run, which did not search. A separate stand-alone Monte Carlo search over all 324 metrics was run afterwards; see `Feature_Importance_and_Monte_Carlo_Results.md`. It ranked a different top 64 (20 shared with C's 64) and its model scored slightly lower.

Checked 2026-09-19. Read-only checks; no pipeline code or data was changed. Scripts:
`decode_c_attributes_vs_python_preds.py` (decodes C's attribute indices and compares names) and
`check_unsorted_layout_c_vs_python.py` (compares Python's unsorted-band tif with C's `metrics.bin` values).

## Answer

1. **Monte Carlo: confirmed it will not run.** The Python training step is `view/rfFitFromPge61.py` (Option B):
   a direct fit against predictor lists hard-coded in the file (`treeCoverPreds`, `barePreds`), no search.
   `MonteCarloSim` / `monteCarloSim.py` is referenced by none of the Python entry points in the pipeline
   (`makeMetrics`, `buildTraining`, `rfFitFromPge61`, `vcfPredict`), by the driver scripts in `Reports/`, or by the
   full-run plan. The only mention is a docstring in `build_xy_from_training.py` saying it reuses the
   `MasterTraining` data class, which does no search. `pcttreeTopN.bin` is just the pickled hard-coded list.
2. **Same subset of metrics: NOT confirmed. The evidence says they differ.**

## C's predictors

C's B3 passes 324 metrics per pixel to the Weka models and selects a fixed subset by index
(`MOD44B3.c` lines 94-99): **64 attributes for tree cover, 21 for bare** (the tree-cover header lists them as
`n000...`). Layout of the 324 (`make_metrics.c`, and the same layout `compare_c_python_metrics_*.py` documents):
band-major, `idx = band*8 + period` for the 72 unsorted values (bands in order Band_1..Band_7, NDVI, Band_31;
periods 2019065, 113, 161, 209, 257, 305, 353, 2020033), then blocks of group statistics, per-band sorted
values, greenest-sorted statistics, and thermal-sorted statistics.

## Python's predictors

`rfFitFromPge61.py`: **60 tree-cover** and **18 bare** names, plus 4 tree and 3 bare names commented out as
`N/A-...` (64 - 4 = 60 and 21 - 3 = 18, which suggests the list was translated from C's list).

## Evidence that the sets differ

- **Python's unsorted values equal C's columns band-major (verified on data).** For h09v05, all 72 bands of
  Python's `UnsortedMonthlyBands.tif` were compared with `h09v05.2019.metrics.bin` on about 20,000 sampled
  pixels. For every one of the 72, the best-matching C column is the band-major index `band*8 + period`
  (Band_1..Band_7 and Band_31 match exactly, 100% of sampled pixels; the date-major alternative matches only
  2 of 72). So Python's name `(band, date)` and C's column `band*8 + period` are the same quantity.
- **Python's predictor list follows a date-major layout instead.** Decoding C's attribute indices as
  `idx = period*9 + band`, with band order Band_1..Band_7, Band_31, NDVI, reproduces Python's unsorted predictor
  names **exactly: 24 of 24 (tree) and 15 of 15 (bare)**. Decoded with C's real band-major layout, only
  **8 of the 24** tree and **5 of the 15** bare unsorted predictors are in C's set; 16 and 10 are not. My
  reading (a hypothesis, not confirmed with the list's author) is that the list was made by translating C's
  attribute numbers with the wrong layout.
- **Other families do not have a same-definition counterpart.** C's remaining attributes are group means of
  2, 4 and 6 ranked periods (`make_metrics.c` lines 114-134, 172-232), amplitude terms, and rank values. Python's
  `Lowest{3,6,8}MeanBandRefl`, `Greenest{3,6,8}MeanBandRefl`, `Warmest{3,6,8}MeanBandRefl` average the 3, 6 or 8
  lowest, greenest or warmest values of its cube (`Metrics.py` lines 819-930), which is a different quantity.
  Only Max, Min, Median, Amp, MaxGreenness, MaxTemp and TempMeanWarmest3 were shown equivalent earlier
  (`Bands17_Fix_Validation_Results.md`).

## How much of C's set has a same-definition Python metric today

| | C attributes | with a Python metric of the same definition | without |
|---|---|---|---|
| Tree cover | 64 | 27 (24 unsorted; Band_31 max; Band_2 and Band_31 max-greenness) | 37 (14 unsorted group stats, 11 greenest-sorted stats, 11 thermal-sorted stats, 1 second-highest Band_31) |
| Bare | 21 | 16 (15 unsorted; NDVI minimum) | 5 |

Not in the count: Python's NDVI unsorted values agree with C's within +/-1 on only 66-83% of the sampled
pixels (Bands 1-7 and 31 are exact). This was seen in the layout check and not investigated.

## Consequence for the C-vs-Python comparison

Even with identical labels and the same tiles, the two models currently see different inputs, so the
full-tile comparison would mix an input-set difference with the learner difference. The earlier two-tile
disagreement (`FinalCompare_2tile_Plan_and_Status.md`) may be partly this. Not tested.

## Options (decided 2026-09-19: option C; see the Resolution section at the end)

- **A. Keep the current Python lists** and document that the comparison is confounded. Cheapest; weakest.
- **B. Use only the C attributes that have same-definition Python metrics** (27 tree, 16 bare) for the Python
  fit: an exact but reduced subset; no new metric code. C still uses all 64/21.
- **C. Implement the missing C group-statistic metrics in Python** (37 tree, 5 bare) in the code copy, validate
  each against `metrics.bin` on the two tiles that have it, then fit on the full 64/21 set. Closest match; needs
  new code and validation before the full run.
- Whichever option is chosen, generate the Python predictor lists from C's `attributes[]` and
  `attributes_bare[]` by a script that is checked against `metrics.bin`, not by hand.

## Update 2026-09-19 (later): option C chosen; formulas validated on two full tiles

**Decision:** the user chose option C (Python metrics that mirror all of C's 64 tree / 21 bare attributes).
**Work area:** `FullRuns/full63_c_vs_py_20260919-0806/` (clean run root; code copy in `code/python_code/`,
nothing written to `Python_code/`, `VCF_outputs/` or the real repo).

**Implementation A (this session):** `code/python_code/modis_vcf/model/CEquivalentMetrics.py`, validator
`code/validation/validate_c_equivalent_metrics.py`, detached launcher `code/validation/run_validation_detached.sh`.
It reproduces `make_metrics.c` blocks 72-323 from the 72 unsorted values, including C's integer truncation
toward zero, its strict-comparison exchange sort for the NDVI and Band_31 re-ordering, and its column layout.
It is **not yet wired into `Metrics.py`** (see the reconciliation item below).

**Validation, run fully detached (`setsid nohup`, parent PID 1, no terminal) on forest203 (h09v05) and forest204
(h12v02), read-only on inputs, write-once outputs in `Python/validation_optionC/full_<tile>_CEquivalentMetrics/`,
status in `status/validate_CEquivalent_<tile>.log`:**

| Check | h09v05 (22,933,948 px) | h12v02 (20,522,188 px) |
|---|---|---|
| Formula check: C's own unsorted values -> derived columns 72-323 (252 columns) | **100% exact, every column, every pixel** | **100% exact, every column, every pixel** |
| End to end, restricted to pixels whose 72 inputs equal C's | 100% exact | 100% exact |
| End to end, all pixels (Python's unsorted tif as input), mean over 72 unsorted columns | 92.5% exact | 89.0% exact |

So the group-statistic formulas are correct; every remaining difference comes from the unsorted inputs.

**What limits parity end to end (input differences, not formulas):**
- **NDVI.** Python's unsorted NDVI equals C's exactly on only 27-34% of h09v05 pixels and 1-19% of h12v02 pixels.
  16 of C's 64 tree predictors and 7 of its 21 bare predictors involve NDVI; their exact fraction is 26-31%
  on average (h09v05) and about 13% (h12v02). All greenest-sorted statistics inherit this (NDVI is the sort key).
  Hypothesis, not tested: C averages the NDVI computed per 16-day composite (`fix_bands.c`), Python takes the
  NDVI of the already-averaged monthly bands (`Metrics.getNdvi`); C also truncates to short and uses 0 (Python
  -1001) when B1 + B2 = 0. A C-style NDVI would need the per-composite bands in `1-Days`.
- **Non-NDVI predictors:** h09v05 tree 48 of 64 with mean exact 99.1% (38 at least 99% exact; the rest are the
  greenest-sorted ones, 93-98%); h09v05 bare all 14 non-NDVI predictors exact. h12v02: tree mean 94.7%, bare 97.0%;
  the shortfall is the already-known `solz<=0` no-observation mechanism (see `Bands17_Fix_Validation_Results.md`,
  `Polar_Night_Zero_Observation_Finding_h12v02.md`), which matters most at h12v02's latitude.

**Reconciliation needed (found during this work):** a second writer, same user account, no other Claude session
listed, has been working in the same run root since 08:07: manifests (`config.env`, `tiles.txt`, `start.marker`,
`code_original_sha256.txt`) and, from 08:28, `modis_vcf/model/CAttributes.py`, `CMirrorMetrics.py`,
`code/validation/validate_cmirror_vs_c.py`, `Python/validation_optionC/cheap_check_h09v05/`, plus an
`import CMirrorMetrics; CMirrorMetrics.install(Metrics)` line at the end of the copy's `Metrics.py` (Implementation B).
Its 300,000-record formula check also found 252 of 252 derived columns exact. Nothing of Implementation B was
modified or removed. Two independent implementations agreeing exactly is useful evidence, but only one should be
wired into the pipeline. Resolved: the user chose implementation B (see the outcome section at the end of this document).

**Not done yet:** wiring one implementation into `Metrics.py`; a check through `getMetricFromRf` on the written tifs;
`generate` predictor lists and `rfFitFromPge61.py --predictorFile`; the NDVI question above; the parallelization
edits; any full-run stage.

## Update 2026-09-19 (later): size of the NDVI differences

Measured on every C pixel and all 8 periods (NDVI x 1000; details in `OptionC_ImplementationB_Validation_Results.md`,
outputs `ndvi_difference_stats_<tile>.txt`): where Python's NDVI is not exactly C's, the absolute difference is
min 1, mean 4.2, median 1, max 1,513 on h09v05 (68.0% of pairs not exact), and min 1, mean 41.0, median 9, max 1,909 on
h12v02 (83.2% not exact). As a percent of C's value the mean is 2.3% (h09v05) and 47.7% (h12v02); as a percent of NDVI 1.0
it is 0.42% and 4.1%.

## Resolution (2026-09-20)

The sections above record the investigation and the decision points as they stood on 2026-09-19; this section says how they ended.
- **Option C was chosen and implemented.** Two implementations of the derived C metrics were written independently (the "implementation B"
  files `CAttributes.py`/`CMirrorMetrics.py`, and the other writer's `CEquivalentMetrics.py`, described as "implementation A" above). Both reproduced
  all 252 derived columns exactly from C's own inputs on two full tiles. **The user chose implementation B**, which is the one wired into `Metrics.py`
  and used in the 63-tile run. Implementation A was not used in the pipeline (kept as a cross-check). The "not yet wired" and "pending the user's
  choice" statements above are therefore out of date, as is the "Not done yet" list: predictor lists were generated from C's own attribute
  tables (64 tree, 21 bare), the parallelization edits were made in the run-root copy, and the full run was done.
- **The mirror bug found by the pilots** (a fresh `makeMetrics` run gave NDVI-related mirror bands that differed from the written unsorted tif) was fixed
  before the run; see `OptionC_ImplementationB_Validation_Results.md`.
- **NDVI:** the remaining difference was accepted for now.
- **Effect on the comparison:** with matched predictors, samp labels and all 63 tiles the h12v02 disagreement from the two-tile test is gone
  (`FullRun_Results_C_vs_Python_63tiles.md`); which of those changes accounts for it was not separated.
- Monte Carlo was not run (direct fit only), as stated.
