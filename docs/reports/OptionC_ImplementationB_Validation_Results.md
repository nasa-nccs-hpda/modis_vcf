# Option C, implementation B: C-mirror metrics validated on two full tiles (h09v05, h12v02)

2026-09-19. Written by the session that made "implementation B" (files named `CAttributes.py`, `CMirrorMetrics.py`,
`validate_cmirror_vs_c.py`). A second session made an independent "implementation A" of the same work in the same
run root (`CEquivalentMetrics.py`, `validate_c_equivalent_metrics.py`) and reported on it in
`Predictor_Subset_C_vs_Python.md`. This page covers implementation B only. Nothing was written to `Python_code/`,
`VCF_outputs/` or the real repo; everything is under
`MOD44B_VCF_Development/FullRuns/full63_c_vs_py_20260919-0806/`.

## What was built (in the run root's copy of the Python code, `code/python_code/modis_vcf/model/`)

- `CAttributes.py`: C's 64 tree-cover and 21 bare attribute tables (checked against `MOD44B3.c` at run time and equal),
  the layout of C's 324 metrics, the mapping from each C attribute to a Python metric name, and the list of families
  needed (19 families; 77 distinct attributes over both models).
- `CMirrorMetrics.py`: the 252 derived C metrics (six group statistics of the unsorted values, each band sorted, and
  the same statistics with the periods ordered by NDVI and by Band_31), using C's integer arithmetic (truncating
  division, exchange sort with no tie swaps). It installs one `metric<Family>` method per needed family on `Metrics`,
  so `getMetric` and `getMetricFromRf` treat them like existing metrics (names such as `CUnsMean13-Band_6`).
- `Metrics.py` (copy): two lines appended at the end to install them.
- `code/validation/validate_cmirror_vs_c.py` and `run_option_c_validation.sh`: the validation and its detached launcher.

## How it was run

Fully detached: `setsid nohup` on forest203 (h09v05) and forest204 (h12v02), parent PID 1, no terminal, write-once
output directories, all inputs bound read-only, status in `status/optionC_validation_<tile>.status`, results in
`Python/validation_optionC/<tile>/`. Each run took about 14 min (h09v05) and 11 min (h12v02); the Metrics class needed
about 3.5 min per tile to compute and write the 19 family files (2.4 GB per tile).
Two earlier launches failed within seconds and were removed and rerun: the container lacked the MOD44C input
directory, and an existing `Metrics.__init__` bug (uses `sys` without importing it when no logger is passed) needed
the script to pass its own logger. The Metrics code itself was not changed for that.

## Results (all C records: 22,933,948 pixels for h09v05, 20,522,188 for h12v02)

| Check | h09v05 | h12v02 |
|---|---|---|
| Formula: C's own unsorted values -> the 252 derived columns | 100% exact, every column | 100% exact, every column |
| Delivered through the Metrics class (tifs written and read back), 77 attributes with 100% exact agreement | 50 of 77 | 37 of 77 |
| The 27 attributes that involve NDVI or the NDVI-ordered families: mean exact fraction | 55% (range 26-97%) | 38% (range 1-97%) |
| The other 50 attributes: mean exact fraction | 100% (all 50 exact) | 98.6% (37 exact; the rest 87-97%) |

Name resolution through `getMetricFromRf` worked for the checked names, and the values read back from the written
tifs are what were compared.

## What limits parity: the inputs, not the formulas

1. **NDVI.** C computes NDVI per 16-day composite (`fix_bands.c` lines 236-247: `1000*(B2-B1)/(B2+B1)`, stored as a
   short, so truncated, and 0 when B1 + B2 = 0) and then averages the composites of each period. Python
   (`Metrics.getNdvi`) computes it from the already-averaged monthly Band_1 and Band_2 and stores it rounded.
   Checked on 300,000 h09v05 pixels: Python's NDVI equals that ratio applied to C's own monthly Band_1 and Band_2
   within +/-1 on 100% of pixels (exact on 50%), while C's NDVI differs from the same ratio by more than 1 on about
   14% of pixels and by more than 5 on about 2%. Python NDVI against C NDVI is exact on 34%, within 1 on 84%.
   NDVI is also the sort key of the greenest-ordered families, so those inherit the mismatch even for other bands
   (96-97.5% exact on h09v05, much lower on h12v02). Making Python match would need NDVI computed per composite from
   the `1-Days` files and averaged like C; not attempted.
2. **h12v02 band values.** Python's unsorted Band_1..Band_7 and Band_31 equal C's on only 87% of pixels for two
   periods (2019353 and 2019305 about 97%). The other session and earlier reports attribute this to the `solz <= 0`
   / no-observation mechanism at this latitude (`Polar_Night_Zero_Observation_Finding_h12v02.md`); I did not verify that here.

## Reconciliation needed (user decision)

Two independent implementations of the same 252-metric derivation both reproduce C exactly from C's own inputs.
Only implementation B is currently wired into the copy's `Metrics.py`. One should be chosen, or the two compared
line by line, before the full run; I have not changed or removed anything of implementation A.

## Not done

Predictor lists for `rfFitFromPge61.py` generated from `CAttributes`; the NDVI fix; the parallelization edits; every
full-run stage (metrics for the other 61 tiles, C chain, fit, predict, comparison).

## NDVI difference statistics (added 2026-09-19)

Python's unsorted NDVI against C's NDVI columns (56-63), every C pixel and all 8 periods; values are NDVI x 1000.
Script `ndvi_difference_stats.py`; full outputs `ndvi_difference_stats_<tile>.txt` (both in `Reports/`).

| | h09v05 | h12v02 |
|---|---|---|
| Pixel-period pairs compared | 183,471,584 | 164,177,504 |
| Exact matches | 32.0% | 16.8% |
| Not exact | 68.0% (124.7M) | 83.2% (136.6M) |
| Not exact, involving a sentinel/fill value | 1,587 | 87,046 |
| Absolute difference of the rest: min / mean / median / p99 / max | 1 / 4.2 / 1 / 75 / 1,513 | 1 / 41.0 / 9 / 264 / 1,909 |
| Same as % of NDVI 1.0: min / mean / max | 0.1 / 0.42 / 151.3 | 0.1 / 4.1 / 190.9 |
| Relative to C, 100*abs(d)/abs(C), C != 0: min / mean / median / max | 0.10 / 2.3 / 0.49 / 20,000 % | 0.10 / 47.7 / 13.6 / 65,400 % |
| Pairs where C = 0 (relative difference undefined) | 51,251 | 3,969,454 |
| Mean of (Python - C); share of pairs where Python is higher | -1.5; 78.8% | -18.2; 49.1% |
| Off by exactly 1 / 2-5 / 6-50 / over 50 | 68.6 / 21.5 / 8.1 / 1.9 % | 27.3 / 17.1 / 29.4 / 26.3 % |

Notes: the maximum relative percent comes from pixels where C is near zero and is not meaningful by itself. The largest
mean gaps by period are 2019065 (7.0), 2019305 and 2019353 (about 6.0) on h09v05, and 2019257 (111.4), 2019113 (63.1) and
2019353 (50.8) on h12v02. On h09v05 most gaps are 1-5 units, consistent with the averaging-order and rounding difference
found earlier; on h12v02 half exceed 5 units and a quarter exceed 50, which is more than rounding and coincides with
Python's band inputs also differing from C there. That link was not tested.

## Resolution (2026-09-20)

- **Reconciliation:** the user chose implementation B (this page); it is the implementation used in the 63-tile run. The other writer's implementation
  was left unused in the pipeline.
- **Bug fixed after this validation:** in a fresh `makeMetrics` run the mirror families were computed from the raw floating-point unsorted values instead of
  the written int16 tif, so NDVI-related bands differed (21 of 44 bands on h10v05). `CMirrorMetrics.computeWanted` now reads the unsorted values back from disk.
  The validation above used the written tif and was not affected. All five pilot tiles were regenerated and match on a 400-row block; the 63-tile run used the fix.
- **"Not done" list above, now done:** predictor lists were generated from `CAttributes` (64 tree, 21 bare); the parallelization edits were made in the run-root
  copy; every full-run stage ran. **Still not done:** a C-style NDVI (the difference was accepted for now).
