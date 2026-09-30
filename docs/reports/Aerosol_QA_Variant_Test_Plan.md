# Plan: test aerosol QA variants on the VCF (bit 15 vs current screen vs none)

**Written 2026-09-29. Status: approved and run; finished 2026-09-29 ~17:00. Results: `Aerosol_QA_Variant_Test_Results.md`. Follow-up: `Aerosol_QA_C_Full63_Retrain_Plan.md`.**

## Why

The PGE72/MOD44C investigation (`/explore/nobackup/people/mcarrol2/MOD44C_PGE72_Investigation/README_START_HERE.md`)
found three things that matter here:

1. **The current PGE61 aerosol test is broken.** It rejects when MOD44C `state` bits 7-8 = 3, but PGE72 ORs
   the BRDF flag into bit 7 and day-of-year parity into bit 8. The test rejects every low-aerosol pixel from an
   odd day: 26-70% of otherwise usable pixels in 32 tile-periods, while only 0.1-16% of usable pixels are truly
   high-aerosol. Every C and Python product since 2026-09-15 used it, including the 63-tile run
   `full63_c_vs_py_20260919-0806` and every model built from it.
2. **The historical MOD44B had no aerosol screen.** The old C code read bits 13-14, which PGE72 never sets.
3. **`state` bit 15 is PGE72's overall QC flag.** It is `tlevel >= 64512`: valid data, all bands valid, no cloud
   or shadow, solar zenith < 65 degrees, and aerosol low or average. It is not affected by the bit overlap.
   At production time, PGE72 reads the internal MOD09GQK, MOD09GHK and MOD09GST (PCF template), which carry more
   observations than the public MOD09GA/GQ. So the investigation's per-pixel checks against MOD09GA, which made bit 15
   look weak as an aerosol flag, used the wrong reference. The project lead prefers testing bit 15 over
   recovering each observation's day from MOD09 for every MOD44C file, which is slow and can never be exact.

The question for this test: **does the QA choice change tree cover enough to matter, and is bit 15 a
reasonable production screen?**

## Variants

All variants keep PGE61's other QA tests unchanged: cloud = 0, no shadow, no adjacency, 0 < solar zenith < 65
(`ProductTypeMod44.createQaMask`, `Python_code/modis_vcf/model/ProductTypeMod44.py` line 68).

| Variant | Aerosol part of the QA mask | Source |
|---|---|---|
| **A: none** | no aerosol test (the historical behaviour) | new run |
| **B: current** | reject when bits 7-8 = 3 | **reuse** the existing 63-tile metrics, no rerun |
| **C: bit 15** | require `state & 0x8000` (bit 15 set) | new run |

Variant C in code: replace `(aerosol != 3)` with `((state & 32768) != 0)`. Variant A: drop the term.
`createCloudMask` is not changed.

## Tiles

Proposed: **h12v09** (Amazon; strongest aerosol signal, fire season; true-high share up to 12%) and
**h09v05** (SW US; low aerosol, the current screen loses 50% of good pixels there). Both have Stage 1 pass-rate
counts in the MOD44C investigation, per-tile deep-dive numbers against the historical MOD44B, and training
labels (`v5.0.3samp`). h19v08 or h18v03 could be added as a third tile if the project lead wants a
high-aerosol, cloudier case.

## Stages

### Stage 0: cheap check before any metrics run (minutes)

On the 23 MOD44CQ periods of the 2019 window for each tile, count the share of valid pixels passing the mask
under A, B and C. This extends the investigation's Stage 1 counts, which did not compute "other QA tests AND
bit 15". It runs on forest203 with plain numpy, reading only `state`, `solz` and a valid-data check.
**Go/no-go:** if C passes almost the same pixels as A (within about 1 percentage point on both tiles), bit 15
adds almost nothing beyond PGE61's other tests. Stage 1 would then be run for A only, and the answer is
"drop the aerosol test". Report back before Stage 1 either way.

### Stage 1: metrics for variants A and C (about 30-35 minutes per tile per variant)

- New run directory: `/explore/nobackup/people/mcarrol2/MOD44B_VCF_Development/FullRuns/aerosol_qa_variants_20260929/`.
  Nothing in `full63_c_vs_py_20260919-0806` is written to.
- Copy that run's `code/python_code` into the new directory twice, `code_A/` and `code_C/`, and change only
  `createQaMask` in each copy. Record `sha256sum` of the changed file and a diff against the original in
  `manifests/`. rlgill's repositories and the workspace `Python_code/` are not touched.
- Run `makeMetrics.py` in the same container, with the same metric list (`metricUnsortedMonthlyBands` plus the
  19 C-mirror families) and the same bind layout as `python_metrics_node.sh`, into fresh output dirs.
  **No symlinks to the old `1-Days` or `2-Composites`**, because the QA mask is applied when day files are
  composited (`CompositeDayFile.py` line 169).
- Memory: a tile peaks near 20 GB; the existing guard (start only with 28 GB free) and watchdog are kept.
  2 tiles x 2 variants = 4 jobs, 2 at a time on forest203 (or split with forest204): about 1-1.2 hours wall clock.
- Launch fully detached (`setsid nohup`, parent PID 1, `timeout 30 ssh -n` for each remote call), with
  `status/*.status` / `.done` / `.failed` markers and logs under the run directory.

### Stage 2: predict with the existing models (a few minutes per tile per model)

Predict tree cover from each variant's metrics with the three trained models already in use: baseline
(`full63.../Python/models/pcttree.bin`), Random Forest and XGBoost tunedB
(`regtree_alt_models_20260927/Python/models/`), using the same patched predict script as the zenith retest.
**Caveat:** these models were trained on variant B metrics. This stage shows the effect of the input change
only. Retraining on variant C metrics needs all 63 tiles rebuilt and would be a separate plan.

### Stage 3: compare (numpy only, minutes)

Per tile, variant and model:
- MAE, bias and correlation of tree cover against the `v5.0.3samp` training labels, as in the zenith retest.
- MAE and bias against the historical MOD44B, as in `Model_vs_Historical_DeepDive_20260929.md`.
- QA pass rate per period, and the number of pixels whose metrics fall back to "no good observation".
- Tree cover difference maps C - B and A - B, to see where the change lands (smoke regions, cloud edges).

## Estimate

Stage 0: about 10 minutes. Stage 1: about 1-1.2 hours on one or two nodes. Stages 2-3: about 30 minutes.
Total about 2 hours, most of it unattended.

## What would change the conclusion

- If A and C give nearly the same tree cover and both beat B against the labels: the broken screen was hurting.
  Bit 15 is optional, and "no aerosol test" is the simplest fix.
- If C beats A on the high-aerosol tile (h12v09) and matches it on h09v05: bit 15 is doing useful screening,
  and it is the candidate for production.
- If B is best: the extra pixels A and C admit are harmful, which would need a closer look (for example BRDF or
  odd-day effects) before changing anything.
- Results against the historical MOD44B are expected to favour A, which is what produced it. The training labels
  are the fairer reference.

## Standing conditions

- rlgill owns pge61, pge72 and `modis_vcf`: no commits or pushes; changes are made only in copies under the run
  directory.
- Nothing is launched until the project lead approves this plan. Stage 0 results are reported before Stage 1.
- Long runs are fully detached on forest203 (forest204/205 if needed), with node load checked immediately
  before launch.
- A results report (`Aerosol_QA_Variant_Test_Results.md`) is written when the run finishes.

## Approval and tile change (2026-09-29 ~14:25)

The project lead approved Stage 0 on h09v05 and h12v09 and added **h16v01 (Greenland) as an outlier case**.
Stage 1 has not been approved yet.

## Stage 0 results (2026-09-29, finished 14:32)

Run folder `FullRuns/aerosol_qa_variants_20260929/` on forest203 (detached, 4 minutes).
Code: `scripts/stage0_qa_variant_counts.py` and `scripts/run_stage0.sh`. Output: `data/stage0/stage0_<tile>.csv`.
Logs: `logs/`. Marker: `status/stage0.done`. MOD44CQ from `/css/.../dev/<year>`. h16v01 is missing A2019337 and
A2019353 (21 of 23 periods); the other tiles have all 23. Pixel-weighted over all periods, as % of valid pixels:

| Tile | Bit 15 set | Pass A (none) | Pass B (bits 7-8) | Pass C (bit 15) | Share of A removed by B (per-period range) | Share of A removed by C (per-period range) |
|---|---|---|---|---|---|---|
| h09v05 (SW US) | 93.1 | 84.1 | 41.9 | 83.8 | 50.1 (26-66) | **0.3** (0-2) |
| h12v09 (Amazon) | 57.4 | 39.5 | 22.7 | 38.7 | 42.6 (20-69) | **2.1** (1-14) |
| h16v01 (Greenland) | 6.6 | 7.7 | 5.1 | 6.1 | 33.8 (2-54) | **20.7** (3-94) |

Observations:
- **Bit 15 is a small, targeted cut where aerosol is low or moderate.** On h09v05 it removes 0.3% of the pixels
  that pass the other tests, and on h12v09 2.1% overall but up to 14% in a single period. The current screen removes
  43-50%. The h12v09 figure is close to the investigation's true-high-aerosol shares for that tile (1.5% at the
  fire peak, up to 12% in December), which is what an aerosol screen should remove.
- **The plan's go/no-go rule, read literally, says C is not needed:** C is within 1 percentage point of A on both
  original tiles (0.3 and 0.8 points). But the h12v09 per-period peak (14%) shows the effect is concentrated in
  specific seasons, which a whole-year average hides.
- **h16v01 behaves differently.** Very few pixels pass any variant (under 8%), and bit 15 removes 21% of them
  (up to 94% in one period). B and C disagree on many pixels there (C removes 17% of A's pixels that B keeps).
  A likely cause is snow and ice: the MOD44C investigation noted that MOD09's internal snow flag also clears
  bit 15 (`MOD44C_PGE72_Investigation/Reports/Stage3_Day_Recovery_Plan.md`). Not verified here.
- Stage 1 for 3 tiles x 2 variants = 6 makeMetrics jobs, about 30-35 minutes each. forest203 had 104 GB free at
  launch, enough for 3 at a time under the 28 GB guard, so about 1.2 hours wall clock.

## Revised scope after Stage 0 (approved 2026-09-29 ~14:50)

The project lead chose **variant C only** (bit 15), accepting the risk of not having A, to save time. This
revised scope replaces Stages 1-3 above:

1. **Stage 1 (C only): metrics for h09v05, h12v09 and h16v01**, built as described in Stage 1 above from a copy of
   the 63-tile run's code (`code_C/`) with only the aerosol term of `createQaMask` changed to
   `((state & 32768) != 0)`. Three tiles run in parallel on forest203, about 35-40 minutes. Output:
   `FullRuns/aerosol_qa_variants_20260929/Python/metrics_C/<tile>/`.
2. **Stage 2: prediction with the existing models, plus a metric-shift check**, chained after Stage 1 in the same
   detached job:
   - Predict tree cover from the C metrics with the three existing models (baseline, RF, XGBoost tunedB), using the
     same predict script and arguments as the zenith retest. Compare with the B predictions of the 63-tile runs:
     MAE, bias and correlation against the `v5.0.3samp` labels, and the C - B tree cover difference.
   - **Metric shift:** at the labelled training pixels, compare every band of every metric file in `3-Metrics`
     between B (the 63-tile run) and C: correlation, and mean absolute difference relative to B's standard
     deviation. Small shifts mean the existing models are a fair test of C; large shifts mean the models must be
     retrained before C can be judged.
   - How to read the result: the existing models were trained on B metrics. If C with these models matches or
     beats B against the labels, that is a strong signal for C. If C looks worse, the result is ambiguous: it
     could be the train/predict mismatch rather than C itself.
3. **Summary for the project lead's decision** on the next step: rebuild C metrics for all 63 tiles and retrain
   the three models (about 2.5-3 hours of metrics on forest201-205, plus training time to be taken from the
   earlier training logs). This is a separate plan and is not run here.
4. **Deferred:** variant A on h16v01, only if h16v01's C results look bad (bit 15 removes 21% of pixels there,
   possibly because of snow). The snow check (bit 15 against state bit 5) is also deferred.

Not included: the comparison against the historical MOD44B (it can be added later from the predictions).

**Resume info (2026-09-29 16:30).** Stages 1-2 (C only) launched 14:48 on forest203, fully detached
(launcher PID 1962892, parent PID 1). Stage 1 finished 15:19 (all 3 tiles rc=0). Stage 2 (metric shift, then
3 model predictions per tile) finished h09v05 and h12v09; h16v01 expected about 17:05. Check
`FullRuns/aerosol_qa_variants_20260929/status/stage12_C.done` (or `.failed`, and `stage12_C.status`).
Results: `data/stage2/stage2_results.jsonl` (predictions vs B) and `data/stage2/stage2_metric_shift.jsonl`
(per metric file and band). Log: `logs/stage2_C.log`. A detached waiter on forest203 (PID 2006845, launched
16:31) writes tables-only summary `Reports/Aerosol_QA_Variant_Stage2_Tables.md` when Stage 2 ends (status
`status/summary.status` / `.done` / `.failed`). Next: summarise for the project lead's decision on step 3
(63-tile C rebuild and retrain), with the label MAEs marked as in-sample (see the correction below).

**Correction (2026-09-29, found on the first Stage 2 result):** the `v5.0.3samp` labels are the models' own
training pixels, so MAE against them is **in-sample for B**: the models were fit on exactly these pixels' B
metrics. C feeds the same pixels different inputs and will score worse even if it is the better screen (first
result, h09v05 baseline: MAE B 1.49 vs C 3.10). The label comparison in Stage 2 therefore measures how far C's
inputs move from what the models memorised, not which screen is better. It is not "the fairer reference", as
written above. A fair test needs retrained models evaluated on held-out pixels or tiles (step 3), or an
independent reference.

## Finished (2026-09-29 ~17:00)

Stage 2 finished for all three tiles; all markers in `status/` are `.done`. Results and the decision summary for step 3
(63-tile C rebuild, retraining, and a held-out B-vs-C comparison) are in `Aerosol_QA_Variant_Test_Results.md`.
Nothing is running. Step 3 needs its own written plan and approval.
