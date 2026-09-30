# Results: aerosol QA variant C (MOD44C state bit 15), three tiles

**Written 2026-09-29 evening. Run finished 2026-09-29 ~17:00; all markers `.done`. Nothing is running.**
Plan and scope history: `Aerosol_QA_Variant_Test_Plan.md`. Auto-generated tables: `Aerosol_QA_Variant_Stage2_Tables.md`.
Run folder: `/explore/nobackup/people/mcarrol2/MOD44B_VCF_Development/FullRuns/aerosol_qa_variants_20260929/`
(results in `data/stage0/`, `data/stage2/stage2_results.jsonl`, `data/stage2/stage2_metric_shift.jsonl`; logs in `logs/`).

## What was tested

Variant C replaces the aerosol part of PGE61's QA mask (`createQaMask`: reject when `state` bits 7-8 = 3, variant B)
with `(state & 0x8000) != 0` (bit 15, PGE72's overall QC flag). All other QA tests are unchanged. The code change was
made only in a copy (`code_C/`) under the run folder. Metrics were rebuilt for h09v05 (SW US), h12v09 (Amazon) and
h16v01 (Greenland), and tree cover was predicted with the three existing models (baseline, Random Forest, XGBoost tunedB),
all of which were trained on variant B metrics. Variant A (no aerosol test) was dropped from Stages 1-2 by the project
lead's decision after Stage 0.

## Findings

1. **Bit 15 removes far fewer pixels than the current screen.** Of the pixels that pass PGE61's other tests, B removes
   50% (h09v05), 43% (h12v09) and 34% (h16v01); C removes 0.3%, 2.1% (up to 14% in one period) and 21%.
   On the two non-snow tiles, C is within 1 percentage point of "no aerosol test" (A). The h12v09 cut is seasonal and
   close in size to the true high-aerosol share found in the MOD44C investigation.
2. **The metrics move moderately, not wholesale.** At labelled pixels, the median correlation between B and C over all
   metric bands is 0.97 (h09v05), 0.95 (h12v09) and 1.00 (h16v01); the lowest single band is about 0.6. The largest shifts
   are on h12v09: CSortedRank0/1 (mean shift +0.4 to +0.5 SD, MAE/SD 0.46-0.61) and the green max, amplitude and 4-6 mean
   metrics. On h16v01 only about 1% of metric values changed, despite C removing 21% of passing pixels.
3. **Tree cover changes are small on h09v05 and h16v01 and moderate on h12v09.**

   | Tile | Mean change C - B (by model) | % pixels within ±5 | % pixels off by > 20 |
   |---|---|---|---|
   | h09v05 | -0.07 to -0.19 | 88-97 | 0.0-0.4 |
   | h12v09 | +1.28 to +1.41 | 69-82 | 0.6-2.2 |
   | h16v01 | about 0 | 99 | 0.0-0.2 |

   XGBoost tunedB is the least sensitive model on every tile.
4. **The label comparison cannot pick a winner.** The `v5.0.3samp` labels are the models' own training pixels, so MAE is
   in-sample for B. C scores worse (for example h09v05 baseline MAE 1.49 B vs 3.09 C; h12v09 baseline 3.71 vs 6.96)
   because its inputs differ from what the models memorised, not necessarily because it is a worse screen. XGBoost tunedB,
   which fits its training data least tightly, shows almost no MAE difference (h09v05 3.65 vs 3.77; h12v09 8.76 vs 8.85),
   which is consistent with that reading.

## What this means

- Keeping B is not a real option: it is known to reject good pixels by bit collision, not aerosol.
- On non-snow tiles, C and A are nearly the same screen. They differ mainly at high latitude (h16v01), where bit 15 also
  appears to drop snow pixels (not verified). So the production choice between A and C matters mostly for snow tiles.
- The existing models cannot judge C. That needs models retrained on C metrics and evaluated on held-out data.

## Decision for the project lead: next step

**Recommended: rebuild C metrics for all 63 tiles, retrain the three models, and compare B and C on held-out data.**

- Metrics: about 2.5-3 hours across forest201-205 (35-40 minutes per tile, three per node under the 28 GB guard),
  from a copy of the 63-tile run's code with only the aerosol term changed, fully detached.
- Training: from earlier logs, RF about 10 minutes, XGBoost tunedB about 30 minutes, baseline about 1 hour (its model files
  were written 16:10 and 17:13 on 09-19). Rebuilding the training table from the new metrics comes first; its time was
  not measured here and should be read from the 63-tile run's logs when the plan is written.
- **Fair comparison:** hold out the same pixels or tiles for both B and C, retrain B on the same split, and compare on the
  held-out part. Otherwise B keeps its in-sample advantage. A leave-tiles-out split is the stricter test.
- Total: roughly half a day, mostly unattended. A written plan comes first; nothing is launched without approval.

Alternatives:
- **Adopt C without retraining:** not advised. The h12v09 shifts are large enough that the B-trained models are
  mismatched to C inputs.
- **Run variant A on h16v01 first** (about 40 minutes) and check bit 15 against the snow bit, to decide A vs C for
  Greenland/Iceland before the 63-tile rebuild. This also connects to the pending Greenland/Iceland fix
  (`Greenland_Iceland_Scoped_Fix_Plan.md`).
