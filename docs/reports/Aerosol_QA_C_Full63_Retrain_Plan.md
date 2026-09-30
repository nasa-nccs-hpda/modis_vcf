# Plan: rebuild bit-15 (variant C) metrics for all 63 tiles, retrain, and compare B vs C on held-out data

**Written 2026-09-29 evening. Status: FINISHED 2026-09-30 10:41 (Stages 0-5b, launched 2026-09-29 21:17, no failures). Results: `Aerosol_QA_C_Full63_Retrain_Results.md`. Stage 6 was replaced by the stable-pixel and lidar references. See "Launch and resume info" at the end.**

**Caveat (2026-09-29): Stage 6, the independent test on non-training tiles, is deferred.** The project lead will
settle it on 2026-09-30 (which tiles, which reference). It is not part of this launch. Stages 0-5b can run without it.
Until Stage 6 is done, the result rests only on the held-out pixels inside the training tiles, which is somewhat
optimistic in absolute terms and cannot show how B and C models do on new areas.

Follows the recommendation in `Aerosol_QA_Variant_Test_Results.md` (step 3 of `Aerosol_QA_Variant_Test_Plan.md`).

## Question

With models trained the same way on each, does the bit-15 QA screen (C) give tree cover at least as accurate as the
current bits 7-8 screen (B) on data the models have not seen? B is known to be broken (it rejects good pixels because
PGE72 stores the BRDF flag and day parity in those bits), so **a tie favours C**. Only a clear loss for C would argue
against adopting it, and would need investigating before anything changes.

## Nodes

**forest201, forest202, forest203 and forest205 only.** forest204 is excluded (checked 20:38: another user's R jobs,
18 GB free). Each node has 12 cores and about 104 GB free. forest201 is shared: another user's Python job has been using
about 8 of its 12 cores for 6 hours. So:
- makeMetrics and training-row jobs take tiles from **one shared claim queue**, so forest201 simply takes fewer tiles
  if it is slow. It never holds a fixed share of the work.
- Model fits (12 threads each) run on forest202, 203 and 205 only. forest201 takes fits only if its other job has
  finished by then.
- Node load and free memory are checked immediately before each launch. If a node is busy at launch time, the plan
  goes ahead on the remaining ones and I report which nodes were used.

## What is reused and what is new

| Item | Source |
|---|---|
| B metrics, B training rows (63 pcttree tiles), predictor list (64) | `FullRuns/full63_c_vs_py_20260919-0806/` (read only) |
| C code (`createQaMask` aerosol term = `(state & 32768) != 0`) | `FullRuns/aerosol_qa_variants_20260929/code_C/` (sha256 checked against its manifest before launch) |
| C metrics for h09v05, h12v09, h16v01 | `aerosol_qa_variants_20260929/Python/metrics_C/` (linked, not rebuilt) |
| Fit and evaluation scripts | `regtree_alt_models_20260927/code/validation/fitAltModel.py`, `eval_heldout.py`; baseline fit and `buildTraining.py` as in `python_phases_node.sh` |
| **New:** C metrics for the other 60 tiles, C training rows, keyed 20% pixel split, 6 fits, evaluation, 63-tile C predictions and historical comparison (independent-tile test deferred, Stage 6) | new run folder `FullRuns/qaC_full63_retrain_<launch date>/` |

Nothing in earlier run folders is written to. rlgill's repositories and the workspace `Python_code/` are not touched.
Model settings are the same as before: baseline (30 estimators, `rfFitFromPge61`), RF (`n_estimators=30, max_features=sqrt`),
XGBoost tunedB (`2000 rounds, depth 6, lr 0.02, subsample 0.8, colsample 0.8`), all `random_state=0`, and the **same
64-predictor list** for B and C. The predictor list was chosen on B metrics. Keeping it fixed gives B a small built-in
advantage, which is acceptable because it biases the test against C, not for it.

## Stages

### Stage 0: prep (about 10 minutes, one node)

Create the run folder, record the code sha256 and the diff against the 63-tile run's code in `manifests/`, write the
60-tile queue (63 minus the three already built), link the three finished tiles, and do a dry run of the queue scripts
(`DRY=1`, as in the 63-tile run). The three Stage 1 tiles from the variant test serve as the canary for the code change.

### Stage 1: C metrics for 60 tiles (about 3-3.5 hours)

`makeMetrics.py` with the same container, bind layout, metric list (`metricUnsortedMonthlyBands` + 19 C-mirror families)
and year (2019) as `run_stage12_C.sh`. Up to 3 tiles at a time per node, each started only with 28 GB free (existing
guard and watchdog). The 09-19 run took 2.6 hours for 63 tiles on 5 nodes. With 4 nodes, one of them shared, expect
about 3-3.5 hours. Disk: about 25 GB per tile, about 1.5 TB in total (panfs has 2.0 PB free).

### Stage 2: C training rows (about 40 minutes)

`buildTraining.py` for pcttree on all 63 tiles, same arguments as `python_phases_node.sh`, from the same queue.
pctbare (non-vegetated) is out of scope. The C metrics will exist if it is wanted later.

### Stage 3: keyed 20% pixel split (about 20 minutes, one node)

**No whole tiles are held out.** Each training tile carries unique geographic information, so all 63 tiles stay in
training (project lead's decision, 2026-09-29).

The 63-tile run's split (`train_test_split(test_size=0.2, random_state=0)` on row order) cannot be reused. If C has
even one row more or fewer than B, the same seed picks different pixels. A new script keys each training row by
`(tid-year, x, y)`, keeps only pixels present in **both** B and C, and holds out 20% of pixels, chosen by a hash of the
key (deterministic, independent of row order). The same pixels are held out for B and for C. The result is comparable
to the earlier held-out table (baseline MAE 7.46). It is somewhat optimistic in absolute terms, because held-out pixels
have training neighbours in the same tile, but this affects B and C equally.

Checks reported here: rows per variant, and the share of B pixels missing from C and the reverse. A mismatch over 5%
pauses the run for review.

### Stage 4: fits (about 1.5 hours)

2 variants (B, C) x 3 models = **6 fits**. **B is retrained too**, on the same training pixels, so neither variant keeps
the in-sample advantage. Times from earlier logs: baseline about 1 hour, XGBoost tunedB about 30 minutes, RF about
10 minutes. That is about 3.3 node-hours, one fit per node at a time, on 3 nodes: about 1.5 hours. The two baseline fits
go first, on separate nodes.

### Stage 5: evaluate on the held-out pixels (about 20 minutes)

`eval_heldout.py` on the held-out pixels: R², MAE, RMSE, correlation and bias for each (variant, model). Also:
- per-tile MAE for B and C on each tile's held-out pixels, and the number of tiles where C beats B;
- MAE by label bin (0-10, 10-30, 30-60, 60-100);
- a summary table, `Reports/Aerosol_QA_C_Full63_Retrain_Tables.md`, written automatically when the run ends.

### Stage 5b: per-tile comparison with the historical MOD44B (about 4 hours; added 2026-09-29 at the project lead's request)

- **Predict all 63 tiles with the three retrained C models** (baseline, RF, XGBoost tunedB), using the same
  `vcfPredictQueue.py` queue as the earlier runs, on forest201/202/203/205. Earlier timings: RF about 2 hours and
  XGBoost tunedB about 2.5 hours for 63 tiles on two nodes each, the baseline about 1.5 hours on five nodes. That is
  about 16 node-hours, so about 4 hours on four nodes. Output: `Python/predictions_C/<model>/`.
- **B side is reused, not re-predicted:** the existing 63-tile predictions used in the model deep dive
  (`full63_c_vs_py_20260919-0806/Python/predictions` for the baseline, `regtree_alt_models_20260927/Python/predictions/rf`
  and `.../xgboost_tunedB`). Those models were also trained on an 80% split of B rows, so they are a fair counterpart to the
  retrained C models. The difference is only which 20% was held out.
- **Comparison:** a copy of `regtree_alt_models_20260927/code/validation/deepdive_vs_historical.py` in the new run folder,
  extended with the three C-model predictions. In that script "C" already means the C-code product, so the bit-15
  products are named **Q15_E, Q15_RF and Q15_TUNEDB** in its output. Per tile, B and bit-15 models are scored against the
  historical MOD44B (C6 `.006`, the same reference as the deep dive) on one common valid-pixel mask: MAE, bias, RMSE,
  correlation, and the Q15 - B tree cover difference. The run gives a 63-tile summary table like
  `Compare/deepdive_all63_summary/`, with pooled MAE for each product and the number of tiles where Q15 beats B per model.
  Output: `Compare/deepdive_q15_vs_b/`. About 20-30 minutes.
- **How much it can show:** it is useful for spotting spatial problems (a tile or region where bit 15 moves tree cover a
  lot), but it is weak evidence on which screen is better. The historical product had no aerosol screen, and bit 15 keeps
  nearly the same pixels as no screen outside snow tiles, so bit 15 is expected to look closer to it. The models were also
  trained on these tiles' labels, and the historical product came from the same operational C code.

### Stage 6: independent evaluation on tiles outside the training set (DEFERRED: to be settled 2026-09-30, not in this launch)

The retrained B and C models predict tree cover on tiles that were never in training, and each is compared with an
independent reference. **Constraint found while planning:** tree cover labels
(`Python_code/modis_vcf/data/legacy/<tile>.bin`) exist only for the 63 training tiles. The 15 other labelled tiles
(h08v06, h09v06, h10v02, h13v02, h17v07, h18v08, h20v07, h21v07, h21v08, h21v09, h26v03, h26v04, h27v03, h30v11,
h31v09) have **bare (non-vegetated) labels only**. So the tiles and the reference are the project lead's choice:
- **Tiles:** for example the 15 above, which cover West Africa, the Congo, Siberia, Australia and Southeast Asia, plus
  the south-western US. That mix includes both high-aerosol and low-aerosol conditions.
- **Reference options:** (a) the historical MOD44B, used for B and C alike, noting that it had no aerosol screen, so it
  leans toward whichever variant keeps more pixels; (b) an external tree cover product, if one is available here;
  (c) the bare labels on the 15 tiles, used indirectly: tree cover should be near 0 where bare is near 100.
- **Cost per tile:** C metrics about 35-40 minutes, and B metrics too if the tile is not among the 63, since B metrics
  exist only for those. Prediction is a few minutes per tile per model. 15 tiles x 2 variants is about 1.5-2 hours on the
  four nodes.

## Estimate

Stages 0-5b: about 10 hours wall clock, almost all unattended. Stage 1 about 3.5 hours, Stages 2-3 about 1 hour,
Stage 4 about 1.5 hours, Stage 5 about 20 minutes, Stage 5b about 4 hours (it starts as soon as the three C fits are done). Stage 6 (deferred) would add about 2 hours when it is planned. Its metrics could run on the same queue
alongside Stage 1 once the tiles are chosen. Everything runs as one detached chain that pauses only for a failure or the
Stage 3 mismatch check.

## How the result will be read

- **Interim reading (Stages 0-5b only):** the held-out-pixel result is reported as provisional. No production decision is
  made on it until Stage 6 is done.
- **C matches or beats B on the held-out pixels for most models, and on the independent tiles:** adopt C as the
  aerosol screen for production. Next: the 63-tile prediction and the historical comparison with C models.
- **Stage 5b (historical) is read for spatial problems only**, not as a vote between B and C, for the reasons given in
  that stage.
- **The two tests disagree:** the independent tiles weigh more, because they are the closer match to predicting new
  areas.
- **C clearly worse** (worse on most tiles for all three models): do not adopt. Look at where it loses (snow tiles,
  label bins) and consider variant A (no aerosol test) before choosing.
- h16v01 and other snow tiles are read separately, because bit 15 also appears to drop snow pixels (not verified).

## Standing conditions

- rlgill owns pge61, pge72 and `modis_vcf`: no commits or pushes; code changes only in copies under the run folder.
- Every stage runs fully detached (`setsid nohup`, parent PID 1, `timeout 30 ssh -n` for each remote call), with
  `status/*.status` / `.done` / `.failed` markers and logs under the run folder, and resume info recorded here.
- Only forest201, 202, 203 and 205 are used. Nothing is launched until the project lead approves this plan.
- A results report, `Aerosol_QA_C_Full63_Retrain_Results.md`, is written when the run finishes.

## Launch and resume info (2026-09-29 21:17)

Approved by the project lead and launched 21:17-21:18, fully detached (`setsid nohup`, parent PID 1, no terminal) on
forest203, 202, 205 and 201. Node chain PIDs: forest203 2092843, forest202 3084352, forest205 1718359, forest201 3284844.
Expected finish: about 07:00-08:00 on 2026-09-30.

- Run folder: `FullRuns/qaC_full63_retrain_20260929/`. One script per node, `scripts/node_chain.sh`, shares work through
  `status/claims/` (mkdir) and markers `status/<step>.done|.failed`. A restarted copy continues where things stand.
- Progress: `status/node_<host>.status`. Finished: `status/run.done`, with tables in
  `Reports/Aerosol_QA_C_Full63_Retrain_Tables.md` (written automatically). Any `status/*.failed` (or
  `status/predict_markers_*/*.failed`) stops every node at its next barrier. `status/split.failed` containing "PAUSED"
  means the B/C pixel mismatch exceeded 5% (Stage 3 check), which is a review point, not a crash.
- Slots: makeMetrics 3 per node (2 on forest201, which is shared), each started only with 28 GB free, one retry per tile;
  buildTraining 4 per node (3 on forest201); fits one per node with 40 GB free, and forest201 joins only if its 1-minute
  load is under 3 when fitting starts.
- Setup done before launch: 60-tile queue in `manifests/tiles_metrics_queue.txt`; h09v05, h12v09 and h16v01 metrics linked
  from the variant test (`status/makeMetrics_C_<tile>.done` = "reused"); B labels linked from the 63-tile run; code_C
  sha256 re-checked (`manifests/code_C_sha256_at_launch.txt`, matches the variant test's manifest).
- Scripts, tested before launch on forest203 with B data standing in for C on h09v05 and h12v09 (all rc=0; split held out
  20.0%; B = C gave zero differences; the historical comparison reproduced the deep dive's h09v05 MAEs: baseline 2.86,
  RF 2.36, XGBoost tunedB 1.82, C 1.41): `split_keyed.py` (Stage 3), `eval_q15.py` (Stage 5), `q15_vs_b_hist.py`
  (Stage 5b, a small script that imports the deep dive's readers and statistics rather than a full copy of it; bit-15
  products are named Q15_*), `summarize_q15.py` (tables).
- Stage 5b predictions use the existing B pctbare model for the non-vegetated layer (required by `vcfPredictQueue.py`);
  only the tree cover layer is compared.
- **2026-09-30 07:06, restart note:** Stage 5b stalled from 01:40 to 07:06 because `Python/predictions_C/<model>/` was never
  created (`VcfPredict` raised "A valid output directory must be provided"), so the node chains looped on rc=1. I created
  the three folders by hand, and the running chains continued with no relaunch. Stages 0-5 finished normally (`eval.done` 01:40).
