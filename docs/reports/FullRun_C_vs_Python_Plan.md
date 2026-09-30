# Plan: full-tile C and Python runs on forest203 + forest204, then compare the predictions

Written 2026-09-19.

> **STATUS (audited 2026-09-20): this plan was carried out and the run has finished.** What actually happened is in
> `FullRun_Launch_and_Status.md` (record) and `FullRun_Results_C_vs_Python_63tiles.md` (results); sections 0c and 0d below summarize it.
> The rest of this file is kept as the planning history and still contains the pre-launch wording and estimates. Where an
> estimate or a statement was wrong or later resolved it is marked "(update ...)" or listed in section 0c. Actual timings: Python metrics
> 2.6 h (12:43-15:19); C B1 66 min, B2 41 min, B3 5.0 h (14:29 to 19:28); training data 31 min; split 2 min; tree fit 81 min and
> bare fit 17 min (12 jobs each); predict 90 min on three nodes; comparison 15 min; total about 7 h against the 10 h (5-node) estimate.
> Measured model sizes: tree 13.5 GB, bare 11.9 GB.

Facts below were checked on 2026-09-19 unless marked "estimate" or "not checked". Companions: `Signoff_Package_2tile_C_vs_Python.md`
(why a full run is needed), `Parallel_CPU_Use_Assessment.md`, `Ensemble_Prediction_C_vs_Python.md`.

## 0. Decisions

**Confirmed by the user 2026-09-19:** (1) use **63 tiles**; (2) Python trains on the **`v5.0.3samp` labels**, to
match C; (3) **apply the parallelization** (in a copy of the code). Also: do not run multiple tiles concurrently
through the parallelized Python fit and predict stages (see "Parallelism in the Python code" below). Items 4-6 were later settled:
(4) Python outputs were left raw and the comparison reports raw and harmonized views; (5) h09v05 and h12v02 were recomputed with the rest;
(6) the quota was not checked (the filesystem showed 5.2 PB free) and the run used about 3 TB or less. The original wording of each decision follows.

**Node pool (user, 2026-09-19): use forest201-forest205 (five nodes)** instead of only forest203 and forest204. Sections
2, 4 and 4a below are revised for it; the two-node schedule and estimate are kept in 4a as the baseline.

1. **Tile count: 62 or 63?** Every source lists 63 tiles: `tile_audit.json` (all 63 pass), the 63 `v5.0.3samp`
   label files, the 63 `data/legacy/<tile>.bin`. The number 62 matches only rlgill's
   `v5.0.3samp-conventionallyNamed/` directory, which lacks **h24v04**. This plan uses a tile manifest
   (`manifests/tiles.txt`), default 63 from `tile_audit.json`. Dropping h24v04 gives 62; say which you want.
2. **Which tree-cover labels does Python train on?** Recommended: `v5.0.3samp` (what C uses, and the file you
   vouched for), so labels are not a confound. `legacy` (Python's current default) can be a second, separate
   Python variant later (its own clean directory). Doing both doubles Python fit and predict time.
3. **Apply the Python speed-ups** (`--nJobs`, `random_state`, threaded and chunked predict) **in a copy of the
   code**. Without them the tree fit on all tiles is estimated at about 15 h on one core (see section 6).
4. **Python post-processing parity** (round half up, water=200 on non-tree veg, tree + bare <= 100): recommended
   to leave the Python outputs raw and apply these only in the comparison step, so both views are reported.
5. **Recompute the two tiles already done (h09v05, h12v02)** for uniformity (recommended, about 3% of the work)
   rather than link the old outputs.
6. **Free-space / quota:** worst case about 3.5 TB (section 5). `/panfs` shows 2.0 PB free; the user/group quota
   was not checked.

## 0a. Blocking issue found 2026-09-19: C and Python do not use the same predictor subset (resolved: option C, see 0c)

Full evidence in `Predictor_Subset_C_vs_Python.md`. C selects 64 (tree) and 21 (bare) of its 324 metrics by
index; Python's `rfFitFromPge61.py` hard-codes 60 and 18 names that follow a different (date-major) numbering,
so only 8 of the 24 tree and 5 of the 15 bare unsorted predictors coincide with C's, and the other families
have no same-definition Python metric. **A decision is needed before the full run** (options A/B/C in that
report); options B and C change Phase 0 and Phase 3 (predictor lists and, for C, new metric code plus validation
against `metrics.bin`). Monte Carlo is confirmed not part of the Python pipeline here (Option B direct fit only).

### Update 2026-09-19 (later): option C chosen and validated on two tiles

- **Decision:** the user chose option C. The run root now exists:
  `FullRuns/full63_c_vs_py_20260919-0806/` (code copy in `code/python_code/`, manifests, status).
- **Two independent implementations** of the 252 derived C metrics were made by two sessions working in that root; both
  reproduce C exactly from C's own inputs. Only implementation B (`CMirrorMetrics.py`, wired into the copy's
  `Metrics.py`) has been run through the Metrics class. One must be chosen before the full run. See
  `OptionC_ImplementationB_Validation_Results.md` and the update in `Predictor_Subset_C_vs_Python.md`.
- **Results on h09v05 and h12v02 (all pixels):** formulas 100% exact; delivered through Metrics, 50 of 77 attributes
  exact on h09v05 and 37 of 77 on h12v02. The rest depend on NDVI (C averages per-composite NDVI, Python takes the NDVI
  of averaged bands) and, on h12v02, on the `solz <= 0` no-observation mechanism.
- **Effects on this plan:** (1) Phase 2 `makeMetrics` gains 19 mirror families per tile (about 3.5 min and 2.4 GB per
  tile, about 150 GB for 63 tiles, added to the section 5 estimate). (2) Phase 3 predictor lists come from
  `CAttributes` (64 tree, 21 bare) instead of the hard-coded lists. (3) **New decision before launch:** either accept
  the NDVI mismatch (27 of 77 attributes only 26-55% exact) or build a C-style NDVI in Python (per 16-day composite,
  truncated, then averaged; needs the `1-Days` files) and revalidate.
- **Still not done:** predictor-list generation, NDVI parity, the parallelization edits, and every full-run stage.
- **NDVI difference size (measured on all pixels):** where Python's NDVI is not exactly C's, the absolute difference (NDVI x 1000) is
  min 1 / mean 4.2 / max 1,513 on h09v05 (68.0% of pairs not exact) and min 1 / mean 41.0 / max 1,909 on h12v02 (83.2% not
  exact); relative to C the mean is 2.3% and 47.7%. See `OptionC_ImplementationB_Validation_Results.md`.
- **Decision (user, 2026-09-19): the current NDVI difference is acceptable for now.** No C-style NDVI is built; the Python
  NDVI stays as is, and results must be read with that input difference in mind (27 of 77 predictors are NDVI-related).

## 0b. Update 2026-09-19 (later): option C chosen, Phase 0 gate (completed, see 0c)

- **Run root already exists** (Phase 0 step 1 done): `FullRuns/full63_c_vs_py_20260919-0806/` with `manifests/`,
  `code/python_code/` (copy of `Python_code` without `modis_vcf/data`), `code/validation/`, `status/`,
  `Python/validation_optionC/`. Its `manifests/start.marker` (08:07) is the reference for the final protected-path
  check; work since then only reads protected trees and writes inside the run root.
- **Decision recorded:** option C (mirror all of C's 64 tree / 21 bare attributes in Python), 63 tiles, samp labels,
  parallelization on. Details and results: `Predictor_Subset_C_vs_Python.md` (update section).
- **Validation result (two full tiles, run detached on both nodes):** the C group-statistic formulas are reproduced
  **100% exactly** (252 derived columns, all 22.9M and 20.5M pixels). End-to-end parity is limited by the unsorted
  inputs, mainly **NDVI** (27-34% exact on h09v05, 1-19% on h12v02), which affects 16 tree and 7 bare predictors
  and every greenest-sorted statistic. New gate for Phase 0: decide whether to add a C-style NDVI (per-composite
  NDVI averaged as in `fix_bands.c`/`make_metrics.c`) to Python before the full run; without it those predictors
  differ from C's on most pixels, and the full comparison stays partly confounded by inputs.
- **Reconcile first:** a second writer has been adding a parallel implementation (`CMirrorMetrics`, `CAttributes`)
  to the same run root; only one implementation should be wired into `Metrics.py`. (Resolved 2026-09-19: the user chose implementation B,
  `CMirrorMetrics`, which is the one wired in and used; the other writer's `CEquivalentMetrics` was kept as an independent cross-check only.)
- **Plan changes this implies:**
  - **Phase 2b (new): C-equivalent metrics per tile**, computed from each tile's `UnsortedMonthlyBands.tif`
    (no need to redo `makeMetrics`): about 5-6 min per tile in the validation runs, memory about 3.3 GB for the
    unsorted values plus blocks; 28 blocks x 9 bands x 46 MB is about 11.6 GB of tifs per tile if all are written
    (about 730 GB for 63 tiles), so write only the blocks the predictor lists need.
  - **`buildTraining.py` runs every available metric** unless given `-m <names>`; use `-m` so the parquet holds only
    the predictor blocks rather than all 263 + 252 columns.
  - **Predictor lists** are generated from C's `attributes[]` / `attributes_bare[]` (parsed from `MOD44B3.c`), not
    typed by hand, and `rfFitFromPge61.py` (copy) takes them via a new `--predictorFile` option.
  - **Run-time and disk additions** are small next to Phase 2 (about 5 min/tile) except the tif size above.
  - `buildTraining.py` already has a `--trainingDir` option, so the samp labels need no symlink changes. **(Wrong: `BuildTraining` accepted
    the option but ignored it; fixed in the run-root code copy, see 0c.)**

## 0c. Status 2026-09-19 12:45: Phase 0 done, full run launched

The pilots, the extended C canary and the code preparation are finished and the run was launched at 12:42:44 on all five
nodes, fully detached. Full record, results and how to check or repair: `FullRun_Launch_and_Status.md`. Changes from this plan:
- Python metrics compute only the unsorted values and the 19 C-mirror families (`-m`), 4 tiles per node (pilot peak 19.8 GB
  per process). Estimated total now about 10 hours (range 9-13) instead of 15.
- C executables rebuilt from the current source in the run root (option A); B1 and B3 now share the aerosol-bit change.
- **Correction:** `BuildTraining` ignored its `trainingDir` argument; fixed in the run-root copy (an earlier line here said
  the samp labels needed no symlink changes). Bare labels exist for 20 of the 63 tiles, not 35.
- Intermediates are kept (no deletion of `1-Days` or `2-Composites`), as instructed.

## 0d. Status 2026-09-19 19:56: run finished

C B1/B2 done; B3 finished for all 63 tiles (three were marked failed by an over-strict 10 MB rule, outputs valid; rule and markers fixed 2026-09-20); Python phases done;
comparison done for all 63 tiles. Results: `FullRun_Results_C_vs_Python_63tiles.md`. Total compute about 7 hours, faster than the 10 hours
estimated (C B3 was the long pole, 14:29 to 19:28).

## 1. Non-overwrite rules (apply to every step)

**One new root, created with plain `mkdir` (no `-p`) so it fails if it exists:**
`/explore/nobackup/people/mcarrol2/MOD44B_VCF_Development/FullRuns/<RUN_ID>/`, with
`RUN_ID=full63_c_vs_py_<YYYYMMDD-HHMM>` (timestamp makes it unique). Nothing below is written outside it.

```
<RUN_ID>/
  manifests/   tiles.txt, config.env, code hashes, input listing, protected-path marker file
  code/        python_code/ (copy of Python_code with the changes in section 6), c_scripts/ (copies of the
               Track A scripts with WORKSPACE repointed), c_templates/ (copy of test_pcf templates)
  C/           workspace/{intermediate/<tile>/, b2_models/, b3_output/, pcf/, logs/, tmp/, run_cwd/<job>/}
  Python/      metrics/<tile>/2019/{1-Days,2-Composites,3-Metrics}, training/, models/, predictions/<tile>/, logs/
  Compare/     per-tile CSV, summary, plots, logs
  status/      one status file per node and stage (never shared between nodes)
```

**Protected, read-only for this run (must show zero new or modified files at the end):**
- `MOD44B_VCF_Development/VCF_outputs/` (3.7 TB, 2,836 files; shared staging, already overwritten once on Sep 17).
- All of `AI_code_sync_09152026/` (existing `C_code`, `Python_code`, all `Metrics/*`, existing `Reports/*`).
- The real repo `/panfs/.../pge61_08312026/pge61/` (owned by rlgill), including the Sep 11 full-run
  `pipeline_logs/` and `pipeline_status.json`.
- `Mark_training/...` labels, `/css/modis`, `/explore/nobackup/projects/ilab/data/MODIS/MOD44C`.
Allowed read-only inputs: `C_code/pge61/bin/*.exe`, `BARE/` models, MCF templates, `static_coeff`, the
`test_pcf` templates, `~/.local` (JDK, weka, HDF, PGS toolkit).

**Mechanical guards**
- All launch scripts start with `set -o noclobber`, and every stage refuses to start if its target directory
  or output file already exists (write-once). Resume works by skipping tiles whose "done" marker exists,
  never by rewriting.
- Container runs bind every shared input with `:ro` (code, data, `/css/modis`); only that stage's output
  directory is bound writable.
- **C hard-codes paths:** `WORKSPACE` in `generate_tile_pcf.py` (line 48), `generate_b2_pcf.py` (line 23) and
  `run_full_pipeline.py` points at the sync copy, and `generate_tile_pcf.py` rewrites `.../VCF_outputs` paths.
  Use copies in `code/c_scripts/` with `WORKSPACE` set to `<RUN_ID>/C/workspace` (one line each), and set
  `PGE_WS` for `env.sh` (its default is the real repo and it `cd`s there).
- **PCF lint step (before any C run):** grep every generated PCF and fail on any path not under
  `<RUN_ID>/C/workspace` (except the read-only inputs above) and on any string `VCF_outputs`.
- **C executables write scratch files into their current directory** (`GetAttr.temp`, `MODIS_37440.t` are
  already untracked strays in the repo). Run each job with its own cwd `C/workspace/run_cwd/<job>/`.
- **After the run:** `find` each protected root with `-newer manifests/start.marker` (metadata only) and
  require empty output; save the result in `manifests/`. Old `VCF_outputs` models are not touched
  because the B2 PCFs point their INTERMEDIATE OUTPUT at `C/workspace/b2_models` (the fix already proven in
  `TrackA_B3_Failure_Root_Cause_and_Fix.md`), verified by the lint step.
- Optionally `chmod -R a-w` each finished stage directory. Not done without your OK.
- One status file per node and stage; the Sep 11 driver wrote a single `pipeline_status.json`, which two
  nodes would race on.

## 2. Node readiness (checked 2026-09-19)

| | forest201 | forest202 | forest203 | forest204 | forest205 |
|---|---|---|---|---|---|
| CPU / RAM | 12 / 112 GB | 12 / 112 GB | 12 / 112 GB | 12 / 112 GB | 12 / 112 GB |
| Free memory when checked | 106 GB | 104 GB | 104 GB | 104 GB | 104 GB |
| Load average when checked | 0.04 | 0.08 | 0.07 | about 0.1 | 0.15 |
| `$HOME` with JDK, weka, HDF libs, PGS toolkit | shared | shared | shared | shared | shared |
| C binary `MOD_PR44B3.exe` libraries missing | 0 | 0 | 0 | 0 | 0 |
| Java | 1.8.0_504 | 1.8.0_504 | 1.8.0_504 | 1.8.0_504 | 1.8.0_504 |
| `/css/modis`, people dir, container, MOD44C | visible | visible | visible | visible | visible |
| `/tmp` free | 36 GB | 37 GB | 36 GB | 37 GB | 38 GB |
| Container Python (numpy 1.24.4, sklearn 1.2.0, gdal 3.8.4) runs | yes | yes | yes | not run on 204 | yes |

All five are the same OS (Rocky Linux 8.10) and CPU (Xeon E5-2697 v3). Other users (wcwagne1, rvieiral) had only idle
processes when checked. These are shared machines, so another user can start work at any time.

Not checked: running a C stage or `makeMetrics` on any node other than forest203/204, memory limits per process, and
whether `mkdir` claims are atomic across all five nodes on panfs (Phase 0 tests it).

## 2a. Parallelism in the Python code (verified 2026-09-19 by reading the code)

| Stage (Python entry point) | Parallelism in the code today | Consequence for the plan |
|---|---|---|
| Fit, `view/rfFitFromPge61.py` | **None.** `BaggingRegressor` has no `n_jobs` (single core); `multiprocessing` is imported (line 5) but never used | One pooled fit per type across all tiles (not per tile). Add `--nJobs` (copy); with 12 jobs it uses every core |
| Predict, `model/VcfPredict.py` | **None.** Saved model has `n_jobs=None`; feature loading is a plain loop | Add threaded `n_jobs` on load (copy); it then uses every core |
| Alternate fit, `model/MonteCarloSim.py` (Option A, not used here) | **Yes**: `ProcessPoolExecutor(max_workers=numCpus)` (line 502) and `n_jobs=-1` in the final fit (line 568) | Not in this pipeline. Its `n_jobs=-1` uses every core of the host, which is another reason not to run it beside other jobs |
| `makeMetrics`, `buildTraining`, `MasterTraining` | No pool or thread settings anywhere (`grep` for multiprocessing, futures, joblib, n_jobs, GDAL/OMP thread variables found none outside the files above); numpy sort/median and GDAL reads | Single process per tile; concurrency across tiles is the only way to use cores here, limited by memory (measured in Phase 0) |

Probe note: a makeMetrics probe on forest204 was stopped after about 60 s on your instruction, so it shows
only the early read phase: one process, 0.1-0.5 cores busy, 13 threads, 0.1-3.2 GB RAM. It says nothing yet
about the compute phase or peak memory; the Phase 0 pilot still has to measure those. Its scratch directory
was removed and no processes remain on forest204.

**Rule for the parallelized stages (fit and predict): one job per node at a time, using all 12 cores; never
run two tiles or two fits at once on the same node.** The pcttree and pctbare fits run one after the other.
Predict runs one tile at a time per node; with two nodes, two tiles at once is fine, one per node. A node may
run several `makeMetrics` or `buildTraining` processes at once only up to the memory limit from the pilot,
and never while a fit or predict is running on it.

## 3. Phases

### Phase 0: preparation and pilots (no full-scale work)
1. Create the run root and manifests; write `start.marker`; record code hashes of the copied code.
2. Copy code, apply the section 6 changes only in `code/python_code/`; C scripts get the `WORKSPACE` change.
3. Run the PCF lint on generated PCFs for two tiles (dry run, nothing executed).
4. Atomic-claim test: two processes, one on each node, `mkdir` the same claim directory; exactly one may win.
5. **Python `makeMetrics` pilot on one new tile plus the waived-input tile h12v01**, with a memory sampler
   (the script from `check_parallel_bagging.py` can be reused). About 46 min. This sets the concurrency
   per node (four parallel regenerations were OOM-killed on Sep 18, so it must be measured).
   The four tiles h12v01, h13v01, h16v01, h21v01 have waived December granules in the C audit; whether
   Python's `makeMetrics` tolerates them is unknown.
6. **C canary:** B1 for one tile into the new workspace (about 6 min), then the protected-path `find`.
7. Scale test of the modified fit and predict on about 1-2M rows for equality with the unmodified code.
Gate: all pass, otherwise stop and report.

### Phase 1: C chain (B1 -> B2 -> B3), reusing the proven `run_full_pipeline.py` logic
Timing reference is the Sep 11 full run (real repo, concurrency 4, all 63 tiles rc=0): B1 245-403 s per tile,
B2 590-719 s per model, B3 524-2,623 s per tile (median 2,150 s).
- **B1, 63 tiles:** 12 `fix.out` files (about 15 GB) and 30 `weka_training.<tile>.2019.NN.arff` per tile.
- **Barrier:** all 63 B1 tiles must be done before B2 (B2 concatenates every tile's arff; running early
  silently trains on a subset). The driver checks the count.
- **B2, 30 models:** pooled over all tiles, into `C/workspace/b2_models/`. Then check the model count and,
  for model 00, the `Total Number of Instances` line in its log (Sep 11-17 pooled runs showed 324,033).
- **B3, 63 tiles:** reads the 30 new models plus the read-only `BARE/` models; writes
  `C/workspace/b3_output/MOD44B.A2019065.<tile>.061.hdf`.
- Estimate on one node at concurrency 4: about 1.6 h + 1.4 h + 9.5 h, about 12.5 h (about 7 h if both nodes
  run C at concurrency 4 each). The Sep 11 numbers were for the real repo; B3 memory per process was not
  recorded, so concurrency is confirmed in Phase 0/1 by watching memory on the first waves.

### Phase 2: Python metrics (independent of C; overlaps Phase 1)
`makeMetrics.py -o Python/metrics/<tile> -y 2019 -t <tile>` per tile inside the container, all binds `:ro`
except the output directory. Reference: h09v05 + h12v02 concurrently took about 46 min and 32 GB each
(1-Days 9.7 GB, 2-Composites 6.8 GB, 3-Metrics 16 GB).
- Work queue: a claim per tile via `mkdir Python/status/claims/<tile>`, so all five nodes can pull from one list.
- Suggested split (five nodes): forest203-forest205 run Python metrics from the start at 2 tiles per node (the
  proven concurrency; raise only if the pilot allows), forest201 and forest202 run the C chain, then join the metrics
  queue when C is done. Estimate: metrics finish about 8 h in (26 node-hours of metrics plus 12.1 of C over five nodes).
  With two nodes only, the earlier estimate was about 19-20 h for this stage.
- Each tile also gets the 19 C-mirror families (about 3.4 min, 2.4 GB) after its metrics, from its own unsorted tif.
- Gate: each tile has its 24 metric tifs and `MOD44-NDVI.bin`; a check script lists any tile missing files.

### Phase 3: Python training data and models
1. `buildTraining.py` per tile and type. It already has a `--trainingDir` option, so the samp labels need no
   symlink edits in shared data: create `Python/training_labels/` with
   `VCF_training_pcttree_<tile>.7.0.0.out` symlinks to the samp files (63, or 62), and for `pctbare` symlinks to
   the existing `data/legacy/<tile>.bare.4.out` (20 of the 63 tiles, not 35; C does not retrain bare, so bare has no samp
   counterpart). Output to `Python/training/`.
2. `build_xy_from_training.py` (copy) for each type, 80/20 split, output to `Python/training/<type>/`.
3. `rfFitFromPge61.py --nEstimators 30 --nJobs 12` (copy), to `Python/models/`. Run tree first, then bare,
   strictly one after the other (12 jobs each), never both at once and never beside a predict on that node.
- Size and time estimates (not measured): samp `pcttree` about 17M rows (about 7.3x the 2.32M of Sep 18),
  so about 15 h single-core, about 1.5-2.5 h at 8x; model about 5 GB. `pctbare` on 35 tiles could be 10-20M
  rows and a model of 10-30 GB. Check the actual row counts and model sizes before Phase 4.
  (Update: measured in the run: bare labels exist for 20 tiles; models 13.5 GB (tree) and 11.9 GB (bare); tree fit 81 min, bare fit 17 min.)
- Held-out score per type from the 20% split, same as `heldout_eval_py_track_b.py`.

### Phase 4: Python prediction
`vcfPredict.py` per tile with the loaded models set to threaded `n_jobs`, chunked rows, outputs to
`Python/predictions/<tile>/`. The driver loads the models once per process and loops over its share of tiles
(a small change, section 6), because reloading a large model for each tile wastes minutes. Memory per process
is the two models plus features (about 3 GB per tile). Threaded `n_jobs=12` already uses every core, so run
**one tile at a time per node** (the two nodes each take half of the tiles). Estimate: about 13 min per tile single-core on Sep 18, so far less with threads; a few hours in total.

### Phase 5: comparison (after both chains are complete)
Generalize `compare_c_vs_python_2tile.py` to the tile manifest (copy, in `code/`), output to `Compare/`:
- Per tile and layer (tree cover, non-vegetated, non-tree vegetation): valid-in-both counts, corr, mean and
  bias (Py - C), MAE, RMSE, share within 5 and 10, water/fill mask disagreement; orientation check per tile.
- Global summary and a ranked table of the worst tiles; h09v05 and h12v02 rows compared with the two-tile result.
- Two views: raw Python output, and Python with C's rules applied in the comparison only (round half up,
  water 200 on non-tree veg, tree + bare <= 100, fill mask).
- Label check on the labeled pixels (samp labels): error of C and of Python against the labels, with the caveat
  that both are partly in-sample (C keeps a random 40-50% subsample; Python's 20% held-out rows are the clean subset).
- (Update: all 63 C HDFs were read, 7 layers each including Quality and the SD layers. The `-3` fill and the SD layers were not analyzed
  in the comparison.)

## 4. Suggested node assignment (five nodes, forest201-forest205)

| When (estimate) | forest201, forest202 | forest203, forest204, forest205 |
|---|---|---|
| Phase 0 (preparation) | C canary: B1 on one tile | makeMetrics memory pilot (one new tile plus h12v01); claim-atomicity test across all five nodes |
| Phases 1-2, hours 0 to about 6.5 | C chain B1, then the B2 barrier, then B3, 4 at once per node | Python metrics with the mirror families, 2 tiles at once per node |
| Phases 1-2, about 6.5 to 8 h | join the Python metrics queue when C is done | continue; the queue empties at about 8 h |
| Phase 3, about 8 to 11 h | training-data extraction on all five nodes (4 at once per node); then the bare fit on one node | the tree fit on one node with 12 jobs (2-3 h); other nodes idle |
| Phase 4, about 11 to 14 h | predict on all five nodes, one tile at a time per node, tiles from a shared claim queue | same |
| Phase 5, about 14 to 15 h | comparison on one node | idle |

Rules that apply on every node:
- Fit and predict use all 12 cores of a node, so nothing else may run on that node while they do.
- Before a stage starts on a node, a guard script checks load average (below 2), available memory for that stage, and that
  no fit or predict is running there; if not, the node is skipped and the queue continues on the others.
- One status file per node and stage (never shared). Claims are `mkdir` directories, used only after the Phase 0 test.
- Stagger model loading in Phase 4 (the bare model may be tens of GB; five nodes loading it at once loads panfs).

## 4a. Revised time estimate for the full 63-tile run (2026-09-19; pre-launch estimate, actual about 7 h, see 0d)

Compute wall-clock only, after code preparation; extrapolated from measured runs unless marked otherwise.

| Stage | Basis | Estimate |
|---|---|---|
| C: B1, B2, B3 (both nodes share) | Sep 11 full run measured: B1 1.5 h, B2 1.3 h, B3 8.8 h at concurrency 4 on one node, 12.1 h total | 6.5-12 h (12 h on one node) |
| Python metrics + 19 mirror families | makeMetrics about 46 min per tile at 2 tiles at once (measured); mirror families 3.4 min per tile (measured); memory limit above 2 at once per node unknown | 26 node-hours |
| Metrics and C sharing the two nodes | (12.1 + 26) node-hours over 2 nodes, perfect packing | about 19-20 h |
| Training data (83 buildTraining runs: tree 63 tiles + bare 20, not 35; measured 31 min) and consolidation | about 6 min per run measured on Sep 18, 4-8 at once | 2-3 h |
| Fit, tree then bare, 12 jobs | single-core 1.8 h for 2.3M rows measured; about 18M rows extrapolated; 7.6-8.3x speedup measured | 3-4 h |
| Predict, 63 tiles, 1 tile at a time per node | about 13 min per tile measured on Sep 18; feature loading dominates unless the unsorted tif is cached | 3-5 h |
| Compare | 34 s for 2 tiles measured | about 1 h |
| **Total** | | **about 30 h (range 26-40 h), 1.2-1.7 days** |

With the NDVI difference accepted (no NDVI rebuild), the compute total stays about 30 h, about 1.25 days (range 1.1-1.7 days).
By node-hours, C is about 12 (Sep 11 measured) and Python about 40-50 (extrapolated), so Python takes roughly 3-4 times longer;
the largest single pieces are C B3 (8.8 h of 12.1 h) and Python makeMetrics (26 node-hours).

Largest uncertainties: makeMetrics memory per process (pilot needed), the bare model size (could reach tens of GB),
repeated reads of the 3.3 GB unsorted tif in predict feature loading (fix by caching), and any NDVI rebuild (not costed).

### Five-node estimate (added 2026-09-19)

Same per-stage node-hours as the table above, spread over five nodes. NDVI difference accepted, no NDVI rebuild.

| Stage | Nodes | Estimate |
|---|---|---|
| C chain plus Python metrics with mirror families (12.1 + 26 node-hours; lower bound 38.1 / 5 = 7.6 h) | 2 on C, 3 on metrics, then all 5 | about 8 h |
| Training data (98 runs at about 6 min, 4 at once per node) and consolidation | 5 | about 1 h |
| Fit, tree (bare fit runs alongside on a second node) | 1 to 2 | 2-3 h (does not shrink with more nodes) |
| Predict, 63 tiles at about 12 min | 5, one tile per node | 2.5-3 h |
| Compare | 1 | about 1 h |
| **Total** | | **about 15 h (range 13-20 h), 0.6-0.85 days** |

Compared with two nodes (about 30 h): about half. It does not shrink further because parts are serial: the C chain
B1 -> B2 -> B3 (about 6 h even on two nodes), the fit (2-3 h), and the comparison (1 h). The upper end of the range covers
shared-storage contention (five nodes reading `/css/modis` and panfs and loading large models at once, not modeled), a large
bare model, and a lower concurrency limit for metrics than assumed. C's concurrency of 4 per node was only shown on
forest203, and Python metrics' 2 per node only on forest203.

## 5. Disk estimate (worst case, not measured for all tiles)

| Item | Estimate |
|---|---|
| C intermediates (about 15 GB per tile x 63) | about 950 GB |
| C arff, models, B3 HDF | about 15 GB |
| Python metrics (about 32 GB per tile x 63) | about 2.0 TB (about 1.0 TB if 1-Days and 2-Composites are deleted afterwards, only with your OK) |
| Python training parquet (samp) | about 20-70 GB |
| Python models | about 5-30 GB |
| Python predictions (about 47 MB per tile) | about 3 GB |
| Total | about 3.2-3.5 TB |

## 6. Code changes needed (in copies only; made in the run-root copy `$RUN/code/`, see 0c)

- `rfFitFromPge61.py`: `--nJobs`, `--randomState` (see `Parallel_CPU_Use_Assessment.md`).
- `VcfPredict.py`: `set_params(n_jobs=N)` after load, threaded backend, row-chunked predict, and a
  multi-tile loop that loads the models once.
- C scripts: `WORKSPACE`, status file location per node, cwd per job, PCF lint.
- Launch drivers: per-node, write-once, `:ro` binds, claim directories.
- No change to model type, features, thresholds or any C source.

## 7. Verification at the end

1. Protected-path `find -newer start.marker` is empty (saved to `manifests/`).
2. 63 (or 62) B3 HDFs and 3 x 63 Python tifs exist; no tile missing in any stage.
3. 30 C models with a two-digit sequence and the instance count matching the pooled B1 arff files.
4. Python held-out scores and label-count checks recorded next to the models.
5. Comparison summary written into `Reports/` as a new document, plus links from `README_START_HERE.md`.

## 8. Risks and unknowns

- Python `makeMetrics` memory (unknown until the pilot) sets the whole Python schedule.
- Model sizes for a full-tile bare model could be large (tens of GB); the prediction design depends on them.
- The Python h12v02 behavior (predictions up to about 44 on unlabeled pixels) is unexplained; more tiles may
  show the same pattern. The full run is the test, but it does not itself explain it.
- The four waived-input tiles (h12v01, h13v01, h16v01, h21v01) may need handling on the Python side.
- Both chains and the metrics read from `/css/modis`, panfs and each other's node at once; I/O contention
  may slow stages and is not modeled.
- Cross-node `mkdir` claims must be shown atomic in Phase 0 before any queue is used.
- Five shared nodes: another user may start heavy work on a node; the guard script skips a busy node but cannot stop work
  that starts mid-stage, so per-node memory should be watched during the first waves.
- C ran at concurrency 4 and Python metrics at 2 per node only on forest203; other nodes are assumed identical, and are
  identical on paper (OS, CPU, RAM, libraries) but were not run.

**Status of these risks (2026-09-20):** metrics memory: measured peak 19.8 GB per process, 4 tiles per node ran without a watchdog kill;
model sizes: 13.5 GB and 11.9 GB, prediction worked with the models loaded once per node; the h12v02 behavior: not present in the 63-tile
comparison (tree cover corr 0.917); waived-input tiles: handled by Python without changes; cross-node claims: tested atomic (400 of 400)
and used all run; C at 4 jobs per node ran on forest201 and forest202 without problems; nodes were not disturbed by other users.
Not resolved: the valid-pixel mask differences on some high-latitude tiles. The three B3 tiles marked failed by the size rule (0d) were fixed on 2026-09-20.
