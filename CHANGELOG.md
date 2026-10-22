# Changelog: MODIS-VCF Python code (modis_vcf)

Changes to the Python MOD44B (PGE61) code since the start of the 2026 development work, newest first. Each entry says what changed,
why, and **whether product values change**. "Exact" means proven identical to the previous version (dailies and composites byte for
byte, metrics and map layers by value) on the named tiles. Commits are on branch `mcarrol2/dev` of
`nasa-nccs-hpda/modis_vcf` (named `mcarrol2/v1.2.0` until 2026-10-19), which starts from `develop` at `b83e302` (2025-07-18). The
version tags are development milestones on this branch, not releases of the repository: what is released is decided by the
repository owner through a pull request into `develop`. Evidence is in the development reports
(the project's `Reports/` folder; a snapshot up to 2026-10-19 is in `docs/reports/` at tag `v1.5.1`).

Version names: v1.0.0 is the first product version. Later versions are cumulative. A version tag marks the code; the models of a
version are fitted with the recipe given here. Maintained with every code commit (see the end of this file).

---

## v1.6.0 (in progress): winter fill for missing inputs, models refitted

- **Missing inputs at polar night** (`model/CompositeDayFile.py`, `fillMissing`; F3, `ed0c13f`, 2026-10-21). A missing MOD44C granule used to enter its 48-day
  period as 10000 at every pixel. In polar night the granules that exist are set to 16000 by the low-sun substitution (LE0), so a
  tile with a missing polar-night granule composited its winter periods lower (e.g. 14000 instead of 16000) than a neighbour without
  the gap. Every v01 tile lacks MOD44CQ days 337 and 353 in 2019, and percent bare showed a step of about 17 points where v01 meets
  v02. Now a missing granule is 16000 wherever every present input of its period is 16000, and 10000 elsewhere, as before; its masks
  stay all bad. **Product change** on tiles with a missing input (composites change only at those pixels); **exact** on tiles with
  all inputs. Tests: `FillMissingTestCase` in `model/tests/test_CompositeDayFile.py`. The models must be refitted with this code
  (the v01 training tiles' predictors change); v1.6.0's models are.
- **Global 2019 build and acceptance** (2026-10-22). All 318 tiles of 2019 mapped with this code and freshly fitted models. The
  bare step at the v01 / v02 boundary went from about −17 to −0.4 points (11 of 12 tile columns within ±5). Held-out labels: tree a
  tie (+0.0014 MAE); bare +0.047 MAE, all of it in the v01 and v02 rows. Greenland ice interior unchanged (100%). Quality and Cloud
  layers identical to v1.5.1 on every tile. Away from the boundary, unlabelled drylands moved by 0.5-2 points of bare (refit
  sensitivity far from the labels; see `docs/ALGORITHM.md`, section 10).
- **Known limitations** written up in `docs/ALGORITHM.md` section 10 (regions far from the labels, low SD is not correctness,
  regional label conventions) (2026-10-22; hash filled in by the next commit).
- This is the version prepared for MODAPS (v1.5.2 was not sent); the hand-over has not happened yet.

## v1.5.2 (tag `v1.5.2`, 2026-10-21): MODAPS delivery clean-up

- **Conda environment and documentation** (`e4f77c4`, 2026-10-20). `environment.yml` (conda-forge only, pinned to the product
  versions; GDAL 3.8.5 for 3.8.4; numba 0.61.0 for the compact forests) and `environment-optional.yml` (xgboost); `docs/BUILD.md`,
  `ALGORITHM.md`, `RUNNING.md`, `TRAINING_LABELS.md` (the label archive set 7.0.1 and the h16v02 row-order note); the v1.5.x predictor
  lists `docs/predictors_tree.txt` (64) and `docs/predictors_bare.txt` (21); README rewritten. **Proof** of the v1.5.2 code in this
  environment (activated; no container, no `core`) against the global 2019 v1.5.1 run on h12v09, h17v01, h16v02, h20v09: composites
  byte for byte, metrics and the 7 map layers by value, V4r inputs, the split rows of those tiles and the three refitted models
  byte-identical; `compose.json` differs in its land / dense statistics only (C1) (`FullRuns/EXP_modaps_cleanenv2_20261020/out/compare.txt` + `compare_addendum.txt`).
- **Unit tests** (`4fa8a85`, 2026-10-20). `model/tests/h08v04-2019-pcttree-training+obsForRF.parq.skip` restored (the sweep
  removed it as unused, but `test_MasterTraining` uses it). `test_VcfPredict_roundingHighLat` brought up to date with v1.4.1-v1.5.1
  (stand-ins for the blend partner, BP10, `chain.json`, Quality / Cloud and the rule-tile guard; assertions unchanged). The 8 test
  files that read no NCCS data pass in the clean conda environment. No code change.
- **Blend A compose statistics without C6** (`8bce306`, 2026-10-20). `V4r.composeStats` read C6 MOD44B for its land and dense
  pixels; they now come from the run's own tree map (land = map <= 100, dense = 60-100). Project lead, C1. `<tile>_compose.json` is
  diagnostic only (nothing reads it back): its land / dense / gate statistics change, **no product value changes**.
- **Tiles with no MOD44W skipped and listed** (`4f2790f`, 2026-10-20). `YearRun` gives such a tile no tasks, writes it to
  `status/skipped_no_mod44w.txt` and logs a warning, instead of a map task that failed twice (h18v14 in 2019). Project lead, Q4 (skip
  and log). **Exact** on every other tile.
- **Input locations in `model/Settings.py`** (`6f42ab4`, 2026-10-20). MOD44C, MOD44W and MOD09A1 folders from `VCF_MOD44_DIR`,
  `VCF_MOD44W_DIR`, `VCF_MOD09A1_DIR`, with the NCCS ADAPT paths as defaults (our runs unchanged); `VcfPredict` re-exports
  `MOD44_DIR` / `MOD44W_DIR`; `view/makeMetrics.py` uses them (its own hard-coded paths removed; `--mod09Dir` defaults to the
  setting). **Exact.**
- **No dependency on ilab `core`** (`69bffd4`, 2026-10-20). `Metrics.readMetric` opened metric GeoTIFFs through `core`'s
  `GeospatialImageFile` (the package's only use of `core`); it now opens them with GDAL, with the same checks (file exists, GDAL opens
  it, valid spatial reference). The project lead chose this over copying the class (MODAPS delivery, plan step 4.2). **Exact.**
- **Comments and help text that named removed files** (`cd33f97`, 2026-10-20). `model/BlendA.py`,
  `model/CompositeDayFile.py`, `model/VcfPredict.py`, `model/YearRun.py`, `view/vcfPredict.py`, `view/fitAltModel.py` and
  `README.md` named `blend/` scripts, `rfFitFromPge61.py`, `vcfPredictQueue.py` or `docs/reports/`; they now name the package step
  (e.g. the cloud counts come from `model/V4r.py` in the metrics task) or say where the old file is (tag `v1.5.1`). `model/V4r.py`'s
  comments follow with its own change. **No value change**: comments, docstrings, `--help` strings and one error message only (the
  code is identical with string constants ignored).
- **Clean sweep: files an operational run does not use removed** (`2345db1`, 2026-10-20). Project lead,
  2026-10-20: v1.5.2 holds only what production runs plus the options kept as testing only (GPU Monte Carlo, MOD09A1 input,
  xgboost), so the MODAPS team sees what they run. Tag `v1.5.1` and the history keep everything. Removed (366 files): `blend/`
  (v1.4.0 standalone chain, replaced by the package steps; its last gap, the acceptance test, is the entry below); rlgill's 2024-25
  training path (`model/BuildTraining.py`, `view/buildTraining.py`, `masterTraining.py`, `trainingStatus.py`, `parquetToCsv.py`;
  replaced by `TrainingManifest`); `view/rfFitFromPge61.py`, `view/vcfPredictQueue.py`, `view/stateDump.py`, `view/stateDump2.py`;
  `model/CEquivalentMetrics.py`, `model/Utils.py`, `model/ProductTypeMod09G.py`, `model/Band.py` and their tests; `projects/`
  (Weka reproduction), `notebooks/`, `view/editLabeledTraining.ipynb`; `data/` (a 2025 copy of label files nothing reads; the
  labels are delivered as a separate tar); `docs/reports/` (snapshot of the internal reports; new docs follow); the ilab
  `requirements/Dockerfile` and Docker Hub CI (MODAPS uses no containers; a conda `environment.yml` follows).
  `model/tests/test_VcfPredict.py` uses 4800 x 4800 directly instead of `Band.ROWS`/`Band.COLS`. **No value change** (no
  production module changed). Check: every kept module imports and every kept command's `--help` runs in the ilab container,
  except the 3 GPU Monte Carlo modules, which need CuPy on a GPU node (unchanged) (`FullRuns/EXP_sweep_import_20261020/out/check.txt`).
  The full exactness proof runs on the finished v1.5.2 tree (`Reports/MODAPS_Delivery_Plan.md`, step 5).
- **Partner split acceptance test moved into the package** (`949f414`, 2026-10-20).
  `TrainingManifest.accept` and `trainingManifest.py accept` replace `blend/accept_split.py`, the one v1.4.0 `blend/` script with no
  package equivalent: the blend A partner's split (V4r root, low-sun off) must have the same keys and labels as the standard split
  of the same manifest, and at rows V4r never replaced, every predictor must equal the standard value. Unlike the old script, the
  value check skips tiles with low-sun substitution (since v1.4.0 the standard split is LE0 there, so values differ by design); it
  reports which tiles it compared. Exit 1 on FAIL. Needed by MODAPS, who refit the partner every year (`Reports/MODAPS_Delivery_Plan.md`,
  section 7.1). **No value change** (a check only; nothing calls it in a run). Evidence: on the global 2019 v1.5.1 splits, old and
  new give the same rows, keys, labels and replaced shares; new PASS (27 tiles value-compared, 0 differing cells); standard = bare
  split gives FAIL, exit 1 (`FullRuns/EXP_accept_proof_20261020/out/compare.txt`).

## v1.5.1 (tag `v1.5.1`, 2026-10-19): memory fixes and the v15-v17 rule tiles

v1.5.0's models with less memory per task, and rows v15-v17 mapped by rule (project lead, 2026-10-19). The models are v1.5.0's
(no refit). Every item is **exact** against v1.5.0 except the v15-v17 tiles, which are new. Memory per task (ngaproc, 2019): split
~10x less, map 18 -> 2.4-3.5 GB, metrics 34.8 -> 30.9 GB on a blend tile (`Reports/Run_Cost_Reference.md`).

- **Rows v15-v17 mapped by rule (Antarctica)** (`2f162db`, 2026-10-19). There is no MOD44C for
  these tiles, so `model/RuleTile.py` marks them and `VcfPredict.forRuleTiles` writes their 7 layers from MOD44W alone, with no
  metrics and no models: land tree cover 0, non-vegetated 100, non-tree vegetation 0, both SD layers 0, Quality and Cloud with all 8
  period bits set (255; as on a v00 tile's periods without input); water 200 and out-of-projection 250 in the cover layers, 0 in the
  others. `YearRun` gives a rule tile only its map task (no metrics task, no model dependencies); `view/vcfPredict.py` needs the
  model files only for other tiles. MOD44W has 27 tiles in these rows (h14v15 and h17v15 are all water). Project lead, 2026-10-19
  (to-do 26, global plan G14). **Product change for rows v15-v17 only** (new tiles; never mapped before; C6 has none). **Exact** on
  every other tile: 7 layers on h12v09, h17v01, h16v02, h20v09 equal the previous code's, and the run driver's tasks, dependencies
  and commands are identical. Evidence: `Reports/V15_V17_Rule_Tiles_Results.md` (`FullRuns/EXP_ruletiles_20261019`).
- **The metrics task frees its main-phase data before the V4r build** (`607d667`, 2026-10-19).
  `YearRun.warmMetrics` deletes the low-sun (LE0) phase's `Metrics` object and collects it before the V4r composites are built on
  blend tiles; before, it stayed alive (about 7 GB) through the V4r phase. **Exact:** h12v09's metrics root (composites byte for
  byte, metrics by value), V4r root and counts equal those of the previous code. Peak of the task 34.8 -> 30.9 GB (main process at
  the V4r start 7.0 -> 3.4 GB; V4r metrics phase 23.8 -> 17.3 GB), 751 -> 705 s. Evidence: `Reports/Metrics_Memory_M1_Results.md`.
- **Maps predicted a block of lines at a time** (`13c14fd`, 2026-10-19). `VcfPredict` predicts the ensemble
  members on about 1 M pixels (208 lines) at a time, each block's float32 predictor rows read straight from the metric files
  (`Metrics.iterTableFromRf`: every file opened once, read a block of lines at a time), as the C code predicts pixel by pixel; no
  whole-tile predictor table is held. `getTableFromRf` and the split reader share the same block reading (`_openBands`, `_readBlock`).
  **Exact:** 7 map layers on h12v09, h17v01, h16v02, h20v09 equal v1.5.0's, also with two map tasks at once; splits unchanged. Map task
  peak (with memory-mapped models) 7.3-8.6 -> 2.4-3.5 GB (from 18.0-18.6 GB at v1.5.0), same time; two tasks on one node 4.7 GB.
  Evidence: `Reports/Map_Block_Prediction_Proof_Results.md`.
- **Compact models memory-mapped by default** (`f91a599`, 2026-10-19). `VcfPredict._loadModel` maps a
  compact model's arrays unless `VCF_COMPACT_MMAP=0` (before: only with `VCF_COMPACT_MMAP=1`). Map tasks on one node share the model
  pages through the file cache, and the pages are not private memory. Identical predictions (v1.4.1 S10; again 2026-10-19: 7 layers on
  4 tiles). About 3 GB less private memory per map task. Evidence: `Reports/Map_Memory_Proof_Results.md`, `Reports/Map_Block_Prediction_Proof_Results.md`.
- **Map inputs read from the metric files in blocks** (`d963313`, 2026-10-19). `Metrics.getTableFromRf`
  builds a map's full-tile float32 predictor table (one row per pixel, one column per distinct predictor) a block of 256 lines at a
  time, straight from the metric files, without holding whole files or a second copy; `VcfPredict` uses it for the ensemble-member
  prediction (production: median + SD layers). Before, every band of every metric file was cached, then copied into a DataFrame and
  again into a float32 array. The block reading is shared with `getRowsFromRf` (`_rfByFile`, `_readBlocks`). **Exact:** the 7 map
  layers equal v1.5.0's on h12v09, h17v01, h16v02, h20v09 (blend on h12v09, h20v09); the tree and bare splits are unchanged. Map task
  peak 18.0-18.6 -> 9.6-10.4 GB. With `VCF_COMPACT_MMAP=1` (memory-mapped compact models, already available): 7.3-8.6 GB, and two map
  tasks on one node 14.4 GB instead of 21 GB. Evidence: `Reports/Map_Memory_Proof_Results.md`.
- **A stopped run ends every process of its tasks** (`3414283`, 2026-10-19). `YearRun` starts each task in
  its own session (process group) and, on STOP, sends the group SIGTERM, then SIGKILL to whatever is left after 30 s; before, only
  the task's first process got SIGTERM, and a split's pool workers were left running (once 215 GB of orphans held a node).
  `TrainingManifest.build` terminates its pool workers on any exception, and `view/trainingManifest.py` turns SIGTERM into a normal
  exit so that path runs. Values unchanged (in the same proof). Stop test: STOP during a 6-worker tree split; 60 s later no process
  of the split's group and none under the run root was left (the group needed the SIGKILL after the 30 s grace). Evidence:
  `Reports/Split_Memory_Fix_Proof_Results.md`.
- **Training rows read from the metric files in blocks, not whole files** (`ae17543`, 2026-10-19).
  `Metrics.getRowsFromRf` returns the predictor values at given pixels without holding whole metric files: each file is read once
  with GDAL in blocks of 256 lines, only the bands the predictors need and only lines holding a training pixel, and GDAL's block
  cache is flushed after each block (the files are pixel-interleaved, so one read gives every band of every pixel in it).
  `TrainingManifest._tileFeatures` uses it instead of `getMetricFromRf` per predictor, which loaded every band of every file (e.g. all
  72 of UnsortedMonthlyBands for 24 used) into the read cache and, with a finished tile not freed until garbage collection, made each
  split worker hold 2-3 tiles at once (108-140 GB per 6-worker split). The name parsing of `getMetricFromRf` moved into `_rfParts` and
  `_bandName`, which both use; prediction still reads through `getMetricFromRf`. **Exact:** splits (tree, bare, partner), models,
  metrics and the 7 map layers equal v1.5.0's on 4 map tiles + 15 metrics tiles. Split alone, 13 tiles, 6 workers: tree 38.3 -> 3.6 GB
  peak and 204 -> 29 s; bare 46.3 -> 4.8 GB and 104 -> 21 s. Evidence: `Reports/Split_Memory_Fix_Proof_Results.md`
  (`Reports/Split_Memory_Step1_Results.md` for the cause).

## v1.5.0 (tag `v1.5.0`, 2026-10-19): new bare training set, no code change

**Model change, bare (Percent_NonVegetated) only.** The code is v1.4.1 (`f401421`) unchanged. The bare model is refitted (RF 60
trees, sqrt, leaf 1, seed 0, the 21 v1.4.1 bare predictors) on 33 label tiles instead of 20:
- v1.4.1's 20 tiles;
- plus 13 legacy bare label tiles (h08v06 h09v06 h13v02 h17v07 h18v08 h20v07 h21v07 h21v09 h26v03 h26v04 h27v03 h30v11 h31v09),
  with h10v02 and h21v08 left out;
- plus the 10,000 h16v02 ice points, re-sampled from that year's metrics.

The manifest is drawn with the fresh-sample mode (P4). The tree model and the blend partner are v1.4.1's, unchanged.

Effects (2019, 69 product tiles, against v1.4.1):
- Quality, Cloud and tree SD are identical. Tree cover changes only through the bare-dependent rules (BP10: 19,960 px; blend raw-bare
  cap), with 0 other tree differences.
- Bare: mean −0.91 (tile range −14.4 to +1.6). The lowering is on the pattern-A tiles (South America, central and southern Africa,
  Australia), which all move closer to ESA WorldCover.
- Held-out bare MAE (old 21-tile set): 3.8022 vs 3.7942 (+0.008; tile bootstrap 95% CI [0.000, +0.017]), accepted as a tie by the
  project lead.
- Ice tiles are unchanged within limits.

Run: `FullRuns/v150_20261018` (84 maps). The models are not in the package. Evidence: `Reports/Version_v1.5.0_Build_Results.md`,
plan `Reports/Version_v1.5.0_Build_Plan.md`, candidate `Reports/T2c_Drop_h10v02_Results.md`.

## v1.4.1 (tag on `f401421`, 2026-10-18): code tidy-up and production benchmark

Goal: the v1.4.0 product from one package, in one driver per year, with exact speed-ups, then a benchmark against the C code (time,
memory, disk). Every item below is **exact** against v1.4.0 unless stated (S7 changes the 2019 blend layers; see its entry). Tagged 2026-10-18 after the parity run (benchmark arm F PASS with the h19v12 exception, `Reports/V141_Production_Benchmark_Results.md`).
Plan: `Reports/V141_Code_Tidyup_And_Full_Retrain_Plan.md`.

- **Fresh-sample mode: the training sample drawn from the label rasters** (P4, `f401421`, 2026-10-17).
  `TrainingManifest.sample` / `trainingManifest.py sample` writes a manifest straight from the label files, using the rule the
  product's splits were made with. Every labelled pixel (value != 255) of each tile is a row, sorted by key (tid-year, x = row,
  y = column). A row is held out when the pandas hash of its key is 0 mod 5. Extra label rows (`--extra`: the 10,000 h16v02 ice
  points, label 100, a fixed label input whose predictors each run recomputes) are appended per part. Row ids are `range` (bare) or
  `position` (tree), as in the v1.4.0 splits. Exact: from the v5.0.3samp tree labels (63 tiles), the 20 legacy bare label files and
  the ice points, both manifests equal v1.4.0's (values, dtypes, order; 11,440,984 tree and 10,799,061 bare rows), and keys-only
  splits are byte-identical. About 25 s per manifest. A year can now be built with no input from an earlier version (benchmark arm F). Evidence: `Reports/V141_P4_Fresh_Sample_Results.md`.
- **V4r inputs and the blend partner built in the year driver** (S7, `548498e`, 2026-10-17). `model/V4r.py` makes the blend A
  partner inputs (V4r composites, snow-aware cloud counts, compose statistics) per tile from the HDF inputs in low-sun mode off,
  without daily files; the arithmetic is `blend/v4r_year.py`'s. `YearRun` builds them in `metrics:<tile>` for the partner's blend
  tiles and training tiles, and the partner is either fitted in the run (`split:partner` from the V4r root + `fit:partner`; the
  default, since a partner is year-specific), given as a fixed model, or borrowed as before (ready-made V4r metrics and counts).
  About 1.5-2.3 min per tile against about 10 for the v1.4.0 path. **Product change for 2019 blend tiles only:** the V4r composites
  and counts equal v1.4.0's (0 differences on 4 tiles; h12v09 composites byte-identical; counts equal on 61 of 63 tiles), but a
  partner refitted from current inputs cannot equal V4R60. V4R60's training rows on h16v01 predate the Greenland zenith fix
  (`13773db`), and on h24v04 / h27v06 its old build read the 2020065 slot that the pipeline never reads. Held-out: refit vs V4R60
  +0.004 MAE (95% tile bootstrap [-0.002, +0.012]; +0.001 on the 60 tiles with identical inputs): a tie, accepted by the project lead.
  On the 4 test maps, tree cover changes on 3.50 M px of h12v09 and 0.10 M px of h20v09 (non-vegetated: 2 and 283 px); h16v02,
  h17v01 and the SD, Quality and Cloud layers are identical. Tree and bare splits and models remain exact.
  Evidence: `Reports/V141_S7_Proof_Partner_Diagnosis_Results.md`, plan `Reports/V141_S6_S7_S8_Plan.md`.
- **Dailies kept in memory** (S6, `50785f2`, 2026-10-17). `BandDayFile` no longer writes the daily inputs to
  `1-Days` (each is read once, by the worker that builds its composite); composites and metrics are written as before, and existing
  daily files are still read. `makeMetrics.py --writeIntermediates` and the year driver's `writeDailiesTiles` write them for
  diagnostics (`VCF_WRITE_DAILIES=1`, seen by the spawned workers); `blend/days_year.py` always writes them (the v1.4.0 V4r path).
  8.48 GB less written per tile (~2.5 TB per 292-tile year); run time unchanged. Exact: composites, metrics and all 7 map layers
  identical to `532f9eb` on 4 tiles; written dailies byte-identical; a resumed root rebuilds identical composites without dailies; an
  unstamped root without dailies is still refused. Evidence: `Reports/V141_S6_Dailies_In_Memory_Results.md`.
- **Per-tile clean-up of intermediates in the year driver** (S8, `532f9eb`, 2026-10-17). `YearRun` deletes a tile's `1-Days`,
  `2-Composites` and `3-Metrics` once every task that reads them is done (its map, and the splits of the manifests it is in); a tile
  whose map failed is kept; only real folders inside the run's metrics root are deleted (links and outside paths refused); stamps and
  low-sun counts stay; each deletion logged with exact bytes (`status/cleanup.log`). Spec `cleanup {mode: after_map | off, keepTiles}`
  (default after_map) and `maxBacklog` (holds map-only metrics while that many tiles wait for their map); training tiles' metrics run
  first; `runYear.py run --keepIntermediates` keeps everything. Needed for the global run (a 292-tile year writes ~8-9 TB of
  intermediates without it). Values unchanged: bare split and refit equal v1.4.0's, 4 maps (28 layers) identical; 22 tiles freed
  430.9 GB, exactly their size. Evidence: `Reports/V141_S8_Cleanup_Results.md`.
- **One driver per year** (P5, `f24f68a`, 2026-10-17). `model/YearRun.py` + `view/runYear.py`: metrics -> splits -> fits (+ compact
  copy) -> maps as a task graph shared by the nodes through a status folder (claims, slots, memory floors, fit hosts, retry of maps,
  STOP markers), with per-task timing (wall, CPU, peak RSS, bytes) in `timing/tasks.csv`. A 2019 year built from scratch reproduced
  v1.4.0's splits, models (every stored test row) and maps (4 tiles, 28 layers). Evidence: `Reports/V141_S5_P5_Results.md`.
- **Training rows built per tile in parallel** (S5, `da96b2a`, 2026-10-17). `TrainingManifest.build(workers=N)`,
  `trainingManifest.py --workers`. Byte-identical to the serial build; tree split of 63 tiles in 824 s (peak 129.5 GB with 6 workers).
- **Training samples as key manifests** (P4, `954045b`, 2026-10-16). `model/TrainingManifest.py`, `view/trainingManifest.py`: a split
  becomes a manifest of pixel keys + labels, and a split is rebuilt from a manifest and a metrics root. Keys, labels and row index equal
  v1.4.0's on every row; predictors equal on every tile with metrics on disk. Fresh-sample mode not yet included.
- **Blend A, raw-bare cap and BP10 inside the map step** (P3 + S3, `79cb19f`, 2026-10-16). `vcfPredict` makes the v1.4.0 product in
  one call (`--partner ... --counts`; BP10 on by default, `--noBp10`); `model/BlendA.py`. The raw non-vegetated value is reused from the
  same prediction (no second bare prediction). The four standalone blend scripts are deprecated (kept one version). Equal to the
  delivered v1.4.0 product on h12v09.
- **Year required everywhere** (P2, `bce3a08`, 2026-10-16). `blend/rawbare.py`, `bp10_apply.py`, `recompute_split_onepass.py`,
  `view/rfFitFromPge61.py` no longer assume 2019.
- **Low-sun substitution (LE0) is the package default** (P1, `d8c3ef9`, 2026-10-16). `model/LowSun.py`: modes `le0` (default),
  `s70`, `off` (= v1.3.0); `--lowSun`, `--lowSunCheck` on the CLIs; the `VCF_LOWSUN_*` variables still work for one version
  (deprecated). Each metrics root gets a stamp (`lowsun.json`) and per-day substitution counts (`lowsun_counts.csv`), and a root built
  in another mode is refused. **Note:** 34 of v1.4.0's 69 tiles were mapped from metrics built without the substitution (found
  2026-10-16, Q6); with P1 every new run applies it. On those tiles, wherever low-sun inputs occur (e.g. h20v09: 653 composite pixels in
  period 2019353), new maps differ from the delivered v1.4.0 maps, by design; h12v09 has no such inputs in 2019 and is unchanged.
- **Quality / Cloud layers read the stored composite masks** (S12, `b0eaf05`, 2026-10-14). No composite rebuild per map; a map no longer
  needs write access to the metrics folder. Fixed Cloud flag stored as `-cloudfb.npy` (old `-cloud.npy` files, which hold the inverted
  flag, are never read).
- **`fitAltModel --minSamplesLeaf`** (`a13d107`, 2026-10-14). Default 1, unchanged. Leaf sizes 5-100 were tested and not adopted
  (values compressed).
- **Compact random-forest models** (S9, `ac6d680`, 2026-10-14). `model/CompactForest.py`: the same trees as flat arrays with a numba
  predictor, 6.4x smaller (29 GB -> 4.6 GB); thresholds stored so every float32 input takes the same branch. Identical predictions per
  tree, MEDIAN and SD; a map 712 s / 81 GB -> 377 s / 22 GB. `VCF_COMPACT_MMAP=1` memory-maps a model.
- **Quality and Cloud layers; Cloud flag fixed** (`986892b`, 2026-10-13). `Quality-<tile>-<year>.tif` (per-period fallback) and
  `Cloud-<tile>-<year>.tif` (fallback with no clear day), period 1 in bit 0. The per-period Cloud flag was inverted and is now the C
  code's definition (equal to the C code's Cloud on every bit of 5 tiles). Cover layers unchanged.

Planned before the benchmark (not committed): per-tile clean-up of intermediates in the driver (S8, in test 2026-10-17), dailies kept
in memory (S6), V4r composites and counts in the composite pass (S7), and the blend partner trained in the year run (decision D4,
2026-10-17). Plan: `Reports/V141_S6_S7_S8_Plan.md`.

## v1.4.0 (tag on `53eb115`, 2026-10-13)

**Product change.** v1.3.0 + low-sun substitution (LE0, refit) + blend A + BP10. Maps: 2019, 69 tiles, 3 cover layers.

- **Low-sun bright substitution, LE0** (`b3a7142`). With `VCF_LOWSUN_T=70 VCF_LOWSUN_LE0=1`, Band_1-7 and Band_31 inputs with solar
  zenith above 70 or at or below 0 become 16000 (the intent of PGE61 `fix_bands.c`); models refitted on these inputs. Fixes the
  Greenland ice interior and the Iceland / h17v01 weak years. With the variables unset the code is v1.3.0.
  `VCF_ZENITH_NDVI=0` switches the zenith-only NDVI rule off. Evidence: `Low_Sun_Bright_Substitution_S70LE0_Results.md`,
  `Bundle_PreDecision_Checks_Results.md`.
- **Blend A, raw-bare cap, BP10 as standalone scripts** (`53eb115`, `blend/`). Tree cover on rows v06-v11 (30N-30S) blended with a
  partner model (V4R60) trained on V4r composites (a cloud-adapted composite), weighted by a cloud count and the clear model's cover;
  the sum cap uses the raw non-vegetated value; BP10 sets tree 0 / non-vegetated 100 where raw non-vegetated is 100 and tree is 1-9.
  Evidence: `BlendA_Full_Product_v130_Results.md`, `BlendA_Other_Years_Results.md`, `Bare_Priority_Cap_Pixel_Count.md`. Moved into the
  package in v1.4.1 (P3).

## v1.3.0 (tag on `7036b6a`, 2026-10-10)

**Product values same as v1.2.0** (speed-ups exact); one fix enables other years.

- **Parallel composites** (`f3deac7`, `7036b6a`). `Metrics.compositeWorkers`, `makeMetrics.py --workers`: a band's composite periods
  built in processes (GDAL serialises HDF4 reads within one process). Metrics 799 s -> 376 s per tile with 8 workers. Exact.
- **Optional MOD09A1 input** (`9e70feb`). MOD09A1 QA rule and cloud mask; `makeMetrics.py --productType`, `--mod09Dir`. Tested on
  h09v05 2019, not adopted as the product input. MOD44C results unchanged.
- **QA and cloud masks built once per day** (`ca764df`). Not once per band: HDF reads per tile 575 -> 253. Exact.
- **Metric GeoTIFFs written once** (`bb2552e`). No per-band `FlushCache`: metric writes 417 s -> 33 s per tile. Pixel values,
  metadata and layout identical (files ~48 KB smaller).
- **Predictor dates mapped to the metrics' year** (`0a5d21a`). `Metrics.getMetricFromRf` did not do what its comment said, so no model
  could run on another year. Identity for 2019. Used for the 2010 and 2014 products.

## v1.2.0 (tag on `45edf62`, 2026-10-07)

**Model change only** (code = v1.1.0). 10,000 non-vegetated training rows labelled 100 on the h16v02 Greenland ice sheet (regular grid
at least 10 km inside the MOD44C ice edge); non-vegetated model refitted with the v1.1.0 recipe. h16v02 interior ice below 100:
8.7% -> 0.007%; held-out labels within seed noise.

## v1.1.0 (tag on `920d06f`, 2026-10-04)

**Product change.** Random forests of 60 trees for both layers, combined with the MEDIAN of the trees (`--tcAgg median --nvAgg
median`); held-out labels better for both layers (`Candidate_60Trees_Median_Results.md`).

- **MODIS sinusoidal grid in the outputs** (`920d06f`). The GeoTIFFs had no geotransform and a WGS84-ellipsoid projection; they now
  carry the MODIS sphere and the tile's geotransform (checked against C6.1 MOD44B granules).
- **uint8 percent layers, uint16 SD layers** (`0298e4e`). As the historical MOD44B (water 200, out of projection 250, no data 253);
  values unchanged.

## v1.0.0 (tag on `16d6474`, 2026-10-02)

**The first product version**: bit-15 QA, random forests (30 trees, sqrt) for tree cover (MEDIAN) and non-vegetated (MEAN), rounded
values, SD layers, Greenland zenith-only NDVI, high-latitude rule on v00 tiles. Definition: `Product_Versions_and_Work_Map.md`.

- **Rounding and v00 rule** (`16d6474`). Values rounded (`floor(v + 0.5)`, as the C code) instead of truncated, which biased both
  layers down by about half a point. On v00 tiles every land pixel is tree 0, non-vegetated 100.
- **Zenith-only NDVI on 6 Greenland tiles** (`13773db`). NDVI set to 0 in composite periods with no day of 0 < solar zenith < 65.
- **Member pass, aggregation and SD layers** (`6684b88`, `f2d9190`, `92c7edc`, `615557d`). Each ensemble member predicted once;
  MEAN / mid20 / MEDIAN per layer (`--tcAgg`, `--nvAgg`); `Percent_Tree_Cover_SD` and `Percent_NonVegetated_SD` as round(100 x SD)
  (`--noSd` to skip). Unit tests; README section on fitting and layers.
- **C code's 21 non-vegetated predictors as the default** (`2778628`). The hand-written list had translated the C code's indices in
  the wrong layout (13 of 18 names were not the C code's predictors).
- **`fitAltModel.py`** (`8f42c83`). Random forest, XGBoost and HGB fits with the same inputs and outputs as `rfFitFromPge61.py`.
- **Predictions clamped to 0-100; layers sum to 100** (`2a37f9e`). Non-vegetated is reduced to absorb any overshoot; water and
  out-of-projection codes carried into non-tree vegetation.
- **CPU models** (`c69c32d`). cuML `RandomForestRegressor` replaced by scikit-learn models (no GPU on the processing nodes); parallel
  prediction in row blocks; `vcfPredictQueue` predicts many tiles with the models loaded once. A design change.
- **Python mirrors of the C code's derived metrics** (`e0e6478`). `CAttributes`, `CMirrorMetrics` reproduce the C code's 252 derived
  attributes exactly; metric files read once per tile.
- **QA and band fixes for C-code parity** (`1c3ac20`). Adjacency is state bit 6; the aerosol test is MOD44C state bit 15 (PGE72's
  overall QC flag; bits 7-8 overlap other fields and rejected 29-50% of good pixels); zenith cutoff 65; Bands 1-7 clamp removed (the C
  code's never fires); Band_31 clamp at 70; `BuildTraining` honours `trainingDir`.

## Documentation snapshots (no code change)

`docs/reports/` snapshots of the development reports: `335ceb2`, `fe56c5f`, `88baf42`, `0959749`, `f99704a`, `3f8097a`, `3ec71c1`,
`3e4f766`, `41556d0`, `6852581` (2026-09-30 to 2026-10-07).

---

## Keeping this file current

Every code commit on the delivery branch adds its entry here **in the same commit**, under "Unreleased" (or the version being built):
what changed, why, whether values change (exact / product change / model change), the commit's evidence report. When a version is
tagged, the "Unreleased" heading becomes the version heading with its tag commit and date.
