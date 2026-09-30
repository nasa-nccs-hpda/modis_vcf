# Python-side two-tile retrain test ("Track B") — plan, code changes, and status

> **Historical (two-tile test, 2026-09-18); superseded 2026-09-20.** The 63-tile run did not use these two-tile models or the hard-coded predictor lists
> described here: it used predictors generated from C's own attribute tables (64 tree, 21 bare) and a further modified code copy in the run root
> (`Reports/FullRun_Launch_and_Status.md`). The code changes described below (`--nEstimators`, predictor-name fixes, CPU predict) remain in the sync copy `Python_code/`.

Status: **completed 2026-09-18; the sections below are the running log and the top half is stale
(Step 4 was later unblocked and both tiles predicted). See the "Update" section and
`Signoff_Package_2tile_C_vs_Python.md`.** Started 2026-09-18. Companion to
`Bands17_Fix_Validation_Results.md` (documents the Python-only
`BandDayFile.py` value-clamp fix this run trains against) and to
`C_code/pge61/build_context/runtime/track_a_launch.log` /
`b2train_2tile/track_a.log` (the C-side equivalent, "Track A" — also
2-tile, also stopped after training, for an unrelated reason: a false-
failure in its own model-count check, see below).

## Why this exists

`Reports/Bands17_Fix_Validation_Results.md` validated a Python-only fix to
`BandDayFile.py`'s zenith value-clamp logic. The natural next question —
does retraining on the corrected metrics produce a sane model? — was
partly explored in an earlier, undocumented ad hoc session (its only
trace was three output files under `Metrics/TrackB_2tile/`: a tree-cover-
only `.joblib` model and two raw `.npy` prediction arrays, produced by a
script that matched no naming convention in the checked-in codebase and
no longer exists anywhere on disk). This report replaces that with a
real, reproducible, documented run using the actual checked-in pipeline
scripts.

## Plan (as approved)

Mirror Track A's structure (extract → consolidate → train → [predict]),
scoped to the same two tiles (h09v05, h12v02), using the already-
validated `Python_bands17_band31fix` metrics as the data source (no
regeneration needed):

1. **Extract** (B1-equivalent): `buildTraining.py`, once per (tile ×
   trainingType) = 4 runs — `pcttree` and `pctbare`, both tiles.
2. **Consolidate**: automatic via `MasterTraining`'s glob, formalized here
   as a small new glue script (see below).
3. **Train** (B2-equivalent): **Option B** — `rfFitFromPge61.py`, direct
   fit against its existing hardcoded predictor list, no Monte Carlo
   search (mirrors Track A's directness; the alternative, Option A
   `monteCarloSim.py`, does a full predictor search and was not chosen).
4. **Predict** (B3-equivalent): `vcfPredict.py` — **blocked**, see below.

## Code changes made for this run

- **`Python_code/modis_vcf/view/rfFitFromPge61.py`**: added a new
  `--nEstimators` CLI argument (default `1`, preserving existing
  behavior for any other caller) and used it in place of the previously
  hardcoded `n_estimators=1`. This run passes `--nEstimators 30`, so one
  invocation produces a complete `BaggingRegressor(DecisionTreeRegressor(),
  n_estimators=30)` ensemble — matching both C's 30-member
  `weka.bare.NN.model` structure and yesterday's `RandomForestRegressor`→
  `BaggingRegressor` fix (see `Sync_Plan_and_Model_Retraining_Alternate.md`
  and yesterday's edits to `MonteCarloSim.py`/`rfFitFromPge61.py`/
  `view/monteCarloSim.py`, all confirmed via diff against the older
  `test_code/modis_vcf/` reference copy since `Python_code/` isn't
  git-tracked).
- **New: `Reports/build_xy_from_training.py`**. `buildTraining.py` writes
  one combined (predictors + label) Parquet per tile; `rfFitFromPge61.py`
  wants separate `xTrain.parq`/`yTrain.parq`. This script uses
  `MasterTraining` (the same class `monteCarloSim.py` uses internally) to
  glob-combine both tiles' per-type files, split label from predictors,
  do an 80/20 train/test split, and write `X/xTrain/xTest/y/yTrain/
  yTest.parq` into a per-trainingType subdirectory.
- **New: `Reports/run_py_track_b_2tile.sh`**. Driver script, orchestrates
  the 4 extract runs → 2 consolidate runs → 2 train runs, each stage in
  parallel, with real per-file existence checks logged to
  `Metrics/PyTrackB_2tile/status.log` (deliberately not a bare
  process-exit-code check — see the Track A bug note below).

## Bug found and fixed during launch: wrong `--metricsDir`

First launch attempt passed `--metricsDir` pointing directly at each
tile's `.../Python_bands17_band31fix/<tile>/2019/3-Metrics` directory.
`Metrics.py`'s constructor (`Metrics.__init__`, line ~88) appends
`tileId/year/3-Metrics` itself (`self._outDir = outDir / tileId /
str(year)`), so passing the `3-Metrics` dir directly produced a doubled,
nonexistent path. Because that path didn't exist, `Metrics.getMetric()`
silently took its regenerate-from-scratch fallback instead of reading
the already-computed `.tif` files — reading raw MODIS granules and
rebuilding composites under a bogus nested
`3-Metrics/<tile>/2019/{1-Days,2-Composites,3-Metrics}` subdirectory
inside the validated output tree. Running 4 of these full regenerations
in parallel exhausted memory: 2 of the 4 processes were OOM-killed after
~25-30 minutes; the job never reached Step 2.

Fix: `--metricsDir` corrected to the parent directory
(`.../Python_bands17_band31fix`, not `.../$TILE/2019/3-Metrics`).
Verified via a capped (90s) foreground smoke test showing it now reads
the existing validated `.tif` files directly (no regeneration). The
bogus nested subdirectory was deleted; the 25 validated top-level files
per tile (24 metric `.tif`s + `MOD44-NDVI.bin`) were confirmed untouched
throughout — this only wasted compute/time, it did not corrupt or
overwrite the validated Bands17/Band31 fix output.

## Blocker, not worked around: Step 4 (predict)

`vcfPredict.py` hard-imports `cuml.ensemble.RandomForestRegressor`
(RAPIDS, GPU-only). Confirmed before launch: forest203 has no GPU
(`nvidia-smi` not found) and `cuml` fails to import in the
`modis-vcf-latest` container (`ImportError: libcuda.so.1`). The driver
script stops after Step 3 and logs `STEP 4: SKIPPED` rather than
attempting an unreviewed GPU-free reimplementation of the prediction
class. This mirrors Track A, which also never reached its own B3/predict
stage (for an unrelated reason — see next section). Needs a decision:
run on a GPU node, or write and validate a CPU-only prediction path.

## Cross-reference: Track A's own Step 4 bug (why it also never predicted)

Track A's C-side driver (`C_code/pge61/build_context/scripts/
run_track_a_2tile.py`, line 169) checked training success by globbing
`ARFF_DIR.glob("weka.%s.*.model" % YEAR)` — but `ARFF_DIR` is the
*input* ARFF directory, not `VCF_outputs/`, where the 30 `weka.2019.NN.model`
files are actually written. All 30 B2 model runs genuinely succeeded
(confirmed: all 30 `.model` files exist and are valid in
`/explore/.../MOD44B_VCF_Development/VCF_outputs/`), but the glob always
found 0, so the script logged "0/30 weka.2019.NN.model files present" and
exited before ever reaching B3. This is why this Python run's own status
log does real per-file existence checks against the actual output paths
rather than trusting exit codes or a similarly fragile count.

Separately (also discovered, unrelated to this run): the C-side
`VCF_outputs/` directory is a large, shared, non-git, non-versioned
staging area used across 60+ tiles and many past sessions, with
generic, non-run-scoped model filenames (`weka.2019.NN.model`) — Track A
almost certainly overwrote a previous, differently-scoped model set
there with no backup. This Python run avoids that failure mode by
writing to a new, scoped directory (`Metrics/PyTrackB_2tile/`) instead
of any shared location.

## Current status (as of this report)

Relaunched (second, corrected attempt) at 2026-09-18 14:35:39, fully
detached on forest203 (`setsid nohup ... & disown`, verified via
independent `ps` — no controlling TTY, survives disconnect). Step 1 is
running: 1 of 4 extract runs complete (`h09v05`/`pctbare`), the other 3
in progress, no errors, all reading the correct validated metrics files
this time.

**Outputs, once complete:**
- `Metrics/PyTrackB_2tile/2-Training/` — 4 per-tile parquet files, then
  `pcttree/` and `pctbare/` subdirs with `xTrain.parq`/`yTrain.parq`/etc.
- `Metrics/PyTrackB_2tile/3-Models/` — `pcttree.bin`/`pcttreeTopN.bin`,
  `pctbare.bin`/`pctbareTopN.bin` (joblib/pickle).
- `Metrics/PyTrackB_2tile/status.log` — step-by-step progress with real
  per-file existence checks.
- `Metrics/PyTrackB_2tile/logs/` — full stdout/stderr per sub-run.

**Not done, and not started without explicit direction:** Step 4
(predict), blocked on GPU/cuML availability as described above.

## Update, later on 2026-09-18

(Supersedes the "Blocker" and "Current status" sections above: Step 4 was
unblocked by CPU changes and has now run for h09v05.)

- **Steps 1-2 completed** (14:41). **Step 3 failed** the first time (14:42): the
  hardcoded predictor lists in `rfFitFromPge61.py` spelled 15 entries
  `...-Band31` while the real column names are `...-Band_31` (confirmed against
  the parquet schema). Fixed by adding the underscore in all 15 places; Step 3
  was then rerun directly (not via the driver script, so `status.log` still
  shows the stale `STEP 3 FAILED` line). `pctbare.bin` finished 15:05 (~24 min);
  `pcttree` was still training at 16:26 (single core: the `BaggingRegressor` in
  `rfFitFromPge61.py` has no `n_jobs`, so its 30 trees fit sequentially).
- **Step 4 CPU changes made (not yet tested):** `VcfPredict.py` now imports
  sklearn `BaggingRegressor` (aliased as `RandomForestRegressor` so the type
  hints still read) and calls `rf.predict(X).astype(...)` without `.to_numpy()`;
  the cuML import and the cuML `.to_numpy()` call are commented out, not
  deleted. New local shim `Python_code/modis_water/model/Utils.py` vendors the
  single `writeRaster` method from the `modis-water-latest` container. The
  smoke test waits on `pcttree.bin`; the container run will need
  `Python_code/modis_water` bound to `/usr/local/ilab/modis_water`.
- **Training-data provenance check** (2,321,403 `pcttree` / 1,191,628 `pctbare`
  rows total): row counts equal the labeled-pixel counts (value != 255) of the
  two tiles' `Python_code/modis_vcf/data/training/VCF_training_<type>_<tile>.7.0.0.out`
  rasters exactly, so no other tiles were picked up. `BuildTraining.py` reads
  those `.7.0.0.out` files; its header comment naming `v5.0.3samp` as the source
  is not what the code uses. The `v5.0.3samp/<tile>.samp.bin` sets are much
  smaller (h09v05 268,039; h12v02 307,939 labeled px) and match the tree-cover
  labels only: for `pcttree` they are a strict subset of the `.7.0.0.out` labels
  with identical values (the `.7.0.0.out` files add 1,284,233 px for h09v05 and
  461,192 for h12v02); for `pctbare` they do not match. Note h12v02's `pcttree`
  labels are all 0 in both sets, so it adds 769,131 zero-cover rows.
- **Step 3 finished:** `pcttree.bin` (670 MB) written 16:32, `pctbare.bin` (1.6 GB)
  15:05.
- **Step 4 mechanics check passed** (h09v05, container with `modis_water`
  bound): both models load, `getMetricFromRf` works for all 60 tree predictors
  (401 s), `predict` on 1,000 rows returns a numpy array (0.0-51.8), water mask
  loads, `writeRaster` shim writes a tif. The full h09v05 prediction was then
  launched detached (`Reports/run_py_track_b_step4_h09v05.sh`, output in
  `Metrics/PyTrackB_2tile/4-VcfProcess/h09v05/`).
- **Step 4 full-tile result, h09v05: completed, rc=0** (16:41 -> 16:54, ~13 min;
  tree-cover raster written 16:49, the other two 16:53). All three tifs exist
  (`Percent_Tree_Cover`, `Percent_NonVegetated`, `Percent_NonTree_Vegetation`,
  4800x4800 int16). Checked with `Reports/check_py_track_b_step4_outputs.py`:
  - Water: all 106,052 water-mask pixels are 200 in tree cover and non-vegetated;
    non-tree vegetation over water is 0 (C gives 200). No 250 pixels.
  - Land ranges: tree 0-79 (mean 13.9), non-veg 0-100 (mean 35.9), non-tree veg
    0-100 (mean 50.2); nothing below 0 or above 100.
  - tree + non-veg exceeds 100 on 100,689 land px (0.44%, max 161): not capped,
    as expected from the code (see `Ensemble_Prediction_C_vs_Python.md`).
  - In-sample agreement with the h09v05 training labels (these pixels were in
    the fit, so this is a plausibility check, not accuracy): tree corr 0.990,
    MAE 0.63, mean(pred-label) -0.17 (n=1,552,076); bare corr 0.986, MAE 1.57,
    mean(pred-label) -0.40 (n=441,516). The near-perfect fit is what unpruned
    trees do to their own training data.
  - The tifs carry the sinusoidal projection but no geotransform (origin 0,0,
    1 px units): `VcfPredict._write` never passes `transform` to
    `Utils.writeRaster`. That is the existing code's behavior, not from the shim.
  - Not yet done: h12v02 prediction; held-out (out-of-sample) evaluation;
    comparison with C or production MOD44B output for the same tile.
- How the 30-tree ensemble becomes the final value, and how that differs from C:
  see `Ensemble_Prediction_C_vs_Python.md`.

## Update, 2026-09-19

- **Which label file Python trains on (verified).** `data/training/VCF_training_pcttree_<tile>.7.0.0.out` are
  symlinks to `data/legacy/<tile>.bin` (mtime 2025-07-18, apparently a bulk copy; no source recorded). C's B1
  reads `v5.0.3samp/<tile>.samp.bin` instead. The samp files are a strict subset of the legacy rows with
  identical values. The user asserts the samp file is accurate based on its 2010 date. rlgill's
  `v5.0.3samp-conventionallyNamed/` directory (April 2025) holds `VCF_training_pcttree_<tile>.7.0.0.out`
  symlinks to the samp files, i.e. the exact names `BuildTraining.py` expects. See "Label provenance" in
  `Signoff_Package_2tile_C_vs_Python.md`. Not changed; a swap would be a symlink repoint plus a rerun of
  Steps 1-4 for `pcttree` into a new output directory.
- **Single-core training and prediction** (Step 3 and Step 4 above): a 100k-row check showed `n_jobs=12`
  fits about 8x faster with identical predictions, and `set_params(n_jobs=12)` works on the existing
  `pcttree.bin`. See `Parallel_CPU_Use_Assessment.md`. No code changed.
- **Scope:** this two-tile run verified the pipeline only; full verification needs all 63 tiles.
