# Using more than one CPU in the Python (Track B) pipeline: assessment and cheap check

> **Status (2026-09-20):** the changes below were applied in the run-root code copy (`FullRuns/full63_c_vs_py_20260919-0806/code/python_code/`) for the
> 63-tile run, not in the sync copy: `--nJobs` and `--randomState` in the fit, `set_params(n_jobs)` with threaded, chunked predict and a metric-read cache (which
> removes the repeated reads of the unsorted file described in item 3), and predict with the models loaded once per node. Measured at full scale: tree fit on
> 63 tiles 81 min and bare fit 17 min with 12 jobs; predict about 4 min per tile per node with the 13.5 GB and 11.9 GB models loaded once. Item 3 (parallel
> feature loading) was not done. The 100k-row results below still stand.

Written 2026-09-19. Assessment plus a small measured check. **No pipeline code was changed and no
full-tile or full-training run was made.** Companion to `PyTrackB_2tile_Plan_and_Status.md`, which noted
that `pcttree` trained on a single core.

## Where the pipeline runs on one CPU

| Place | Why | Evidence |
|---|---|---|
| Training, `Python_code/modis_vcf/view/rfFitFromPge61.py` line 171 | `BaggingRegressor(...)` is built without `n_jobs`, so its 30 trees fit one after another | `pcttree` took about 1.8 h (single core) on 2.32M rows |
| Prediction, `Python_code/modis_vcf/model/VcfPredict.py` line 189 | `rf.predict(X)` uses the model's saved `n_jobs`, which is `None` because the model was trained without it | checked: the saved `pcttree.bin` reports `n_jobs None` |
| Feature loading, `VcfPredict.py` line 141 | a plain `for` loop calls `getMetricFromRf` for each of the 60 tree predictors | 401 s for the 60 predictors on h09v05 (`PyTrackB_2tile_Plan_and_Status.md`) |

Host facts (checked 2026-09-19): forest203 and forest204 each have 12 cores and 112 GB RAM and were idle
(load about 0.1). Inside the `modis-vcf-latest` container all 12 CPUs are visible (`cpu_count` 12, affinity
12), so no singularity option is needed. sklearn is 1.2.0, joblib 1.3.2. forest204 answers SSH without a
password and sees the same `/explore/nobackup/people/mcarrol2` and container paths; I did not run the
container there.

## The check (script and log in `Reports/`)

`check_parallel_bagging.py`, log `check_parallel_bagging.log`; run on forest203 in the container.
Standalone: reads the real `pcttree` training rows and real saved model, writes nothing.

**Part 1, fit** on a 100,000-row random subset of the real `pcttree` training rows (60 predictors,
30 trees, `random_state=0`):

| Setting | Wall time | Peak RSS, all processes | Speedup | Predictions identical to `n_jobs=1` |
|---|---|---|---|---|
| `n_jobs=1` | 183.2 s | 1.0 GB | 1.0x | reference |
| `n_jobs=12`, default (loky processes) | 24.0 s | 2.7 GB | 7.6x | yes |
| `n_jobs=12`, threading backend | 22.2 s | 3.0 GB | 8.3x | yes |

**Part 2, predict** with the real saved `pcttree.bin` (670 MB, 30 trees) on the real held-out `xTest`
rows (464,281 rows):

| Setting | Wall time | Peak RSS, all processes | Speedup | Output identical to `n_jobs=None` |
|---|---|---|---|---|
| saved `n_jobs=None` | 7.8 s | 4.2 GB | 1.0x | reference |
| `set_params(n_jobs=12)`, loky processes | 2.9 s | 6.2 GB | 2.7x | yes |
| `set_params(n_jobs=12)`, threading | 1.1 s | 5.6 GB | 7.1x | yes |

What this shows:

- Fitting scales to about 8x on 12 cores, and fixing `random_state` gives the same model whatever
  `n_jobs` is (identical predictions on 100k rows).
- Setting `n_jobs` on an already-saved model with `set_params` works and gives identical output, so old
  `.bin` models need no retraining to predict in parallel.
- The threading backend was as fast as or faster than processes for both fit and predict, with less
  memory in predict. The tree code releases the interpreter lock, so threads are enough.

## Limits of this check

- 100k training rows, not the 2.32M used for the real model, and 464k predict rows, not the 23M of a
  full tile. Speedups at full size are expected to be similar for fit, but were not measured.
- Peak memory at full-tile size was **not** measured and cannot be extrapolated from 464k rows (part of
  the increase is a fixed cost from sending model pieces to workers). Predict in row chunks of about
  1-2M rows, and measure on one tile first.
- Linear extrapolation of the 7.8 s single-core predict time to 23M rows gives about 6.5 min. That does
  not fit with the earlier log timings (tree-cover raster written about 8 min after launch, of which 401 s
  was feature loading), so how prediction time splits between feature loading and tree evaluation is
  uncertain. Measure it on one tile before deciding whether feature loading (item 3) matters more than
  parallel predict (item 2).
- `random_state` is not set in `rfFitFromPge61.py`, so real trainings are not reproducible today,
  regardless of `n_jobs`.

## Changes needed (not made when written; applied later in the run-root copy, see the status note above)

1. **Training (biggest, one line plus a flag).** In `rfFitFromPge61.py` add a `--nJobs` argument
   (default 1, like `--nEstimators`, so other callers are unchanged) and pass `n_jobs=args.nJobs` to
   `BaggingRegressor`. Optionally add `--randomState`. Expected: about 8x faster; the 1.8 h `pcttree` fit
   would fall to roughly 15-25 min (an estimate from the subset result).
2. **Prediction.** In `VcfPredict.py`, after `joblib.load`, call `rf.set_params(n_jobs=N)`, and predict in
   row chunks to bound memory. Prefer the threading backend
   (`joblib.parallel_backend('threading', n_jobs=N)`), which was fastest and lightest here.
3. **Feature loading.** Parallelize the 60-predictor loop in `_getMetrics` with a thread or process pool.
   Not checked: whether `Metrics.getMetricFromRf` is safe to call from several threads, and reading from
   panfs may limit the gain.
4. **Driver scripts.** The drivers already run 2 fits and 2 tile predictions at once. Share the 12 cores
   (for example `n_jobs=6` each, or run one at a time with 12) so the host is not oversubscribed. The
   drivers need a copy with the flag passed through.

Not needed: BLAS/OpenMP thread variables (trees do not use them), cuML, or container options.

## Open decisions

- Whether to apply items 1-4 in the sync copy (`Python_code/`), and whether to set `random_state`.
- Backend choice for fit (threads or processes were equal here; processes are the sklearn default).
- Whether to run a one-tile timing to settle the feature-loading versus predict split.
