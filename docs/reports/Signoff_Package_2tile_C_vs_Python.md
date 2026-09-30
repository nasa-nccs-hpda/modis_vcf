# Signoff package: two-tile C vs Python MOD44B parity work (h09v05, h12v02, year 2019)

Prepared 2026-09-18. This page summarises what was done, what was verified, what was not, and what
needs a decision. It links to the detailed reports rather than repeating them. Nothing here has been
committed to any repository.

> **Status update 2026-09-20.** This package was written for the two-tile run (2026-09-18). The full 63-tile run has since finished; its results
> are in `FullRun_Results_C_vs_Python_63tiles.md` and item 8 below. Items 3, 6a, 6b and 6c were resolved by that run (each is annotated where it
> appears). Items 1, 2 and the remaining differences in item 8 are still open. The sections between (verification table, recommendation) describe
> the two-tile evidence as it stood on 2026-09-18.

## Purpose and scope of this test

The two-tile run (h09v05, h12v02, 2019) was intended to verify that both pipelines (C Track A, Python
Track B) run end to end, and to give a quick check of the relationship between the C and Python results.
It is not a full verification. **A full test with all 63 tiles is needed for true verification**; nothing
in this package should be read as establishing C-vs-Python agreement or accuracy beyond these two tiles.

## Recommendation

- **Ready to sign off: the tooling and the pipeline.** Both sides now run end to end on both tiles:
  the Python metrics fix (Bands 1-7 and Band_31) is validated against C's `metrics.bin`; the C
  B1 -> B2 -> B3 chain trains 30 models and writes B3 products for both tiles; the Python chain
  trains, predicts and produces a held-out score.
- **Not ready to sign off: product parity.** h09v05 tree cover is comparable (corr 0.90), but the two
  products disagree strongly on h12v02 (unexplained; two-tile result, gone in the 63-tile run, see item 8), and Python lacks post-processing steps that C
  applies. See "Open items". This recommendation is my reading of the evidence; the call is the
  reviewer's.

## What was verified, with evidence

| Item | Result | Where |
|---|---|---|
| Bands 1-7 zenith clamp removed, Band_31 clamp added in Python `BandDayFile.py` | Validated on both tiles against C `metrics.bin`; h12v02 Bands 1-7 to exact parity, h09v05 not regressed | `Bands17_Fix_Validation_Results.md` |
| Track A (C) B2/B3 failure root cause | `generate_b2_pcf.py` pointed at the wrong workspace, and the STEP 4 model check globbed the wrong dir; both fixed | `TrackA_B3_Failure_Root_Cause_and_Fix.md` |
| Track A rerun STEP 3-6 | rc=0, 30 models trained, B3 HDFs written for both tiles (104 MB h09v05, 64 MB h12v02), ~37 min for B3 | `Metrics/FinalCompare_2tile/status.log`, `c_track_a_resume.log` |
| Python Track B retrain and predict | Trained on the two tiles' labeled pixels only (row counts match the `.7.0.0.out` labels exactly); predicted h09v05 and h12v02, rc=0 | `PyTrackB_2tile_Plan_and_Status.md` |
| Python held-out score (20% of rows) | Tree cover corr 0.969, MAE 0.86; non-vegetated corr 0.897, MAE 7.3 | `Metrics/FinalCompare_2tile/heldout_eval.txt` |
| C vs Python on pixels valid in both | See table below | `Metrics/FinalCompare_2tile/compare_c_vs_python.txt`, `FinalCompare_2tile_Plan_and_Status.md` |

C vs Python (Py - C), pixels valid in both:

| Tile / layer | corr | mean Py vs C | MAE |
|---|---|---|---|
| h09v05 tree cover | 0.901 | 13.93 vs 14.78 | 4.82 |
| h09v05 non-vegetated | 0.470 | 35.88 vs 27.75 | 17.11 |
| h09v05 non-tree veg | 0.185 | 50.23 vs 57.47 | 18.84 |
| h12v02 tree cover | 0.065 | 6.97 vs 1.00 | 7.25 |
| h12v02 non-vegetated | 0.540 | 15.91 vs 47.36 | 32.44 |
| h12v02 non-tree veg | 0.259 | 77.13 vs 51.63 | 32.06 |

## Code changes made (all in the `AI_code_sync_09152026` sync copy, per the reports)

I have not diffed these against the owners' repos for this package; confirm before copying anywhere.

- Python `BandDayFile.py`: zenith value clamp removed for Bands 1-7; `solz > 70 -> 16000` added for
  Band_31 (`Bands17_Fix_Validation_Results.md`). Backup of an earlier `Metrics.py` is in `Reports/`.
- Python `rfFitFromPge61.py`: 15 predictor names corrected from `...-Band31` to `...-Band_31`; new
  `--nEstimators` flag (30 for Track B).
- Python `VcfPredict.py`: cuML random forest import and `.to_numpy()` call commented out (not
  deleted) so prediction runs on CPU with sklearn `BaggingRegressor`; new shim
  `Python_code/modis_water/model/Utils.py` supplying `writeRaster`.
- C Track A scripts (`C_code/pge61/build_context/scripts/`): `generate_b2_pcf.py` workspace fixed;
  `run_track_a_2tile.py` gained `--from-step/--to-step`, STEP 4 stale-file cleanup and overwrite
  guards, STEP 5 PCF patching by section; `launch_track_a_2tile.sh` passes arguments through.
- New run scripts and comparison/evaluation scripts are in `Reports/` (`run_final_compare_master.sh`,
  `compare_c_vs_python_2tile.py`, `heldout_eval_py_track_b.py`, and others).
- Further code changes for the 63-tile run were made only in a copy inside the run root (`FullRuns/full63_c_vs_py_20260919-0806/code/`), not in the sync
  copy or any repository: `CAttributes.py` and `CMirrorMetrics.py`, a metric-read cache in `Metrics.py`, threaded chunked predict and `--nJobs` in
  `VcfPredict.py`, `--predictorFile`, `--nJobs` and `--randomState` in `rfFitFromPge61.py`, `BuildTraining` honoring `trainingDir`, a queue-based
  predict view, rebuilt C executables (`C/build`), and a claim-based C driver. See `FullRun_Launch_and_Status.md`.

## Known limits of this evidence

- **No ground truth** in the C-vs-Python comparison: it shows how the products differ, not which is closer to the truth.
- **Different training data:** C models each saw about 18k instances (B1 sample files); Python models
  saw about 1.86M (tree cover) and 0.95M (bare) pooled rows, so some of the gap is training volume, not only M5P vs bagged trees.
  The tree-cover label files also differ (open item 6a): C uses `v5.0.3samp`, Python the larger legacy superset.
- **Two tiles, one year (2019).** Nothing here says how either path behaves elsewhere. The test was a
  pipeline check plus a quick C-vs-Python comparison; a full 63-tile test is needed for true verification.
- **Python tifs have no geotransform** (`VcfPredict._write` never passes `transform`); this is existing behavior.
- The Python held-out score for tree cover is dominated by h09v05: h12v02's tree labels are all 0.
- Some C post-processing was not read: the Greenland block in `MOD44B3.c` from line 411, and the
  `-3` fill rule was only skimmed (`Ensemble_Prediction_C_vs_Python.md`).

## Open items

**Needs the owner's attention (not a modeling question)**
1. **Sep 17 Track A overwrote shared files.** It overwrote `VCF_outputs/weka.2019.NN.model` and
   `weka_training.all.2019.NN.arff` (and any earlier `generated_b2` PCFs from 2026-09-11 in the real
   repo). Neither location is versioned or backed up, so the earlier contents are unrecoverable. The
   models now in `VCF_outputs/` were trained on a many-tile pool (324,033 instances for model 00), not on these two tiles.
   The rerun did not touch them (`TrackA_B3_Failure_Root_Cause_and_Fix.md`).
2. **Stray files in the real repo:** 30 `build_context/test_pcf/generated_b2/MOD_PR44B2.NN.pcf` and 30
   `build_context/runtime/logs/b2_NN/` dirs. Deleting was blocked by the permission classifier; they are untracked.

**Decisions and unexplained results**
3. **h12v02 disagreement is unexplained (two-tile result; not present in the 63-tile run, see item 8).** Python predicts tree cover up to about 44 (95th percentile)
   although every h12v02 tree label is 0; C predicts almost none. Hypotheses only: extrapolation
   beyond the labeled rows, mismatch between training and prediction inputs (see
   `Polar_Night_Zero_Observation_Finding_h12v02.md`), or the training-volume gap. Suggested cheap first
   checks: map where Python's h12v02 tree predictions are above 0, then trace one such pixel.
   **Label provenance (checked 2026-09-19; the accuracy call is the user's, not verified by me):**
   - Two h12v02 tree-cover label files exist; both contain only 0 and 255 (unlabeled), so every labeled pixel is 0:
     - `v5.0.3samp/h12v02.samp.bin` (`/explore/nobackup/projects/ilab/data/MODIS/MODIS_VCF/Mark_training/VCF_training_adjusted/tile_adjustment/v5.0.3samp/`):
       307,939 zero px, file mtime 2010-09-16 12:49:12. **The user (mcarrol2) asserts this file is accurate,
       based on its historical date.** The date is a filesystem mtime; birth time is not recorded on this mount.
     - `Python_code/modis_vcf/data/legacy/h12v02.bin` (the target of the symlink
       `data/training/VCF_training_pcttree_h12v02.7.0.0.out`, and the file `BuildTraining.py` actually trains on):
       769,131 zero px, mtime 2025-07-18 10:31:36, ctime 2026-09-15. The bytes differ from `h12v02.samp.bin`.
       Its mtime matches a bulk copy of `legacy/` (neighbouring files are seconds apart), so the date says
       nothing about when its labels were made, and no source for it is recorded.
   - Consequence (my inference): with the samp labels accepted as accurate, "h12v02 tree cover is 0" is not a
     labeling error, so Python predicting tree cover up to about 44 there points to the model or its
     prediction inputs, not the labels. Python's training file is not the accepted one; its extra 461,192 zero
     pixels (769,131 - 307,939) have no stated provenance, though they are also 0.
   - Python h12v02 tree-cover prediction on the 307,939 samp zero pixels (2026-09-19,
     `Reports/eval_samp_h12v02.py`; alignment checked on h09v05 samp, in-sample corr 0.979 vs 0.046 transposed):
     761 are water-mask pixels (predicted 200, excluded); on the other 307,178 the predicted range is **0 to 19**
     (mean 0.00, median 0, 99.99th percentile 0); 307,153 (99.99%) are exactly 0, 25 are >0, 4 are >=5, 2 are >=10,
     none >=20. RMSE vs 0 is 0.047. Caveat: these pixels are (nearly all) inside the training rows
     (the samp labels are a subset of the `legacy/h12v02.bin` rows Python fits), so this is a fit check, not an
     out-of-sample test. The predictions of about 44 (p95) are on pixels with no label, not on these.
4. **Python post-processing parity** (confirmed in the products, predicted earlier from the code):
   - `Percent_NonTree_Vegetation` is 0 over water where C gives 200 (106,052 px in h09v05; 2,517,812 in h12v02).
   - tree + non-veg is not capped at 100 (100,689 px, 0.44% of valid, in h09v05; 3,242 in h12v02).
   - No SD layer or `-3` fill: on 11,192 h12v02 px C is fill while Python has values.
5. **Rounding:** Python truncates, C rounds half up. Measured on held-out rows, truncation shifts mean
   error from about +0.02 to -0.12 (tree) and +0.14 to -0.34 (bare); rounding half up gives +0.01 / +0.15.
6. Python leaf size (`min_samples_leaf`, C's M5P uses 500) and whether to match how B1 builds each of the 30 samples.
   (`MOD44B1.c` lines 219-242 skips labels 240/255 and keeps 50% of zero-labeled and 40% of non-zero-labeled pixels.)
6a. **C and Python train on different tree-cover label files (verified 2026-09-19).** C's B1 reads
   `v5.0.3samp/<tile>.samp.bin` (the PCF path is a real-repo symlink to the same `Mark_training` files). Python
   reads `data/training/VCF_training_pcttree_<tile>.7.0.0.out`, symlinks to `data/legacy/<tile>.bin`. The samp
   files are a strict subset with identical values (h09v05 268,039 of 1,552,272 labeled px; h12v02 307,939 of
   769,131). The legacy set is more zero-heavy on h09v05 (70.5% vs 48.7% zeros); the extra h12v02 pixels are all
   zero. Bare labels are not comparable: C does not retrain bare (static `BARE/weka.bare.NN.model`) and the samp
   directory has no bare labels. To use the samp labels in Python: repoint the `pcttree` symlinks in
   `Python_code/modis_vcf/data/training/` (rlgill's `v5.0.3samp-conventionallyNamed/` already has the expected
   names; 62 entries), then rerun extract, split, fit and predict for `pcttree` into a NEW output directory
   (the `MasterTraining` glob would otherwise pick up the old parquets), reusing `pctbare.bin`. Estimate about
   1.5 h on forest203 (about 576k rows for the two tiles, so a shorter fit than Sep 18). It will probably not fix
   h12v02 (both label sets are all zero there); it may move h09v05 tree cover up. **Done in the 63-tile run** (Python trained on the samp labels through a separate labels directory of symlinks; `BuildTraining` had
   to be fixed in the run-root copy because it ignored its `trainingDir` argument; bare labels exist for only 20 of the 63 tiles).
6b. **Python fit and predict use one CPU.** Measured on a 100k-row subset: `n_jobs=12` fits about 8x faster with
   identical predictions, and `set_params(n_jobs=12)` works on the saved `pcttree.bin`. Changes needed and limits:
   `Parallel_CPU_Use_Assessment.md`. **Applied in the run-root code copy** for the 63-tile run: threaded, chunked predict (189 s per tile in the test) and a
   12-job fit (tree fit on all 63 tiles took 81 min, bare fit 17 min).
6c. **C and Python use different predictor subsets (found 2026-09-19).** C uses 64 tree / 21 bare of its 324
   metrics; Python hard-codes 60 / 18 names that match C's attribute numbers only under a date-major layout, while
   C and Python's own unsorted tif are band-major (verified on h09v05). Only 8 of 24 tree and 5 of 15 bare unsorted
   predictors coincide. Other families lack same-definition Python metrics. This may explain part of the two-tile
   disagreement (untested) and must be resolved before a full-tile comparison is meaningful. Options and evidence:
   `Predictor_Subset_C_vs_Python.md`. Python does not run the Monte Carlo search (direct fit only).
   NDVI gap measured on all pixels: not exact on 68.0% (h09v05) and 83.2% (h12v02) of pixel-periods; mean absolute
   difference 4.2 and 41.0 (NDVI x 1000), max 1,513 and 1,909.
   **Resolved 2026-09-19:** option C was implemented (Python mirrors of C's derived metrics, implementation B) and used in the 63-tile run; the NDVI
   difference was accepted for now.
8. **Update, 63-tile run (2026-09-19, `FullRun_Results_C_vs_Python_63tiles.md`):** with all 63 tiles, matched predictors and samp labels the
   h12v02 tree-cover disagreement (item 3) is gone (corr 0.917, was 0.065); pooled tree cover corr 0.96 (MAE 4.5), non-vegetated 0.81, non-tree
   vegetation 0.74. Still open from that run: valid-pixel masks differ on h12v01, h13v01, h21v01 and h11v02; the non-vegetated layer is not a
   like-for-like comparison (C static bare models versus Python trained on 20 tiles). The three B3 tiles that an over-strict size rule marked failed,
   although their outputs were valid, were corrected on 2026-09-20 (B3 now 63/63 ok).

**Housekeeping**
9. `PyTrackB_2tile_Plan_and_Status.md` and `TrackA_B3_Failure_Root_Cause_and_Fix.md` carried stale
   "in progress" / "not yet re-run" status lines; they now point here.

## Sign-off checklist (unchecked; for the reviewer)

- [ ] Metrics fix (Bands 1-7, Band_31) accepted
- [ ] Track A script fixes accepted
- [ ] Python CPU predict path (BaggingRegressor, `Utils.py` shim) accepted
- [ ] Owner informed of item 1 (overwritten shared files)
- [ ] Stray-file cleanup (item 2) done or waived
- [ ] Decision on items 3-6c: resolved by the 63-tile run as annotated; reviewer to confirm
- [ ] Full 63-tile test: done (2026-09-19); results reviewed, including the open differences in item 8

Reviewer: ______________________  Date: ____________
