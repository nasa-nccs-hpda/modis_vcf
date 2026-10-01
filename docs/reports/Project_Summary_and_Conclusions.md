# C vs Python MOD44B VCF: what was done and what was concluded (to 2026-09-30)

This is the front page for the whole body of work. It states the goal, the sequence of work, the conclusions with their numbers, what is still open, and what changed in code. It links to the detailed
reports for the evidence rather than repeating them; `README_START_HERE.md` is the index of those reports.

**Status of code (2026-09-30):** the Python changes are committed on local branch `mcarrol2/c-python-parity-fixes` in the delivery clone `$SYNC/Git_delivery/modis_vcf`,
based on `origin/develop` (`b83e302`). Pushed to GitHub by the project lead on 2026-09-30 (`335ceb2`); see `Git_Delivery_Clone_modis_vcf.md`. `$SYNC/Python_code`
matches the branch. **Nothing has been applied to rlgill's checkouts or the pge61 repository.** The one C change (`fix_bands.c`) is in `$SYNC/C_code/pge61` only and is now known to be the wrong aerosol screen (section 10).

Paths used below: `$SYNC` = `/explore/nobackup/people/mcarrol2/MOD44B_VCF_Development/AI_code_sync_09152026`; `$RUN` = `/explore/nobackup/people/mcarrol2/MOD44B_VCF_Development/FullRuns/full63_c_vs_py_20260919-0806`.

## 1. Summary

- **Goal.** Establish whether the Python reimplementation of the MOD44B VCF pipeline (`modis_vcf`) reproduces the operational C pipeline (pge61: MOD_PR44B1, B2, B3), and find and fix the differences. Year 2019, 63 tiles.
  The 2019 product uses the 16-day composites from day of year 065 of 2019 through day 049 of 2020, not the calendar year.
- **Bugs found and fixed.** Six code-level differences were identified. The ones that mattered were fixed: a Python zenith value clamp that wiped out `BandReflMax` on high-latitude tiles, mismatched QA bit reads
  (aerosol and adjacency) and a zenith cutoff in Python, and the aerosol shift in C. After the fix Python's band metrics match C exactly on h09v05; on h12v02 they match apart from an understood residual on `BandReflMin` and `AmpBandRefl` (about 87% exact), and NDVI-related metrics still differ by design (finding 4).
- **Both pipelines now run end to end** on all 63 tiles, and the products were compared. Tree cover: **correlation 0.96, mean absolute difference (MAE) 4.5** between C and Python on 1.11 billion pixels.
  Non-vegetated: correlation 0.81, and that comparison is not like for like.
- **Against the historical MOD44B (Collection 6) tree cover, both track it closely and C is consistently closer**: MAE 3.49 for C and 5.00 for Python, C lower on 60 of 63 tiles.
- **Predictor selection was also examined** (2026-09-20): a stand-alone Monte Carlo over all 324 metrics picks a different top 64 than the one the model uses (20 shared), and a model built from it is slightly worse, on held-out pixels and on maps of three tiles (section 5). Per-tile figures and tables comparing Python with the historical product are in `figures/`.
- **Python is comparable to the historical MOD44B (agreed with the project lead, 2026-10-01).** On the 63 training tiles for 2019, the Python RF product reproduces the historical MOD44B tree cover to a pooled mean absolute difference of 4.2 points with negligible overall bias, within about 0.6 points of the operational C code. Agreement is weakest in dense forest, where Python reads about 3 points lower. "Comparable" means within a few points, not identical; qualifications in section 10 and `C_to_Python_Conversion_Assessment.md`.
- **The conclusion also holds against C6.1 (2026-10-01):** on 54 unaffected tiles, Python RF vs C6.1 MAE 4.17, the C code 3.42; C6 and C6.1 differ by 2.64 (`Historical_C6_C61_Four_Way_Comparison_Results.md`). Python RF is a sound replacement candidate for the C code for tree cover; before calling it operational: an out-of-sample test (non-training tiles, a second year), the SD layer, and the non-vegetated (bare) layer (`C_to_Python_Conversion_Assessment.md`, "Recommendation").
- **Not established:** which product is right (no ground truth), whether the remaining differences matter for use, or agreement in any year other than 2019.
- **Still open** (section 6): valid-pixel masks differ on four high-latitude tiles; non-vegetated is not comparable; Python post-processing gaps; NDVI still differs; a published-granule check of the QA bit
  decision; and repository clean-up that belongs to the repository owner.

## 2. What was compared

| | C (operational) | Python |
|---|---|---|
| Code | pge61: `MOD_PR44B1` (per-tile training samples and metrics), `MOD_PR44B2` (30 models), `MOD_PR44B3` (prediction) | `modis_vcf` and `core` (rlgill's checkout): `makeMetrics`, `buildTraining`, `rfFitFromPge61`, `vcfPredict` |
| Model | 30 M5P models (Weka, bagged) | 30-member scikit-learn bagged regression trees on CPU |
| Predictors | 64 attributes (tree cover), 21 (bare), chosen by index out of 324 per-pixel metrics | 60 and 18 names hard-coded, later replaced by mirrors of C's exact attributes (section 4, phase D) |

## 3. What was found: the six code-level differences

Written up in `C_code/pge61/MOD_PR44B_Metrics/C_vs_Python_Code_Differences.md` (status table at its top).

| # | Difference | Outcome |
|---|---|---|
| 1 | Zenith cutoff: C 65, Python larger (70 in `BandDayFile.py`) | Python changed to 65 (2026-09-15). |
| 2 | Aerosol and adjacency QA bits read from different bits | Decision: adjacency bit 6, aerosol bits 7-8. Python changed (`ProductTypeMod44.py`); C `fix_bands.c` aerosol shift changed `>> 13` to `>> 7` in both `MOD_PR44B1` and `MOD_PR44B3`. **The check against a published MOD44C granule is still open.** |
| 3 | An uninitialized read in C: its zenith value clamp never fires for Bands 1-7 (reads the zenith array before it is filled) | Python's Bands 1-7 clamp removed to match C's real behavior; a clamp added for Band_31, where C's clamp does fire (2026-09-17). C source unchanged. |
| 4 | NDVI zero-division and averaging differ | Unchanged, measured on every pixel and accepted for now. |
| 5 | QA-mask coverage differs | Measured, no change. |
| 6 | Python did not exclude its masking sentinel before reducing | A windowing redesign was tried and reverted; the final handling is in `Bands17_Fix_Validation_Results.md`. |

## 4. The work, in order

**A. Per-tile metric comparisons (2026-09-12 to 09-14).** For all 63 tiles, the per-pixel annual metrics from C (`gen_metrics`, reusing `make_metrics.c`) were compared with Python's
(`makeMetrics.py`) for the seven metrics that can be reconstructed exactly. This produced findings 1 to 6 and one report per tile (`C_code/pge61/MOD_PR44B_Metrics/C_vs_Python_VCF_Metrics_Comparison_<tile>.md`).
Those reports predate every fix and each carries a note saying so. Three tiles (h18v03, h13v01, h16v01) showed anomalies the findings did not explain; they were not re-examined.

**B. Fix and validate on two tiles (2026-09-15 to 09-17).** h09v05 (clean) and h12v02 (a high-latitude tile where `BandReflMax` collapsed to zero valid pixels). Findings 1 and 2 were applied first, and a
windowing redesign for finding 6 was tried and reverted. A single-pixel trace of h12v02 (row 50, column 2264) then showed the true cause: Python's Bands 1-7 zenith value clamp, where C's never fires.
Removing it, and adding the Band_31 clamp, gave (`Bands17_Fix_Validation_Results.md`):
- h09v05: `BandReflMax`, `BandReflMin` and `AmpBandRefl` for Bands 1-7 and Band_31 at bias 0.000, correlation 1.0000, **100% exact**.
- h12v02: `BandReflMax` Bands 1-7 **100% exact** (was zero valid pixels); `BandReflMedian` 99.99 to 100%; `BandReflMin` and `AmpBandRefl` about **87% exact**. The 87% is one understood residual: C accepts a
  no-observation placeholder day (`solz = 0`) as valid, Python correctly rejects it (`Polar_Night_Zero_Observation_Finding_h12v02.md`). Solar zenith at or below zero is a known unreliable-data case in
  this product, not a defect in Python's QA logic.

**C. Two-tile end-to-end test (2026-09-17 to 09-18).** The full chain on both tiles: C (Track A: B1, B2 training of 30 models, B3) and Python (Track B: metrics, training, prediction). A C B2/B3 failure was
traced to `generate_b2_pcf.py` pointing at the wrong workspace and a model check globbing the wrong directory; both fixed (`TrackA_B3_Failure_Root_Cause_and_Fix.md`). Python held out 20% of rows: tree cover
correlation 0.969, MAE 0.86. C against Python: h09v05 tree cover correlation 0.90; **h12v02 correlation 0.065**, an unexplained disagreement (Python predicted tree cover up to about 44 where every label is 0).
The package `Signoff_Package_2tile_C_vs_Python.md` records the evidence and its limits (two tiles, no ground truth, different training data).

**D. The 63-tile run (2026-09-19, launch 12:42, compute ended 19:28, about 7 hours).** Run in a clean run root, with nothing written outside it (checked by a file-change scan of the protected trees).
- *Matched predictors.* Python mirrors of all of C's derived attributes (`CAttributes.py`, `CMirrorMetrics.py`) reproduce C's 252 derived metrics exactly from the 72 unsorted values, so C and Python train on
  the same attribute definitions (`Predictor_Subset_C_vs_Python.md`, `OptionC_ImplementationB_Validation_Results.md`). Python uses `v5.0.3samp` tree labels like C. Bare labels exist for only 20 of the 63 tiles.
- *Speed.* Python originally used one CPU; the measured fix was applied in the run-root code copy (`Parallel_CPU_Use_Assessment.md`).
- *Result.* All 63 tiles produced by both sides. See section 5 and `FullRun_Results_C_vs_Python_63tiles.md`. The two-tile h12v02 disagreement is **gone** (correlation 0.917); which of the changes (C trained on 63
  tiles instead of two, matched predictors, samp labels) removed it was not isolated.
- *One process fault.* Three C B3 tiles (h12v01, h12v05, h13v01) were marked failed by a driver rule that demanded more than 10 MB; they had finished correctly. The rule and markers were corrected on
  2026-09-20 (B3 now 63 of 63 ok). The comparison had used them.

**E. Historical comparison and audit (2026-09-20).** Tree cover from C and Python against the historical MOD44B, Collection 6 (`/css/modis/Collection6/L3/MOD44B-VCF/2019/065`), on the pixels valid in all three
products (`Tree_Cover_vs_Historical_MOD44B_Results.md`). All documents in `Reports/` and `MOD_PR44B_Metrics/` were audited for stale statements and corrected; a scan of the protected trees found no changes.

**F. Figures and predictor selection (2026-09-20, after E).** 63 per-tile figures (Python, historical, difference) with CSV tables in `figures/`. Then feature importance of the existing 64-predictor model (`Feature_Importance_Part_A_Results.md`); a rebuild of the tree-cover training table with all 324 metrics, checked against the metric rasters for all 63 tiles and by fresh regeneration of h09v05 and h12v02; a stand-alone Monte Carlo (702 trials on five nodes, a stop rule, a decision not to continue to `minVarUsage` 15) and two final models; and a check of those models on maps of h12v02, h09v05 and h12v09 with a control that reproduced the existing maps exactly (`Feature_Importance_and_Monte_Carlo_Results.md`, plan in `Feature_Importance_and_Monte_Carlo_Plan.md`). No product code was changed for this; the code written is in the run root's `code/validation/`.

## 5. Results

**C against Python, 63 tiles, pooled** (Python minus C, pixels valid in both, `FullRun_Results_C_vs_Python_63tiles.md`):

| Layer | Correlation | Bias | MAE | RMSE |
|---|---|---|---|---|
| Tree cover (1.11 billion pixels) | **0.960** | +0.37 | 4.53 | 7.04 |
| Non-vegetated | 0.808 | -6.82 | 13.02 | 19.00 |
| Non-tree vegetation | 0.736 | +6.29 | 13.88 | 19.75 |

Per tile, tree cover correlation has median 0.919 and maximum 0.975; eight tiles are below 0.8 (h12v01, h13v01, h16v01, h21v01, h19v11, h20v06, h21v06, h29v11). Non-vegetated differs more because C uses its
static, previously trained bare models while Python retrained on bare labels from 20 tiles; non-tree vegetation is computed from the other two and inherits both. This is a reading, not a tested result.

**Both against the historical MOD44B Collection 6, tree cover, 63 tiles, 1,106,719,917 common-valid pixels:**

| | Correlation | Bias | MAE | RMSE | Within 5 |
|---|---|---|---|---|---|
| C vs historical | 0.972 | -0.12 | **3.49** | 6.02 | 78.6% |
| Python vs historical | 0.953 | +0.25 | **5.00** | 7.77 | 67.0% |
| Python vs C | 0.960 | +0.38 | 4.52 | 7.03 | 70.1% |

- Both track the historical product closely; **C is consistently closer**: about 30% lower MAE, lower on 60 of 63 tiles, and Python's error is larger in every historical-cover bin.
- Both are biased high at low cover and low at high cover; Python's error is larger everywhere by 0.6 to 2.5 MAE points.
- Seven tiles are among the ten worst for both products (h11v08, h19v08, h19v09, h20v09, h21v10, h26v06, h27v06), which points more to the reference or shared inputs than to either implementation. Not investigated.
- The historical product is a Collection 6 reference with different inputs and training data, not ground truth; no significance test was run; Python is a reimplementation of C, so C being closer is not
  evidence that Python is wrong.

**Predictor selection (Monte Carlo) on all 324 metrics, tree cover** (`Feature_Importance_and_Monte_Carlo_Results.md`; done 2026-09-20 after the 63-tile run):
- The stand-alone Monte Carlo (702 trials, every metric used at least 10 times) ranks a **very different top 64 from the 64 the 63-tile model uses: only 20 are shared.**
- On the same 2,288,197 held-out rows the model built from the new top 64 scores R-squared 0.848 and mean absolute error 7.63, against 0.854 and 7.46 for the existing model; a top-20 model scores 0.792. The gap between the two 64-predictor models is small; both scores are optimistic because the held-out rows are a random split of pixels.
- The ranking is not "conclusive" by the rule fixed beforehand (about 25 predictors are nearly tied at the top-64 cutoff), but more trials would not resolve that, so the run was **not** continued to `minVarUsage` 15.
- **Map-level check on three tiles** (h12v02, h09v05, h12v09; against the historical MOD44B on the same 65.3 million pixels): mean absolute error C 3.67, existing Python 4.98, new top 64 5.22, top 20 5.55. The new top-64 map is worse than the existing map on all three tiles, by about a fifth of the Python-to-C gap, and differs from it pixel by pixel by a mean 4.6 points, so the maps change without improving. A control (the existing model through the new prediction path) reproduces the existing maps exactly.
- Not tested: a run-to-run baseline for that 4.6-point pixel difference (same predictors, different random seed), a random-64 null, a split of the held-out data by tile, a redundancy-aware selection, and the other 60 tiles.

## 6. Open items

1. **Resolved 2026-09-27. Valid-pixel masks differ:** C is valid where Python is not on h12v01 (17.96 M pixels), h13v01 (10.32 M), h21v01 (2.86 M) and h11v02 (2.79 M). This is the MOD44W out-of-projection
   area (`water_mask == 250`); Python writes code 250 there on purpose (`Valid_Mask_Disagreement_MOD44W_Root_Cause.md`).
2. **Non-vegetated is not comparable** (static bare models in C, Python retrained on 20 tiles).
3. **Mostly resolved 2026-09-27 (on the branch):** predictions are clamped to 0-100, the three layers sum to 100 (non-vegetated absorbs any overshoot), and water/out-of-projection codes carry to all layers
   (`NonTreeVeg_Water_OutOfProj_Code_Propagation_Plan.md`). Truncation instead of rounding is unchanged.
4. **Python NDVI still differs from C's** (finding 4); accepted for now.
5. **Superseded 2026-09-29/30: finding 2's aerosol decision was wrong.** In MOD44C `state`, bits 7-8 overlap other fields; the screen rejected 29-50% of good pixels. Python now uses `state` bit 15 (PGE72's
   overall QC flag) instead (section 10). The historical MOD44B had no working aerosol screen. The C copy still has the bits 7-8 shift.
6. **Per-tile metric reports (section 4, A) predate all fixes** and were not regenerated; the three anomalous tiles from that batch were not re-examined.
7. **Repository clean-up belongs to the repository owner** (section 8).
8. **Not explained:** a second writer created the 63-tile run root at 08:06:56 on 2026-09-19. Presumed, not proven, to be the previously disconnected Claude session (`FullRun_Launch_and_Status.md`).
9. **Label provenance for h12v02:** two tree-cover label files exist (`v5.0.3samp/h12v02.samp.bin` and the legacy file `modis_vcf` trains on); both contain only zeros and unlabeled pixels. The project lead accepts the samp file as accurate
   from its 2010 modification date; that is a file date, not independent evidence (`Signoff_Package_2tile_C_vs_Python.md`, item 3).
10. **Whether to adopt Monte Carlo predictor selection** in the Python pipeline. The 63-tile run used C's 64 predictors (no search). The search gave a different, slightly worse set here; a run-to-run baseline for the map differences (same predictors, another seed) has not been done. Section 5 has the numbers.
11. **Per-pixel SD layer needed (to do; noted 2026-09-30).** The historical MOD44B has `Percent_Tree_Cover_SD` (and `Percent_NonVegetated_SD`): the SD across its 30 bagged models. The Python pipeline writes no SD layer. RF and the baseline (30 bootstrap trees) can produce the equivalent from per-tree predictions (`model.estimators_`) at almost no cost; it needs a small addition to `VcfPredict`, then a comparison with the historical SD on a few tiles (expect a different scale). XGBoost has no equivalent without bagging about 30 models. Because the project lead considers RF and XGBoost equivalent in accuracy, this favours RF. Not planned yet.

## 7. What changed in code, and where it lives

State on 2026-09-30. Python rows marked "branch" are committed on `mcarrol2/c-python-parity-fixes` (not yet pushed):

| Change | Where | Kind |
|---|---|---|
| C `fix_bands.c` aerosol shift `>> 13` to `>> 7` (`MOD_PR44B1`, `MOD_PR44B3`) | `$SYNC/C_code/pge61` | Bug fix (finding 2). The only C source change. |
| Python `BandDayFile.py` (cutoff 65, Bands 1-7 clamp removed, Band_31 clamp), `ProductTypeMod44.py` (adjacency bit 6; QA `state` bit 15 instead of an aerosol test), `BuildTraining.py` (`trainingDir` was ignored) | branch | Bug fixes |
| Python `Metrics.py` read cache, `vcfPredict.py`, new `vcfPredictQueue.py` | branch | Performance |
| New `CAttributes.py`, `CMirrorMetrics.py`, `CEquivalentMetrics.py` | branch | Feature: mirrors of C's derived metrics |
| `VcfPredict.py`: clamp to 0-100, layers sum to 100, codes 200/250 on all layers | branch | Bug fix (2026-09-27) |
| New `view/fitAltModel.py` (Random Forest, XGBoost, HistGradientBoosting fits) | branch | Feature: reproduces the RF and XGBoost tunedB models |
| cuML random forest replaced by scikit-learn `BaggingRegressor` in `VcfPredict.py`, `MonteCarloSim.py`, its test, `monteCarloSim.py`, `rfFitFromPge61.py` | branch (separate commit) | **Design change**, made because forest203 has no GPU; not a bug fix |
| A one-method `modis_water` shim | `$SYNC/Python_code` | Workspace only, not for delivery |
| Predictor-selection scripts: 324-metric wrapper (`p0_makemetrics_all.py`), table builders, `run_monte_carlo_standalone.py`, `analyze_mc_stability.py`, `predict_tree_cover_model.py`, comparison and figure scripts | `$RUN/code/validation` | Run harness; no edit to `modis_vcf`. The wrapper only widens which C-mirror metrics are computed. |
| C run driver, PCF generators, Track A scripts | `$SYNC/C_code/pge61/build_context/scripts` and `$RUN/code/c_scripts` | Run harness |

`$SYNC/Python_code` was brought up to the branch on 2026-09-30 (replaced files in `Python_code/_backup_before_sync_20260930/`). Run roots keep the code each run used.

## 8. For discussion with rlgill

These need the repository owner's view; none has been acted on.
1. **Which of the changes to take.** The bug fixes (section 7) are the natural minimum. The `BaggingRegressor` swap changes the model type and is a design decision. The C-mirror files and the read cache are optional.
2. **The aerosol shift in C** (`>> 13` to `>> 7`) is in the workspace copy only and implements the bits 7-8 screen that was later found wrong (section 10). Do not apply it; whether C should also move to bit 15 is undecided.
3. **A Sep 17 Track A run overwrote shared model files** (`VCF_outputs/weka.2019.NN.model` and `weka_training.all.2019.NN.arff`) that are neither versioned nor backed up; the earlier contents are unrecoverable
   (`TrackA_B3_Failure_Root_Cause_and_Fix.md`, `Signoff_Package_2tile_C_vs_Python.md` item 1).
4. **Untracked files in the pge61 repository directory** from that run: 30 files at `/explore/nobackup/projects/ilab/projects/MODIS-VCF/pge61_08312026/pge61/build_context/test_pcf/generated_b2/MOD_PR44B2.NN.pcf`
   and 30 directories at `.../pge61/build_context/runtime/logs/b2_NN/` (NN 00 to 29). The pge61 working tree also has 72 files modified against its last commit and 34 untracked entries that this work did not create as far as can be told.
5. **Whether the valid-pixel mask difference is a Python or a C question**, and the Python post-processing gaps (open items 1 and 3).
6. **Six defects in the shipped `MonteCarloSim`** found while running it stand-alone (none changed; the stand-alone driver avoids them): `view/monteCarloSim.py` calls `saveFinalModel(finalModel)` though the method takes no argument, so the final model is never saved; `topnMet` is read before assignment when the top-N check fails; `numCompleted` is reset every batch, so the `maxTrials` guard never stops the loop; permutation importance is computed on all rows including the training rows; the non-default branch of `_initXY()` uses a member that does not exist; and `tests/test_MonteCarloSim.py` expects an older interface. Details and line numbers: `Feature_Importance_and_Monte_Carlo_Plan.md`, "Defects in the Monte Carlo code".

## 9. Where the evidence is

| Question | Document |
|---|---|
| Index of everything, and current state | `README_START_HERE.md` |
| The code-level differences and their status | `C_code/pge61/MOD_PR44B_Metrics/C_vs_Python_Code_Differences.md` |
| The metrics fix chain | `Reduced_Scope_Validation_Plan.md`, `Phase5_6_Before_After_Summary.md`, `Zenith_Value_Clamp_Root_Cause_Full_Picture.md`, `Bands17_Fix_Validation_Results.md` |
| The two-tile test | `Signoff_Package_2tile_C_vs_Python.md`, `FinalCompare_2tile_Plan_and_Status.md`, `PyTrackB_2tile_Plan_and_Status.md`, `TrackA_B3_Failure_Root_Cause_and_Fix.md` |
| The 63-tile run | `FullRun_C_vs_Python_Plan.md`, `FullRun_Launch_and_Status.md`, `FullRun_Results_C_vs_Python_63tiles.md`, `Predictor_Subset_C_vs_Python.md`, `OptionC_ImplementationB_Validation_Results.md`, `Parallel_CPU_Use_Assessment.md` |
| Predictor selection (feature importance and Monte Carlo) | `Feature_Importance_Part_A_Results.md`, `Feature_Importance_and_Monte_Carlo_Plan.md`, `Feature_Importance_and_Monte_Carlo_Results.md` |
| Comparison with the historical MOD44B | `Tree_Cover_vs_Historical_MOD44B_Plan.md`, `Tree_Cover_vs_Historical_MOD44B_Results.md` |
| Delivery to the main repositories | `Git_Delivery_Clone_modis_vcf.md`, `Sync_Back_and_Cleanup_Plan.md` |
| RF and XGBoost models; model vs historical | `Regression_Tree_Methods_RF_XGBoost_Results.md`, `Model_vs_Historical_DeepDive_20260929.md` |
| h16v01 over-prediction and the zenith-only fix | `H16v01_TreeCover_OverPrediction_Investigation.md`, `Zenith_Only_Fix_63Tile_Results.md`, `Greenland_Iceland_Scoped_Fix_Plan.md` |
| Aerosol QA (bit 15) and the B/C decision | `Aerosol_QA_Variant_Test_Results.md`, `Aerosol_QA_C_Full63_Retrain_Results.md`, `Lidar_Reference_Coincidence_20260930.md`, `Stable_Pixel_Reference_Selection.md` |
| Cloudy-region drops (h12v09) | `C_vs_Historical_h12v09_20260930.md` |
| Chronological working log up to 2026-09-16 | `execution_log.md` |

## 10. Work from 2026-09-27 to 2026-09-30 (Python model quality)

From 2026-09-27 the goal shifted from matching C to making the Python product as good as possible.
- **Regression tree methods.** Random Forest and XGBoost ("tunedB": 2000 rounds, depth 6, learning rate 0.02) against the baseline bagged trees, same predictors and training data.
  Against the historical MOD44B on 63 tiles, pooled MAE: baseline 5.06, RF 4.48, XGBoost 4.02 (C 3.54). XGBoost is the best Python model against the historical product on 59 of 63 tiles, but it is the worst against held-out training labels (see the correction below)
  (`Regression_Tree_Methods_RF_XGBoost_Results.md`, `Model_vs_Historical_DeepDive_20260929.md`). The non-vegetated layer is still the baseline model.
- **h16v01 (Greenland) over-prediction.** Most of its compositing periods have no pixel passing the 65-degree zenith cutoff. A zenith-only NDVI-default fix
  cures h16v01 but hurts at least 10 other tiles, so it is not adopted; the direction is to scope it to Greenland (and possibly Iceland) tiles, which is blocked on MOD44C data.
- **Model choice is not settled (corrected 2026-09-30).** XGBoost tunedB is the best Python model only against references built from the historical MOD44B: best C (bit-15) model on 56 of 63 tiles against the historical map (pooled MAE baseline 4.87 / RF 4.20 / XGBoost 3.61), and best on the h11v02 stable historical pixels (2.74 / 2.55 / 2.36) and on the 3-tile stable set (5.72 / 5.13 / 4.81; gaps are ties; `Stable_Set_Model_Eval_Results.md`). Against the held-out 20% training labels it is **last on every tile** (best: baseline 48, RF 15, XGBoost 0; pooled MAE 7.25 / 7.32 / 8.38). G-LiHT lidar on h11v02 does not separate the models. The references disagree, so by the relative-difference rule no model is chosen yet. The historical product had no working aerosol screen, so references built from it favour whatever resembles it.
- **NEON lidar (2026-09-30):** on 50,561 pixels at 19 sites, RF (bit 15) ties the historical MOD44B (3 m MAE 20.39 vs 20.43). All products and the training labels read far below lidar in dense forest, which reflects the cover definition, not the model (`NEON_Lidar_Eval_Results.md`).
- **Model choice and SD layer (2026-09-30).** The project lead treats RF and XGBoost as equivalent in accuracy. RF is favoured because it can produce the per-pixel SD layer the historical MOD44B has; that layer still has to be built (open item 11).
- **Aerosol QA.** The 2026-09-15 aerosol screen (MOD44C `state` bits 7-8) was shown to reject 29-50% of good pixels (bit overlap; investigation in
  `/explore/nobackup/people/mcarrol2/MOD44C_PGE72_Investigation/`). A 63-tile rebuild and retrain with `state` bit 15 (C) against the bits 7-8 screen (B):
  **B and C are equivalent (within about 1 point MAE on every reference); a tie favours C, which is now on the branch.** Watch h16v01 (C baseline +3.8 MAE vs historical).
- **Independent references.** A 10,000-pixel stable-pixel reference set and G-LiHT lidar on h11v02. Every product compresses the range against lidar; the labels match lidar at a 3 m height cutoff.
- **Cloudy regions.** On h12v09 the sharp dense-forest drops are persistently cloudy pixels whose composites fall back to all-day means; every Python model has them. This is the next area for model improvement (not started).
- **Comparability with the historical MOD44B (2026-10-01).** Using the results in hand before the h12v09 cloudy-pixel work, the project lead and I agreed the Python product is comparable (within a few points) to the historical MOD44B on the 63 training tiles. Bit-15 retrain vs historical C6, about 1.07 billion common-valid pixels (`Aerosol_QA_C_Full63_Retrain_Tables.md`, Stage 5b):

  | Product | Pooled MAE | Pooled bias | Per-tile median MAE | Tiles over 5 | Tiles over 7 | Worst tile |
  |---|---|---|---|---|---|---|
  | Operational C (rebuilt) | 3.56 | -0.13 | 3.52 | 12 | 2 | 8.0 (h19v08) |
  | Python RF | 4.20 | +0.28 | 4.39 | 23 | 2 | 7.5 (h19v08) |
  | Python XGBoost tunedB | 3.61 | -0.08 | 3.64 | 10 | 2 | 7.6 (h19v08) |
  | Python baseline | 4.87 | +0.34 | 4.90 | 31 | 8 | 10.2 (h16v01) |

  Qualifications: (1) the claim is about average error; about 10% of pixels differ by more than 10 points and RMSE is about 6.4; (2) bit-15 RF reads low in dense forest (-3.0 at 71-80% and 81-90% cover, -2.8 above 60%) and +1.1 to +1.7 high at 1-30% cover, as does the C code (-2.8 above 60%); figures corrected 2026-10-01 from the older bits 7-8 RF values (-3.8 / -4.2, 12%, RMSE 7); (3) this is agreement with a reference that had a broken aerosol screen, not accuracy, and NEON lidar (RF ties historical, 20.39 vs 20.43) is the only independent support; (4) 2019 and the 63 training tiles only; (5) it holds for RF or XGBoost, not the baseline (h16v01 10.2); (6) h12v09-type cloudy drops are inside these numbers (h12v09 RF 5.5) but smoothed by tile averages. Full write-up: `C_to_Python_Conversion_Assessment.md`.
- **Four-way comparison with C6.1 (2026-10-01).** C6, C6.1, the C code and Python RF (bit 15) on 63 tiles; 9 tiles flagged for the C6.1 high-latitude rule (tree cover and non-tree vegetation set to 0, non-vegetated to 100: the 5 v02 tiles and the 4 v01 tiles) and kept out of the pooled numbers. Results by 4 cover bins and by continent and latitude band (location only; land cover was dropped). `Historical_C6_C61_Four_Way_Comparison_Results.md`.
- **Compute.** Nodes ngaproc201 and ngaproc202 (24 CPUs, 225 GB each) became available on 2026-09-30; singularity is loaded by `module load singularity/4.3.1` there.
