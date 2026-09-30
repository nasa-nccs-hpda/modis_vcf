# Start here: MODIS-VCF development workspace

Last updated 2026-09-30 ~13:30. Read this section, then `Project_Summary_and_Conclusions.md`. The dated status entries are kept word for word under "Status history" at the end.

## Current state (2026-09-30)

- **Nothing is running.** All run markers are `.done`. The last run, the 63-tile bit-15 rebuild and retrain, finished at 10:41 with no failures.
- **Aerosol QA decision:** B (`state` bits 7-8) and C (`state` bit 15) are equivalent in accuracy, and a tie favours **C**. C is now in the code. Before production, look at h16v01 (snow): C's baseline model is +3.8 MAE there against historical.
- **Best Python model:** XGBoost tunedB, best on 59 of 63 tiles against the historical MOD44B. The non-vegetated layer is still the baseline model.
- **Code:** the Python changes are **committed on local branch `mcarrol2/c-python-parity-fixes`** (6 commits off `origin/develop` `b83e302`) in `Git_delivery/modis_vcf`. **The project lead pushes it**, because the cluster has no GitHub credentials (`Git_Delivery_Clone_modis_vcf.md`). `Python_code/modis_vcf` matches the branch.
- **Compute:** forest201-205 as before, plus **ngaproc201 and ngaproc202** (24 CPUs, 225 GB each, from 2026-09-30). On those two, ssh needs `-o CheckHostIP=no` (a stale IP entry in `/etc/ssh/ssh_known_hosts`), and singularity needs `module load singularity/4.3.1`. The existing launchers call `singularity` directly and need that line before they will run there.

## Waiting on the project lead

1. Push the branch and open a pull request for rlgill (`Git_Delivery_Clone_modis_vcf.md`, "To push").
2. NEON lidar steps 2-5 (optional, about 1-2.5 h; `NEON_Lidar_Eval_Plan.md`).
3. Plan the cloudy-region work: cloud-fallback composites (`C_vs_Historical_h12v09_20260930.md`, last section).
4. Greenland/Iceland scoped fix: blocked on MOD44C files (`Greenland_Iceland_Scoped_Fix_Plan.md`).
5. The non-vegetated layer has not been tried with RF or XGBoost.

## Standing conditions

- Nothing in rlgill's checkouts (pge61, pge72, `modis_vcf`, `core`) is edited, committed or pushed. The branch lives in our own clone and is pushed by the project lead.
- Plans are written and approved before anything runs. Long runs are fully detached (`setsid nohup`, status and done markers). Model choices rest on relative differences, and a difference of about 1 point MAE or less counts as a tie.
- New work goes into new folders; nothing already written is overwritten.

## Document map

**Start with `Project_Summary_and_Conclusions.md`**: one page on what was done, the results, the open items, what changed in code and what to discuss with the repository owner.

**63-tile run (2026-09-19)**
- `FullRun_Results_C_vs_Python_63tiles.md`: comparison results.
- `FullRun_Launch_and_Status.md`: launch record, decisions, pilots, canary, corrections, final status.
- `FullRun_C_vs_Python_Plan.md`: the plan, kept with its estimates; the top status block says what became of each part.
- `Predictor_Subset_C_vs_Python.md`, `OptionC_ImplementationB_Validation_Results.md`: why C and Python used different predictors,
  the option C implementation (Python mirrors of C's derived metrics, implementation B) and its validation.
- `Parallel_CPU_Use_Assessment.md`: why Python used one CPU and the measured fix (applied in the run-root code copy).
- `figures/`: one figure per tile (63) comparing the Python tree cover with the historical MOD44B and their difference, with `tile_stats.csv` and a README (2026-09-20).
- `Feature_Importance_Part_A_Results.md`: importance of the 64 predictors in the existing tree-cover model, from impurity and from full-sample permutation on all held-out rows (done 2026-09-20).
- `Feature_Importance_and_Monte_Carlo_Results.md`: the stand-alone Monte Carlo on all 324 metrics (702 trials): ranking, top 64 against the existing model's 64 (20 shared), final-model scores on the same held-out rows (top 64 R-squared 0.848 against 0.854 for the existing model; top 20 0.792), stability analysis, the decision not to continue to `minVarUsage` 15, and a map-level check of the new models on three tiles against the historical MOD44B (existing model MAE 4.98, top 64 5.22, top 20 5.55, C 3.67; the maps differ pixel by pixel by a mean 4.6 points; control exact) (done 2026-09-20).
- `Feature_Importance_and_Monte_Carlo_Plan.md` (all steps done 2026-09-20; results in `Feature_Importance_and_Monte_Carlo_Results.md`): the plan - feature importance from the existing 63-tile tree model, and a stand-alone Monte Carlo predictor selection on the full 324-metric set (includes a rebuild of the training table, the defects found in the shipped code, and a split across compute nodes).
- `Sync_Back_and_Cleanup_Plan.md`: **plan only, not executed** - stale-reference scan of the documents, what differs from the main repositories, cleanup candidates with sizes, and the phased sync-back with its conditions.
- `Git_Delivery_Clone_modis_vcf.md` (2026-09-21): a fresh sparse clone of upstream `nasa-nccs-hpda/modis_vcf` at `AI_code_sync_09152026/Git_delivery/modis_vcf/`, with the 14 Python files from the run-root code copy on local branch `mcarrol2/c-python-parity-fixes` (base `origin/develop` `b83e302`). **Nothing committed or pushed**; the user's SSH key has no GitHub access yet, so the commit waits for access from rlgill. Says what was left out and what to do before committing.
- `Tree_Cover_vs_Historical_MOD44B_Results.md` (and its plan, `Tree_Cover_vs_Historical_MOD44B_Plan.md`): C and Python tree cover against the historical MOD44B (Collection 6, `/css/modis`), 63 tiles, run 2026-09-20. C is closer (MAE 3.49 against 5.00 on the common-valid pixels).

**Regression tree methods experiment (2026-09-27)** - a separate run root, `FullRuns/regtree_alt_models_20260927/`; the 63-tile run above is untouched.
- `Regression_Tree_Methods_RF_XGBoost_Results.md` (and its plan, `Regression_Tree_Methods_RF_XGBoost_Plan.md`): tried Random Forest and XGBoost against the baseline
  Python model (`BaggingRegressor(DecisionTreeRegressor)`), same 64 predictors, same training data. Both beat the baseline at full 63-tile scale against the historical
  MOD44B (corr 0.9627 Random Forest, 0.9685 XGBoost tunedB, against the baseline's 0.9522 and C's 0.9714), and fit far faster (10-30 minutes against the baseline's 81).
- `Valid_Mask_Disagreement_MOD44W_Root_Cause.md`: a side investigation from looking at this experiment's high-latitude tiles - the 63-tile run's open item 2 (C has
  values where Python has none on h12v01, h13v01, h21v01, h11v02) is a real MODIS-sinusoidal-projection out-of-bounds area (MOD44W's fill code), not a defect; also found
  the historical MOD44B product disguises the same area as the ordinary-looking value 18 in `Percent_Tree_Cover`, which contaminates the original
  `Tree_Cover_vs_Historical_MOD44B_Results.md`'s C-vs-historical pairwise numbers on those four tiles (flagged there, not corrected). `figures/c_vs_python_mask_disagreement/`
  has one full-tile image per affected tile.
- `NonTreeVeg_Water_OutOfProj_Code_Propagation_Plan.md`: **implemented and validated** (not just planned) - `Percent_NonVegetated` is now reduced, never tree cover, to
  keep all three output layers summing to exactly 100 on ordinary pixels, and raw predictions are clamped to 0-100 before the water/out-of-projection codes are applied
  (XGBoost genuinely produced out-of-range values in real output - RF and the baseline never do). Staged in `Git_delivery/modis_vcf/model/VcfPredict.py`, uncommitted.
- `H16v01_TreeCover_OverPrediction_Investigation.md`: root cause (5 of 8 compositing periods 100%
  unfiltered tile-wide at 69-80 degrees N, a structural zenith-cutoff effect) confirmed 2026-09-27. A
  first candidate fix (NDVI-default on any QA failure) was disqualified 2026-09-28 (broke h12v09). A
  narrower **zenith-only variant passed 2-tile validation the same day** (h16v01 mean 5.66->1.47,
  h12v09 provably untouched) - see the report's "Zenith-only NDVI-default variant" section. **Checked
  across 31 of 63 tiles and all three models** (baseline/RF/XGBoost tunedB) in
  `Zenith_Only_Fix_63Tile_Plan.md`; results in `Zenith_Only_Fix_63Tile_Results.md` (2026-09-28): fixes
  h16v01 but degrades label accuracy on at least 10 other tiles, consistently across models - **not a
  safe general fix as designed**. Still a diagnostic, not adopted; no code changed.
  A 70-degree cutoff retest on h16v01 and h20v03 (`Zenith_Only_Fix_Cutoff70_Retest_Plan.md` and
  `_Results.md`, 21:09) reduced but did not remove the damage. **Direction from the project lead: scope
  the fix to Greenland tiles, and consider Iceland** (note at the top of this README; not yet designed).

**Two-tile test (2026-09-18, historical)**
- `FinalCompare_2tile_Plan_and_Status.md`, `PyTrackB_2tile_Plan_and_Status.md`, `Ensemble_Prediction_C_vs_Python.md`,
  `TrackA_B3_Failure_Root_Cause_and_Fix.md`.

**Metrics-fix investigation (2026-09-15 to 09-17)**
- Final state: `Bands17_Fix_Validation_Results.md` (Bands 1-7 zenith value clamp removed, Band_31 clamp added, validated).
- History, in order: `Reduced_Scope_Validation_Plan.md` and `Phase5_6_Before_After_Summary.md` (Findings 1-2 fix),
  `Finding6_Fix_Plan_h12v02.md` (executed then reverted), `Single_Pixel_Trace_*` and `BandReflMax_*` (root-cause trace),
  `Zenith_Value_Clamp_Root_Cause_Full_Picture.md`, `Polar_Night_Zero_Observation_Finding_h12v02.md`, `execution_log.md`.

**Code-level comparison of the two codebases**
- `C_code/pge61/MOD_PR44B_Metrics/`: `C_vs_Python_Code_Differences.md` (six findings, with a 2026-09-20 status table),
  `HOWTO_C_vs_Python_Comparison.md`, `Metrics_Array_Reconstruction_Deep_Dive.md`, and 63 per-tile metric comparisons written
  2026-09-13/14 (before the fixes above; each carries a note saying so).

**Model quality work (2026-09-28 to 09-30)**
- `Model_vs_Historical_DeepDive_20260929.md`: baseline, RF and XGBoost tunedB against the historical MOD44B on all 63 tiles.
- `Greenland_Iceland_Scoped_Fix_Plan.md`: scoped zenith-only fix for the Greenland tiles, and possibly Iceland; plan only, blocked on MOD44C data.
- `MOD44C_State_Aerosol_Bit_Collision.md`: stub. The report moved to `/explore/nobackup/people/mcarrol2/MOD44C_PGE72_Investigation/`.
- `Aerosol_QA_Variant_Test_Plan.md`, `_Results.md`, `Aerosol_QA_Variant_Stage2_Tables.md`: bit 15 (C) against the bits 7-8 screen (B) on three tiles.
- `Aerosol_QA_C_Full63_Retrain_Plan.md`, `_Results.md`, `_Tables.md`: 63-tile C rebuild and B/C retrain; **decision: equivalent, and a tie favours C**.
- `Stable_Pixel_Reference_Selection.md`: the 10,000-pixel stable reference set, which stands in for Stage 6.
- `Lidar_Reference_Coincidence_20260930.md`: G-LiHT on h11v02, the decision section, and the lidar coverage survey.
- `NEON_Lidar_Eval_Plan.md`: NEON evaluation, step 1 done, steps 2-5 optional.
- `C_vs_Historical_h12v09_20260930.md`: dense-forest check, and cloud-fallback drops in the eastern Amazon.
- `Git_Delivery_Clone_modis_vcf.md`: the local branch with the Python changes and how to push it.

## Open items
1. **Resolved 2026-09-20: three B3 tiles (h12v01, h12v05, h13v01) had been marked `failed`** in the run's markers although all three finished with rc=0 and
   valid HDFs. The driver's success rule required more than 10 MB, which is wrong for mostly-ocean tiles. The rule in `$RUN/code/c_scripts/run_c_pipeline.py` now requires
   more than 100 KB and the markers were reclassified to `ok` (originals in `$RUN/C/production/build_context/markers/_reclassified_20260920/`); B3 reads 63/63 ok.
2. **Resolved 2026-09-27: valid-pixel masks differ on some high-latitude tiles** (C has values where Python has none on h12v01, h13v01,
   h21v01, h11v02). The disagreement matches MOD44W's fill/unclassified code (`water_mask == 250`) exactly, pixel for pixel, on all four
   tiles - confirmed to be a genuine out-of-bounds area of the MODIS sinusoidal projection at high latitude, not a data-quality defect.
   **Corrected the same day:** Python is not producing "no value" there - `VcfPredict` already writes an explicit code 250
   ("out of projection") to `Percent_Tree_Cover` and `Percent_NonVegetated` for exactly these pixels, deliberately and correctly
   (`MASK_OUT_OF_PROJ`/`PRED_OUT_OF_PROJ` in `VcfPredict.py`, unit-tested). An earlier pass of the investigation read only part of that
   file and said otherwise; see `Valid_Mask_Disagreement_MOD44W_Root_Cause.md` for the correction. The C-side mechanism was not pursued,
   since the goal is Python's own quality, not matching C.
3. **Non-vegetated is not a like-for-like comparison** (C: static bare models; Python: retrained on bare labels for 20 of the 63 tiles).
4. **Resolved in staged code 2026-09-27** (not yet regenerated into the existing 63-tile prediction files - see below): Python's
   post-processing had two related gaps, found/designed/fixed/validated the same day while re-checking item 2. `Percent_NonTree_Vegetation`
   was `max(0, 100 - Percent_Tree_Cover - Percent_NonVegetated)` with no masking step, so a real land pixel where tree cover and non-veg
   overshot 100 together silently showed 0 there while the other two layers were left overshot on disk, and a masked (water/out-of-projection)
   pixel showed 0 instead of the 200/250 code; predictions were also never clamped to 0-100 before writing (harmless for the
   baseline/Random Forest, which are mathematically bounded to the training range, but XGBoost tunedB genuinely produced out-of-range
   values - min -22, 1.89% of all pixels). Both fixed in `Git_delivery/modis_vcf/model/VcfPredict.py` (staged, uncommitted): predictions
   are clamped to `[0,100]` before the water/out-of-projection codes are applied, and `Percent_NonVegetated` (never tree cover) is reduced
   to absorb any remaining overshoot so all three layers always sum to exactly 100 on an ordinary pixel. Validated against the real
   63-tile Random Forest and XGBoost tunedB output (0.63%/0.47% of pixels needed the overshoot fix, 0%/1.89% needed the clamp).
   Comparison's raw/harmonized-views decision in `Ensemble_Prediction_C_vs_Python.md`; full detail and validation numbers:
   `NonTreeVeg_Water_OutOfProj_Code_Propagation_Plan.md`.
5. **Python NDVI still differs from C's** (accepted for now; `OptionC_ImplementationB_Validation_Results.md`).
6. **Real repo clean-up not done:** 30 stray `build_context/test_pcf/generated_b2/MOD_PR44B2.NN.pcf` and 30 `runtime/logs/b2_NN/` dirs
   from Track A's Sep 17 run (untracked); and Sep 17 overwrote shared `VCF_outputs` models (unrecoverable). See the Signoff package.
7. **Presumed resolved 2026-09-20 (user's instruction): the second writer that created the run root at 08:06:56 on 2026-09-19 is presumed to be the previously disconnected
   Claude session.** This is a presumption, not a proof: only session `8cc8fcae` was active at that time, but nothing records who created the directory
   (see `FullRun_Launch_and_Status.md`). The "other writer" of `CEquivalentMetrics.py` in the predictor documents may be the same session; that was not checked.
8. **Predictor selection (2026-09-20):** whether to adopt Monte Carlo selection in the Python pipeline is undecided. The stand-alone search over 324 metrics gave a different top 64 (20 shared with C's), and a model from it was slightly worse on held-out pixels and on three tiles' maps. A run-to-run baseline for the map differences was not done (see "Resume here"). Results: `Feature_Importance_and_Monte_Carlo_Results.md`.
9. **Six defects in the shipped `MonteCarloSim`** (unchanged in `MAIN_PY` and here; the stand-alone driver avoids them): for the repository owner. List: `Feature_Importance_and_Monte_Carlo_Plan.md`, "Defects in the Monte Carlo code", and `Project_Summary_and_Conclusions.md`, section 8.
10. **Updated 2026-09-30: Python committed locally, push pending (project lead).** Previously: **Sync to the main repositories is on hold** until the project lead has spoken to the repository owner (the Python side is staged, uncommitted, in `Git_delivery/modis_vcf`; see `Git_Delivery_Clone_modis_vcf.md`; the C change in pge61 is not staged) (`Sync_Back_and_Cleanup_Plan.md`; six open decisions).
11. **Superseded 2026-09-27 (same day, later): already implemented.** This item originally asked for an explicit 250 ("out of projection") code in Python's tree-cover output; checking the full code
    found `VcfPredict._maskPrediction` already does exactly that (and the same for `Percent_NonVegetated`), correctly, unit-tested. Rolled into item 4, which covers the one part that is genuinely not
    implemented (`Percent_NonTree_Vegetation` not propagating the water/out-of-projection codes). Plan: `NonTreeVeg_Water_OutOfProj_Code_Propagation_Plan.md`.
12. **Cloud-fallback composites** (2026-09-30): persistently cloudy pixels fall back to all-day means with clouds, and every Python model drops there. Improving this is the next model-quality area; not planned yet.
13. **The C workspace copy still has the bits 7-8 aerosol shift** (`C_code/pge61/.../fix_bands.c`). Do not deliver it. Whether C should use bit 15 is undecided.
14. **The qaC and aerosol-variant runs used code without the 2026-09-27 `VcfPredict` output fix** (no clamp; layers not forced to sum to 100). B and C are affected alike; the branch has the fix.

## Documentation audit (2026-09-20)

Every document in `Reports/` and `C_code/pge61/MOD_PR44B_Metrics/` was checked against the final state. The audit was interrupted by a disconnect and finished in a second session.
- Corrected in place: the run timings (C compute ended 19:28, not 19:16), the plan and sign-off status wording, the launch record, and the option C and parallelization documents.
- Given status notes pointing to the final state: `Phase5_6_Before_After_Summary.md`, `BandReflMax_Bug_Investigation_Plan.md`, `Polar_Night_Zero_Observation_Finding_h12v02.md`, and all 63 per-tile metric comparisons. `Finding6_Fix_Plan_h12v02.md`, `Bands17_Fix_Validation_Results.md`, `Zenith_Value_Clamp_Root_Cause_Full_Picture.md`, `execution_log.md` and the two how-to and deep-dive docs already carried correct status headers.
- Verified by find, after the fact: no file in `C_code`, `Python_code`, `Metrics`, `VCF_outputs` or the shared `pge61` repo changed between the production start marker (2026-09-19 12:42:45) and `compare_only.done` (19:56:00). Zero changes in each.
- Checked that the Python code copy used in the 63-tile run matches the fixed `Python_code/` copy (Bands 1-7 clamp removed, Band_31 clamp kept).
- Later the same day the first and last open items above were closed (item 1 fixed, item 7 presumed resolved).
- `C_code/pge61/MOD_PR44B_Metrics/HOWTO_C_vs_Python_Comparison.md` was revised to the current layout: it now binds the fixed `Python_code` copy (rlgill's checkout still has the unfixed clamps), uses the `$SYNC/Metrics/<tile>/` paths that `gen_metrics.c` and the compare scripts actually use, gives a test-compiled build command, and no longer attributes the BandReflMax collapse to Finding 6 alone. It still does not describe a comparison against the archived MOD44B-VCF product in `/css/modis`; no document does.
- Second pass 2026-09-20 (see `Sync_Back_and_Cleanup_Plan.md`, Part 1): status notes added to the two `Sync_Plan_*` documents in `C_code/pge61`, the stale "pending the user's choice" line in `Predictor_Subset_C_vs_Python.md` resolved, and three references to scripts that no longer exist marked. The cleanup and sync-back phases of that plan are **not** executed.
- A search for the superseded phrases (old timings, "not applied", "not executed") finds none in the current-state documents.

## Suggested wording for a new session
"Read Reports/README_START_HERE.md (including its 'Resume here' section) and Reports/Project_Summary_and_Conclusions.md, and tell me the open items and what is not done. Do not execute anything without asking."

## Status history (newest first; kept word for word)

**2026-09-30 10:41: bit-15 (C) 63-tile run FINISHED, no failures. Nothing is running.** `Aerosol_QA_C_Full63_Retrain_Results.md`. Decision: B and C are equivalent, and a tie favours C. Watch h16v01 (snow): C baseline is +3.8 MAE vs historical there. Waiting for the project lead: NEON steps 2-5 (optional), and planning the cloudy-area work.

**2026-09-30 ~11:00: h12v09 dense-forest drops explained, NOT QA.** Pixels with C <= HIST-30 are persistently cloudy (eastern Amazon, Pará). Composites there fall back to all-day means with clouds. Retrained B drops at the same pixels, and more often (RF 94k vs C 38k pixels). C has fewer cloud fallbacks than B. Open question: why the C-code historical product is less affected. `C_vs_Historical_h12v09_20260930.md`.

**2026-09-30 ~10:30: C vs historical on h12v09 by bin:** no meaningful dense-forest reduction. Binned by HIST, C is 1.3-1.8 lower at 60-100, but this is mostly a binning artifact (binned by C: 0 to +0.6). The tile mean is unchanged, and C is less low than the old B. `C_vs_Historical_h12v09_20260930.md`.

**2026-09-30: DECISION: B and C are equivalent in accuracy** (every reference puts them within about 1 point of MAE; differences <= ~1 point are treated as equivalent). A tie favours C (bit 15), per the retrain plan. See the Decision section in `Lidar_Reference_Coincidence_20260930.md`. The NEON plan is now about describing accuracy, not the B/C choice.

**2026-09-30 ~10:05: NEON lidar evaluation PLANNED, step 1 done.** `NEON_Lidar_Eval_Plan.md`. 25 NEON sites (mostly 2019) are local in `ilab/data/AGB/Airborne_Lidar`. The FSD CC layer turns out to be the share of CHM > 2 m in a 3x3 window (reproduced exactly), so it is derived from the CHM, not a gap fraction. NEON mapping script tested on UNDE 2019: 3,053 pixels on h11v04 in 58 s. Steps 2-5 (about 1-2.5 h) are waiting for the project lead. Automatic in the meantime: the bit-15 run finishes and the final h11v02 G-LiHT evaluation runs.

**2026-09-30 ~09:20: G-LiHT truth on h11v02, preliminary.** `Lidar_Reference_Coincidence_20260930.md`. 2,663 non-label pixels, 2014 lidar vs 2019 products. The labels match lidar best at a 3 m height cutoff (r 0.86, bias 0). All products compress the range (MAE ~19-20, r ~0.55; stable pixels MAE ~11-13, r ~0.78). C is WORSE than B on this tile (baseline +0.8, XGB +1.1 MAE, significant), even though C was better on the held-out labels. No significant difference on stable pixels. Fire ruled out (32 of 2,663 pixels burned after the flights). The high-cover offset fits canopy cover (MODIS) vs crown cover (CHM); the low-end offset is not explained. Final run with C RF starts automatically when that prediction lands.

**2026-09-30 ~08:15: stable-pixel reference set (Stage 6 substitute) DONE with option (a).** `Stable_Pixel_Reference_Selection.md`. `FullRuns/stable_pixels_20260930/stable_pixels_subbin_10000.csv` has 10,000 pixels on h11v04, h18v03 and h19v09, about 588 in each 5-point sub-bin from 0 to 85. Stable means SD over the 21 C6 years <= max(3, the 10th percentile in that sub-bin); each pixel is >= 3 px from any label. h19v09 mid-range pixels (SD up to 8.5) kept as they are by decision; results will also be reported per tile. Bit-15 Stage 5b predictions are still running.

**STATUS 2026-09-30 07:08: bit-15 (C) 63-tile run: Stages 0-5 FINISHED; Stage 5b predictions RUNNING (restarted 07:06, about 4 h, done about 11:00-11:30).** The run stalled at 01:40: `vcfPredictQueue.py` requires its output folder to exist, and `node_chain.sh` never created `Python/predictions_C/<model>/`, so all 4 nodes retried the baseline predict every ~3 s for 5.5 h (rc=1, 0/63, no tile claimed, nothing else affected). Fixed by creating `predictions_C/{baseline,rf,xgboost_tunedB}`; the running chains picked it up with no relaunch. **Held-out 20% pixels (provisional; Stage 6 still deferred): C beats B on all three models.** MAE B→C: baseline 7.45→7.25, RF 7.52→7.32, XGBoost tunedB 8.58→8.38. R² +0.006 to +0.007, bias unchanged. Per tile, C is better on 53/63 (baseline), 54/63 (RF) and 47/63 (XGBoost) tiles. C is lower in every label bin for every model. Results are in `FullRuns/qaC_full63_retrain_20260929/Python/eval/`. The full tables and results report follow when `status/run.done` appears. Still needed from the project lead: the Stage 6 tiles and reference.

**FINISHED 2026-09-30 10:41 (was RUNNING, launched 2026-09-29 21:17): 63-tile bit-15 (C) rebuild, B and C retrained, held-out comparison.** Resume from "Launch and resume info" at the end of the plan; finished = `FullRuns/qaC_full63_retrain_20260929/status/run.done`. `Aerosol_QA_C_Full63_Retrain_Plan.md`. forest201/202/203/205 only (204 busy). 20% pixel split only (no whole-tile holdout, project lead's decision), 6 fits, then 63-tile predictions with the C models and a per-tile comparison with the historical MOD44B (Stage 5b), about 10 hours. The independent test on non-training tiles is DEFERRED to 2026-09-30 (tiles and reference to be chosen; tree labels exist only for the 63 tiles); until then any result is provisional. Awaiting approval; nothing run.

**FINISHED (2026-09-29 ~17:00): aerosol QA variant C (bit 15) on h09v05, h12v09, h16v01. Nothing running.** `Aerosol_QA_Variant_Test_Results.md`. Bit 15 removes 0.3-2% of good pixels on the two non-snow tiles (current screen: 43-50%), 21% on h16v01. Tree cover with the existing models moves little on h09v05 and h16v01 and about +1.3 on h12v09. The label MAE is in-sample for B, so it cannot pick a winner. Awaiting the project lead's decision: recommended next step is a 63-tile C rebuild, retraining, and a held-out B-vs-C comparison (needs its own plan).

**VCF WORK RESUMED; PLAN WRITTEN (2026-09-29 ~14:00): aerosol QA variant test.** `Aerosol_QA_Variant_Test_Plan.md`.
It compares, on h12v09 and h09v05, the QA mask with no aerosol test (historical), the current bits 7-8 screen (reuse the 63-tile run) and
MOD44C `state` bit 15 (PGE72's overall QC flag). The bit-15 case rests on the MOD44C investigation. At production time, PGE72 reads the
internal MOD09GQK/GHK/GST, which carry more observations than the public MOD09GA/GQ, so exact per-pixel day recovery from MOD09 is slow and
incomplete. Awaiting approval; nothing run.

**IMPORTANT FOR ALL VCF RESULTS (2026-09-29 ~10:45):** the PGE61 aerosol test adopted on 2026-09-15 (MOD44C `state` bits 7-8 = 3)
has been shown on real data to reject 29-50% of otherwise good pixels (every low-aerosol pixel from an odd day), and 99.8% of its rejections are
not high-aerosol. Every C and Python product since 09-15, including the 63-tile run and everything built on it, used it. Details:
`/explore/nobackup/people/mcarrol2/MOD44C_PGE72_Investigation/Reports/Aerosol_Theories_Test_Results.md`.

**VCF WORK PARKED (2026-09-29, ~09:50). Nothing running; all run markers are `.done`.** The project lead split the PGE72/MOD44C
issue (the aerosol bits in MOD44C `state` overlap other fields, which decides whether MOD44C can be used for aerosol screening) into a
separate investigation: **`/explore/nobackup/people/mcarrol2/MOD44C_PGE72_Investigation/README_START_HERE.md`**. The aerosol report
`MOD44C_State_Aerosol_Bit_Collision.md` moved there (a stub with its new path is left here). It bears on every comparison against
the historical MOD44B, including `Model_vs_Historical_DeepDive_20260929.md`.
When the VCF work resumes, the open threads are:
- **Greenland/Iceland fix:** `Greenland_Iceland_Scoped_Fix_Plan.md`, waiting for MOD44C data.
- **Model choice:** XGBoost tunedB is best on 59/63 tiles (deep dive).
- **Non-vegetated:** the layer is weak and hasn't been tried with RF/XGBoost.

**FINISHED (2026-09-29 08:51): model vs historical deep dive on all 63 tiles.** `Model_vs_Historical_DeepDive_20260929.md` (last section),
summary in `FullRuns/regtree_alt_models_20260927/Compare/deepdive_all63_summary/`. Pooled MAE: baseline 5.06, RF 4.48, XGBoost tunedB 4.02, C 3.54.
XGBoost is best of the Python models on 59/63 tiles and below C on 9. Caveat: the historical product had no working aerosol screen (see the report's top section).

**FINISHED (2026-09-29 08:06, extended 08:26 to h12v02, h12v09 and h20v06): model vs historical deep dive.** `Model_vs_Historical_DeepDive_20260929.md`.
XGBoost tunedB is best of the three Python models (MAE 1.82 vs RF 2.36 and baseline 2.86; C 1.41). All three over-predict sparse woodland
(historical 11-50%) in the same places, the northern NM/SW CO uplands (block error maps correlate 0.83-0.97). No edge or striping artifacts.
The non-vegetated layer is the baseline model for all three. The script takes any tile; awaiting the project lead's review.

**PLAN WRITTEN (2026-09-29): Greenland/Iceland-scoped fix, blocked on input data.** `Greenland_Iceland_Scoped_Fix_Plan.md`.
Greenland tiles: h15v02, h15v03, h16v00, h16v01, h16v02, h17v00, h17v01. Iceland is h17v02 (plus the edge of h16v02), not h15v02.
MOD44C lives in `/css/modis/Collection6.1/L3/MOD44B-VCF/dev/<year>/` and in the pipeline's `/explore/nobackup/projects/ilab/data/MODIS/MOD44C/`;
both have the 2019 window only for h16v01 (21 of 23; A2019337 and A2019353 missing). The project lead will fetch the other 322 files. Nothing run.

**SESSION END (2026-09-28, late evening). Nothing running, no code changed.** This session, after
the cutoff-70 retest: recorded the Greenland/Iceland direction below in four reports; reviewed the
Random Forest / XGBoost tunedB results against the historical MOD44B, including h12v09 per-tile numbers
(`FullRuns/regtree_alt_models_20260927/Compare/regtree_63tile/`; on h12v09 XGBoost tunedB MAE 5.71 vs
C 5.55, Random Forest 6.23, baseline 7.04). The project lead is thinking through next steps; nothing is
waiting on a job. Next step on the h16v01 work: design the Greenland (and possibly Iceland) tile list
and scope, as a written plan, before running anything.

**PROJECT LEAD'S DIRECTION (2026-09-28, after the cutoff-70 retest): scope the fix to Greenland, and consider Iceland.**
h16v01 is a Greenland tile. The project lead's view is that applying this zenith-only NDVI-default fix
only to Greenland tiles is likely to be the solution. Iceland tiles should be considered for inclusion
as well. This has not been designed or tested yet. Points to check when it is:
- **None of the 10 tiles the fix hurt is in Greenland or Iceland** (h11v02, h11v03, h18v03, h18v04,
  h20v02, h20v03, h21v02, h23v02, h23v03, h24v03 are northern Canada, Europe and Russia). That is
  consistent with the direction, but it is not a test of it.
- **The C code already scopes a rule this way.** `MOD44B3.c` lines 411-421 (the "Greenland" block, MSZ
  4/30/2026) sets non-vegetated to 100 when some metrics are zero, but only for `vtile < 3 && htile < 18`.
  That box also contains h11v02, one of the hurt tiles, so a Python scope copied from C's box would
  include a tile the fix damages. The list of Greenland (and Iceland) tiles needs to be explicit.
- **The tile list is still to be written.** Iceland is about 63-67 N and 13-24 W, which should put it in
  h16v02 and possibly the edge of h15v02; confirm against the tile grid. Check which of these tiles are
  among the 63 run tiles and the 31 in the 65-degree test.

**FINISHED (2026-09-28 21:09): cutoff-70 retest, two tiles (h16v01, h20v03).** Loosening the zenith bar from
65 to 70 degrees keeps the fix working on h16v01 and reduces, but does not remove, the damage on h20v03
(baseline MAE vs labels: real 6.71, 65 deg 15.93, 70 deg 13.77). Neither cutoff is a clean answer; the
recommendation to scope the fix by failure pattern stands. Nothing is running. Results and an account of a
duplicate-launch incident: `Zenith_Only_Fix_Cutoff70_Retest_Results.md`.

**JOB FINISHED, RESULT NEGATIVE FOR BLANKET ADOPTION (2026-09-28 18:29):** the 31-tile x 3-model
validation of the zenith-only NDVI-default fix (launched 16:30, all three node jobs done 18:20-18:29) is
complete and its cross-tile summary has been written: `Zenith_Only_Fix_63Tile_Results.md`. It resolves
h16v01's over-prediction as expected, but it also measurably **degrades accuracy against real training
labels on at least 10 of the other 30 flagged tiles**, consistently across all three models - not the
narrow, h16v01-only effect the earlier two-tile test suggested. Still a diagnostic; no code changed, no
fix adopted. `Zenith_Only_Fix_63Tile_Plan.md`'s "Progress and resume info" section has the run's
mechanics; the results document has the numbers and the tile-by-tile breakdown.

**PLAN WRITTEN (2026-09-28, same day; since run as a two-tile retest, finished 21:09 - see the FINISHED block above):** the run used a 65-degree zenith cutoff, which
checking turned up matches production's real `BandDayFile.ZENITH_CUTOFF` (the actual "is this day usable"
test) - not an error - but the codebase also has two other, different-purpose zenith cutoffs (70 for a
Band_31 value clamp, 72 in a debug script that duplicates the QA formula without referencing the shared
constant). `Zenith_Only_Fix_63Tile_Results.md`'s "Zenith cutoff used, and a codebase discrepancy found
while checking it" section has the detail. A sensitivity retest at a looser 70-degree cutoff is planned in
`Zenith_Only_Fix_Cutoff70_Retest_Plan.md` - scoped to h16v01 plus the 10 hurt tiles (~11 tiles instead of
31, roughly 30-45 minutes), starting with a cheap Stage 0 rescan before any model compute. ~~Awaiting
approval to launch.~~ Run, scoped down to h16v01 and h20v03 at the project lead's direction.

Last updated 2026-09-28 (this status block; the results doc is new). Previously updated 2026-09-20 20:55 (after the predictor-selection work below; earlier the same day, audit of every document in `Reports/` and `C_code/pge61/MOD_PR44B_Metrics/`). Read the docs below rather
than searching the tree.

### State on 2026-09-20 (63-tile run)

The full **63-tile C vs Python run is finished** (launched 2026-09-19 12:42, compute ended 19:28, comparison 19:40-19:55; about
7 hours). Everything ran in a clean run root, `MOD44B_VCF_Development/FullRuns/full63_c_vs_py_20260919-0806/` (`$RUN` below);
nothing outside it was written (checked). Headline (tree cover, 1.11 billion valid pixels): correlation 0.96, MAE 4.5, bias +0.4;
non-vegetated correlation 0.81 (C uses static bare models, Python retrained on 20 tiles). The h12v02 disagreement seen in the
two-tile test is gone.

- Results: `FullRun_Results_C_vs_Python_63tiles.md` (and `$RUN/Compare/c_vs_python_63/`).
- What was run, how, how to check or repair it: `FullRun_Launch_and_Status.md`. Status any time: `bash $RUN/code/validation/run_status.sh`.
- Sign-off view with evidence, limits and checklist: `Signoff_Package_2tile_C_vs_Python.md` (item 8 has the 63-tile update).

**Later on 2026-09-20** (all in the run root `$RUN` and `Reports/`; no product code changed): comparison with the historical MOD44B (`Tree_Cover_vs_Historical_MOD44B_Results.md`); 63 per-tile figures and CSV tables (`figures/`); feature importance of the existing model, a rebuilt 324-metric training table, a stand-alone Monte Carlo predictor selection with two final models, and a map-level check of those models on three tiles (`Feature_Importance_*.md`). The sync to the main repositories is **on hold** until the project lead has spoken to the repository owner (`Sync_Back_and_Cleanup_Plan.md`); nothing has been committed or copied.

### Resume notes written 2026-09-20 21:00

**State.** Nothing is running or queued on any node; every marker under `$RUN/status/` is `.done` except the old `compare.failed` (2026-09-19 18:44), which is explained in `FullRun_Launch_and_Status.md` (the driver's own compare step was stopped by the too-strict size rule; `compare_only.done` is the rerun that succeeded). No pending decision is waiting on a running job. The only thing waiting is you.

**Standing conditions** (the project lead's; also in `Sync_Back_and_Cleanup_Plan.md` and the plans):
- Nothing outside the sync workspace `AI_code_sync_09152026` and the run root `$RUN` is overwritten. New work goes into new folders that refuse reuse.
- **The sync to the main repositories is on hold** until the project lead has spoken to the repository owner (rlgill, who owns pge61, pge72 and the `modis_vcf` repository). Nothing has been committed, pushed or copied; do not commit or push there.
- Plans are written and approved before anything is executed; long runs are launched fully detached (`setsid nohup ... & disown`, status and done marker files) so they survive an SSH or VS Code disconnect; give a time estimate when there is one.

**Options not done, with what each would take** (none has been approved; costs are measured from today's runs except where marked):

| Option | Question it answers | What exists / what is needed | Cost |
|---|---|---|---|
| Seed baseline | How much of the 4.6-point map difference between the Monte Carlo top-64 and existing models is just random tree-building? | Refit the existing 64 predictors with another seed: the original fit was `rfFitFromPge61.py --randomState 0 --nEstimators 30 --predictorFile $RUN/manifests/predictors_pcttree.txt` on `Python/training/pcttree/xTrain.parq` (`python_phases_node.sh`, lines 98 to 106); use `--randomState 1` and a new output folder. Then predict the same three tiles with `run_predict_model_node.sh` and compare (`compare_model_variants.py` has fixed labels and needs a variant added). **No launcher written yet.** | one 64-predictor fit took about 75 to 80 minutes at full data (Monte Carlo final fit), then 2 to 5 minutes per tile |
| Random 64-predictor null | Does the choice of 64 predictors matter at all? | New small script; fit two or three random 64-sets from the 324 in `Python/training_324/pcttree/` | about 75 to 80 minutes per fit, parallel over nodes |
| Split by tile | Are the held-out scores inflated by neighbouring pixels? | New script; per-tile Parquet files are in `Python/training_324/` (63 files) | 15 fits for 5 folds and 3 predictor sets, about 3 to 4 hours over 5 nodes (estimate) |
| Redundancy-aware selection | Would a selection that avoids near-duplicate predictors beat the top 64? | New script (correlation clustering or forward selection), then the two tests above | not estimated |
| More Monte Carlo trials | Not recommended (see the results document) | Procedure written in the plan, "Decision rule for continuing" (move the minvar-10 outputs into `results_minvar10/` first) | 214 to 321 more trials, about 0.8 to 1.2 hours on five nodes |
| More maps | Other 60 tiles for the Monte Carlo models, a non-vegetated layer, figures of the new maps, a colorblind-safe (blue to orange) version of the difference figures | `run_predict_model_node.sh` takes a tile list; `run_figures_node.sh` makes figures | 2 to 5 minutes per tile per model; figures not timed here |

**Environment.**
- Compute nodes `forest201` to `forest205` (12 cores, 112 GB each); the login nodes have no `singularity`, NumPy or GDAL. Container `/explore/nobackup/projects/ilab/containers/modis-vcf-latest`. Every launcher here binds the project tree read-only plus its own new output folder only.
- **Usable memory is about 65 to 70 GB per node** when running heavy work, not 105 GB: two 20 GB metric workers plus a 35 GB table build got 9 tiles killed by the memory watchdog (it kills the newest metrics job when free memory drops under 8 GB). A 13 to 15 GB model loads in 1.5 to 2 minutes; predicting with the top-64 model took about 26 GB in total on a node (free memory fell from 101 to 75 GB).
- **Other users share these nodes.** On 2026-09-20 user another user used about 800% CPU on several of them for hours; check `uptime` and free memory before launching, and do not assume the load is yours.
- Shell pitfalls seen: `noclobber` makes a retried `>` redirect onto an existing log fail at once (use `>>` or a new name); a claim directory left by a stopped runner blocks its tile until removed; `cp` is aliased to interactive (use `\cp -pf`); `ssh` prints a login banner on standard output, so filter output; `pgrep -f` matches the ssh command that ran it; do not edit a `.sh` launcher while it is running.
- Status: `bash $RUN/code/validation/run_status.sh` reports **only the 63-tile run**. For 2026-09-20 work list `$RUN/status/`: `mc_*` (Monte Carlo), `predict_<label>` and `variant_compare` (map check), `figures_*`, `hist_compare_*`, `p0_*` (table build).

**Where things are:** the Monte Carlo and its models in `$RUN/Python/monte_carlo_standalone/tree/`; the 324-metric rasters in `$RUN/Python/metrics_324/` (1.1 TB, kept on purpose) and tables in `$RUN/Python/training_324/`; the map check in `$RUN/Compare/mc_model_prediction/`; all scripts in `$RUN/code/validation/`. `$RUN` = `/explore/nobackup/people/mcarrol2/MOD44B_VCF_Development/FullRuns/full63_c_vs_py_20260919-0806`.

