# Plan: stale-reference scan, cleanup candidates, and sync back to the main repositories

**Update 2026-09-30:** the Python part of the sync is done a different way: the changes are committed on a branch in the delivery clone for the project lead to push (`Git_Delivery_Clone_modis_vcf.md`), and `$SYNC/Python_code` was brought up to the branch (phase 1, Python part). Phases 2 to 5 as written below (cleanup tiers, staging folder, patch verification, pge61 hand-off) have not been executed. The C `fix_bands.c` change should **not** be delivered: it implements the bits 7-8 aerosol screen, which was shown on 2026-09-29 to reject 29-50% of good pixels.

Written 2026-09-20. **Status: the document fixes of Part 1 (findings 1 to 3) were carried out later the same day, with approval; nothing else in this plan has been executed.** The scan behind it was read-only (`diff -rq`, `du`, `find`, `grep`, and a script that checks every file name and
absolute path cited in the documents). The scan changed no file, and no file was moved or deleted. The document fixes edited six documents (below); copies of the originals are in the session scratchpad, not in the workspace.

Names used: `$SYNC` = `/explore/nobackup/people/mcarrol2/MOD44B_VCF_Development/AI_code_sync_09152026`; `$RUN` = `.../FullRuns/full63_c_vs_py_20260919-0806`;
`MAIN_C` = `/explore/nobackup/projects/ilab/projects/MODIS-VCF/pge61_08312026/pge61` (the pge61 git repository); `MAIN_PY` = `/explore/nobackup/people/rlgill/innovation-lab-repositories/{modis_vcf,core}`.

## Standing conditions for any sync

1. **`MAIN_C` and `MAIN_PY` belong to rlgill.** Nothing is committed, pushed, or edited in place in either. The sync produces a staging folder and patch files outside both repositories; rlgill (or the user,
   by explicit decision) applies them.
2. **Nothing is deleted or written without the user's approval of that specific step.** Each phase below is a separate approval.
3. **Clean up before syncing**, so that only intended files can reach the staging folder.
4. **Re-baseline at sync time.** `MAIN_C` is not at a clean commit: `git status` shows 72 files modified against HEAD `f3405bc` and 34 untracked entries, none made by this work as far as this scan can tell
   (for example `MOD_PR44B1/fix_bands.c` already has an unrelated `SDstart` change and a file-mode change there). Patches are made against the tree as it is when the sync starts.

## Part 1. Stale-reference scan of the documents

Scope: 29 documents (all of `Reports/*.md`, the three documents in `C_code/pge61/MOD_PR44B_Metrics/`, the two `Sync_Plan_*` documents and `C_code/pge61/README.md`); the 63 per-tile metric comparisons carry a
dated note and were checked only for that note.

**Cross-references between documents: none broken.** Every `*.md` name cited in a backtick resolves. Of 64 distinct unresolved file or path citations, 3 are real (findings 3 below) and 61 are false positives (generic names such as
`metrics.bin`, `.fix.out` or `MOD_PR44B2.NN.pcf`; abbreviated paths such as `/panfs/.../pge61`; paths that exist only inside the container, such as `/usr/local/ilab/modis_vcf`; helper files that live outside
the workspace but exist, such as `VCF_outputs/batch_remaining_tiles/orchestrate_remaining_tiles.sh`).

**Real findings** (findings 1, 2 and 3 were fixed on 2026-09-20; 4 to 6 needed no change):

| # | Where | Problem | Proposed fix |
|---|---|---|---|
| 1 | `C_code/pge61/Sync_Plan_and_Model_Retraining.md` and `..._Alternate.md` | Both say "**planning only, not executed**" (dated 2026-09-14). They were carried out: the 65-degree zenith cutoff, the adjacency bit 6 and aerosol bits 7-8 reads, and the BaggingRegressor swap all exist in the code. The audit of 2026-09-20 did not cover these two files. | **Fixed 2026-09-20.** Each now opens with a dated status note; the first has a table of what became of phases 0 to 6, the second says the recommendation was followed in part (no permanent regression harness was built). Added a line saying "sync" there means harmonizing C and Python, not delivering to the main repositories. |
| 2 | `Reports/Predictor_Subset_C_vs_Python.md` line 121 | "Pending the user's choice." The choice was made (implementation B), and line 141 of the same document says so. | **Fixed 2026-09-20.** The line now says the user chose implementation B and points to the outcome section of the same document. |
| 3 | `Reports/Single_Pixel_Trace_Plan_h12v02.md` line 23; `Reports/execution_log.md` lines 337 and 398 | Cite `sample_c_bin.py`, `run_b1_both.sh` and `run_py_metrics_both.sh`, which exist nowhere in the workspace, the run root or `VCF_outputs`. They were one-off scripts. | **Fixed 2026-09-20.** Each is marked "[script no longer present]" at the cited line; the historical text is otherwise unchanged. |
| 4 | `Reports/FinalCompare_2tile_Plan_and_Status.md` line 146, `Signoff_Package_2tile_C_vs_Python.md` line 95, `TrackA_B3_Failure_Root_Cause_and_Fix.md` line 78 | "Deleting was blocked by the permission classifier" for the 30 stray `generated_b2` PCFs and 30 `runtime/logs/b2_NN/` directories. Still true; it is README open item 6. | Leave; item 6 is handled in Part 3 and phase 2. |
| 5 | Bodies of `Finding6_Fix_Plan_h12v02.md`, `BandReflMax_Bug_Investigation_Plan.md`, `PyTrackB_2tile_Plan_and_Status.md` | Still contain "not yet applied", "not yet decided", "not started" in the body, but each has a top-of-file note saying the state is historical. | No change needed. Listed so nobody re-flags them. |
| 6 | `C_code/pge61/MOD_PR44B_Metrics/HOWTO_C_vs_Python_Comparison.md` (revised earlier today) | Its historical paragraph still cites `VCF_outputs/batch_remaining_tiles/` and `VCF_outputs/Metrics/`, which live outside the workspace. | Fine as history; if the workspace is archived, `VCF_outputs` must be archived with it or the paragraph annotated. |

Also worth knowing: the earlier audit's assertion "no superseded phrases remain in the current-state documents" holds for the documents it covered; findings 1 and 2 above were outside or missed by that search.

## Part 2. What differs from the main repositories (what a sync would carry)

### C side (`$SYNC/C_code/pge61` against `MAIN_C`, build_context excluded): 73 files differ, 0 exist only in the sync copy

- **One real source change, in two files:** `MOD_PR44B1/fix_bands.c` and `MOD_PR44B3/fix_bands.c`, line 187: `aerosol = (... >> 13) & 3;` becomes `>> 7`. This is the aerosol decision of 2026-09-14 (bits 7-8 of the repacked
  state; see `C_vs_Python_Code_Differences.md`).
- **Not source:** `bin/MOD_PR44B1.exe` (rebuilt 2026-09-15), `MOD_PR44B1/fix_bands.o`, `MOD_PR44B_Metrics/gen_metrics` (binary). `bin/` and the `.o` files are untracked in `MAIN_C` too.
- **`MOD_PR44B_Metrics/` (untracked in `MAIN_C`, so it was never part of the repository):** newer `gen_metrics.c` (has the workspace path `SYNC_ROOT` compiled in, so it is a workspace tool, not deliverable code),
  `compare_c_python_metrics.py`, the three documents and the 63 annotated per-tile comparisons.
- **`build_context/` (untracked in `MAIN_C`, 15 GB there):** three scripts differ (`generate_b2_pcf.py`, `generate_tile_pcf.py`, `run_full_pipeline.py`), two exist only in the sync copy
  (`launch_track_a_2tile.sh`, `run_track_a_2tile.py`), and every generated PCF differs because it carries workspace paths. Personal build harness, not repository content.

### Python side: the current code is in `$RUN/code/python_code`, not in `$SYNC/Python_code`

`$SYNC/Python_code` was last touched 2026-09-18 and is **behind** the run-root copy used for the 63-tile run (which has the option C mirror files, the read cache and the parallel prediction queue). A sync must
start from the run-root copy, or first bring `$SYNC/Python_code` up to date (phase 1). `MAIN_PY`'s files are unchanged since 2025 (modification dates 2025-07-18 and 2025-12-18), so every difference is ours.

| File (against `MAIN_PY`) | Change | Kind |
|---|---|---|
| `model/BandDayFile.py` | `ZENITH_CUTOFF` 70 to 65; Bands 1-7 value clamp removed; Band_31 clamp added (`VALUE_CLAMP_ZENITH_CUTOFF = 70`) | **Bug fixes**, validated in `Bands17_Fix_Validation_Results.md` |
| `model/ProductTypeMod44.py` | adjacency `state & 8192` becomes `state & 64`; aerosol `(state & 192) >> 6` becomes `(state & 384) >> 7` | **Bug fixes** (Findings 1-2) |
| `model/BuildTraining.py` | the `trainingDir` argument was accepted but never used; now used | **Bug fix**, independent of the C comparison |
| `model/Metrics.py` | per-metric read cache so each file is read once | **Performance**, behavior unchanged |
| `view/vcfPredict.py`, new `view/vcfPredictQueue.py` | predict many tiles in one process, models loaded once | **Performance / new file** |
| `model/VcfPredict.py`, `model/MonteCarloSim.py`, `model/tests/test_MonteCarloSim.py`, `view/monteCarloSim.py`, `view/rfFitFromPge61.py` | cuML `RandomForestRegressor` replaced by scikit-learn `BaggingRegressor` (CPU; `--nEstimators`; `Band31` names corrected to `Band_31`) | **Design change**, made because forest203 has no GPU. Rlgill's code uses cuML. Not a bug fix. |
| new `model/CAttributes.py`, `CMirrorMetrics.py`, `CEquivalentMetrics.py` | Python mirrors of C's derived metrics (implementations B and A) | **New feature**, needed for matched predictors; implementation A is a cross-check only |
| `modis_water/model/Utils.py`, `modis_water/__init__.py`, `model/__init__.py` | vendored one-method shim because the container has no `modis_water` | **Workspace-only. Do not sync.** The real package exists in `MAIN_PY`. |
| `modis_vcf/data/` (2.6 GB) | copy of `MAIN_PY`'s data | **Do not sync** (identical to the source) |

**Added 2026-09-20 after this scan:** the predictor-selection work changed no `modis_vcf` file, so it adds nothing to this table; its scripts are in `$RUN/code/validation` (harness, never synced). It found six defects in the shipped `MonteCarloSim` (in `MAIN_PY` too, unchanged there and here); whether to fix them is for the repository owner (`Project_Summary_and_Conclusions.md`, section 8, item 6). The metric rasters and training tables it added are kept by decision ("keep the metric rasters for now"), so they are not in any cleanup tier; their removal would be a separate decision.

### Never synced

Binaries and objects (`*.exe`, `*.o`, `*.a`, `gen_metrics`), `__pycache__`, the workspace-path copy of `gen_metrics.c`, all `Metrics/` data, everything under `Reports/` that is not a document, the run-root harness
(`$RUN/code/c_scripts`, `$RUN/code/validation`), and the generated PCFs.

## Part 3. Cleanup candidates (sizes from `du` today; nothing deleted)

The workspace is **448 GB**, of which `Metrics/` is about 416 GB and `build_context/runtime` 30 GB.

| Tier | What | Size | Why it can go | Risk |
|---|---|---|---|---|
| 0, no information lost | 56 `__pycache__` directories, 386 `.pyc` files | 20 MB | regenerated on import | none |
| 0 | `GetAttr.temp` files (66: workspace root, `bin/`, `runtime/`, one per tile log directory), `MODIS_37440.t` | few MB | scratch left by the PGS toolkit | none; also untracked in `MAIN_C` |
| 1, temporary | `Metrics/_smoke_test_tmp` (738 files, last written 09-15) | 30 GB | smoke test scratch | low; check `smoke_test_getbandcube.py` output is not cited |
| 1 | `Metrics/_spotcheck_h12v02_tmp` (93 files, 09-15) | 3.7 GB | spot-check scratch | low |
| 1 | 30 stray `build_context/test_pcf/generated_b2/MOD_PR44B2.NN.pcf` and 30 `runtime/logs/b2_NN/` from Track A's Sep 17 run (README open item 6) | small | untracked by-products | low. The same 30 + 30 exist in `MAIN_C/build_context` too. |
| 2, superseded evidence (regenerable, about 45-50 min per tile each) | `Metrics/{h09v05,h12v02}/Python_finding6` (44 GB each), `Python_zenithfix` (32 GB each), `Python_bands17fix` (32 GB each) | 216 GB | each was superseded by `Python_bands17_band31fix`; they back the before/after numbers in the reports | medium: the numbers stay in `compare/` and the documents; only the raw rasters go |
| 2 | `Metrics/{h09v05,h12v02}/Python` (original baseline) | 64 GB | the "before" of the metrics fix | medium; keep if the before/after must be rerunnable without regenerating |
| 2 | `Metrics/PyTrackB_2tile` (7.6 GB), `Metrics/TrackB_2tile` (383 MB) | 8 GB | two-tile Track A/B test outputs, historical | low to medium |
| 3, keep unless told otherwise | `Metrics/*/Python_bands17_band31fix` (32 GB each), `Metrics/*/C/*.metrics.bin` (18 and 16 GB), `Metrics/*/compare`, `build_context/runtime/intermediate` (30 GB, the `.fix.out` inputs of `gen_metrics`) | about 130 GB | current state and the inputs needed to reproduce it | do not delete |
| Docs directory | `Reports/`: 67 non-document files: one-off `check_*.py`, `run_*.sh` launchers, 26 `.log` files (7 of them empty), `ndvi_difference_stats_*.txt` (674 KB and 57 KB) | about 1 MB | the scripts are cited by the documents as provenance | keep the scripts; only the 7 empty `.log` files (and possibly the tiny launch logs) are removable |
| Outside the workspace, for information | `$RUN/code/validation/_audit_*.py` and other one-off scripts (14 files), `$RUN/Python` (1.6 TB at the time of this scan, **plus about 1.3 TB added on 2026-09-20 by the predictor-selection work**: `metrics_324` 1,121 GB, `metrics_324_fresh` 106 GB, `training_324` 21 GB, `monte_carlo_standalone` 36 GB including two 14 GB models), `$RUN/Compare/mc_model_prediction` (0.7 GB), `$RUN/C` 967 GB; `MAIN_C` untracked build products (`*.o`, `PGS_*`, `api-trans-7.0.4`, `build_context` 15 GB, `bin/`, `.claude/scheduled_tasks.lock`) | large | the run root is the 63-tile evidence; `MAIN_C` items are rlgill's repository | **not touched by this plan without a separate decision** |

Tier 0 and 1 total about 34 GB. Tier 2 is another 288 GB.

## Part 4. The sync, in phases (each phase is its own approval)

**Phase 0. Freeze and re-baseline (read only).** Record the HEAD and `git status` of `MAIN_C`, the modification times of `MAIN_PY`, and SHA-256 sums of every candidate file in the sync workspace and run root.
Re-run the two `diff -rq` scans of Part 2. If `MAIN_C` or `MAIN_PY` changed since 2026-09-20, stop and revise the list.

**Phase 1. Fix the stale documents (Part 1) and bring `$SYNC/Python_code` up to date.**
- *Documents: done 2026-09-20* (findings 1 to 3, six documents edited, references re-scanned; the new citations resolve).
- *Python code copy: not done, needs its own approval.* Copy the run-root `python_code` changes into `$SYNC/Python_code`, listing each file copied, so the sync workspace becomes the single source of the Python
  delivery. Backups first. Nothing under `$SYNC/Python_code` has been touched.

**Phase 2. Cleanup, tier by tier.** Tier 0, then tier 1, each with the exact file list shown beforehand and an approval, then a check that no document cites a removed path. Tier 2 only if the user chooses, one directory
group at a time, after confirming the `compare/` results are intact. Nothing in tier 3, the run root or `MAIN_C` is touched.

**Phase 3. Build the staging folder (outside both repositories).** Proposed location: `$SYNC/../SYNC_BACK_STAGING_20260920/`. Contents, grouped so each group can be taken or refused on its own:
- `A_bug_fixes/`: the two `fix_bands.c` line changes, `BandDayFile.py`, `ProductTypeMod44.py`, `BuildTraining.py`, each as a unified diff against the tree recorded in phase 0, plus the changed file;
- `B_performance/`: `Metrics.py`, `vcfPredict.py`, `vcfPredictQueue.py`;
- `C_c_mirror_metrics/`: `CAttributes.py`, `CMirrorMetrics.py`, `CEquivalentMetrics.py`;
- `D_cpu_bagging_swap/`: the five files of the BaggingRegressor swap, flagged as a design change;
- `docs/`: `Reports/` documents and the three `MOD_PR44B_Metrics` documents, if wanted;
- `MANIFEST.txt` with SHA-256, size, source path and reason for every file, and `NOT_INCLUDED.txt` listing what was left out and why (Part 2, "Never synced").

**Phase 4. Verify the staging folder without touching the repositories.** Copy `MAIN_C` and `MAIN_PY` into a scratch area, apply the patches there, then: compile the two `fix_bands.c` files; import all changed Python modules
and run the modules' existing unit tests in the container; run one tile end to end before anything larger: `makeMetrics.py` with the patched Python on h09v05 (about 45-50 minutes, detached) and compare the result with the existing `Metrics/h09v05/compare` numbers for `Python_bands17_band31fix`, which must match. Report failures instead of adjusting patches silently.

**Phase 5. Hand-off.** Write a change note for rlgill (per file: what, why, evidence document, risk), list the groups that can be applied independently, and hand it to the user. How it reaches rlgill (patch files by
message, a branch made by rlgill, or a merge request) is the user's decision. This plan does not commit, push or apply anything.

**Rollback.** Phases 0, 3 and 4 write only new folders. Phase 1 and 2 keep backups (documents) or delete only what was listed and approved, so the recorded checksums and the manifest show exactly what changed.

## Decisions needed before the remaining phases run

1. **Where does "main repo" point?** Both `MAIN_C` (pge61) and `MAIN_PY` (modis_vcf), as this plan assumes, or only one?
2. **Which groups to deliver?** A is the natural minimum. B, C and D are separable. D changes the model type and should be rlgill's decision.
3. **Cleanup depth:** tier 0 and 1 only (about 34 GB), or also tier 2 (another 288 GB, regenerable)? If tier 2, keep the baseline `Python` directories or not?
4. **Should the documents travel with the code** (as `docs/` in the staging folder), or stay in the sync workspace?
5. **Delivery mechanism** for the hand-off (phase 5).
6. **The 30 + 30 stray Track A files in the pge61 repository directory** (untracked; `MAIN_C` is `/explore/nobackup/projects/ilab/projects/MODIS-VCF/pge61_08312026/pge61`). Delete them, leave them, or only list them for rlgill?
   - 30 PCF files: `/explore/nobackup/projects/ilab/projects/MODIS-VCF/pge61_08312026/pge61/build_context/test_pcf/generated_b2/MOD_PR44B2.NN.pcf` (NN 00 to 29)
   - 30 log directories: `/explore/nobackup/projects/ilab/projects/MODIS-VCF/pge61_08312026/pge61/build_context/runtime/logs/b2_NN/` (NN 00 to 29)
