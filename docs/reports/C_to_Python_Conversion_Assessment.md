# MODIS-VCF C-to-Python Conversion Assessment

2026-10-01

## Summary

In four weeks the project went from an operational C pipeline nobody could build to both C and Python running on all 63 training tiles, with the Python product now improving on its own merits. The C toolkits and executables were built from source in about 10 days, exposing 5 latent C bugs and several missing configuration files. Python then reached exact agreement with C on band metrics, correlation 0.96 with C on tree cover, and a tie with the historical MOD44B against NEON lidar. On the 63 training tiles for 2019, the Python RF tree cover is comparable to the historical MOD44B: a pooled mean absolute difference of 4.2 points with negligible bias, within about 0.6 points of the operational C code, and weakest in dense forest (about 3 points low). Open work: the SD layer, the non-vegetated model, high-latitude and cloudy regions, and delivery to the repository owner.

## Problem statement

The operational MOD44B Vegetation Continuous Fields product (PGE61, MOD\_PR44B) is produced by legacy C code that is complicated and hard to maintain. It runs as three executables chained by intermediate files. It depends on the NASA SDP Toolkit, HDF4, HDF-EOS2/5, api-trans and an embedded Java VM running Weka M5P regression trees.

A conversion to Python (the `modis_vcf` repository) had been started but was blocked. Without a working C build there was no way to run the operational code side by side with Python. That meant no way to check that Python reproduces the C metrics, to find where the two differ, or to tell which differences were Python bugs and which were C behaviour.

The C code could not be built easily because its toolkits are old, partly undocumented and no longer packaged for current Linux systems. Starting in early September 2026, work in VS Code with AI coding assistants got the toolkits and all three C executables built and running. That opened the way to a direct comparison between C and Python, and from there to improving the Python pipeline itself.

## Building the toolkits and the C code

It took about 10 days (2026-09-03 to 2026-09-12) to go from source tarballs to a 63-tile operational C run: the toolkits built 2026-09-03 to 09-08, the 3 executables running end to end on 2026-09-11, and all 63 tiles through B1, B2 and B3 at 05:15 on 2026-09-12. Most of the difficulty was not compiling. It was the runtime configuration (process control files, metadata files, model files) that operations had always supplied and that was not in the repository.

### Toolkit stack, built from source into a user directory

No system packages were usable, so every dependency was built from source into `~/.local` on the ADAPT cluster, with no root access and no rpath baked into the executables.

| Component | Version | Why it was hard |
| --- | --- | --- |
| NASA SDP Toolkit (`libPGSTK.a`, `smfcompile`, message catalogs, leap-second and time databases) | 5.2.20 | Distributed as a `.tar.Z` (needed `ncompress` 5.0 built first). The leap-second path in the shipped config pointed at a dead operations path (`/data/landops20`). |
| HDF4 | 4.2.13 | Needs Sun RPC/XDR, which modern glibc no longer ships; `libtirpc` 1.3.4 had to be built first. |
| HDF-EOS2 / HDF-EOS5 | 3.0 / 2.0 | HDF-EOS2 3.0 changed function prototypes; `MOD_PR44B3/GDapi.c` needed a compatibility patch (`ease.h`, const-qualified prototypes). |
| HDF5, SZIP, JPEG, zlib | 1.8.19, 2.1.1, 9b, 1.2.11 | Old versions matched to the HDF stack; built as shared libraries, so every run needs `LD_LIBRARY_PATH`. |
| api-trans | 7.0.4 | MODAPS-internal library; taken from an in-tree copy. |
| Java (embedded JVM) | Temurin JDK 8 | B2 and B3 start a JVM through JNI to run Weka; libjvm path passed to make by hand. |
| Weka | 3.6.14 (not 3.8.6) | See "Weka version archaeology" below. |

The PGE's message header (`PGS_MODIS_37440.h`) is not checked in. It is generated from `MODIS_37440.t` with `smfcompile`, and the same seed file is needed to build PGE72 (MOD\_PR44C).

### Runtime configuration that operations had always supplied

- **Process control files (PCFs).** The test PCFs used logical IDs (299xxx) that the 7.x code no longer reads; they were remapped to the 244xxx IDs in `MOD44B1.h`. The tile ID is passed as a number (`51009005` for h09v05) whose characters are sliced to get h and v.
- **A blank line poisons a PCF.** One blank line inside a file-table section silently stops the Toolkit from resolving every later entry in that section. Found with a small stand-alone C harness that calls `PGS_PC_GetReference`.
- **Path field conventions.** An empty path field means "use the section default", so a full path in the filename field produced a nonexistent concatenated path for the MOD44W water mask.
- **Missing metadata file.** There was no ODL `MOD44B.mcf` anywhere in the tree (the `SHR_MCF/*.C7.mcf` files are api-trans manifests, not ODL). One was written from `MOD44CH.mcf` and the C7 manifest. Modelling the 9 additional attributes as one array made `PGS_MET_SetAttr` segfault at index 7; the correct structure (9 separate CLASS-numbered containers) was found in the operational Perl script `COMB/PGE61T.pl`.
- **Missing `header.bare.arff`.** Not in the tree or the 2023 archive copy; rebuilt from the 21 bare-attribute indices in `MOD44B3.c`.
- **The processing year is not the calendar year.** It is 23 sixteen-day composites, day 065 of the year to day 049 of the next.

### Weka version archaeology

- B2 called M5P with `-P 6`, an option Weka 3.8.6 rejects. `MOD44B2.c` never checks the return of `system()`, so B2 reported success while training no models. The option was removed.
- The 30 shipped bare-ground models (`BARE/weka.bare.NN.model`) were serialized with an old Weka (`FastVector`) and would not load in 3.8.6. Weka 3.6.14 from Maven Central loads them and trains new models, so both B2 and B3 were moved to it.
- B3 could not find `Predict.class` unless it sat beside `weka.jar`.
- Under `gdb`, the JVM start-up crashes; that is a debugger artefact, not a bug, and cost time before it was recognised.

### Latent C bugs exposed by actually running the code

| Where | Bug | Effect |
| --- | --- | --- |
| `MOD44B3.c` ~line 134 | `getenv("MODAPS_INSTANCE")` with no NULL check | Segfault outside the operational environment |
| `MOD44B3.c` ~line 133 | Bare-model directory hard-coded to `/MODAPSx/PGE/AM1M/coeff/...` | Cannot run off the operations system |
| `create_output_file.c` ~738 | `version` not reset inside the `PGS_PC_GetReference` loop | Only the first of 30 model LUNs resolves |
| `fix_bands.c:137` (B1 and B3) | Inputs opened read-write (`DFACC_RDWR`) though only read | Fails on every tile whose inputs the user does not own: nearly all 63 tiles |
| `MOD44B2.c:131` | Unchecked `system()` and the `-P 6` option | Silent failure: exit 0, no models |

### Scale-up to 63 tiles

Python scripts generate a PCF per tile, reading each granule's real file name rather than substituting text, and give each tile its own log directory so concurrent runs do not corrupt each other. An audit found 59 of 63 tiles with all 23 composites; the 4 northernmost (v01) tiles lack the two December periods (polar night). B3 takes 10 to 36 minutes a tile at about 2.6 GB of memory, depending on how much land the tile has. The full 63-tile pipeline (B1, then 30 B2 models, then B3) finished 63/63 at 05:15 on 2026-09-12.

## Python milestones, 2026-09-12 to 2026-09-30

With C running, the work went in two phases. Through 2026-09-20 the goal was to make Python reproduce C. From 2026-09-27 it shifted to making the Python product as good as possible, with C as one reference among several. The h12v09 cloud-composite variants of 2026-09-30 are left out here.

| Date | Milestone | Result |
| --- | --- | --- |
| 09-12 to 09-14 | Per-pixel metrics compared between C and Python on all 63 tiles | Six code-level differences identified (`C_vs_Python_Code_Differences.md`); one report per tile |
| 09-15 | Zenith cutoff and QA bit reads aligned | Python zenith cutoff 70 changed to C's 65; adjacency bit 6 fixed |
| 09-16 to 09-17 | Single-pixel trace of h12v02 finds the real cause of the high-latitude collapse | Python clamped Bands 1-7 on zenith; C's clamp never fires (uninitialized read). After the fix, h09v05 band metrics are 100% exact; h12v02 `BandReflMax` goes from 0 valid pixels to 100% exact |
| 09-17 to 09-18 | Two-tile end-to-end test, C (B1, B2, B3) and Python (metrics, training, prediction) | Python held-out tree cover correlation 0.969, MAE 0.86 |
| 09-19 | Python mirrors of C's derived attributes (`CAttributes.py`, `CMirrorMetrics.py`) | Reproduce all 252 of C's derived metrics exactly, so both sides train on the same predictor definitions |
| 09-19 | 63-tile C and Python run, about 7 hours | Tree cover C vs Python: correlation 0.960, MAE 4.53 on 1.11 billion pixels. Python parallelized (it had used one CPU) |
| 09-20 | Both compared with the historical MOD44B Collection 6 | MAE 3.49 for C, 5.00 for Python; C closer on 60 of 63 tiles |
| 09-20 | Feature importance and a Monte Carlo predictor search over all 324 metrics (702 trials) | A different top 64 (20 shared) scored slightly worse; C's 64 predictors kept. Six defects found in the shipped `MonteCarloSim` |
| 09-21 | Delivery clone of `nasa-nccs-hpda/modis_vcf` set up | Branch `mcarrol2/c-python-parity-fixes` off `develop` |
| 09-27 | Random Forest and XGBoost tried against the baseline bagged trees | Pooled MAE vs historical: baseline 5.06, RF 4.48, XGBoost 4.02 (C 3.54); fits take 10-30 minutes instead of 81 |
| 09-27 | Output post-processing fixed in `VcfPredict.py` | Predictions clamped to 0-100, the 3 layers sum to 100, water and out-of-projection codes on all layers |
| 09-27 | Valid-mask disagreement on 4 high-latitude tiles explained | MOD44W out-of-projection area, not a defect |
| 09-27 to 09-28 | h16v01 (Greenland) tree cover over-prediction diagnosed | Zenith cutoff leaves 5 of 8 periods unfiltered. A zenith-only fix cures h16v01 but hurts 10 other tiles; not adopted, to be scoped to Greenland/Iceland |
| 09-29 | Aerosol bit collision in MOD44C `state` found | The bits 7-8 screen rejected 29-50% of good pixels; the historical MOD44B had no working aerosol screen |
| 09-29 to 09-30 | 63-tile rebuild and retrain comparing aerosol screens | Bits 7-8 and bit 15 within about 1 MAE point everywhere; bit 15 adopted |
| 09-30 | Independent references: stable-pixel set (10,000 pixels, 3 tiles), G-LiHT on h11v02, NEON lidar (50,561 pixels, 19 sites) | NEON: RF ties the historical MOD44B (3 m MAE 20.39 vs 20.43) |
| 09-30 | Model choice | XGBoost leads on historical-based references, trails on held-out labels; RF and XGBoost treated as equivalent, RF favoured for its SD layer |
| 09-30 | Non-vegetated predictors corrected | Default list had 13 of 18 names wrong; now C's 21 bare attributes |
| 09-30 | Branch pushed to GitHub | `335ceb2`; 2 further local commits waiting to be pushed |

The comparison results above measure agreement with C or with the historical product, not accuracy. Neither is ground truth, and the historical product carries the aerosol defect found on 09-29.

## Is the Python product comparable to the historical MOD44B?

**Conclusion (agreed with the project lead, 2026-10-01):** On the 63 training tiles for 2019, the Python RF product reproduces the historical MOD44B tree cover to a pooled mean absolute difference of 4.2 points with negligible overall bias, within about 0.6 points of the operational C code. Agreement is weakest in dense forest, where Python reads about 3 points lower.

"Comparable" here means within a few points of the historical product, not identical to it. The evidence comes from the results in hand before the h12v09 cloudy-pixel work started.

### Supporting evidence

Bit-15 retrain against the historical MOD44B C6, 63 tiles, about 1.07 billion common-valid pixels (`Aerosol_QA_C_Full63_Retrain_Tables.md`, Stage 5b). The per-tile columns were computed from that report's per-tile MAE table.

| Product | Pooled MAE | Pooled bias | Per-tile median MAE | Tiles over 5 | Tiles over 7 | Worst tile |
| --- | --- | --- | --- | --- | --- | --- |
| Operational C (rebuilt) | 3.56 | -0.13 | 3.52 | 12 | 2 | 8.0 (h19v08) |
| Python RF | 4.20 | +0.28 | 4.39 | 23 | 2 | 7.5 (h19v08) |
| Python XGBoost tunedB | 3.61 | -0.08 | 3.64 | 10 | 2 | 7.6 (h19v08) |
| Python baseline | 4.87 | +0.34 | 4.90 | 31 | 8 | 10.2 (h16v01) |

- The pooled average differs from historical by about 4 points for RF and under 4 for XGBoost, with almost no overall bias.
- Python RF is within about 0.6 points of what the operational C code itself produces against its own historical product. XGBoost is within 0.05.
- The two tiles over 7 (h19v08, h19v09) are just as bad for C. That points to the reference or the inputs, not to Python.
- The earlier run (bits 7-8 screen) gave RF a correlation of 0.963 and 88% of pixels within 10 points (`Regression_Tree_Methods_RF_XGBoost_Results.md`).
- On independent NEON lidar (50,561 pixels, 19 sites), RF ties the historical product: 3 m MAE 20.39 vs 20.43 (`NEON_Lidar_Eval_Results.md`).

### Qualifications to state with the claim

1. **Average versus per pixel.** "Within a few points" is true of the average error. It is not true of every pixel: about 10% of pixels differ by more than 10 points, and RMSE is about 6.4 (bit-15 RF vs C6, 63 tiles).
2. **Dense forest.** Bit-15 RF reads systematically low there against C6: -2.5 at 61-70% cover, -3.0 at 71-80% and -3.0 at 81-90% (-2.8 over all cover above 60%). Sparse cover reads +1.1 to +1.7 high at 1-30% cover. The C code shows the same pattern (-2.8 above 60%). Figures from the 63-tile four-way comparison, `Historical_C6_C61_Four_Way_Comparison_Results.md`. (Corrected 2026-10-01: the dense-forest and per-pixel figures first given here, -3.8 / -4.2, +1.5 to +2, 12% and RMSE 7, came from the older bits 7-8 RF; the figures now shown are the bit-15 RF, `Historical_C6_C61_Four_Way_Comparison_Results.md`.)
3. **Agreement, not accuracy.** The historical product isn't ground truth, and it had a broken aerosol screen. "Comparable to historical" should not be read as "equally accurate". NEON is the only independent support, and it covers 19 sites.
4. **Scope.** The evidence covers only 2019 and only the 63 training tiles, which are the same tiles whose labels trained the model. Nothing outside the training tiles or for another year has been tested.
5. **h16v01 (Greenland).** RF is fine there (5.2), but the baseline model under bit 15 is not (10.2). The claim needs RF or XGBoost, not the baseline.
6. **Cloudy regions.** The h12v09-type dense-forest drops are already inside these numbers (h12v09 RF MAE is 5.5). They don't break the claim, but they are local errors that the tile averages smooth over.

**Against C6.1 too (2026-10-01, `Historical_C6_C61_Four_Way_Comparison_Results.md`).** On the 54 tiles not affected by C6.1's high-latitude rule (C6.1 sets those pixels' tree cover and non-tree vegetation to 0 and non-vegetated to 100), Python RF differs from C6.1 by a pooled MAE of 4.17 (bias +0.25) and from C6 by 4.10; the C code by 3.42 and 3.44. The two historical collections differ from each other by 2.64. By cover bin and region, Python RF and the C code share the same pattern (high at low cover, about -2.8 above 60%); Python RF reads about 2 points high at 21-40% in most regions, +3 to +5 on sparse and mid-cover pixels in equatorial South America, and low in Europe.

## Recommendation: Python RF as the replacement for the C code (2026-10-01)

For tree cover, Python RF is a sound replacement candidate. It is within about 0.7 MAE points of the C code against both historical collections, which is a tie under the project rule (about 1 point or less), and it moves the product by less than twice what one reprocessing of the historical product (C6 to C6.1) did. Before it is called an operational replacement:

1. Test outside the training set: a few non-training tiles and a second year, against C6 and C6.1.
2. Build the per-pixel SD layer for RF.
3. Replace the baseline model for the non-vegetated (bare) layer.
4. Rebuild the C code with bit 15, or keep stating that the C code reference used the bits 7-8 shift.
5. Investigate the 21-40% over-reading and the equatorial South America sparse-cover over-reading.

The out-of-sample test and the SD layer are the two items that could still change this conclusion.

## Current state and decisions

- **Pipelines.** C and Python both run end to end on all 63 training tiles for 2019. The C build is reproducible from `build_context/env.sh` and the PGE72 recipe in `BUILD_forest203.md`.
- **Python code.** All Python changes are on branch `mcarrol2/c-python-parity-fixes` in our own clone, pushed to GitHub at `335ceb2`. The pull request into `develop` has not been opened. Nothing was changed in rlgill's checkouts.
- **Aerosol QA.** `state` bit 15 is in the code. The one C source change (aerosol shift `>> 13` to `>> 7`) implements the screen later found wrong and should not be applied.
- **Model.** RF and XGBoost tunedB are treated as equal in accuracy; RF is favoured because it can produce a per-pixel SD layer like the historical product. The non-vegetated layer still uses the baseline model.
- **Decision rule.** Model choices rest on differences that agree across references; about 1 MAE point or less is a tie.

## Open items

- [ ] Push the 2 local commits and open the pull request into `develop`
- [ ] Build the per-pixel SD layer for RF and compare it with the historical SD
- [ ] Try RF or XGBoost for the non-vegetated (bare) layer (next session)
- [ ] Out-of-sample test: non-training tiles and a second year
- [ ] Greenland/Iceland scoped zenith fix (blocked on MOD44C files)
- [ ] Check h16v01 (snow) under bit 15 before production (+3.8 MAE vs historical)
- [ ] Persistently cloudy regions (dense tropical forest), the next area for model improvement
- [ ] For rlgill: which changes to take, the `BaggingRegressor` design change, the 6 `MonteCarloSim` defects, and the C bugs listed above

## Sources

- C build: `pge61_08312026/pge61/build_context/native_build_context.txt`, `env.sh`, `pipeline_logs/full_run.log`; `pge72/pge72-main/BUILD_forest203.md`; toolkit sources in `~/.local/src`
- Early comparison: `pge61_08312026/pge61/MOD_PR44B_Metrics/C_vs_Python_Code_Differences.md` and the 63 per-tile reports
- Python work: `MOD44B_VCF_Development/AI_code_sync_09152026/Reports/`, especially `Project_Summary_and_Conclusions.md` and `README_START_HERE.md`
