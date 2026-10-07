# Start here: MODIS-VCF development workspace

**2026-10-06 afternoon: `Band31_Unclamped_Results.md` written. Band 31 unclamped does NOT pass D9 (b): tree held-out +0.022 MAE
[+0.011, +0.036] (seed change −0.002); bare −0.016 [−0.033, −0.001]; Greenland ice fixed on h16v02 but broken on h16v01/h17v01. Recommendation:
do not adopt (v1.1.0 unchanged). Decision C7 (summer-snow rule) CLOSED: dropped. Ice training rows on h16v02 approved (T1-T6 as suggested) and
running as `FullRuns/EXP_ice_rows_h16v02_20261006` -> `Ice_Training_Rows_h16v02_Results.md`.**

**START HERE ON 2026-10-06 (updated 22:05, project lead logged off):** check status of two detached runs, both EXPERIMENTAL:
(1) `FullRuns/EXP_fillonly_20261005` (fill-only test, expected end ~01:00) -> write its results into `Composite_Unusable_Inputs_Results.md`
(a new section; Greenland tiles already unchanged: h16v01 <100 154,353 -> 161,530; h17v01 148,200 -> 148,995);
(2) `FullRuns/EXP_b31_unclamped_20261005` (`Band31_Unclamped_Plan.md`, launched 22:01; self-test + gate due ~23:15-23:45; full run
~7-9 h) -> `Band31_Unclamped_Results.md`, remembering the caution note (the clamp was a design decision; adopt only with care).
Earlier note: **START HERE ON 2026-10-06:** (1) DONE 2026-10-05 evening (see the entry below; `Composite_Unusable_Inputs_Results.md`): check status of `FullRuns/EXP_unusable_inputs_20261005` (EXPERIMENTAL; started 2026-10-05 11:18 on
ngaproc201/202, expected end about 19:00-19:30) and write `Composite_Unusable_Inputs_Results.md` against the D9 (b) rule;
(2) **the push is DONE** (verified 2026-10-05 with `git ls-remote origin`; tag `v1.1.0` on `920d06f`); still open in section A of `Decisions_For_20261005.md`: A4 (maps on disk) and A5 (pull request).

**2026-10-05 late evening: `Ice_Sheet_Diagnosis_Results.md` written (EXP_icesheet_diag_20261005, read-only, all checks exact).** Two problems:
the ice margin (MCD12 class-15 pixels that look like ground in summer; probably right as < 100) and interior ice just below 100 (h16v02 1.9 M px),
driven by the **band-31 solar-zenith clamp** (16000 = 160 K on an emissive band) and by h16v02's ice learning from Canadian tundra rows
(only h16v01 has ice training rows). H3 (decision 19.6) does not matter. Also: MOD44C inputs on the ice carry solz ~86 in June (all fallback;
upstream question). **Next: `Band31_Unclamped_Plan.md` (U1-U6), awaiting approval.**

**2026-10-05 late evening: `Ice_Sheet_Diagnosis_Plan.md` written (read-only diagnosis of why the Greenland ice sheet is below 100 bare in v1.1.0;
the project lead wants a data/pipeline fix, no masks or override rules; MCD12 is a reference only). **E1 approved, E3 yes** (MCD12 ice counts = reference only, not the ice sheet's boundary); E2 withdrawn.**

**2026-10-05 evening (project lead): full unusable-input rule NOT adopted (v1.1.0 unchanged); fill-only short test approved and launched
(`FullRuns/EXP_fillonly_20261005`, EXPERIMENTAL; −28672 only, v1.1.0 models, 22 tiles; results to write when done).**

**2026-10-05 evening: `Composite_Unusable_Inputs_Results.md` written. EXP_unusable_inputs_20261005 is done and does NOT pass D9 (b).**
Tree cover is unchanged (−0.001 MAE). Bare is slightly worse: +0.069 MAE [+0.003, +0.159], +0.109 on the rebuilt tiles. The cause is the
polar-night tiles h12v01, h12v02, h13v01 and h21v01 (+0.31 to +0.46 each). On Greenland the labels and C6 disagree (h16v01 labels better;
h16v01 and h17v01 worse against C6). Recommendation: do not adopt; options 2-3 in the report. The h15v02 map ran out of memory and was re-run alone; nothing lost.

**2026-10-05 (project lead):**
- **D1-D12 all decided** (`Composite_Unusable_Inputs_Plan.md` section 9): all as recommended except **D9 = (b), the stricter rule**
  (bare improves AND tree cover unchanged or better). D4 approved after clarifying that the fill test is per band (a fill in band 5
  never removes another band's value). The overnight count sent D10 to the full test.
- **Run launched** on the ngaproc nodes only (forest nodes busy with other users): self-test PASS (56/56 band-periods on h16v02 and h12v09
  match the count exactly), gate PASS (h12v09, h11v04, h20v06 training rows identical), 2 setup stops fixed (missing folders; nothing lost).
  The screen of the other 51 tiles added 9 tiles with tiny kind-C changes (at most 0.005% of land); 24 tiles rebuilt.
- **Update 2026-10-05 evening: A1-A3 DONE** (tag moved to `920d06f`, both branches and tags pushed; verified 2026-10-05 with `git ls-remote origin`). A4 and A5 still open.
- **Push deferred:** the project lead will get help with Git from someone who knows the NCCS/GitHub setup. Section A of
  `Decisions_For_20261005.md` still open (A1 tag move, A2 push, A3 WV2 footprint names, A4 maps on disk (suggested: leave), A5 PR).
- **To-do 21 added:** figures of the code's progression, from the original C code and the incomplete Python to the final version.
- Still open from `Decisions_For_20261005.md`: D (other years after this run; Quality/Cloud/HDF).


Last updated 2026-10-03 afternoon (versions named; Monte Carlo rerun written up; cloudy-pixel experiments of 2026-10-03 written up: V4r blend, snow/low-sun check, QA bits, 60 trees and counts, gated 63-tile blend, h12v09 drop pixels). **Read `Product_Versions_and_Work_Map.md` first:** it defines **v1.0.0** (the current product, called BASE in the cloudy-pixel reports; local tag `v1.0.0` on `16d6474`), **v1.0.1** (EXPERIMENTAL, not adopted: v1.0.0 with tree cover blended with V4R under the gated cloud, cover and snow rules; other layers v1.0.0's; `Gated_V4r_Blend_63Tile_Results.md`), **v0.0.0** (bagged trees, MEAN of 30, bit-15 inputs; the earlier "baseline"; in the project lead's view the closest match to the C code's intent), **v0.0.0-pre.1** (rlgill's incomplete upstream Python) and **v0.0.0-pre.2** (bagged trees on the bits 7-8 aerosol test, the September runs), and sorts the work into experiments to improve v1.0.0 (cloudy pixels, Monte Carlo rerun), validation (NGA WV2, AlphaEarth, lidar) and assessment (held-out labels, map comparisons, stable set, parity). Then read this section, then `Project_Summary_and_Conclusions.md`. The dated status entries are kept word for word under "Status history" at the end.

## Current state (2026-09-30)
- **2026-10-07: v1.2.0 BUILT and tagged locally** (`FullRuns/v120_20261007`, all 69 tiles pass; `Version_v1.2.0_Build_Results.md`). Branch
  `mcarrol2/v1.2.0` + tag `v1.2.0` (`45edf62`) in the delivery clone; **the project lead pushes**, then the PR into `develop`. v00 rule kept
  (`EXP_v00_norule_20261007`: bare ~100 on ice but the tree model adds 1-2%). Closed today: A4, B6, E12; band 31 unclamped not adopted.
- **2026-10-07: experiment to do (not yet run): `Tree_Ice_Rows_and_Bare_Priority_Plan.md`** (tree rows = 0 on ice; bare = 100 trusted over small tree).
- **2026-10-07: NEXT PRIORITY after the v1.2.0 build: figures of the whole process of the past month** (open item F13 in `Decisions_For_20261005.md`).
- **DONE (verified 2026-10-05 with `git ls-remote origin`): the push.** Remaining: open the pull request into `develop`. *Old reminder:* **REMINDER FOR THE NEXT SESSION (project lead, 2026-10-01): start by pushing the delivery branch to GitHub** (`Git_delivery/modis_vcf`, branch `mcarrol2/c-python-parity-fixes`, 9 commits ahead as of 2026-10-02 (`13773db` Greenland fix, `16d6474` rounding + v00 rule); the cluster has no GitHub credentials, so the project lead pushes; see `Git_Delivery_Clone_modis_vcf.md`, "To push"). First refresh the `docs/reports` snapshot if it has not been done. Then review the overnight results (to-do items 2-5). **Still not pushed as of 2026-10-02 morning.**
- **2026-10-02 MORNING: overnight results (reports written):**
  - **Delivery (item 1): all 63 tiles pass** (`Delivery_SD_RF_Median_Results.md`). Branch ready to push.
  - **h16v01 (item 3)** (`H16v01_Delivered_Product_Check_Results.md`): MEDIAN solves the tree over-prediction (0.40, C6 MAE 0.44). Bare is still 89.0 against C6 ~98.7 (bias -9.6), so non-tree vegetation is 10.6.
  - **Greenland/Iceland zenith fix v2 (item 4)** (`Greenland_Iceland_Scoped_Fix_v2_Results.md`): both gates passed, and the polar-night tile h17v00 runs (100% valid). **Greenland criterion met:** bare C6 MAE roughly halves (h16v00 26.9 -> 12.6, h17v00 21.2 -> 10.2, h16v01 10.2 -> 5.5); tree drops on the polar-night tiles (6.95 -> 2.94, 5.75 -> 2.25); h15v02 is a tie. **Iceland does not qualify** (bare 10.6 -> 11.5). **DECIDED 2026-10-02: adopted for the 6 Greenland tiles with a fixed tile list, Iceland excluded; committed locally as `13773db`.** NDVI equals the run's on all 6 tiles; h17v00 end to end is identical in all 5 layers. About 2-3% tree cover remains on h16v00/h17v00 (periods with no input are filled with 10000 in every band); **DECIDED 2026-10-02: the high-latitude rule is adopted for v00 tiles only** (tree 0 / non-vegetated 100 on land; `VcfPredict.HIGH_LATITUDE_BARE_VTILES = (0,)`).
  - **Bare in the C regions (item 5)** (`Bare_C_Regions_Investigation_Results.md`):
    - **High latitude:** RF is 6-27 below C6 inside the v7 mask on Greenland. On the small Arctic islands (h12v01, h13v01) RF is ~24, the labels are 23-52 and C6 is ~99. Forcing 100 would be wrong there.
    - **Sahara stretch:** it matches C6 better (h18v07 MAE 13.0 -> 8.0), but **worsens the held-out labels on h18v07 (7.3 -> 11.1)**.
    - **Conflict:** C6 does not show bare = 100 on the v6.0.3-only tiles (C6 bare 10-17 there), so last night's side finding "C6 includes the all-v<3 rule" is unconfirmed.
    - Options are listed for the project lead.
  - **Rounding (item 2): DONE after a retry** (`Rounding_vs_Truncation_Results.md`). All differences are ties: tree cover ROUND −0.03 MAE against C6, C6.1 and the C code; non-vegetated −0.20 against C6/C6.1; lidar split (NEON better, G-LiHT worse); held-out labels ±0.04. ROUND removes truncation's downward bias. **DECIDED 2026-10-02: ROUND adopted** (`VcfPredict._roundValue`, both layers). The first pass lost 4 tiles to out-of-memory (3 workers per node); they were rerun with 1 per node, and a pandas bug in the held-out step was fixed.
- **OVERNIGHT RUNS 2026-10-01/02 (all launched about 20:27-20:36, fully detached; none depends on the Claude session). Check each done marker in the morning:**
  1. **Delivery run** (`FullRuns/delivery_sd_rf_median_20261001`), **DONE 20:43: all 63 tiles pass every check** (`out/summary63.md`). Tree equals the aggregation MEDIAN map, the SDs equal the run's SD, the capped bare is exact, sums are 100. The sum cap applies to 472,428 pixels with MEDIAN tree against 1,723,065 with MEAN tree.
  2. **Plan 2, rounding** (`FullRuns/rounding_vs_truncation_20261001`; ngaproc, 6 workers):
     - starts on the delivery `ALL.done`;
     - automatic gate on h08v05 (`status/step0.done` / `.failed`);
     - `status/ALL.done`; results in `out/summary_tree`, `out/summary_bare`, `out/ref_eval`, `out/heldout`.
     - The comparison scripts were adapted from the aggregation run and have not been tested. If `status/final.failed`, the maps in `out/TRUNC` and `out/ROUND` are still valid; fix and rerun the final step.
  3. **Plan 3, h16v01** (`FullRuns/h16v01_delivered_check_20261001`), `status/ALL.done`, `out/summary.md`.
     - **A smoke run on the final h16v01 map already gave the answer:** delivered tree mean 0.40 (C6 MAE 0.44), against RF MEAN 4.78 and the bit-15 baseline 9.20. So MEDIAN removes the h16v01 tree over-prediction.
     - But RF bare reads 89.0 against C6 about 98.6 (bias -9.6), so non-tree vegetation takes about 10.6.
  4. **Plan 4, Greenland/Iceland v2** (`FullRuns/greenland_zenith_v2_20261001`; forest201-205, 1 worker each):
     - gate = h17v00 `makeMetrics` + predict (`status/gate.done` / `.failed`);
     - then 5 more tiles, then zenith-fixed variants of 7 tiles, then `compare4.py`;
     - `status/ALL.done`, `out/compare/summary.md`. `compare4.py` has not been tested; on `compare.failed`, the maps in `out/real` and `out/fixed` are still valid.
  5. **Plan 5, bare in the C regions** (`FullRuns/bare_c_regions_20261001`; forest204):
     - phase 1 starts on the delivery `ALL.done` (`out/regionA.csv`, `regionB.csv`, `regionB_by_C6bin.csv`, masks);
     - phase 2 starts on the plan-4 `ALL.done` (new Greenland tiles; gives up at 09:00).
     - **Smoke-tested on h12v01 and h20v06:**
       - the rebuilt v7 mask matches the C code (bare = 100 on 100% of masked land);
       - the MOD44CQ-derived flags match the C intermediates 100% at the 65-degree cutoff;
       - on h20v06 the Sahara stretch takes RF's MAE against C6 from 1.03 to 0.24.
     - **Side finding:** the C6-era code (v6.0.3) also has the Sahara stretch and the all-v<3 high-latitude rule, so C6 includes both.
  - **Not done tonight:** the `docs/reports` snapshot commit on the delivery branch (do it before the push, item 8).
- **2026-10-01: delivery plan APPROVED, with non-vegetated = MEAN (provisional; project lead).** Reason: this layer already changes a lot at once (dynamic models instead of static ones, RF instead of bagging, and the QA changes from the tree-cover work), so its combination method stays as it is for now. It may be revisited after more evaluation. Tree cover = MEDIAN. Step 0 starting.
- **2026-10-01: delivery plan written** (`Delivery_SD_RF_Median_Plan.md`; awaiting approval). It adds a member pass to `VcfPredict` that gives the value (mean, mid-20 or median per layer) and the SD in one pass, with CLI options and unit tests. It runs a one-tile exactness check against the aggregation run, then commits on our clone branch (no push). The 63-tile run waits for the non-vegetated method decision.
- **2026-10-01 DECISIONS (project lead), after the RF aggregation run:** (1) **SD layer approved:** delivery code plus 63-tile SD, using the `SD_Layer_Plan.md` defaults. (2) **RF approved for non-vegetated.** (3) **MEDIAN of the 30 RF trees approved for tree cover.** Reason: the project lead values the greater spread, and believes the historical product is compressed toward the mean, so map references built from it favour MEAN for structural reasons. **Still open:** the final-pixel method for non-vegetated (per-tile and region-by-bin detail in `RF_Aggregation_Maps_63Tile_Results.md`, "Non-vegetated detail"), and the priority of to-do item 9.
- **2026-10-01: RF aggregation run done (18:46)** (`RF_Aggregation_Maps_63Tile_Results.md`). All 63 tiles; MEAN equals the stored maps everywhere. **Non-vegetated: MEAN is clearly better** (MID20 +1.3, MEDIAN +1.8 MAE against C6 and C6.1; consistent on every map reference). **Tree cover: the references disagree, and nearly every gap is a tie.** The maps and the stable set favour MEAN (MEDIAN +0.9-1.0). Lidar and held-out labels favour MEDIAN or MID20 (-0.2 to -0.6 overall; up to -2.4 in the 61+ bin, where the median eases dense-forest compression). **Suggested: keep MEAN for both; MID20 is the dense-forest alternative.** Decisions pending (project lead): final-pixel method, SD, bare RF, then the priority of to-do item 9.
- **2026-10-01: plan written for the `Quality` and `Cloud` layers and a single HDF product file** (`Quality_Cloud_Layers_and_HDF_Output_Plan.md`; awaiting approval). These are the last 2 of the 7 historical SDSs that Python does not write. Python already saves the per-period fallback flag, but its cloud flag (`outCloud`) is inverted and unused. The historical file (and the C code) is HDF4 / HDF-EOS2 with DEFLATE level 8 on every SDS; Python now writes 5 Int16 LZW GeoTIFFs at 135 MB per tile against the C code's 82 MB for 7 layers. Recommendation: keep HDF-EOS2 for C7 (written with `pyhdf`, not yet in the container) with a writer that can add HDF5 later; confirm what MODAPS / LP DAAC expect for C7.

- **2026-10-01: plan written for 63-tile maps of mean / middle-20 / median of the RF trees; approved and launched ~15:30; gate passed 15:34 (all 7 checks)** (`RF_Aggregation_Maps_63Tile_Plan.md`). Scored against C6, C6.1, the C code, NEON, G-LiHT and the stable set. 18 workers across forest201-205 and ngaproc201/202 (all idle at 15:35); under 1 hour after a one-tile check.
  - **Approved and launched 2026-10-01 ~15:45, fully detached** (`FullRuns/rf_aggregation_maps_20261001/`): 18 workers (forest201-205 x 2, ngaproc201/202 x 4).
    - **The step 0 check is an automatic gate,** because the project lead signed off during the run. On h08v05, MEAN must equal both existing bit-15 maps exactly, the SD must equal the SD test, and MID20/MEDIAN must move in the expected direction at the ends of the range (`scripts/gate_eval.py`). Any failure stops all workers (`status/step0.failed`).
    - **After the gate:** predictions on 63 tiles (MEAN, MID20 and MEDIAN products plus SD), then comparisons against C6, C6.1 and the C code for both layers, then NEON, G-LiHT and stable-set scoring (`scripts/ref_eval.py`, which also checks that MEAN equals the stored `C_rf`).
    - **Done marker:** `status/ALL.done`. Results in `out/summary_tree`, `out/summary_bare` and `out/ref_eval`.
    - **The bare RF decision waits for these results** (project lead).
    - **Attempt 1 stopped 16:03: 51/63 tiles OK, 12 failed with numpy out-of-memory errors** (all on forest202/203/204, 2 workers x 6 threads per forest node; none on ngaproc). The 51 OK tiles match the stored maps exactly (MEAN mismatch 0). No comparisons or ref_eval ran. Failed markers, claims and worker status files moved to `status/attempt1_oom_forest/`. **Rerun of the 12 tiles (h20v03, h20v08, h21v02, h22v03, h26v06, h27v04, h27v06, h27v07, h29v11, h29v12, h30v12, h31v11) launched 2026-10-01 18:06, fully detached on ngaproc201/202 only** (3 workers each, ids `r2_*`, same scripts). The same workers then run the 126 comparisons, summaries and ref_eval; done marker `status/ALL.done`.
- **2026-10-01: non-vegetated run done (15:11)** (`NonVegetated_Bit15_Models_Results.md`). **Recommended: RF.** On held-out labels it ties the baseline (4.20 against 4.18). On maps it is best against C6, C6.1 and the C code alike: against C6 it is 0.95 better than the baseline and 0.75 better than the current model, with the same sign on every reference. XGBoost is worse on both. The static models are worst on held-out labels (28.9). All retrained models read about 5 points less bare than C6, matching the labels.
- **2026-10-01: SD layers done** (`SD_Layer_Results.md`). The member SD leaves the maps unchanged (5 tiles exact) and costs about the same as `predict()`. It tracks held-out error better than C6.1's SD (Spearman: tree 0.87 against 0.78, non-vegetated 0.60 against 0.45). Levels are about 1.7-3x C6.1. One analysis bug (label column named `y`) was found and fixed; the v1 output is kept with an `_invalid` suffix. Next: delivery-branch code and 63-tile SD, awaiting approval.
- **2026-10-01: combining RF trees by median instead of mean** (`RF_Tree_Aggregation_Test_Results.md`, held-out, no refit). The median lowers MAE by 0.61 (tree cover, 61/63 tiles) and 0.37 (non-vegetated, 20/20), with CIs clear of zero. That is a tie under the 1-point rule. It reduces compression at both ends (61+ tree bias -8.5 to -6.3) at a small cost in the middle bins. Decision pending.
- **2026-10-01: SD layer plan written** (`SD_Layer_Plan.md`); **approved and run the same day; results in `SD_Layer_Results.md`**. It covers member-level SD for RF and the baseline, with tests on forest nodes only. Historical findings: C6's tree SD layer is a copy of the bare SD; the C code's SD has a +0.5 offset that sets a 0.5% floor; the `valid_range` attribute 0-275 is wrong.
  - **Approved; step 0 passed 2026-10-01 14:41 on forest203** (`FullRuns/sd_layer_test_20261001/out/step0_h08v05.json`). On h08v05, the 30-tree mean equals `rf.predict` exactly (max difference 0.0), and the int16 values equal the existing bit-15 map on all 17,991,862 land pixels. Timings: member pass 33 s against 25 s for `predict()` (12 threads); metric reading 154 s dominates. RF tree SD on land: mean 5.1%, median 2.3%, p95 17.9%, 15% exactly 0. C6.1 on the same tile: mean 2.2%, median 0.9%. Waiting for the go-ahead on steps 1-5.
  - **Steps 1-5 launched 14:48, fully detached on forest201-205** (`FullRuns/sd_layer_test_20261001`). The patched `VcfPredict` copy (`code/`, diff in `manifests/VcfPredict_sd.diff`) writes the `_SD` layers. Value layers still come from `predict()`. 5 tiles: h08v05, h12v09, h19v11, h20v06, h12v02. Analysis (unit check, maps unchanged, SD against held-out error, levels against C6.1 and the C code) runs on forest203 after the predictions; done marker `status/ALL.done`.
- **2026-10-01: static bare models traced** (`Bare_Static_Model_Pixel_Trace.md`, run on forest202/203).
  - The C code applies them correctly: a C `make_metrics` harness reproduces the C map exactly on 8 pixels.
  - They seem to expect a calendar shifted by half a year: shifting the inputs 4 periods cuts the MAE against C6 from 23.1 to 8.9 (best shift on 14 of 20 tiles, both hemispheres).
  - They are also fragile to small input differences. A Weka/Java version mismatch is ruled out. Supports retraining.
  - **C code bare post-processing rules** in `MOD44B3.c` (sum cap, high-latitude bare = 100, Sahara stretch; trace report section 3) are not in Python. **Deferred (project lead): consider them only after the models are assessed without them.** `bare_earth3` was an early attempt at understanding the bare models; ignore it.
- **2026-10-01: the static Weka bare models now run in pure numpy** (`Weka_M5P_Bare_Models_to_Python.md`; files in `AI_code_sync_09152026/Weka_M5P_Export/`). On 145,000 test rows they match Weka to within 1e-12, and the C code's rounded value is identical on every row. Side finding: the C code needs Weka 3.6.14 (`~/.local/share/weka-3.6.14`), not the 3.8.6 named in `build_context/env.sh`.
- **2026-10-01: non-vegetated (bare) plan written:** `NonVegetated_Bit15_Models_Plan.md`; **approved and run the same day; results in `NonVegetated_Bit15_Models_Results.md`**. It retrains the baseline and tries RF and XGBoost on the existing bit-15 metrics (no new metrics needed), on 20 bare-label tiles, with a 20% pixel split, then predicts all 63 tiles. Runs on ngaproc201/202 and starts with a one-tile cheap check.
  - **Approved 2026-10-01; step 0 (cheap check) passed at 11:31** in `FullRuns/nonveg_bit15_models_20261001/` (`Python/eval/step0_check_h20v06.json`). h20v06 bit-15 bare table: 170,067 rows, the same pixel keys and labels as the 2026-09-19 table, all 21 predictors present with no missing values, built in 262 s. Earlier runs are mounted read-only. Two earlier attempts are kept in `status/attempt*`: Lmod was not loaded, then the check used the wrong label column. Waiting for the go-ahead on steps 1-7.
  - **Steps 1-5 launched 2026-10-01 11:35, fully detached on ngaproc201 and ngaproc202** (`scripts/chain.sh`; node logs in `status/node_*.status`). ngaproc201 builds the tables, splits, fits the 3 models (at the same time) and runs the held-out evaluation. Both nodes then share the 3 x 63 predictions (bare model varies; tree cover is the bit-15 RF throughout). Markers: `eval.done`, `predict.done`. The map comparison (step 6) and the NDVI check (step 7) are launched separately once their scripts are written and tested. Pre-launch checks: table keys map to raster row = x, column = y, for both the Python and the C code maps (exact against the current model).
  - **Steps 6 and 7 launched 11:41-11:43, detached.** Step 6 (`scripts/step6.sh`, both nodes) waits for `predict.done`, then compares the 63 tiles and writes `Compare/summary/summary.md` and `status/ALL.done`; it was tested on h20v06 (`Compare/smoke_test_h20v06/`). Step 7 (`scripts/ndvi_check.py`, ngaproc202) tests the two NDVI definitions on the C code's own 16-day files. The C code averages the 16-day NDVIs (`make_metrics.c:65-83`); Python takes the NDVI of the 48-day mean B1 and B2 (`Metrics.getNdvi`). **Its first attempt read the files with the wrong layout and is invalid** (kept with an `_invalid_layout` suffix). The files are stored pixel by pixel, 23 periods per pixel. The rerun uses that layout and the C code's clear flag (`state` = 1).
  - **Step 7 done 11:47** (`Python/eval/ndvi_definition_check.csv`). The two definitions were applied to the same inputs: 1 M pixels, 3 tiles, 8 windows each.
    - Agreement within 1 NDVI unit: h20v06 68-80%; h19v11 54-65%; h12v02 3-58%. Within 10: 99%, 73-80% and 21-90%.
    - The differences are a heavy tail, not a general offset. On h20v06 the 95th percentile is 1.4 while the MAE is 6-11. On h19v11 the MAE is 180-390 with a 95th percentile of 650-1900.
    - On h12v02 the worst windows are those where 72-96% of pixels have no clear period, so the C code averages all periods.
    - This matches the earlier 66-83%, so the definition difference explains it. It concerns parity, not Python correctness, because Python's models are trained on Python's own NDVI. Report only.
  - **Step 4 (held-out) done 12:19** (`Python/eval/`; 2.16 M held-out pixels on 20 tiles).
    - Pooled MAE: baseline 4.18, RF 4.20, XGBoost 5.40, current 5.01, C code 28.3 (bias +21.9).
    - RF vs baseline +0.02, a tie. XGBoost is worse than RF by 1.19 (worse on all 20 tiles; CI clears zero).
    - RF vs current -0.81 (better on 16 of 20 tiles; CI clears zero), but under the ~1-point tie rule. The current model may have been trained on some of these pixels (different split), so this gap is if anything understated.
    - **C code side check** (`heldout_vs_historical_check.csv`): at the same pixels, historical C6 is MAE 11.9 (bias +6) from the labels, but 21.8 from the C code. So the 2026-09-19 C run's bare map does not reproduce C6, although both use the same static models. The worst tiles are h12v02, h21v01, h12v01, h13v01, h11v02 and h19v12. Cause not tested.
    - C6.1 vs labels is 33.7, from the high-latitude rule (non-vegetated = 100) on 6 of the 20 tiles.
  - **12:22: 2 extra prediction workers per node** (`scripts/extra_predict.sh`). Predictions ran at about 3.7 min per tile with load 2-6 of 24 cores. The extra workers use the same command and tile claims, so outputs are unchanged. Status in `status/extra_*.status`.
  - **Static-model check (approved 2026-10-01; `logs/eval_static.log`, `Python/eval/static_heldout_per_tile.csv`):** the C code's 30 static bare models, run in numpy on the Python bit-15 held-out inputs.
    - They are MAE 7.6 (bias -3.0) from the C code's map, but 23.1 (bias +12.9) from historical C6 and 28.9 from the labels. The C code itself is 21.7 from C6.
    - So the C run is not broken: on today's inputs, from either code, the static models give values unlike C6. Retraining is clearly better (RF 4.1 from the labels).
    - Not tested: whether the inputs changed since the static models were made (e.g. MOD44C collection or scaling), or whether C6 used other models.
  - **Fourth candidate STATIC launched 13:20, detached on both nodes** (`scripts/predict_static.py`, `static_chain.sh`). The inputs and output format are the same as `VcfPredict`; tree cover and the water/out-of-projection codes are copied from the bit-15 RF product.
    - Checked on h20v06: the map equals the held-out static predictions on all 34,022 pixels, and the copied tree cover equals this run's own.
    - Markers are in `status/static_markers` and cannot stop the main chain. The comparison scripts now include STATIC; the v1 copies are kept, and `compare_bare.py` waits for each static tile's marker.
- **2026-10-01: four-way comparison done** (C6, C6.1, the C code, Python RF; `Historical_C6_C61_Four_Way_Comparison_Results.md`). The comparability conclusion holds against C6.1 (54 unaffected tiles: Python RF 4.17, the C code 3.42; C6 vs C6.1 2.64). Python RF is the recommended replacement path for tree cover, pending an out-of-sample test, the SD layer and the non-vegetated layer (`C_to_Python_Conversion_Assessment.md`, "Recommendation"). **Next session: the non-vegetated (= bare) layer.**
- **2026-10-01: Python judged comparable to the historical MOD44B** (agreed with the project lead): on the 63 training tiles for 2019, Python RF tree cover is within a pooled MAE of 4.2 points of historical with negligible bias, about 0.6 points from the operational C code, weakest in dense forest (about 3 points low; corrected 2026-10-01 from 4, which was the older bits 7-8 RF). Evidence and qualifications: `C_to_Python_Conversion_Assessment.md` and `Project_Summary_and_Conclusions.md` section 10.
- **V4r** (lowest-red pick when all 3 MOD44C inputs are cloudy or cloud-adjacent), h12v09, steps 1-3 of the amendment in `Cloud_Composite_Alternatives_Plan.md`.
  - **Done 2026-09-30 ~20:39; results in `V4r_Darkest_Red_h12v09_Results.md`.**
    - Against historical, the drop-pixel bias moves from -21.7 to -4.9, but the all-land bias rises from +0.4 to +3.0.
    - Against the h12v09 held-out labels, MAE gets worse, 6.84 to 8.70, entirely at labels of 80-100.
    - The references disagree and the models were not retrained, so there is no decision. Next step proposed: retrain on V4r metrics (awaiting approval).
  - **First run failed** (`FullRuns/v4r_darkest_red_h12v09_20260930/`, `status/compose.failed`, kept as is). Its alignment check was exact in 7 of the 8 periods but 0.0 in 2020033. The fault was in the check, not the data: the 2020065 slot does not exist (no MOD44C file and no 1-Days file), and the stored 2020033 composite is exactly the mean of the 2 inputs that exist (2020033 and 2020049; identical at 100% of fallback pixels on Band_1, 2, 3 and 31). The check had filled the missing slot with 10000.
  - **Rerun (launched 2026-09-30 ~20:00, fully detached on ngaproc201):** `FullRuns/v4r_darkest_red_h12v09_20260930_r2/`. The code is the same except for the corrected check, which now averages only the inputs that exist. The chain runs compose, then metrics, then compare; `status/ALL.done` when finished. A second fix was made at about 20:20: the chain passed `makeMetrics -o $R/metrics` instead of `-o $R/metrics/h12v09`, so makeMetrics ignored the V4r composites and started rebuilding the standard ones. It was stopped, the path corrected and the chain restarted from metrics (compose kept). The log now shows "Reading Composite from .../h12v09/h12v09/2019/2-Composites". The partial tree from the wrong path is in `metrics/h12v09/STRAY_wrong_path_2019_unused/` (unused). Logs:
    - `logs/compose.log`: cheap check, alignment check and shadow/water gate;
    - `logs/compare.log`: map and held-out comparison, with exactness checks;
    - tables in `eval/`.
- **Daily-data variants** (R0 re-implementation of PGE72, V1, V2, V3, V3m, and V4 on R0), cloudiness only. **Done 2026-09-30 17:15** (`FullRuns/cloud_composites_h12v09_20260930/`, `out/measures_summary.txt`).
  - R0 reads "VALID: False" against the code threshold of 0.90: its exact band 1+2 match with the real MOD44CQ is 0.853-0.936 by window (median 0.881).
  - **The project lead accepted the about 88% match for this evaluation (2026-09-30).** The likely cause is that the public MOD09GA files hold only the first observation per day.
  - R0's fallback count matches the pipeline (3.290 against 3.289).
- **Both results reports written 2026-09-30:** `V4r_Darkest_Red_h12v09_Results.md` and `Cloud_Composite_Alternatives_Results.md`. On the daily-data variants: V4 gets most of V3's cloud reduction with no daily data and no change to clear pixels; V1 and V2 barely help.
- **Aerosol QA decision:** B (`state` bits 7-8) and C (`state` bit 15) are equivalent in accuracy, and a tie favours **C**. C is now in the code. Before production, look at h16v01 (snow): C's baseline model is +3.8 MAE there against historical.
- **Model choice is not settled (corrected 2026-09-30).** XGBoost tunedB is the best Python model only against references built from the historical MOD44B: best C (bit-15) model on 56 of 63 tiles against the historical map (pooled MAE baseline 4.87 / RF 4.20 / XGBoost 3.61), and best on the h11v02 stable historical pixels (2.74 / 2.55 / 2.36). Against the held-out 20% training labels it is **last on every tile** (best: baseline 48, RF 15, XGBoost 0; pooled MAE 7.25 / 7.32 / 8.38). G-LiHT lidar on h11v02 does not separate the models. The references disagree, so by the relative-difference rule no model is chosen yet. The historical product had no working aerosol screen, so references built from it favour whatever resembles it. The non-vegetated layer is still the baseline model.
- **Model choice (project lead, 2026-09-30):** RF and XGBoost tunedB are close enough to be treated as equivalent. RF is favoured because it can produce the per-pixel SD layer that the historical product has (item 8 below, still to be built).
- **Code:** the Python changes are **committed on local branch `mcarrol2/c-python-parity-fixes`** (6 commits off `origin/develop` `b83e302`) in `Git_delivery/modis_vcf`. **Pushed to GitHub 2026-09-30** (`335ceb2`; pull request into `develop` still to open). The cluster has no GitHub credentials (`Git_Delivery_Clone_modis_vcf.md`). `Python_code/modis_vcf` matches the branch. **Local commit `2778628` (2026-09-30, not pushed):** the non-vegetated default predictors in `rfFitFromPge61.py` are now C's 21 bare attributes (`NonVegetated_Model_Predictors_C.md`). A second local commit refreshes `docs/reports` with all the reports as of 2026-09-30 ~21:00, with the same redaction of other users' usernames. **The branch is 2 commits ahead of GitHub; the project lead pushes.** **Update 2026-10-05: all of it is pushed** (`0959749`; verified 2026-10-05 with `git ls-remote origin`).
- **Compute:** forest201-205 as before, plus **ngaproc201 and ngaproc202** (24 CPUs, 225 GB each, from 2026-09-30). On those two, ssh needs `-o CheckHostIP=no` (a stale IP entry in `/etc/ssh/ssh_known_hosts`), and singularity needs `module load singularity/4.3.1`. The existing launchers call `singularity` directly and need that line before they will run there.

## To-do list (updated 2026-10-01 evening, after the project lead's decisions)

**Decided 2026-10-01:**
- SD layers: approved.
- RF for non-vegetated: approved.
- Tree cover = MEDIAN of the 30 RF trees (greater spread; the historical product is seen as compressed toward the mean).
- Non-vegetated = MEAN, provisional.
- Delivery plan approved (`Delivery_SD_RF_Median_Plan.md`).

**In progress**
1. **[DONE 2026-10-01: all 63 tiles pass; `Delivery_SD_RF_Median_Results.md`]** **Delivery of the SD layers, RF non-vegetated, and MEDIAN tree cover** (`Delivery_SD_RF_Median_Plan.md`).
   - Patch and unit tests done (5/5) in `FullRuns/delivery_sd_rf_median_20261001/code`.
   - **Step 0 passed 2026-10-01 19:43 on ngaproc201** (`logs/step0_check_h08v05.json`):
     - tree cover equals the aggregation run's MEDIAN map on every pixel;
     - both SDs equal the run's SD;
     - non-vegetated equals min(MEAN bare, 100 − MEDIAN tree);
     - water/out-of-projection codes are unchanged;
     - layers sum to 100 on all 17,991,862 land pixels, with no negatives.
     - The cap changes 228 pixels (mean reduction 7.3), against 1,266 with MEAN tree cover: MEDIAN lowers tree cover where bare is high.
     - Run time 6 min 23 s; **peak memory 42.6 GB per process**, so at most 3 workers per ngaproc node and 2 per forest node would be too many.
     - Reviewed and approved by the project lead.
   - **Step 2 passed 19:49:** all 5 layers from the branch checkout are identical to step 0.
   - **Committed locally** (no push; the branch is now 7 ahead): `6684b88` member pass and SD, `f2d9190` CLI options, `92c7edc` unit tests, `615557d` README.
   - **63-tile run launched 19:53, fully detached** (`FullRuns/delivery_sd_rf_median_20261001`):
     - ngaproc201/202, 3 workers each;
     - frozen copy of commit `615557d` in `code_commit_615557d/`;
     - each tile is checked against the aggregation run (`scripts/check0.py`);
     - summary in `out/summary63.md`; done marker `status/ALL.done`.
     - Each process loads both RF models (14.6 + 13.3 GB on disk), which accounts for most of its ~42 GB.
   - Still to do: refresh the `docs/reports` snapshot once the run's results are in.

**Overnight 2026-10-01/02: plans 2-5 APPROVED by the project lead (about 20:15), with automatic step-0 gates (a failure stops that run with a `.failed` marker)**
2. **[DONE; ROUND ADOPTED 2026-10-02]** **Rounding instead of truncating the value cast** (`Rounding_vs_Truncation_Plan.md`). 63-tile rounded products, compared with the delivered (truncated) maps against C6, C6.1, the C code, lidar, the stable set and held-out labels. About 1.5 h on ngaproc. The expected MAE effect is a tie, so the choice is one of principle.
3. **[DONE 2026-10-01; `H16v01_Delivered_Product_Check_Results.md`]** **h16v01 check of the delivered product** (`H16v01_Delivered_Product_Check_Plan.md`). Read-only, about 10 min, after the delivery run.
4. **[DONE; ADOPTED 2026-10-02 for 6 Greenland tiles, Iceland excluded; commit `13773db`]** **Greenland/Iceland zenith-only fix on the delivered product, plan v2** (`Greenland_Iceland_Scoped_Fix_Plan_v2.md`). The MOD44C files are now in `/css/.../dev/` for 6 tiles; h15v03 has none and is out. Gate on the polar-night tile h17v00. About 3-4 h.
5. **[DONE 2026-10-01; `Bare_C_Regions_Investigation_Results.md`. Decided 2026-10-02: high-latitude rule on v00 tiles only. Still open: the rest of the C code's high-latitude mask, the Sahara stretch, and the v6.0.3-vs-C6 conflict]** **What the new RF bare layer does in the C code's two special regions** (high latitude and the Sahara stretch; `Bare_C_Regions_Investigation_Plan.md`). Describe only, to decide whether anything is needed. About 45 min, plus the Greenland tiles once item 4 has them.

**Next up after the overnight runs** (ranking from the project lead; not to be run yet)
6. **Cloudy regions, single-tile h12v09 test bed** (`Single_Tile_h12v09_Scenario_Plan.md`).
7. **Retrain on V4r metrics** (h12v09; `V4r_Darkest_Red_h12v09_Results.md`).
   - **2026-10-02: items 6 and 7 combined in `Cloudy_Pixels_Continuation_Plan.md`; revised after the project lead's decisions (no filling from neighbours or other periods; test down-weighting; all 63 tiles; delivered 64 predictors; forest201-205; no C6 check; no interim warning). Candidates BASE / BASE-C (existing model N) / BASE-W / V4R; not yet approved to run.**
   - **2026-10-02 15:38: run DONE** (`FullRuns/cloudy_v4r_weights_20261002`; restarted twice: a too-strict alignment check on h12v01, then a missing Mark_training bind for the tables). Figures: `figures_cloudy_v4r_weights_20261002/`. First look and **the project lead's review notes** are in `Cloudy_Pixels_Continuation_Results.md`: C6 is a comparison, not truth (also cloud-affected), so V4R's increases are not to be read as over-prediction; the sharp decreases need investigating. No decision; full write-up to do. New read-only check on h12v09: the delivered product (MEDIAN, rounded) does not fix the cloudy pixels. Dense forest still falls from +3.6 against C6 at 4 fallback periods to -13.1 at 8, and the held-out MAE at 5+ periods is 9.8 (bias -8.0). The tree SD flags them: 69% of drop pixels are in the top SD decile, against 4.5% of controls.
   - **2026-10-03: follow-up experiments, all FULLY EXPERIMENTAL, nothing adopted** (names: `Product_Versions_and_Work_Map.md`):
     - `V4r_Cloud_Blend_Results.md` (read its correction section first): v1.0.0 on V4r composites, and blends; the pooled label result was dominated by snowy h10v04; without it, V4r raises under-predicted dense pixels at 5+ cloud periods.
     - `Snow_vs_Cloud_V4r_Trigger_Check.md`: northern V4r firing is about half low sun (removed by a zenith test); the MOD44C snow bit is partial.
     - `QA_Bits_Dense_vs_Open_Check.md`: no QA bit separates open from dense in cloudy pixels; v1.0.0 already ranks them (AUC 0.99).
     - `Trees_and_Counts_2x2_Results.md`: 60 trees -0.119 held-out MAE (57/63 tiles better); cloud counts raise dense cloudy pixels slightly without raising open land.
     - `Gated_V4r_Blend_63Tile_Results.md`: cloud x cover x snow gated blend on 63 tiles. Gates work; held-out dense labels +0.3 (n.s.); h12v09 drop pixels raised +8.6 to +9.7 (C6 gap halved, C6.1 improved, C code worse); control unchanged.
     - **2026-10-03: the gated blend's main configuration is named v1.0.1** (experimental, not adopted).
     - **Open for the project lead:** whether to pursue the gated blend (e.g. weight 1 at 5+ periods), T60, or the counts; the h10v04 winter firing; non-vegetated.

**Other open items**
8. **Push: DONE 2026-10-05** (verified 2026-10-05 with `git ls-remote origin`). **Open: the pull request** into `develop`. *Original item:* **Push the branch and open the pull request** into `develop` for rlgill (`Git_Delivery_Clone_modis_vcf.md`, "To push"). It is 9 commits ahead of GitHub as of 2026-10-02. The `docs/reports` snapshot is deferred (project lead, 2026-10-02: do it later, before the push). **Reminder: do this at the start of the next session.**
9. **Rerun the Monte Carlo predictor selection with all of the new changes to metrics and composites** (bit-15 QA composites, zenith-only NDVI on h16v01), with RF as the model, for **both tree cover and non-vegetated**, and a run-to-run (seed) baseline. No lidar evaluation. The old 2026-09-20 324-metric rasters (~1.2 TB) are deleted at the end. Runs on **ngaproc201/202 only**. Plan: `Monte_Carlo_Rerun_Plan.md` (revised 2026-10-02 with the project lead's decisions: results go to the project lead for review, 63-tile maps if the tile maps complete without error; about 14-19.5 h).
   - **2026-10-02 08:23-19:37: run DONE** (`FullRuns/mc_rerun_20261002`, ngaproc201/202, about 11.25 h; all gates and acceptance tests passed, no failures). Results: `Monte_Carlo_Rerun_Results.md`. **Neither layer's MC set beats C's predictors on held-out labels** (seed yardstick about 0.01-0.02): tree MC top 64 +0.28 MAE (worse in every bin and region; on 63 tiles further from the C code, C6 and C6.1), top 20 +1.49; non-vegetated MC top 64 +0.11 (mixed against references), top 21 +2.15. Top 64 sets are ties under the 1-point rule. The tree ranking reproduces 2026-09-20 (Spearman 0.97, 56/64 shared) and still shares only 19/64 with C's set (bare: 1/21). **Awaiting the project lead: keep or change each layer's predictor set, and go-ahead for step 8 given and done 2026-10-03 (1,076.8 GB of old 2026-09-20 324-metric rasters deleted; list in `mc_rerun_20261002/status/delete_old_metrics324.list`).**
   - **DECIDED 2026-10-04 evening (project lead): keep the C code's predictor sets for both layers** (tree 64, non-vegetated 21).
10. **DECIDED 2026-10-04 (project lead): non-vegetated uses MEDIAN, as tree cover does** (evidence: `Bare_Mechanisms_Results.md`, held-out 3.77 vs 4.15 MAE). Previously: **Revisit later:** the non-vegetated method (MEAN is provisional; MEDIAN is better on held-out labels but worse against C6/C6.1; see `RF_Aggregation_Maps_63Tile_Results.md`, "Non-vegetated detail").
11. **`Quality` and `Cloud` layers and a single HDF product file** (to do; `Quality_Cloud_Layers_and_HDF_Output_Plan.md`). Writer: `pyhdf` or a C packer. Format: HDF-EOS2 or HDF5 (HDF-EOS2 recommended for C7; confirm with MODAPS / LP DAAC). uint16 SD storage comes with it.
12. **Run more years with per-year training and models** on the 63 training tiles: for each year, build and save the training set, fit that year's own tree and non-vegetated models (same settings and predictors as 2019), and make that year's product. No lidar testing. Keeps the option of a later test of one pooled multi-year model against per-year models (held-out locations kept the same in every year). Complete locally: 2000, 2004, 2009, 2010, 2014. 2001, 2005, 2015 and 2023 each need DOY 001-049 of the next year from the private repository. Plan: `Additional_Years_Plan.md` (revised 2026-10-02; clean-up of intermediates accepted; not yet approved to run; about 16-20 h for the 5 years; about 2.3 TB with the suggested clean-up). Suggested order: after the Monte Carlo decision (item 9).
13. **Decided 2026-10-04 (project lead): keep 60 trees for both tree cover and non-vegetated.** Tree cover evidence: `Trees60_then_MC64_Gated_Results.md` (A −0.118 MAE vs v1.0.1, 57/63 tiles). Non-vegetated at 60 trees has not been fitted or scored yet; a 60-tree non-vegetated model is still to be built. Not yet a version; naming is separate. First non-vegetated figures (v1.0.0 and MC top 64 vs C6, two rows per tile): `NonVeg_Figures_v100_MC64_Results.md`.
14. **Reminder (set 2026-10-04): deeper dive into the MC 64 runs** (plan written 2026-10-04 evening: `MC64_Deep_Dive_Plan.md`, awaiting approval) (`Monte_Carlo_Rerun_Results.md`, `Trees60_then_MC64_Gated_Results.md`): why MC64 is consistently worse (+0.28 MAE in both settings, worst on dense cloudy labels), by predictor, region and cover bin. Not planned yet.
15. **Non-vegetated where C6 is 100 and the new models are not** (2026-10-04, project lead; plan APPROVED 2026-10-04, step 0 (60-tree bare) DONE 2026-10-04 09:55 (`Bare60_Step0_Results.md`: held-out −0.046 MAE, 19/20 tiles; no change where C6 = 100 — mean 95.4, 38% exactly 100, 12% < 90; MEAN aggregation now the lead suspect); steps 2-3 DONE 2026-10-04 10:55 (`Bare_Mechanisms_Results.md`: where C6 = 100 the labels side with the new model (MAE 1.95 vs C6 6.04); MEDIAN beats MEAN on held-out (3.77 vs 4.15); sum cap no factor; ice inputs never pass QA (all fallback); h16v02 edge = polar-night input boundary in DOY 353): `Bare_Where_C6_Is_100_Investigation_Plan.md`, bare to-do B1-B11 incl. additional predictors; external data on ADAPT only, no NGA, 69 tiles only). Greenland ice sheet below 100 (incl. h16v02 straight edge), desert drops on h20v06/h21v06, Australia 9-14 lower than C6; MEAN/sum-cap effect; pixel traces and QA; inventory of the C code's post-model rules (v7 high-latitude, Sahara stretch, sum cap, v6.0.3 conflict); external maps as comparison only (ESA WorldCover, MCD12Q1, C6 series local; CGLS-LC100 2019, ice masks need download approval). Read-only, about 1 day plus 3-4 h for the 60-tree bare model.
16. **2026-10-04 follow-ups to item 15** (project lead):
    - **Step 1, candidate product:** 60 trees and MEDIAN for both layers, 69 tiles. DONE 13:48 (`Candidate_60Trees_Median_Results.md`):
      held-out better for both layers (bare 4.18 -> 3.76, tree 6.70 -> 6.58); tree better against C6/C6.1/C code;
      bare worse against all three (about +1.7, lower in unlabelled boreal/temperate tiles); C6 = 100 at exactly 100 38% -> 79%; decision pending.
    - **Step 2, MOD44C bit 15:** settled from existing reports. Bit 15 is PGE72 QC and requires "not snow", so ice fails it by construction.
      Layout skill: `~/.claude/skills/mod44c-state-bits`.
    - **Step 3, polar-night / low-sun fill (fill72 experiment):** DONE 16:00, FAILS the pass rule, not adopted (`Fallback_Fill72_Results.md`).
      MEDIAN alone already removes the h16v02 edge and most ice bare < 100; the fill raises tree cover on the ice.
      Affects 13 tiles at v <= 02, only the Nov-Feb periods.
    - **Step 4, C code rule inventory:** `C_Code_Bare_Rules_Inventory.md`. C6 2019 = PGE61 6.0.4 (narrowed high-latitude rule);
      C6.1 = 6.1.4 (all v < 3). This resolves the v6.0.3 conflict.
17. **DECIDED 2026-10-04 16:13 (project lead): adopt MEDIAN for non-vegetated**, after the labels and the ESA WorldCover quick look
    (`Candidate_60Trees_Median_Results.md`). With the earlier decisions the candidate product is the v1.0.0 recipe with:
    - tree T60 (60 trees, MEDIAN);
    - non-vegetated B60S0 (60 trees, MEDIAN);
    - everything else unchanged.
    Maps: `FullRuns/cand60med_20261004/out/maps/CAND` (69 tiles, no SD). Version name: **v1.1.0** (confirmed 2026-10-04).
18. **2026-10-04 evening:**
    - **v1.1.0 confirmed** by the project lead. Branch `mcarrol2/v1.1.0` and tag `v1.1.0` exist in the Git_delivery clone; the docs
      snapshot is committed. **Push pending (project lead): attempted 2026-10-04 evening and failed; `git ls-remote` shows
      nothing reached GitHub (branch still `335ceb2`, no tags). To be pushed from the office on 2026-10-05.** **DONE 2026-10-05** (verified 2026-10-05 with `git ls-remote origin`).
    - **SD maps:** `FullRuns/v110_sd_20261004` DONE 17:55: 69 tiles x 5 layers; the 3 cover layers identical to the candidate maps on
      all 69 tiles; checksums in `manifests/v1.1.0_maps_sha256.txt`.
    - **Snow flag and the ice margin:** the summer snow flag (MOD44C bit 5) separates ice from margin almost perfectly on h16v01
      (`SnowFlag_IceMargin_Results.md`). The h16v02 90-99 band is flagged ice: the remaining shortfall. A summer-snow ice rule is a
      possible next test (not adopted).
19. **DECIDED 2026-10-04 evening (project lead), on the 7 open questions:**
    1. **WV2 report stays out of the public snapshot** (NGA rule). `WV2_Reference_Scenes_Provenance.md` stays on ADAPT only.
    2. **Cloud blend "A" (v1.0.1) is not in v1.1.0; the code stays as the experimental v1.0.1. STILL NEEDS INVESTIGATION:** overall it
       ties plain 60 trees on held-out (−0.118 vs −0.119 against v1.0.0); in the cloudiest pixels it raises tree cover about 2.2 points
       and is better for open labels but worse for dense ones. Why it splits by cover is open.
    3. **Monte Carlo 64 predictor set dropped from the product. STILL NEEDS INVESTIGATION:** worse in both tests (+0.28 MAE); the deep
       dive (item 14) stays on the list.
    4. **No post-model bare rules** (no Sahara stretch, no C code high-latitude box): the labels do not support them where they exist.
       The v00 rule (2026-10-02) is unchanged.
    5. **Summer-snow ice rule (bare 100, tree 0 where summer snow share >= 0.25): test it**, starting with a cheap screen of where it
       triggers outside Greenland and whether tree cover is hurt there; full tiles only if the screen is clean.
    6. **h16v01 training inputs: leave as is for v1.1.0** (trained on the original NDVI, predicted with the Greenland-corrected NDVI);
       revisit at the next retrain.
    7. **Order of work:** (a) in-file fill values (−28672) entering the composites (`Fallback_Solz_LE0_Plan.md`, side finding);
       (b) other years with the v1.1.0 settings (item 12); (c) `Quality`/`Cloud` layers and HDF output (item 11).
    - **(a) first screen DONE 2026-10-04 evening** (`Fill_Value_Screen_Results.md`, 5 tiles, read-only): band 5 fill passes QA
      (83% on h11v04/h16v02) but no product predictor set uses band 5; fill in bands 1-4, 6, 7 never passes QA and appears only on
      high-latitude tiles (Nov-Jan, ice), entering through the fallback mean. Next (awaiting OK): composite-level count on the v <= 02
      tiles, then decide the fix (treat −28672 as missing in every band).
    - **Plan written 2026-10-04 evening: `Composite_Unusable_Inputs_Plan.md`** (awaiting decisions D1-D12): one rule for every kind of
      unusable input (missing file, polar-night zeros, −28672 fill) before other years; full history of the zenith/fallback work kept
      in its section 3. Step 1 is a read-only composite-level count (about 1 h) with a gate (D10).
    - **Output data type (question, 2026-10-04 evening):** the v1.1.0 GeoTIFFs are **Int16** (LZW; `VcfPredict.py` casts to int16,
      `Utils.NUMPY_DTYPE`), not uint8; no NoData value is set. Cover layers hold 0-100 plus 200/250 and would fit uint8 like the
      historical MOD44B; SD layers are percent x 100 (e.g. max 3632 on h08v04) and need 16 bits. Changing the storage belongs to
      item 11 (Quality/Cloud + HDF, with uint16 SD). Not changed.
      **DECIDED 2026-10-04 evening (project lead): the output data type should be 8-bit unsigned** (uint8), as in the historical
      MOD44B (`Percent_Tree_Cover`, `Percent_NonTree_Vegetation`, `Percent_NonVegetated`: uint8, valid 0-100, water 200, _FillValue 253;
      `Quality_Cloud_Layers_and_HDF_Output_Plan.md`). Open: the SD layers (historical uint16 with scale_factor 0.01; uint8 would need
      whole-percent SD), and whether to change the code now or with item 11.
      **Then decided (project lead): SD layers uint16 (round(100 x SD)), as historical; change now, before the push.**
      **DONE 18:25: commit `0298e4e` on `mcarrol2/v1.1.0`** ("Write the percent layers as uint8 and the SD layers as uint16"):
      `VcfPredict._write` now writes its own GeoTIFFs (modis_water's `writeRaster` hard-codes Int16): percent layers uint8 with
      nodata 253, SD layers uint16, LZW; values that do not fit are refused. New tests `model/tests/test_VcfPredict_write.py`
      (8/8 with the member tests). **One-tile check** (`FullRuns/uint8_check_20261004`, forest201, h08v04, v1.1.0 models and flags):
      all 5 layers identical pixel for pixel to `v110_sd_20261004`; percent files 26-27% smaller. The tag `v1.1.0` stays on
      `1ac9bc7` (Int16 output) unless the project lead moves it. Side notes: (1) **the output GeoTIFFs have no geotransform**
      (upper left 0, 0; before and after this change; modis_water's writer never set one) - not fixed, for the project lead;
      (2) the workspace copy `Python_code/modis_vcf` is older than the SD-layer commits and no longer matches the branch.
20. **2026-10-04 evening (project lead), answers to the status check:**
    - **Push:** from the office on 2026-10-05 (unchanged).
    - **Tag `v1.1.0` should mark the uint8 output.** Moving the tag was blocked by the session's permission guard; the project lead moves it
      (`git tag -f -a v1.1.0 -m "v1.1.0: 60 trees and MEDIAN for both layers (project lead, 2026-10-04)" 920d06f`). It is still on `1ac9bc7`. **DONE 2026-10-05: on `920d06f`, pushed.**
    - **Geotransform = MODIS sinusoidal: DONE, commit `920d06f`** on `mcarrol2/v1.1.0` ("Write the MODIS sinusoidal grid into the output
      GeoTIFFs"). `VcfPredict._write` sets the tile's geotransform from its hXXvYY ID and the MODIS sphere (R = 6371007.181 m; the old
      string said WGS84 ellipsoid). Matches the C6.1 MOD44B granules for h08v04, h16v01, h31v11 (corner and pixel size within 1 micrometre,
      same SRS). New test `testModisSinusoidalGrid`; 9/9 writer and member tests pass. Existing v1.1.0 maps were not rewritten.
    - **`Composite_Unusable_Inputs_Plan.md` D1-D12:** explained in the session; still awaiting the decisions.
    - **Summer-snow ice rule: cheap screen DONE 19:31** (`SummerSnow_Rule_Screen_Results.md`, `FullRuns/summersnow_screen_20261004`,
      69 tiles, no failures). **Not clean as specified:** June-August everywhere fires on h12v12 (Patagonian Andes, southern winter;
      about 2.5 M tree points). The local-summer version hardly touches trees, but the few bare labels outside Greenland side with
      v1.1.0. The gain is a few bare points on h16v02. No full-tile run. Decision pending (drop / Greenland only / 0.5 everywhere).
    - **Predictor sets:** keep the C code's sets (item 9).
    - **Other years (12), Quality/Cloud + HDF (11):** to be decided.
    - **Synced 2026-10-04 evening:** the workspace copy `Python_code/modis_vcf` = `920d06f` (the notebook `view/editLabeledTraining.ipynb`
      keeps its local edits); the docs snapshot on `mcarrol2/v1.1.0` is refreshed (see `Git_Delivery_Clone_modis_vcf.md`).
    - **Overnight 2026-10-04 (approved): EXPERIMENTAL step 1 of `Composite_Unusable_Inputs_Plan.md`** (read-only composite-level count),
      `FullRuns/EXP_unusable_inputs_count_20261004`, forest201/203, 18 tiles, code @ 920d06f, gate on h12v09. Tagged EXPERIMENTAL
      (directory prefix `EXP_`, `EXPERIMENTAL.md`) so that it can be isolated; it changes no product file.
    - **`MC64_Deep_Dive_Plan.md` written** (to-do 14; not approved; nothing run).
    - **ALL DECISIONS FOR 2026-10-05 are listed in `Decisions_For_20261005.md`** (push and tag, rewriting the on-disk maps, D1-D12 + the D10
      threshold, the summer-snow rule, other years, HDF output).
21. **Figures of the code's progression (added 2026-10-05, project lead; to do, not planned yet).** A figure set showing how the
    product changed from the starting point to the final version, whatever that turns out to be: the C code as first compiled and
    run here, and rlgill's incomplete Python (v0.0.0-pre.1), through v0.0.0-pre.2, v0.0.0, v1.0.0 and v1.1.0, to the final version.
    Versions and their run folders are defined in `Product_Versions_and_Work_Map.md`. Make it when the final version is settled;
    write a short plan (tiles, layers, references, which runs hold each version's maps) before making the figures.

    **Done for F13 2026-10-07 (project lead: "figures are good for now"):** h12v02 and h09v05 tree and non-vegetated, from the first Python maps to v1.2.0, against
    the C code and C6, plus h16v02 non-vegetated v1.1.0 to v1.2.0: `AICoE_10142026/fig7_*.png`, `make_tile_evolution.py`,
    `tile_evolution_stats.csv`.
22. **If time permits (added 2026-10-07, project lead): run rlgill's Python as found (`develop` `b83e302`, v0.0.0-pre.1) on h12v02 or
    h09v05**, to show its real errors in the progression figures. No map was ever made with that code (GPU cuML, incomplete), so fig7
    uses the first Python maps (`Metrics/PyTrackB_2tile`, 2026-09-18) in its place. On a CPU node, with cuML swapped for scikit-learn
    only where the code cannot run otherwise; if it fails, the failure is the result. About 1-3 h, detached. Write a short plan first.

23. **Revisit later (added 2026-10-07, project lead): the decision not to adopt the cloud variants (V1-V4C, V4R, BLEND-2, v1.0.1).**
    Goal: better dense-forest tree cover in cloudy areas without really hurting predictions elsewhere. Review findings and a suggested
    first step (score on dense labels that v1.0.0 under-predicts, plus open-land and clear-pixel checks) are under item E11 in
    `Decisions_For_20261005.md`. Not scheduled.

24. **NOT ADOPTED, to be revisited (2026-10-07, project lead): cold-ice rule for the ice-sheet interior** (`Cold_Ice_Rule_Check_Results.md`,
    run `FullRuns/EXP_cold_ice_rule_check_20261007`). Works on the interior by both ice masks; a rule; redundant with v1.2.0 on h16v02; v00
    low-sun land (non-tree vegetation up to 84, judged artefacts) still unsolved. Decisions item 15.

**Done (from the earlier lists):**
- NEON lidar evaluation (2026-09-30);
- stable-set scoring (2026-09-30);
- non-vegetated with RF/XGBoost (2026-10-01; RF chosen);
- model choice (RF; 2026-09-30/10-01);
- SD layer (approved 2026-10-01, delivered in item 1);
- one-tile sum-cap check with MEDIAN tree (item 1, step 0);
- the Greenland/Iceland data block, now that the files have arrived (item 4).

**Later, if time permits:**
- **The v00 high-latitude rule** (adopted 2026-10-02: on every v00 tile, land is set to tree cover 0, non-vegetated 100, non-tree vegetation 0). It rests on the knowledge that there are no trees and very likely no vegetation north of about 80 N, not on an evaluation. It could be investigated further, for example:
  - whether any v00 land has real vegetation;
  - whether periods with no input (filled with 10000 in every band) should instead be no-data in the metrics, so the model could be trusted there;
  - whether the rule should extend to other high-latitude tiles (the C code's v < 3, h < 18 mask would overwrite real cover on h11v02, h12v01, h12v02 and h13v01; `Bare_C_Regions_Investigation_Results.md`).
## Standing conditions

- Nothing in rlgill's checkouts (pge61, pge72, `modis_vcf`, `core`) is edited, committed or pushed. The branch lives in our own clone and is pushed by the project lead.
- Plans are written and approved before anything runs. Long runs are fully detached (`setsid nohup`, status and done markers). Model choices rest on relative differences, and a difference of about 1 point MAE or less counts as a tie.
- New work goes into new folders; nothing already written is overwritten.

## Document map

**Start with `Project_Summary_and_Conclusions.md`**: one page on what was done, the results, the open items, what changed in code and what to discuss with the repository owner.

**`Historical_C6_C61_Four_Way_Comparison_Plan.md` / `_Results.md`** (2026-10-01): C6, C6.1, the C code and Python RF on 63 tiles; C6.1 high-latitude rule; 4 cover bins; continent x latitude band.

**`C_to_Python_Conversion_Assessment.md`** (2026-10-01): the problem statement, the challenges of building the C toolkits and code, the Python milestones to 2026-09-30 (excluding the h12v09 cloud variants), and the agreed conclusion that Python is comparable to the historical MOD44B, with its evidence and qualifications.

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
- `Cloudy_Fallback_Diagnosis_Plan.md`: why the Python models drop on cloud-fallback composites, how widespread it is, and QA information layers for RF (plan).
- `Cloudy_Fallback_Diagnosis_Results.md`: stage A results (steps 1, 2 and 4), 2026-09-30.
- `Single_Tile_h12v09_Scenario_Plan.md`: single-tile RF test bed for cloud-fallback scenarios (plan).
- `RF_Fallback_Counts_Results.md`: 63-tile RF with fallback and cloudy counts as predictors (done 2026-09-30): small consistent gain, no tile worse, drop reduced but not fixed.
- `Cloud_Composite_Alternatives_Plan.md`: plan to test less-cloudy compositing rules on h12v09 from MOD09GA/GQ (cloudiness only, no model; awaiting approval).
- `Stable_Set_Model_Eval_Plan.md`, `Stable_Set_Model_Eval_Results.md`: the models scored on that set (done 2026-09-30).
- `Stable_Pixel_Reference_Selection.md`: the 10,000-pixel stable reference set, which stands in for Stage 6.
- `Lidar_Reference_Coincidence_20260930.md`: G-LiHT on h11v02, the decision section, and the lidar coverage survey.
- `NEON_Lidar_Eval_Plan.md`, `NEON_Lidar_Eval_Results.md`: NEON lidar evaluation of RF (bit 15), 19 sites, done 2026-09-30.
- `C_vs_Historical_h12v09_20260930.md`: dense-forest check, and cloud-fallback drops in the eastern Amazon.
- `Git_Delivery_Clone_modis_vcf.md`: the local branch with the Python changes and how to push it.

**Overnight 2026-10-01/02 results:**
- `Delivery_SD_RF_Median_Results.md`: 63-tile delivery run (tree = MEDIAN, non-vegetated = MEAN, SD layers); all checks pass on all 63 tiles.
- `H16v01_Delivered_Product_Check_Results.md`: the delivered product on h16v01. Tree over-prediction gone; bare 9.6 below C6.
- `Greenland_Iceland_Scoped_Fix_v2_Results.md`: the zenith-only NDVI fix on 6 Greenland tiles and Iceland, using the delivered product. Meets the criterion for Greenland, not for Iceland.
- `Bare_C_Regions_Investigation_Results.md`: the RF bare layer in the C code's high-latitude and Sahara-stretch regions; options for the project lead; v6.0.3 conflict.
- `Rounding_vs_Truncation_Results.md`: rounding against truncation of the value cast. All ties; ROUND adopted 2026-10-02.
- `Monte_Carlo_Rerun_Plan.md`: plan to rerun the Monte Carlo predictor selection on the current inputs, with a seed baseline (to-do item 9).
- `Trees60_then_MC64_Gated_Plan.md` (2026-10-03, FULLY EXPERIMENTAL, DONE 2026-10-03 22:36, score rerun 2026-10-04; results `Trees60_then_MC64_Gated_Results.md`: **A (60 trees) −0.118 MAE vs v1.0.1, 57/63 tiles; B (MC64) +0.28 worse than A** — awaiting the project lead): v1.0.1 recipe with 60-tree models (A), then with the Monte Carlo 64 predictors (B); 63 tiles, held-out by bin, maps, drop pixels, figures; ~10-11 h on ngaproc.
- `figures_gated_v4r_63_20261003/` (2026-10-03, FULLY EXPERIMENTAL): **v1.0.1** tree cover (V4R gated blend, main) against C6, 63 tiles, same layout as `figures_final_round_v00_20261002/` (v1.0.0); 9 tiles change; README in the folder.
- `Gated_V4r_Blend_63Tile_Results.md` (2026-10-03, FULLY EXPERIMENTAL; **main configuration named v1.0.1**): gated V4r blend on 63 tiles. Gates work (open land protected, no seams, NDSI snow test cuts the h10v04 loss from -0.92 to -0.20); maps raise dense cloudy pixels about +4, but at independent dense labels the raise is small and not significant (+0.3 without h10v04, gap 7-9); the weighted count is not better. Plan: `Gated_V4r_Blend_63Tile_Plan.md`.
- `QA_Bits_Dense_vs_Open_Check.md` (2026-10-03, diagnostic): in cloud-caused fallback pixels no MOD44C QA bit or combination separates open (0-20) from dense (61+) labels usefully (single fields AUC <= 0.58; multi-field 0.77-0.87 via location/season proxies); v1.0.0's prediction already separates them (AUC 0.99). The cloud problem is dense values too low, not confusion.
- `Trees_and_Counts_2x2_Results.md` (2026-10-03, FULLY EXPERIMENTAL): 60 trees (T60) -0.119 held-out MAE, better on 57/63 tiles (seed noise ~0.01) but no raise of dense cloudy pixels; counts as predictors (C30) raise dense cloudy pixels (+0.6 at 5+ periods, same bin +1.5 points) without raising open land, but close little of the ~11-point gap; effects add (C60 -0.152). Plan: `Trees_and_Counts_2x2_Plan.md`.
- `Snow_vs_Cloud_V4r_Trigger_Check.md` (2026-10-03): h12v03, h24v03, h23v03, h11v03, h12v04 vs h12v09. Most northern V4r firing is low sun (all inputs zenith >= 65, flagged as cloud), not cloud; a zenith test removes it (h12v03 44% -> 21% of pixel-periods) and changes nothing in the tropics. The MOD44C snow bit (5) catches only a minority of snow.
- `V4r_Cloud_Blend_Results.md` (2026-10-03; read its correction section first: the pooled label result was dominated by h10v04, a snowy Rockies tile; without it, V4r raises under-predicted dense pixels at 5+ cloud periods on 11-12 of 14 tiles, consistent with the maps): with low sun excluded by a zenith test, the V4r blends do raise tree cover where clouds caused fallback (+4 to +5 at 5+ periods on maps), but the raise is broad: as large on open land, and at independent (Landsat, by-bin) dense-label pixels, where v1.0.0 is too low (44% a bin low at 5+), it does not rise. Cover-gated version (v1.0.0 > 20/30/40, ramp) removes the open-land raise but dense labels are still not raised. Side finding: test a 60-tree forest for v1.0.0.
- `V4r_Cloud_Blend_Plan.md` (2026-10-03, run the same day): v1.0.0 applied to V4r composites (V4r only where clouds caused the fallback) and cloud-gated blends with v1.0.0 (one model preferred); 7-tile maps, 63-tile held-out labels.
- `Product_Versions_and_Work_Map.md` (2026-10-03): version names (v0.0.0, v1.0.0), v1.0.0's full processing rules, the change list between them, and the map of experiments and validation.
- `Monte_Carlo_Rerun_Results.md` (2026-10-03): results of that rerun for tree cover and non-vegetated; MC sets do not beat C's; step 8 deletion done 2026-10-03.
- `WV2_Reference_Scenes_Provenance.md` (2026-10-02): 30 WorldView-2 scene parts over cloudy-forest tiles (`FullRuns/wv2_reference_scenes_20261002`, footprint index from `FullRuns/nga_wv02_m1bs_index_2019_20261002`), processed to ortho TOA and pan-sharpened products with a basic canopy assessment, plus AlphaEarth tests. Products good; cloud mask and canopy thresholds not yet robust, so no comparison with MODIS products yet. NGA data and derivatives stay on ADAPT.
- `Cloudy_Pixels_Continuation_Results.md`: first-look numbers and the project lead's review notes for `FullRuns/cloudy_v4r_weights_20261002`; its open questions were taken up in the 2026-10-03 follow-up reports (see to-do item 7); no separate full write-up of that run.
- `Additional_Years_Plan.md`: plan for per-year training, models and products for 2000, 2004, 2009, 2010 and 2014 on the 63 tiles, set up for a later pooled-model test (to-do item 12).
- `figures_final_round_v00_20261002/`: tree-cover figures of the final product (RF MEDIAN, rounded, Greenland zenith fix, v00 rule) against C6, for 69 tiles (the 63 plus the 6 new Greenland tiles). Each file is `<tile>_tree_cover_final_round_v00_20261002_vs_C6.png`, and the stats are in `tile_stats_final_round_v00_20261002.csv`. Made with a tagged copy of the 2026-09-20 figure script (`FullRuns/final_round_v00_20261002/scripts/make_tree_cover_figures_tagged.py`, which never replaces a file). Inputs: the final code for h08v05 and the 7 Greenland/Iceland tiles; the rounding run's ROUND maps for the other 62, which the final code reproduces exactly on h08v05. The 2026-09-20 set in `figures/` is unchanged.

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
15. **Per-pixel SD layer needed (to do; noted 2026-09-30).** The historical MOD44B has `Percent_Tree_Cover_SD` (and `Percent_NonVegetated_SD`): the SD across its 30 bagged models. The Python pipeline writes no SD layer. RF and the baseline (30 bootstrap trees) can produce the equivalent from per-tree predictions (`model.estimators_`) at almost no cost; it needs a small addition to `VcfPredict`, then a comparison with the historical SD on a few tiles (expect a different scale). XGBoost has no equivalent without bagging about 30 models. Because the project lead considers RF and XGBoost equivalent in accuracy, this favours RF. Not planned yet.

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
"First remind me to open the pull request into develop (to-do item 8; the push is done). Then read Reports/README_START_HERE.md (including its 'Resume here' section) and Reports/Project_Summary_and_Conclusions.md, and tell me the open items and what is not done. Do not execute anything without asking."

## Status history (newest first; kept word for word)

### "Waiting on the project lead" list, as it stood on 2026-10-01 before the evening decisions

1. Open the pull request into `develop` for rlgill (the branch is pushed) (`Git_Delivery_Clone_modis_vcf.md`, "To push").
2. **Done 2026-09-30:** NEON lidar evaluation of RF (bit 15) (`NEON_Lidar_Eval_Results.md`). RF ties the historical MOD44B (3 m MAE 20.4 vs 20.4). Every product and the labels compress the range against lidar (dense forest about -28).
3. Cloudy-region work (cloud-fallback composites): **stage A done 2026-09-30** (`Cloudy_Fallback_Diagnosis_Results.md`). C and Python use the same fallback rule; the current C code drops too, and more than RF at high fallback counts. Only the historical C6 product is less affected (it fell back about 1 period less at the drop pixels). About 14% of land outside the high latitudes has 3 or more fallback periods; in dense forest RF is biased about -7 at 5 or more. Side finding: Python `outCloud` is inverted (unused). Addendum: 2-input periods and h12v09 label distribution. **Next (project lead, 2026-09-30): single-tile h12v09 test bed instead of the literature search:** `Single_Tile_h12v09_Scenario_Plan.md` (awaiting approval).
4. Greenland/Iceland scoped fix: blocked on MOD44C files (`Greenland_Iceland_Scoped_Fix_Plan.md`).
5. The non-vegetated layer has not been tried with RF or XGBoost. C's 21 bare predictors, decoded (2026-09-30): `NonVegetated_Model_Predictors_C.md`. Our runs already use them through a predictor file, but the default list in `rfFitFromPge61.py` is still wrong and needs replacing.
6. Score the models on the 10,000-pixel stable set (h11v04, h18v03, h19v09). It was selected but never scored; only h11v02 stable pixels were. **Done 2026-09-30** (`Stable_Set_Model_Eval_Results.md`): XGBoost 4.81, RF 5.13, baseline 5.72 MAE. The ranking matches the other historical-based references, but every pooled gap is a tie (at most 1 point).
7. Model choice (baseline, RF or XGBoost): the references disagree (see above). The 3-tile stable set agrees with the other historical-based references (XGBoost first, gaps are ties). An independent reference is needed; NEON lidar (item 2) is the one available.
8. **Per-pixel SD layer needed (to do; noted 2026-09-30).** The historical MOD44B has `Percent_Tree_Cover_SD` (and `Percent_NonVegetated_SD`): the SD across its 30 bagged models. The Python pipeline writes no SD layer. RF and the baseline (30 bootstrap trees) can produce the equivalent from per-tree predictions (`model.estimators_`) at almost no cost; it needs a small addition to `VcfPredict`, then a comparison with the historical SD on a few tiles (expect a different scale). XGBoost has no equivalent without bagging about 30 models. Because the project lead considers RF and XGBoost equivalent in accuracy, this favours RF. **Update 2026-10-01: planned and tested on 5 tiles** (`SD_Layer_Plan.md`, `SD_Layer_Results.md`); delivery code and 63-tile SD wait on the project lead's decision after the RF aggregation comparisons.
9. **`Quality` and `Cloud` layers and a single HDF product file (to do; plan written 2026-10-01, not yet prioritized):** `Quality_Cloud_Layers_and_HDF_Output_Plan.md`. It covers the last 2 of the 7 historical SDSs, the `outCloud` fix, and one granule like the original with DEFLATE on every SDS. Open questions: `pyhdf` or a C packer, and HDF-EOS2 or HDF5 (recommended: HDF-EOS2 for C7, after confirming with MODAPS / LP DAAC). **The project lead will set its priority after the RF aggregation comparisons, together with the SD, bare RF and final-pixel (mean / mid-20 / median) decisions.**

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
- **Other users share these nodes.** On 2026-09-20 user `another user` used about 800% CPU on several of them for hours; check `uptime` and free memory before launching, and do not assume the load is yours.
- Shell pitfalls seen: `noclobber` makes a retried `>` redirect onto an existing log fail at once (use `>>` or a new name); a claim directory left by a stopped runner blocks its tile until removed; `cp` is aliased to interactive (use `\cp -pf`); `ssh` prints a login banner on standard output, so filter output; `pgrep -f` matches the ssh command that ran it; do not edit a `.sh` launcher while it is running.
- Status: `bash $RUN/code/validation/run_status.sh` reports **only the 63-tile run**. For 2026-09-20 work list `$RUN/status/`: `mc_*` (Monte Carlo), `predict_<label>` and `variant_compare` (map check), `figures_*`, `hist_compare_*`, `p0_*` (table build).

**Where things are:** the Monte Carlo and its models in `$RUN/Python/monte_carlo_standalone/tree/`; the 324-metric rasters in `$RUN/Python/metrics_324/` (1.1 TB, kept on purpose) and tables in `$RUN/Python/training_324/`; the map check in `$RUN/Compare/mc_model_prediction/`; all scripts in `$RUN/code/validation/`. `$RUN` = `/explore/nobackup/people/mcarrol2/MOD44B_VCF_Development/FullRuns/full63_c_vs_py_20260919-0806`.

