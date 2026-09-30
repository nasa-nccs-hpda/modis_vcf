# Full 63-tile C vs Python run: launch record and how to check it

Launched **2026-09-19 12:42:44**, fully detached on five nodes. Run root:
`/explore/nobackup/people/mcarrol2/MOD44B_VCF_Development/FullRuns/full63_c_vs_py_20260919-0806/` (called `$RUN` below).
Written for anyone picking this up, including a new session. (Pre-launch estimate: about 10 hours, range 9-13. The run finished sooner; see the
final status below.)

## Final status (2026-09-20)

**Finished.** Compute ended at 19:28 on 2026-09-19 (the last C B3 tile; 6.75 h after launch); the comparison ran 19:40-19:55. All 63 tiles have a C B3 HDF and a
Python prediction and were compared: `FullRun_Results_C_vs_Python_63tiles.md`.

| Stage | When (2026-09-19) | Duration |
|---|---|---|
| Python metrics (63 tiles, 4 per node on forest203-205) | 12:43 to 15:19 | 2.6 h |
| C B1 (forest201/202, 4 per node) | 12:43 to 13:48 | 66 min |
| C B2 (30 pooled models) | 13:49 to 14:29 | 41 min |
| C B3 (63 tiles, 8 at a time) | 14:29 to 19:28 | 5.0 h |
| Python training data (83 runs), split | 15:19 to 15:50, then 2 min | 31 min |
| Python fits (12 jobs each, on two nodes) | 15:53 to 17:14 (tree), 16:10 (bare) | 81 min and 17 min |
| Python predict (63 tiles, three nodes, models loaded once each) | 17:14 to 18:44 | 90 min |
| Comparison | 19:40 to 19:55 | 15 min |

Model sizes: tree 13.5 GB, bare 11.9 GB. No memory-watchdog kills; all 63 metrics tiles and all 83 training runs succeeded.

**Two things did not go through the automatic chain.** (1) Three C B3 tiles (h12v01, h12v05, h13v01) were marked `failed` because my success rule
required an HDF larger than 10 MB. All three exited with rc=0 and produced valid 7-layer HDFs of 3.9, 4.7 and 0.9 MB (the Sep 11 run's HDFs for the
same tiles were 4.3, 4.5 and 0.9 MB); the size rule is wrong for mostly-ocean tiles. That stopped the driver's own compare step, so the comparison
was run separately (`code/validation/run_compare_only.sh`, which does not read the markers). **Resolved 2026-09-20:** the rule in `code/c_scripts/run_c_pipeline.py` now requires more than 100 KB (the smallest valid tile HDF is about 0.9 MB; original saved in
`C/production/build_context/markers/_reclassified_20260920/`), and the three markers were reclassified from `failed` to `ok`, with the original `.failed` files kept in that same
folder. `run_status.sh` now reports B3 ok 63/63, failed 0. The status logs still contain the original "B3 ... FAILED" lines; they are history.
(2) The Python side finished all its phases; only the compare step was affected.

## What is running

| Node | Runs | Detail |
|---|---|---|
| forest201, forest202 | C chain, then the Python phases | `node_c_then_python.sh`: B1 (63 tiles) -> barrier -> B2 (30 pooled models) -> barrier -> B3 (63 tiles), 4 jobs per node, tiles claimed with `mkdir`; then joins the Python phases |
| forest203, forest204, forest205 | Python phases | `node_python.sh` -> `python_phases_node.sh`: metrics (4 tiles per node), then training data, split, fit, predict, compare |

Everything was started with `setsid nohup` through `ssh -n` (parent PID 1, no terminal), so it survives disconnects.
All work is shared through claims (`mkdir`) and marker files, so a restarted script resumes and never redoes finished items.

## Decisions in force (from the user)
63 tiles; Python trains on the `v5.0.3samp` labels (bare uses the legacy bare labels, which exist for only 20 of the 63
tiles); Python predictors mirror all of C's 64 tree and 21 bare attributes (implementation B, `CMirrorMetrics`); the
Python NDVI difference is accepted for now; only the unsorted values and the 19 C-mirror families are computed (`-m`);
C executables rebuilt from the current source (option A); intermediates are NOT deleted; nothing outside `$RUN` is written.

## Check progress
`bash $RUN/code/validation/run_status.sh` (read-only): counts of finished items per stage, per-node load, memory and running
jobs, watchdog kills, disk. Per-node logs: `$RUN/status/node_<host>.status` and `.out`, `python_phases_<host>.status`,
`python_metrics_<host>.status`; C: `$RUN/C/production/build_context/status/<host>.status` and `pipeline_logs/`.
A marker `*.failed` in `$RUN/status` (Python) or `$RUN/C/production/build_context/markers` (C) means an item failed and the
chain stops on purpose.

## If something fails
Read the item's log, fix the cause, move the item's partial output aside (never delete), remove its claim directory
(`Python/status/claims/<tile>`, `Python/status/claims_phase/<name>`, or `C/production/build_context/claims/<name>`) and
its `.failed` marker, then rerun that node's `node_*.sh` (detached). The memory watchdog kills the newest `makeMetrics`
if a node drops below 8 GB free; the killed tile is marked failed and must be rerun this way.

## Outputs (all under `$RUN`)
`C/production/build_context/runtime/{intermediate,b2_pool,b3_output}` (fix.out and arff per tile, 30 models, B3 HDFs);
`Python/metrics/<tile>` (metrics, kept), `Python/training`, `Python/models`, `Python/predictions`;
`Compare/c_vs_python_63/{per_tile.csv,summary.txt}` when done. Older evidence kept: `C/workspace` (first canary, B1 only),
`C/canary2` (extended canary), `C/canary2_attempt_B1only_*`, `C/build` (rebuilt executables), `Python/test_phase3`,
`Python/dry_test_state`, `Python/validation_optionC`.

## Phase 0 results (done before launch)
- **Metrics pilots** (5 tiles on forest203/204, 2-3 at a time): all rc=0 and 44 files each, including the waived-input tile
  h12v01; 52-60 min per tile with all 24 standard metrics; peak 19.8 GB per process, 47 GB per node at 3 at once; no slowdown
  from 3 versus 2 at once. Stage times: composites 15-21 min, standard metrics about 27 min, unsorted values 7-9 min,
  C-mirror families 3.5 min. The production run computes only unsorted + mirror (about 33 min per tile).
- **Bug found by the pilots and fixed:** in a fresh `makeMetrics` run the mirror families used the raw floating-point unsorted
  values, so NDVI-related bands differed from the written unsorted tif (21 of 44 bands on h10v05). Now the code always reads the
  unsorted values back from disk. All five pilot tiles were regenerated and match exactly on a 400-row block.
- **C executables (option A):** rebuilt from the current sources in `C/build`. B1 and B2 are byte-identical to the sync-copy
  ones; B3 differs by 24 bytes, the aerosol-bit change (`>> 13` to `>> 7` in `fix_bands.c`). That change is Finding 2 of
  `C_code/pge61/MOD_PR44B_Metrics/C_vs_Python_Code_Differences.md`: it was decided on 2026-09-14 and applied to the sources of B1 and B3 on Sep 15 in
  the sync copy, but only B1 was rebuilt then, so the sync copy's B3 executable (and the Sep 11 full C run) still had the old shift. The clean run
  therefore uses the intended, documented C code in all three executables. B1 output is reproducible (byte-identical between two runs and both executables).
- **Extended C canary** (h10v05, `C/canary2`): B1, all 30 B2 models and B3 completed; the 127 MB HDF has 7 layers with sane
  values and water flagged; no scratch files left; none of the five protected roots changed. Every generated process control
  file is linted before it runs; the lint caught three leftover paths that would have written into the sync copy (Track A's
  B2 template, and the /panfs form of the real repo runtime path) and all were fixed before any production job ran.
- **Python chain test** on the five pilot tiles (small fit): training data 136 s per tile and type at 18 GB peak; split; fit with
  12 jobs; threaded predict of a full tile in 189 s (about 13 min before). Predictions in range with water = 200.
- **Orchestrator dry run** with two concurrent copies passed all barriers. Claim atomicity across five nodes passed (400 of 400).

## Corrections to earlier statements
- `BuildTraining` accepts a `trainingDir` argument but ignored it (labels always came from its package `data/training`). The
  copy in `$RUN/code/python_code` now honors it; the earlier note that the samp labels needed no symlink changes was wrong.
- Bare labels exist for 20 of the 63 tiles, not 35 (the 35 included tiles outside the 63).
- The Python models came out larger than estimated (13.5 GB tree, 11.9 GB bare, against 5 GB and up to 27 GB), and the whole run was faster than the
  10-hour estimate (about 7 hours).
- I said earlier that I could not tell which aerosol shift was correct. The 2026-09-14 decision in the code-differences document says `>> 7` (bits 7-8 of
  the repacked state) is the intended one for both C and Python; the shipped B3 executable simply had not been rebuilt.
- A second writer created this run root at 08:06:56 on 2026-09-19, before my first command; I believed it was a leftover of the interrupted session.
  Only one Claude process existed afterwards and I could not identify the writer. Its files were left as found; nothing of the final run depends on them.
  **Presumed resolved (2026-09-20, on the user's instruction):** this writer is presumed to be the previously disconnected Claude session. This is a presumption, not a
  proof. What supports it: the session transcripts in `~/.claude/projects/` show only one session, `8cc8fcae`, active at 08:06 local on 2026-09-19 (it ran from 06:33
  that morning to 06:31 on 2026-09-20 and is the one that was disconnected), and the run root name carries that 0806 stamp. What does not settle it: the transcript
  does not record who created the directory, and no process record from that moment exists. The item is closed as a presumption; nothing was changed on disk for it.

## Known limits
Python NDVI still differs from C's (accepted). Python output keeps its known post-processing gaps (no 200 in non-tree
vegetation, no cap at 100, truncation); the comparison reports raw and "harmonized" views (rounding cannot be harmonized).
The Python fit uses all labeled pixels, C uses about 40-50% random subsamples per model; the learners differ (bagged trees
versus M5P). No ground truth: the comparison shows how the products differ, not which is closer.
