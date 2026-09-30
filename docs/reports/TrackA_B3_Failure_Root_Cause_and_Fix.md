# Track A (C, 2-tile B1->B2->B3): why B3 never ran, and the fix

> **Update 2026-09-20:** the 63-tile run used its own claim-based C driver, a separate workspace and a path check on every process control file, so this
> failure (and the overwrite of shared files) did not recur there. The real-repo clean-up under "Still open" is still not done.

Status: **code fix applied and checked; Track A re-run from STEP 3 completed 2026-09-18 17:46 (see
`FinalCompare_2tile_Plan_and_Status.md`); cleanup of the real repo only partly done (see "Still open").** Written 2026-09-18. Companion
to `PyTrackB_2tile_Plan_and_Status.md` (the Python-side equivalent).

## What happened

Track A (`C_code/pge61/build_context/scripts/run_track_a_2tile.py`, run
2026-09-17 12:40-13:37) finished B1 and all 30 B2 model runs (every one rc=0),
then logged `0/30 weka.2019.NN.model files present` / `STEP 4 FAILED` and
exited before STEP 5/6 (B3 PCF generation and run). B3 never executed; there is
no `b3_output/` directory.

## Root cause: one bug, two symptoms

`build_context/scripts/generate_b2_pcf.py` hardcoded

    WORKSPACE = Path("/panfs/ccds02/nobackup/projects/ilab/projects/MODIS-VCF/pge61_08312026/pge61")

i.e. the **original repo**, not this sync copy. Its sibling
`generate_tile_pcf.py` (used for B1 and B3) correctly points at the sync copy.
Because `generate_b2_pcf.py` ignores the caller's cwd, on Track A's STEP 3:

1. The run script patched the **sync copy's** B2 template (`VCF_outputs` ->
   `arff_input`), but `generate_b2_pcf.py` then read the **original repo's**
   unpatched template. All 30 generated PCFs kept `INTERMEDIATE OUTPUT` =
   `VCF_outputs`. `MOD44B2.c` (lines 87-133) takes `int_dir` from that PCF
   section and runs `java ... weka.classifiers.trees.M5P ... -d %s/weka.%s.%02i.model`
   with it, so all 30 real models were written to the shared `VCF_outputs/`.
2. STEP 4's check globs `ARFF_DIR` (`build_context/runtime/b2train_2tile/arff_input/`),
   the intended location. It was empty, hence "0/30" and the early exit.
   This is the same bug as (1), not an independent one.
3. `generate_b2_pcf.py`'s `OUT_DIR` and per-run log root derive from the same
   `WORKSPACE`, so the 30 generated `MOD_PR44B2.NN.pcf` files and the
   `runtime/logs/b2_NN/` scratch dirs (LogStatus/LogReport/LogUser/ShmMem)
   were written into the **original repo's** untracked `build_context/`.

B1 was unaffected (correct `WORKSPACE`), and B3's PCF generation/patching
(STEP 5, never reached) is already scoped to the sync copy.

## Fix applied

`generate_b2_pcf.py` line ~20: `WORKSPACE` now
`/explore/nobackup/people/mcarrol2/MOD44B_VCF_Development/AI_code_sync_09152026/C_code/pge61`
(same as `generate_tile_pcf.py`), with a short comment on why.

Checked cheaply before any re-run: on forest203, `generate_b2_pcf.py --model 0`
wrote `MOD_PR44B2.00.pcf` into the sync copy's `test_pcf/generated_b2/`; it
contains no `VCF_outputs` string and has the 4 expected `b2train_2tile/arff_input`
paths; no new file appeared in the original repo.

## Where outputs will go when Track A is re-run (from STEP 3)

All inside `AI_code_sync_09152026`, none in `VCF_outputs/`:

- B2 models (30): `.../C_code/pge61/build_context/runtime/b2train_2tile/arff_input/weka.2019.NN.model`
- B2 consolidated training ARFFs and scratch: same `arff_input/` dir
- B2 generated PCFs: `.../C_code/pge61/build_context/test_pcf/generated_b2/`
- B2/B3 per-run scratch logs: `.../C_code/pge61/build_context/runtime/logs/b2_NN/`
- B3 products: `.../C_code/pge61/build_context/runtime/b2train_2tile/b3_output/MOD44B.A2019065.h09v05.061.hdf` and `...h12v02.061.hdf`
- Driver/step logs: `.../runtime/b2train_2tile/track_a.log`, `logs/`

(`/panfs/ccds02/nobackup/people/mcarrol2/...` is the same storage as the
`/explore/nobackup/people/mcarrol2/...` paths the C toolkit sees.)

B1 does not need re-running. Resume at STEP 3, e.g. a small driver calling
steps 3-6, or `run_track_a_2tile.py` with steps 1-2 skipped.

## Still open

- **Stray files in the original repo were NOT removed.** The 30
  `build_context/test_pcf/generated_b2/MOD_PR44B2.NN.pcf` files (all
  2026-09-17 12:48, the only contents of that dir) are still there; my delete
  was blocked by the permission classifier and needs to be done or allowed by
  the user. The dir is untracked (`?? build_context/`), so nothing can be
  committed by accident.
- The 30 `build_context/runtime/logs/b2_NN/` dirs in the original repo (4 small
  scratch files each, 2026-09-17) are also Track A byproducts. Not touched, since
  they were outside what was agreed; it is unclear whether any pre-dated Track A.
- Files Track A overwrote in the original repo (any earlier `generated_b2` PCFs
  from 2026-09-11) and in `VCF_outputs/` (previous `weka.2019.NN.model`,
  `weka_training.all.2019.NN.arff`) are unrecoverable: neither location is
  versioned or backed up.
- The 30 `weka.2019.NN.model` files currently in `VCF_outputs/` are from the
  buggy run; the re-run will not touch them.
