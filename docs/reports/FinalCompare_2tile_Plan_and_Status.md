# C (Track A, resumed) vs Python (Track B) two-tile comparison: what was launched

> **Historical (two-tile test, 2026-09-18); superseded 2026-09-20 by the 63-tile run** (`FullRun_Results_C_vs_Python_63tiles.md`). The h12v02 disagreement
> recorded here is not present in that run (tree cover corr 0.917); the Python post-processing gaps listed here remain (`README_START_HERE.md`, open item 4). Of
> the "Still open" items below: the real-repo clean-up is still open; the other two are answered by the 63-tile run or carried into its open items.

Status: **launched 2026-09-18 17:07, finished 17:46 (all chains rc=0). Results are in the "Results" section below.**
Companions: `TrackA_B3_Failure_Root_Cause_and_Fix.md`, `PyTrackB_2tile_Plan_and_Status.md`,
`Ensemble_Prediction_C_vs_Python.md`.

## What is running

Master: `Reports/run_final_compare_master.sh` (PID 1291382 at launch, parent init, no TTY).
Two chains run in parallel; a failure in one does not stop the other.

**Python chain**
1. `vcfPredict.py` for h12v02 with the existing two-tile models (`Metrics/PyTrackB_2tile/3-Models`),
   writing `Metrics/PyTrackB_2tile/4-VcfProcess/h12v02/` (~13 min expected, as for h09v05).
2. `Reports/heldout_eval_py_track_b.py`: scores the models on the 20% held-out rows
   (`xTest/yTest.parq`), overall and per tile, plus truncate-vs-round and an in-sample contrast.
   Output: `Metrics/FinalCompare_2tile/heldout_eval.txt`.

**C chain** (`launch_track_a_2tile.sh --from-step 3 --to-step 6`): STEP 3 regenerate the 30 B2 PCFs
(now scoped to the sync copy), STEP 4 train 30 M5P models, STEP 5 generate/patch the B3 PCFs,
STEP 6 run B3 for both tiles. Log: `Metrics/FinalCompare_2tile/c_track_a_resume.log` and
`C_code/pge61/build_context/runtime/b2train_2tile/track_a.log`.

**Compare** (after both chains exit): `Reports/compare_c_vs_python_2tile.py` ->
`Metrics/FinalCompare_2tile/compare_c_vs_python.txt`. Compares tree cover, non-vegetated and
non-tree vegetation on pixels valid (0-100) in both products, per tile (correlation, bias,
MAE/RMSE, share within 5 and 10, percentiles), reports validity/water disagreement, an
orientation check, and how often tree + non-veg > 100. It is a comparability check, not an
exact-match test (M5P on B1 sample files vs bagged unpruned trees on pooled labeled pixels).

Progress: `Metrics/FinalCompare_2tile/status.log` (timestamped; ends with
`=== FINAL COMPARE MASTER ALL DONE ===`).

## Where outputs go (all inside AI_code_sync_09152026; nothing to VCF_outputs)

- C models: `C_code/pge61/build_context/runtime/b2train_2tile/arff_input/weka.2019.NN.model`
- C B3 products: `C_code/pge61/build_context/runtime/b2train_2tile/b3_output/MOD44B.A2019065.<tile>.061.hdf`
- Python h12v02: `Metrics/PyTrackB_2tile/4-VcfProcess/h12v02/*.tif` (h09v05 already in `.../h09v05/`)
- Held-out and comparison text: `Metrics/FinalCompare_2tile/`

## Changes made to Track A for this run (`build_context/scripts/`)

- `run_track_a_2tile.py`: added `--from-step/--to-step`.
- STEP 4: deletes stale `weka_training.all.*.arff` and `weka.2019.*.model` from `arff_input/` first
  (`MOD44B2.c` concatenates `weka_training.*.<year>.<NN>.arff` from the same dir it writes the
  "all" file to, so a leftover file would be concatenated into itself); after every batch of 8 it
  aborts if `VCF_outputs/weka.2019.*.model` changed or if models are not appearing in `arff_input/`.
- STEP 5: the old patch matched the string `.../VCF_outputs`, which no longer appears in the B3
  PCFs (`generate_tile_pcf.py` already repoints those directories to `intermediate/<tile>`), so it
  changed nothing. As generated, B3 would have read the models from `intermediate/<tile>` and written
  its HDF there. Now patched by PCF section: PRODUCT OUTPUT -> `b3_output`, INTERMEDIATE OUTPUT ->
  `arff_input` (where B2 writes the models), TEMPORARY I/O -> `intermediate/<tile>`. Verified in both
  generated PCFs before launch.
- `launch_track_a_2tile.sh` now passes its arguments through.

## Preflight results (B3 has never run in this copy)

Present: 12 `fix.out` inputs per tile in `intermediate/<tile>` (from Track A's B1), 30
`BARE/weka.bare.NN.model` (read from the original repo's `BARE/`, read-only), the `MOD44B.mcf`
template, the MOD44W water-mask granule path. Not checked: whether the B3 executable runs to
completion (a comment in the PCF template warns that `create_output_file`'s `PGS_MET_Init` needs an
ODL `MOD44B.mcf`; the file exists, but B3 output creation is untested). Runtime of B3 is unknown:
it makes 60 Java model calls per pixel over two passes per tile, so it could take hours.

## CONFIRMED (17:09): the earlier Track A models were trained on a many-tile pool

The resumed STEP 4 finished 30/30 models in about 1 minute (Sep 17: ~49 min), all in `arff_input/`,
`VCF_outputs/` untouched. Model 00's log now shows `Total Number of Instances 18051` (the two-tile
`all` arff is 5.5 MB) versus 324,033 on Sep 17. So the `VCF_outputs/weka.2019.NN.model` files from
Sep 17 came from the pooled many-tile arff files, as suspected below. Note for the comparison: the
new C models see ~18k instances each (B1 sample files), the Python models ~1.86M/0.95M pooled rows,
so a large part of any C-vs-Python difference can come from training volume, not only from
M5P vs bagged trees.

## (Original reasoning) earlier Track A models (Sep 17)

Those B2 runs used `int_dir = VCF_outputs`, and `MOD44B2.c`'s `cat` glob there matches every tile's
`weka_training.<tile>.2019.NN.arff` ever written to that shared directory (65 tiles present, 30 files
each). The earlier logs show `cat: .../weka_training.all.2019.00.arff: input file is output file`
and `Total Number of Instances 324033` for model 00, and the `all` arff files are ~98 MB, far more
than two tiles' ~2.5-3 MB files. So the models now in `VCF_outputs/weka.2019.NN.model` were very
probably trained on a many-tile pool, not the two tiles. Confirm from this run: the new B2 logs
(`.../b2train_2tile/logs/b2_NN.log`) should show a much smaller `Total Number of Instances`.

## Results (recorded 2026-09-18 ~17:55 from `Metrics/FinalCompare_2tile/`)

Timeline: Python h12v02 predict 17:07-17:22, held-out eval done 17:23; C STEP 6 (B3, both tiles)
17:09-17:46 (~37 min; the HDFs grew from 203 KB skeletons to 104 MB h09v05 and 64 MB h12v02);
compare 17:46:12-17:46:46. Master exited cleanly.

**Python held-out (`heldout_eval.txt`)**: tree cover corr 0.969, MAE 0.86, RMSE 2.7 overall
(h12v02 held-out labels are all 0, so its corr is n/a); non-vegetated corr 0.897, MAE 7.3, RMSE 12.0
(h09v05 MAE 3.1, h12v02 MAE 9.8). Truncate vs round-half-up moves MAE by only ~0.05 (tree) and ~0.05-0.1 (bare).

**C vs Python on pixels valid in both (`compare_c_vs_python.txt`)**

| Tile / layer | corr | mean Py vs C | bias (Py-C) | MAE | within 5 |
|---|---|---|---|---|---|
| h09v05 Tree cover | 0.901 | 13.93 vs 14.78 | -0.85 | 4.82 | 69.8% |
| h09v05 Non-veg | 0.470 | 35.88 vs 27.75 | +8.13 | 17.11 | 24.2% |
| h09v05 Non-tree veg | 0.185 | 50.23 vs 57.47 | -7.24 | 18.84 | 19.7% |
| h12v02 Tree cover | 0.065 | 6.97 vs 1.00 | +5.96 | 7.25 | 73.7% |
| h12v02 Non-veg | 0.540 | 15.91 vs 47.36 | -31.46 | 32.44 | 6.2% |
| h12v02 Non-tree veg | 0.259 | 77.13 vs 51.63 | +25.49 | 32.06 | 8.7% |

Observations (what the numbers show; causes below are hypotheses, not tested):

1. **h09v05 tree cover agrees reasonably** (corr 0.90, as-is orientation is clearly the right one:
   transposed 0.25, flipud 0.18). Non-veg and non-tree veg agree poorly (corr 0.47 / 0.18).
2. **h12v02 disagrees strongly on every layer.** No orientation fixes it (as-is 0.065, transposed
   -0.055, flipud -0.083), so this is a real content difference, not a flipped grid. C predicts almost
   no tree cover (p95 = 3); Python's p95 is 44 even though every h12v02 training/held-out tree label
   is 0 and Python scores MAE 0.001 on those rows. Python is therefore extrapolating on pixels unlike the
   labeled sample rows, or its prediction inputs for h12v02 differ from the training features
   (see `Polar_Night_Zero_Observation_Finding_h12v02.md`). Label note (2026-09-19): the user (mcarrol2)
   asserts the all-zero h12v02 tree labels in `v5.0.3samp/h12v02.samp.bin` (mtime 2010-09-16) are accurate,
   based on that historical date; see the "Label provenance" bullet in `Signoff_Package_2tile_C_vs_Python.md`
   for the two label files and which one Python trains on. C's ~18k-instance models and Python's
   ~0.95M-row pool also saw very different amounts of h12v02 data.
3. **Post-processing parity gap (concrete, not a modeling difference):** Python's
   Percent_NonTree_Vegetation is *not* set to 200 where tree cover and non-veg are 200 (water/no-data).
   That is 106,052 px in h09v05 and 2,517,812 px in h12v02 that are valid in Python but 200 in C. In
   h12v02 a further 11,192 px are C = fill (-3/253) but Python has values for all three layers.
4. **Python does not enforce tree + non-veg <= 100**: 100,689 px (0.44% of valid) in h09v05 and
   3,242 px (0.02%) in h12v02; C has none.
5. Valid/fill masks otherwise match on h09v05 (C-only and Py-only both 0 for tree and non-veg).

Scope: this two-tile comparison was meant to verify the pipelines and give a quick check of the C-vs-Python
relationship. A full 63-tile test is needed for true verification.

Caveat carried from the plan: this compares M5P on B1 sample files (~18k instances/model) with bagged
unpruned trees on pooled labeled pixels (~1.86M/0.95M rows), and there is no ground truth in this
comparison, so it cannot say which product is closer to the truth, only how they differ.

Suggested next checks (cheap first): map where h12v02's Python tree predictions are >0 (latitude
band? rows with zero observations?), and trace one high-tree-cover h12v02 pixel through the Python
features against the training rows before any further full-tile run.

## Still open

- Real-repo cleanup: the 30 stray `build_context/test_pcf/generated_b2/MOD_PR44B2.NN.pcf` files
  (delete was blocked by the permission classifier) and the 30 `runtime/logs/b2_NN/` dirs there.
- Post-processing parity (items 3 and 4 above): NonTree_Veg water/no-data masking and the
  tree + non-veg <= 100 constraint in the Python path.
- Explain the h12v02 disagreement (item 2).

## Update 2026-09-19: label files and CPU use

- **The two sides trained on different tree-cover label files.** C's B1 reads `v5.0.3samp/<tile>.samp.bin`;
  Python trained on `Python_code/modis_vcf/data/legacy/<tile>.bin`. The samp set is a strict subset with identical
  values (h09v05: 268,039 vs 1,552,272 labeled px; h12v02: 307,939 vs 769,131), and the legacy set is more
  zero-heavy on h09v05 (70.5% zeros vs 48.7%). So part of the h09v05 difference (Python's lower tree cover,
  13.9 vs 14.8) may come from label choice, not only the model type; untested. It cannot explain the h12v02
  tree-cover gap, because both h12v02 sets are all zero. Details and the swap plan: "Label provenance" in
  `Signoff_Package_2tile_C_vs_Python.md` (item 6a).
- **Python predicts 0 to 19 on the 307,178 valid samp-zero pixels of h12v02** (99.99% exactly 0), so its
  large h12v02 predictions are on unlabeled pixels (fit check, not out-of-sample).
- **Python used one CPU.** Measured fixes are in `Parallel_CPU_Use_Assessment.md`; none applied yet.
