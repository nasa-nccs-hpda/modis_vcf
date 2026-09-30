# How the 30-model ensemble becomes the final VCF value: C vs Python

> **Status of the open decisions below (2026-09-20):** in the 63-tile run Python still truncates (no rounding change), still uses unpruned trees
> (`min_samples_leaf` 1), and its outputs were not post-processed to C's rules; instead the comparison reports raw and "harmonized" views
> (`FullRun_Results_C_vs_Python_63tiles.md`), where harmonizing changes almost nothing. The Greenland block in `MOD44B3.c` was still not read.

Status: **findings from reading the code, 2026-09-18. Nothing changed as a
result; the differences below are open decisions.** Written while running the
Python two-tile retrain (`PyTrackB_2tile_Plan_and_Status.md`). Resolves the
"base learner not located" caveat in `C_code/pge61/Sync_Plan_and_Model_Retraining_Alternate.md`.

## Python (current code)

Training, `Python_code/modis_vcf/view/rfFitFromPge61.py`:
`BaggingRegressor(estimator=DecisionTreeRegressor(), n_estimators=30, bootstrap=True)`
(30 via the new `--nEstimators` flag; default 1). Everything else is sklearn
default: each tree is fit on a bootstrap resample of all training rows (same
size, with replacement), considers all predictors at every split, and is grown
unpruned (no depth limit, min 1 sample per leaf). A leaf value is the mean label of its
samples. Separate ensembles for `pcttree` (60 predictors) and `pctbare` (18).

Prediction, `Python_code/modis_vcf/model/VcfPredict.py`:
1. `rf.predict(X)` on a `BaggingRegressor` returns the **unweighted mean of the
   30 trees**. Leaves are label means, so each tree, and the mean, stays in the
   0-100 label range.
2. `_runPrediction`: `.astype(np.int16)` (**truncation**, not rounding), reshape
   to 4800x4800, then `_maskPrediction`: water (mask 1) -> 200, out-of-projection
   (mask 250) -> 250.
3. `runTileForYear`: tree cover and non-vegetated are predicted independently;
   non-tree vegetation = `np.maximum(0, 100 - pctTc - pctNonveg)`. Writes
   `Percent_Tree_Cover`, `Percent_NonVegetated`, `Percent_NonTree_Vegetation`
   tifs via `Utils.writeRaster`.

## C

- **Training** (`MOD_PR44B2/MOD44B2.c` lines 129-135): per model number NN,
  `java ... weka.classifiers.trees.M5P -M 500.0 -x 3 -t weka_training.all.<year>.NN.arff -d weka.<year>.NN.model`.
  M5P model tree: linear models at the leaves, at least 500 instances per leaf
  (not REPTree). As far as MOD44B2.c shows, each of the 30 models trains on its
  own NN-th sample file (one per tile, concatenated), not on bootstraps of one
  pool. How B1 draws each NN subsample was not examined.
- **Per-pixel prediction** (`MOD_PR44B3/Predict.java` lines 20-81): tree cover uses
  `classifier[0..29]` (the retrained `weka.<year>.NN.model`), bare uses
  `classifier[30..59]` (static `BARE/weka.bare.NN.model`, loaded in MOD44B3.c
  lines 316-322). Each member's output is **clamped to [0,100]**, then the 30
  are summed and divided by 30.
- **Conversion and post-processing** (`MOD44B3.c`):
  - lines 363-364, 403-404: `floor(prediction + 0.5)` (**round half up**).
  - lines 366-371: value set to -3 if any of the 23 values of the first input
    band (`array[0]`) equals -28672 (my reading; not traced further).
  - lines 376-382, 489: per-pixel SD across the 30 members written as
    `Percent_Tree_Cover_SD`.
  - lines 406-410: bare capped so tree + bare <= 100; a Greenland-specific
    block follows (comment at lines 411-413; **not read**).
  - lines 476-482: tree capped the same way.
  - lines 495-503: non-tree vegetation = `100 - tree - bare`, clamped to 0-100;
    water -> 200 in all layers.

## Differences (C vs Python)

| | C | Python |
|---|---|---|
| Member learner | M5P, linear leaves, min 500 instances/leaf | unpruned DecisionTreeRegressor, constant leaves, min 1/leaf |
| Member training data | its own NN-th sample file | bootstrap resample of one pooled set |
| Per-member clamp to 0-100 | yes | none (moot for constant-leaf trees) |
| Ensemble combine | mean of 30 | mean of 30 |
| Float -> int | round half up | truncate (about 0.5 low on average) |
| tree + bare <= 100 | enforced, both directions | not enforced (only non-tree veg floored at 0) |
| Non-tree veg over water | 200 | 0 (water-masked 200s make `100-200-200` -> 0) |
| SD layer, -3 fill, Greenland rule | yes | none |

Unpruned trees are also the likely reason the Python model files are so large
(`pctbare.bin` 1.6 GB, `pcttree.bin` 670 MB).

## Open decisions (not made)

- Rounding: change the truncation in `VcfPredict._runPrediction` to round
  half up (one line).
- Python tree limits: e.g. `min_samples_leaf=500` would move toward M5P `-M 500`,
  but cannot reproduce M5P's linear-model leaves.
- Post-processing parity (cap tree + bare at 100, water -> 200 in the derived
  layer, SD layer, -3 fill) if C-vs-Python comparison of the final products is a goal.
- Read the Greenland block (`MOD44B3.c` from line 411) before comparing high-latitude tiles.
- Check how B1 builds each of the 30 sample files if matching ensemble-member
  training data matters. (Update 2026-09-19: read in `MOD_PR44B1/MOD44B1.c` lines 219-242: pixels labeled 240 or
  255 are skipped, and of the rest 50% of zero-labeled and 40% of non-zero-labeled pixels are kept at random. How
  the kept pixels are split over the 30 files was still not examined.)
- Training labels (update 2026-09-19): C reads `v5.0.3samp/<tile>.samp.bin`, Python `data/legacy/<tile>.bin`, a
  larger, more zero-heavy superset with identical values on the overlap. Whether to switch Python to the samp
  labels and whether to copy C's 50%/40% subsampling are open; see `Signoff_Package_2tile_C_vs_Python.md` item 6a.
