# Mean, middle-20 or median of the RF trees: 63-tile maps against every reference (plan)

**Written 2026-10-01. Status: plan only, awaiting approval.**

## Why

The held-out test (`RF_Tree_Aggregation_Test_Results.md`) used labelled pixels only. There, the median of the 30 trees and the mean of the middle 20 beat the plain mean on MAE: tree cover −0.61 and −0.51, non-vegetated −0.37 and −0.33. They also reduce compression at both ends, at a small cost in the middle bins.

The project rule asks for differences with the same sign across independent references before changing a default. This run makes full maps with each way of combining the trees and scores them against:
- C6, C6.1 and the C code;
- NEON lidar;
- G-LiHT lidar;
- the stable-pixel set.

## What is compared

**Three ways of combining the 30 trees,** each written as a complete product: tree cover, non-vegetated, non-tree vegetation and the two SD layers.

| Name | Value layer |
|---|---|
| MEAN | mean of the 30 trees (today's default; identical to `predict()`) |
| MID20 | mean of the middle 20 trees (drop 5 at each end) |
| MEDIAN | median of the 30 trees |

**Models:**
- tree cover: the bit-15 RF (`qaC_full63_retrain_20260929/Python/models/C/rf`);
- non-vegetated: the new bit-15 RF (`nonveg_bit15_models_20261001/Python/models/rf`, recommended and pending approval).

The SD layer is the same in all three products (the SD of the 30 trees).

**Code:** a copy of the SD-patched `VcfPredict` from `sd_layer_test_20261001/code` (`manifests/VcfPredict_sd.diff`), extended so that one member pass per model and tile writes all three products. The trees are predicted once, then the mean, middle 20 and median are taken from the same array.
- The MEAN product must equal the existing bit-15 maps exactly: tree cover against `qaC.../predictions_C/rf`, non-vegetated against `nonveg.../predictions/rf`. That is checked per tile.
- The layer-sum rule is the one in the run's code copy, non-tree vegetation = max(0, 100 − tree − non-vegetated), applied to each product separately.
- Nothing goes on the delivery branch.

## Compute (all seven nodes; checked idle 2026-10-01 ~15:35)

- forest201-205: 12 CPUs and about 104 GB free each. 2 workers per node.
- ngaproc201/202: 24 CPUs and about 217 GB free each. 4 workers per node. ssh needs `-o CheckHostIP=no`, and scripts need `source /etc/profile.d/z00_lmod.sh; module load singularity/4.3.1`.

That makes 18 workers sharing the 63 tiles through `mkdir` claims, as in the earlier runs.
- **Time per tile** (measured on h08v05): metric reading about 2.5 minutes, plus two member passes of about 0.5 minutes each, plus writing 3 × 5 layers. About 4-5 minutes per tile per worker, with 6 threads each.
- **Expected:** about 63 × 4.5 / 18 ≈ 16 minutes, plus slack for shared-disk contention; under 45 minutes in practice.

The run is fully detached (`setsid nohup`), in the new run root `FullRuns/rf_aggregation_maps_20261001/`. Earlier runs are mounted read-only. Each step writes `.done`/`.failed` markers.

## Steps

0. **Cheap check, then stop for review:**
   - One tile (h08v05, forest203), all three products.
   - Checks: MEAN equals both existing maps exactly; MID20 and MEDIAN differ from MEAN in the expected direction at the ends of the range; the SD equals the SD-test output for the same tile.
   - Record the time per tile.
1. **Predict 63 tiles** on 18 workers (about 30-45 minutes).
2. **Historical comparison, tree cover:**
   - Four-way method (`hist4way_c6_c61_20261001/code/four_way.py`, adapted to the three products), against C6, C6.1 and the C code.
   - The 9 tiles flagged for the C6.1 high-latitude rule are reported per tile and kept out of pooled numbers.
   - Breakdowns: 4 cover bins of the reference (0-20, 21-40, 41-60, 61+), and continent × latitude band (location only).
3. **Historical comparison, non-vegetated:** the same, with `compare_bare.py` and `summarize_bare.py` adapted to the three products. The caveat about the C code's bare rules on 12 tiles carries over.
4. **Lidar and stable references, tree cover:**
   - **NEON:** the same 50,561 pixels on 8 tiles, 19 sites (`stable_pixels_20260930/neon_eval/neon_eval_pixels.csv`). 2, 3 and 5 m cover; MAE, bias and bins (`code/neon_eval.py` method).
   - **G-LiHT on h11v02:** the pixels and method of `code/gliht_eval.py`.
   - **Stable set:** the 10,000 pixels on h11v04, h18v03 and h19v09 (`code/stable_set_eval.py` method). Stable-pixel references are built from the historical product, so they favour whatever resembles it.
5. **Summary and decision table:** for each reference, MID20 − MEAN and MEDIAN − MEAN, with 95% block-bootstrap (100 × 100-pixel) and tile-bootstrap intervals, and counts of tiles or sites where each wins. Report: `Reports/RF_Aggregation_Maps_63Tile_Results.md`. README updated at launch and at the end.

## How the result is judged

The project rule:
- A change of default needs the same sign across independent references, with intervals clear of zero.
- About 1 point of MAE or less is a tie.

**The references here:**
- held-out labels (already done);
- C6, C6.1 and the C code (one lineage);
- NEON and G-LiHT lidar (independent);
- the stable set (built from the historical product).

**Expected outcome:** the differences will probably be ties by size. The decision then rests on behaviour:
- **compression at the ends:** the dense-forest under-reading against lidar, which was −26 to −30 points at 60%+ cover;
- **middle-bin cost;**
- **consistency across references.**

If MID20 or MEDIAN reduces the lidar compression without losing elsewhere, that supports changing the default.

## Not in scope

- Changing any model or retraining.
- The C code's bare side rules (deferred).
- Putting code on the delivery branch. That follows a decision on the default.
- Out-of-sample tiles or other years.
