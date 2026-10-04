# 60 trees, then the Monte Carlo 64 predictors, in the gated V4r blend (v1.0.1 recipe): a combined plan

> **FULLY EXPERIMENTAL.** Nothing here changes v1.0.0, v1.0.1 or the delivery branch. All products are test products only; any
> adoption or version naming is a separate decision by the project lead.

**Written 2026-10-03, requested by the project lead (60 trees first, then the Monte Carlo 64; ngaproc201/202). Status: approved and launched
2026-10-03 evening (`FullRuns/trees60_mc64_gated_20261003`).** Names as in `Product_Versions_and_Work_Map.md`.

## Why

- **More trees:** the 60-tree tree-cover model (T60) was the clearest gain found today: −0.119 held-out MAE, better on 57 of 63 tiles
  (`Trees_and_Counts_2x2_Results.md`). Nothing so far uses it; v1.0.0 and v1.0.1 both use 30 trees.
- **Predictor set:** the Monte Carlo top 64 (MC64) was worse than the C code's 64 with v1.0.0's inputs (`Monte_Carlo_Rerun_Results.md`).
  It has not been tried inside the gated V4r blend, where its greenest- and warmest-period predictors might suffer less from cloud
  fallback.
- **Order:** 60 trees first, so that MC64 is tested on the forest size that would be used.

## Candidates

All use the v1.0.1 recipe unchanged:
- the same V4r composites and snow-aware counts (`gated_v4r_63_20261003/out/counts_snow`);
- the same gates: cloud (snow-free zenith-gated V4r count ≥ 5, w = count / 8), cover (ramp from 20 to 50 on the **clear-sky model's
  own prediction**), and the pick snow test;
- the same blend and rounding;
- other layers v1.0.0's, as for v1.0.1.

| ID | Clear-sky model | V4r model | Phase |
|---|---|---|---|
| v1.0.0 | 30 trees, C code's 64 (delivered) | | reference |
| v1.0.1 | 30 trees, C code's 64 | V4R: 30 trees, C code's 64, on V4r metrics | reference |
| **A: v1.0.1 at 60 trees** | **T60** (exists; seed 0) | **V4R60** (new: 60 trees, C code's 64, on the existing V4r split) | 1 |
| **B: A with MC64** | **MC64-60** (new: 60 trees, MC top 64, on the Monte Carlo 324-column split) | **V4R-MC64-60** (new: 60 trees, MC top 64, on new V4r tables) | 2 |
| Seed yardstick | T60, seed 1 (new; held-out only) | | 1 |

Model settings throughout: RF, `max_features='sqrt'`, MEDIAN, rounded; the same keyed 80/20 pixel split as v1.0.0.

## Phase 1: 60 trees (about 4.5-5 h)

1. **Fits** (about 15 min, 2 at once): V4R60 on `cloudy_v4r_weights_20261002/Python/split_v4r` with the 64 predictors; T60 seed 1 on
   v1.0.0's split.
2. **Maps, 63 tiles** (about 4 h; 3 at once per node, about 45 GB each):
   - T60 on the bit-15 metrics;
   - V4R60 on the V4r metrics.
   - Same predictor script as before (`map_predict.py`). Check: T60's map of h12v09, re-predicted, must be reproducible.
3. **Gated blend A** (about 45 min): the v1.0.1 code (`gated_tile.py`, main configuration only), with T60 as the clear-sky model and
   V4R60 as the partner.
   - **Check:** the same script run on v1.0.0 and V4R must reproduce v1.0.1's maps pixel for pixel. This verifies the swap of inputs.
4. **Held-out labels, 63 tiles** (minutes): T60, V4R60, blend A, against v1.0.0 and v1.0.1, with the T60 seed yardstick. Scored by
   label bin, by zenith-gated cloud count, per tile, and pooled MAE with tile bootstrap.

## Phase 2: Monte Carlo 64 under A (about 5-6 h; starts when phase 1's maps are done)

5. **V4r metrics for the 7 missing families, 63 tiles** (about 45 min).
   - MC64 needs CThermMean79, CThermMax, CGreenMean79 and CSortedRank2-5, which the V4r run did not build.
   - They are built with `makeMetrics.py` from the existing V4r composites into a **new** metrics root. That root links the existing
     21 families' rasters and the composites read-only; the earlier run is not written to.
   - **Check:** at pixels where V4r replaced nothing, each new family equals the bit-15 family in `mc_rerun_20261002/Python/metrics_324`
     cell for cell.
   - **Revised before launch (pre-launch self-tests, 2026-10-03):**
     - The V4r run's family rasters hold only the bands the C code's models use, so MC64 lacked 27 columns. Step 5 now builds all 26
       C families that the delivered 64 and MC64 need, with exactly the needed bands (`code/p0_makemetrics_needed.py`). The check
       compares every band by name.
     - Self-test on h12v09: 85 bands equal the bit-15 values at every unchanged pixel; the table has every needed column. The step
       takes about 7.5 min per tile, about 1-1.5 h for 63 tiles.
6. **V4r training tables and split** (about 1 h): `buildTraining.py` on the new root, then `split_v4r.py` with its acceptance test
   against the Monte Carlo split:
   - same keys and labels;
   - where V4r changed nothing, every column equals the bit-15 value.
7. **Fits** (about 25 min, 2 at once):
   - MC64-60 on `mc_rerun_20261002/Python/split/tree`;
   - V4R-MC64-60 on the new V4r split.
8. **Maps, 63 tiles** (about 4 h):
   - MC64-60 on `mc_rerun_20261002/Python/metrics_324`;
   - V4R-MC64-60 on the new V4r metrics root.
9. **Gated blend B** (about 45 min), with the cover gate on MC64-60's prediction. Then held-out scoring as in step 4, with all
   candidates side by side.

## Evaluation and report (both phases)

**Held-out labels (63 tiles):**
- MAE and same-bin share for each product against v1.0.0, with tile bootstrap;
- by label bin × cloud group;
- dense labels (≥ 41) and open labels (≤ 40) at 5+ cloud periods, per tile, with and without h10v04.

**Maps (63 tiles), by C6 class × cloud group:**
- change from v1.0.0 and v1.0.1;
- mean difference from C6, C6.1 and the C code (direction only);
- the h12v09 drop pixels and the dense control pixels;
- seams;
- a figure set of each blend against C6, in the same layout as `figures_gated_v4r_63_20261003/`.

**Questions for the report:**
- **A against v1.0.1:** does 60 trees carry its −0.12 gain through the blend, without losing the cloud raise?
- **B against A:** does MC64 do better under cloud, i.e. more dense cloudy labels in the right bin and the drop pixels raised, without
  losing clear-sky accuracy? Its earlier +0.28 MAE disadvantage on v1.0.0's inputs is the bar.

**Report:** `Trees60_then_MC64_Gated_Results.md`, and the README and work map are updated.

## Disk and compute

**Disk:**

| Item | Size |
|---|---|
| New 60-tree models: V4R60, MC64-60, V4R-MC64-60, T60 seed 1 | about 34 GB each, about 135 GB in all |
| New V4r metrics, 7 families × 63 tiles | about 300 GB |
| New tables and split | about 25 GB |
| Maps | small |

The 1,077 GB freed today covers this.

**Compute:** ngaproc201/202 only, both idle now. Fully detached, with the usual claims, memory waits, slots, both watchers and status
files.

**Total:** about 10-11 h, an overnight run.

## Not in this plan

- non-vegetated;
- other years (they follow, using whichever version is chosen);
- the weight-1 option at 5+ periods;
- naming A or B as versions.
