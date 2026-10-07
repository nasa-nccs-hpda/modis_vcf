# Seed yardstick for the band-31 maps: plan

**Status: APPROVED by the project lead 2026-10-06: Y1 (c) all four nodes, Y2 (a) blocks, Y3 (a) keep all layers, Y4 "yes" read as (b), add the C6 / C6.1 comparison.**
Follows: `Band31_Unclamped_Plan.md` (run `EXP_b31_unclamped_20261005`) and its map figures
(`FullRuns/EXP_b31_unclamped_20261005/out/figures/maps/<tile>_b31_vs_v110.png`).

## 1. Why

The b31 map figures (v1.1.0 | b31 unclamped | b31 minus v1.1.0) show changes of 3-4 points on average on many tiles. They show them on
tiles where band 31 was **never clamped** about as much as on tiles where it was:

| Layer | Largest mean abs. change, clamped tiles | Largest mean abs. change, never-clamped tiles |
|---|---|---|
| Tree cover | 4.4 (h12v03) | 4.2 (h27v06) |
| Non-vegetated | 3.7 (h12v12) | 4.0 (h20v11) |

So most of what the difference panels show is retraining noise: a random forest refit on the same rows moves many pixels by a few points.
The band-31 effect is only what stands out **above** that noise. To see it in the maps we need the noise itself as a map: v1.1.0 refit
with a different seed, minus v1.1.0.

## 2. What is already there (no training needed)

Both seed-1 models of the v1.1.0 recipe (60 trees, sqrt, MEDIAN) already exist:
- tree cover: `trees60_mc64_gated_20261003/Python/models/T60S1` (seed 1, `qaC_full63_retrain_20260929/Python/training_split/C`, the same
  split as v1.1.0's T60);
- non-vegetated: `bare60_20261004/Python/models/B60S1` (seed 1, the v1.0.0 bare split, as v1.1.0's B60S0).

Their held-out scores differ from seed 0 by about 0.002 MAE (`Trees60_then_MC64_Gated_Results.md`, `Bare60_Step0_Results.md`). What is
missing is their **maps**. This plan makes them with exactly the commands that made v1.1.0 (`cand60med_20261004`), swapping only the two
model paths.

## 3. Steps

Run: `FullRuns/EXP_b31_seedmaps_20261006/` (EXP_ prefix and EXPERIMENTAL.md: a yardstick, not a product). Fully detached, with status
markers, as usual (adapt-detached-run skill). Inputs read-only.

**Step 0, checks (minutes, fatal).**
- The b31 tree split (rebuilt from `mc_rerun_20261002/Python/split/tree`) and v1.1.0's tree split (`qaC.../training_split/C`) have the
  same training keys. If not, say so in the results: the seed yardstick then still measures refit noise, but b31 also differs in rows.
- Same for non-vegetated (`mc_rerun_20261002/Python/split/bare` vs `nonveg_bit15_models_20261001/Python/training_split`).

**Step 1, exactness gate (about 5 min, fatal).** With this run's copy of `cand60med`'s `predict()` and the **seed-0** models (T60, B60S0),
predict h12v09 and require it to equal the v1.1.0 map (`cand60med_20261004/out/maps/CAND`) pixel for pixel, all three layers
(`maps_equal.py` from `v110_sd_20261004`). This proves the commands, container, code and metrics match v1.1.0, so any seed-1 difference
is the seed alone.

**Step 2, seed-1 maps (about 1-1.5 h).** 69 tiles (`tiles69.txt`), T60S1 + B60S1, MEDIAN both layers, the same metrics directories per
tile as `cand60med` (`manifests/tasks.txt`), output `out/maps/SEED1`. As in cand60med: 1 map at a time on a forest node, 2 on an ngaproc
node, 90 GB free memory before each start, 2 attempts per tile. For reference, cand60med's 69 maps took 1 h 25 min across its nodes.

**Step 3, figures and numbers (about 10 min).** Per tile, one figure with two rows (tree cover, non-vegetated) of four panels:

| v1.1.0 | b31 minus v1.1.0 | seed 1 minus v1.1.0 (noise) | b31 change beyond the noise |

- Panels 2 and 3 use the same colors as now (red lower, green higher, gray within ±2, ending at ±10), so they can be compared by eye.
- Panel 4: per 6 x 6 block (1.5 km), the b31 mean change, shown only where it is larger than the seed mean change on that block by more
  than the tile's seed spread (the 95th percentile of |seed block mean change| on that tile); elsewhere gray. That is the band-31 signal.
  (Option: show panel 4 per pixel instead; see decision Y2.)
- The subtitle keeps the tile's held-out MAE change (b31 vs CTRL) and adds the seed-1 held-out change on that tile, so better/worse can
  also be read against the noise.

Table `tile_stats_seed.csv` per tile and layer: mean change and mean |change| for b31 and for seed 1, share of blocks beyond the noise,
mean shift on those blocks, band-31 clamped yes/no. Summary: clamped vs never-clamped tiles. On never-clamped tiles, panel 4 should be
almost empty; if it is not, the yardstick is too loose and the results say so.

**Step 4, write-up.** `Reports/Band31_Unclamped_Results.md` (not yet written): the held-out result (fails D9 (b): non-vegetated better
-0.016, tree cover worse +0.022), the statistics figures already made (`out/figures/fig1-5`), and these maps, with the tiles worth
looking at.

## 4. Time and nodes

About 1.5-2 h in total, almost all of it step 2. No training. Disk: about 69 x 3 x 46 MB = 10 GB of Int16 maps (or delete the
NonTree layer after the figures; see Y3).

## 5. Decisions

| # | Question | Options | Suggested |
|---|---|---|---|
| Y1 | Nodes | (a) forest203 and forest205 (b) ngaproc201/202 (c) all four | **(c)** if no one else is on them at launch, else whatever is free; the other-user watcher runs as usual |
| Y2 | Panel 4 | (a) 6 x 6 blocks, beyond the tile's seed spread (b) per pixel, beyond ±(seed |change| 95th pct) | **(a)**: per-pixel noise is too large to separate a real shift from scatter |
| Y3 | Keep the seed-1 maps | (a) keep all three layers (b) keep tree and non-vegetated only | **(a)**: they are the yardstick for any later v1.1.0-recipe experiment, not only this one |
| Y4 | Also a seed-1 vs seed-0 map pair for C6 / C6.1 comparisons (compare_maps.py) | (a) no (b) yes, about 15 min more | **(a)** for now; the question here is b31 against v1.1.0 |
