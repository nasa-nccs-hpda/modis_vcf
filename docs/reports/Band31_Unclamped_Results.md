# Keep band 31's real values (no solar-zenith clamp): results

**Status 2026-10-06: done. The change does NOT pass D9 (b); recommendation: do not adopt.** v1.1.0 is unchanged.
Plan: `Band31_Unclamped_Plan.md` (U1-U6 approved 2026-10-05 late evening). Runs (both EXPERIMENTAL, no failures):
`FullRuns/EXP_b31_unclamped_20261005` (count, rebuild of 38 tiles, refit, 69 maps, evaluation; ended 2026-10-06 06:15) and
`FullRuns/EXP_b31_seedmaps_20261006` (seed-1 maps of the v1.1.0 recipe as the noise yardstick, `B31_Seed_Yardstick_Maps_Plan.md`; ended 10:30).
The ice-sheet reading uses `Ice_Extent_MOD44C_Reference_Results.md` (MOD44C ice reference, June-blue second pass).

## Bottom line

1. **Tree cover gets worse beyond refit noise, so D9 (b) fails.** Held-out tree MAE +0.022 [+0.011, +0.036] against the control. The seed
   change is −0.002 [−0.007, +0.002]. The loss is on the 38 tiles where the clamp fired (+0.040, 23 of 32 labelled tiles worse); elsewhere it
   is noise (+0.004).
2. **Non-vegetated gets slightly better.** Held-out bare MAE −0.016 [−0.033, −0.001] against a seed change of +0.0002 [−0.0045, +0.0042].
   Most of the gain is on the polar tundra tiles (h12v02 −0.083, h11v02 −0.068, h13v01 −0.053, h12v01 −0.039).
3. **On the Greenland ice it fixes h16v02 and breaks the tiles that were right.** On MOD44C interior ice, bare < 100 goes from 8.7% to 0.2% on
   h16v02, but from 0.04% to 45.7% on h17v01 and from 0.02% to 17.7% on h16v01. Seed 1 alone gives 13.6% and 4.3% there, so b31 is far
   beyond noise. On the MCD12-selected ice sheet, pixels with bare < 100 go from about 2.35 M to 4.65 M.
4. **The maps are not closer to C6, C6.1 or the C code.** Tree cover moves further from all three, by more than the seed change. Non-vegetated
   moves by about the seed change.

The caution note in the plan (the clamp was a deliberate design choice; look for its reason and for implausible unclamped values before any
adoption) was not worked through, because the change fails on the evaluation alone. It stays open only if the idea comes back.

## What was run

- **Change:** `BandDayFile._getRaster` keeps Band_31's raw value at any solar zenith (no ">70° -> 16000" clamp). Nothing else changed
  (QA test, compositing, −28672 handling, Greenland NDVI rule, v00 rule, predictors, 60 trees / seed 0 / sqrt / MEDIAN).
- **Count:** the clamp changes at least one band-31 composite on **38 of 69 tiles** (`manifests/tiles_rebuild.txt`), all high latitude or
  mid-latitude winter; h16v02: 100% of land.
- **Checks, all passed:** the h16v02 self-test (rebuilt band 31 equals the diagnosis's independent unclamped rebuild; other bands
  byte-identical); the gate (h12v09, h20v06: tree and bare training rows reproduced exactly); the seed-map run's row check (b31 and v1.1.0
  splits have the same keys and labels) and map gate (v1.1.0 models reproduce the v1.1.0 h12v09 map, three layers, 0 differences).
- **Control for held-out:** fill72_20261004 CTRL (the same base split refit with the same recipe). **Yardstick:** v1.1.0 seed 0 against
  seed 1 (T60S1 + B60S1), on the held-out labels and as maps on all 69 tiles.

## 1. Held-out labels (MEDIAN; base held-out sets)

| Layer | Scope | n | b31 − control MAE [95% over tiles] | Tiles better / worse | Seed 1 − seed 0 |
|---|---|---|---|---|---|
| Tree | all 63 tiles | 2,289,990 | **+0.022 [+0.011, +0.036]** | 22 / 38 | −0.002 [−0.007, +0.002] |
| Tree | the 38 rebuilt tiles (32 labelled) | 1,170,407 | +0.040 [+0.020, +0.064] | 8 / 23 | |
| Tree | other tiles | 1,119,583 | +0.004 [−0.002, +0.011] | 14 / 15 | |
| Bare | all 20 tiles | 2,160,056 | **−0.016 [−0.033, −0.001]** | 14 / 6 | +0.0002 [−0.0045, +0.0042] |
| Bare | rebuilt tiles (10 labelled) | 930,522 | −0.035 [−0.062, −0.004] | 8 / 2 | |
| Bare | other tiles | 1,229,534 | −0.001 [−0.004, +0.002] | 6 / 4 | |

Bare by tile (largest changes): h12v02 −0.083, h11v02 −0.068, h13v01 −0.053, h12v01 −0.039, h20v09 +0.054 (3,282 labels), h16v01 −0.006.
The "other tiles" lines are a built-in check: where the clamp never fired, b31 behaves like a refit (noise-sized).

Files: `EXP_b31_unclamped_20261005/out/compare/` (compare.csv, per_tile_*), `EXP_b31_seedmaps_20261006/out/heldout/{tree,bare}/`.

## 2. Maps against v1.1.0, with the seed yardstick (69 tiles)

Per tile, a 6 × 6-block change counts as "beyond noise" when it exceeds the 95th percentile of the seed-1 block changes on that tile;
noise alone gives about 5% (`EXP_b31_seedmaps_20261006/out/figures/`, one figure per tile, `tile_stats_seed.csv`).

| Layer | Blocks beyond noise, mean of tiles | Clamped tiles (38) | Unclamped tiles (31) | Seed 1 (same rule) |
|---|---|---|---|---|
| Tree | 8.4% | 11.2% | 5.0% | 4.5% |
| Bare | 8.7% | 11.9% | 4.9% | 4.8% |

Tiles above 15% beyond noise: bare h17v01 38, h12v01 31, h13v01 28, h16v01 28, h23v02 21, h12v02 19; tree h18v03 27, h12v03 24, h18v04 21,
h23v03 20. On the unclamped tiles b31 is at the noise level, as it should be.

## 3. Maps against C6, C6.1 and the C code (comparisons, not truth; 63 tiles, 1.107 G pixels)

| Layer | Reference | v1.1.0 MAE | seed 1 | b31 |
|---|---|---|---|---|
| Tree | C6 | 4.653 | 4.655 | **4.680** |
| Tree | C6.1 | 5.698 | 5.698 | **5.774** |
| Tree | C code | 5.144 | 5.144 | **5.175** |
| Bare | C6 | 10.645 | 10.709 | 10.664 |
| Bare | C6.1 | 17.268 | 17.326 | 17.335 |
| Bare | C code | 14.063 | 14.131 | 14.142 |

Tree cover moves away from all three references by 0.03-0.08, against a seed change of at most 0.002. Non-vegetated moves by about the seed
change (b31 and seed 1 are within 0.05 of each other). The eval's 69-tile pooled tables (`out/eval/refs_pooled.csv`) agree.
Files: `EXP_b31_seedmaps_20261006/out/compare_maps/summary_*/pooled.csv`.

## 4. Greenland ice

**MOD44C ice reference** (June-blue, all ice bodies; bare < 100, %, v1.1.0 / b31 / seed 1):

| Tile | Interior (> 2 km) | Edge (≤ 2 km) |
|---|---|---|
| h16v02 | 8.7 / **0.2** / 3.5 | 26.8 / **3.1** / 14.3 |
| h16v01 | 0.02 / **17.7** / 4.3 | 0.9 / **15.7** / 7.2 |
| h17v01 | 0.04 / **45.7** / 13.6 | 0.2 / **13.6** / 3.8 |
| h15v02 | 6.0 / **17.5** / 4.5 | 6.7 / **12.5** / 4.7 |
| h16v00, h17v00 | 0 / 0 / 0 | 0 / 0 / 0 |

**MCD12-selected ice sheet** (reference only; pixels with bare < 100): h16v02 1,921,746 → 201,773; h16v01 154,353 → 1,599,511;
h17v01 148,200 → 2,595,020; h15v02 → 257,193; total about 2.35 M → 4.65 M. Tree > 0 on the sheet: 426,157 pixels with b31 (tree mean ≤ 0.023).
Most of the new < 100 values are 90-99 (bare < 90: 681,061 pixels in all).

**Reading.** Unclamped band 31 helps exactly where h16v02's ice had no training rows that looked like it, and hurts where the model was
already right (h16v01, h17v01), because the refit changes how those tiles' ice is learned. This is why the next step targets the training
rows on h16v02 instead (`Ice_Training_Rows_h16v02_Plan.md`, approved and running 2026-10-06 as `EXP_ice_rows_h16v02_20261006`).

## 5. Decision

- **D9 (b) (bare better and tree unchanged or better): fails on tree cover.** Bare is better on held-out labels, but tree is worse beyond the
  seed yardstick; the maps are not closer to any reference; the Greenland ice gets worse overall.
- **Recommendation: do not adopt.** Keep the clamp as in v1.1.0 (and the C code). No version name needed (U6).
- If band 31 comes back later, the caution note in the plan applies first (reason for the clamp; implausible unclamped values).

## Files

- `FullRuns/EXP_b31_unclamped_20261005/`: `EXPERIMENTAL.md`, `manifests/tiles_rebuild.txt`, `out/count/`, `out/selftest/h16v02.log`,
  `logs/gate_*.json`, `Python/models/NEW_{tree,bare}`, `out/maps/NEW/`, `out/compare/`, `out/eval/`, `logs/ice_counts.log`, `out/figures/`.
- `FullRuns/EXP_b31_seedmaps_20261006/`: `out/heldout/`, `out/figures/` (69 yardstick figures, `tile_stats_seed.csv`), `out/compare_maps/`,
  `out/maps/SEED1/`.
- `FullRuns/EXP_ice_extent_june_20261006/out/tables/score_all3.csv` (MOD44C ice zones for v1.1.0, b31, seed 1).
