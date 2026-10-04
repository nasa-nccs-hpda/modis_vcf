# Mean, middle-20 and median of the RF trees on 63 tiles (results)

**Run 2026-10-01, 15:45-18:46. Plan: `RF_Aggregation_Maps_63Tile_Plan.md` (approved).** Folder: `FullRuns/rf_aggregation_maps_20261001/`.
- Results: `out/summary_tree/summary.md`, `out/summary_bare/summary.md`, `out/ref_eval/summary.txt`.
- Held-out comparison of the same three methods (no refit): `RF_Tree_Aggregation_Test_Results.md`.

Products:
- **MEAN** is the current product: the mean of the 30 trees, equal to `predict()`.
- **MID20** is the mean of the middle 20 trees.
- **MEDIAN** is the median of the 30 trees.

MAE differences are MAE(A) − MAE(B), so a negative value means A is better. By the project rule, about 1 point or less counts as a tie.

## Summary

- **Non-vegetated: MEAN is better, and the result is consistent.** Against C6 and C6.1, MID20 is 1.3 points worse and MEDIAN 1.8-1.9 worse. Both are beyond the tie threshold, and MEAN is better on 47 of 54 tiles. Against the C code the differences are 1.0 and 1.5. On held-out labels the median is better by only 0.37, a tie. The median also deepens the existing low bias (-5.1 for MEAN, -7.9 for MEDIAN against C6).
- **Tree cover: the references disagree, and nearly every difference is a tie.**
  - **Map references (C6, C6.1, C code) and the stable set favour MEAN.** MID20 is 0.28-0.39 worse and MEDIAN 0.86-1.0 worse; MEAN is better on 42-49 of 54 tiles. On the stable set, MID20 is 0.37 worse and MEDIAN 0.91 worse.
  - **Lidar (NEON, G-LiHT) and held-out labels favour MEDIAN/MID20.** On all lidar pixels the gains are 0.2-0.6, and NEON's site-level CIs include zero. On the 379 stable G-LiHT pixels the gains are 1.8-2.5, though that set has only 6 groups. On held-out labels the median is 0.61 better.
  - **What the median changes:** it reduces dense-forest compression. In the 61+ bin, MAE against lidar improves by 1.4-2.4 points, and bias against C6 moves from -2.8 to +0.9. The cost is low cover: in the 0-20 bin, bias against C6 moves from +1.0 to -1.9, and MAE in the 21-40 bin is 2.5 points worse.
- **Suggested decision: keep MEAN for both layers.**
  - Non-vegetated is a clear result.
  - For tree cover there is no consistent winner, so the relative-difference rule gives no reason to change. MEAN is also what the model is trained to produce, and the SD layer is centred on it.
  - Caveat: the historical-based references are themselves means of 30 bagged models, so they may favour MEAN for structural reasons. This is the same caution as with the aerosol-era references.
  - If dense-forest compression is the priority, **MID20 is the compromise.** It loses about 0.3 against the maps and gains 1.4-1.5 against lidar in the 61+ bin.

## Run notes

- Step 0 gate passed at 15:34 on all 7 checks.
- **Attempt 1:** 51 of 63 tiles OK. 12 tiles failed with numpy out-of-memory errors on forest202, 203 and 204 (2 workers × 6 threads per node). Their markers are kept in `status/attempt1_oom_forest/`.
- **Rerun:** the 12 tiles were rerun 18:06-18:23 on ngaproc201/202 only (3 workers each), with the same scripts.
- **Integrity checks:** all 63 tiles have MEAN mismatch 0 against the stored bit-15 maps for both layers. ref_eval confirms that MEAN equals the stored `C_rf` at every reference pixel (8 of 8 references, share 1.0).

## Tree cover: maps (54 tiles without the C6.1 high-latitude flag)

| Reference | MEAN MAE | MID20 MAE | MEDIAN MAE | MID20 − MEAN | MEDIAN − MEAN | Tiles where MEAN is better (vs MID20 / MEDIAN) |
|---|---|---|---|---|---|---|
| C6 | 4.10 | 4.39 | 4.96 | +0.28 | +0.86 | 42 / 47 |
| C6.1 | 4.17 | 4.50 | 5.10 | +0.34 | +0.93 | 44 / 48 |
| C code | 4.43 | 4.82 | 5.42 | +0.39 | +1.00 | 45 / 49 |

Bias against C6 by C6 bin, for MEAN / MID20 / MEDIAN:

| C6 bin | MEAN | MID20 | MEDIAN |
|---|---|---|---|
| 0-20 | +1.0 | -1.1 | -1.9 |
| 21-40 | +1.0 | -0.6 | -1.2 |
| 41-60 | -1.2 | +0.2 | +1.3 |
| 61+ | -2.8 | -0.5 | +0.9 |

## Tree cover: independent and stable references

| Reference | n | MEAN | MID20 | MEDIAN | MID20 − MEAN (group CI) | MEDIAN − MEAN (group CI) | 61+ bin: MEDIAN − MEAN |
|---|---|---|---|---|---|---|---|
| NEON 3 m | 50,561 (19 sites) | 20.39 | 20.05 | 19.89 | -0.34 (-0.82, +0.19) | -0.51 (-1.29, +0.24) | -2.32 |
| G-LiHT 3 m | 2,663 (8) | 20.39 | 19.98 | 19.97 | -0.41 (-0.91, -0.01) | -0.43 (-1.07, +0.10) | -1.69 |
| G-LiHT 3 m stable | 379 (6) | 12.97 | 11.20 | 10.69 | -1.77 (-2.53, -0.21) | -2.28 (-3.32, -0.11) | -2.43 |
| Stable set (C6 2000-2020 mean) | 9,997 (3 tiles) | 5.13 | 5.50 | 6.03 | +0.37 (+0.15, +0.85) | +0.91 (+0.60, +1.60) | -0.49 |

NEON 2 m and 5 m, and G-LiHT 5 m, give the same pattern (`out/ref_eval/summary.txt`).

## Non-vegetated: maps (54 tiles)

| Reference | MEAN MAE (bias) | MID20 MAE (bias) | MEDIAN MAE (bias) | MID20 − MEAN | MEDIAN − MEAN | Tiles where MID20 / MEDIAN is better |
|---|---|---|---|---|---|---|
| C6 | 9.04 (-5.1) | 10.31 (-7.4) | 10.88 (-7.9) | +1.27 | +1.84 | 7 / 7 |
| C6.1 | 9.12 (-5.4) | 10.41 (-7.7) | 10.98 (-8.2) | +1.30 | +1.86 | 7 / 7 |
| C code | 10.96 (-4.6) | 11.94 (-6.9) | 12.46 (-7.4) | +0.98 | +1.49 | 14 / 8 |

The C code against C6 is 10.2. C6.1 against C6 is 2.0, so the two historical products agree closely on non-vegetated.

## Decisions for the project lead

1. **Final-pixel method:** MEAN suggested for both layers (MID20 as the dense-forest alternative for tree cover).
2. **Bare RF:** unaffected by this run. MEAN is the stored RF product already recommended in `NonVegetated_Bit15_Models_Results.md`.
3. **SD layer:** unaffected. The SD is the spread around MEAN.
4. **Then:** prioritize `Quality_Cloud_Layers_and_HDF_Output_Plan.md` (to-do item 9).

## Decisions (project lead, 2026-10-01)

- **SD layer: approved** (delivery code and 63-tile SD; `SD_Layer_Plan.md` defaults).
- **RF for non-vegetated: approved.**
- **Tree cover: MEDIAN of the 30 trees,** not the MEAN suggested above. Reason given: the project lead values the greater spread, and believes the historical product is compressed toward the mean, so the map references built from it favour MEAN for structural reasons.
- **Non-vegetated method: open.** The project lead asked for the per-tile and bin detail below.

## Non-vegetated detail (for the open decision)

Full tables:
- `out/bare_detail/per_tile.md`: 63 tiles; MAE of MEAN, and the MID20 and MEDIAN differences, against C6, C6.1 and the C code;
- `out/bare_detail/region_bin.md`: region × C6 bin, MAE and bias.

Script: `scripts/bare_detail.py`, read-only on `out/summary_bare`.

**Per tile (54 unflagged tiles, against C6):**
- MEDIAN − MEAN has a median of +2.10 (range -1.39 to +5.66). It is above +1 on 41 tiles and negative on 7.
- MID20 − MEAN has a median of +1.38.
- **MEDIAN is better on 7 tiles,** all dry: h18v07, h20v06, h19v10, h19v11, h17v05, h21v06, h21v05. The flagged h16v01 is also better (-2.34).
- **MEDIAN is worst on temperate and boreal vegetated tiles:** h18v03 +5.66, h20v03 +4.77, h11v03 +4.42, h11v04 +3.93, h18v04 +3.39, h31v11 +3.36, h27v04 +3.31, h12v03 +3.30.
- C6.1 gives the same picture. Against the C code the differences are smaller (median +1.34).

**By bin, the median spreads the values here too:**
- **Desert (C6 61+), lifted toward C6:** Africa 10-30N bias goes from -6.3 to -4.6 (MAE -1.6), Africa 30-50N -12.8 to -10.4 (MAE -2.2), Asia 10-30N -2.0 to -0.5 (MAE -1.3).
- **Vegetated land (C6 0-40), pushed lower:** all products already read below C6 there, and the median goes further. North America 30-50N 0-20 bias goes from -6.7 to -9.5 (MAE +2.5). Europe 50-80N 21-40 goes from -13.9 to -20.1 (MAE +6.1). South America 10N-10S 21-40 goes from -19.0 to -24.8 (MAE +5.6).
- The 0-40 bins hold most of the land, so the pooled result follows them.

**The same structural caveat as for tree cover applies, but more weakly:**
- The retrained models read about 5 points less bare than C6, and so do the held-out labels (`NonVegetated_Bit15_Models_Results.md`). Part of the MEDIAN penalty against C6 may therefore be C6's own level, not error.
- On held-out labels the median is better on 20 of 20 tiles. In the 0-20 bin its bias goes from +1.9 to +0.4, so on labels it moves the low bin toward the truth, not away from it.
- Against C6 the median penalty (+1.8 pooled) is about five times its held-out gain (-0.37), and it is beyond the 1-point rule. For tree cover the penalty was +0.9 and the gain -0.61.

**Consistency between layers:** `Percent_NonTree_Vegetation` is the remainder (100 − tree − bare). Mixing MEDIAN tree with MEAN bare is valid, but that combination has not been produced. In this run, each product used one method for both layers. A one-tile check of the mixed remainder (negative values, sum cap) would be needed before production.
