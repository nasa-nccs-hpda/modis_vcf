# Results: models scored on the 10,000-pixel stable reference set (2026-09-30)

Plan: `Stable_Set_Model_Eval_Plan.md` (approved without step 1, so there are no retrained-B predictions).
Pixel set: `Stable_Pixel_Reference_Selection.md`.
- **Code:** `FullRuns/stable_pixels_20260930/code/stable_set_eval.py`, run on forest203 in the container.
- **Output:** `stable_pixels_20260930/stable_set_eval/` (`stats.csv`, `paired.csv`, `pixels.csv`), log `logs/stable_set_eval.log`.
- **Reference:** the historical MOD44B C6 2000-2020 mean.
- **Products:** the bit-15 (C) Python models, the C-code 63-tile product, and historical 2019. Historical 2019 shows how close any 2019 map can get.
- **Weighting:** pooled numbers weight each 5-point sub-bin equally. Unweighted values differ by 0.1 or less pooled, and by at most 0.8 per tile.

## Answer

**On these pixels XGBoost tunedB ranks first, RF second and the baseline last. The gaps are small: by our rule (a difference of about 1 point MAE or less is a tie) every pooled difference is a tie.** The only difference over 1 point is XGBoost against the baseline in dense cover (60-100). This matches the other references built from the historical product. It does not settle the model choice, because the held-out training labels rank the models the other way (baseline best, XGBoost last by about 1.1). This is the outcome stated in the plan before the run, and the model choice stays open.

## Pooled (9,997 pixels; h11v04 2,989, h18v03 2,946, h19v09 4,062)

| Product | MAE | Bias | RMSE | r |
|---|---|---|---|---|
| historical 2019 (floor) | 3.39 | +0.64 | 4.60 | 0.983 |
| C-code product | 4.70 | +1.22 | 6.71 | 0.964 |
| C baseline | 5.72 | -1.67 | 7.52 | 0.955 |
| C RF | 5.13 | -1.41 | 6.74 | 0.964 |
| **C XGBoost tunedB** | **4.81** | -1.32 | 6.38 | 0.969 |

Paired MAE differences (95% bootstrap over 50x50-pixel blocks):

| Pair | All | h11v04 | h18v03 | h19v09 | 0-10 | 10-30 | 30-60 | 60-100 |
|---|---|---|---|---|---|---|---|---|
| XGBoost - RF | **-0.32** [-0.41, -0.23] | -1.07 | 0.00 | -0.16 | +0.11 | +0.06 | -0.19 | -0.96 |
| XGBoost - baseline | **-0.91** [-1.00, -0.82] | -1.18 | -0.72 | -0.80 | -0.42 | -0.70 | -0.63 | **-1.62** |
| RF - baseline | **-0.59** [-0.68, -0.50] | -0.11 | -0.72 | -0.64 | -0.54 | -0.75 | -0.45 | -0.66 |

## What the detail shows

- **XGBoost's advantage over RF comes from dense cover (60-100) and from h11v04.** At 0-30% cover RF is as good as or slightly better than XGBoost. On h18v03 the two are identical.
- **All Python models read low** (bias -1.3 to -1.7 pooled), most of all on h18v03 (Europe, -2.9 to -3.6) and at 60-100 (-1.8 to -3.0). The C-code product reads high (+1.2), most of all on h19v09 (+3.6).
- **XGBoost is about as close to this reference as the C-code product** (4.81 against 4.70). On h11v04 and h19v09 it is closer; on h18v03 it is 1.2 worse.
- **h19v09 (savanna)**, whose mid-range pixels vary by SD 7-8.5 over 21 years, ranks the models the same way as the other tiles, only noisier.

## Caveats

- **The reference is built from the historical product**, which had no working aerosol screen, so it favours whatever resembles it. Every reference of this kind (the full historical map, h11v02 stable pixels, this set) ranks XGBoost first. The held-out labels, the only reference here not built from the historical product, rank it last.
- **All three are training tiles.** Pixels are at least 3 pixels from any label, but the models saw other pixels of each tile. There are no pixels above 85% cover.
- **XGBoost values were clamped to 0-100 on 128 pixels** (h11v04 14, h18v03 69, h19v09 45; all were -2 to -7). The qaC predictions were made before the 2026-09-27 clamp fix, and the delivered code clamps the same way. Without the clamp these pixels would have to be dropped, which would flatter XGBoost. The same unclamped values were used in the earlier qaC scores of XGBoost (against historical and the held-out labels); clamping could only lower XGBoost's error slightly there.
- **Preflight:** CSV and raster grids agree at all 10,000 pixels; 3 pixels are water in every product.

## Where this leaves the model choice

| Reference | Ranking (best first) | Size of gaps |
|---|---|---|
| Historical MOD44B, 63 tiles | XGBoost, RF, baseline | 0.7-1.3 |
| Stable pixels, h11v02 | XGBoost, RF, baseline | 0.2-0.4 |
| **Stable set, 3 tiles (this report)** | XGBoost, RF, baseline | 0.3-0.9 |
| Held-out 20% labels, 63 tiles | **baseline, RF, XGBoost** | 0.1-1.1 |
| G-LiHT lidar, h11v02 | no separation | < 0.5 |

The historical-based references agree with each other, but they are not independent of each other. A reference that doesn't come from the historical product is needed to break the tie. NEON lidar (`NEON_Lidar_Eval_Plan.md`, steps 2-5) is the one available now.
