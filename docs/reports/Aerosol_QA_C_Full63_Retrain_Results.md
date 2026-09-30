# Results: 63-tile bit-15 (C) rebuild and B/C retrain (2026-09-30)

Plan: `Aerosol_QA_C_Full63_Retrain_Plan.md`. Tables (auto-generated): `Aerosol_QA_C_Full63_Retrain_Tables.md`.
Run: `FullRuns/qaC_full63_retrain_20260929/`, launched 2026-09-29 21:17, finished 2026-09-30 10:41 (`status/run.done`, no failures).
Delay: Stage 5b stalled 01:40-07:06 because the prediction output folders were not created. It was fixed by hand, with no relaunch (see the plan's restart note).
B = current screen (state bits 7-8, known to reject 29-50% of good pixels). C / Q15 = state bit 15.

## Decision (project lead, 2026-09-30)
**B and C are equivalent in accuracy. A tie favours C (bit 15).** Differences of about 1 point of tree cover MAE or less count as equivalent,
and every reference puts B vs C within that range. Details: `Lidar_Reference_Coincidence_20260930.md` (Decision section).

## Evidence
| Reference | Result (C minus B) |
|---|---|
| Held-out 20% pixels, 63 tiles (Stage 5) | MAE baseline 7.45 -> 7.25, RF 7.52 -> 7.32, XGBoost 8.58 -> 8.38; C better on 53 / 54 / 47 of 63 tiles; lower in every label bin |
| Historical MOD44B, 63 tiles (Stage 5b; leans toward C by construction) | pooled MAE baseline 5.08 -> 4.87, RF 4.51 -> 4.20, XGBoost 4.04 -> 3.61; C better on 48 / 53 / 58 tiles |
| G-LiHT lidar, h11v02 (2014 lidar, 2,663 pixels) | +0.77 / +0.76 / +1.09 (B better); stable pixels: not significant |
| Stable historical pixels, h11v02 | -0.23 / -0.13 / -0.14 (C better) |

## Things to watch before production
- **h16v01 (snow tile):** vs historical, C baseline MAE 10.19 vs B 6.36 (+3.8). RF +0.3, XGBoost -0.3. This is the one tile where one C model is clearly
  worse. Bit 15 removes about 21% of good pixels on this tile (variant test). Look at it before adopting C for snow and ice tiles.
- Largest mean tree cover shifts, C minus B (XGBoost), are within +/-1.6 points (h11v09 +1.6, h27v07 -1.4, h20v03 -1.4). The share of pixels changing
  by more than 20 points is <= 1.1% on every tile.
- h12v09 dense-forest drops are **not** caused by QA: they are cloud-fallback composites in the eastern Amazon, and B has them too, more often
  (`C_vs_Historical_h12v09_20260930.md`). Improving cloud-fallback handling is an open area for making all models better in cloudy regions.

## Not done
- Stage 6 (independent tiles): replaced by the stable-pixel and lidar references above. The NEON lidar plan (`NEON_Lidar_Eval_Plan.md`) is optional now,
  for describing accuracy by biome, not for the B/C choice.
