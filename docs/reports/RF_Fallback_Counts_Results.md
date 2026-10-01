# RF with fallback and cloudy counts as predictors: results (2026-09-30)

Plan: the amendment in `Single_Tile_h12v09_Scenario_Plan.md`. Run root: `FullRuns/rf_fallback_counts_20260930/` (code, logs, status, `eval/`). The run took 15:45-16:05 on ngaproc201 and 202, with no failures.

## What was done

- **Counts for all 63 tiles** (`out/counts/<tile>_counts.npz`):
  - `fallback_count` = the number of periods (0-8) where no input passed QA (the stored `qa.npy`);
  - `cloudy_count` = the number of periods where no input has MOD44CQ state cloud bits = clear (zenith ignored).
  - Check: no pixel on any tile has more cloudy than fallback periods. On h12v09, 47% of fallback periods are cloud-flagged.
- **Split:** rebuilt with the two columns. The 116 existing columns and the labels are identical to `qaC.../training_split/C`, row for row (9,150,994 train and 2,289,990 test).
- **Models** (RF, 30 trees, `max_features` sqrt, as in the qaC run):
  - **G**, the existing model, seed 0;
  - **G1**, the same 64 predictors with seed 1 (seed noise);
  - **N**, the 64 predictors plus the 2 counts, seed 0.
- **Feature assembly check (h12v09 map):** all 66 features matched the training rows exactly at 24,860 pixels. G re-predicted from the rasters is within 1 of its saved map on 100% of pixels, with a mean of +0.50. That offset is a rounding convention: this script rounds, while the saved map truncates. G, G1 and N are all rounded the same way, so they compare directly; `G_saved` is 0.5 lower throughout.

## Held-out 20% labels (all 63 tiles)

| MAE | G | G1 | N |
|---|---|---|---|
| Pooled (2.29M px) | 7.319 | 7.316 | **7.284** |
| Fallback 0 / 1-2 / 3-4 / 5+ | 6.14 / 8.23 / 10.04 / 3.84 | 6.14 / 8.22 / 10.03 / 3.84 | 6.13 / 8.18 / 9.97 / 3.81 |
| Cloudy 0 / 1-2 / 3-4 / 5+ | 6.67 / 7.55 / 11.01 / 8.50 | 6.67 / 7.55 / 11.01 / 8.48 | 6.64 / 7.52 / 10.95 / 8.44 |

- **Per tile:** N is better than G by more than 0.1 on 8 tiles (best: h20v02 -0.26, h18v04 -0.15, h24v04 -0.15, h12v09 -0.14, h19v04 -0.14). No tile is worse by more than 0.01. The largest per-tile seed difference |G1 - G| is 0.21.
- **High-latitude and snow tiles are unchanged:** h16v01 0.02/0.02, h12v01 0.04/0.05, h13v01 0.13/0.14, h21v01 0.34/0.34, h11v02 7.30/7.26, h12v02 0.09/0.09 (G/N).

## h12v09 map (dense = historical >= 60; differences from the historical product)

| Pixels | G | G1 | N |
|---|---|---|---|
| Drop (103,011): mean minus historical | -21.7 | -21.4 | **-20.2** |
| Drop: share <= historical - 30 | 18.7% | 19.8% | **14.5%** |
| Dense, 5 fallback periods | -4.7 | -4.2 | -3.7 |
| Dense, 6 | -8.1 | -8.0 | -7.2 |
| Dense, 7 | -10.9 | -10.8 | -9.8 |
| Dense, 8: mean / share <= -30 | -15.8 / 12.8% | -16.0 / 14.2% | **-14.4 / 9.3%** |
| Dense, 0-4 | -1.2 to +0.1 | same | -1.3 to +0.2 |
| All dense / all land | -1.20 / +0.44 | -1.11 / +0.50 | -0.92 / +1.02 |

## Reading

- The counts help in the right place and hurt nowhere. At the most-fallback pixels, N is about 1-1.5 points less low. The share of drop pixels 30 or more below the historical product falls by about a quarter (18.7% to 14.5%). This is beyond seed noise, which is up to 0.6 in the same groups.
- They do not fix the drop. At 8 fallback periods N is still about 14 points low, and at the drop pixels about 20 low. The size of the gain is about the 1-point tie level. The RF gets information that a pixel is fallback-heavy, but it cannot recover the clear-sky signal the contaminated composites have lost.
- Held-out labels: a consistent, small gain (0.03 pooled, 0.07 at 3-4 fallback periods), and no tile worse. Greenland and the snowy tiles are unaffected, so the cloudy count does no harm there. Whether the cloudy count adds anything beyond the fallback count was not isolated (no fallback-only model).
- **Suggested next step:** keep the counts (free, safe, slightly positive). To fix the drop, change the inputs rather than just label them: S3 (fill fallback periods from the pixel's clean periods) is the next scenario in the plan. Optionally, add a fallback-only fit to see whether `cloudy_count` is needed.
