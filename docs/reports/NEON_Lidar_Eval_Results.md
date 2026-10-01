# Results: NEON airborne lidar evaluation of RF (bit 15) (2026-09-30)

Plan: `NEON_Lidar_Eval_Plan.md`, reduced to RF with the bit-15 screen (project lead, 2026-09-30). Steps 2-3 ran on ngaproc201/202 from 14:28 to 14:33, about 4 minutes, with no failures (`code/neon_ngaproc_chain.sh`). Step 5 ran automatically after them (`code/neon_eval_wait.sh`, `code/neon_eval.py`).
Output: `FullRuns/stable_pixels_20260930/neon_eval/` (`neon_eval_stats.csv`, `neon_label_check.csv`, `neon_eval_pixels.csv`), log `logs/neon_eval.log`.

- **Reference:** NEON 1 m CHM canopy cover (share of CHM cells above 2, 3 or 5 m) per 250 m MODIS pixel. Pixels are at least 90% covered by lidar. Lidar years: 2019 for 21 sites, 2018 for DEJU, GRSM and MLBS, and 2020 for RMNP.
- **Products (2019):** C_rf (Random Forest trained on the bit-15 screen), HIST (historical MOD44B C6) and Ccode (the C-code 63-tile product).
- **Scored pixels:** 50,561 on 8 tiles and 19 sites. They are not training labels, at least 3 pixels from any label, and valid in all products.
- **Not scored:** ABBY, SERC, SJER, WREF and YELL, because every lidar pixel there is a training label or next to one; they are used in the label check below. PUUM (Hawaii) is not on a training tile.

## Answer

**RF (bit 15) is as accurate as the historical MOD44B against NEON lidar.** At 3 m, MAE is RF 20.39, historical 20.43 and C-code 21.04. At 5 m it is 19.68, 18.63 and 19.20; at 2 m, 21.27, 21.86 and 22.45. Every difference is within about 1 point, so each is a tie. **All three products share the same large error pattern against lidar:** they read about 17-22 points too high where lidar sees under 10% cover, and about 26-30 points too low where it sees over 60%. That is the same compression of the range seen with G-LiHT on h11v02. RF compresses slightly less (bias -10.0 against historical -12.5 at 3 m).

## Pooled

| Pixels | Height | RF MAE / bias | Historical MAE / bias | C-code MAE / bias |
|---|---|---|---|---|
| all 50,561 | 2 m | 21.27 / -12.9 | 21.86 / -15.4 | 22.45 / -16.0 |
| | 3 m | **20.39 / -10.0** | 20.43 / -12.5 | 21.04 / -13.1 |
| | 5 m | 19.68 / -4.6 | 18.63 / -7.0 | 19.20 / -7.6 |
| stable (historical SD <= 5) | 3 m | 18.27 / -13.3 | 17.95 / -15.2 | 18.98 / -16.0 |

By lidar cover at 3 m (MAE / bias):

| Lidar cover | RF | Historical | C-code |
|---|---|---|---|
| 0-10 | 19.4 / +18.8 | 17.3 / +16.5 | 17.0 / +16.5 |
| 10-30 | 14.5 / +7.4 | 13.3 / +3.9 | 13.2 / +4.6 |
| 30-60 | 12.0 / -5.1 | 13.5 / -10.4 | 13.5 / -10.6 |
| 60-100 | 28.4 / -28.4 | 28.6 / -28.6 | 30.1 / -30.1 |

- RF is about 2 points worse than the historical product at low cover (0-30).
- RF is 1.5 points better at 30-60.
- RF and historical are equal in dense cover.

## By site (3 m MAE; RF minus historical)

- **RF better by more than 1:**
  - SOAP -5.9, BART -5.5, HARV -2.7, TEAK -1.6 (Sierra Nevada, New England);
  - SCBI -1.5, JERC -1.4, MLBS -1.2.
- **Tie (within 1):** BONA, CHEQ, CLBJ, DELA, GRSM, LENO, RMNP, STEI, UNDE.
- **RF worse by more than 1:** TALL +4.1, UKFS +2.0, DEJU +1.4.

## Training labels against lidar (19,671 label pixels inside the lidar, all 24 mapped sites)

| Height | MAE | Bias (label minus lidar) | r |
|---|---|---|---|
| 2 m | 16.1 | -9.8 | 0.885 |
| 3 m | 15.3 | -8.2 | 0.886 |
| 5 m | 14.3 | **-4.6** | 0.885 |

- The labels follow lidar closely (r 0.885), but they read lower than CHM-based cover at every height.
- 5 m fits best, and on h11v02 (G-LiHT) 3 m fitted best, so the best height cutoff differs by biome.
- The gap is largest at BART, WREF and SERC (labels 14-22 lower).
- **ABBY labels do not match the 2019 lidar** (r -0.17). ABBY is managed forest in Washington, and harvesting since the labels were made is the likely reason.
- Because the labels themselves sit below CHM cover, part of every product's gap against lidar is a difference in definition (MODIS canopy cover vs CHM crown cover), not model error.

## What this means

- **The choice of RF with the bit-15 screen holds up.** Against independent lidar it is as accurate as the historical MOD44B overall, better at 7 sites and worse at 3.
- **The largest error is shared by every product and by the labels:** dense forest reads 26-30 points below lidar, and open land reads 17-19 above it. Model or QA choices within our pipeline will not fix this. It comes from how cover is defined and from the labels. A calibration curve (from labels or products to lidar cover) would be the way to address it, if lidar-equivalent cover is ever the goal.
- **Caveat:** these are training tiles. The scored pixels are not labels, but the models saw the tile. The lidar is mostly from 2019, the same year as the products.
