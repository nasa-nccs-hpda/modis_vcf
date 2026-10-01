# Plan: score the models on the 10,000-pixel stable reference set (h11v04, h18v03, h19v09)

**Written 2026-09-30. Status: DONE 2026-09-30 (approved without step 1). The preflight stopped on 128 negative XGBoost values (unclamped qaC predictions). The project lead approved clamping them to 0-100 as the delivered code does, and the scoring then ran. Results: `Stable_Set_Model_Eval_Results.md`.**

## Why

The set `FullRuns/stable_pixels_20260930/stable_pixels_subbin_10000.csv` was built on 2026-09-30 (`Stable_Pixel_Reference_Selection.md`) to score the models. It was never scored; only the h11v02 stable pool was (`Lidar_Reference_Coincidence_20260930.md`, "Same tile, stable-pixel reference").

The references now disagree about the tree-cover model:
- Against the historical MOD44B, and on h11v02 stable pixels, XGBoost tunedB is best.
- Against the held-out 20% labels, XGBoost is last on every tile, by about 1.1 MAE.
- Lidar on h11v02 does not separate the models.

This set adds three biomes (eastern US, Europe, central African savanna) and evenly covers 0-85% cover, so the answer is not dominated by 0-10% pixels as it was on h11v02 (79% of pixels there).

**Main question:** with the bit-15 (C) screen, how do baseline, RF and XGBoost tunedB rank on these pixels?
**Secondary question (optional step 1):** do B and C still tie here?

## What exists and what is needed

| Product | Status for h11v04, h18v03, h19v09 |
|---|---|
| C baseline / RF / XGBoost tunedB (`qaC_full63_retrain_20260929/Python/predictions_C/<model>/Percent_Tree_Cover-<tile>-2019.tif`) | **exist** |
| Historical MOD44B 2019, and the 2000-2020 mean (the reference) | in the CSV (`hist_2019`, `mean_2000_2020`) |
| C-code product (63-tile run, read as in `qaC_full63_retrain_20260929/scripts/q15_vs_b_hist.py`) | exists |
| Retrained B baseline / RF / XGBoost | **missing**: made so far only for h11v02 and h12v09 |

## Steps

**Step 0: preflight check (read only, about 2 minutes).**
- Check that every product file exists for the three tiles, and that it is on the same 4800x4800 grid as the CSV's row and column (spot-check a few pixels: `hist_2019` in the CSV must equal the historical 2019 raster).
- Count how many of the 10,000 pixels are valid in every product (not water 200, not out of projection 250, not fill). Stop and report if more than about 2% are lost, or if one sub-bin loses more than 10%.

**Step 1 (optional): retrained-B predictions for the three tiles.**
- Use `stable_pixels_20260930/code/run_predict_B_tile.sh <tile>` unchanged. It writes only under `gliht_eval/predictions_B/`, with markers in `gliht_eval/status/`.
- It took about 12 minutes per tile (3 models) on h12v09. One tile per node on forest nodes, fully detached (`setsid nohup`), so about 12-15 minutes of wall time.
- ngaproc201/202 are not used: the launcher calls `singularity` without `module load`.
- Needed only for the secondary question. B and C are already decided as a tie, so skipping this step is reasonable.

**Step 2: scoring (new script `stable_pixels_20260930/code/stable_set_eval.py`, a few minutes, on a forest node in the container).**
- **Reference:** `mean_2000_2020`. `hist_2019` is scored as a product too; it shows how close any 2019 map can get to the reference (the floor).
- **Per product:** MAE, bias, RMSE and r:
  - pooled over the set, with each 5-point sub-bin weighted equally as designed;
  - per tile, so h19v09 (noisy mid-range, decided 2026-09-30) can be read separately;
  - per coarse bin (0-10, 10-30, 30-60, 60-100).
- **Paired differences in MAE**, each with a 95% bootstrap interval. Resampling is by 50x50-pixel blocks within each tile, since pixels are at least 3 pixels apart but neighbours still correlate.
  - XGBoost minus RF, XGBoost minus baseline, and RF minus baseline (C models).
  - C minus B per model, if step 1 ran.
- **Output:** a new folder `stable_pixels_20260930/stable_set_eval/` (the script refuses to reuse it), with CSV tables and a log in `logs/stable_set_eval.log`.

**Step 3: results report.** `Stable_Set_Model_Eval_Results.md`, plus a one-line update to `README_START_HERE.md` and to the model-choice note in `Project_Summary_and_Conclusions.md`.

## How the result will be read (fixed before running)

- A difference of about 1 point MAE or less is a tie.
- A model is preferred only when references agree. If XGBoost again wins here, all the references built from the historical product favour it, but the held-out labels still do not. The model choice then stays open until an independent reference (NEON lidar, or labels on new tiles) breaks the tie.
- If XGBoost does not win here, that is evidence its historical-map advantage is a matter of resemblance, not accuracy.
- **Caveats:**
  - This reference is built from the historical product, so it favours whatever resembles it.
  - All three are training tiles (the pixels are at least 3 pixels from any label, but the models saw the tile).
  - There are no pixels above 85% cover.
  - h19v09 mid-range pixels have an SD of 7-8.5 over 21 years.

## Cost, and what is not touched

- About 15 minutes without step 1; about 30-35 minutes with it.
- Everything is written under `FullRuns/stable_pixels_20260930/` (new files and folders only) and `Reports/`.
- Nothing in rlgill's repositories, the pushed branch, or any existing run output changes.
