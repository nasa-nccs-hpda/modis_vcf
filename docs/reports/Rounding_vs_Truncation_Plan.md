# Rounding instead of truncating the predicted value (plan, to-do item 6)

**Written 2026-10-01 evening. Status: plan only, awaiting approval. Nothing has run.**

## The question

`VcfPredict` turns each pixel's float prediction into an integer by **truncation** (`astype(np.int16)`): 37.9 becomes 37. The C code **rounds**: `floor(x + 0.5)` for bare (`MOD44B3.c`), so 37.9 becomes 38.
- If the fractional parts are spread evenly, truncation lowers every value by about 0.5 on average.
- That is a systematic bias of about -0.5 on every layer, including the sum cap and the remainder layer.

The delivery commit kept truncation on purpose, so its maps could be checked exactly against the aggregation run. This plan measures what rounding would change on the delivered configuration: tree = MEDIAN of the 30 RF trees, non-vegetated = MEAN, sum cap as on the branch.

**Expected outcome, stated in advance:** the MAE change will be well under 1 point, a tie by the project rule. The decision will then rest on principle (unbiased rounding, and matching the C code's rounding of bare), not on accuracy. The run gives the measured size of the effect, so that choice is made on numbers.

## Method

Run root: `FullRuns/rounding_vs_truncation_<date>/`, new; earlier runs are mounted read-only.

0. **Automatic gate (h08v05, about 7 minutes).**
   - Predict with the frozen delivery commit (`delivery_sd_rf_median_20261001/code_commit_615557d`), keeping the float values for both layers.
   - The truncated result must equal the delivered map (`delivery_sd_rf_median_20261001/out/full63`) on every pixel for all three layers.
   - The rounded result must differ from the truncated one by 0 or +1 only, before the sum cap.
   - Any failure writes `status/step0.failed` and stops all workers.
1. **63 tiles, rounded products.** A small driver (`predict_round.py`) uses `VcfPredict._memberPredict` from the frozen commit, rounds with `floor(x + 0.5)`, then applies the branch's sum cap and remainder unchanged. It writes `out/ROUND/<layer>-<tile>-2019.tif` for the three layers. The SD is not recomputed, because it does not change.
   - The share of pixels whose value moves by +1 is recorded per tile and layer.
   - ngaproc201/202, 3 workers each, fully detached, with claims, markers and `ALL.done`.
2. **Comparisons, truncated (delivered) against rounded,** using the scripts from the aggregation run (`compare_agg.py`, `summarize_agg.py`, `ref_eval.py`), copied and pointed at the two product sets:
   - maps against C6, C6.1 and the C code: pooled over the 54 unflagged tiles and all 63, per tile, by the 4 reference bins, and by continent × latitude band; block and tile bootstrap CIs;
   - NEON (2/3/5 m), G-LiHT (3/5 m, all and stable), and the 10,000-pixel stable set;
   - **held-out labels:** tree cover on the bit-15 20% split, non-vegetated on its 20% split, with the same truncated/rounded pair computed from the member predictions on those rows.
   - Measures: MAE, bias, RMSE, and the change in the sum-cap count.
3. **Report:** `Reports/Rounding_vs_Truncation_Results.md`, and the README is updated.

## Expected cost

About 1.5 hours, unattended: step 0 about 7 minutes; 63 tiles on 6 ngaproc workers about 70 minutes, since metric reading dominates; comparisons about 20 minutes.

## Not in scope

- Changing the branch. If rounding is chosen, it is a one-line change in `VcfPredict` plus a test, and the delivered maps would be regenerated.
- Rounding the SD layers, which already use `np.round`.
