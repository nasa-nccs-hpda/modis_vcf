# 60-tree non-vegetated model (results; step 0 of `Bare_Where_C6_Is_100_Investigation_Plan.md`)

**Status 2026-10-04: done.** Run `FullRuns/bare60_20261004`, 08:13-09:55 (about 1 h 40 min) on ngaproc202 and forest201-205.
ngaproc201 was unreachable. No adoption decision is made here; 60 trees was already adopted as the setting on 2026-10-04.

## What ran

- **Gate (passed):**
  - A 30-tree seed-0 refit with this run's commands reproduces v1.0.0's bare held-out predictions exactly: 2,160,056 rows, 0 differ.
  - The delivered models through this run's code reproduce v1.0.0's maps on h12v09 and h17v00 (Greenland metrics, v00 rule): all
    3 layers, 0 pixels differ.
  - The first gate attempt failed only because the map step did not create its output folder. The fix and the set-aside attempt are
    in `status/attempt1_gate_outdir/`; about 2 min lost.
- **Fits:** B60S0 and B60S1 are 60 trees, seeds 0 and 1, the C code's 21 predictors, MEAN, on the v1.0.0 bare split.
- **Maps:** 69 tiles (the 63 plus h15v02, h16v00, h16v02, h17v00, h17v01 and h17v02). Each map uses the delivered 30-tree tree model
  with B60S0, so only the bare model changes; everything else is as delivered.
- **Outputs:**
  - held-out scores: `out/heldout/`;
  - fully bare labels: `out/heldout/label100.csv`;
  - C6 = 100 screen: `out/c6_100_screen.csv`;
  - figures: `Reports/figures_nonveg_v100_b60_20261004/` (two rows: v1.0.0 and B60 against C6).

## Results

**Held-out labels (2.16 M pixels, 20 tiles):**
- **Overall:** B60 is a small, consistent gain. MAE is 4.195 → 4.149 (−0.046, tile interval −0.055 to −0.037), better on 19 of 20
  tiles.
- **Seed yardstick:** seed 1 gives −0.048, so the gain is far outside run-to-run noise (about 0.002 between seeds).
- **Bins and regions:** the gain is in every cover bin (largest in 21-40, −0.08) and every region; the biggest region is Africa
  10N-10S, −0.15 on a small n.

**Fully bare labels:** no change.
- On label = 100, both models predict 100 on about 87% of pixels, with a mean of 99.4 and an MAE of 0.62.
- Where labels exist, both models are right at 100 except in a few regions with few 100-labels:
  - Australia (n = 1,421): mean 90.4, 35% at 100;
  - southern and equatorial Africa (n ≈ 500-600): mean 77-88.

**Maps against C6 (69 tiles):**
- **Overall:** MAE is 8.78 → 8.63 and the share within ±5 is 44.3 → 45.0%. B60 is closer to C6 on 57 of 69 tiles, but 17 of those
  are ties (< 0.05).
- **Where C6 = 100** (107.6 M pixels): **nothing changes.**

  | | Mean | Exactly 100 | 90-99 | < 90 |
  |---|---|---|---|---|
  | v1.0.0 | 95.4 | 38% | 50% | 12% |
  | B60 | 95.4 | 38% | 50% | 12% |

## What it means for the bare investigation

- **60 trees does not address the "C6 = 100, new < 100" concern.** The gap stays: across all C6 = 100 pixels, half are 90-99 and 12%
  are below 90.
- **Aggregation (MEAN) is now the lead suspect for the 90-99 band, and the 60-tree result points the same way.**
  - On the ice tiles, more trees make exact 100s **rarer**:
    - h15v02: 3.6 → 1.3%;
    - h16v02: 2.3 → 0.5% (MAE against C6 4.58 → 4.90);
    - h17v01: 15.2 → 13.0%.
  - That is what a mean over trees does: one tree below 100 is enough to pull a pixel below 100, and more trees means more chances.
  - The deserts move the other way (h20v06 exact 100s 78.6 → 82.5%), so this needs the per-tree check (plan step 2a) before any
    conclusion.
- **Australia is not a C6 = 100 problem.** C6 is at 100 on less than 0.2% of land on all four Australian tiles (≥ 95: at most 0.14%).
  v1.0.0 is 9-14 lower than C6 across the middle of the range. This moves to the mid-fraction question. Note the theory that C6 was
  tuned for fully bare pixels, and the static models' half-year calendar shift (both in the plan's "Considerations").
- **Below-90 pixels** are concentrated in a few places:
  - h12v01 (small Arctic islands, where the labels say these pixels are not fully bare);
  - h11v02;
  - h18v07, h17v05 and h21v05 (the Sahara-stretch tiles);
  - h17v02 (Iceland);
  - h15v02.

  These are rule or label questions (plan step 4), not tree-count questions.

## Next

Plan steps 1-3 on the 60-tree model:
1. the per-tree check of MEAN against MEDIAN and upper quantiles at C6 = 100 pixels (2a);
2. the sum cap (2b);
3. C6 against the labels by bin, including label = 100 (2d);
4. the pixel traces.
