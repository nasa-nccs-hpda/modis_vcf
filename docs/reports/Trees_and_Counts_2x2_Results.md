# More trees and cloud counts for v1.0.0's tree model: a 2 × 2 test (results)

> **FULLY EXPERIMENTAL.** Nothing here changes v1.0.0 or the delivery branch. The models, counts and scores are test products
> only. No candidate is adopted by this work; any adoption is a separate decision by the project lead.

**Run 2026-10-03, 08:57-09:36, ngaproc201/202, fully detached (`FullRuns/trees_counts_2x2_20261003`). Plan:
`Trees_and_Counts_2x2_Plan.md`.** All checks passed, and no task failed. Tree cover only; held-out labels only (no maps).

## Summary

| Candidate | What changes | Held-out MAE vs v1.0.0, all pixels | Dense labels (61+) at 5+ cloud-caused periods | Open labels (0-20) at 5+ |
|---|---|---|---|---|
| SEED1 / SEED2 (yardstick) | seed only | −0.009 / −0.002 | +0.06 / +0.02 MAE; same bin unchanged | no change |
| **T60** | 60 trees | **−0.119** [−0.138, −0.100]; better on 57 of 63 tiles | −0.08 MAE; same bin unchanged; mean change −0.02 | −0.15 MAE; no raise |
| **C30** | + fallback count + zenith-gated cloud count | **−0.037** [−0.049, −0.027]; better on 46 of 63 | **−0.56 MAE; same bin +1.5 points; raised +0.6** | −0.06 (interval spans 0); **not raised** (+0.00) |
| **C60** | both | **−0.152** [−0.176, −0.129]; better on 56 of 63 | **−0.68 MAE; same bin +1.5 points; raised +0.7** | −0.15 MAE; not raised (−0.08) |

Brackets are 95% bootstrap intervals over tiles. Labels are Landsat (early 2000s) and are judged by bin.

1. **More trees is a clear precision gain.** T60 lowers MAE by 0.12, about 12 times the seed noise, on almost every tile and in every
   cloud group. Same-bin agreement rises by 0.3 points overall.
   - It does not raise dense cloudy pixels: it reduces scatter, not the low bias.
   - The gain is larger than the seed-blend control suggested (−0.047), which used only partial weights.
2. **The counts do what the V4r blends did not:**
   - they raise dense cloudy pixels (+0.6 points at 5+ periods, +0.26 at 3-4) and move them into the right bin more often;
   - they **leave open land alone** (+0.00 at 5+);
   - the model learned a correction that depends on cover.
3. **But the correction is small.** At 5+ cloud-caused periods, dense-label pixels are on average about 11 points below the label
   (72.7 against 61.4; `V4r_Cloud_Blend_Results.md`, R1). The counts close about 0.6 of that, and 42.7% of those pixels stay a bin
   low (v1.0.0: 44.2%).
4. **The two effects add up.** T60 (−0.119) plus C30 (−0.037) ≈ C60 (−0.152).
5. **Cost of 60 trees:**
   - model files double (17 GB to 34 GB per tree model);
   - fits take about 12 min instead of about 7 here;
   - prediction time about doubles.

   **Cost of counts:**
   - two extra rasters per tile;
   - the cloud count needs the MOD44CQ state and zenith, which the pipeline already reads.

## Checks

- **Cloud count (63 tiles):** `cloud_z_count` ≤ `fallback_count`, and the V4r zenith-gated count ≤ `cloud_z_count`, on every pixel.
  Self-test on h12v09 before launch.
- **Split:**
  - rows and keys equal the 2026-09-30 counts split;
  - the 64 predictors and the labels equal v1.0.0's split (9,150,994 train, 2,289,990 test);
  - `cloud_z_count` ≤ `fallback_count` on every row.
- **Held-out scoring:** the same pixels and keys for every model (checked by `heldout_multi.py`).
- **Models:** RF, `max_features='sqrt'`, `random_state=0`, MEDIAN, rounded. T60 and v1.0.0 are trained on v1.0.0's split; C30 and C60
  on the counts split (the same rows plus the two counts).

## Details (held-out labels, tiles v04 and south unless stated)

**MAE difference from v1.0.0 by cloud-caused periods** (tiles better / worse in brackets):

| | 0 | 1-2 | 3-4 | 5+ |
|---|---|---|---|---|
| n | 1,126,300 | 523,513 | 158,276 | 34,483 |
| SEED1 | −0.006 | −0.013 | −0.002 | +0.006 |
| T60 | −0.098 (41/3) | −0.136 (39/3) | −0.159 (30/6) | −0.161 (22/6) |
| C30 | −0.021 (28/15) | −0.027 (31/10) | −0.104 (25/10) | −0.178 (18/12) |
| C60 | −0.119 (40/4) | −0.165 (39/2) | −0.252 (30/8) | −0.310 (21/9) |

**Dense and open labels in cloudy pixels:** same / lower / higher bin, and mean change from v1.0.0.

| Cloud periods, label | n | v1.0.0 | T60 | C30 | C60 |
|---|---|---|---|---|---|
| 3-4, 61+ | 66,230 | 0.774 / 0.226 / 0 | 0.773 / 0.227, +0.02 | 0.780 / 0.220, **+0.26** | 0.781 / 0.219, **+0.29** |
| 3-4, 0-20 | 40,922 | 0.794 / 0 / 0.206 | 0.797 / 0.203, −0.13 | 0.793 / 0.207, +0.08 | 0.794 / 0.206, −0.02 |
| 5+, 61+ | 5,232 | 0.558 / 0.442 / 0 | 0.557 / 0.443, −0.02 | **0.573 / 0.427, +0.60** | **0.573 / 0.427, +0.66** |
| 5+, 0-20 | 17,645 | 0.870 / 0 / 0.130 | 0.873 / 0.127, −0.13 | 0.870 / 0.130, +0.00 | 0.872 / 0.128, −0.08 |

MAE at 5+, 61+: T60 −0.08, C30 −0.56 [−1.62, −0.31], C60 −0.68 [−1.69, −0.43]. These intervals are wide, because there are few dense
held-out pixels at 5+ periods.

**Fit times:** T60 12.2 min, C30 6.6 min, C60 11.9 min. v1.0.0's 30-tree fit took about 12 min in the Monte Carlo run under different
load, so times are indicative only.

## Points for the project lead

1. **T60 is the strongest single result of the cloud and model experiments so far:** a robust −0.12 MAE on almost every tile.
   Adopting it would change v1.0.0's model (60 trees instead of 30) and double the model size and prediction cost.
   - Before any adoption, it would need maps (the 63 tiles, or the 7 cloudy tiles first) and the non-vegetated question (60 trees
     there too?).
2. **The counts are the first approach that raises dense cloudy pixels without raising open land.** The raise is real but small, a
   fraction of a point against an 11-point gap. Worth keeping if there is a use for it, but not a fix for the cloud problem on its own.
3. **The size of the gap, set against these small gains,** suggests that the cloudy dense pixels lack information that neither more
   trees nor counts can recover from the fallback composites. The `QA_Bits_Dense_vs_Open_Check.md` result points the same way: the
   model already ranks dense above open; it is the level that is too low.

## Caveats

- Held-out pixels are a random 20% of pixels, neighbours of training pixels, so the scores are optimistic. Only 5,232 dense held-out
  pixels have 5+ cloud-caused periods.
- No maps: the effect on full tiles, and against C6, C6.1 and the C code, is not measured.
- Tree cover only.

## Where the results are

All under `FullRuns/trees_counts_2x2_20261003/`:
- `Python/models/{T60,C30,C60}`;
- `Python/split_cz/` (with `split_check.json`);
- `Python/cloudz/` (63 tiles);
- `out/heldout/` (`predictions.parq`, `scores.csv`, `differences.csv`);
- `out/bins/` (`bins.csv`, `diffs.csv`);
- `logs/`, `status/`.
