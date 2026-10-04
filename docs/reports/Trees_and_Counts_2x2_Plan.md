# More trees and cloud counts for v1.0.0's tree model: a 2 × 2 test (plan)

> **FULLY EXPERIMENTAL.** Nothing here changes v1.0.0 or the delivery branch. The models, counts and scores are test products
> only. No candidate is adopted by this work; any adoption is a separate decision by the project lead.

**Written 2026-10-03. Approved the same day by the project lead (2 × 2 design, zenith-gated cloud count) and run 08:57-09:36; results in `Trees_and_Counts_2x2_Results.md`.** This is an
experiment to improve v1.0.0 (names in `Product_Versions_and_Work_Map.md`). Tree cover only.

## Why

1. **More trees.** In the blend test (`V4r_Cloud_Blend_Results.md`, section 2b), blending v1.0.0 with a seed refit lowered held-out MAE
   by 0.047 (−0.15 at 5+ fallback periods), with only partial weights. Averaging two forests is in effect a larger forest. It reduced
   scatter but did not move dense cloudy pixels into the right label bin.
2. **Counts as predictors.** Model N (2026-09-30) added a fallback count and a cloudy count to the 64 predictors. It gave a small but
   consistent gain (−0.03, no tile worse) and did not fix the h12v09 drop. Its cloudy count ignored zenith, so it counted low sun as
   cloud.
   - Counts let the model learn a correction that depends on both cloud count and cover type. That is what the V4r blends lacked: they
     raised open land as much as dense cover.
3. **A 2 × 2 design keeps the two effects separate.**

## Candidates (RF, `max_features='sqrt'`, `random_state=0`, MEDIAN, rounded; same 80/20 pixel split as v1.0.0)

| ID | Trees | Predictors |
|---|---|---|
| **v1.0.0** | 30 | the 64 C-code predictors (existing model `qaC_full63_retrain_20260929/Python/models/C/rf`) |
| **T60** | 60 | the 64 |
| **C30** | 30 | the 64 + `fallback_count` + `cloud_z_count` |
| **C60** | 60 | the 64 + the two counts |
| SEED1, SEED2 | 30 | the 64, seeds 1 and 2 (existing, `mc_rerun_20261002`): the run-to-run yardstick |

**The counts (0-8 per pixel):**
- **`fallback_count`:** periods where no input passed v1.0.0's QA. This is the 2026-09-30 count, from `qa.npy`, reused.
- **`cloud_z_count` (new):** fallback periods where every input that exists is cloud-flagged (cloud bits ≠ 0 or adjacency bit 6), at
  least one input exists, and **at least one input has 0 < solar zenith < 65**. So low-sun fallback is not counted as cloud.
  - Unlike the V4r count, it does not require an eligible pick.
  - **Checks:** cloud_z_count ≤ fallback_count, and the V4r zenith-gated count (CUR_Z) ≤ cloud_z_count, on every pixel of every tile.

## Steps (run root `FullRuns/trees_counts_2x2_<date>/`, ngaproc201/202, fully detached)

1. **Count rasters, 63 tiles** (about 10 min): `cloud_z_count`, with the checks above.
2. **Split** (about 10 min): the 2026-09-30 counts split (`rf_fallback_counts_20260930/split`, row for row identical to v1.0.0's
   split, with keys), plus `cloud_z_count` attached by pixel key (orientation count[x, y], as verified in `v4r_blend_20261003`).
   **Checks:**
   - rows and keys equal the counts split;
   - the 64 predictors and the labels equal v1.0.0's split;
   - `fallback_count` equals the 2026-09-30 column.
3. **Fits** (about 30 min; at most 2 per node): T60 on v1.0.0's split; C30 and C60 on the new split. Same `fitAltModel.py` command as
   the seed refits, except for `--nEstimators` and the predictor list.
4. **Held-out scoring, all 63 tiles** (`heldout_multi.py`, as before, with the same pixels and keys checked). Then the by-bin scoring:
   - groups: cloud_z_count (0 / 1-2 / 3-4 / 5+) × label bin, and v04+ against v00-v03;
   - measures: same / lower / higher bin share, bias and MAE;
   - differences from v1.0.0 with bootstrap intervals over tiles and blocks, and tiles better or worse, next to SEED1-SEED2.
5. **Report:** `Trees_and_Counts_2x2_Results.md`, and the README is updated.

**Not in this plan:** maps (they can be added for the 7 cloudy tiles if a candidate looks worth it), non-vegetated, and a 60-tree seed
yardstick.

## Reading the results (the project lead decides)

- **T60 against v1.0.0:** the effect of more trees alone. Judge it against the seed noise, and against the cost: model file and
  prediction time about double.
- **C30 against v1.0.0:** the counts alone. The key question is whether dense-label pixels at 5+ cloud-caused periods move into the
  right bin, without pushing open-land labels up.
- **C60 against T60 and C30:** whether the two effects add up.
- **Labels are judged by bin** (Landsat, early 2000s, sites chosen to avoid change). They are comparisons, not truth.

**Cost (estimate):** about 1-1.5 h, all on ngaproc201/202.
