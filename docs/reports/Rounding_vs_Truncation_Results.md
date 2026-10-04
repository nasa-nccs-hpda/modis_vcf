# Rounding instead of truncating the value cast (results, to-do item 2)

**Run 2026-10-01 20:43 to 2026-10-02 06:09, ngaproc201/202, fully detached. Plan: `Rounding_vs_Truncation_Plan.md` (approved).** Folder: `FullRuns/rounding_vs_truncation_20261001/`. Results are in:
- `out/summary_tree/summary.md` and `out/summary_bare/summary.md` (maps against C6, C6.1 and the C code);
- `out/ref_eval/summary.txt` (lidar and the stable set);
- `out/heldout/` (held-out labels).

Same models and settings as the delivered product (tree = MEDIAN, non-vegetated = MEAN, sum cap). The only change is how the float value is cast to an integer:
- `TRUNC` is the delivered product, `int16(v)`, which truncates;
- `ROUND` is `floor(v + 0.5)`, which matches the C code's `prediction + 0.5`.

## Summary

- **Every difference is a tie** (well under 1 point MAE), so the choice is one of principle, as the plan expected.
  - **Tree cover:** ROUND is better by 0.03 MAE against C6, C6.1 and the C code (on 40-41 of 54 tiles).
    - Lidar is split: NEON is 0.05-0.18 better, G-LiHT 0.25-0.33 worse.
    - The held-out labels show no difference (−0.001).
  - **Non-vegetated:** ROUND is better by 0.20-0.21 against C6 and C6.1 (48-49 of 54 tiles), and by 0.13 against the C code. Held-out labels: +0.04 (worse), a tie.
- **Rounding removes the downward half-point bias of truncation**, as expected.
  - Held-out bias: tree cover −0.48 → −0.17; non-vegetated −0.36 → +0.07.
  - Against C6, the bias is about 0.3-0.5 smaller for both layers.
- **The sum cap fires a little more often with ROUND:** 570,461 pixels against 472,428.
- **TRUNC reproduces the delivered maps exactly** on all 63 tiles, in all 3 layers.
- **Suggested:** adopt ROUND on principle, for unbiased integers and agreement with the C code (`MOD44B3.c` adds 0.5 before `floor`). No result argues against it. This is a project-lead decision; nothing in the delivered code has changed.

## Run notes

- The first pass (2026-10-01) lost 4 tiles to out-of-memory with 3 workers per node:
  - h27v04, h27v07 and h30v12 raised `ArrayMemoryError`;
  - h27v06 was killed by the kernel.

  The prediction script built the 64-column feature array twice per tile, once for each layer, on top of the ~42 GB of RF models. The 4 tiles were rerun on 2026-10-02 (05:36-05:50) with 1 worker per node. The old claims and markers are in `status/retry1_20261002/`.
- The final held-out step failed once because of a pandas name clash: a column named `at` collides with the `DataFrame.at` indexer. It was fixed (columns renamed; original kept as `scripts/heldout_round.py.orig_20261001`) and rerun alone (`scripts/rerun_heldout.sh`).

## Maps against C6, C6.1 and the C code (54 tiles without the C6.1 high-latitude flag)

| Layer | Reference | TRUNC MAE | ROUND MAE | ROUND − TRUNC (95% tile bootstrap) | Tiles where ROUND is better | TRUNC bias | ROUND bias |
|---|---|---|---|---|---|---|---|
| Tree cover | C6 | 4.961 | 4.927 | −0.034 (−0.053 to −0.013) | 41 / 54 | −0.97 | −0.66 |
| Tree cover | C6.1 | 5.097 | 5.067 | −0.030 (−0.050 to −0.009) | 40 / 54 | −0.91 | −0.60 |
| Tree cover | C code | 5.423 | 5.396 | −0.028 (−0.051 to −0.004) | 40 / 54 | −0.85 | −0.53 |
| Non-vegetated | C6 | 9.040 | 8.837 | −0.203 (−0.251 to −0.154) | 48 / 54 | −5.07 | −4.61 |
| Non-vegetated | C6.1 | 9.115 | 8.902 | −0.213 (−0.263 to −0.167) | 49 / 54 | −5.38 | −4.91 |
| Non-vegetated | C code | 10.962 | 10.833 | −0.128 (−0.190 to −0.066) | 35 / 54 | −4.60 | −4.14 |

All 63 tiles give the same picture (tree cover against C6: 4.920 → 4.890).

## Tree cover against lidar and the stable set

| Reference | n | TRUNC MAE | ROUND MAE | ROUND − TRUNC | Groups where ROUND is better |
|---|---|---|---|---|---|
| NEON 2 m | 50,561 | 20.62 | 20.44 | −0.18 | 16 / 19 sites |
| NEON 3 m | 50,561 | 19.89 | 19.75 | −0.13 | 15 / 19 |
| NEON 5 m | 50,561 | 19.41 | 19.36 | −0.05 | 14 / 19 |
| G-LiHT 3 m | 2,663 | 19.97 | 20.21 | +0.25 | 1 / 8 |
| G-LiHT 5 m | 2,663 | 23.70 | 24.03 | +0.33 | 1 / 8 |
| G-LiHT 3 m stable | 379 | 10.69 | 10.91 | +0.23 | 1 / 6 |
| Stable set (C6 2000-2020 mean) | 9,997 | 6.03 | 6.00 | −0.04 | 2 / 3 |

The split follows the sign of each reference's bias. NEON reads higher than the product (bias −4 to −12), so rounding up helps. G-LiHT reads lower (bias +11 to +18), so rounding up hurts.

## Held-out labels (20% split)

| Layer | n | TRUNC MAE | ROUND MAE | TRUNC bias | ROUND bias | Tiles where ROUND is better |
|---|---|---|---|---|---|---|
| Tree cover (MEDIAN) | 2,289,990 | 6.707 | 6.705 | −0.48 | −0.17 | 29 / 63 |
| Non-vegetated (MEAN) | 2,160,056 | 4.158 | 4.195 | −0.36 | +0.07 | 10 / 20 |

By label bin, ROUND is slightly better in the 61+ bin for both layers (tree cover 8.40 → 8.20, non-vegetated 3.91 → 3.76). It is slightly worse in the 0-20 bin (tree cover 3.77 → 3.86, non-vegetated 3.07 → 3.27), where truncation's downward push happens to help.

## Decision for the project lead

Adopt rounding (`floor(v + 0.5)`) in `VcfPredict` for both layers? It is a one-line change in the value cast plus a unit test, and the delivered maps would change by at most 1 point per pixel. Suggested: yes, on principle; the accuracy effect is a tie.

## Decision (2026-10-02)

**Project lead: adopt ROUND.** `VcfPredict._roundValue` (`floor(v + 0.5)`) replaces the `int16` truncation on both paths: the member pass and plain `predict()`. Unit tests are in `model/tests/test_VcfPredict_roundingHighLat.py`. Check: on h08v05 the new code must reproduce `out/ROUND` exactly (`FullRuns/final_round_v00_20261002`).
