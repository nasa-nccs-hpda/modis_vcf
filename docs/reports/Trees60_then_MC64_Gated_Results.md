# 60 trees, then the Monte Carlo 64 predictors, in the gated V4r blend (results)

> **FULLY EXPERIMENTAL.** Nothing here changes v1.0.0, v1.0.1 or the delivery branch. A and B are test products; adoption and
> version naming are for the project lead.

**Status 2026-10-04: finished.** Run `FullRuns/trees60_mc64_gated_20261003` (ngaproc201/202) launched 2026-10-03 15:34 and reached
`ALL.done` at 22:36, about 7 h against the plan's 10-11 h. Plan: `Trees60_then_MC64_Gated_Plan.md`.

## Summary

- **A (v1.0.1 recipe with 60 trees) is better than v1.0.0 and v1.0.1 on held-out labels:** −0.118 MAE overall, better on 57 of 63
  tiles. That is the whole T60 gain (−0.119) carried through the blend. It also holds in the cloudiest pixels (5+ periods): −0.13 for
  dense labels (41 and up), −0.16 for open labels.
- **B (A with the Monte Carlo 64) is worse:** +0.28 MAE against A overall (better on 9 of 63 tiles). The gap is widest where MC64 was
  supposed to help, dense labels under heavy cloud: +0.57 MAE, and 2 points fewer pixels in the correct bin. MC64's earlier +0.28
  disadvantage on v1.0.0's inputs is unchanged inside the blend.
- **Seed yardstick:** a second T60 seed and its blend differ from the first by about 0.002 MAE. Both of the results above are far
  outside that.
- **Maps:** A agrees with every reference more closely than v1.0.0 does overall (MAE against C6 4.72 vs 4.92; C6.1 5.79 vs 5.96;
  C code 5.24 vs 5.40). At 5+ cloud periods, A is raised by 2.2 points against v1.0.0 and sits further from all three references.
  That raise was also seen in v1.0.1 and is the intended effect, since C6 itself is cloud-affected. B agrees less with every
  reference than A does.
- **Reading:** 60 trees is worth keeping; MC64 is not. A is the candidate the project lead should look at.

## What ran

All steps completed and every check passed:
- self-test on h12v09;
- v4m band-by-band checks on 63 tiles;
- split acceptance test;
- blend R (v1.0.1 rebuilt) reproducing v1.0.1 on all 63 tiles.

Products:
- fits: V4R60, T60 seed 1, MC64-60 and V4R-MC64-60;
- maps: 252;
- blends: 189;
- held-out scores and figure sets A and B.

**One failure, now fixed:** the score step (`score_ab.py`) failed at 22:27 on 2026-10-03. `lane.sh` passed `-o` instead of
`--o`, so it stopped at argument parsing with rc=2. The script was corrected and the step was rerun alone on the existing held-out
predictions (`scripts/rerun_score.sh`, ngaproc202, done 2026-10-04 07:05:37). The rerun finished with `score.done`. The original marker and
log are kept as `*.20261003`.

ngaproc201 did not answer ssh on 2026-10-04 ("No route to host"). Check it before the next run.

## Held-out labels (2,289,990 pixels, 63 tiles)

MAE difference (negative = better). The interval is the tile bootstrap. "Better" counts tiles.

| Product | vs | All pixels | Clear (cloud 0, v04+) | 5+ periods, label ≥ 41 | 5+ periods, label ≤ 40 |
|---|---|---|---|---|---|
| v1.0.1 | v1.0.0 | −0.000 | 0.000 | +0.05 [−0.28, +0.22] | −0.03 [−0.10, +0.03] |
| T60 | v1.0.0 | −0.119 [−0.137, −0.102], 57/63 | −0.098 | −0.16 | −0.18 |
| **A** | v1.0.0 | **−0.118 [−0.137, −0.099], 57/63** | −0.098 | −0.08 [−0.42, +0.05] | −0.19 [−0.25, −0.07] |
| **A** | v1.0.1 | **−0.118 [−0.135, −0.101], 57/63** | −0.098 | −0.13 [−0.21, −0.07], 18/23 | −0.16 [−0.20, −0.07], 18/27 |
| MC64 | T60 | +0.282 [+0.219, +0.351], 9/63 | +0.197 | +0.60 | +0.16 |
| **B** | A | **+0.282 [+0.223, +0.352], 9/63** | +0.197 | +0.57 [+0.34, +1.74], 5/23 | +0.15 [−0.02, +0.21] |
| B | v1.0.1 | +0.164 [+0.111, +0.221], 15/63 | +0.099 | +0.43 | −0.01 |
| A seed 1 | A | −0.002 [−0.007, +0.003] | +0.001 | +0.01 | +0.01 |

- **Same-bin share:**
  - A against v1.0.1: +0.3 points overall, +0.7 for dense cloudy labels.
  - B against A: −0.7 overall, −2.1 for dense cloudy labels.
- **Per-model scores** (`out/heldout/differences.csv`) agree:
  - V4R60 is about even with v1.0.0 (+0.004) and gains 0.12 over 30-tree V4R.
  - V4R-MC64-60 is +0.29.

## Maps (63 tiles, pixel-weighted)

| | MAE vs C6 | vs C6.1 | vs C code | Change vs v1.0.0 at 5+ periods | Seam step (blend / clear-sky) |
|---|---|---|---|---|---|
| v1.0.0 | 4.92 | 5.96 | 5.40 | | |
| A | 4.72 | 5.79 | 5.24 | +2.22 | 9.05 / 9.79 |
| B | 5.08 | 6.05 | 5.54 | +2.61 | 8.61 / 9.45 |

At 5+ periods, all MAEs are higher than v1.0.0's (A: C6 8.64 vs 8.18). This is the cloud raise, and the references are themselves
cloud-affected, so it is not counted against A (see `vcf-c6-comparison-not-truth`).

Figures:
- `Reports/figures_trees60_mc64_A_20261003/`
- `Reports/figures_trees60_mc64_B_20261003/`

## Points for the project lead

1. Adopt 60 trees in the gated blend (A)? Held-out labels and all three references agree.
2. Drop MC64 for tree cover? This matches `Monte_Carlo_Rerun_Results.md`.
3. Non-vegetated and other years are not covered here (see the plan, "Not in this plan").

## Where the results are

`FullRuns/trees60_mc64_gated_20261003/`:
- `out/heldout/` (per-model scores);
- `out/scores/` (`diffs.csv`, `bins.csv`, `per_tile.csv` for the blends);
- `out/stats/` (map stats and seams per tile);
- `out/maps/`.
