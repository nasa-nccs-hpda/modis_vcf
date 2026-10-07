# Greenland/Iceland: zenith-only NDVI fix on the delivered product (results, plan v2, to-do item 4)

**Run 2026-10-01, 20:36-21:51, forest201-205 (1 worker each), fully detached. Plan: `Greenland_Iceland_Scoped_Fix_Plan_v2.md` (approved).** Folder: `FullRuns/greenland_zenith_v2_20261001/` (comparison in `out/compare/summary.md` and `stats.csv`; maps in `out/real` and `out/fixed`). Code: the frozen delivery commit `615557d`; tree = MEDIAN, non-vegetated = MEAN.

## Summary

- **Both gates passed.**
  - The polar-night tile h17v00 has 2 compositing periods with no input at all. `makeMetrics` and `vcfPredict` still completed, with valid predictions on 100% of 4,349,621 land pixels.
  - On h16v01 the fix lowered tree cover (0.40 → 0.01).
- **Greenland: the fix moves every tile toward C6 or leaves it level, and damages none.** The success criterion is met.
  - Non-vegetated, C6 MAE: h16v00 26.9 → 12.6, h17v00 21.2 → 10.2, h17v01 11.5 → 6.4, h16v01 10.2 → 5.5, h16v02 6.5 → 5.0. h15v02 is a tie (13.7 → 14.3).
  - Tree cover, C6 MAE: h16v00 6.95 → 2.94, h17v00 5.75 → 2.25. h16v01 and h17v01 were already below 0.5 and drop to ~0.0. h15v02 and h16v02 are unchanged.
- **Iceland (h17v02): no gain.** Tree cover is level (3.06 → 3.04). Non-vegetated gets slightly worse (10.6 → 11.5, +0.8). That counts as a tie, but the plan's rule for Iceland is "adopt only if MAE against C6 does not get worse", so **Iceland does not qualify.**
- **What remains after the fix:**
  - Non-vegetated stays biased low on every Greenland tile (−2.7 to −16.6 against C6).
  - The two polar-night tiles keep about 2-3% tree cover against C6's 0.

  The fix halves the bare error but does not reach C6. C6 and C6.1 reach ~100 on these tiles largely through the C code's high-latitude bare rule (`Bare_C_Regions_Investigation_Results.md`).
- **Suggested:** adopt the zenith-only fix for the Greenland tiles (h15v02, h16v00, h16v01, h16v02, h17v00, h17v01), not for Iceland. Whether a bare rule is also needed on top is a separate decision (see the bare-regions report).

## Zenith scan (step 1): share of land with no valid zenith per compositing period

"No valid zenith" means no raw day in the period with `0 < solz < 65`. Periods 5 and 6 on h16v00 and h17v00 have no input files at all (polar night). Every tile has at least one changed period, so all 7 went through the fix.

| Tile | P0 | P1 | P2 | P3 | P4 | P5 | P6 | P7 |
|---|---|---|---|---|---|---|---|---|
| h15v02 | 0.1 | 19.8 | 8.8 | 0.0 | 29.4 | 100 | 100 | 100 |
| h16v00 | 100 | 47.8 | 47.8 | 64.0 | 100 | 100 (no input) | 100 (no input) | 100 |
| h16v01 | 73.8 | 5.6 | 0.0 | 24.4 | 100 | 100 | 100 | 100 |
| h16v02 | 0.0 | 20.8 | 19.8 | 0.6 | 35.1 | 100 | 100 | 100 |
| h17v00 | 100 | 22.7 | 19.9 | 66.6 | 100 | 100 (no input) | 100 (no input) | 100 |
| h17v01 | 65.1 | 2.6 | 0.1 | 12.7 | 100 | 100 | 100 | 100 |
| h17v02 (Iceland) | 0.0 | 2.8 | 0.7 | 0.0 | 36.7 | 100 | 100 | 100 |

Per-period details are in `logs/zenith_<tile>.json`.

## Real against fixed (land pixels)

C6 is the primary reference. C6.1 sets tree = 0 / bare = 100 on v < 3 tiles. On h16v00 and h16v01, Ellesmere (Canada) and Greenland are split by an approximate Nares Strait line.

**Tree cover**

| Tile | Region | Real mean | Fixed mean | Real > 10 % | Fixed > 10 % | C6 MAE real → fixed | C6.1 MAE real → fixed |
|---|---|---|---|---|---|---|---|
| h15v02 | all | 0.36 | 0.41 | 1.30 | 1.38 | 0.96 → 0.94 | 0.34 → 0.38 |
| h16v00 | all | 6.95 | 2.94 | 17.15 | 1.72 | **6.95 → 2.94** | 6.97 → 2.92 |
| h16v00 | Greenland | 6.91 | 2.93 | 14.75 | 0.48 | 6.91 → 2.93 | |
| h16v00 | Ellesmere | 6.95 | 2.94 | 17.42 | 1.86 | 6.95 → 2.94 | |
| h16v01 | all | 0.40 | 0.01 | 0.58 | 0.02 | 0.44 → 0.06 | 0.39 → 0.01 |
| h16v01 | Greenland | 0.26 | 0.01 | 0.30 | 0.02 | 0.31 → 0.07 | |
| h16v01 | Ellesmere | 1.18 | 0.02 | 2.15 | 0.05 | 1.17 → 0.02 | |
| h16v02 | all | 0.03 | 0.03 | 0.10 | 0.10 | 0.16 → 0.15 | 0.03 → 0.03 |
| h17v00 | all | 5.75 | 2.26 | 8.92 | 0.80 | **5.75 → 2.25** | 5.76 → 2.25 |
| h17v01 | all | 0.31 | 0.01 | 0.31 | 0.01 | 0.33 → 0.03 | 0.31 → 0.01 |
| h17v02 | Iceland | 4.92 | 4.44 | 16.49 | 14.45 | 3.06 → 3.04 | 4.63 → 4.21 |

**Non-vegetated**

| Tile | Region | Real mean | Fixed mean | C6 MAE real → fixed | C6 bias real → fixed | C6.1 MAE real → fixed |
|---|---|---|---|---|---|---|
| h15v02 | all | 75.58 | 77.59 | 13.72 → 14.30 | +5.8 → +7.8 | 24.03 → 22.03 |
| h16v00 | all | 73.11 | 87.44 | **26.93 → 12.59** | −26.9 → −12.6 | 27.00 → 12.54 |
| h16v00 | Greenland | 71.13 | 83.47 | 28.90 → 16.55 | −28.9 → −16.6 | |
| h16v00 | Ellesmere | 73.34 | 87.89 | 26.71 → 12.14 | −26.7 → −12.1 | |
| h16v01 | all | 88.98 | 93.93 | **10.21 → 5.49** | −9.6 → −4.7 | 10.97 → 6.02 |
| h16v01 | Greenland | 91.95 | 95.58 | 7.11 → 3.75 | −6.4 → −2.8 | |
| h16v01 | Ellesmere | 72.85 | 85.01 | 27.20 → 15.01 | −27.2 → −15.0 | |
| h16v02 | all | 90.69 | 92.86 | 6.48 → 5.02 | −4.8 → −2.7 | 9.19 → 7.03 |
| h17v00 | all | 78.86 | 89.79 | **21.18 → 10.22** | −21.2 → −10.2 | 21.23 → 10.23 |
| h17v01 | all | 87.25 | 92.92 | **11.50 → 6.36** | −10.9 → −5.2 | 12.72 → 7.04 |
| h17v02 | Iceland | 54.87 | 59.07 | 10.63 → 11.45 | −2.8 → +1.4 | 44.26 → 40.06 |

The h16v01 labels (all 0% tree) give a tree MAE of 0.00 for both real and fixed.

## Notes

- **h15v02 (SW Greenland)** is the one Greenland tile where the fix does nothing useful. It has the most ice-free land among these tiles, and RF bare is already above C6 there (+5.8). The fix adds 2 points of bare. Both changes are under 1 point MAE, so they count as ties.
- **Iceland:** the fix shifts bare from 2.8 below C6 to 1.4 above it, so the bias improves while the MAE gets slightly worse. Iceland's periods 0-3 have almost no zenith problem, so little is expected to change there.
- **The polar-night tiles are the weakest.** On h16v00 and h17v00 two periods have no input and two more have no valid zenith anywhere. Even after the fix about 2-3% tree cover remains, where C6 has 0.
- The fix needs the MOD44CQ zenith data and a scoped switch in the pipeline (a tile list or a latitude test). Neither is in the delivered code; this run used a per-tile `metricsDir` built by the run's scripts.

## Decision for the project lead

1. Adopt the zenith-only NDVI fix for the 6 Greenland tiles? (Suggested: yes.)
2. Exclude Iceland? (Suggested: yes, by the plan's own criterion.)
3. If yes, how to put it into the pipeline: a fixed tile list, or a rule based on the zenith scan (for example, "any period with ≥ X% no valid zenith")? This needs its own plan before any code change.

## Decision and implementation (2026-10-02)

**Project lead: adopt the fix for the 6 Greenland tiles with a fixed tile list; exclude Iceland.**

- **Code:** commit `13773db` on `mcarrol2/c-python-parity-fixes` (local, not pushed; pushed since: on GitHub as of 2026-10-05).
  - `Metrics.ZENITH_ONLY_NDVI_TILES` lists h15v02, h16v00, h16v01, h16v02, h17v00 and h17v01.
  - `Metrics.getNdvi` calls `_applyZenithOnlyNdvi` for those tiles. It reads solar zenith for each day from MOD44CQ and counts a missing day as no valid zenith.
  - Unit test: `model/tests/test_Metrics_zenithOnlyNdvi.py`. It passes, and so do the 5 member-pass tests.
- **Checks** (`FullRuns/greenland_fix_adopt_20261002/`):
  - The patched NDVI is identical to this run's fixed NDVI on all 6 tiles, and identical to the real NDVI on h17v02.
  - On h17v00, end to end (makeMetrics + vcfPredict with the delivered settings), all 5 output layers are identical to `out/fixed`.
- **Note:** the step runs only when NDVI is computed. A metrics folder that already holds an unfixed `MOD44-NDVI.bin` for one of these tiles must be rebuilt.
- **Not covered by the fix:** about 2-3% tree cover remains on h16v00 and h17v00. Two compositing periods there have no input files at all, and the pipeline fills them with the substitute value 10000 in every band (NDVI 0). The C code's high-latitude rule (bare = 100, tree = 0) covers 100% of the land on both tiles; whether to use it is still open.
