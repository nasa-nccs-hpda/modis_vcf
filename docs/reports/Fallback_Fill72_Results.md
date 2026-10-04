# Polar-night / low-sun fill experiment (fill72): results

**Status 2026-10-04: done; does not pass the project lead's rule. Not adopted.** Run `FullRuns/fill72_20261004`, 12:51-16:00, on
ngaproc201/202 and forest201-205. Plan: `Fallback_Solz_LE0_Plan.md` (revised version).

**What was tested:**
- **Rule:** a pixel's period with no usable input (0 < solz < 72 in that band's own inputs; a missing file is unusable) takes the
  same pixel's nearest valid period; mixed periods are unchanged.
- **NDVI:** the zenith-only NDVI rule is replaced by the fill on the fill tiles.
- **Variants:**
  - **G:** the 6 Greenland tiles;
  - **GA:** G plus h12v01, h13v01 and h21v01;
  - **CTRL:** the same models without fill.
- **Models:** 60 trees, MEDIAN for both layers, seed 0; same base splits for all three.

**Pass rule (project lead):** bare helped **and** tree cover unchanged or better.

## Checks

- **Gate:** with no fill, the recomputed predictors equal the stored split values at every training and held-out row of the 4
  labelled scope tiles (0 cells differ, tree and bare).
- **Runs:** 6/6 fits and 27/27 maps.
- **One failure, fixed:** `fitAltModel.py` was not copied into the run; about 2 min lost.

## Results

**Held-out labels (variant − CTRL, MAE; 95% tile-bootstrap interval).** The seed-to-seed yardstick is about 0.002-0.005.

| Layer | Variant | All tiles | Scope tiles | Other tiles |
|---|---|---|---|---|
| Tree | G | −0.006 [−0.010, −0.001], 37 better / 21 worse | +0.003 (0 better, 2 worse) | −0.006 |
| Tree | GA | +0.002 [−0.003, +0.006], 27 / 34 | +0.007 (4 of 4 worse) | +0.002 |
| Bare | G | −0.001 [−0.005, +0.004] | −0.005 [−0.039, +0.009], 3 of 4 better | −0.001 |
| Bare | GA | +0.007 | **+0.058** [+0.036, +0.324], 4 of 4 worse | −0.002 |

All differences except GA bare on the scope tiles are at the size of seed noise.

**Maps, Greenland tiles (CTRL → G):**
- **Tree cover rises on the ice:** the share with tree = 0 drops from 99.7 to 94.1% on h16v01, 99.4 to 96.7% on h16v02, 99.8 to 96.6%
  on h17v01, and 93.9 to 89.9% on h15v02.
- **Bare against C6 is mixed:**
  - better: h15v02 14.4 → 11.6, h17v01 5.1 → 4.6;
  - worse: h16v01 4.8 → 5.7, h16v02 1.5 → 2.9.
- **The h16v02 edge:** the share of pixels ≥ 5 below C6 is about 0 in CTRL and in the step-1 candidate, both with MEDIAN. It is 0.7%
  above the edge and 1.1% below with G. **The MEDIAN models alone already remove the edge seen in the MEAN maps** (3% → 11%), and the
  fill brings back a little of it.
- **v00 tiles:** unchanged (v00 rule).
- **Islands (h12v01, h13v01, h21v01):** changes of about 0.1-1 point.

## Verdict

- **G fails the rule:**
  - bare is not clearly helped: a tie on labels, mixed against C6;
  - tree cover gets worse where it matters, with more non-zero tree on the ice sheet and the scope tiles' tree held-out slightly
    worse.
- **GA fails on both layers.**
- **Recommendation:** do not adopt the fill. Keep the zenith-only NDVI rule on the 6 Greenland tiles as it is.

## What we learned

1. **Of the polar-night / low-sun symptoms seen in the MEAN products, MEDIAN removes most:**
   - the h16v02 edge;
   - ice bare below 100 (exactly 100 on the C6 = 100 pixels: 38% → 79% overall).
2. **Filling unusable periods with the nearest valid period shifts the ice pixels towards "summer-like" inputs** that the models read
   as slightly vegetated. With labels only on h16v01, the models cannot learn that it is ice.
3. **Side finding:**
   - On Greenland, the step-1 candidate (models trained on the original split, h16v01 with unfixed NDVI) is closer to C6 for bare
     than CTRL (models trained on the MC split, h16v01 with the zenith-fixed NDVI): h16v01 MAE 2.7 against 4.8, and exactly 100 85%
     against 72%.
   - So training on the zenith-fixed h16v01 rows lowers Greenland bare a little.
   - Worth noting when the delivered models are next retrained. Not investigated further here.

## Files

`FullRuns/fill72_20261004/out/`:
- `compare/` (`compare.csv`, per-tile CSVs);
- `heldout/`;
- `map_check.csv`;
- `maps/{CTRL,G,GA}`.
