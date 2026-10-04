# In-file fill values (−28672) in the composites: first screen (2026-10-04)

Decision 7 (README item 19) puts this first in the work order. This is a read-only count. No code was changed and no run was launched.

## What the code does

`CompositeDayFile._computeComposite` builds each period's composite from its inputs in two cases:
- **Case 1 (QA pass):** the mean of the inputs that pass `createQaMask` (cloud, shadow and adjacency clear; bit 15 good; 0 < solz < 65).
- **Case 2 (fallback):** if no input passes, the mean of **all** inputs, whatever their QA.

A missing file is replaced with 10000 in every band. A **value of −28672 inside a file is never tested for**. Wherever it passes
QA, or wherever the pixel falls back, it enters the mean as a number. One fill value among three inputs pulls the mean to about −9500.
The C code drops a pixel from prediction when band 1 is −28672 in any input (its −3 rule; `C_Code_Bare_Rules_Inventory.md`, row 0).

## Screen

Script: `Reports/fill_count.py`. For each of the 23 MOD44CQ/CH inputs of 2019 (DOY 065 2019 to 049 2020),
it reads bands 1-7 and 31, then counts the pixels at −28672 and how many of those pass the pipeline's QA mask (CQ `state` and
`solar_zenith`, as in `createQaMask`). The counts are per input pixel, not per final composite, so they are an upper bound on what
reaches the metrics. Run on adaptlogin201, about 5 min per tile.

| Tile | Region | Tile pixels with fill in any band or period | Band 5 fill (pixel-periods / of which pass QA) | Other bands with fill (all fail QA) |
|---|---|---|---|---|
| h16v02 | Greenland edge | 26.4% | 6,555,368 / 5,627,219 | B1 21k, B2 1.7k, B3 0.8k, B4 46k, B6 192k, B7 766k |
| h12v02 | Canada, high latitude | 5.8% | 902,157 / 210,180 | B1 12k, B4 36k, B6 70k, B7 465k |
| h11v04 | USA, temperate | 4.1% | 1,510,342 / 1,250,989 | none |
| h20v06 | Sahara / Sahel | 0.5% | 249,085 / 194,905 | B6 15, B7 38 |
| h12v09 | Amazon, cloudy | 0.01% | 1,870 / 213 | none |

Band 31 never has fill: it is clamped to 16000 above solz 70 before the mean.

## Findings

1. **Band 5 fill passes QA almost everywhere it occurs** (83% of band 5 fill on h11v04 and h16v02). The QA mask cannot catch it,
   because band 5 fill does not set any `state` bit. This fits Terra band 5's known bad detectors. **The v1.1.0 product does not use
   band 5:** neither the 64 tree predictors (`predictors_pcttree.txt`) nor the 21 non-vegetated predictors use it, and neither does
   the MC64 set. So it affects only the full metric pool (the Monte Carlo search's 324 metrics) and any future predictor set.
2. **Fill in the other bands (1, 2, 3, 4, 6, 7) never passes QA**, so it can enter only through the case 2 fallback mean. It occurs
   only on the high-latitude tiles (h16v02, h12v02), mostly in the Nov-Jan periods (DOY 337, 353, 001) and some summer periods on
   h16v02 (ice). These periods and tiles are the same ones as the polar-night and low-sun fallback work (`Fallback_Solz_LE0_Plan.md`,
   `Fallback_Fill72_Results.md`). Bands 1, 2, 6 and 7 are in the tree predictors; 2, 3, 4, 6 and 7 are in the non-vegetated predictors.
3. **Temperate, desert and tropical tiles have essentially no non-band-5 fill**, so the fix would not touch most of the product.

## Not yet known

- How many **final composite values** in the predictor bands actually include a fill value. That is the case 2 periods only, and
  needs the composite-level count on the high-latitude tiles.
- Whether those values change the v1.1.0 maps. Most of these pixels are ice or polar-night land, where the v00 rule, MEDIAN and the
  labels already decide the outcome.

## Suggested next step (not run; needs the project lead's OK)

**Superseded 2026-10-04 evening by `Composite_Unusable_Inputs_Plan.md`**, which covers fill values, polar-night zeros and missing
files together. The text below is kept as written.

1. **Composite-level count** on the 15 tiles at v ≤ 02 (and h11v04 as a control): for each predictor band and period, the pixels
   whose case 2 mean includes −28672, and the metric values that result. Read-only, about 1 h on one node.
2. If any meaningful number of predictor values is affected: **treat −28672 like a missing input** (left out of both means; if every
   input is fill, use today's 10000 substitute, consistent with the solz ≤ 0 recommendation) and do it for **every band**, band 5
   included, so the metric pool is clean. Then a one-tile exactness check (no change outside fill pixels) and the 69-tile maps
   against v1.1.0.

Other years (to-do 12) should wait for this decision: rerunning them after a composite fix would double the cost.
