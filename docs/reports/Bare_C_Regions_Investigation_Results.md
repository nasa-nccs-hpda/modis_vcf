# What the new RF bare layer does in the C code's special regions (results, to-do item 5)

**Run 2026-10-01: phase 1 finished 20:46, phase 2 (new Greenland tiles) finished 21:53. forest204, read-only, fully detached. Plan: `Bare_C_Regions_Investigation_Plan.md` (approved).** Folder: `FullRuns/bare_c_regions_20261001/` (`out/regionA.csv`, `regionA_newtiles.csv`, `regionB.csv`, `regionB_by_C6bin.csv`; masks in `out/masks`). Product: the delivered non-vegetated layer (RF, MEAN, sum cap) and its SD.

This report describes what the layer does. Nothing in the product is changed. The options at the end are for the project lead.

## Summary

- **Both C code regions were rebuilt exactly.**
  - The rebuilt v7 high-latitude mask gives C code bare = 100 on 100% of masked land on all 5 v7 tiles.
  - On the 6 new Greenland tiles, the MOD44CQ-derived flags match the C intermediates 100% at the 65° cutoff.
- **Region A (high latitude): RF bare is well below 100 inside the C code's mask, by different amounts on different tiles.**
  - **Ice sheet (h16v01, h16v02, h17v01, h15v02):** RF bare is 88-94, about 6-10 below C6. The held-out labels on h16v01 (mean 98.6) are matched closely (RF MAE 0.16), so RF is right on the labelled ice pixels and low elsewhere.
  - **Polar-night tiles (h16v00, h17v00):** RF bare is 73-79, about 21-27 below C6. These tiles are 100% inside the mask.
  - **Small high-Arctic islands (h12v01, h13v01):** RF bare is ~24 inside the mask, where C6 and the C code are ~98-100. The few labels there are 23-52, not 100. RF and the labels both say these pixels are not fully bare.
  - **h11v02 and h12v02:** inside the mask (2-7% of land), RF is 74-83 against C6's 97-99. The labels are 61-81, with only 41-59% of them at 90 or more. On these tiles, forcing bare to 100 would overwrite real vegetation for part of the mask.
  - **Land cover inside the v7 mask:** 98-100% of these pixels have C6 tree cover 0. They are ice, rock or barren tundra, not forest.
- **Region B (Sahara stretch): the stretch matches C6, but not the labels.**
  - Against C6, applying the stretch to RF lowers MAE on every desert tile: h20v06 1.03 → 0.24, h21v06 2.17 → 0.36, h18v07 13.0 → 8.0, h17v05 10.6 → 8.0, h21v05 8.4 → 7.3. It does nothing on h19v08 and h20v08, where almost no pixels have bare ≥ 55.
  - **Against held-out labels it hurts h18v07:** MAE 7.3 → 11.1, bias −0.6 → +8.5. On h20v06 and h21v06 the labels are already ~100 and RF is within 0.1 of them either way.
  - So **RF's low desert bias against C6 is largely the stretch built into C6**, not an RF error. The labels do not support stretching.
- **A conflict to reconcile:** the rebuilt v6.0.3 mask (all v < 3 tiles) does **not** match C6. On h20v02, h21v01, h21v02 and h23v02, which lie inside it but outside v7, C6 bare is 10-17, not 100. On h11v02, h12v01 and h13v01, C6 is 18-26 inside the v6.0.3 mask, but 97-99 inside the v7 mask. The C6 maps look like the v7-type rule, or something close to it, and not the v6.0.3 rule as rebuilt here. Last night's side finding ("C6 includes the all-v<3 rule") should not be quoted until this is understood.

## Region A: high latitude

**Rule (C code, `MOD44B3.c`):** bare = 100 where the slot-11 snow flag is 0, or where slots 11-13 (DOY 241-273) have no clear observation. In v7 this applies to h < 18, v < 3. The v6.0.3 version as rebuilt here applies to every v < 3 tile and uses the snow flag in any of slots 11-13.

**Inside the v7 mask** (the pixels the C code forces to 100):

| Tile | Land in mask | RF bare mean | RF bare median | RF ≥ 90 | RF tree | RF bare SD | C6 bare | C6.1 bare | C code bare | RF − C6 | Held-out labels (n, mean, RF MAE) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| h11v02 | 7.0 % | 74.0 | 94 | 57.8 % | 0.15 | 10.9 | 99.1 | 100 | 100 | −25.0 | 6,881, 60.7, 7.8 |
| h12v01 | 16.0 % | 23.7 | 18 | 0.2 % | 0.03 | 14.8 | 98.9 | 100 | 100 | −75.9 | 7, 52.3, 9.0 |
| h12v02 | 1.6 % | 82.7 | 92 | 56.1 % | 0.09 | 11.0 | 96.9 | 100 | 100 | −14.2 | 10,671, 81.4, 6.7 |
| h13v01 | 1.5 % | 24.4 | 19 | 0.0 % | 0.0 | 11.4 | 97.4 | 100 | 100 | −74.5 | 166, 23.1, 9.0 |
| h16v01 | 97.4 % | 89.9 | 94 | 59.2 % | 0.40 | 9.0 | 100.0 | 100 | 100 | −10.1 | 93,560, 98.6, 0.16 |
| h15v02 | 42.6 % | 92.3 | | 81.1 % | 0.00 | | 99.6 | 100 | | −7.2 | |
| h16v00 | 100 % | 73.1 | | 8.0 % | 6.9 | | 100.0 | 100 | | −26.9 | |
| h16v02 | 92.9 % | 94.1 | | 93.6 % | 0.00 | | 100.0 | 100 | | −5.9 | |
| h17v00 | 100 % | 78.9 | | 18.5 % | 5.8 | | 100.0 | 100 | | −21.2 | |
| h17v01 | 96.3 % | 88.6 | | 52.0 % | 0.32 | | 100.0 | 100 | | −11.4 | |
| h17v02 (Iceland) | 31.3 % | 88.7 | | 72.3 % | 0.32 | | 99.3 | 100 | | −10.5 | |

The last 6 rows are the new tiles from the Greenland run, using their real (unfixed) delivered maps. They have no C code run and no labels.

Outside the v7 mask on the same tiles, RF and C6 agree closely: RF − C6 is between −4.5 and +8.7 on the original 5 tiles, and h12v02 is at +0.06.

**Inside the v6.0.3-only area** (v < 3 tiles outside the v7 box, where the rebuilt v6.0.3 mask covers 84-100% of land):

| Tile | Land in mask | RF bare | C6 bare | C6.1 bare | C code bare | C6 tree = 0 |
|---|---|---|---|---|---|---|
| h20v02 | 87.5 % | 14.4 | 10.4 | 100 | 39.2 | 20 % |
| h21v01 | 100 % | 14.6 | 14.1 | 100 | 61.3 | 66 % |
| h21v02 | 97.2 % | 17.3 | 13.6 | 100 | 43.4 | 16 % |
| h23v02 | 84.2 % | 23.5 | 17.2 | 100 | 52.0 | 18 % |

Here RF agrees with C6 within 0-6 points. Only C6.1 is at 100, because of its own high-latitude rule.

## Region B: Sahara stretch

**Rule (C code):** on h16-h23, v05-v08, bare ≥ 55 becomes 2 × bare − 55, capped at 100, with hand-drawn exemptions on h22/h23 v05/v06 (none among these tiles). The what-if below applies it to RF in memory.

| Tile | RF bare mean | RF share ≥ 55 | C6 MAE: RF → RF + stretch | C6 bias: RF → RF + stretch | C code MAE: RF → RF + stretch | Held-out MAE: RF → RF + stretch (n) | Stretched pixels over the sum cap |
|---|---|---|---|---|---|---|---|
| h17v05 | 52.8 | 45.6 % | 10.6 → **8.0** | −6.9 → −3.6 | 14.7 → 17.3 | | 4,185 |
| h18v07 | 58.6 | 57.2 % | 13.0 → **8.0** | −7.1 → −1.6 | 13.4 → 8.8 | **7.3 → 11.1** (121,087) | 93,870 |
| h19v08 | 7.3 | 0.1 % | 5.5 → 5.5 | −0.1 → −0.1 | 9.6 → 9.6 | | 637 |
| h20v06 | 98.2 | 99.0 % | 1.03 → **0.24** | −0.8 → +0.1 | 2.7 → 2.2 | 0.04 → 0.0 (34,022) | 6,155 |
| h20v08 | 6.1 | 0.0 % | 4.8 → 4.8 | +0.1 → +0.1 | 10.0 → 10.0 | | 116 |
| h21v05 | 62.8 | 57.7 % | 8.4 → **7.3** | −1.5 → +3.1 | 20.2 → 24.0 | | 9,428 |
| h21v06 | 97.5 | 99.6 % | 2.17 → **0.36** | −1.8 → +0.3 | 4.5 → 3.9 | 0.08 → 0.0 (68,041) | 31,625 |

C6.1 gives the same picture as C6, within 0.5.

**By C6 bare bin** (`regionB_by_C6bin.csv`): the stretch helps only in the 61+ bin, where RF is low against C6:
- h18v07: bias −15.6 → −6.2;
- h17v05: −12.8 → −6.3;
- h21v05: −7.0 → +0.9.

In the 41-60 bin it makes things worse everywhere: h20v06 MAE 15.9 → 26.0, h21v06 14.2 → 23.4, h21v05 8.9 → 12.2.

**Held-out labels on h18v07** are the only labelled test where the stretch changes anything. They say the stretch is wrong there: unstretched RF is unbiased (−0.6), and the stretch adds +8.5.

## Options for the project lead (no recommendation is acted on)

**Region A, high latitude**
1. **Do nothing.** RF stays 6-27 below C6 inside the mask on Greenland. The zenith-only fix (`Greenland_Iceland_Scoped_Fix_v2_Results.md`) closes about half of that gap.
2. **Port the v7 rule (bare = 100 inside the mask).** This matches C6/C6.1 and the C code on the ice sheet. But it would also force 100 on the small high-Arctic islands (h12v01, h13v01) and part of h11v02/h12v02, where the labels say 23-81.
3. **Something narrower,** for example the rule only where RF bare is already high, or only on the Greenland tiles, after the zenith fix. This needs a plan.

**Region B, Sahara**
1. **Do nothing** (suggested by the labels). RF's low bias against C6 in the 61+ bin is mostly the stretch that C6 contains.
2. **Port the stretch** to match the historical product's look. This brings RF closer to C6 by up to 5 MAE on h18v07, but makes it worse against the labels and in the 41-60 bin.

**Before quoting the v6.0.3 side finding:** find out why C6 does not show bare = 100 on the v6.0.3-only tiles. The C6 production may have used a code version other than the v6.0.3 found, or the rebuilt v6.0.3 condition may not be the one that ran.

## Decision (2026-10-02)

**Project lead: adopt the high-latitude rule for v00 tiles only** (tree cover 0, non-vegetated 100, non-tree vegetation 0 on land; the SD layers keep the members' spread, as in the C code). Reason: there are no trees and very likely no vegetation there. h16v00 and h17v00 are 100% inside the C code's mask anyway. No other tile, and no other part of the C code's v < 3 mask, is affected. The Sahara stretch has not been decided; nothing changes there. A further look at the rule is on the to-do list under "Later, if time permits".
