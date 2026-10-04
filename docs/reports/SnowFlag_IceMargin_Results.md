# Does the MOD44C snow flag identify the ice-sheet margin? (2026-10-04)

**Read-only quick analysis.** Requested by the project lead. Run `FullRuns/snowflag_margin_20261004` (ngaproc202, about 3 min).
Product: v1.1.0 (the `cand60med_20261004` CAND maps).

**Flag:** MOD44C `state` bit 5 (MOD09 bit 12, snow/ice; see the `mod44c-state-bits` skill), read from MOD44CQ for the 23 inputs of the
2019 window. Inputs with `state` = 0 (no observation) are not counted.

**Per pixel:** the share of present inputs with the flag set, for the whole year and for **summer** (inputs dated 2019161-2019241,
June-August).

## h16v01, against the held-out bare labels (the only labelled Greenland tile)

| Labels | n | Snow share, year | Snow share, summer | Summer share ≥ 50% | Summer share < 10% | v1.1.0 mean |
|---|---|---|---|---|---|---|
| = 100 | 90,323 | 77% | 73% | 97.0% | 1.4% | 100.0 |
| 90-99 | 18 | 31% | 1% | 0% | 94% | 77.4 |
| 50-89 | 6,654 | 35% | 0.2% | 0% | 99.2% | 70.6 |
| < 50 | 3,727 | 31% | 0.0% | 0% | 99.9% | 46.0 |

**Using "summer snow share ≥ threshold" as an ice test:**

| Threshold | Label-100 pixels caught | Label < 90 pixels wrongly called ice |
|---|---|---|
| 0.10 | 98.6% | 0.6% |
| 0.25 | 98.0% | 0.1% |
| 0.50 | 97.0% | 0.0% |

**Yes: the summer snow flag separates the ice sheet from its margin almost perfectly on h16v01.** The margin pixels (labels 41-89,
mostly MCD12Q1 grassland or barren: the dark, saturated soils) are essentially never flagged in summer. The whole-year share separates
less well, since winter snow covers the margin too.

## The other Greenland tiles (no labels), where C6 = 100

| Tile | v1.1.0 group | Pixels | Summer share ≥ 50% | Summer share < 10% |
|---|---|---|---|---|
| h16v02 | = 100 | 6.07 M | 92% | 2% |
| h16v02 | **90-99** | **1.99 M** | **93%** | 3% |
| h16v02 | < 90 | 0.02 M | 31% | 38% |
| h17v01 | = 100 | 8.39 M | 79% | 1% |
| h17v01 | 90-99 | 0.20 M | 12% | 27% |
| h17v01 | < 90 | 0.80 M | 0.6% | 64% |
| h15v02 | = 100 | 2.53 M | 67% | 14% |
| h15v02 | 90-99 | 0.61 M | 18% | 59% |
| h15v02 | < 90 | 0.33 M | 0.8% | 93% |

**Reading:**
- **h17v01 and h15v02:** the pixels below 90 have almost no summer snow. By the flag they are margin, as on h16v01, so v1.1.0 is
  right not to put them at 100.
- **h16v02:** the 2 M pixels at 90-99 have summer snow on 93% of them. **By the flag they are ice, and v1.1.0 is a few points
  short.** This is the remaining ice-sheet shortfall, consistent with the dark low-sun inputs in mixed periods found in the traces.
- **h15v02:** its 90-99 band is mostly not flagged (59% under 10%). It is more likely margin or mixed ground than ice.

## What this could be used for (nothing adopted)

**A physically based ice rule** would set bare = 100 and tree = 0 where the summer snow share is at least about 0.25.
- **Precedent:** the C code's high-latitude rule also used the snow flag, at slot 11 (DOY 241), but as a blanket box rule.
- **Gain:** a summer-share test would follow the actual ice edge.
- **Cost:** on h16v01 it would change almost nothing (the labelled ice is already 100) and would mislabel only 0.1% of margin pixels.
  It would fix the h16v02 90-99 band.

**Before considering it:**
1. **Screen where the test triggers outside Greenland:** mountain glaciers, summer snow patches, Arctic islands.
2. **Check the tree-cover side:** it sets tree to 0. Your rule is that bare must be helped and tree not made worse.
3. **Check it against the h16v01 labels:** nearly neutral, by construction.

It would be a post-model rule. The project's evidence so far has argued against rules in general, but here labels and a physical flag
agree.
