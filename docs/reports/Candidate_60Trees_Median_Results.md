# Candidate product: 60 trees and MEDIAN for both layers (results; step 1 after `Bare_Mechanisms_Results.md`)

**Status 2026-10-04: done.** Run `FullRuns/cand60med_20261004`, 11:33-13:48, on ngaproc202, forest201-205 and, from 12:56,
ngaproc201.

**Candidate:** the v1.0.0 recipe with:
- tree cover from T60 (60 trees, MEDIAN);
- non-vegetated from B60S0 (60 trees, MEDIAN);
- the sum cap and v00 rule as delivered.

69 tiles. No version name is given; that is the project lead's decision.

## Checks

- **Gate (h12v09):**
  - the tree layer equals the stored T60 map (0 pixels differ);
  - the bare layer equals min(round(median of 60 trees), 100 − tree) at 10,000 sampled pixels (0 differ).
- **69/69 maps, no failures.**
- **Labels read from the maps at the held-out keys:** the v1.0.0 map reproduces its stored held-out predictions on 99.5% of valid
  pixels.

## Results

**Held-out labels, read from the maps (sum cap and v00 rule included):**

| Layer | Label bin | v1.0.0 | Candidate |
|---|---|---|---|
| Bare | all | 4.18 | **3.76** |
| | 0-20 | 3.26 | **2.51** |
| | 21-40 | **6.64** | 6.87 |
| | 41-60 | 5.89 | 5.85 |
| | 61-99 | 5.71 | **5.32** |
| | 100 | 0.53 | **0.29** |
| Tree | all | 6.70 | **6.58** |
| | every bin | — | 0.08-0.67 better |

- **Bare** is better in every region except Greenland, a tie (+0.03), for example North America 50-80N −0.76 and Asia 50-80N −0.62.
- **Tree** is better in every region (−0.01 to −0.22).

**Maps against the C-lineage references (69 tiles, pixel-weighted MAE):**

| Layer | Reference | v1.0.0 | Candidate |
|---|---|---|---|
| Tree | C6 / C6.1 / C code | 4.75 / 5.75 / 5.36 | **4.52 / 5.54 / 5.15** |
| Bare | C6 / C6.1 / C code | **8.78 / 15.10 / 12.40** | 10.48 / 17.02 / 14.07 |

- **Tree** is better against all three, consistent with the labels.
- **Bare** is worse against all three by about 1.7. Its bias against C6 goes from −3.8 to −6.8, and only 12 of 69 tiles move closer to
  C6.

**Where C6 = 100 (107.6 M pixels):** exactly 100 rises from 38% to **79%**; the mean is 95.4 → 97.2; below 90 is 11.7% → 9.3%.

## Where bare moves away from C6, and what the labels can say there

- **Where:** the largest increases in MAE against C6 are in vegetated temperate and boreal tiles. C6's bare there is a few points
  above 0, and the candidate is lower.

  | Tile | MAE against C6 (v1.0.0 → candidate) |
  |---|---|
  | h18v03 | 10.2 → 16.0 |
  | h20v03 | 9.7 → 14.8 |
  | h11v03 | 11.9 → 16.6 |
  | h11v04 | 13.7 → 18.0 |
  | h27v04 | 14.4 → 18.2 |

  The same pattern holds on h18v04, h23v03, h12v03, h22v03, h24v03, h10v06 and h12v05. A median of trees that mostly predict about 0
  comes out at 0, where the mean kept a few points.
- **Labels in these tiles:** bare labels exist on only 20 tiles. Of these 12 tiles, only h27v04 has them, and there (Asia 30-50N) the
  labels favour the candidate: 0.99 → 0.51. On every labelled region, lower bare in vegetated land agrees with the labels.
- **Labels missing:** Europe and the Canadian/European v03-v04 tiles have **no** bare labels. The only evidence there is the three
  references, which share the static-model lineage and agree with each other.
- **Effect on the sum cap:** the share of land where non-tree vegetation is 0 (sum cap or v00 rule binding) rises from 5.7% to 10.7%.
  That follows from more exact 100s in bare.

## Reading

- **Tree cover:** the candidate is better on every measure (labels and all three references).
- **Bare:** labels against references disagree.
  - The labels (independent, 20 tiles) say the candidate is better everywhere they exist, including at 100 and in 0-20.
  - The three C-lineage references say it is worse, mostly through lower values in vegetated boreal and temperate land that has no
    bare labels.
  - By the project's rule (act on differences consistent across references), the references are consistent with one another but
    not with the labels. This is for the project lead.
  - **Possible ways forward:**
    - accept MEDIAN (labels);
    - check the boreal/temperate bare level against an external reference on ADAPT (ESA WorldCover bare/sparse fraction) before
      deciding;
    - or use MEDIAN only where it matters most (a rule; not proposed).

## Files

- `FullRuns/cand60med_20261004/out/`: `eval/` (`labels_from_maps.csv`, `refs_per_tile.csv`, `refs_pooled.csv`, `checks.json`),
  `c6_100_screen.csv`, `maps/CAND`.
- Figures:
  - `Reports/figures_nonveg_v100_cand60med_20261004/` (two rows: v1.0.0 and the candidate against C6);
  - `Reports/figures_tree_cand60med_20261004/`.

## Quick look against ESA WorldCover 2020 (added 2026-10-04; comparison only, not truth)

**Requested by the project lead** for the unlabelled boreal and temperate tiles. Run `FullRuns/worldcover_bare_20261004` (read-only,
ngaproc201, about 1 min).

**Method:**
- Per tile, 9 windows of 400 × 400 MODIS pixels, resampled from WorldCover 10 m to 64 samples per 250 m pixel.
- "WorldCover non-vegetated" is the share of classes 50 (built-up), 60 (bare/sparse vegetation) and 70 (snow/ice), out of non-water
  samples.
- **Caveat:** a class share is not percent cover. WorldCover is another satellite product, and class 60 allows up to about 10%
  vegetation, while shrub and grass classes can contain bare ground.

**Results** (mean over the sampled land pixels; MAE and bias against the WorldCover share × 100):

| Tile | WorldCover | v1.0.0 mean / MAE / bias | Candidate mean / MAE / bias | C6 mean / MAE / bias |
|---|---|---|---|---|
| h18v03 | 4.7 | 9.8 / 10.3 / +5.1 | **3.1 / 5.3 / −1.6** | 20.1 / 17.8 / +15.3 |
| h20v03 | 1.3 | 10.7 / 10.6 / +9.4 | **4.7 / 5.2 / +3.4** | 19.5 / 18.8 / +18.2 |
| h11v03 | 4.6 | 14.8 / 11.6 / +10.2 | **9.4 / 6.5 / +4.9** | 23.8 / 21.2 / +19.2 |
| h11v04 | 2.8 | 9.8 / 9.8 / +7.0 | **4.7 / 5.9 / +1.9** | 22.4 / 20.7 / +19.6 |
| h27v04 | 2.8 | 6.4 / 6.0 / +3.6 | **3.1 / 3.3 / +0.3** | 21.1 / 19.1 / +18.3 |
| h18v04 | 8.5 | 11.3 / 10.9 / +2.8 | **7.5 / 8.3 / −1.0** | 20.5 / 17.0 / +12.0 |
| h23v03 | 0.6 | 10.9 / 10.6 / +10.4 | **6.2 / 6.1 / +5.6** | 16.5 / 16.1 / +15.9 |
| h12v03 | 0.3 | 8.0 / 7.8 / +7.7 | **3.4 / 3.3 / +3.1** | 14.7 / 14.5 / +14.4 |
| h22v03 | 1.2 | 16.9 / 16.5 / +15.7 | **12.7 / 12.6 / +11.5** | 21.8 / 21.2 / +20.5 |
| h24v03 | 0.4 | 7.9 / 7.8 / +7.6 | **3.8 / 3.7 / +3.4** | 12.6 / 12.4 / +12.3 |
| h10v06 | 7.2 | 10.0 / 11.0 / +2.8 | **3.6 / 7.5 / −3.6** | 15.9 / 14.4 / +8.7 |
| h12v05 | 4.0 | 7.6 / 7.8 / +3.6 | **4.5 / 5.6 / +0.5** | 17.2 / 15.1 / +13.2 |
| *controls* | | | | |
| h12v09 (forest) | 0.0 | 2.6 / 2.6 / +2.6 | **0.2 / 0.2 / +0.1** | 13.3 / 13.3 / +13.3 |
| h20v06 (desert) | 99.9 | 99.7 / 0.3 / −0.3 | **99.9 / 0.1 / 0.0** | 99.9 / 0.0 / 0.0 |
| h08v05 (SW US, labelled) | 33.0 | 48.5 / 22.0 / +15.6 | 48.8 / 21.6 / +15.8 | 49.1 / 24.0 / +16.2 |
| h09v05 (SW US, labelled) | 11.7 | 35.6 / 27.0 / +23.9 | 35.1 / 26.4 / +23.4 | 37.3 / 29.0 / +25.6 |

**Reading (a quick look, not a conclusion):**
- **On all 12 tiles that moved away from C6, the candidate is closer to WorldCover than v1.0.0.** MAE is lower on every tile, by 2.3 to
  5.4.
- **C6 is the furthest from WorldCover on every one of them:** bias +9 to +20 where WorldCover shows 0.3-8.5%.
- **So WorldCover sides with the held-out labels, not with the C-lineage references,** on where the median moves bare.
- **Forest:** in the Amazon control, C6 shows 13% bare where WorldCover shows almost none.
- **Drylands (SW US):** all three products are far above WorldCover's class share, and the candidate equals v1.0.0. This is most
  likely the class-share caveat (sparse shrubland), not a product difference.

## Ice sheet: are the remaining pixels below 100 errors? (h16v01 held-out labels, added 2026-10-04)

The question: are the pixels where C6 = 100 but the candidate is below 100 errors, or real ice-free margin? Only h16v01 has labels.

**Check:** held-out bare labels where C6 = 100 (93,546 pixels), grouped by the candidate's value.

| Candidate | n | Label mean | Labels = 100 | MCD12Q1 | Candidate MAE | C6 MAE |
|---|---|---|---|---|---|---|
| = 100 | 90,317 | 100.0 | 100% | 100% snow/ice | 0.00 | 0.00 |
| 90-99 | 5 | 78.8 | 0% | barren | 16.4 | 21.2 |
| < 90 | 3,224 | 60.0 | 0% (26% below 50) | 81% grassland, 19% barren | 12.7 | 40.0 |

**Result:**
- **The ice sheet itself is solved on h16v01:** every labelled ice pixel is 100, as the label is.
- **The pixels still below 90 are ice-free margin,** where the labels are 41-78 (10th-90th percentile). The candidate is right not to
  put them at 100, and C6 is 40 points off there.
- **The project lead's note fits:** saturated soil at the margins is spectrally dark.
- **The candidate is slightly high on the margin:** its values are 62-79 against labels of 41-78. Over all C6 = 100 labelled pixels
  the candidate's MAE is 0.44, against 0.39 for v1.0.0, a small difference that comes from these margin pixels.

**Still open:** the 90-99 band on h16v02 (25%) and h15v02 (18%) cannot be checked, since those tiles have no labels. Mixed low-sun
inputs remain the likely cause.
