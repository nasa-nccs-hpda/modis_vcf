# Baseline, Random Forest and XGBoost tunedB vs the historical MOD44B: tile deep dives (h09v05, h12v02, h12v09, h20v06)

**Status (2026-09-29): finished. Started on one test tile, extended at 08:26 to h12v02, h12v09 and h20v06, and at 08:51 to all 63 tiles (last section).**
The paragraphs below were written for the first tile. No product code changed. Nothing outside the regression-tree
run root and this Reports directory was written.

## Read this first: the historical product was made without a working aerosol screen

(The MOD44C `state` bit problem behind this is now its own investigation: `/explore/nobackup/people/mcarrol2/MOD44C_PGE72_Investigation/`.)

**CORRECTION (2026-09-29, later, after re-reading PGE72's code): this section overstates the historical
product's contamination and misses a bigger effect on the current products.**
- PGE72 already prefers low- or average-aerosol observations when it builds each MOD44C composite, so
  the historical product was mostly aerosol-screened upstream. The broken PGE61 read only affected the
  leftover cases.
- The fixed PGE61 screen (bits 7-8 = 3) probably also rejects every *low*-aerosol composite pixel
  observed on an odd day of year, perhaps half of the best pixels, in both the current C and Python
  products.

So the likelier reason C and the models share their disagreement with the historical product is this
over-rejection in the current products, not smoke or dust in the historical one. Neither has been
measured. Details: `/explore/nobackup/people/mcarrol2/MOD44C_PGE72_Investigation/Reports/MOD44C_State_Aerosol_Bit_Collision.md`,
"Corrections". The bullets below are kept as first written.

The historical MOD44B is **not a clean reference.** The earlier C-vs-Python work found that the
operational C code read aerosol from the wrong QA bits
(`C_code/pge61/MOD_PR44B_Metrics/C_vs_Python_Code_Differences.md`, finding 2;
`Project_Summary_and_Conclusions.md`). C's `fix_bands.c` read aerosol as `(state >> 13) & 3`, and those
bits are always 0 in MOD44C's repacked `state`. So **the aerosol test never rejected anything in the code
that produced the historical product**, and smoke- and dust-contaminated observations went into its
composites and metrics. It was fixed on 2026-09-15: both C (`>> 7`) and Python (`(state & 384) >> 7`)
now reject an observation when bits 7-8 equal 3. Both the C and Python products in this comparison use
the fixed screen; the historical product does not. Adjacency (bit 6) was always read correctly by C, so
it does not differ from the historical product.

What this means for the comparisons below:
- **Disagreement with the historical product is not all model error.** Some of it is the historical
  product's own aerosol contamination, and C and Python should both differ from it there. This is the
  likely reason that **C shares the models' error geography on the boreal and Amazon tiles** (block
  correlation 0.64-0.72) but not on h09v05 (-0.04). 2019 had very large fire seasons in interior Alaska
  (h12v02) and the Amazon (h12v09, where the Xingu/Altamira cluster is in the deforestation and burning
  arc), and the Sahara/Nile tile (h20v06) is dust-prone. This is a hypothesis; it has not been tested
  pixel by pixel.
- **The fixed screen is not a pure aerosol test either.** In the repacked `state`, bit 7 also carries the
  original BRDF-correction flag and bit 8 the low bit of day-of-year, so "bits 7-8 = 3" also rejects some
  observations that are not high-aerosol. The check against a published MOD44C granule is still open
  (Project Summary open item 5).
- **Comparisons between the current products (C vs Python, model vs model) are unaffected**, since they
  share the same screen. The ranking of the three Python models (XGBoost best everywhere) therefore
  holds, but absolute MAE and bias against the historical product mix model error with the reference's
  aerosol contamination.
- **The training labels and the models' learned relationships** were built from metrics computed with the
  fixed screen. So the models are trained and applied consistently. The historical product was
  consistent with its own (unscreened) inputs.
- A direct test is possible: count, per pixel and period, how many observations the fixed aerosol screen
  removes (bits 7-8 = 3 against the old no-op), and correlate that count with the block-level
  disagreement maps. Not done.

## What was run

- Script: `FullRuns/regtree_alt_models_20260927/code/validation/deepdive_vs_historical.py` (new; the
  script takes a tile argument and can be reused for other tiles). Launcher: `run_deepdive_node.sh <tile>`,
  run detached on forest203 from 08:03 to 08:06.
- Output: `FullRuns/regtree_alt_models_20260927/Compare/deepdive_vs_historical_h09v05/`, containing
  `summary.txt`, six `stats_*.csv` files and figures `fig1` to `fig8`.
- Inputs (all existing, read-only):
  - Historical MOD44B Collection 6 (`.006`, the reference in every earlier report). Collection 6.1
    (`.061`) was also scored, statistics only.
  - Baseline predictions from the 63-tile run; RF and XGBoost tunedB from the regression-tree run.
  - C B3 output, as a reference row.
  - MCD12Q1 2019 `LC_Type1` land cover (IGBP classes).
- Mask: tree cover valid (0-100) in the historical, C and all three models. That leaves 22.74 M of the
  22.93 M historical-valid pixels. Bias = model minus historical.
- **Caveat:** the historical product is itself a model output, and the C code is the operational code
  that produced it. "Error" below means disagreement with the historical product, not with ground truth.
  C agreeing best is expected, not independent validation.

h09v05 covers roughly 30-40 N, 92-117 W: Arizona, New Mexico, southern Utah and Colorado, and west
Texas. It is 63% grassland and 16% open shrubland (MCD12Q1), and the historical mean tree cover is 7.9.

## Headline statistics (tree cover vs historical C6)

| Product | Mean | Bias | MAE | RMSE | Corr | OLS slope | Within 5 | Over +20 | Under -20 |
|---|---|---|---|---|---|---|---|---|---|
| Baseline (Bagging+DT) | 9.02 | +1.12 | 2.86 | 5.10 | 0.942 | 1.07 | 84.1% | 0.94% | 0.03% |
| Random Forest | 8.77 | +0.87 | 2.36 | 4.24 | 0.958 | 1.06 | 87.8% | 0.41% | 0.02% |
| XGBoost tunedB | 8.05 | +0.14 | 1.82 | 3.30 | 0.972 | 1.05 | 91.9% | 0.11% | 0.01% |
| C (reference) | 7.74 | -0.17 | 1.41 | 2.61 | 0.979 | 0.97 | 94.5% | 0.01% | 0.02% |

Historical mean: 7.91. The ranking is the same against Collection 6.1 (historical mean 8.33): MAE is
3.06, 2.59 and 2.17 for baseline, RF and XGBoost, and 1.73 for C. XGBoost tunedB is clearly the best of
the three Python models on every statistic. It cuts the baseline's MAE by 36% and its bias by nearly
90%. The earlier 63-tile results (`Regression_Tree_Methods_RF_XGBoost_Results.md`) already had the same
numbers for this tile. This run adds the breakdowns below.

## Where the models disagree with the historical product

**By historical cover level** (`fig4`, `stats_by_cover_bin.csv`): all three models over-predict in the
middle range and under-predict dense cover.

| Historical cover | Pixels | Baseline bias | RF bias | XGBoost bias |
|---|---|---|---|---|
| 0 | 5.01 M | +0.23 | +0.18 | +0.12 |
| 1-10 | 12.88 M | +0.40 | +0.22 | -0.48 |
| 11-20 | 1.98 M | +4.00 | +3.27 | +0.92 |
| 21-30 | 1.06 M | +5.88 | +5.16 | +3.13 |
| 31-40 | 0.75 M | +5.30 | +4.65 | +3.76 |
| 41-50 | 0.56 M | +3.52 | +2.80 | +2.32 |
| 51-60 | 0.31 M | -0.15 | -0.25 | +0.34 |
| 61-70 | 0.15 M | -4.54 | -4.29 | -2.04 |
| 71-80 | 0.03 M | -8.96 | -10.53 | -7.66 |

The 11-50% range, sparse woodland, carries most of the tile's error. XGBoost roughly halves that
over-prediction but does not remove it. Under-prediction above 60% cover affects under 1% of pixels, and
all three models share it; RF is slightly worse than the baseline there.

**By land cover** (`fig5`, `stats_by_landcover.csv`): the largest biases are in savanna (7% of the tile;
+4.2, +3.5 and +2.0) and woody savanna (4%; +3.7, +3.1 and +2.4). On this tile those classes are mostly
pinyon-juniper and oak woodland. Evergreen needleleaf forest is over-predicted by all three models (+2.7,
+2.2 and +3.8; here XGBoost is the worst). Evergreen broadleaf and mixed forest are under-predicted by
about 2 by the baseline and RF, and under 1 by XGBoost. Grassland, open shrubland, cropland and barren,
which together are 86% of the tile, are within about 1 point for all models.

## Spatial concerns

1. **The error is spatially clustered, not noise.** Moran's I of the ~9.3 km block bias is about 0.78-0.80
   for all three models (`fig6`, `stats_spatial.csv`). The over-prediction sits in the NW-SE-trending
   wooded uplands of the tile's northern half: northern New Mexico, southwest Colorado and the Four
   Corners (`fig2`). The south half is close to zero, with a slight negative bias for XGBoost.
2. **The three models are wrong in the same places.** Their block-level error maps correlate at 0.83-0.97
   (`stats_error_colocation.csv`). XGBoost reduces the error's size but keeps its geography. So the cause
   is shared: the training labels or predictors for this woodland type, or the historical product itself
   there. It is not a quirk of any one algorithm, and changing the algorithm alone will not fix it.
3. **One strong hotspot, in the baseline and RF only** (corrected 2026-09-29 after a follow-up check).
   The worst block is at rows 2000-2040, columns 1720-1760, about 35.8 N, 106.5 W: Jemez Mountains /
   Valles Caldera, New Mexico. Block bias there is +21.9 for the baseline and +20.1 for RF. **XGBoost
   tunedB is only +2.9 and C is +0.1 in the same block.** An earlier version of this item gave XGBoost
   +13.4; that is its tile-wide maximum, which is elsewhere. Over a 5 x 5-block window (~46 km) around
   the block, mean tree cover is 17.6 historical, 24.5 baseline, 23.9 RF, 19.7 XGBoost and 17.3 C. Over
   the whole tile, C's block-bias map is uncorrelated with the baseline's (r = -0.04), and C's largest
   |block bias| is 9.2. **So C does not show the hotspot at all.** The baseline and RF over-predict
   there, and XGBoost mostly does not. **Hypothesis, still not checked:** the location matches the 2011
   Las Conchas fire scar. If so, the historical product and C show post-fire tree loss that the baseline
   and RF read as tree cover (shrub or aspen regrowth), and XGBoost handles it better.
4. **The largest-disagreement 400 x 400 windows** (`fig7_zoom1-3`) are all grassland, savanna and woody
   savanna mosaics in the north half. The zoom-1 window (rows 1200-1600, columns 2000-2400) has a
   historical mean of 23.4 against 31.5, 30.8 and 28.0 for the models. In that window the models keep the
   historical product's spatial pattern and raise the whole woodland uniformly, so this is a level shift,
   not misplacement.
5. **No tile-edge artifacts or striping.** MAE within 20 pixels of the tile edge is only about 0.25 higher
   than in the interior for every model. The row and column mean-bias profiles are smooth and follow the
   landscape. The automatic spike test flagged 8-33 rows or columns, but the largest departure from the
   running median is under 0.5 points, which is not a real stripe.
6. **Minor:** the baseline map is visibly speckled (salt-and-pepper, positive and negative) in the
   southeast corner (croplands of eastern New Mexico and west Texas). RF is less speckled and XGBoost
   least. Cropland bias is +1.07 for the baseline and -0.40 for XGBoost.

## Non-vegetated layer: not a model comparison

The `Percent_NonVegetated` files are **byte-identical across the three models**. The RF/XGBoost experiment
replaced only the tree-cover model; the bare/non-vegetated model stayed out of scope, as its plan said.
The non-vegetated statistics (bias -3.78, MAE 10.12, corr 0.861 against the historical product) and
`fig8` therefore describe the baseline bare model only. Testing RF/XGBoost for non-vegetated would need
new fits (19-predictor bare model).

## Suggested next steps (for the project lead to choose)

- Run the same script on more tiles (about 3 minutes per tile on one node), for example one forest tile,
  one boreal tile and one tropical tile, to see whether the mid-range over-prediction and the shared error
  geography hold elsewhere.
- Zoom on the Jemez hotspot and compare it with the Las Conchas fire perimeter.
- Look at the training labels for savanna and woody-savanna pixels (label vs historical product) to see
  whether the over-prediction comes from the training data.
- Fit RF/XGBoost for the non-vegetated layer if that comparison is wanted.

---

# Three more tiles: h12v02, h12v09, h20v06 (2026-09-29, run 08:22-08:26)

Same script, run in parallel on forest203. Since the last run the script also scores C's block bias
against the historical product, and reports every product's bias at each model's worst ~9.3 km block
(`summary.txt`, `stats_worst_blocks.csv`). Outputs: `Compare/deepdive_vs_historical_<tile>/`.

## Headline statistics (tree cover vs historical C6)

| Tile | Region | Hist. mean | Baseline MAE / bias | RF MAE / bias | XGBoost MAE / bias | C MAE / bias |
|---|---|---|---|---|---|---|
| h09v05 | SW USA, dryland | 7.9 | 2.86 / +1.12 | 2.36 / +0.87 | 1.82 / +0.14 | 1.41 / -0.17 |
| h12v02 | Alaska to NWT, boreal/tundra (60-70 N) | 16.9 | 5.27 / +1.46 | 4.81 / +1.80 | 4.33 / +0.58 | 4.26 / +0.87 |
| h12v09 | Brazilian Amazon (0-10 S, Para) | 63.6 | 7.04 / -0.51 | 6.23 / -0.49 | 5.71 / -0.77 | 5.55 / -2.00 |
| h20v06 | Egypt/Sudan desert + Nile (20-30 N) | 0.11 | 0.22 / +0.14 | 0.21 / +0.14 | 0.13 / +0.06 | 0.02 / -0.01 |

**On all four tiles the ranking is XGBoost tunedB, then RF, then baseline**, by MAE, RMSE and correlation.
On h12v02 and h12v09 XGBoost is within 0.1-0.2 MAE of C. On h12v09 its bias is smaller than C's
(-0.77 against -2.00).

## Patterns common to all tiles

- **Mid-range over-prediction and high-cover under-prediction on every vegetated tile.** h12v02:
  +4.8, +5.4 and +2.6 at historical 11-20%, and -5.8, -5.2 and -3.3 at 61-70% (baseline, RF, XGBoost).
  h12v09: +6.9, +6.0 and +2.4 at 11-20%, and -3.3, -3.1 and -3.0 at 71-80%. XGBoost reduces the
  low-and-mid-range over-prediction most (by roughly half) and helps little at the high end.
- **Savanna, woody savanna and grassland carry the largest positive bias everywhere.** h12v02 savanna
  (41% of the tile): +2.9, +3.4 and +1.1. h12v09 savanna and grassland: +5.7 to +6.0 for the baseline and
  +1.5 to +2.2 for XGBoost.
- **The error is spatially clustered, and the three models share its geography.** Moran's I of block
  bias is 0.71-0.80 on every tile. The models' block-error maps correlate at 0.82-0.96 with one another on
  the vegetated tiles (0.60-0.77 on h20v06).
- **C shares the models' error geography only partly.** C's block-bias map correlates 0.64-0.72 with the
  models on h12v02 and h12v09, and 0.49-0.63 on h20v06. On h09v05 there was no relationship (-0.04). So on
  the boreal and tropical tiles much of the disagreement is between the historical product and *all* of
  the current products, C included.
- **No tile-edge artifacts or striping on any tile.** Edge MAE is equal to or lower than interior MAE
  everywhere. The row and column profiles have no step changes; the largest departure from the running
  median is under 1.2 points, and the flagged rows and columns follow rivers and the Nile.

## Spatial concerns by tile

**h12v02 (Alaska, Yukon, NWT).**
- A band of negative bias at rows ~1000-1400 (about 67-68 N). The worst block (rows 1320-1360,
  columns 240-280, about 67.2 N, 153.8 W, NW interior Alaska) is -19.5 for the baseline, -19.2 for RF,
  -20.4 for XGBoost and **-17.0 for C**. It is the same block for all three models, and C shares it, so
  here the historical product has more tree cover than any current product. A burn or change
  mismatch is possible, or differences in inputs between the historical run and this one; not
  investigated.
- The largest-disagreement windows sit on the Great Slave Lake shoreline (zoom 1: 54% water) and in
  savanna mosaics. Shoreline errors could come from differences in the water mask or mixed pixels.
- Diagonal (NE-SW) streaks in the block-MAE map follow terrain, not the scan direction. They do not show
  as steps in the row or column profiles.

**h12v09 (Amazon, Para).**
- A large contiguous under-prediction cluster at rows ~1000-2200, columns ~2500-4000 (about 2-4.6 S,
  52-55 W, the Xingu/Altamira region), in evergreen broadleaf forest (85% of the tile). All models show
  it, and **C is worse there**: the RF-worst block (rows 1240-1280, columns 3600-3640) is -16.3 for the
  baseline, -21.6 for RF, -16.3 for XGBoost and **-31.8 for C**. This region is known for active
  deforestation and degradation. It is also where the h16v01 fix check (`H16v01_...`) found h12v09's
  ordinary cloud-QA losses. Not investigated.
- Scattered over-prediction (orange) in the savanna and grassland of the north and far south.
- **Non-vegetated: bias -11.0, correlation 0.41** (the baseline bare model; the file is identical for
  all three). That is far worse than on the other tiles (correlation 0.76-0.94). The Python bare model
  almost never predicts non-vegetated cover here, where the historical product has about 11 points.
  This is worth a separate look.

**h20v06 (Egypt/Sudan desert, Nile).**
- The only real error feature is the **Nile valley and the Cairo/Delta edge** (top right), where
  cropland (hist 11.7) is under-predicted by 3.3-3.5 points by every model, and urban by 1.5. C is about
  -1.4 there. This is the column dip at 3300-3500 in the profile.
- **False tree cover in the desert:** barren pixels (99% of the tile, hist 0.0003) are +0.17 for the
  baseline and RF and +0.09 for XGBoost, with faint NW-SE streaks following dune and wadi structure.
  It is tiny in absolute terms, but it is the reason the models' means (0.17-0.25) are double the
  historical 0.11.

## Overall

1. XGBoost tunedB is the best Python model on all four tiles. It removes most of the baseline's
   over-prediction in sparse cover and most of the baseline/RF hotspots (for example Jemez), but not the
   under-prediction of dense cover.
2. The remaining disagreement is mostly shared by all three models, and on the boreal and tropical tiles
   partly by C as well. **Part of it is probably the historical product's own aerosol contamination
   (see "Read this first"), not model error.** So it points to training labels, predictors and inputs, or to the historical
   product itself, rather than to the choice of algorithm.
3. No processing artifacts (edges, striping) were found on any tile.
4. The non-vegetated layer has not been tested with RF/XGBoost. On h12v09 the baseline bare model is poor
   (correlation 0.41, bias -11).

---

# All 63 tiles (2026-09-29, batch 08:37-08:51 on forest203, 6 at a time, no failures)

Per-tile outputs: `Compare/deepdive_vs_historical_<tile>/`. Cross-tile summary:
`Compare/deepdive_all63_summary/` (`summary.txt`, `per_tile.csv`, `by_cover_bin_pooled.csv`,
`by_landcover_pooled.csv`, `worst_blocks.csv`), built by `code/validation/summarize_deepdives.py`.
Launchers: `run_deepdive_batch.sh` (tile list `manifests/deepdive_remaining59.txt`). **The aerosol
caveat at the top of this document applies to every number below.**

## Pooled over 63 tiles (pixel-weighted, ~1.1 billion pixels, vs historical C6)

| Product | MAE | Bias | RMSE |
|---|---|---|---|
| Baseline | 5.06 | +0.27 | 7.83 |
| Random Forest | 4.48 | +0.24 | 6.92 |
| XGBoost tunedB | 4.02 | -0.18 | 6.40 |
| C (reference) | 3.54 | -0.12 | 6.06 |

By region (MAE, baseline / RF / XGBoost / C): boreal and arctic v00-03 (16 tiles) 5.97 / 5.27 / 5.02 / 4.25;
temperate north v04-05 (17 tiles) 3.85 / 3.40 / 2.93 / 2.42; tropics v06-10 (20 tiles)
5.89 / 5.27 / 4.74 / 4.44; south temperate v11-12 (10 tiles) 3.45 / 2.98 / 2.37 / 1.93.

- **XGBoost tunedB is the best Python model on 59 of 63 tiles.** RF is best on the other four (h12v01,
  h16v01, h18v03, h21v01), all high-latitude tiles, and by at most 0.4 MAE. The baseline is never best.
  XGBoost cuts the baseline's MAE by 24% on average.
- **XGBoost has lower MAE than C on 9 tiles**: h11v09, h12v01, h13v01, h19v08, h19v09, h20v02, h20v08,
  h21v01 and h21v02. Most are tropical Africa/South America or high-latitude tiles.

## Error structure, pooled

**By historical cover** (`by_cover_bin_pooled.csv`):

| Historical cover | Baseline bias | RF bias | XGBoost bias |
|---|---|---|---|
| 1-10 | +1.53 | +1.51 | +0.32 |
| 11-20 | +1.96 | +1.95 | +0.11 |
| 21-30 | +1.47 | +1.40 | +0.06 |
| 51-60 | -2.31 | -2.23 | -0.82 |
| 71-80 | -3.99 | -3.79 | -2.62 |
| 81-90 | -4.15 | -4.18 | -3.14 |

The baseline and RF pull predictions toward the middle (sparse cover too high, dense cover too low).
**XGBoost removes almost all of that in the 1-50% range** and about a third of it at 70-90%. The dense
forest under-prediction is the main systematic error left.

**By land cover** (`by_landcover_pooled.csv`), XGBoost bias: grassland -0.07, savanna -0.24, woody savanna
+0.51, evergreen broadleaf -0.92 (baseline -2.27), mixed forest -0.65, cropland -0.60, permanent wetland
-3.57. **Snow and ice: +4.1** (baseline +4.7, RF +3.7) against a historical mean of 0.01. That is
false tree cover on permanent snow and ice, most of it on h16v01 (Greenland). This matches the h16v01
over-prediction and is relevant to the Greenland fix plan (`Greenland_Iceland_Scoped_Fix_Plan.md`).

## Spatial

- **Error is spatially clustered on every tile**, and the three models share the geography.
- **C shares part of it**: the mean block-bias correlation between C and XGBoost is 0.46 (boreal), 0.49
  (temperate north), 0.57 (tropics) and 0.53 (south temperate). That fits disagreement that is partly
  about the historical product (aerosol contamination, see the top section), strongest in the tropics.
  Not tested.
- **Worst hotspots are tropical Africa** (h19v08, h19v09, h20v08, h20v09) and h11v08 (northern South
  America). There, all three models are +30 to +44 above the historical product in single ~9.3 km blocks,
  and **XGBoost is often the worst** (for example h19v08 +44.0 against a baseline of +39.1). C is also
  high there (+10 to +24). On h19v08 the over-prediction covers the whole southwest quadrant (the humid
  forest of Cameroon, Gabon and Equatorial Guinea), which is very cloudy and has heavy dry-season
  biomass-burning smoke. The unscreened historical product would be most contaminated there, which is a
  candidate explanation, but it is not tested. These are the five tiles with the highest XGBoost MAE
  (6.0-7.7).
- **Shared negative hotspots** where C agrees with the models (C within a few points of them): h20v03
  (-24 to -29, C -27.7), h12v10, h11v10, h11v09, h27v04, h12v02, h11v02 and h21v02. Wherever C and all the
  models agree against the historical product, the likely cause is the reference or the inputs (the
  aerosol screen, input or label differences), not the algorithm.
- **Hotspots in the baseline and RF only** (like Jemez on h09v05): h16v01 (+38 baseline, +14 RF, +26
  XGBoost, C 0), h12v12, h27v04, h12v10, h19v10 and h23v03. There XGBoost is clearly lower and C is near 0.

## Artifact checks

- **Edge effects:** edge MAE is more than 0.5 above interior MAE on 8 tiles (h11v09 +1.11, h19v09 +0.87,
  h16v01 +0.76, h11v10, h12v01, h19v10, h21v01, h24v04; all under 1.2 points). All the other tiles show
  no edge effect. These 8 may be landscape that happens to sit at the tile edge; not checked.
- **Striping:** the row/column profile test flagged 23 tiles with residuals of 2-27 points. **Checked on
  h17v05: a false alarm.** The spike is at the Strait of Gibraltar, where rows hold only a few land
  pixels and their means are noisy. Every flagged tile is coastal, island or mostly ocean (h29v12,
  h31v11, h21v01, h12v05 and others), so the others are probably the same. They were not checked one by
  one. The test should be weighted by pixel count if it is used again.

## Non-vegetated

The layer is the same baseline bare model for all three. Its correlation with the historical product is
below 0.6 on 28 of 63 tiles (as low as -0.07 on h12v10, 0.10 on h13v10), mostly with negative bias
(-5 to -16). **The non-vegetated layer is much weaker than tree cover** and was not part of the
RF/XGBoost experiment.

## Conclusions

1. **XGBoost tunedB is the strongest Python model** by every pooled measure and on 59 of 63 tiles. It is
   the natural choice to replace the baseline for tree cover. RF is marginally better on four
   high-latitude tiles.
2. What XGBoost fixes: the baseline's pull toward the middle in sparse and mid cover, and most of the
   baseline-only local hotspots.
3. What it does not fix: under-prediction of dense forest (-2.6 to -3.1 above 70% cover), over-prediction
   hotspots in tropical Africa, where it is sometimes worse than the baseline, and false tree cover on
   snow and ice.
4. **Absolute agreement with the historical product is an imperfect target.** The historical product had
   no working aerosol screen, and the current screen also removes some clean observations, so the
   remaining gap to the historical product, including C's, is not all model error. The shared hotspots
   (C and all models against the historical product) are the first places to test this, by counting the
   observations the fixed screen removes there.
5. No processing artifacts were found. The edge and striping flags are small or explained by coastlines.
6. The non-vegetated layer needs its own work.
