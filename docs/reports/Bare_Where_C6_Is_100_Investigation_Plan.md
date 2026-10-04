# Non-vegetated (bare) where C6 is 100 and the new models are not: investigation plan

**Written 2026-10-04 at the project lead's request. Status: approved 2026-10-04 with the decisions below; step 0 launched first. Read-only: no product, model or branch changes.**
Names follow `Product_Versions_and_Work_Map.md`. "Bare" and "non-vegetated" are the same layer. C6, C6.1, the C code and every
external map are **comparisons, not truth**. The held-out labels (Landsat 30 m) are the only independent check, and they cover
20 tiles.

## Project lead's decisions (2026-10-04)

1. **Build the 60-tree non-vegetated model first** (step 0) to see what it changes, before the rest.
2. **External data: use only what is on ADAPT** (ESA WorldCover 2020, MCD12Q1 2019, C6.1, the C6 series). These are reference, not a
   decision gate. If no conclusion can be reached, downloading more external data is a decision for a later round of testing.
3. **No NGA** for now (too much work for the return).
4. **Stay on the 69 tiles** (the 63 plus h15v02, h16v00, h16v02, h17v00, h17v01 and Iceland h17v02; h16v01 is one of the 63). There is no local MOD44C outside them, so no new tiles.
5. **Added:** predictors beyond the C code's 21 and the MC 64 are part of the thinking. More information is available to train on
   if needed (new step 7).

## Why

On the first non-vegetated figures (`NonVeg_Figures_v100_MC64_Results.md`), the project lead's concern is the large areas where
**C6 is at or near 100 and v1.0.0 is not**:

1. **Greenland ice sheet:** large areas below 100.
   - Earlier result: RF bare inside the C code's high-latitude mask is 88-94 on the ice sheet tiles and 73-79 on the polar-night tiles
     (`Bare_C_Regions_Investigation_Results.md`).
   - h16v02 shows a straight-edged lower-than-C6 patch.
2. **Deserts (h20v06, h21v06):** C6 is 100 over most of the land, and v1.0.0 drops below it in places.
   - Tile MAE against C6 is small (0.56 and 1.72), so the drops are local.
   - The earlier analysis found the 41-60 C6 bin hard on these tiles.
3. **Australia (h29v11, h29v12, h30v12, h31v11):** v1.0.0 is 9-14 lower than C6 on average, with 30-43% of pixels more than 15
   points lower. The C code has no known Australian rule; this is to be confirmed.
4. **In general:** the C code reset values with **post-model rules** where the models fell short. It did not fix the models. Those
   rules need a deeper look before deciding between a model fix, a rule, or nothing:
   - the v7 high-latitude mask (bare = 100);
   - the Sahara stretch;
   - the sum cap.

This is the "bare side rules" work deferred until after model assessment (memory: bare side rules deferred). The C code's
`bare_earth3` stays ignored.

## Considerations when reading any C6 comparison (project lead, 2026-10-04; not separate work items)

- **The static bare models may expect a calendar shifted by about half a year.**
  - **Earlier finding** (`Bare_Static_Model_Pixel_Trace.md`, `Weka_M5P_Bare_Models_to_Python.md`): rotating the unsorted inputs by
    4 periods (about 192 days) cut the static M5P models' MAE against C6 from 23.1 to 8.9. The best shift was 4 on 14 of 20 tiles, 5 on
    h12v01, h20v06 and h21v06, 3 on h11v10 and h16v01, and 6 on h27v04.
  - **Not resolved:** where the shift came from (operational composite order, or a different training convention).
  - **Use here:** a possible source of difference between C6 (static models plus rules) and any retrained model, separate from the
    rules. No deep dive. The existing per-tile rotation results (`trace/period_rotation_per_tile.csv` of that work) are read alongside
    the step 1 screen.
- **C6 may have been tuned for fully bare pixels** (a theory; cannot be confirmed). C6's bare product may have aimed to be as right as
  possible at 100% bare, with less effort on the middle fractions.
  - **Use here:** a possible explanation when C6 agrees at 100 and disagrees in the middle.
  - **Cheap check** (in step 2d): C6 against the held-out labels by label bin, including label = 100. If C6 is close at 100 and weak
    in 21-40 and 41-60, that is consistent with the theory, though it does not prove it.

## Product under study

- **v1.0.0 non-vegetated** (RF, 30 trees, MEAN, rounded): the 63 tiles, plus the 6 Greenland tiles and h17v02.
- **The 60-tree non-vegetated model**, adopted on 2026-10-04 but not yet built. Step 0 builds it first, so that the findings apply
  to the model that will ship.

## Steps

### 0. Setup (about 1 h)

- **List the bare-label tiles.** Get the 20 tiles that carry bare labels from `mc_rerun_20261002/Python/split/bare/keys*.parq`.
  Record which named tiles have labels (Australia, h20v06, h21v06, Greenland: h16v01 only).
- **Fit the 60-tree non-vegetated RF.** Use the C code's 21 predictors, MEAN, the same keyed 80/20 pixel split, and seed 0. Map
  the 69 tiles detached (run `FullRuns/bare60_20261004`, launched 2026-10-04 08:13; ngaproc202 + forest201-205, since ngaproc201 is unreachable). Score it on held-out labels against v1.0.0 with a seed-1 yardstick, as was done for T60.
  - About 3-4 h.
  - Approved: runs first.

### 1. Global screen: where is C6 = 100 and the new map not? (about 45 min)

**Per tile and per location stratum** (continent and latitude band from h/v, not land cover), on land:
- the share with C6 bare = 100 and ≥ 95;
- among those pixels, the distribution of the new bare value (100 / 90-99 / 75-89 / < 75), and the same for C6.1 and the C code
  (63 tiles);
- where labels exist, the label distribution;
- the reverse case: new = 100 where C6 < 100.

**Outputs:**
- a ranked table of where the gap is largest;
- a "C6 = 100, new < 90" mask per tile;
- a figure set in the bare colours (two rows: v1.0.0 and the 60-tree model).

This decides which tiles beyond the named ones enter steps 2-5.

### 2. Mechanism checks on the model output, no refit (about 1-2 h)

| Check | Question | How |
|---|---|---|
| a. Aggregation | Is MEAN pulling 100s down? With MEAN, a pixel is 100 only if every tree says 100. | Per-tree predictions on a sample (about 20 k pixels per region: ice, desert, Australia, controls). Share of trees at 100; MEAN against MEDIAN against an upper quantile. Links to-do 10 (MEAN is provisional). |
| b. Sum cap | Does forcing tree + non-tree + bare ≤ 100 cut bare? | Bare before and after the cap; tree and non-tree predicted at C6 = 100 pixels (for example, about 6-7 tree on the polar-night tiles). |
| c. Training labels | Are labels at 100 rare in these settings? | Training-label histogram by region. Leaf composition: how often a leaf reached by a C6 = 100 pixel holds only 100s. |
| d. Held-out labels | Where labels exist, who is right? | v1.0.0, the 60-tree model **and C6** against labels at the C6 = 100 pixels and overall, by cover bin (0-20 / 21-40 / 41-60 / 61+, plus label = 100) and stratum. C6 by bin tests the "tuned for 100" theory. |
| e. Out of range | Are these pixels outside the training range? | For the 21 predictors, the share of C6 = 100 pixels outside the 1-99 percentile range of the training rows. Ice and bright desert are the likely cases. |

### 3. Pixel traces and the QA (about 2-3 h)

**Pixels:** about 20-30 per region.
- **Gap pixels:** C6 = 100, new < 90.
- **Controls:** C6 = 100 and new = 100 nearby.
- **Special cases:**
  - both sides of the h16v02 straight edge;
  - the polar-night tiles;
  - h18v07 stretch pixels;
  - Australian pixels at C6 high and labels present, if any.

**Each trace, per 16-day period:**
- MOD44C band values;
- QA state: the bit-15 cloud handling, cloud, snow and aerosol bits, solar zenith (`solz <= 0` is a known edge case);
- days contributing, and the fallback flag and count;
- NDVI;
- the metrics behind the 21 bare predictors, with each one's percentile in the training data;
- per-tree predictions and SD;
- the C code value and its rule flags (v7 mask, slot-11 snow flag, stretch);
- C6, C6.1 and nearby labels.

**Hypotheses to confirm or reject:**
- **Ice:**
  - snow and ice observations handled as cloud, leading to fallback;
  - low sun or no valid zenith;
  - saturated visible bands;
  - a composite-period or input boundary behind the h16v02 edge.
- **Desert:**
  - bright surfaces or dust flagged as cloud or aerosol, leading to fallback to cloud-affected means;
  - rock against sand, where the labels are below 100 and C6 is stretched.
- **Australia:**
  - Is C6's high bare there rule-driven or data-driven? (C6 2000-2020 series)
  - season timing;
  - burn scars (`ilab/data/MODIS/AnnualBurnScar`);
  - sparse shrubs below the model's detection.

**Region-scale QA summary:** per period, the share of land with fallback, cloud, snow or aerosol flags in the C6 = 100 zones against
the rest of the same tile.

**Caveat:** C6 carries the historical aerosol-bit bug (memory: historical MOD44B aerosol bug caveat), so C6's own composites differ
from ours in exactly these bright and hazy areas.

### 4. The C code's post-model rules (about 2-3 h, mostly reading and reruns on hand)

**Inventory every post-model reset** in the C code (`MOD_PR44B3`: `MOD44B3.c`, `Predict.java`, `fix_bands.c`) and its
`HISTORY.txt`. For each rule, record:
- region, condition, effect and version;
- whether it shows in C6 and C6.1;
- what the labels say;
- what the external maps say.

**Known rules:**
- the v7 high-latitude mask (h < 18, v < 3, slot-11 snow flag or no clear observation on DOY 241-273, bare = 100);
- the Sahara stretch (h16-h23, v05-v08, bare ≥ 55 becomes 2 × bare − 55, with hand exemptions on h22/h23 v05/v06);
- the sum cap.

**Find any others,** for example Australia, Antarctica or water edges.

**Resolve the open conflict:** the rebuilt v6.0.3 mask does not match C6 on h20v02, h21v01, h21v02 and h23v02.

**Output:** a rule table with columns rule | region | effect on C6 | labels | external maps | options.

### 5. External comparison maps (comparison only; about 2-3 h for local data)

All maps are aggregated to the MODIS 250 m grid on ADAPT and compared by stratum and by C6 bin. The project acts only on
differences that are consistent across references.

**On ADAPT now:**
- **ESA WorldCover 2020, 10 m** (`ilab/data/ESA_WorldCover`): the fraction of class 60 (bare / sparse vegetation) and class 70
  (snow and ice) per 250 m cell. A class fraction is not percent bare; class 60 allows up to about 10% vegetation.
- **MCD12Q1 C6.1, 2019** (`/css/modis/Collection6.1/L3/MCD12Q1-Landcover`): IGBP 15 (snow and ice) and 16 (barren), and the
  LCCS layers.
- **MOD44B C6.1, 2019** (on hand), and the **C6 2000-2020 series** (`MOD44B-VCF_reconstructed`, never C6.1). The series asks
  whether C6 = 100 is stable across years, which would suggest a rule; noisy values would suggest data.
- **The C code's 63-tile outputs** (on hand).

**Not on ADAPT: deferred** (project lead, 2026-10-04). Downloading these is a decision for a later round, only if the local data
cannot settle a question:
- Copernicus CGLS-LC100 2019 **bare cover fraction** and snow fraction (100 m, same year). This is the closest concept to percent bare.
- A Greenland ice mask (GIMP or PROMICE), for ice-sheet extent.
- RGI 7 glacier outlines, for Iceland.
- GLC_FCS30D, a Landsat-based check.

**NGA very-high-resolution scenes: not used** (project lead, 2026-10-04).

### 7. Additional predictors (scoped after steps 2-3)

**The problem:** the C code's 21 bare predictors, and the MC 64 search over the 324-metric pool, may not carry what separates
fully bare from almost bare on ice, bright desert and arid shrubland. More information is available to train on.

**The pool on hand:** the 324 metrics (`mc_rerun_20261002/Python/metrics_324`, 63 tiles; the bare split already holds all 324
columns). The 7 Greenland/Iceland tiles would need these metrics built from their MOD44C.

**What the pool lacks:** the 324 metrics are built from bands 1-7, band 31 and NDVI. NDVI is the only index, so nothing combines the
mid-IR (SWIR) bands with the others. Band 5 is in the pool but is known to have bad detectors on Terra. Neither the C code's 21 nor
either MC bare set uses it, and new indices should not use it either.

**Candidate indices (project lead, 2026-10-04): mid-IR focused, bands 6 (1.64 µm) and 7 (2.13 µm), never band 5.** They are cheap to
make: the same 8 periods and the same metric families (unsorted per period, calendar means, sorted ranks, greenness-ordered) as the
existing bands, so each index adds a family of columns.

| Index | Bands | Why it may help here |
|---|---|---|
| NDSI | (B4 − B6) / (B4 + B6) | Snow and ice against rock and soil (the ice sheet below 100; snow-affected periods) |
| NDII / NDMI | (B2 − B6) / (B2 + B6) | Canopy and soil moisture: sparse shrubs against dry bare ground |
| NBR | (B2 − B7) / (B2 + B7) | Vegetation against bare soil; burn scars (Australia) |
| SWIR ratio / NDTI | B6 / B7, or (B6 − B7) / (B6 + B7) | Dry, non-green vegetation and litter against bare soil and rock; also clay and carbonate minerals |
| BSI (bare soil index) | ((B6 + B1) − (B2 + B3)) / ((B6 + B1) + (B2 + B3)) | Bare soil directly |
| NDBI-type | (B6 − B2) / (B6 + B2) | Bright bare surfaces (rock, sand, built-up) |
| SWIR brightness | B6 + B7, or an albedo-type sum of B1-B4, B6, B7 | Bright desert against darker sparse shrubland |

**Other candidates,** chosen from what the traces (step 3) and the range check (2e) show is missing:
- thermal seasonality;
- greenest and driest period metrics at the extremes;
- QA-derived counts (fallback periods, clear observations) as predictors.

**When to test them:** only if steps 1-3 point to a model shortfall (for example, gap pixels out of range or indistinguishable in
the 21 predictors), not to aggregation, the sum cap, labels or C6's own rules. Then:
1. Build the index families on a few tiles from each region first.
2. Run importance on the gap strata.
3. Refit at 60 trees on the same split, with the C code's 21 plus a short list of index columns.

Location or ancillary layers (latitude, elevation) are noted but not proposed; the project lead decides whether they are acceptable.

**Method:**
- **Importance:** region-restricted importance of the 324 within the gap pixels' strata.
- **Additions:** add a few targeted predictors to the C code's 21 (or a short list).
- **Refit:** 60 trees on the same split.
- **Scoring:** held-out labels overall and by region and cover bin, maps against the references, and the seed yardstick.

The MC 64 for non-vegetated was a tie overall (+0.11). This step asks whether specific predictors help **in the problem regions**
without costing elsewhere.

### 6. Candidate remedies: evaluate only, no adoption (scoped after steps 1-5 and 7)

1. **Aggregation:** MEDIAN or an upper quantile for bare.
2. **Predictors:** the C code's 21 plus targeted additions from step 7.
3. **Training:** add 100-labels in ice and desert, following the existing label rules (stable sites).
4. **QA and compositing:** handle snow and bright surfaces so they do not trigger fallback.
5. **Post-model rules:**
   - a high-latitude rule restricted to the ice sheet (mask from an external ice map instead of the snow flag);
   - the stretch (the labels already argue against it on h18v07);
   - the sum cap order.
6. **Do nothing,** where the labels side with the new model against C6.

**How each remedy is scored:**
- held-out labels (20% pixel split; never whole-tile hold-outs);
- maps against C6, C6.1, the C code and the external maps;
- by stratum and cover bin;
- with a seed yardstick.

Adoption is for the project lead.

## To-do list (bare)

| # | Item | Step | Depends on |
|---|---|---|---|
| B1 | List the 20 bare-label tiles; fit and map the 60-tree non-vegetated model | 0 | approval |
| B2 | Global screen of C6 = 100 / new < 100, ranked by tile and stratum, with figures | 1 | B1 (or v1.0.0 only) |
| B3 | Greenland ice sheet below 100 (incl. the h16v02 straight edge, the polar-night tiles, and Iceland's E Greenland strip) | 2, 3, 5 | B2 |
| B4 | Desert drops where C6 = 100 (h20v06, h21v06; then h17v05, h18v07, h21v05) | 2, 3, 5 | B2 |
| B5 | Australia lower than C6 (h29v11, h29v12, h30v12, h31v11): rule- or data-driven? | 2, 3, 5 | B2 |
| B6 | Aggregation and sum-cap effect on 100s (MEAN against MEDIAN; to-do 10) | 2a-b | B1 |
| B7 | Inventory of the C code's post-model rules; resolve the v6.0.3 conflict | 4 | none |
| B8 | External maps on ADAPT as reference (not a gate); downloads deferred to a later round | 5 | none |
| B11 | Additional predictors: importance of the 324 in the problem regions, targeted new mid-IR indices from bands 6-7 (NDSI, NDII, NBR, SWIR ratio, BSI, NDBI, SWIR brightness; never band 5), QA counts, refits at 60 trees | 7 | B2, B3-B5 traces |
| B9 | Candidate remedies and their evaluation | 6 | B2-B8, B11 |
| B10 | Results report `Bare_Where_C6_Is_100_Investigation_Results.md`; README and work map | all | |

## Compute, disk and time

- **Compute:** ngaproc201/202 (ngaproc201 was unreachable on 2026-10-04; check first) and forest nodes. Fully detached, with
  status files and the usual watchers.
- **Time:** steps 1-5 are read-only analysis, about 1 working day, plus about 3-4 h for step 0 if approved.
- **Disk:** under 50 GB (the 60-tree bare model is about 34 GB; masks, traces and aggregated external maps are small).

## Questions for the project lead

Answered 2026-10-04 (see "Project lead's decisions" above).
