# What the new RF bare layer does in the C code's special regions (plan, to-do item 5)

**Written 2026-10-01 evening. Status: plan only, awaiting approval. Nothing has run.**

**Scope (project lead, 2026-10-01): describe what the new non-vegetated (bare) layer does in the regions where the C code applies special rules, so that we can decide whether Python needs anything there.** Both regions are covered. Nothing is changed in the product. A what-if is shown only where it helps describe the region.

Product examined: the delivered non-vegetated layer (RF, MEAN, sum cap) from `delivery_sd_rf_median_20261001/out/full63`, with its SD layer.

## The C code's two regions (`MOD_PR44B3/MOD44B3.c`; `Bare_Static_Model_Pixel_Trace.md`, section 3)

**A. High latitude ("Greenland" rule):**
- Bare is set to 100 where the slot-11 snow flag is 0 (snow, or zenith over the cutoff, at DOY 241), or where slots 11-13 (DOY 241-273) have no clear observation.
- In v7 this applies only to h < 18 and v < 3; in v6.0.3 (C6 era) to every v < 3 tile.
- **Tiles among the 63:**
  - v7 box: h11v02, h12v01, h12v02, h13v01, h16v01;
  - v6.0.3 box adds h20v02, h21v01, h21v02, h23v02.
- **Later:** the 6 new Greenland/Iceland tiles from item 11 (all inside the v7 box) are added if item 11 has produced them.

**B. Sahara stretch ("volcanic soils and varnish"):**
- On h16-h23, v05-v08, bare ≥ 55 becomes 2 × bare − 55 (capped at 100). Hand-drawn lines exempt parts of h22/h23 v05/v06.
- **Tiles among the 63:** h17v05, h18v07, h19v08, h20v06, h20v08, h21v05, h21v06. None is an exempted h22/h23 tile.

## Method (read-only; forest nodes; after the delivery run ends)

**Region A, high latitude:**
1. **Rebuild the rule mask as the C code defines it,** from the MOD44CQ state for DOY 241, 257 and 273 (snow flag and clear flag, as `MOD44B3.c` reads them).
   - **Check:** on the v7 tiles, the C code's 2019 bare map must be 100 on every masked pixel. Any mismatch is reported before going on.
   - Both the v7 mask and the v6.0.3 mask (all v < 3) are kept.
2. **Inside and outside the mask, per tile:**
   - the RF bare distribution: mean, median, share ≥ 90, share < 50;
   - RF tree cover;
   - the SD of both layers;
   - the same for C6, C6.1 and the C code;
   - the training labels where the tile has them;
   - the size of the gap between RF and 100 inside the mask: what the rule would overwrite, and by how much.
3. **What kind of land the mask covers:** the C6 cover classes of the masked pixels, and the labels' tree cover and bare there. This shows whether the masked pixels are ice/rock (where 100 is right) or tundra/forest (where 100 would be wrong).

**Region B, Sahara:**
1. **Per tile and by C6 bare bin (0-20, 21-40, 41-60, 61+):**
   - the RF bare distribution and the share ≥ 55 (the pixels the stretch would touch);
   - RF against C6, C6.1 and the C code: MAE, bias.
2. **Is C6's desert level the stretch?** Find whether the C6-era code (v6.0.3) applied the stretch, since it is in the code tree searched in the trace report. If it did, RF's low desert bias against C6 (Africa 10-30N 61+ bin: -6.3) is at least partly the stretch.
   - **What-if:** apply the stretch to RF (read-only, in memory) and report MAE and bias against C6, C6.1, the C code, and the held-out bare labels on these tiles.
3. **Sum cap interaction:** how many stretched pixels would exceed 100 − tree.

## Output

- `FullRuns/bare_c_regions_<date>/`: tables per tile, region and bin; masks as GeoTIFFs for inspection.
- `Reports/Bare_C_Regions_Investigation_Results.md`. For each region it states what RF does there and the options (do nothing / port the rule / something else), for the project lead to decide.

## Expected cost

About 45 minutes on forest nodes: masks for 9 tiles about 15 minutes, statistics about 20 minutes, report extra. Region A is repeated for the 6 new tiles once item 11 has produced their maps, about 10 minutes more.

## Not in scope

Porting either rule into Python. That is a decision after this report.
