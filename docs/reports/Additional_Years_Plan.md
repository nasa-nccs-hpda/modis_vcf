# Additional years: per-year training, per-year models and products on the 63 training tiles (plan, to-do item 12)

**Written 2026-10-02; revised the same day (project lead: build new models for each year, save each year's training, no lidar testing; clean-up of intermediate files accepted). Status: plan only. Not approved for execution yet. Nothing has run.**

## Goal

For each extra year:
1. build and **save** that year's training set;
2. fit that year's own models (tree cover and non-vegetated);
3. make that year's product with them.

The saved per-year training sets keep open a later test, not part of this plan: is **one model trained on all years pooled** better than **individual models for each year**? This plan is set up so that test is clean (see "Set up for the pooled-model test").

## What data there is (checked 2026-10-02, read-only)

A processing year N needs MOD44CH and MOD44CQ for 23 dates: DOY 065-353 of year N and DOY 001-049 of year N+1. Both local locations were checked: `/css/modis/Collection6.1/L3/MOD44B-VCF/dev/<year>/` and `/explore/nobackup/projects/ilab/data/MODIS/MOD44C/`.

| Year N | 63 tiles | Note |
|---|---|---|
| **2000, 2004, 2009, 2010, 2014** | **complete** | Same as 2019: h12v01, h13v01, h16v01 and h21v01 lack only the polar-night DOY 337 and 353, which are never produced |
| 2001, 2005, 2015, 2023 | year N complete; DOY 001-049 of N+1 missing (504 files per year) | Need the 2002, 2006, 2016 and 2024 files from the private repository |
| 2011, 2020 | partial | Not usable without more downloads |

- **Labels:** the training labels are per tile, not per year: tree cover on 63 tiles and non-vegetated on 20 (`VCF_training_pcttree_<tile>.7.0.0.out`, `VCF_training_pctbare_<tile>.7.0.0.out`). The C production used the same labels in every year. So a per-year training set is the same label pixels with that year's metrics.
- **Reference:** the reconstructed C6 MOD44B (`/css/modis/Collection6/L3/MOD44B-VCF_reconstructed/<year>/065/`) has all 63 tiles for all 5 years.
- **Greenland tiles:** h15v02, h16v00, h16v02, h17v00, h17v01 and h17v02 have MOD44C only for 2019, so they are out of scope. h16v01 (one of the 63) gets the zenith-only NDVI fix in every year through its tile list.

**This plan covers the 5 complete years: 2000, 2004, 2009, 2010 and 2014.** The 4 years that need downloads can be added the same way, at the same cost per year.

## Method

Run root: `FullRuns/more_years_<date>/`, new. Code: frozen copy of the delivery branch at `16d6474` or later: RF, tree = MEDIAN, non-vegetated = MEAN, rounding, SD layers, the Greenland zenith fix, and the v00 rule (which touches none of these tiles).

Model settings are identical to the delivered 2019 models: Random Forest, 30 trees, the same hyper-parameters, seed 0, and the same predictors (the delivered 64 for tree cover, C's 21 bare attributes for non-vegetated). **Only the year of the metrics changes,** so differences between years' models come from the data, not the setup.

Everything for year N goes under `<year>/` subfolders, kept and never overwritten:
- `Python/metrics/<year>/`
- `Python/training/<year>/{pcttree,pctbare}/` (per-tile tables)
- `Python/training_split/<year>/{pcttree,pctbare}/`
- `Python/models/<year>/{pcttree,pctbare}/`
- `out/<year>/` (products)

**Step 0. Preflight** (about 15 min)
- Re-run the inventory and confirm all 23 dates for each tile and year.
- Check free disk.
- Freeze the code; write manifests of the 315 tile-years.
- Record the 2019 split keys (see the pooled-model setup).

**Step 1. Gate: h08v05 for 2010, from end to end** (about 1 hour)
- `makeMetrics` from the daily inputs, then `buildTraining` for that tile (tree and bare).
- **Automatic checks** (a failure stops the run):
  - all 23 dates were read, and the DOY 001-049 files came from year N+1;
  - the 21 metric files are complete, with no NaN columns;
  - the tile's training table has the same label pixels (same keys and label values) as its 2019 table, and the metric columns are filled.

**Step 2. Metrics, one year at a time** (2010, 2014, 2009, 2004, then 2000 last; see the 2000 caveat)
- `makeMetrics` builds only the families the two models need: the 19 C families, the unsorted bands and NDVI.
- Workers: forest201-205 (2 per node) and ngaproc201/202 (4 per node), 18 at once.
- **Time:** about 25-35 minutes per tile-year from the daily inputs (h17v00 measured at 21.5 minutes on 2026-10-01 and -02; in September a full 324-metric build took 34-37 minutes). One year ≈ 4 waves ≈ **about 2 to 2.5 hours.**

**Step 3. Training, split and fits, per year** (as soon as that year's metrics are done; overlaps the next year's metrics)

| Sub-step | What | Measured in the 2019 retrains | Per year |
|---|---|---|---|
| 3a. Build training | `buildTraining` per tile: tree on 63 tiles, bare on 20, from that year's metrics and the static labels | about 4-5 min per tile, 4 at a time per node | about 20-30 min |
| 3b. Keyed split | Held-out pixels chosen by location, the same as 2019 (see below) | tree 14 min, bare under 1 min | about 15 min |
| 3c. Acceptance test | Train and test keys equal the 2019 split's keys; row counts equal 2019, minus pixels with no valid metrics that year (reported) | | a few min |
| 3d. Fit | Tree RF and bare RF, settings as delivered; run in parallel on 2 nodes | tree 8 min, bare 4.5 min | about 10-15 min |
| 3e. Held-out score | That year's models on that year's held-out rows | | about 5 min |

**Per year: about 1 hour**, mostly hidden behind the next year's metrics.

**Step 4. Products, per year** (with that year's own models)
- `vcfPredict` with the delivered settings (MEDIAN / MEAN, rounding, SD), using `models/<year>/`. Runs on ngaproc201/202 (3 per node, about 42 GB each) and on forest nodes when free (1 per node).
- **Time:** 6.5-8 minutes per tile-year (measured today on h08v05). One year: **about 45-85 minutes**, overlapped.
- **Checks on every tile-year:** 5 layers present; the layers sum to 100 on land; water and out-of-projection codes the same as in 2019; SD is 0 off land.

**Step 5. Evaluation** (no lidar, as instructed)

**a. Held-out labels, model year × data year** (the evidence base for the pooled-model test)
- Every year's tree and bare models (2019 included, so 6) score every year's held-out rows: a 6 × 6 table per layer.
  - The diagonal is "individual models".
  - The off-diagonal shows how well a model carries to other years.
- About 2.3 M tree rows and 0.4 M bare rows per year; one model on one year takes about 1-2 minutes. **About 1 to 1.5 hours** on 2-3 nodes.

**b. Products against C6 for the same year**
- MAE, bias and RMSE: pooled, by tile, by C6 bin (0-20 / 21-40 / 41-60 / 61+), and by continent and latitude band. This reuses `summarize_agg.py` from the rounding run, with a year parameter.
- **About 20 minutes per year**, overlapped.

**c. Stability across years**
- On the stable set (pixels whose C6 SD over 2000-2020 is ≤ 5), the spread of the Python products across the 6 years against C6's spread on the same pixels.
- The year-to-year change, Python against C6, per tile.
- **About 30 minutes.**

**d. Figures**
- Tree cover against C6 for each year, with a further tagged copy of the 2026-10-02 figure script. Its C6 reader is fixed to `A2019065`, so the copy gets a `--year` option and reads the reconstructed C6.
- File names `<tile>_tree_cover_<tag>_<year>_vs_C6.png` go in a new folder; nothing is overwritten.
- **About 5 minutes per year.**

**Step 6. Report:** `Reports/Additional_Years_Results.md`, and the README is updated. **About 1.5 hours.**

## Set up for the pooled-model test (later, not in this plan)

- **Same held-out locations in every year.** The split is keyed on tile and pixel (x, y) only, not on year. So a pixel held out in 2019 is held out in every year. Without this, a pooled model could train on a location in one year and be tested on the same location in another, which would inflate its score. The step 3c acceptance test enforces it.
- **Same predictors, settings and seed in every year,** so a pooled model differs from the individual models only by its data.
- **Saved per year:** the per-tile tables (with the year in the `tid-year` key), the split files and the models. A pooled model is then a concatenation of the years' `xTrain`/`yTrain`, with no metric rebuild.
- **Watch for later:** a pooled tree RF on 6 years is about 55 M training rows. The 2019 tree RF (9.2 M rows) is already 17 GB on disk, so a pooled one could be about 100 GB, which won't fit in memory on these nodes. The later plan will need to subsample rows (to the size of one year, for a like-for-like test) or limit the leaf size.

## Ordering with the Monte Carlo rerun

If the Monte Carlo rerun (`Monte_Carlo_Rerun_Plan.md`) changes the predictor set, every year's training tables, models and products would need redoing.

- **Option A (suggested): run the years after the Monte Carlo decision.**
- **Option B: run now and keep each year's composites.** If the predictors change, rebuild the metrics from the composites (13-16 minutes per tile) and redo steps 3-4. That is about 6 to 8 hours for all 5 years.

## Time estimate

| Phase | Wall time |
|---|---|
| Preflight and gate | about 1.25 h |
| Metrics, 5 years, with each year's training, fits and products overlapping | about 10 to 12.5 h (2 to 2.5 h per year) |
| The last year's training, fits and products (not overlapped) | about 1.5 to 2 h |
| Model year × data year scoring, stability, summary across years | about 2 to 2.5 h |
| Report | about 1.5 h |
| **Total** | **about 16 to 20 hours**: two overnight runs, or one overnight run plus a day |
| Each further year (after downloads) | about 3 to 3.5 h |

## Disk and clean-up of intermediate files (clean-up accepted by the project lead, 2026-10-02)

Each tile-year has three intermediate folders under `Python/metrics/<year>/<tile>/<year>/` in the run root. The run's own scripts delete two of them, only inside the new run folder, and only after the step that needs them has passed its check. Sizes were measured on h17v00.

| Folder | Contents | Per tile-year | 5 years (315) | Action | When | Rebuild cost |
|---|---|---|---|---|---|---|
| `1-Days/` | Each band (1-7 and 31) for each 16-day date, decoded from the MOD44C HDFs into `.bin` files (136 files): a cache of the inputs | ~7.2 GB | ~2.3 TB | **Delete** | as soon as the tile's composites are built | re-read from MOD44C (part of the ~18 min composite stage) |
| `2-Composites/` | The 8 compositing periods per band, with QA and cloud flags (192 files); the input to every metric | ~6.8 GB | ~2.1 TB | **Keep** | - | ~18 min per tile from MOD44C |
| `3-Metrics/` | The 19 metric rasters the models use, `UnsortedMonthlyBands.tif` (3.3 GB) and `MOD44-NDVI.bin` (1.5 GB) | ~7.8 GB | ~2.5 TB | **Delete** | once that year's training tables are built **and** its products pass their checks | 13-16 min per tile from the kept composites |

**Never deleted:**
- the MOD44C inputs in `/css/...` and `/explore/.../MOD44C` (read-only);
- the labels;
- anything for 2019, and anything from earlier runs;
- each year's training tables and splits (~7.5 GB per year), models (~33 GB per year) and products (~9 GB per year).

**Why the composites are kept:** they are the slow part to rebuild, and every metric set can be rebuilt from them in 13-16 minutes per tile (against about 35 minutes from MOD44C). That is what a changed predictor set (Monte Carlo) or other metric families would need.

**Totals:**
- about **2.3 TB at the end** (composites ~2.1 TB, plus training, models and products ~250 GB);
- a peak of about **3 TB** during the run, because one year's metrics and the next year's daily files exist together;
- without the clean-up, about 6.9 TB.

Each deletion is logged to `status/cleanup_<year>.log` (folder, size, time). A deletion is skipped, and logged as such, if its condition is not met.

The filesystem shows 2.0 PB free; no per-user quota was visible. Check the project quota before starting.

## Caveats

- **The labels are static.** They come from imagery of one period, and are used with every year's metrics, as the C production did. Where land changed between the label imagery and year N (deforestation, regrowth), that year's training pairs a stale label with new spectra. The model year × data year table and the stability check will show how much this matters. A later option, not in this plan, is to drop label pixels whose C6 tree cover changes strongly across years.
- **C6 is not a clean reference.** The historical product used no working aerosol test, so it includes some high-aerosol pixels (0.1-16% depending on region and season). Differences against C6 in hazy regions are partly the reference.
- **2000:** the first Terra year, with early-mission data gaps. The share of fallback (all-bad-QA) composites per period is reported; 2000 is run last and read with care.
- **h16v01** gets the zenith-only NDVI fix in every year, and its polar-night dates are missing in every year, as in 2019.
- **Held-out scores are optimistic:** the held-out rows are neighbours of training pixels. This affects every year equally, so the comparisons between years and models still hold.
