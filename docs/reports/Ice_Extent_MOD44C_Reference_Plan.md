# An ice-extent reference from MOD44C (NDSI + thermal), to score non-vegetated on the ice: plan

**Status: APPROVED 2026-10-06 (I1-I4 as suggested), then REVISED the same morning (project lead chose option (b)); running as
`FullRuns/EXP_ice_extent_mod44c_20261006` on forest205 from 08:35.**

> **Revision after the self-test (2026-10-06, option (b)).** On h17v01 the NDSI test failed on the ice-sheet interior: 20,000 pixels more
> than 50 km inside the MCD12 ice have a median NDSI of only 0.27-0.34 (clean snow is about 0.8-0.9), band 6 about 0.35 (ice should be
> < 0.1), green below red, and in late August half of them have red reflectance near 0.03. These composites are fallback means of low-sun
> inputs (no input passes QA on the interior ice), so their reflectances are unreliable; band 31, emitted, is not affected (median 251-255 K,
> all 4 periods pass). **New test:** ice = band 31 <= 275 K in >= 3 of 4 periods **and** red (B1) >= 0.40 in >= 2 of 4 periods, and **not
> MOD44W water** (the pipeline's water_mask.bin, which equals the maps' water exactly). NDSI >= 0.4 is kept only as a diagnostic.
> Sensitivity: red 0.30 / 0.50, thermal 273 / 277 K, 2 of 4 thermal periods, 3 of 4 bright periods. The sea-water B2 check is reported only.
> Open question for later (upstream, not chased now): band 6 at about 0.35 on ice is extreme enough to confirm on a non-polar snow tile.
Follows: `Ice_Sheet_Diagnosis_Results.md` (2026-10-05) and `Band31_Unclamped_Plan.md`.

> **For evaluation only.** The mask defines where ice is so that we can score the product against it. It is **not** a rule, mask or
> override for the product: the ice-sheet fix stays data-driven (inputs, composites, training rows, models).

## 1. The question

The ice-sheet counts so far use MCD12Q1 2019 class 15 (largest connected region per tile) to say where the ice is. The diagnosis found that
part of it is probably not ice: the margin pixels where the model gives < 90 non-vegetated are, in June, dark (blue reflectance about 850
against about 7,000 on clean ice), warm (band 31 about 28,800 against 25,000-25,300) and slightly green. If MCD12 puts its edge too far out,
part of the "bare < 100 on the ice" we count is the model being right.

**Question:** with an ice extent built from our own inputs (MOD44C), how much of the non-vegetated layer is really wrong on the ice? And is
what is left the interior problem (h16v02), the margin, or both?

## 2. Data (already on disk; nothing new to fetch)

- **MOD44C 16-day composites, all bands, band 31 unclamped:** `FullRuns/EXP_b31_unclamped_20261005/Python/metrics_new/<tile>/2019/1-Days/`
  (`MOD44-<tile>-<period>-Band_<b>.bin`, int16, 4800 x 4800, 23 periods). Band 31 here is the real value (the clamp removed), which the thermal
  test needs.
- **Tiles (7):** Greenland h15v02, h16v00, h16v01, h16v02, h17v00, h17v01, and Iceland h17v02 (ice caps; a second, independent case).
- **MCD12Q1 2019 class 15:** read with `deepdive_vs_historical.read_lc` (`regtree_alt_models_20260927/code/validation`), largest connected
  region, exactly as in `ice_counts.py`. A reference, not the boundary.
- **Maps scored:** v1.1.0 (`cand60med_20261004/out/maps/CAND`), b31 unclamped (`EXP_b31_unclamped_20261005/out/maps/NEW`), and seed 1
  (`EXP_b31_seedmaps_20261006/out/maps/SEED1`) if that run has finished.

## 3. The test (per pixel)

**Periods:** late summer, 2019177, 2019193, 2019209, 2019225 (late June to late August). By then seasonal snow off the ice has mostly
melted, so persistent snow/ice signal means permanent ice.

**Per period:**
- **Snow/ice:** NDSI = (B4 − B6) / (B4 + B6) ≥ 0.4, **and** B2 (NIR) ≥ 0.11, **and** B4 (green) ≥ 0.10 (reflectance; stored × 10,000). These are the
  MOD10 snow thresholds; the B2 test keeps water out (water also has positive NDSI). Band 6, not band 5 (Terra band 5 has bad detectors).
- **Thermal:** band 31 ≤ 275 K. Melting ice cannot be warmer than 273.15 K; 2 K allows for mixed pixels. (Step 0 confirms the band-31 scale.)
- A period with a fill or missing value in any of these bands does not count.

**Classes:**
- **Ice:** both tests pass in at least 3 of the 4 periods (one cloudy period allowed; cloud over ice fails NDSI because cloud is bright in B6).
- **Ambiguous:** one test passes in ≥ 3 periods and the other does not, or fewer than 3 usable periods. Reported separately, never counted as ice.
- **Not ice:** the rest.
- The **largest connected ice region** per tile is the ice sheet (or ice cap), as for MCD12; the smaller ones are reported as other ice.

## 4. Steps

Run `FullRuns/EXP_ice_extent_mod44c_20261006/` (EXPERIMENTAL.md), detached, one forest node, read-only on its inputs. About 15-20 minutes.

**Step 0, checks (minutes, fatal):**
- **Band-31 scale.** The stored value is taken as 0.01 K (so 27,500 = 275 K). Check that this gives physical temperatures: open ocean next to
  Greenland in August should be about 271-280 K, and the clean interior ice in June about 240-265 K. If not, stop and report.
- **Composites match.** For one tile and period, bands 4 and 6 in `1-Days` equal the clamped run's `greenland_zenith_v2_20261001` composites
  (only band 31 should differ).
- **Orientation.** The MOD44C arrays, MCD12 and the maps align (row, column), checked on a coast: water in the map is water in MCD12 and
  fails the B2 test.

**Step 1, the mask:** per tile, the class map, the per-period pass counts, and these diagnostics:
- pixels that pass NDSI and thermal but fail the B2 ≥ 0.11 test (the dark bare-ice melt zone in west Greenland could fall here);
- pixels with fewer than 3 usable periods (h16v02's composites are all fallback means of low-sun inputs, solz about 86°).

**Step 2, extent against MCD12** (pixels and km², per tile):
- both ice | MCD12 only (we expect the margin ring) | MOD44C only | ambiguous;
- distance of the MCD12-only pixels to the MOD44C ice edge (median, 90th percentile).

**Step 3, non-vegetated and tree on the ice,** for each map (v1.1.0, b31, seed 1):

| Zone | Reported |
|---|---|
| MOD44C ice sheet (interior: > 2 km from its edge) | share bare = 100, < 100, < 90; mean bare; share tree > 0 |
| MOD44C ice sheet (edge band: ≤ 2 km) | the same |
| MCD12 only (MCD12 says ice, MOD44C does not) | the same, plus mean NDVI and band 31 in July |
| MOD44C only | the same |

The interior row is the cleanest test of "are we getting bare right on the ice". The MCD12-only row shows how much of the earlier count
was the reference being wrong.

**Step 4, sensitivity:** repeat steps 2-3 (counts only) with NDSI 0.3 and 0.5, thermal 273 K and 277 K, and "2 of 4" periods instead of 3.
The conclusions stand only if they hold across these.

**Step 5, figures** (per tile, MODIS data only):
- the non-vegetated map (v1.1.0) with the MCD12 ice edge and the MOD44C ice edge drawn over it;
- the class map (both / MCD12 only / MOD44C only / ambiguous);
- the number of periods passing each test.

**Step 6, write-up:** `Reports/Ice_Extent_MOD44C_Reference_Results.md`, and the ice counts in `Band31_Unclamped_Results.md` reported against
both references.

## 5. Caveats (stated in the results)

- **h16v02's low sun.** No input passes QA there in any period; its composites are means of low-sun inputs. NDSI is a ratio and fairly robust to
  illumination, but the reflectances are noisier; its "ambiguous" count is reported on its own.
- **Dark ice.** Bare ice in the west Greenland melt zone can be darker than 0.11 in B2; step 1 counts how many pixels that test removes.
- **Mixed pixels at 250 m** make the edge a band, not a line; hence the 2 km edge zone in step 3.
- **One year (2019)**, as the product.

## 6. Decisions

| # | Question | Options | Suggested |
|---|---|---|---|
| I1 | Thresholds | (a) as above: NDSI ≥ 0.4, B2 ≥ 0.11, B4 ≥ 0.10, B31 ≤ 275 K, 3 of 4 late-summer periods (b) other values | **(a)**, with the sensitivity in step 4 |
| I2 | Tiles | (a) the 6 Greenland tiles + h17v02 (Iceland) (b) Greenland only | **(a)** |
| I3 | Maps | (a) v1.1.0, b31, seed 1 if ready (b) v1.1.0 only | **(a)** |
| I4 | Node | (a) one forest node, after the seed-map lanes there finish (b) start at once beside them | **(a)** if the seed maps finish within the hour; else (b) on forest205, whose map lane uses one slot (about 42 GB of 112) |
