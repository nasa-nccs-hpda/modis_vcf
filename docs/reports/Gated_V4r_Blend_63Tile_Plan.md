# Cloud-, cover- and snow-gated V4r blend on all 63 tiles (plan)

> **FULLY EXPERIMENTAL.** Nothing here changes v1.0.0 or the delivery branch. All products are test products only; any adoption is a
> separate decision by the project lead.

**Written 2026-10-03. Approved by the project lead the same day and run 10:37-12:19; results in `Gated_V4r_Blend_63Tile_Results.md`.** Builds on
`V4r_Cloud_Blend_Results.md` (read its correction section first), `Snow_vs_Cloud_V4r_Trigger_Check.md` and
`QA_Bits_Dense_vs_Open_Check.md`.

## Question

Can a blend of v1.0.0 with a V4r-based prediction raise the dense pixels that v1.0.0 under-predicts under heavy cloud, without
touching open land or snowy tiles, across all 63 tiles?

## Names

- **v1.0.0:** the delivered tree model and its maps.
- **V4R:** the model retrained on V4r metrics, applied to V4r composites.
- **V1-V4C:** v1.0.0's model applied to V4r composites (one model).
- **V4r composites:** the existing ones from `cloudy_v4r_weights_20261002`, unchanged.

## Gates (project lead's choices, 2026-10-03; main setting first, the others reported for comparison)

Weight w = cloud weight × cover weight; value = floor((1 − w) × v1.0.0 + w × partner + 0.5). Land pixels only; water and
out-of-projection codes are kept.

1. **Snow (inside the cloud count).** A replaced period counts only if the V4r pick is **not** snow.
   - **Main rule:** NDSI = (B4 − B6) / (B4 + B6) ≥ 0.4 **and** NIR (Band 2) ≥ 0.11 on the pick (the standard MOD10 thresholds) means
     snow.
   - Also reported: the MOD44C snow bit (5) of the picked input, and no snow test.
2. **Cloud.** The count is the zenith-gated, snow-free replaced periods (0-8).
   - **Main rule:** heavy only, w_cloud = count / 8 where the count is ≥ 5, otherwise 0.
   - Also reported: count / 8 at all counts.
   - **Project lead, 2026-10-03:** if the results with the hard count (5+) are unsatisfactory, revisit the weighted count (count / 8
     at all counts) as the main setting.
3. **Cover** (v1.0.0's first pass).
   - **Main rule:** ramp, 0 at v1.0.0 = 20 rising to 1 at 50.
   - Also reported:
     - the same ramp on the higher of v1.0.0 and V1-V4C;
     - a hard gate at > 40;
     - no cover gate.

**Partners:** V4R and V1-V4C, each with every gate combination (2 partners × 3 snow × 2 cloud × 4 cover = 48 variants).

## Steps (run root `FullRuns/gated_v4r_63_<date>/`, fully detached, ngaproc201/202)

1. **Snow-aware counts, 63 tiles** (about 15 min).
   - For each period: the zenith-gated V4r trigger, rebuilt exactly as `counts_zen.py`; the pick, rebuilt as `v4r_compose.py`; its
     NDSI and NIR from the V4r composite; its snow bit from its MOD44CQ state.
   - **Checks:** the rebuilt zenith-gated count equals the stored `CUR_Z` on every pixel, and the rebuilt pick's Band_1 equals the V4r
     composite's Band_1 at every replaced pixel.
2. **Maps of V4R and V1-V4C, all 63 tiles** (about 2 h; 3 at a time per node). Same predictor as before (`map_predict.py`: MEDIAN,
   rounded).
   - **Check:** the 7 tiles mapped earlier must reproduce exactly.
3. **Gated blends, 63 tiles** (post-processing).
   - Statistics for all 48 variants:
     - by C6 class (< 30 / 30-59 / ≥ 60) × cloud group, all land, by continent and latitude band, and per tile;
     - the mean change from v1.0.0 and the mean difference from C6, C6.1 and the C code (direction only: all three are
       cloud-affected).
   - Seams at weight edges.
   - GeoTIFF maps are written for the main configuration only (both partners).
4. **Held-out labels, 63 tiles** (minutes). All 48 variants plus a seed-refit control for the main configuration. Scored:
   - by label bin (same / lower / higher), by cloud group and per tile;
   - especially dense labels (≥ 41) at 5+ periods, and **h10v04 with and without the snow test**;
   - pooled MAE differences with tile bootstrap.
5. **Snow check on h10v04** (`Snow_vs_Cloud` method plus the NDSI share of its picks), to confirm or reject the snow explanation for
   its decrease.
6. **Report:** `Gated_V4r_Blend_63Tile_Results.md`, and the README and work map are updated.

## Reading the results (the project lead decides)

The main configuration is worth taking further if, on the held-out labels:
- dense labels (41-60, 61+) at 5+ cloud periods move up into the right bin on most tiles, beyond the seed control;
- open labels (0-20, 21-40) do not move up;
- h10v04 no longer loses with the snow test.

And, on the maps:
- the raise sits in C6 ≥ 60 and C6 30-59 dense pixels, not in C6 < 30;
- no seams appear.

## Not in this plan

Non-vegetated, the SD layer, other years, and changing the V4r composites themselves.

**Total:** about 2.5-3 h, mostly maps.
