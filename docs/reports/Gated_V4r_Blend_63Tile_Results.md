# Cloud-, cover- and snow-gated V4r blend on all 63 tiles (results)

> **FULLY EXPERIMENTAL.** Nothing here changes v1.0.0 or the delivery branch. All products are test products only; any adoption is a
> separate decision by the project lead.

> **The main configuration's tree cover is named v1.0.1 (experimental, not adopted; project lead, 2026-10-03).** See
> `Product_Versions_and_Work_Map.md`, section 1b, and `FullRuns/gated_v4r_63_20261003/manifests/VERSION_v1.0.1.md`.

**Run 2026-10-03, 10:37-12:19, ngaproc201/202, fully detached (`FullRuns/gated_v4r_63_20261003`). Plan:
`Gated_V4r_Blend_63Tile_Plan.md`.** All checks passed, and no task failed. The watchers saw no other users and no alerts.

**Names:**
- **V4R:** the model retrained on V4r metrics.
- **V1-V4C:** v1.0.0's model on V4r composites.
- **Main configuration** (project lead's choices):
  - snow test on the V4r pick: NDSI ≥ 0.4 and NIR ≥ 0.11;
  - cloud gate: heavy only (5+ snow-free, zenith-gated replaced periods), weight = count / 8;
  - cover gate: v1.0.0 ramp from 20 to 50.

Held-out labels (Landsat, early 2000s, sites chosen to avoid change) are judged by bin. The seed refit SEED1, put through the same
gates, is the control.

## Summary

1. **The gates work as designed.**
   - On the maps, the main configuration changes only cloudy pixels: 1% of all land, and about half of the pixels with 5+ cloud
     periods.
   - **Open land is protected:** C6 < 30 at 5+ periods gets +0.5 / +1.0, against +2.5 ungated. Held-out open labels (≤ 40) are not
     raised: −0.04 / −0.02.
   - **No seams.**
   - **The NDSI snow test reduces the loss on the snowy tile h10v04:** V4R −0.92 → −0.20, V1-V4C −0.84 → −0.15. The MOD44C snow bit
     helps less (−0.39).
2. **On the maps, dense cloudy cover is raised clearly.** At 5+ periods: C6 ≥ 60 +3.8 (V4R) / +4.3 (V1-V4C); C6 30-59 +4.3 / +5.7.
3. **At the independent labels, the raise in dense cover is small and not significant.**
   - **Labels ≥ 41 at 5+ periods** (9,928 pixels, 23 tiles):
     - all tiles: −0.02 (V4R) and +0.02 (V1-V4C);
     - without h10v04: +0.32 and +0.36, against a seed control of −0.10;
     - raised on 16 and 15 of 23 tiles.
   - **MAE changes are within noise** in every group.
   - **The gap at those pixels is 7-9 points** (label 61.3 against v1.0.0 52.0; 68.6 against 61.1 without h10v04). The main
     configuration closes about 5% of it.
4. **The weighted count does not do better.** That is the alternative the project lead asked to revisit if the hard count
   disappointed.
   - For V4R: dense labels at 5+ −0.29; pooled MAE −0.014. But the seed control gains −0.021, so that gain is again averaging.
   - For V1-V4C: dense −0.20; pooled MAE +0.013 (worse).
5. **Reading:** on this evidence a gated V4r blend is safe (open land, seams and snow are under control) but not effective at the
   independent labels. The large map raise in dense cloudy pixels (+4) is supported only weakly by the labels (+0.3, a fraction of the
   gap).

## Checks (all passed)

| Check | Result |
|---|---|
| Snow-aware counts, 63 tiles: rebuilt zenith-gated count = stored `CUR_Z` | 0 pixels differ on every tile |
| Rebuilt V4r pick: Band_1 = V4r composite Band_1 at every counted pixel-period | 0 differ on every tile |
| V4R and V1-V4C maps, 63 tiles each (126) | all produced; the 14 maps of the 7 earlier tiles are **pixel-identical** to the earlier runs |
| Gated blends, 63 tiles; held-out scoring; h10v04 snow check | done |

## 1. Held-out labels (63 tiles)

**Main configuration, change from v1.0.0** (95% bootstrap over tiles for MAE):

| Group (tiles v04+, 5+ cloud periods) | n | V4R blend: change / MAE / tiles raised | V1-V4C blend: change / MAE / tiles raised | Seed control: change / MAE |
|---|---|---|---|---|
| Labels ≥ 41 | 9,928 | −0.02 / +0.05 [−0.27, +0.22] / 16 of 23 | +0.03 / +0.21 [−0.00, +0.77] / 15 of 23 | −0.06 / +0.01 |
| Labels ≤ 40 | 19,370 | −0.04 / −0.03 / 12 of 27 | −0.02 / −0.01 / 13 of 27 | −0.03 / −0.03 |
| All held-out pixels | 2,289,990 | 0.00 / 0.000 | 0.00 / +0.001 | 0.00 / 0.000 |

**Per tile, labels ≥ 41 at 5+ periods** (change from v1.0.0, main configuration; tiles with ≥ 100 pixels):

| Tile | n | Label / v1.0.0 | V4R: no snow test → NDSI | V1-V4C: no snow test → NDSI | Seed |
|---|---|---|---|---|---|
| h10v04 (northern Rockies) | 6,595 | 57.7 / 47.4 | −0.92 → **−0.20** | −0.84 → **−0.15** | −0.05 |
| h19v09 | 1,054 | 59.9 / 53.3 | +0.44 | +1.71 | −0.24 |
| h12v09 | 470 | 81.4 / 72.5 | +0.26 | −1.26 | −0.14 |
| h11v09 | 465 | 79.8 / 77.9 | +0.77 | +0.37 | +0.06 |
| h09v04 | 441 | 62.6 / 50.8 | −0.21 → +0.16 | +0.16 → −0.02 | +0.03 |
| h26v06 | 223 | 84.7 / 84.0 | −0.06 | −0.66 | −0.20 |
| h08v05 | 200 | 50.2 / 38.7 | −0.29 → −0.32 | +0.29 → −0.23 | −0.26 |
| h11v08 | 156 | 84.3 / 67.0 | −0.36 | −1.20 | +0.26 |

The 15 smaller tiles (20-82 pixels each) are mostly raised, by +0.5 to +2.3.

**Snow tests on h10v04's dense labels (V4R blend):**

| Snow test | Change |
|---|---|
| None | −0.92 |
| MOD44C snow bit | −0.39 |
| NDSI (main) | −0.20 |

**Cover-gate variants** (dense ≥ 41 / open ≤ 40 at 5+, change from v1.0.0, V4R blend):

| Cover gate | Dense ≥ 41 | Open ≤ 40 |
|---|---|---|
| Ramp 20-50 (main) | −0.02 | −0.04 |
| Ramp on the higher of v1.0.0 and V1-V4C | +0.13 | +0.06 |
| Hard > 40 | −0.06 | −0.03 |
| None | +0.15 | +0.17 |

For the V1-V4C blend without a cover gate, open land is raised (+0.44, MAE +0.29, worse on 22 of 27 tiles). The cover gate is needed
for that partner.

## 2. Maps (63 tiles, 1.11 billion land pixels)

**Main configuration, pixels with 5+ cloud periods.** Mean difference from each reference, with v1.0.0's in brackets:

| C6 class (n) | Weighted share | Change, V4R / V1-V4C | Raised ≥ 5 points (V4R) | vs C6 | vs C6.1 | vs C code |
|---|---|---|---|---|---|---|
| < 30 (12.4 M) | 20% | +0.45 / +0.97 | 7% | +5.1 (+4.6) | +5.2 (+4.8) | +2.9 (+2.5) |
| 30-59 (5.4 M) | 79% | +4.29 / +5.69 | 42% | +11.3 (+7.0) | +16.1 (+11.8) | +13.5 (+9.2) |
| ≥ 60 (6.2 M) | 99% | +3.82 / +4.33 | 39% | +2.3 (−1.5) | +7.6 (+3.8) | +10.6 (+6.8) |

- **Unchanged:** pixels with fewer than 5 cloud periods, and all pixels on the 63 tiles outside the gates. Over all land the mean
  change is +0.05.
- **Ungated** (no snow, no cover gate): C6 < 30 at 5+ +2.5, 30-59 +5.9, ≥ 60 +4.0.
- **Seams:** at neighbouring pixel pairs whose weights differ (15.5 M pairs), mean |difference| is 9.9 (V4R blend) and 10.2 (V1-V4C
  blend), against 10.9 for v1.0.0.
- **Against the references** (all cloud-affected): in C6 ≥ 60 cloudy pixels the blend moves from 1.5 below C6 to 2.3 above it, and
  further above C6.1 and the C code. In C6 30-59 cloudy pixels v1.0.0 is already 7-12 above all three, and the blend adds 4 more.

## 3. h10v04 snow check

From `Snow_vs_Cloud`, h10v04:
- **Winter V4r triggers** cover 15-47% of land in November-February.
- **Low sun** (every input at zenith ≥ 65) explains 34% of the December triggers and 8% of November's.
- **The snow bit** removes 20-30% of the triggers.
- **The NDSI test on the picks** marks about 20-24% of the winter replaced periods as snow (self-test).
- **Much of the winter firing on this forested mountain tile remains** after both tests. That is consistent with snow under or on a
  forest canopy, which reads with lower NDSI, or with real winter cloud.

## 4. The h12v09 drop pixels (added 2026-10-03)

Files: `code/drop_pixels.py`, `out/drop/`.
- **Drop pixels:** the 103,011 dense h12v09 pixels from the earlier sessions (`stable_pixels_20260930/drop_diag/drop_mask_h12v09.npy`).
- **Control:** 200,000 dense pixels where the v0.0.0 baseline is within 3 points of C6, chosen at random (seed 0), as in the
  cloudy-pixel runs.

Mean difference from each reference (MAE in brackets):

| Set | Product | Mean | Change from v1.0.0 | vs C6 | vs C6.1 | vs C code |
|---|---|---|---|---|---|---|
| Drop (103,011) | v1.0.0 | 53.2 | | −18.9 (19.4) | −7.7 (12.1) | −0.3 (9.7) |
| | V4R, ungated | 69.4 | +16.2 | −2.7 (7.6) | +8.5 (13.4) | +15.9 (18.2) |
| | V1-V4C, ungated | 72.0 | +18.7 | −0.1 (6.9) | +11.1 (14.1) | +18.4 (20.2) |
| | **V4R blend, gated (main)** | 61.8 | **+8.6** | −10.3 (11.5) | +0.9 (10.7) | +8.2 (12.5) |
| | **V1-V4C blend, gated (main)** | 62.9 | **+9.7** | −9.2 (10.5) | +2.0 (10.6) | +9.3 (13.2) |
| Control (200,000) | v1.0.0 | 79.1 | | +2.4 (3.3) | +2.7 (3.6) | +4.9 (5.5) |
| | gated blends (main) | 79.3 | +0.2 | +2.7 (3.4) | +2.9 (3.7) | +5.1 (5.7) |

**How the gates treat the drop pixels:**
- **Cloud gate:** 75.6% pass it (5+ snow-free periods). The snow test removes almost nothing here.
- **Cover gate:** 62% get full cover weight (v1.0.0 ≥ 50), and 37% are on the ramp (v1.0.0 20-50). Only 0.5% have v1.0.0 below 20.
- **Overall:** the mean blend weight is 0.46. The count / 8 weight gives 5-6 periods only 0.63-0.75.

**Reading:**
- The gated blend closes about half of the drop against C6 (−18.9 → −9 to −10).
- It moves the drop pixels from below C6.1 to just above it, with MAE improving from 12.1 to 10.6-10.7.
- It moves them away from the C code, which agreed with v1.0.0 there (MAE 9.7 → 12.5-13.2).
- **Clean dense forest (the control) is unchanged** (+0.2).
- So two of the three references improve, and the C code, which shares v1.0.0's cloud-contaminated inputs, gets worse.
- The ungated V4r products close the gap to C6 almost fully, but overshoot C6.1 and the C code.
- **If a fuller correction in heavy-cloud pixels were wanted,** a weight of 1 at 5+ periods (instead of count / 8) would move the
  gated blend toward the ungated values on the pixels that pass the gates. That has not been tested on the held-out labels.

## Points for the project lead

1. **The gating design is sound,** and any future cloud correction could reuse it. A cover ramp on v1.0.0 protects open land, a pick
   NDSI test reduces snow damage, and no seams appear.
2. **The correction it gates, a V4r-based raise, does little where independent labels can judge it:** +0.3 against a 7-9 point gap,
   not significant, raised on two thirds of tiles. The maps raise dense cloudy pixels by about +4. Whether that is right cannot be
   settled with these references: the labels support it only weakly, and C6, C6.1 and the C code are all cloud-affected.
3. **The weighted count was revisited, as asked.** It is not better than the hard count.
4. **Not addressed here:**
   - why V4r lowers some dense tropical label pixels (h11v08, h12v09 for V1-V4C);
   - the remaining winter firing on h10v04;
   - non-vegetated.

## Caveats

- Only 9,928 dense held-out pixels at 5+ periods, two thirds on h10v04. Several tiles have 20-80 pixels.
- Held-out pixels are neighbours of training pixels.
- Label age (early 2000s) is a minor caveat at these sites, but it remains one.

## Where the results are

All under `FullRuns/gated_v4r_63_20261003/out/`:
- `counts_snow/` (63 tiles);
- `maps/V4R`, `maps/V1-V4C` (63 tiles each), `maps/V4R_MAIN`, `maps/V1-V4C_MAIN` (main-configuration blends);
- `gated/<tile>_stats.csv` and `_seams.csv`;
- `heldout/` (`bins.csv`, `diffs.csv`, `per_tile.csv`);
- `snowcheck/`;
- `map_check.txt`.
