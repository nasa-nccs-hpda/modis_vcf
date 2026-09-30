# Stable-pixel reference set from the 2000-2020 historical MOD44B (2026-09-30)

**Purpose.** There are no independent tree cover labels, so this builds a reference from the historical MOD44B itself: pixels in
training tiles whose tree cover barely changed over 21 years, that are not training labels, and not next to them. The B and C
models will be scored against these pixels (Stage 6 substitute for `Aerosol_QA_C_Full63_Retrain_Plan.md`).

**Status (2026-09-30 ~08:15): option (a) approved and done. `stable_pixels_subbin_10000.csv` is the current set.** The first
set (`stable_pixels_10000.csv`, fixed SD <= 3) is kept for reference only. h19v09 mid-range pixels kept as they are (project lead, 2026-09-30).

## Data
- Historical MOD44B **C6 (.006)**, `Percent_Tree_Cover`, **2000-2020 (21 years)**, from `/css/modis/Collection6/L3/MOD44B-VCF_reconstructed/`.
  The standard C6 folder lacks 2003-2006. The project lead rebuilt the reconstructed files from an external source, so their
  timestamps are all 2017. Checked: tree cover is identical to the standard C6 files (h11v04, h19v08; 2000 and 2019; 100% of pixels).
  C6.1 is not used (project lead's instruction).
- Training labels: `Python_code/modis_vcf/data/legacy/<tile>.bin` (pcttree) and `<tile>.bare.4.out` (pctbare), 4800x4800 uint8,
  255 = no label, on the same grid as MOD44B (r(label, C6 2019) = 0.78-0.91 on the tiles used; 0.62 on h12v04, which was left out).
- Caveat: the historical product used PGE61 with no working aerosol test (see the aerosol-bit caveat), so this reference is a
  "no screen" product. It is stable over time, but it is not independent of the historical algorithm.

## Method
1. Pixel valid (0-100) in all 21 years.
2. Not a pcttree or pctbare label pixel, and not within 2 pixels of one (so >= 3 pixels, about 700 m, from any label).
3. Stable: population SD over 21 years <= 3 percentage points.
4. Bin by the 21-year mean: 0-10, 10-30, 30-60, 60-100; 2,500 pixels per bin, split as evenly as possible across tiles.
5. Spacing: row and col 1-3 mod 5, one pixel per 5x5 block, so selected pixels are >= 3 pixels apart. Random, seed 0.

Pools were built for h11v04, h12v04, h12v09, h18v03, h19v08, h19v09 (`logs/pool_*.log` has counts at SD <= 1, 2, 3, 5).
SD <= 2 leaves almost nothing in the middle bins (e.g. 397 pixels at 30-60 on h11v04), so SD <= 3 was used.
Tiles used: **h11v04** (eastern US), **h18v03** (Europe), **h19v09** (central Africa).

| Bin | h11v04 | h18v03 | h19v09 | Median 21-yr mean |
|---|---|---|---|---|
| 0-10 | 834 | 833 | 833 | 5.5 |
| 10-30 | 834 | 833 | 833 | 11.1 |
| 30-60 | 1233 | 1233 | 34 (all it had) | 57.9 |
| 60-100 | 834 | 833 | 833 | 64.5 |

## Problem
A fixed SD cutoff selects mostly the ends of the bins. MOD44B varies more from year to year at mid-range cover, so of the 10,000
pixels only 246 have a mean of 15-50. 10-30 is in practice 10-15, and 30-60 is in practice 50-60. A score on this set would say
little about the 15-50 range. Options:
- **(a) Stratify in 5-point sub-bins with a relative stability rule** (e.g. the most stable 10% of each sub-bin, or SD <= 3 or
  SD/mean <= 0.1, whichever is looser), so the whole 0-100 range is covered evenly. Recommended.
- (b) Keep SD <= 3 but loosen it to SD <= 5 for 15-50 only.
- (c) Use the set as is, and report results for the 15-50 range as not covered.

## Files
`FullRuns/stable_pixels_20260930/`: `code/stable_pool.py`, `code/select_stable.py`, `pools/<tile>.npz`,
`stable_pixels_10000.csv` (tile, row, col, bin, lat, lon, 21-yr mean, SD, SD 2000-2019, min, max, 2019 value), `logs/`.
`pools_17yr_superseded/` is the first try on the 17 standard C6 years; it is not used.

## Option (a) result (current set): `stable_pixels_subbin_10000.csv`
`code/select_stable_subbin.py`, log `logs/select_subbin.log`. The same three tiles, pools, label buffer and spacing as above.
5-point sub-bins on the 21-year mean. Stable within a (tile, sub-bin) means SD <= max(3, the 10th percentile of SD in that
tile's sub-bin). 588-589 pixels in each of the 17 sub-bins from 0 to 85. These three tiles have no historical MOD44B mean above 85.
Pixels per tile: h11v04 2989, h18v03 2947, h19v09 4064 (h19v09 supplies almost all of 75-85).

SD cutoffs used (h11v04 / h18v03 / h19v09):

| Sub-bin | 0-10 | 10-15 | 15-20 | 20-30 | 30-50 | 50-60 | 60-70 | 70-75 | 75-85 |
|---|---|---|---|---|---|---|---|---|---|
| cutoff | 3/3/3 | 3.4/4.0/3.5 | 4.1/4.9/4.6 | 4.7-5.0/5.7-6.2/5.7-6.9 | 4.2-4.9/5.2-6.2/7.6-8.2 | 3.5-3.9/3.5-4.4/8.2-8.5 | 3.2-3.4/3.1-3.5/7.1-8.2 | 3.0/3.3/5.5 | about 3 |

By coarse bin: 0-10 n=1178 (mean SD 2.1), 10-30 n=2354 (4.4), 30-60 n=3528 (5.1), 60-100 n=2940 (3.4).

**Open point:** in the savanna tile h19v09 even the most stable 10% of mid-range pixels vary by an SD of 7-8.5 points over
21 years, compared with 3-6 on h11v04 and h18v03. Those pixels are "stable relative to their surroundings", but their
reference value is less certain. Options: keep them (the error at those pixels will be noisier), or cap the cutoff (for example
SD <= 6) and let h11v04 and h18v03 fill the shortfall in 20-70. The set is usable as is; results will also be reported per tile.

**Decision (2026-09-30):** keep the set as it is. The h19v09 mid-range pixels stay in, accepted as genuinely noisy. Results will also be
reported per tile so that h19v09 can be read separately.
