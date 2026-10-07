# v1.2.0 build: results

**Status 2026-10-07: done. All 69 tiles pass every check. v1.2.0 is built.** Run `FullRuns/v120_20261007`, 07:44-09:29 (1 h 45 min) on ngaproc201/202
and forest202-205, with the Greenland/Iceland tiles first. Plan: `Version_v1.2.0_Build_Plan.md` (approved, V1-V6 as suggested). Recipe:
`v120_20261007/manifests/VERSION_v1.2.0.md`.

## What v1.2.0 is

v1.2.0 is v1.1.0 with one change: the non-vegetated model adds 10,000 bare training rows labelled 100 on h16v02's MOD44C ice (A10S0 from
`EXP_ice_rows_h16v02_20261006`; evidence in `Ice_Training_Rows_h16v02_Results.md`). Everything else is unchanged:
- **Tree model:** T60.
- **Inputs:** band 31 clamped, bit-15 composites, zenith 65.
- **Rules:** the Greenland zenith-only NDVI rule and the v00 rule.
- **Code:** delivery tag `v1.1.0` = `920d06f`.

v1.2.0 is the first full map set written by that code: uint8 percent layers (nodata 253), uint16 SD × 100, and the MODIS sinusoidal grid.

## Checks (all passed)

| Check | Result |
|---|---|
| Gate: h16v02 and h12v09 before the other tiles | pass |
| Format: Byte / UInt16, nodata 253, MODIS grid geotransform and sphere, every layer and tile | 69/69 |
| Tree cover and tree SD values equal v1.1.0 | 69/69 (0 pixels differ) |
| Non-vegetated and non-tree vegetation equal the EXP A10S0 maps (15 tiles) | 15/15 (0 pixels differ) |
| Water / out-of-projection / fill pixels unchanged from v1.1.0 | 69/69 |
| MOD44C ice-zone scores of the 7 Greenland/Iceland tiles reproduce the A10S0 column | 42/42 rows equal |

## How much non-vegetated changed (v1.2.0 against v1.1.0)

`out/summary/bare_change_vs_seed.csv` compares the change with a pure seed change (seed 1 against v1.1.0, `EXP_b31_seedmaps_20261006`) on all
69 tiles.
- **The change is the size of a refit.**
  - At the median tile, 20.7% of land pixels move by more than 2 points; a seed change moves 21.2%.
  - The ratio of the two is 0.94-1.10 on 64 of 67 tiles (0.95-1.06 on 60). h16v00 and h17v00 are excluded: the v00 rule sets them.
  - The median absolute mean shift is 0.075 for v1.2.0 and 0.083 for seed 1.
  - A random forest refit moves individual pixels this much on its own; held-out labels and zone counts are the measures that matter.
- **Exceptions, as expected:**
  - **h16v02:** 1.31 times the seed change, mean +0.33. This is the intended ice fix: interior ice below 100 goes from 8.7% to 0.007%.
  - **h16v01 and h17v01:** 0.7 times the seed change. They are steadier than a refit.
- **Largest mean shifts:**
  - h31v11 +1.25 (seed 1 alone −0.60), h12v12 +0.71, h19v11 −0.66.
  - Seed 1 alone moves tiles by up to ±0.74 (h19v10 +0.74, h22v03 −0.70), so these are at or slightly beyond seed range.
  - None has held-out labels in the 15-tile EXP set. Noted, not a blocker.

## The sum cap in v1.2.0

v1.2.0 keeps the C code's first sum cap (rule C2): on land, bare = min(bare_raw, 100 − tree). Tree cover is never lowered, and non-tree
vegetation = 100 − tree − bare. Water and out-of-projection pixels are skipped. The SD layers are computed before the cap.

`out/summary/cap_count.csv` counts where the cap could have fired. The raw bare is not stored, so the count is an upper bound: land pixels
with tree > 0 and non-tree vegetation 0.

| Tree on capped pixels | Pixels | Note |
|---|---|---|
| all | 170,160 (0.015% of land; 66 tiles) | |
| 1-5 | 48,798 | bare 95-99: bare near or at 100, pulled down by a little tree cover |
| 6-20 | 41,021 | |
| > 20 | 80,341 | overshoot in vegetated areas (h11v02, h26v06, h11v03) |

The bare-near-100 cases are concentrated on h21v06 (9,656), h12v12 (9,030), h17v02 Iceland (7,591), h13v12 (4,163) and h16v01
(3,913).

## v00 tiles without the v00 rule (`FullRuns/EXP_v00_norule_20261007`, EXPERIMENTAL)

The project lead asked whether v1.2.0's models still need the v00 rule. The test was v1.2.0 exactly with the rule switched off, on h16v00 and
h17v00.

| | h16v00 | h17v00 |
|---|---|---|
| Bare mean: v1.0.0 era without the rule → v1.2.0 without the rule | 73 → **92.4** | 79 → **94.6** |
| Bare MAE against C6: zenith fix only → v1.2.0 without the rule | 12.6 → **7.6** | 10.2 → **5.4** |
| Tree mean (C6 and C6.1 are 0) | 2.4 | 2.1 |
| Land with tree exactly 0 | 9.6% | 16.5% |
| MOD44C interior ice: bare below 100 / tree above 0 | 82% / 82% | 71% / 71% |

- **On the ice the bare model predicts about 100. The tree model adds 1-2% spurious tree cover, and the sum cap lowers bare to 98-99.**
  - The share with bare below 100 equals the share with tree above 0.
  - Interior-ice bare averages 98.4-98.5.
- In the cells that MCD12 calls ice and MOD44C does not, bare is lower (88-94) and tree is above 0 on 96%. This fits the polar-night
  inputs: two of eight periods have no data.
- **Decision (project lead, 2026-10-07): keep the v00 rule in v1.2.0.**
  - The data-driven path to retire it is the tree-side experiment in `Tree_Ice_Rows_and_Bare_Priority_Plan.md`.
  - That experiment has two parts: ice tree rows (label 0), and a variant that trusts bare = 100 over small tree cover.

## Added label file (decision V6)

`v120_20261007/Python/training_labels_add/h16v02.bare.ice_rows_A10.out` (README beside it):
- Legacy form: 4800 × 4800 uint8, row-major, 255 = no sample.
- 10,000 cells set to 100, the same rows as `Python/split_A10`.
- Orientation was checked against h16v01's labels: 100% match row-major, 33% transposed.
- No NGA-derived values.

## Files

- **Maps:** `v120_20261007/out/maps/v1.2.0` (69 tiles, 5 layers, sha256 in `manifests/v1.2.0_maps_sha256.txt`).
- **Checks:**
  - `out/summary/checks_per_tile.csv`
  - `out/summary/ice_vs_A10S0.csv`
  - `out/summary/bare_change_vs_seed.csv`
  - `out/summary/cap_count.csv`
- **Model and split:** `Python/models/A10S0`, `Python/split_A10` (sha256 identical to the EXP run).
- **Code:** `code/modis_vcf` = delivery clone `920d06f`. `code/check_v120.py`, `make_label_file.py`, `summarize_v120.py`, `seed_compare.py`,
  `cap_count.py`.

## Next

1. Delivery clone:
   - Branch `mcarrol2/v1.2.0` from `mcarrol2/v1.1.0`.
   - One docs commit (the README gives v1.2.0).
   - Local annotated tag `v1.2.0`.
   - The project lead pushes, then opens the PR into `develop` from v1.2.0 (V5).
2. The tree-side experiment plan (to be run later).
3. F13 figures (`AICoE_10142026/`).
