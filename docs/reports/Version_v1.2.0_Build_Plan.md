# v1.2.0 build: plan

**Status: APPROVED 2026-10-07 by the project lead, V1-V6 as suggested.** Name v1.2.0 chosen by the project lead on 2026-10-07.

## 1. What v1.2.0 is

v1.2.0 is v1.1.0 with one change: **the non-vegetated model is A10S0** from `EXP_ice_rows_h16v02_20261006`. That model is the v1.1.0 bare split
plus 10,000 ice rows on h16v02, all labelled 100, fit with the v1.1.0 recipe (RF, 60 trees, sqrt, seed 0, MEDIAN).

Evidence and decisions are in `Ice_Training_Rows_h16v02_Results.md` (A10 kept and h12v02 accepted as seed noise, both on 2026-10-07).

| Part | v1.1.0 | v1.2.0 |
|---|---|---|
| Tree cover model | `trees_counts_2x2_20261003/Python/models/T60` | same |
| Non-vegetated model | `bare60_20261004/Python/models/B60S0` | `EXP_ice_rows_h16v02_20261006/Python/models/A10S0` (copied, not refit) |
| Bare split | `nonveg_bit15_models_20261001/Python/training_split` | that split + 10,000 rows (`.../Python/split_A10`) |
| Composites, QA, band 31 | bit 15, zenith 65, band 31 **clamped** | same (band 31 unclamped is **not** used) |
| Code and flags | delivery clone at tag `v1.1.0` (`920d06f`), `--tcAgg median --nvAgg median`, SD on, uint8/uint16 on the MODIS grid | same code (no code change) |

The model is copied into a product folder, so the product does not depend on an `EXP_` run directory.

**Output format (checked 2026-10-07):** the v1.1.0 maps on disk (`v110_sd_20261004`, `cand60med_20261004`) and the EXP run's maps are
**Int16 with no geotransform**. They were made with code older than `0298e4e` (uint8/uint16) and `920d06f` (MODIS grid). This build runs the
delivery clone at the tag `v1.1.0` = `920d06f`, so v1.2.0 is the first full map set in uint8/uint16 with a geotransform. That supersedes
open decision A4 for the current product. The two commits change only the writer (checked on h08v04: uint8 values equal the v1.1.0 maps).

## 2. Steps

The run goes in `FullRuns/v120_20261007/`. It is a product run, so the folder has no `EXP_` prefix. It runs detached, with the gate first.

0. **Copy and freeze:** copy A10S0 and split_A10 into `v120_20261007/Python/` and record sha256 against the EXP run. Write
   `manifests/VERSION_v1.2.0.md` in the same form as `VERSION_v1.1.0.md`.
1. **Gate (fatal):** predict h16v02 and h12v09.
   - The outputs must be uint8 (percent layers, nodata 253) and uint16 (SD) on the MODIS sinusoidal grid (geotransform present).
   - Compare **values** after mapping nodata: the bare layer against the EXP run's A10S0 maps, and tree cover and tree SD against v1.1.0.
     Byte equality is not possible, because those maps are Int16 with no geotransform (see section 1).
2. **Maps:** all 69 tiles for 2019, with SD, written to `out/maps/v1.2.0`. v1.1.0 took 1 h 25 min on 6 nodes; the nodes are checked at launch.
3. **Checks:**
   - Tree cover and tree SD values equal v1.1.0 on all 69 tiles (after mapping nodata), and every file is uint8/uint16 with a geotransform.
   - For the bare layer, per tile: the share of pixels changed and the mean change against v1.1.0. Outside Greenland and Iceland, the changes
     should match the EXP run's 15-tile beyond-noise shares (about 5%).
   - The ice-zone table (`ice_extent.py score`) for the 7 Greenland/Iceland tiles must reproduce the A10S0 column.
4. **Write-up:** `Reports/Version_v1.2.0_Build_Results.md`. Also add a v1.2.0 line to `vcf-version-names`, the README and the to-do list.
5. **Delivery clone (after you look at step 4):**
   - Create the new branch `mcarrol2/v1.2.0` from `mcarrol2/v1.1.0`.
   - Make one docs commit: the README gives v1.2.0, its non-vegetated model and the added rows. No code change.
   - Add the local annotated tag `v1.2.0`.
   - **You push** (commands go in `Git_Delivery_Clone_modis_vcf.md`).

The run takes about 2 h, mostly step 2.

## 3. Decisions

| # | Question | Options | Suggested |
|---|---|---|---|
| V1 | Bare model | (a) A10S0 as fit (seed 0, as v1.1.0) (b) refit | **(a)**: it is the evaluated model |
| V2 | Tiles and year | (a) the 69 v1.1.0 tiles, 2019 (b) also the other years now | **(a)**; other years follow under `Additional_Years_Plan.md` with v1.2.0 |
| V3 | Model and split files off ADAPT | (a) stay on ADAPT, as for v1.1.0 (the repo holds code and docs only) (b) add them to the repo's `data/` | **(a)**. They carry no NGA-derived values (T5 (a)), so (b) is possible later |
| V4 | Branch | (a) new branch `mcarrol2/v1.2.0` from `mcarrol2/v1.1.0` (b) commit on `mcarrol2/v1.1.0` | **(a)**: v1.1.0 stays as tagged |
| V6 | A standalone label file for the h16v02 rows | (a) yes: write the 10,000 rows (keys, label 100, how chosen) as a per-tile label file in the same form as the other training labels, so any future rebuild of the training set from labels includes them (b) no: the split `split_A10` is the record | **(a)** |
| V5 | PR into `develop` (open item A5) | (a) one PR from `mcarrol2/v1.2.0` (b) as planned from v1.1.0 | **(a)**: rlgill gets the current version |
