# Plan: NEON airborne lidar as tree cover truth for the historical MOD44B and the B and C models (2026-09-30)

**Status: Step 1 DONE (2026-09-30 ~10:05). Steps 2-5 NOT started; awaiting the project lead's go-ahead.**

## Why
The G-LiHT test on h11v02 (`Lidar_Reference_Coincidence_20260930.md`) had three limits: one boreal tile, lidar 5 years older than the
products (2014 vs 2019), and ~2,700 pixels. It showed that all products share a large error against lidar (MAE ~19-20; per-pixel errors
correlate 0.92-0.96), and that on that tile C was worse than B (+0.7 to +1.1 MAE). This contradicts the held-out labels. NEON gives 2019 lidar
at 25 sites in most US biomes, on 10+ of our training tiles, so the B-vs-C difference can be checked across references and regions.
That is the rule we agreed on: act only when the sign is the same across independent references.

## Data (local; nothing to download)
`/explore/nobackup/projects/ilab/data/AGB/Airborne_Lidar/` (gtamkin, from the earlier AGB work):
- `FSD/<year>_NEON_<Dxx>_<SITE>_DP3_<E>_<N>_{CHM,CC,RUG}.tif`: 1 km tiles, 1 m, UTM. 25 sites, 2013-2022.
- `DP1.30003.001/`: NEON classified point clouds (LAZ), about 957 GB for 2019 alone.
- **What CC is (established from the data, `code/neon_cc_probe.py`):** CC = share of CHM cells > 2 m in a 3x3 window. It is reproduced
  exactly on a UNDE 2019 tile (r = 1.0000, MAE 0.0000) and on a HARV 2019 tile (r = 0.9999; the rest is edge handling). The values are multiples of 1/9.
  So CC is derived from the CHM with a 2 m cutoff (the "deep gap" idea of NEON's Forest Structural Diversity tutorial, applied per cell).
  It is **not** a point-cloud gap fraction and adds nothing beyond the CHM. It is not used; cover is computed from the CHM at 2, 3 and 5 m.
- Years used (closest to 2019; later year on ties), from `FullRuns/stable_pixels_20260930/neon/site_years.txt`: 2019 for 21 sites,
  DEJU 2018, GRSM 2018, MLBS 2018, RMNP 2020. 5,659 CHM tiles in total.

Expected MODIS tiles (approximate, from site coordinates; step 2 gives the exact list): h11v02 BONA, DEJU (same tile as the G-LiHT test,
now with 2018/2019 lidar); h11v04 UNDE (confirmed), STEI, CHEQ; h12v04 HARV, BART; h12v05 SERC, SCBI; h11v05 MLBS, GRSM; h10v05 JERC,
TALL, DELA, LENO, UKFS; h09v05 CLBJ; h09v04 WREF, ABBY, RMNP; h10v04 YELL; h08v05 SJER, SOAP, TEAK; PUUM (Hawaii) probably not a training tile.

## Steps
All work stays in `FullRuns/stable_pixels_20260930/` on forest204 (and forest201/202/203/205 once the bit-15 run releases them),
fully detached, with status markers. No changes to rlgill's repositories.

1. **DONE: adapt the G-LiHT mapping to NEON and test on one site.** `code/neon_modis_coverage.py` reuses the G-LiHT sampling
   (3 m sample grid, projected to MODIS sinusoidal, binned to 250 m pixels). Per pixel: coverage, and CHM cover at >= 2, 3 and 5 m.
   Test UNDE 2019 (`neon/neon_modis_pixels_UNDE-2019.csv`): 208 CHM tiles, all on h11v04; 3,053 pixels >= 90% covered; 3 m cover bins
   0-10/10-30/30-60/60-100 = 188 / 226 / 557 / 2,082; 58 s.
2. **Map all 25 site-years** (one process per site, detached). From the UNDE timing: about 25 minutes of CPU in total, **about 5 minutes wall clock** in parallel.
   Output: exact site -> MODIS tile list and pixel counts per site.
3. **Pixel pools for each MODIS tile** (`code/stable_pool.py`: excludes pcttree/pctbare labels plus a 2-pixel buffer, 21-year historical
   mean/SD). About 10 tiles in parallel, **about 10 minutes**.
4. **Retrained B predictions** for the tiles not yet done (h11v02 done), same method as `code/run_predict_B_h11v02.sh` (B metrics bound
   read-only). About 11 minutes per tile for the three models on one node: **about 1.5-2 h on forest204 alone, or 30-40 minutes** once the other
   four nodes are free (the bit-15 run is due about 10:30-11:00). C predictions for all 63 tiles come from the bit-15 run (Stage 5b).
5. **Evaluate** (`code/gliht_eval.py` generalised to several tiles and to NEON): pixels >= 90% covered, not labels/buffer, valid in all
   products. HIST, B_*, C_* and Bold_* against NEON cover at 2, 3 and 5 m. Per site and tile, pooled, by cover bin, stable-pixel subset;
   C-B MAE differences with block-bootstrap intervals; label-vs-lidar check on the label pixels inside the lidar to fix the height cutoff
   per biome (3 m on h11v02). **About 15 minutes.**

**Total:** about 1-1.5 hours wall clock after the go-ahead if the four nodes are free, or about 2-2.5 hours on forest204 alone.

## Optional step 6: canopy cover from the point clouds (true gap fraction)
This is the direct test of canopy cover (MOD44B's definition, light interception) vs crown cover (CHM-based). Per 250 m pixel: the share of
first returns (or all returns) above 3-5 m. The container has no LAZ reader (no laspy, lazrs or PDAL), so it needs a separate Python
environment first. 2019 alone is about 957 GB. **Recommended scope:** 3 sites (for example BONA, UNDE, HARV), about 1 hour including setup;
all sites about 2-3 hours.

## Decisions for the project lead
- Go-ahead for steps 2-5.
- Step 6 now, later, or not at all; and which sites.
- Whether PUUM (Hawaii) is included if it is not on a training tile (it would need B and C metrics built first).
