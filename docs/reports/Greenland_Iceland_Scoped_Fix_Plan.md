# Plan: zenith-only NDVI-default fix, scoped to Greenland (and possibly Iceland) tiles

**Status (2026-09-29): plan only. Nothing run, no code changed. Blocked on input data:** the 2019
MOD44C composites for every Greenland/Iceland tile except h16v01 are missing (Stage 0 below). The project
lead will fetch them from the private repository they live in.

Background: `Zenith_Only_Fix_63Tile_Results.md` (the fix is right for h16v01 but hurts 10 other tiles),
`Zenith_Only_Fix_Cutoff70_Retest_Results.md` (70 degrees does not rescue it), and the project lead's
direction of 2026-09-28: apply the fix only to Greenland tiles, and consider Iceland.

## Stage 0 (done 2026-09-29, read-only): tile list and input-data check

### Which tiles contain Greenland and Iceland

Computed from the MODIS sinusoidal grid (R = 6371007.181 m, tile 1111950.52 m) on 22 Greenland and 8
Iceland coastline/interior points. This replaces the guess in the README ("Iceland in h16v02, maybe
h15v02"), which was wrong for Iceland.

| Tile | Contents | In 63-tile run | In 31-tile 65-degree test | Training labels (v5.0.3samp) | Historical MOD44B C6.1 2019 |
|---|---|---|---|---|---|
| h16v01 | Greenland (northwest and interior, 72-80 N); also part of Ellesmere Island (Canada) | yes | yes | yes (all 0) | yes |
| h16v00 | Northernmost Greenland edge; mostly Ellesmere Island (Canada) | no | no | no | yes |
| h17v00 | North Greenland (Peary Land, 81-84 N) | no | no | no | yes |
| h17v01 | East Greenland (70-80 N) | no | no | no | yes |
| h15v02 | Southwest Greenland (Nuuk, 61-65 N) | no | no | no | yes |
| h16v02 | Southeast and west-central Greenland (62-70 N); also the Reykjanes tip of Iceland | no | no | no | yes |
| h15v03 | Cape Farewell, Greenland's southern tip (below 60 N); tiny land area | no | no | no | yes |
| h17v02 | Iceland (almost all of it) | no | no | no | yes |

Notes:
- **Greenland: h15v02, h15v03, h16v00, h16v01, h16v02, h17v00, h17v01. Iceland: h17v02 (plus the edge of h16v02).**
- h15v01 contains no Greenland; the west coast at 70-75 N falls in h16.
- None of the 10 tiles the fix hurt is on this list. C's `vtile < 3 && htile < 18` box would add
  h11v02, which is one of the hurt tiles, so the scope must be an explicit tile list, not a box.
- **Two tiles are shared with Canada.** h16v00 and h16v01 also contain Ellesmere Island. A tile-level
  scope would apply the fix to those Canadian pixels too. Recommendation: scope at tile level first,
  since Ellesmere is also high-Arctic barren land with almost no trees, and consider a pixel-level
  Greenland mask only if the results on those pixels look wrong.
- **Iceland is a different case.** It has real (if sparse) birch woodland at 63-67 N, closer to the
  Europe tiles the fix hurt than to Greenland's ice sheet and tundra. Test it separately and do not
  assume it behaves like Greenland.
- Only h16v01 has training labels, so on every other tile the only accuracy check is against the
  historical MOD44B (Collection 6.1, 2019), which exists for all eight tiles.

### Input data: MOD44C for processing year 2019 (A2019065 to A2020049, 23 composites)

- **MOD44C also lives in `/css/modis/Collection6.1/L3/MOD44B-VCF/dev/`, in one directory per year**
  (note from the project lead, 2026-09-29). Years present: 2000, 2001, 2004, 2005, 2009-2011, 2014,
  2015, 2019, 2020, 2023, plus a `modis_download.py`. `dev/2019/` holds 3,158 files. A first pass
  missed this location because it looked only for a top-level MOD44C product directory under
  `Collection6.1`. **Checked 2026-09-29 (`dev/2019` + `dev/2020`, 2019 window):** same coverage as the
  `/explore` directory below. h16v01 has 21 of 23 (A2019337 and A2019353 also missing here). The other
  seven Greenland/Iceland tiles have none.
- The pipeline actually reads MOD44C from `/explore/nobackup/projects/ilab/data/MODIS/MOD44C/` (a flat
  directory of `MOD44CH.A<date>.<tile>.061.<prod>.hdf` and `MOD44CQ...` files; 1,626 of each). That is
  where the 63-tile run got its inputs.
- In that directory, for the 2019 window:

| Tile | MOD44CH | MOD44CQ |
|---|---|---|
| h16v01 | 21 of 23 (missing A2019337, A2019353) | 21 of 23 (same) |
| h15v02, h15v03, h16v00, h16v02, h17v00, h17v01, h17v02 | 0 of 23 | 0 of 23 |

**Files needed from the private repository:** MOD44CH and MOD44CQ for A2019065 through A2020049 (23
dates, every 16 days) for h15v02, h15v03, h16v00, h16v02, h17v00, h17v01 and h17v02. That is 7 tiles x 23
dates x 2 = 322 files. For h16v01, A2019337 and A2019353 are missing. The earlier h16v01 work treated
these as absent archive files. If the private repository has them, they are worth fetching too (4 files).
Doing so would change h16v01's inputs, so a rerun of h16v01's real (unfixed) prediction would then be
needed as the new baseline.

Where to put them: in a new folder, not in the shared `/explore/.../MOD44C/` directory, unless the
project lead prefers otherwise, e.g.
`MOD44B_VCF_Development/Inputs/MOD44C_greenland_2019/`. The launcher will point at it. Nothing outside the
sync workspace and new run roots is written (standing condition).

## Stage 1 (after the data arrives): cheap check before any model compute

For each new tile, run the existing Stage 0 scan (per-period percent of pixels whose NDVI would be
zeroed: no raw day with `0 < solz < 65`). It reads only the MOD44C/QA data, taking minutes per tile, not
the full pipeline. Expected result: large, near-tile-wide zeroed fractions in the winter and shoulder
periods on the v00/v01 tiles, like h16v01; less on h15v02/h16v02/h17v02. If a tile shows zero
fraction everywhere, the fix cannot change it and it needs no model run.

## Stage 2: full pipeline, real vs zenith-only fixed, three models

New run root, e.g. `FullRuns/greenland_zenith_fix_20260929/`, refusing reuse. Per tile:
1. Build composites and metrics from the new MOD44C (the full Python metrics step; the 63-tile run did
   not cover these tiles, so nothing is cached).
2. Predict with the existing baseline, Random Forest and XGBoost tunedB models, first with real inputs
   and then with the zenith-only NDVI default (65-degree cutoff, the production value), using the same
   driver as `Zenith_Only_Fix_63Tile_Plan.md`.
3. Compare each prediction against the historical MOD44B C6.1 2019: MAE, bias, correlation, mean tree
   cover and fraction of pixels above 20%. On h16v01, also compare against the training labels, as
   before.

Success criterion for Greenland: the fixed prediction moves toward the historical product (lower tree
cover over ice sheet and tundra, lower MAE) on every Greenland tile and damages none. For Iceland
(h17v02) the fix is adopted only if it does not worsen MAE against the historical product. Iceland has
no training labels, so this is a weaker test than the one that caught the 10 hurt tiles. That limit
will be stated in the results.

Launch: fully detached (`setsid nohup`, status and done markers), one `ssh` per node, each wrapped in
`timeout`, after checking `uptime` and free memory (lesson from the cutoff-70 duplicate launch).
Nodes forest203-205.

Time estimate: not measured for a from-scratch tile. The Stage 1-3 driver on an already-built tile took
about 20-25 minutes per tile for all three models (h20v03, 2026-09-28). The metrics build from MOD44C adds
to that and will be timed on the first tile before the rest are launched. With 8 tiles over 3 nodes,
expect a few hours, with a firmer estimate after tile one.

## Stage 3 (only if Stage 2 passes): how the scope would go into code

An explicit tile set (for example `ZENITH_NDVI_DEFAULT_TILES = {h15v02, h15v03, h16v00, h16v01, h16v02,
h17v00, h17v01}` plus h17v02 if Iceland passes), checked where NDVI is composited. This would be written
as a proposed change in `Git_delivery/modis_vcf`, uncommitted, like the `VcfPredict.py` fix. The sync to
rlgill's repositories stays on hold.

## Open decisions for the project lead

1. Fetch the 322 files (and the 4 missing h16v01 files if they exist), and say where to put them.
2. Tile-level scope, accepting the Ellesmere pixels in h16v00/h16v01, or a pixel-level Greenland mask.
3. Whether to include h15v03 (a tiny land area) and h16v00 (mostly Canadian).
4. Approval to launch Stages 1-2 once the data is in place.
