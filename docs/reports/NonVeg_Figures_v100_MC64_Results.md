# Non-vegetated figures: v1.0.0 and Monte Carlo top 64 against C6 (2026-10-04)

**Status: done.** The first figure set for the non-vegetated (bare) layer, requested by the project lead on 2026-10-04.

- 63 tiles, one figure per tile.
- Run `FullRuns/nonveg_figures_20261004` (ngaproc202, detached, about 9 min).
- Figures: `figures_nonveg_v100_mc64_20261004/`. Per-tile statistics: `tile_stats_nonveg.csv` in that folder.

## Layout

Each figure has two rows of the usual three panels (product | MOD44B C6 | product minus C6):
- **Row 1:** v1.0.0 (delivered, 30 trees).
- **Row 2:** Monte Carlo top 64 (30 trees, `mc_rerun_20261002/out/maps63/bare_MC_TOP64`).

Colors:
- **Non-vegetated:** light yellow at 0 to brown at 100.
- **Difference:** red (lower than C6) to green (higher than C6), gray within ±5, ending at ±15.

Code: `FullRuns/nonveg_figures_20261004/code/make_nonveg_figures_2row.py`.

## 63-tile summary (pixel-weighted, against C6)

| Product | MAE | Mean diff | Within ±5 | > 15 higher | > 15 lower |
|---|---|---|---|---|---|
| v1.0.0 | 8.85 | −3.96 | 43.5% | 4.8% | 12.2% |
| MC top 64 | 8.92 | −4.39 | 42.9% | 4.4% | 12.7% |

- MC top 64 is closer to C6 on 19 of 63 tiles.
- Both products are lower than C6 on average.
- The two are a tie against C6. This matches `Monte_Carlo_Rerun_Results.md`, where MC top 64 for non-vegetated was +0.11 on held-out labels and
  mixed against the references.

C6 is a comparison, not truth.

## Note

The project lead decided on 2026-10-04 to use 60 trees for non-vegetated too (README to-do 13). No 60-tree non-vegetated map exists yet.
A third row can be added once one is built.

## Greenland and Iceland (v1.0.0 only, one row; added 2026-10-04)

- **What:** v1.0.0 non-vegetated against C6 for the 6 Greenland tiles and Iceland, the 7 tiles outside the 63. The maps come from
  `FullRuns/final_round_v00_20261002/out/final`.
- **Where:** `figures_nonveg_v100_greenland_iceland_20261004/`, with stats in `tile_stats_nonveg.csv` in that folder. Same script, run with
  `--tag v100` and no second row (`run_gl.sh`).
- **Missing tile:** h15v03 (Cape Farewell) has no MOD44C input and no map.

| Tile | Area | MAE vs C6 | Mean diff | Note |
|---|---|---|---|---|
| h15v02 | SW Greenland | 14.30 | +8.32 | Higher than C6 over the ice-free land |
| h16v00 | Mostly Ellesmere, N Greenland edge | 0.00 | 0.00 | v00 rule (bare = 100), as in C6 |
| h16v01 | NW Greenland, Ellesmere | 5.15 | −4.34 | |
| h16v02 | SE / W-central Greenland | 4.58 | −2.17 | Straight-edged lower-than-C6 patch on the ice sheet (see below) |
| h17v00 | N Greenland (Peary Land) | 0.00 | 0.00 | v00 rule, as in C6 |
| h17v01 | E Greenland | 5.94 | −4.73 | |
| h17v02 | Iceland and a strip of E Greenland | 10.48 | −2.33 | No Greenland fix: Iceland was excluded, so the E Greenland strip is unfixed too |

**Consistency check:** the values match `Greenland_Iceland_Scoped_Fix_v2_Results.md` ("fixed" maps; h17v02 "real"), within the
small change from rounding. On h16v00 and h17v00 the v00 rule takes them to 0.

**h16v02:** the difference panel has a lower-than-C6 area (−5 to −10) on the ice sheet with a straight northern edge. The edge is
faintly visible in the v1.0.0 panel too, so it may be a tile-internal input or zenith-period boundary. Not investigated.
