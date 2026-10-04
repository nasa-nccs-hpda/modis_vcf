# Delivery run: SD layers, RF non-vegetated, MEDIAN tree cover (results, to-do item 1)

**Run 2026-10-01, 19:53-20:43, ngaproc201/202 (3 workers each), fully detached. Plan: `Delivery_SD_RF_Median_Plan.md` (approved).** Folder: `FullRuns/delivery_sd_rf_median_20261001/`. Code: frozen copy of commit `615557d` (`code_commit_615557d/`), branch `mcarrol2/c-python-parity-fixes` in `Git_delivery/modis_vcf`.

## Summary

- **All 63 tiles pass every check.** The delivered code reproduces the aggregation run exactly: tree cover = MEDIAN of the 30 RF trees, non-vegetated = MEAN, both SD layers, sum cap.
- **MEDIAN tree cover cuts the sum cap by almost 4×:** 472,428 capped pixels across the 63 tiles, against 1,723,065 with MEAN tree cover. Most of the drop is on three tiles: h16v01 (7,649 against 1,014,739), h20v06 (731 against 280,628) and h12v12 (10,217 against 86,086). MEDIAN puts tree cover at 0 where the MEAN gave a few percent of spurious tree cover on snow and desert.
- On some boreal tiles the cap count rises (for example h11v03: 39,915 against 17,330; h20v02: 74,150 against 30,574), where MEDIAN tree cover is above the MEAN. The mean reduction on capped pixels is 3-14 points.
- **The branch is ready to push** (7 commits ahead of origin). Push it before the `docs/reports` snapshot gets any older.

## Checks (`out/summary63.md`, from `scripts/summarize63.py`)

| Check | Tiles passing |
|---|---|
| Tree cover = aggregation MEDIAN map, every pixel | 63 / 63 |
| Tree SD = aggregation run's SD | 63 / 63 |
| Non-vegetated SD = aggregation run's SD | 63 / 63 |
| Non-vegetated = min(MEAN bare, 100 − MEDIAN tree), every pixel | 63 / 63 |
| Water / out-of-projection codes equal in all 3 layers | 63 / 63 |
| Layers sum to 100 on land | 63 / 63 |
| No negative values | 63 / 63 |
| SD = 0 off land | 63 / 63 |
| **All checks** | **63 / 63** |

The per-tile cap table is in `out/summary63.md`.

## Resources

Each process loads both RF models (14.6 + 13.3 GB on disk). Peak memory was about 42 GB per process, so 3 workers per ngaproc node (225 GB) is the safe limit for full-size tiles. The rounding run hit this limit when its script held extra arrays (`Rounding_vs_Truncation_Results.md`).
