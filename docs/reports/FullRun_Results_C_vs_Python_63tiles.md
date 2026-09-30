# Full 63-tile C vs Python comparison: results (2026-09-19)

Run: `MOD44B_VCF_Development/FullRuns/full63_c_vs_py_20260919-0806/` (launch record: `FullRun_Launch_and_Status.md`).
Compute finished at 19:28 (last C B3 tile); the comparison ran 19:40-19:55 (`code/validation/run_compare_only.sh`, option B: it reads the C HDFs
and the Python predictions and does not touch stage markers). Outputs: `Compare/c_vs_python_63/{summary.txt,per_tile.csv}`.
All 63 tiles were compared; none had a missing product. Year 2019 only. No ground truth: this shows how the two products differ,
not which is closer to the truth.

## What was compared
C: 30 M5P models trained on the pooled B1 sample files of all 63 tiles plus the static bare models, B3 products.
Python: bagged unpruned regression trees (30 members) trained on all labeled pixels of the 63 tiles (`v5.0.3samp` tree labels;
bare labels for 20 tiles only), with Python metrics that mirror C's 64 tree-cover and 21 bare attributes (NDVI-related values still
differ from C's, accepted). Pixels valid (0-100) in both products, 1.11 billion for tree cover.

## Pooled results (raw Python output)

| Layer | corr | mean Python vs C | bias (Py - C) | MAE | RMSE | within 5 | within 10 |
|---|---|---|---|---|---|---|---|
| Percent_Tree_Cover | **0.960** | 23.82 vs 23.45 | +0.37 | 4.53 | 7.04 | 70.0% | 88.1% |
| Percent_NonVegetated | 0.808 | 22.80 vs 29.62 | -6.82 | 13.02 | 19.00 | 36.6% | 54.9% |
| Percent_NonTree_Vegetation | 0.736 | 51.82 vs 45.53 | +6.29 | 13.88 | 19.75 | 32.8% | 51.8% |

The "harmonized" view (C's rules applied to Python: water 200 in the derived layer, tree + non-vegetated capped at 100) changes
almost nothing (non-vegetated corr 0.8076, bias -6.86). Rounding could not be harmonized (Python rasters were truncated when written).

## Per tile
- Tree cover corr per tile: median 0.919, max 0.975, min 0.012. Eight tiles are below 0.8: h12v01, h13v01, h16v01, h21v01 (the four
  high-latitude tiles with waived December inputs), and h19v11, h20v06, h21v06, h29v11. Tree-cover MAE by tile ranges from 0.1
  (h21v06) to 8.0 (h19v09); largest: h19v09 8.0, h19v08 7.8, h26v06 6.8, h24v03 6.7, h20v03 6.7. Per-tile bias is within about
  +/-3 except h16v01 (+5.6).
- Non-vegetated corr per tile: median 0.581, range -0.04 to 0.94; 22 tiles are below 0.5.
- **The two tiles of the earlier two-tile test:** h09v05 tree cover corr 0.945 (was 0.90), mean 8.96 vs 7.69 (was 13.93 vs 14.78).
  **h12v02 tree cover corr 0.917 (was 0.065), mean Python 17.76 vs C 17.27 (was 6.97 vs 1.00), MAE 4.6.** The earlier large h12v02
  disagreement is not present in the 63-tile run. What changed (C now trained on all 63 tiles instead of two, matched predictors,
  samp labels) was not separated, so the cause is not isolated.
  h12v02 non-vegetated still differs (corr 0.51, bias -27.4), as does h09v05 non-vegetated (corr 0.45).

## Why non-vegetated differs more (reading, not tested)
C's non-vegetated models are the static, previously trained `BARE` models; Python's are retrained here on bare labels from only 20
of the 63 tiles. The two sides therefore do not share a training set for this layer, unlike tree cover. Non-tree vegetation is
computed from the other two layers, so it inherits both differences.

## Follow-up items
1. **Valid-pixel masks differ on some tiles** (tree cover): C valid but Python not valid on h12v01 (17.96M pixels), h13v01 (10.32M),
   h21v01 (2.86M) and h11v02 (2.79M). The Python water/no-data handling versus C's on high-latitude tiles was not examined.
2. **Three B3 tiles were marked `failed` in the run's markers (fixed 2026-09-20)** (h12v01, h12v05, h13v01) although all three finished with rc=0 and
   valid 7-layer HDFs (3.9, 4.7, 0.9 MB, like the Sep 11 run's 4.3, 4.5, 0.9 MB). My driver's success rule required more than 10 MB,
   which is wrong for mostly-ocean tiles. The comparison used those three HDFs. On 2026-09-20 the rule was changed to require more than 100 KB
   and the three markers were reclassified to `ok` (originals kept in `markers/_reclassified_20260920/`); B3 now reads 63/63 ok. The comparison itself was
   unaffected.
3. The bare-model difference above (a like-for-like bare comparison would need C's bare models retrained on the same 20-tile labels,
   or Python using C's static bare training set).
4. Python post-processing gaps remain (non-tree vegetation is 0, not 200, over water; no cap at 100; truncation instead of rounding).
