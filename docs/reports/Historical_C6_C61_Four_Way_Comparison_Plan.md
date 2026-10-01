# Plan: tree cover from historical C6, historical C6.1, the C code and Python RF, 63 tiles

Status: **done 2026-10-01** (approved by the project lead; control passed; 63/63 tiles). Results in `Historical_C6_C61_Four_Way_Comparison_Results.md`.

Naming: earlier reports used a "phase C" and shortened it to "C", which is easy to confuse with the C code. In this plan and its results, everything about the operational pge61 code is called **the C code**. The historical collections are C6 (.006) and C6.1 (.061).

## 1. Goal

Compare 2019 `Percent_Tree_Cover` from four products on all 63 training tiles, per tile and pooled, in the same way as the earlier historical comparisons (`Tree_Cover_vs_Historical_MOD44B_Results.md`, `Aerosol_QA_C_Full63_Retrain_Tables.md` Stage 5b):

| Key | Product | Source |
|---|---|---|
| H6 | Historical MOD44B Collection 6 (.006) | `/css/modis/Collection6/L3/MOD44B-VCF/2019/065/` |
| H61 | Historical MOD44B Collection 6.1 (.061) | `/css/modis/Collection6.1/L3/MOD44B-VCF/2019/065/` |
| CCODE | The C code (pge61 B1/B2/B3), 63-tile run of 2026-09-19 | `FullRuns/full63_c_vs_py_20260919-0806/C/production/build_context/runtime/b3_output/` |
| RF | Python Random Forest, bit-15 QA, 63-tile retrain | `FullRuns/qaC_full63_retrain_20260929/Python/predictions_C/rf/` |

All 4 inputs exist for all 63 tiles (checked).
- **RF timing:** the RF files were written 2026-09-30 between 09:31 and 10:30, before the cloudy-pixel work started (`cloudy_fallback_20260930` was created at 15:00). Nothing from the h12v09 cloud-composite runs is used.
- **The C code product's aerosol screen (accepted by the project lead; stated in the results report):** this is the C code product used in every earlier comparison. It was built with the bits 7-8 aerosol shift (`>> 7`), not the operational `>> 13` and not bit 15. It is not rebuilt for this comparison.

## 2. The C6.1 high-latitude rule and how those tiles are handled

**The rule in C6.1 (from the project lead, 2026-10-01):** the previous developer could not solve a high-latitude problem, so C6.1 sets those pixels to 0, not to fill, in all of the percent-cover products.

**Handling:** the affected tiles are kept in the per-tile comparison, with a note on each. They are left out of the overall (pooled) assessment.

## 3. Cheap pre-check (done 2026-10-01, forest203, about 5 minutes)

This step compared H6 and H61 tree cover on all 63 tiles. The scripts, `c61_check.py` and `c61_zero.py`, are in the session scratchpad and will be copied into the run root.

- **Valid masks are almost identical.** The C6.1 fill code 253 covers at most 0.4% of any tile, which fits the rule above: the high-latitude pixels are 0, not fill.
- **On the v02 tiles, C6.1 sets most tree cover to 0.** On h20v02, 69.8% of valid pixels have C6 tree cover above 0 (mean 29) but C6.1 = 0; on h11v02 the figure is 65.1%. The share of zeros in C6.1 is 0.875 on h20v02, against 0.177 in C6. The bias grows southward across the tile: on h20v02 it is -0.6 in the top 800 rows and -40 at rows 3200-3999.
- **On the v01 tiles the pattern is different:** a near-uniform offset (h13v01 -5.0 everywhere, h12v01 -4.9, h21v01 -2.3), with almost no extra zeros (h13v01: 0.5% of pixels zeroed). h16v01 is unaffected (98.5% exact).
- **Elsewhere C6.1 is close to C6.** Per-tile MAE is 0.01 to 5.7, and bias is within about ±1.3 on most tiles; it reaches -2 to -3 on h23v03 and h24v03.

C6.1 minus C6 by tile (tiles over 5 MAE or beyond ±2 bias):

| Tile | MAE | Bias | Share of valid pixels with C6 > 0 and C6.1 = 0 | Pattern |
|---|---|---|---|---|
| h20v02 | 21.57 | -21.32 | 69.8% | zeroed |
| h11v02 | 18.10 | -17.82 | 65.1% | zeroed |
| h21v02 | 15.02 | -14.90 | to be measured | zeroed (by bias) |
| h12v02 | 12.32 | -11.72 | to be measured | zeroed (by bias) |
| h23v02 | 12.30 | -11.93 | to be measured | zeroed (by bias) |
| h19v08 | 5.67 | -0.34 | | scatter, no bias |
| h24v03 | 5.41 | -3.09 | | mild negative bias |
| h19v09 | 5.17 | -0.11 | | scatter, no bias |
| h13v01 | 5.00 | -4.96 | 0.5% | uniform offset |
| h12v01 | 4.88 | -4.87 | | uniform offset |
| h23v03 | 4.51 | -2.07 | 0.1% | mild negative bias |
| h21v01 | 2.35 | -2.32 | | uniform offset |

**Rule for flagging tiles, fixed before the run.** A tile is *C6.1 high-latitude zeroed* if more than 10% of its common-valid pixels have C6 tree cover above 0 and C6.1 tree cover equal to 0. This follows the stated rule (zeroing) and is measured, not assumed.
- **Expected to flag:** the 5 v02 tiles, h11v02, h12v02, h20v02, h21v02 and h23v02. The run computes the share for every tile, and whatever the rule flags is what gets excluded.
- **Check on the other layers:** on the flagged tiles, the run also reads C6.1 `Percent_NonVegetated` and `Percent_NonTree_Vegetation`, to confirm that the zeroing covers all percent-cover products as stated.
- **The v01 tiles (h12v01, h13v01, h16v01, h21v01): excluded too (decided by the project lead, 2026-10-01).** They do not match the zeroing measure, but the project lead believes the previous developer's high-latitude rule affects them in some way. All four v01 tiles are therefore flagged *C6.1 high-latitude (v01)*: they stay in the per-tile comparison, each with a note on its pattern, and are left out of the overall assessment. That includes h16v01, which was 98.5% exact against C6 in the pre-check; the report will say so.
- **Overall assessment tile set:** the 63 tiles minus the zeroed tiles and the 4 v01 tiles. That is 54 tiles if the zeroing measure flags exactly the 5 expected v02 tiles.

## 4. Method

For each tile, the run reads the 4 layers, builds one common valid mask (pixels where all 4 are in 0-100), and computes these pairs:

| Pair | Why |
|---|---|
| RF vs H6, the C code vs H6 | Repeats the existing assessment on the new mask |
| RF vs H61, the C code vs H61 | The new comparison |
| H61 vs H6 | How much the reference itself moves |
| RF vs the C code | Python against the code it replaces |

**Statistics per pair** use the existing functions: `deepdive_vs_historical.stats`, plus the readers and `cover_bin` in `regtree_alt_models_20260927/code/validation/`. They are:
- n and the mean of each product;
- bias, MAE, RMSE, correlation and R²;
- the share within 5 and within 10, and the fractions over +20 and under -20;
- bias and MAE by 10-point bin of the reference's cover.

**Control (must pass before the full run):** on h09v05 and h12v09, run the new script with the old Stage 5b mask (H6, the C code and the 6 Python products). It must reproduce the Stage 5b numbers for the C code and Q15_RF exactly. That proves the readers and statistics are unchanged, so only the mask and the added reference differ.

## 5. Outputs

New run root `FullRuns/hist4way_c6_c61_20261001/` (nothing existing is overwritten):

- `code/`: the comparison script, the summarizer, and copies of the pre-check scripts.
- `per_tile/<tile>_stats.csv` and `per_tile/<tile>_by_bin.csv`.
- `summary/`:
  - `per_tile_all.csv`: one row per tile and pair, with the zeroed share and a `c61_flag` column (`zeroed`, `v01` or empty).
  - `pooled.csv`: every pair pooled over (a) the unflagged tiles, the headline, and (b) all 63 tiles, for reference.
  - `by_bin_pooled.csv`: the by-bin results pooled.
  - `tables.md`: the generated tables.

H6 results are given on both tile sets: the unflagged set is like for like with H61, and all 63 tiles give continuity with earlier reports.

**Report:** `Reports/Historical_C6_C61_Four_Way_Comparison_Results.md`, containing:
- the pooled table;
- per-tile MAE for every pair on all 63 tiles, with the flagged tiles marked;
- a section on the flagged tiles: the zeroed share, whether the other percent-cover layers are zeroed too, and the RF and C code numbers against both references;
- a note on the C code product's bits 7-8 aerosol shift;
- whether the 2026-10-01 conclusion that Python is "comparable to historical" holds against C6.1.

After review, the conclusion goes into `C_to_Python_Conversion_Assessment.md`, `Project_Summary_and_Conclusions.md` and `README_START_HERE.md`.

## 6. Execution

- **Where:** forest203, or ngaproc201 after `module load singularity/4.3.1`, using the `modis-vcf-latest` container with read-only binds on `/css/modis`.
- **How:** 8 tiles in parallel, launched fully detached (`setsid nohup`), with a status file per tile and an `ALL.done` marker.
- **Estimate:** 4 reads and 6 pairs a tile (plus 2 extra layers on the flagged tiles), about 1 to 2 minutes a tile, so 15 to 30 minutes for all 63.
- **Order:** the control on 2 tiles, then the full run, then the summary, then the report.
- **No changes:** no product, model or code in `modis_vcf`, pge61 or the delivery clone is changed.

## 7. What this will not tell us

- Neither historical collection is ground truth; this measures agreement, as before.
- C6 and C6.1 share the historical training data and the aerosol screen problem.
- It covers 2019 and the 63 training tiles only.
