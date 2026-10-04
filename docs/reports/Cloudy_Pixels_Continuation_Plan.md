# Cloudy pixels (h12v09 and the cloud-fallback composites): what we know, and a plan to continue

**Written 2026-10-02; revised the same day after the project lead's decisions (section 3). Status: plan, not yet approved to run. Nothing run except the read-only check in section 2.**
This continues to-do items 6 and 7 of `README_START_HERE.md`, using these reports:
- `Cloudy_Fallback_Diagnosis_Results.md`;
- `RF_Fallback_Counts_Results.md`;
- `Cloud_Composite_Alternatives_Results.md`;
- `V4r_Darkest_Red_h12v09_Results.md`;
- `Single_Tile_h12v09_Scenario_Plan.md`.

## 1. What has been done (2026-09-30)

**The problem.** A composite period "falls back" when none of its three 16-day MOD44C inputs passes QA. It then takes the mean of all three inputs, clouds included. Cloud-contaminated periods look like low tree cover, and persistently cloudy dense forest is predicted too low. h12v09 (Amazon) is the test tile, with 103,011 "drop" pixels.

**Stage A diagnosis** (`Cloudy_Fallback_Diagnosis_Results.md`):
- **C and Python use the same fallback rule.** The C code's `Quality` and `Cloud` bytes are written only as output layers and never change a prediction.
- **The 2026 C code drops too,** and by more than RF at high fallback counts. Only the historical C6 is less affected: at the drop pixels it fell back in 4.1 of 8 periods, against 5.1 for Python and 6.0 for the C code. The likely reason is its C6 MOD44C inputs; this was not checked.
- **Prevalence:** outside the high latitudes, 10.8% of land pixels have 3-4 fallback periods and 3.7% have 5 or more. These pixels are concentrated in tropical forest and v04 winter tiles.
- **Effect on accuracy:** in dense forest, RF is biased about -7 at 5 or more fallback periods.
- **Hypothesis, untested:** low-label training pixels are the most fallback-heavy (5.8 fallback periods at label 0, against 2.4-3.0 at 50-99). The training data may therefore teach the model that "cloudy signature = low tree cover".
- **Side finding:** Python's `outCloud` flag is inverted (unused today).

**Fallback and cloudy counts as predictors** (`RF_Fallback_Counts_Results.md`, 63-tile RF "N"):
- Small, safe gain: held-out MAE 7.319 → 7.284, no tile worse by more than 0.01.
- At the drop pixels, about 1-1.5 points less low. The drop is not fixed.
- Not adopted.

**Compositing alternatives, cloudiness only** (`Cloud_Composite_Alternatives_Results.md`, daily MOD09GA, h12v09):
- V4 (in an all-cloudy period, take the lowest-red guarded pick) gets 79% of the cleanest variant's gain with no daily data.
- Clear composites are untouched, and no water or shadow picks were found.
- Changing PGE72's priority bits (V1, V2) does little.

**V4r through the model, without retraining** (`V4r_Darkest_Red_h12v09_Results.md`):
- Against C6, the drop is largely removed (bias -21.7 → -4.9).
- Against the held-out labels, the existing model gets worse (MAE 6.84 → 8.70, worse in every fallback group).
- All land moves +2.6 against C6.
- The references disagree, and the test is handicapped because the model was trained on contaminated inputs. **Retraining on V4r metrics was proposed (to-do item 7) and never run.**

The single-tile scenario test bed (to-do item 6; S1-S5) was planned and then replaced by the 63-tile counts run. Its S2 (drop or down-weight contaminated training rows) and S3 (fill fallback periods from the pixel's clean periods) were never run.

## 2. What we have learned since then

### New check (2026-10-02): the delivered product on h12v09

Read-only. Script: `FullRuns/cloudy_fallback_20260930/code/delivered_product_check_20261002.py`; log in `logs/`.
- **Delivered** = RF, MEDIAN of the 30 trees, rounded.
- **G** = the earlier RF, MEAN, truncated.
- SD is the delivered tree-cover SD layer, in points.

| h12v09 | n | G minus C6 | Delivered minus C6 | Delivered, share <= C6 - 30 | Tree SD (mean) |
|---|---|---|---|---|---|
| Drop pixels | 103,011 | -22.2 | **-18.9** | 18.0% | **23.4** |
| Control pixels | 200,000 | -0.8 | +2.4 | 0.0% | 11.9 |
| Dense forest (C6 >= 60) | 15.9 M | -1.7 | +2.1 | 0.2% | 13.7 |
| All land | 22.3 M | -0.1 | +2.5 | 0.2% | 16.4 |

| Dense forest, fallback periods | 0-2 | 3 | 4 | 5 | 6 | 7 | 8 |
|---|---|---|---|---|---|---|---|
| G minus C6 | -1.7 to -0.6 | -0.4 | -0.9 | -5.2 | -8.6 | -11.4 | -16.3 |
| Delivered minus C6 | +1.9 to +2.5 | +3.1 | +3.6 | -0.2 | -4.0 | -7.1 | -13.1 |

**Held-out labels on h12v09** (4,932 pixels; MAE / bias):

| | All | 0-2 fallback periods | 3-4 | 5+ (666 px) |
|---|---|---|---|---|
| G | 7.08 / -4.3 | 7.73 / -3.9 | 5.59 / -3.8 | 10.18 / -7.5 |
| Delivered | **6.41 / -2.7** | 7.14 / -1.9 | 4.76 / -1.9 | **9.76 / -8.0** |

- **MEDIAN and rounding did not fix the cloudy pixels.**
  - They raise the whole map by about 3-4 points, which shrinks the drop from -22 to -19.
  - The slope with fallback count is unchanged: dense forest still falls about 16 points from 4 to 8 fallback periods.
  - On the labels, the 5+ group is barely better (10.2 → 9.8) and is still biased about -8. The gain is in the other groups.
- **The SD layer marks the problem pixels.** 69% of drop pixels are in the top decile of dense-forest SD, against 4.5% of controls. Even without a fix, the SD and a future `Cloud` layer can warn users.
- **The delivered map is 5.3 M pixels (24% of h12v09 land) with 5 or more fallback periods.** There, 6.6% of pixels are more than 15 points below C6 and 10.0% more than 15 points above it, against 1.4% and 3.9% at 0-2 periods. So cloudiness widens the spread around C6 in both directions, not only downward.

### Other things learned since 2026-09-30 that bear on this work

- **Method decisions now fixed:** bit-15 QA, RF, tree cover = MEDIAN, non-vegetated = MEAN, rounding, and the SD layers. Every candidate fix must be judged as a change to this delivered product.
- **A run-to-run yardstick is coming.** The Monte Carlo rerun (running now, `FullRuns/mc_rerun_20261002`) refits the delivered tree and bare models with seeds 1 and 2 and maps them. This gives the seed noise to judge cloudy-fix differences against, at no extra cost.
- **The predictor set may change.** If the project lead adopts an MC set, any cloudy fix must be confirmed on that set.
  - The ranked (greenest/warmest) predictors are the robust ones (stage A), so the set matters.
  - A fix tested now on the delivered 64 is still informative, because differences are relative.
- **More years depend on this** (`Additional_Years_Plan.md`). A compositing change (such as V4r) changes `2-Composites` and every metric. It should be settled before multi-year production, or that production would have to be rebuilt.
- **The `Quality`/`Cloud` layer plan** (`Quality_Cloud_Layers_and_HDF_Output_Plan.md`) fixes `outCloud`. The corrected per-period cloud flag is the same information as the cloudy count, so the two pieces of work share code.
- **Decision rule (project lead):**
  - act on relative differences, and only when the references agree;
  - about 1 point MAE or less is a tie;
  - comparisons against the historical product carry the aerosol caveat.
  
  V4r failed exactly this test: it was better against C6 and worse against the labels. So the next test must be built to separate those two readings.
- **Compute:** ngaproc201/202 are busy with the MC rerun until about tomorrow morning; forest201-205 are free. The V4r machinery (compose → `makeMetrics` → predict) worked on h12v09 after two fixes. The S3 code does not exist yet.

## 3. Decisions (project lead, 2026-10-02)

| Question | Decision |
|---|---|
| Candidates | **No filling of pixels from neighbours or from other periods** (S3 is dropped). Down-weighting cloudy training rows is to be tested. V4r stays: it picks one of the period's own three inputs, not another period or a neighbour. |
| Where | **forest201-205, if free** (all 5 idle on 2026-10-02, 12 CPUs and 112 GB each). ngaproc stays with the Monte Carlo rerun. |
| Scale | **All 63 tiles** |
| Predictors | **The existing (delivered) 64** |
| Why C6 fell back less | **Not tested** |
| Interim warning in the product | **Not needed now** |

## 4. Plan

**Goal:** find a change that makes dense forest with 5 or more fallback periods less biased on the held-out labels, without hurting the other pixels, and with the map references agreeing.

**Scope:** tree cover only. If a candidate is adopted, the non-vegetated model is refitted on the same inputs afterwards (separate step).

### Candidates (all RF, 30 trees, `max_features` sqrt, seed 0, the delivered 64 predictors, the keyed 20% split; scored as MEDIAN, rounded)

| ID | Composites | Extra predictors | Training rows | Fit needed? |
|---|---|---|---|---|
| BASE | current (bit 15) | none | as delivered | no: the delivered model |
| BASE-C | current | fallback + cloudy counts | as delivered | **no:** model N from `rf_fallback_counts_20260930` is exactly this |
| BASE-W | current | none | weight = share of clean periods, (8 - fallback count) / 8, floor 0.1 | yes |
| V4R | V4r (darkest-red guarded pick in all-cloudy periods) | none | as delivered | yes, after V4r composites, metrics and tables for 63 tiles |

**Run-to-run yardstick:** the delivered model refitted with seeds 1 and 2. These come free from the Monte Carlo rerun (`mc_rerun_20261002/Python/models/tree/SEED1`, `SEED2`), fitted on the same rows. A candidate's difference from BASE counts only if it is larger than the SEED1-SEED2 and SEED-BASE differences.

### Step 0: gate (forest node, about 30-45 min)

1. **Generalize the V4r composer.** The 2026-09-30 version is hard-wired to h12v09 and copies `1-Days`. The new copy:
   - takes a tile argument and links `1-Days` instead of copying it;
   - reads the bit-15 composites and `1-Days` from `qaC_full63_retrain_20260929` and the MOD44CQ state from `ilab/data/MODIS/MOD44C`.
   
   The four polar v01 tiles (h12v01, h13v01, h16v01, h21v01) have 21 of 23 inputs, the same polar-night gaps as the delivered build. A missing input is never a candidate pick.
2. **Regression gate (h12v09):** the new code's composites must be byte-identical to the 2026-09-30 V4r run's (`v4r_darkest_red_h12v09_20260930_r2/metrics`).
3. **Arid check (h20v06):** report the share of pixel-periods V4r replaces, which is expected to be near 0.
4. **Weights check:** the training-row weights match the fallback counts at the row keys (x = row, y = col, as established).

A gate failure stops the run.

### Step 1: V4r composites and metrics for 63 tiles (about 1.5-2 h)

- **Build:** V4r composites, then the unchanged `makeMetrics` for the metric families the delivered models use (`metric_names_subset.txt` of the qaC run). On tiles with little cloud, almost nothing changes.
- **Checks per tile:**
  - alignment (the mean of the available inputs equals the stored composite where V4r did not fire);
  - metrics and composites identical where V4r never fired;
  - the shadow/water gate (median NIR of the picks over clear dense-forest NIR, flagged below 0.6), where a tile has enough dense pixels;
  - the trigger share by period.
- **Workers:** 2 per forest node (10 at once), with free-memory checks and tile claims.

### Step 2: tables, split, fits (about 1.5-2 h)

- **Tables:** V4R tree-cover training tables for 63 tiles, from the V4r metrics.
- **Split:** the keyed split as delivered. **Acceptance:**
  - row counts and held-out keys equal the delivered split;
  - every column is equal, cell for cell, at rows where V4r never fired.
- **Fits:** BASE-W (a copy of `fitAltModel.py` with a `--weightFile` option; nothing else changed) and V4R. Each takes about 20 min on a forest node.

### Step 3: evaluation (about 1.5 h)

**Held-out labels, all 63 tiles** (the main reference: independent of the historical product):
- BASE, BASE-C, BASE-W, V4R, SEED1 and SEED2 on the same rows;
- pooled, by fallback group (0-2 / 3-4 / 5+), by cover bin (0-20 / 21-40 / 41-60 / 61+), by continent × latitude band, and per tile;
- bootstrap intervals over tiles and blocks.

**Maps** of h12v09, h11v08, h19v09, h19v08, h26v06, h12v04 (winter check) and h20v06 (arid control), for all six models:
- against C6, C6.1 and the C code;
- by fallback count, at the h12v09 drop and control pixels, on the stable set, and as the share of pixels more than 15 points above and below C6;
- the map predictor for BASE-C adds the two count rasters, as in the counts run, but with MEDIAN and rounding.

**V4r's all-land shift** (+2.6 against C6 on h12v09 without retraining): rechecked after retraining, by MCD12Q1 land cover.

**Compute:** one map prediction at a time per forest node (two caused out-of-memory failures on 2026-10-01). About 42 predictions at about 7 min each over 5 nodes, roughly 1 h.

### Reading the results (for the project lead to decide; no automatic adoption)

A candidate is worth adopting if:
- the 5+ fallback group improves on the held-out labels by more than the seed noise;
- the 0-2 and 3-4 groups get no worse than about 1 point;
- the cloudy-tile maps move the same way against C6, C6.1 and the C code.

If the labels and the maps disagree again, as with V4r untrained, the report says so and makes no recommendation.

### Later, and only if a candidate is adopted

- Refit the non-vegetated model on the same inputs.
- Confirm on the predictor set chosen after the Monte Carlo rerun.
- Map all 63 tiles.
- Settle this before `Additional_Years_Plan.md` runs, because a compositing change alters every metric.

### Practicalities

- **Run root:** `FullRuns/cloudy_v4r_weights_<date>/`, new; the scripts refuse to reuse it.
- **Code:** copies only. No product code edited; nothing in rlgill's checkouts or on the delivery branch.
- **Runs:** fully detached, with status and done markers. Earlier runs are mounted read-only.
- **Total:** about 5-7 h on forest201-205.
- **Results:** `Reports/Cloudy_Pixels_Continuation_Results.md`, with the README updated.
