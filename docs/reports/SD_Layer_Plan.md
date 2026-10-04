# Per-pixel SD layers for the Python VCF (plan)

**Written 2026-10-01. Status: plan only, awaiting approval. Nothing has run apart from the read-only check of the historical layers below.**
This is to-do item 8 in `README_START_HERE.md`: the historical MOD44B has `Percent_Tree_Cover_SD` and `Percent_NonVegetated_SD`, and the Python pipeline writes no SD layer.

## How the C code makes the SD (`MOD_PR44B3/MOD44B3.c`)

- **Ensemble:** the 30 models' predictions for a pixel, each clipped to 0-100 (`Predict.java`). For tree cover these are the 30 M5P models trained that year, one per training subset. For non-vegetated they are the 30 static models.
- **Formula:** `SD = floor(100 * sqrt(mean_k (m + 0.5 - p_k)^2) + 0.5)`, where `m` is the ensemble mean and `p_k` is model k's prediction (`MOD44B3.c:376-382`, `459-465`).
  - The rounding offset (+0.5) is added to the mean before the spread is taken, so the stored SD is `sqrt(variance + 0.25)`. **It can never fall below 0.5%.**
- **Storage:** `uint16` in units of 0.01% (`scale_factor` 0.01, `_FillValue` 0). Water and fill pixels get 0.
- **Rounding:** the SD is taken from the raw predictions, before the layer-sum caps and the bare side rules.

## What the historical layers look like (checked 2026-10-01 on forest203, read-only)

Script and output: `FullRuns/sd_layer_prep_20261001/` (`logs/hist_sd_check.log`).

| Tile | C6 tree SD | C6 bare SD | C6.1 tree SD | C6.1 bare SD | C code tree SD | C code bare SD |
|---|---|---|---|---|---|---|
| h08v05 | 6.74 | 6.74 | 2.18 | 6.78 | 2.22 | 9.20 |
| h12v09 | 5.23 | 5.23 | 5.87 | 5.22 | 6.25 | 3.08 |
| h19v11 | 9.46 | 9.46 | 0.68 | 9.48 | 0.70 | 6.73 |
| h20v06 | 0.90 | 0.90 | 0.56 | 0.91 | 0.56 | 3.24 |

Values are mean SD on land, in percent.

**Findings:**
- **In C6, `Percent_Tree_Cover_SD` is a copy of `Percent_NonVegetated_SD`.** The statistics are identical on every tile. C6.1 and the C code write separate layers. C6's tree SD layer should not be used as a reference.
- **The SD never goes below 0.5%, because of the +0.5 offset.** The median in the desert (h20v06) is 0.51%, which means the models agree almost exactly there.
- **The `valid_range` attribute (0-275) is wrong.** 47-96% of land pixels exceed 275 (2.75%), except on h19v11 and h20v06 tree cover, where the models agree closely.
- **The C code's tree SD closely matches C6.1's** (the same kind of models, retrained each year). Its bare SD does not, which is consistent with the static-model problems in `Bare_Static_Model_Pixel_Trace.md`.

## Proposed Python method

**For RF and the baseline (`BaggingRegressor`)**, each of which has 30 members:
- In `VcfPredict._runPrediction`, replace the single `predict` call with one pass over the members, chunk by chunk:
  - RF: `estimators_[k].predict(X)`;
  - baseline: `estimators_[k].predict(X[:, estimators_features_[k]])`.
- Clip each member to 0-100. This changes nothing for RF and the baseline, whose members are averages of training labels.
- Keep the mean (the prediction) and the population SD (ddof = 0) of the members.

**Proposed decisions** (approve or change):
1. **Use the true SD, without the C code's +0.5 offset.** That follows the project goal (Python quality, not parity), and it lets SD fall to 0 where all members agree. A parity variant, `sqrt(var + 0.25)`, would be one line if it is ever needed for a historical comparison.
2. **Keep the historical storage:** `uint16`, value = round(100 × SD), scale factor 0.01, water/out-of-projection/fill = 0. Use the same file naming as the other layers: `Percent_Tree_Cover_SD-<tile>-<year>.tif` and `Percent_NonVegetated_SD-<tile>-<year>.tif`. Do not copy the wrong `valid_range`.
3. **Take the SD of the raw members, before the layer-sum cap.** That is the same as the C code: it describes model disagreement, not the post-processing.
4. **Get the prediction from the same pass**, so there is no extra prediction cost. The mean of the members must equal `predict()`. This is checked in step 1 below; if any pixel differs after the integer cast, the code keeps `predict()` for the value layer and uses the members only for the SD.
5. **XGBoost:** no SD layer (it has no member ensemble). The layer is skipped and the log says so.

**Expected difference in scale:** RF trees also sample predictors at random (`max_features='sqrt'`), so a single tree varies more than one of the C code's M5P models, each fitted to a different training subset. The Python SD will probably run higher than C6.1's. That is a difference of definition, so the comparison focuses on whether the SD tracks real error, not on matching levels.

## Steps (all on forest nodes; the ngaproc run is not touched)

**Code location:** a copy of the Python code in a new run root, `FullRuns/sd_layer_test_<date>/code/`. Nothing goes on the delivery branch until the results are approved. Nothing in rlgill's checkouts is touched.

0. **Cheap check (one tile, forest203, about 10 minutes, then stop for review):**
   - On h08v05, with the bit-15 tree-cover RF, does the members' mean, cast to int16, equal the existing prediction on every pixel? (The comparison is against `qaC_full63_retrain_20260929/Python/predictions_C/rf`.)
   - Record the run time against the plain prediction, and the SD distribution.
1. **Write the code:**
   - `VcfPredict` gets SD output (on by default, with an option to switch it off);
   - a small helper computes the member spread for RF and the baseline;
   - a unit check on 1,000 rows compares it with `np.std` over `estimators_`.
2. **Five test tiles, with the RF tree-cover model and the new RF bare model** (from `nonveg_bit15_models_20261001`):
   - tiles: h08v05 (western US), h12v09 (Amazon), h19v11 (southern Africa), h20v06 (desert), h12v02 (high latitude);
   - spread over forest201-205 (12 CPUs, 112 GB each); about 5-10 minutes per tile;
   - outputs in the new run root only.
3. **Does the SD track error?** This is the main test.
   - **Held-out pixels:** the bit-15 tree-cover 20% split (`qaC_full63_retrain_20260929/Python/training_split/C`) and the bare split from `nonveg_bit15_models_20261001`.
   - **Measure:** the rank correlation of SD with |prediction − label|, and the mean |error| by SD decile.
   - **Same test on C6.1's tree SD**, against |C6.1 − label|. C6's tree SD is not used, because it is a copy of the bare SD.
4. **Level comparison** with C6.1 and the C code SD on the 5 tiles:
   - per-tile distributions, correlation and ratio of medians;
   - by the 4 cover bins and by continent and latitude band, as in the four-way comparison.
5. **Report:** `Reports/SD_Layer_Results.md`. If approved, a separate step adds the code to the delivery branch, with the same checks.

## Expected cost

About 1 hour wall time for steps 0-4: step 0 about 10 minutes, the 5 tiles about 30 minutes on 5 forest nodes, the analysis about 15 minutes. The ngaproc run is unaffected.

## Not in scope

- An SD for XGBoost (it would need about 30 separately bagged models).
- Back-filling SD for all 63 tiles. That would follow later, through the normal prediction run, once the code is approved.
- Any change to how the C code or the historical product computes SD.
