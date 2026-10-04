# Why the C code's static bare models disagree with historical C6: pixel trace and period test

**Written 2026-10-01.** Requested by the project lead while the non-vegetated run was in progress. All work ran on forest202 and forest203, away from the ngaproc run. Files are in `FullRuns/nonveg_bit15_models_20261001/trace/`. Nothing in rlgill's checkouts was changed: `make_metrics.c` was copied unchanged (same md5).

## Short answer

The C code applies its 30 static bare models correctly. Run on today's inputs, though, the models give answers far from historical C6, which uses the same models: MAE 23, bias +13, against 11.9 between C6 and the labels. Two things are wrong:

1. **The models seem to expect a calendar shifted by about half a year.**
   - Shifting the unsorted inputs by 4 periods (about 192 days) cuts the static models' MAE against C6 from 23.1 to 8.9. The bias goes from +12.9 to -0.8 and the correlation from 0.61 to 0.93.
   - Against the labels the shift gives 12.8, about the same as C6 itself (11.9).
   - 4 is the best shift on 14 of 20 tiles, and 3-6 on the rest, in both hemispheres.
   - The training means stored in the models point the same way: NDVI highest and Band_31 coolest where today's data have the opposite.
2. **The models are fragile.** Even with the shift, single pixels are often far off. Inputs from the C code and Python that differ only by screening in one or two periods move the prediction from 14 to 45 on a desert pixel whose label is 96.

**A Weka or Java version mismatch is not the cause:**
- The numpy port, which uses neither, matches Weka 3.6.14 to within 1e-12.
- Every leaf model uses all 21 attributes, the sign that M5P's smoothing was built in when the models were trained. So 3.6.14 applies them as intended.
- Java floating point is deterministic.

## 1. Pixel trace (C code and Python side by side)

**Harness:**
- `trace/c_harness/`: an unmodified copy of `MOD_PR44B3/make_metrics.c`, a three-line stand-in header, and a driver.
- The driver reads one pixel's 23 composites (bands, state, cloud) from the C run's intermediate files, which are stored pixel by pixel as `MOD44B3.c` reads them. It calls the C code's own `make_metrics()` and prints all 324 metrics.
- The static models (numpy port) were then run on the C code's 21 bare attributes and on Python's 21 inputs.

**Validation:** on all 8 pixels, the static models on the C code's inputs give exactly the C code's map value after rounding.

| Tile, row, col | Type | Label | C6 | C code map | Static, C inputs | Static, Python inputs | Static, Python shifted 4 | RF |
|---|---|---|---|---|---|---|---|---|
| h08v05 3343 1664 | bad | 96 | 98 | 14 | 14.2 | 45.1 | 78.4 | 90.6 |
| h08v05 3301 1659 | bad | 97 | 97 | 14 | 14.3 | 57.6 | 91.7 | 90.6 |
| h08v05 3339 1615 | bad | 95 | 94 | 13 | 13.3 | 45.8 | 75.9 | 92.1 |
| h08v05 3299 1660 | bad | 96 | 94 | 14 | 13.7 | 42.6 | 73.2 | 91.8 |
| h08v05 830 2409 | control | 20 | 20 | 19 | 18.9 | 59.0 | 20.9 | 13.0 |
| h19v12 313 3865 | bad | 39 | 39 | 96 | 95.8 | 95.8 | 70.8 | 41.8 |
| h19v12 428 3832 | bad | 37 | 36 | 92 | 91.5 | 91.6 | 58.2 | 35.5 |
| h19v12 215 4152 | control | 68 | 73 | 71 | 71.4 | 71.2 | 62.7 | 49.7 |

**How the pixels were chosen:** held-out pixels where the label agrees with C6 (within 5) and the C code differs most from C6, plus one control per tile. All 21 input values for each pixel, from the C code and from Python, are in `trace/trace_detail.txt`.

**What the trace shows:**
- **h19v12 (southern Africa):** the C code's and Python's inputs are almost identical, and both predictions are 92-96 bare against a truth of about 37. The models fail here regardless of which code supplies the inputs. The 4-period shift moves the prediction most of the way.
- **h08v05 (western US desert):**
  - The inputs from the two codes differ in a few slots, mainly November Band_7 and Band_31 (by up to about 1,000). That is a screening difference: bits 7-8 shift in the C run, bit 15 in Python.
  - The models turn those small differences into 14 against 45, for a bright pixel (Band_2 about 3,000-3,500) with low NDVI (about 100-250) that is about 96% bare.
  - With the shift the prediction is 73-92.
  - A 14% bare value for that pixel is physically implausible, which marks the models as working outside their range.

## 2. Period test (`trace/period_rotation_test.py`, held-out pixels, 2.13 M)

**Method:**
- Each unsorted input was taken from period (n + r) mod 8 instead of period n.
- The two calendar-mean predictors were recomputed from the shifted values.
- The sorted and greenness-ordered predictors were left unchanged, because they don't depend on calendar order.

| Shift r | MAE vs C6 | Bias vs C6 | Correlation with C6 | MAE vs labels |
|---|---|---|---|---|
| 0 (as run) | 23.1 | +12.9 | 0.61 | 28.9 |
| 1 | 19.7 | +9.5 | 0.72 | 23.9 |
| 2 | 16.0 | +4.8 | 0.80 | 19.1 |
| 3 | 14.2 | +2.2 | 0.83 | 17.4 |
| **4** | **8.9** | **-0.8** | **0.93** | **12.8** |
| 5 | 12.4 | +3.1 | 0.88 | 16.4 |
| 6 | 17.2 | +8.3 | 0.79 | 22.8 |
| 7 | 19.6 | +10.4 | 0.72 | 25.4 |

Reference: C6 against the labels is 11.9. Per tile, the best shift is 4 on 14 tiles, 5 on 3 (h12v01, h20v06, h21v06), 3 on 2 (h11v10, h16v01) and 6 on 1 (h27v04). Full table: `trace/period_rotation_per_tile.csv`.

**Seasonal profiles** (`trace/seasonal_profile.csv`):

| Slot | Today's NDVI | NDVI training mean (models) | Today's Band_31 | Band_31 training mean (models) |
|---|---|---|---|---|
| 0 (2019065) | 186 | 378 | 27,898 | 30,066 |
| 2 (2019161) | 380 | - | 29,598 | 27,948 |
| 4 (2019257) | 272 | 340 | 27,379 | - |
| 6 (2019353) | 191 | 414 | 23,988 | 29,788 |

Today's NDVI peaks in slots 2-3 (June-July). The models' training means peak in slot 6.

## What this means and what is not known

- **Where the shift came from is not known.**
  - Today's code and the v6.0.3 code (the older checkout in `ilab/projects/MODIS-VCF/pge61/`, the era that produced C6) slot composites the same way: `iindex = day/16 - 4`, so slot 0 is DOY 65.
  - So either the operational C6 runs got their composites in a different order (for example through the input dates the code reads, `RangeBeginningDate`), or the models were trained with another period convention and C6 was made some other way.
  - This can't be settled without the operational PCFs or logs, or the training data behind the 2012 models.
- **Even with the shift (MAE 8.9 against C6), the static models are no better than the retrained Python models** (RF 4.1 against the labels, against 12.8 for the shifted static models), and they are much more sensitive to small input differences.
- **For the non-vegetated decision this supports retraining.** It also explains why the C code's non-vegetated layer disagrees with the historical product, which affects any C-against-historical comparison of that layer.
- **Possible next steps (not run):**
  - add the shift-4 static models as a fifth map candidate;
  - look for the operational PGE61 logs or PCFs that show composite order;
  - ask the model's originators (2012, v5.1.0) which period convention they trained on.

## 3. Search of every PGE61 code copy for bare-related processing (added 2026-10-01)

**Copies searched** (read-only):
- today's v7 checkout (`pge61_08312026/pge61`; `AI_code_sync_09152026/C_code/pge61` is a copy of it);
- the older v6.0.3 checkout (`ilab/projects/MODIS-VCF/pge61`, the C6 era);
- `pge61.tar`;
- the driver scripts in `COMB/` (`PGE61*.pl`, `CoeffFiles.txt`).

`pre-alpha-2010*` holds Python-pipeline data, not code. `bare_earth3` was an early attempt at understanding the bare models. The project lead says to ignore it (2026-10-01).

**There is no separate bare module.** Outside `MOD_PR44B3`, the COMB driver only passes `header.bare.arff` (LUN 244105) and stages the 30 model files. Everything bare-specific is in `MOD_PR44B3/MOD44B3.c`, right after the bare prediction. In order:

1. **Prediction:** the mean of the 30 static M5P models, each clipped to 0-100 (`Predict.java`), then `floor(x + 0.5)`.
2. **Sum cap:** if tree + bare > 100, then bare = 100 - tree.
3. **High-latitude rule ("Greenland"):** bare = 100 when the pixel's slot-11 snow flag is 0 (snow, or solar zenith over the cutoff, at DOY 241), or when slots 11-13 (DOY 241-273) have no clear observation.
   - v6.0.3 applies it to every tile with v < 3. That matches the C6.1 high-latitude rule found in the four-way comparison, including h20v02, h21v01, h21v02 and h23v02.
   - v7 limits it to h < 18 and drops the slot-12/13 snow checks. The code note reads "MSZ 4/30/2026 ... consistent with PGE61 V6.0.4".
   - Tiles affected in v7 among the 63: h11v02, h12v01, h12v02, h13v01, h16v01.
4. **Sahara "volcanic soils and varnish" stretch:** on tiles h16-h23, v05-v08, a bare value of 55 or more becomes 2 x bare - 55 (capped at 100). Hand-drawn line boundaries exempt parts of h22v05, h22v06, h23v05 and h23v06.
   - Tiles affected among the 63: h17v05, h18v07, h19v08, h20v06, h20v08, h21v05, h21v06.
5. **Tree cap:** after these rules, tree = 100 - bare where tree + bare > 100. Non-tree vegetation = 100 - tree - bare, kept within 0-100.
6. **Percent_NonVegetated_SD:** 100 x the SD of the 30 model predictions.

**The metric code that feeds the bare models is the same in v6.0.3 and v7.** Comparing the code with whitespace and comments ignored, `make_metrics.c` and `load_multiple_array.c` are identical. `fix_bands.c` differs only in opening input files read-only. The composite slotting (`iindex = day/16 - 4`, by the file's `RangeBeginningDate`) is the same. The available MOD44C (061) files carry correct metadata dates: `A2019065` ends 2019-03-21, `A2019257` ends 2019-09-29. The C6-era (006) MOD44C files are not on disk, so their dates cannot be checked.

**What this means:**
- **None of this code explains the half-year period shift.** It remains unexplained.
- **The Python pipeline has none of rules 3 and 4.** Neither the run's code copy, the delivery branch nor rlgill's checkout has them. Rule 2 exists only on the delivery branch, in a different form: non-vegetated absorbs the overshoot.
- **The C code's non-vegetated map includes rules 2-5.** So C-against-Python comparisons of non-vegetated on the 12 affected tiles mix these rules with the model differences.
- **The C6.1 high-latitude zeroing is this rule 3.** It is a deliberate rule, since narrowed in v7, not a data artefact.

**Decision (project lead, 2026-10-01):** rules 2-4 (sum cap, high-latitude rule, Sahara stretch) are to be considered only after the non-vegetated models have been assessed without them.
