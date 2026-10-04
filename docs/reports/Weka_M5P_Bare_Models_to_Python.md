# The C code's static bare models, translated to Python (no Weka, no C)

**Written 2026-10-01.** Investigation requested by the project lead while the non-vegetated run (`NonVegetated_Bit15_Models_Plan.md`) was in progress. Nothing in rlgill's checkouts was changed.

## Result

**Yes, it can be done, and it has been done.** The 30 static bare models (`pge61/BARE/weka.bare.00-29.model`) now exist as plain arrays with a pure-numpy predictor. The predictor reproduces Weka exactly:
- On 145,000 test rows, every one of the 30 models agrees with Weka to within 1.4 × 10⁻¹², which is floating-point rounding.
- The C code's rounded ensemble value, `floor(mean of the 30 clipped predictions + 0.5)`, is identical on all 145,000 rows.

At run time the predictor needs only numpy: no Java, Weka or C. Weka was used once, offline, to read the models out of Java serialization.

## Where it is

`AI_code_sync_09152026/Weka_M5P_Export/` (new folder):

| File | What it is |
|---|---|
| `json/bare_m5p_30.npz` (2.9 MB) | The 30 models as arrays: splits, leaf linear models, missing-value means, attribute names. This is the file to use. |
| `json/bare_m5p_30.json` (20 MB) | The same, human-readable: the direct export. |
| `json/source_models_md5.txt` | Checksums of the 30 source `.model` files. |
| `python/m5p_numpy.py` | The predictor: `M5PEnsemble.from_npz(...)`, then `.predict(X)` (clipped 30-model mean), `.predict_c_rounded(X)` (as written by the C code), or `.predict_each(X)` (each model's raw value). `X` is (n, 21) in the C code's attribute order, `n014` … `n223`. |
| `java/ExportM5P.java` | The one-time exporter (reflection on Weka's private fields). |
| `java/WekaScore.java`, `validation/` | The check against Weka: test rows, Weka's scores, the comparison script and its output (`compare_result.txt`). |

## How the models work (read from Weka's bytecode, then confirmed by the exact match)

Each file is a `weka.classifiers.trees.M5P`: a model tree with 1,139-1,282 leaves on the 21 bare attributes. A prediction takes three steps:
1. **Missing values:** a missing attribute takes the training mean stored in the model (`ReplaceMissingValues`). The C code treats any input of -30000 or less as missing (`Predict.java:58`). The two other filters stored in the model (`NominalToBinary`, `RemoveUseless`) change nothing for these models; the exporter checks this.
2. **The tree:** at each split, go left when the value is ≤ the split value.
3. **The leaf:** a linear model, intercept + Σ coefficient × value. M5P's smoothing (on by default) is built into the leaf models when the tree is trained, so the leaf model is the whole prediction.

The C code then clips each model's prediction to 0-100, averages the 30, adds 0.5 and floors (`Predict.java:62-78`, `MOD44B3.c:403-404`).

## How it was checked

- **Test rows (145,000):**
  - 125,000 real bit-15 bare predictor values from five tiles (h20v06, h19v11, h12v02, h29v11, h11v08);
  - 10,000 rows with 10% of their values set to missing;
  - 10,000 rows with one value set exactly equal to a split threshold, which tests the ≤ rule.
- **Reference:** Weka 3.6.14 scored the rows with the original `.model` files, building instances from `header.bare.arff` the same way `Predict.java` does.
- **Agreement:** the largest raw difference is 9 × 10⁻¹³ on the missing and split-threshold rows and 1.4 × 10⁻¹² on the real rows.
- **Speed:** about 47,000 rows a second for all 30 models on one core, so a full 4800 × 4800 tile takes about 8 minutes on one core and splits easily across cores.

## Found on the way: the C run used Weka 3.6.14, not 3.8.6

- **Weka 3.8.6 cannot read these models.** Deserialization fails with a `ClassCastException` in `weka.core.RelationalLocator`, because the stored format is older. It also fails on the tree-cover models that the 2026-09-19 C run trained itself.
- **Which jar the C run actually used:**
  - The C code takes its classpath from the PCF's `WEKA_DIRECTORY` (`MOD44B3.c:279`).
  - In the 2026-09-19 run that was `C/production/build_context/runtime/weka/`.
  - Its `weka.jar` links to `~/.local/share/weka-3.6.14/weka.jar`.
- **The build notes are out of date.** `pge61/build_context/env.sh` and `native_build_context.txt` name `~/.local/share/weka/weka.jar` (3.8.6). Anyone rebuilding or rerunning the C code needs 3.6.14.

## What it means for the non-vegetated work

- **It doesn't reproduce the C code's bare map on its own.** It reproduces the C code's bare *models*. The bare map also depends on the inputs. Python's 21 bare predictors mirror the C code's attributes, but two things differ:
  - NDVI: 7 of the 21 predictors use it, and the two codes define the 48-day NDVI differently (step 7 of the non-vegetated run);
  - the aerosol QA: bit 15 in Python, the bits 7-8 shift in the 2026-09-19 C run.

  On Python's bit-15 metrics, the translated models give "the historical bare models on Python inputs".
- **It allows a fourth candidate for the non-vegetated layer.** The static models could run inside the Python pipeline with no Weka, scored alongside the retrained baseline, RF and XGBoost. That would show directly whether retraining on 20 tiles beats the models behind the historical product. It is not in the approved plan.
- **The tree-cover models don't need this.** The C code retrains them every year (`MOD_PR44B2`), so there is no static set to translate. The same exporter would work on them, with 3.6.14.
