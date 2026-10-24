# How the MODIS VCF product is made (modis_vcf v1.7.0)

This describes what the code computes, in the order it runs, for a reader who knows remote sensing but not this algorithm. File and
function names point to where each step lives. How to run each step is in [RUNNING.md](RUNNING.md); the training labels are in
[TRAINING_LABELS.md](TRAINING_LABELS.md).

## 1. What the product is

For each MODIS land tile and year, seven 4800 x 4800 GeoTIFFs on the MODIS sinusoidal 250 m grid (231.656 m pixels; LZW, single band):

| Layer | Type | Values |
|---|---|---|
| `Percent_Tree_Cover` | uint8 | 0-100 % of the pixel covered by tree canopy |
| `Percent_NonVegetated` | uint8 | 0-100 % bare (also called non-vegetated) |
| `Percent_NonTree_Vegetation` | uint8 | 0-100 %, = 100 - tree - bare |
| `Percent_Tree_Cover_SD` | uint16 | spread of the tree model's trees, round(100 x SD) |
| `Percent_NonVegetated_SD` | uint16 | the same for the bare model |
| `Quality` | uint8 | bit n set = composite period n+1 had no good input (it fell back to a mean of all inputs) |
| `Cloud` | uint8 | bit n set = period n+1 fell back and had no clear (cloud-free) day |

In the three percent layers, water = 200, outside the projection = 250, no data = 253. In the SD, Quality and Cloud layers, water and
outside the projection are 0. File names: `<Layer>-<tile>-<year>.tif`, e.g. `Percent_Tree_Cover-h12v09-2019.tif`.

The three percent layers sum to exactly 100 on every land pixel.

## 2. Inputs

| Input | What is used | Where |
|---|---|---|
| **MOD44CH / MOD44CQ** (Collection 6.1, PGE72) | 16-day composites: reflectance bands 1-7, band 31 (brightness temperature), solar zenith, the `state` QA word | `model/ProductTypeMod44.py` |
| **MOD44W** (Collection 6.1) | land / water / outside-projection mask, the file for day 001 of the year | `model/ProductTypeMod44W.py` |
| Training labels | percent tree and percent bare per pixel on 98 tiles | [TRAINING_LABELS.md](TRAINING_LABELS.md) |

Input folders are set in `model/Settings.py` (environment variables `VCF_MOD44_DIR`, `VCF_MOD44W_DIR`).

### The processing year is not a calendar year

"Year 2019" uses the 23 sixteen-day composites from **day 065 of 2019 to day 049 of 2020**: 19 in 2019 and 4 in 2020. A year can only
be run once the next year's first four composites exist.

## 3. Daily inputs and QA (per 16-day composite)

Each 16-day MOD44C granule is one "day file" (`model/BandDayFile.py`). For each band and pixel a value is **good** when all of these
hold (`ProductTypeMod44.createQaMask`):

- state bits 0-1 (cloud) = 0, bit 2 (cloud shadow) = 0, bit 6 (adjacent to cloud) = 0;
- state bit 15 = 1 (PGE72's overall QC flag; in MOD44C state, bits 7-8 are not an aerosol field, so they are not used);
- solar zenith 0 < solz < 65.

A separate **clear-day** mask uses only the cloud bits (0-1 = 0) and solz < 65 (`createCloudMask`). The same masks apply to every
band, band 31 included.

**Low-sun substitution (LE0, `model/LowSun.py`).** Inputs of bands 1-7 and 31 with solar zenith above 70, or at or below 0 (polar
night), are replaced by 16000 before compositing. This is the intent of the operational C code (`fix_bands.c`) and stops dark,
low-sun values from looking like forest at high latitude. Every metrics folder records the mode (`lowsun.json`) and how many pixels
were replaced per day (`lowsun_counts.csv`). A missing granule becomes a day with no good pixels. Its value is 10000 ("substitute"), except where every input of the period that
is present was set to 16000 by the low-sun substitution: there the missing day is 16000 as well (`CompositeDayFile.fillMissing`).
Without this, tiles whose polar-night granules are missing (every v01 tile in 2019 lacks days 337 and 353) got darker winter periods
than their neighbours whose polar-night granules exist, which showed as a step in percent bare at the v01 / v02 boundary.

## 4. Composite periods (8 per year)

The 23 inputs are grouped three at a time into **8 periods of 48 days**, starting on days 065, 113, 161, 209, 257, 305, 353 and 033
of the next year (the last period has two inputs). For each band, pixel and period (`model/CompositeDayFile.py`):

- the composite is the **mean of the good inputs**;
- if no input is good, it is the **mean of all inputs** ("fallback"); the period's Quality bit is set, and its Cloud bit too when no
  input was a clear day.

NDVI is computed from the composites of bands 1 and 2. On six Greenland tiles (h15v02, h16v00, h16v01, h16v02, h17v00, h17v01) NDVI is
set to 0 in any period with no input of 0 < solz < 65 (`Metrics.ZENITH_ONLY_NDVI_TILES`).

## 5. Metrics (the predictors)

From the 8 periods of 9 bands (bands 1-7, band 31, NDVI), `model/Metrics.py` and `model/CMirrorMetrics.py` write per tile:

- `UnsortedMonthlyBands`: the 72 period values themselves (named e.g. `UnsortedMonthlyBands-Band_2-Day_2019353`);
- statistics of the periods, computed exactly as the operational C code does (integer arithmetic): means over period groups
  (`CUnsMean13` = mean of periods 1-2, `Mean16` = 1-4, `Mean19` = 1-6, `Mean46` = 3-4, `Mean79` = 5-6) and an amplitude (`Amp15`),
  over the periods in time order (`CUns...`), ordered by NDVI high to low (`CGreen...`, `CGreenMax` = the greenest period), or by band
  31 high to low (`CTherm...`); and each band sorted on its own (`CSortedRank0`-`7`).

The models use 64 of these for tree cover and 21 for bare (`docs/predictors_tree.txt`, `docs/predictors_bare.txt`). Predictor names
carry 2019 dates; for another year the code maps each date to the same period of that year (`Metrics.getMetricFromRf`), so the same
lists serve every year.

## 6. Models

Two Random Forest regressions (scikit-learn 1.2.0, `view/fitAltModel.py`): 60 trees, `max_features='sqrt'`, minimum leaf 1, seed 0.

- **Tree cover** model: trained on the tree labels (63 tiles).
- **Bare** model: trained on the bare labels (35 tiles, including 10,000 points labelled 100 on the Greenland ice sheet and 2,560 points
  labelled 100 in the Atacama Desert and the Namib Sand Sea).

Training rows are every labelled pixel; 20% are held out for testing by a hash of the pixel key (`model/TrainingManifest.py`). The
predictors at each row are read from that year's metrics, so **models are refitted every year**; the labels stay the same.

Each model is also saved as a **compact forest** (`model/CompactForest.py`: the same trees as flat arrays, about 6x smaller, predicted
with numba). Its predictions are identical to the scikit-learn model's, tree by tree.

**Prediction** (`model/VcfPredict.py`): every tree predicts every pixel; the layer value is the **median of the 60 trees**, rounded
(`floor(v + 0.5)`), clamped to 0-100; the SD layer is the population SD of the 60 trees x 100, rounded.

## 7. Rules applied after prediction (in this order, `model/VcfPredict.py`)

1. **Blend A** (tree cover only, tile rows v06-v11 = about 30 N to 30 S; `model/BlendA.py`). Persistent cloud in the tropics makes the
   standard composites fall back to cloudy means. A second tree model, the **partner**, is trained on **V4r composites**
   (`model/V4r.py`): in a period where every input is cloudy and the period fell back, V4r takes the single input with the lowest red
   reflectance among those that are not shadow and look vegetated (NDVI >= 0, NIR >= 1500), instead of the mean. V4r is built without
   the low-sun substitution. The partner has the same predictors and recipe as the tree model and is refitted every year on V4r metrics
   of the same tree labels. For each pixel:

   `tree = floor((1 - w) x clear + w x partner + 0.5)`, with `w = cloud weight x cover weight`,
   cloud weight = CUR_ZS / 8 when CUR_ZS >= 5 of the 8 periods, else 0 (CUR_ZS = periods that used a V4r pick with a usable sun angle
   and no snow), cover weight = clip((clear - 20) / 30, 0, 1).

   So the blend acts only where at least 5 periods were persistently cloudy and the standard model already sees more than 20% tree.
   The tree SD layer stays the standard model's. The partner's training split is checked against the standard split
   (`trainingManifest accept`).
2. **Sum cap.** Bare is lowered (never tree) so that tree + bare <= 100; non-tree vegetation = 100 - tree - bare.
3. **High-latitude rule.** On row v00 (north of about 80 N) every land pixel is tree 0, bare 100 (the C code's rule).
4. **Water and outside projection** from MOD44W: 200 / 250 in the percent layers.
5. **BP10** (the last step). Where the bare model's raw (uncapped) value is 100 and the final tree cover is 1-9: tree 0, bare 100,
   non-tree vegetation 0. Removes small spurious tree values on bare ground.

## 8. Tiles without models

- **Rows v15-v17 (Antarctica)** have no MOD44C. They are written from MOD44W alone (`model/RuleTile.py`): land = tree 0, bare 100,
  non-tree vegetation 0, SD 0, every Quality and Cloud bit set on land.
- **A tile with no MOD44W file** for the year cannot be masked: it is skipped and listed in the run's
  `status/skipped_no_mod44w.txt` (in 2019: h18v14).

## 9. Differences from the operational C code (PGE61)

The C code trains M5P model trees (Weka) and is the source of the historical MOD44B. The Python code keeps its inputs, QA, periods and
derived metrics (the C-mirror metrics are bit-identical), and differs in: Random Forest models with the median of the trees; the SD,
Quality and Cloud layers; low-sun substitution as the C code intended; blend A; BP10; the Greenland ice training points; the
v15-v17 rule tiles. The development history and evidence are in `CHANGELOG.md`.

## 10. Known limitations

- **Regions far from the training labels.** The bare labels cover 35 tiles and the tree labels 63. Many drylands have no bare label
  within several hundred to a few thousand kilometres (for example Patagonia, the Monte, the Sahel edge, India and Pakistan, north-west
  Mexico, the Kazakh and Mongolian steppes, the Tibetan plateau). There the models extrapolate: the spread between the trees (the SD
  layers) is high, and the values move by a point or two whenever the training changes, including the yearly refit. In 2019 the
  v1.6.0 refit raised bare by 0.5-2 points on 11 such dryland tiles compared with v1.5.1.
- **A low SD does not mean a correct value.** The SD shows how much the trees disagree. Where the labels themselves follow a different
  convention, the trees agree on it and the SD is low.
- **Label conventions differ by region.** Pure desert is labelled 100 in the Sahara and Arabia but about 40-80 in the southern
  hemisphere deserts (since v1.7.0, desert-core points labelled 100 in the Atacama and the Namib Sand Sea bring those two to about
  85-92 % bare, close to C6); lichen and prostrate-shrub tundra is about half bare in the North American and Greenland labels but low in the
  Siberian ones. The maps follow the labels in each region.
