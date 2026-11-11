# Running a year (modis_vcf v1.8.0)

One processing year (see [ALGORITHM.md](ALGORITHM.md), section 2: "2019" = 2019 day 065 to 2020 day 049) goes through five steps:

```
labels ──► 1 sample ──► manifests ─┐
MOD44C ──► 2 metrics (+ cloud-fallback composites on blend and tree-training tiles) ──► 3 splits ──► 4 fits ──► 5 maps ──► 1 C7 .h5 per tile
MOD44W ────────────────────────────────────────────────────────────────────────────────────┘
```

Models are refitted every year (Terra only). Steps 2 and 5 are per tile; 3 and 4 run once per model (tree, bare, cloud-fallback tree).

There are two ways to run it:

- **The year driver** `view/runYear.py` (`model/YearRun.py`) runs all steps as a task graph from one JSON spec, on one or several
  machines sharing a run folder. It is how every product since v1.4.1 was made, and it is the reference for the order and the
  exact commands. Section A.
- **Step by step**, with the commands the driver runs (section B), when your own scheduler runs the tasks.

Set up the environment first ([BUILD.md](BUILD.md)). Input folders: `VCF_MOD44_DIR` (MOD44CH / MOD44CQ, `<dir>/<year>/*.h5` for
Collection 7 or `*.hdf` for Collection 6.1, the year and the next year) and `VCF_MOD44W_DIR` (MOD44W, `<dir>/<year>/001/*.hdf`).

The product uses the Collection 7 inputs (the default collection). There are no default folders; set, for every step:

```bash
export VCF_MOD44_DIR=<folder of the C7 MOD44CH / MOD44CQ .h5 files>   # <dir>/<year>/MOD44C[HQ].A<year><doy>.<tile>.007.*.h5
export VCF_MOD44W_DIR=<folder of the MOD44W files>                     # <dir>/<year>/001/MOD44W.A<year>001.<tile>.061.*.hdf
```

All variables: [Environment variables](#environment-variables).

Fit the models on metrics from the same collection that the maps use; do not mix collections.

## Step 1: training manifests (once per year)

A manifest lists every training pixel (tile-year, row, column, label) and whether it trains or tests. It comes from the label rasters
([TRAINING_LABELS.md](TRAINING_LABELS.md)):

```bash
python -m modis_vcf.view.trainingManifest sample --labels '<labels>/pcttree/VCF_training_pcttree_{tile}.7.0.1.out' \
    --tiles tree_tiles.txt --trainingType pcttree -y 2021 --index position -o manifest_tree_2021.parq --version v1.6.0-2021
python -m modis_vcf.view.trainingManifest sample --labels '<labels>/pctbare/VCF_training_pctbare_{tile}.<7.0.0 or 7.0.1>.out' ...
```

`--labels` takes one pattern; for the bare set, whose files carry two version numbers, make a folder of links with one name pattern,
or one file per tile. `tree_tiles.txt` / `bare_tiles.txt` = the tiles in the label archive (63 / 35). The tree manifest uses
`--index position`, the bare one the default `range` (as in our runs; it only affects row ids).

> **Row order of the added points.** Sampled from the rasters, the 10,000 h16v02 ice points and the 2,560 desert points (h11v11,
> h19v11) sit in tile order. Our runs appended them after all other rows (`--extra`, a parquet list). Same rows, same train/test parts, different order, so a bare model fitted from a fresh sample
> is statistically equivalent to ours but **not byte-identical**. To reproduce our 2019 models exactly, use the 2019 manifests in the
> label archive instead of step 1.

## Step 2: metrics (per tile)

Reads the year's 23 MOD44C composites, applies QA and the low-sun substitution, builds the 8 period composites and the metrics. On
**blend tiles** (rows v06-v11) and on the **tree training tiles** it also builds the cloud-fallback composites and metrics (the cloud-fallback tree model's
inputs) and the cloud counts.

```bash
python -m modis_vcf.view.runYear metrics --spec spec.json --tile h12v09
```

Output per tile: `<metrics root>/<tile>/<year>/{2-Composites, 3-Metrics, lowsun.json, lowsun_counts.csv}`; cloud-fallback tiles also
`<cloud-fallback root>/<tile>/<year>/...` and `<counts dir>/<tile>_{counts_snow.npz, cloudfallback_used_count.npy, cloudfallback_periods.npz, cloudfallback_build.json}`.
(`view/makeMetrics.py` builds the standard metrics alone, without the cloud-fallback composites.)

## Step 3: splits (per model, after the metrics of its training tiles)

```bash
# tree and bare: from the standard metrics
python -m modis_vcf.view.trainingManifest build --manifest manifest_tree_2021.parq --predictors docs/predictors_tree.txt \
    --root <metrics root> -y 2021 -o splits/tree --workers 6 --lowSun le0
python -m modis_vcf.view.trainingManifest build --manifest manifest_bare_2021.parq --predictors docs/predictors_bare.txt \
    --root <metrics root> -y 2021 -o splits/bare --workers 6 --lowSun le0
# cloud-fallback tree model: the tree manifest and predictors, from the cloud-fallback metrics, low-sun off
python -m modis_vcf.view.trainingManifest build --manifest manifest_tree_2021.parq --predictors docs/predictors_tree.txt \
    --root <cloud-fallback root> -y 2021 -o splits/cloudFallbackTree --workers 6 --lowSun off
# check that the cloud-fallback tree split lines up with the tree split (exit 1 = FAIL)
python -m modis_vcf.view.trainingManifest accept --cloudFallbackTree splits/cloudFallbackTree --standard splits/tree \
    --counts <counts dir> --root <metrics root> -y 2021 -o splits/cloudFallbackTree_accept.json
```

Output: `xTrain / yTrain / xTest / yTest / keysTrain / keysTest .parq`.

## Step 4: fits (per model)

```bash
python -m modis_vcf.view.fitAltModel --algo rf -o models/tree --xTrainPath splits/tree/xTrain.parq \
    --yTrainPath splits/tree/yTrain.parq --trainingType pcttree --predictorFile docs/predictors_tree.txt \
    --nEstimators 60 --maxFeatures sqrt --nJobs 12 --randomState 0 --minSamplesLeaf 1
python -m modis_vcf.model.CompactForest models/tree/pcttree.bin models/tree/pcttree.cf
```

The same for `bare` (`pctbare`, `predictors_bare.txt`) and `cloudFallbackTree` (`pcttree`, `predictors_tree.txt`, from `splits/cloudFallbackTree`).
Output: `<type>.bin` (scikit-learn, joblib), `<type>TopN.bin` (predictor order), `<type>.cf` (compact copy, used for the maps).

## Step 5: maps (per tile, after its metrics and the fits)

```bash
python -m modis_vcf.view.vcfPredict -o maps --metricsDir <metrics root> -t h12v09 -y 2021 \
    --tc models/tree/pcttree.cf --tcn models/tree/pcttreeTopN.bin \
    --nv models/bare/pctbare.cf --nvn models/bare/pctbareTopN.bin \
    --nJobs 8 --tcAgg median --nvAgg median --lowSun le0 \
    --cloudFallbackTree models/cloudFallbackTree/pcttree.cf --cloudFallbackTreeTopN models/cloudFallbackTree/pcttreeTopN.bin \
    --cloudFallbackMetricsDir <cloud-fallback root> --counts <counts dir>          # the last two lines on blend tiles (rows v06-v11) only
```

- **`--tcAgg median --nvAgg median` are required**: the command's defaults are `mean`, which is not the product.
- The bare-priority rule is always applied (`--noBarePriority` switches it off, for testing only). Rows v15-v17 need only `-o maps -t <tile> -y <year>` (rule tiles, no metrics, no models).
- A tile with no MOD44W file for the year cannot be mapped; the driver skips it and lists it in `status/skipped_no_mod44w.txt`.
- Output: the Collection 7 product file `MOD44B.A<year>065.<tile>.071.<production time>.h5` (collection set by
  `VCF_OUTPUT_COLLECTION`, default `071`; use `007` for an operational MODAPS run) (the 7 layers; layout and corrections
  in [BUILD.md](BUILD.md), section 6) and `<tile>-<year>-chain.json` (cloud-fallback tree blend and bare-priority pixel counts). `--format both` also writes the
  7 layers as GeoTIFFs (`<layer>-<tile>-<year>.tif`), and `--format geotiff` writes only those. In a `runYear` spec this is
  `"mapFormat"`, default `"h5"`. `--h5Attr NAME=VALUE` (spec: `"h5Attrs": {NAME: VALUE}`) sets root attributes of the file, e.g.
  the production `PGEVersion`.
- After the map of a blend tile, `python -m modis_vcf.model.CloudFallback compose ...` writes diagnostic statistics (optional; nothing reads
  them).

## A. The year driver

```bash
python -m modis_vcf.view.runYear run --spec spec.json        # on each machine; they share <runRoot>/status
python -m modis_vcf.view.runYear status --spec spec.json     # progress
```

Spec (all paths absolute):

```json
{
 "year": 2021,
 "runRoot": "/data/vcf/runs/2021",
 "lowSun": "le0",
 "models": {
  "tree": {"manifest": ".../manifest_tree_2021.parq", "predictors": ".../docs/predictors_tree.txt", "trainingType": "pcttree",
           "fit": {"nEstimators": 60, "maxFeatures": "sqrt", "nJobs": 12, "randomState": 0, "minSamplesLeaf": 1}},
  "bare": {"manifest": ".../manifest_bare_2021.parq", "predictors": ".../docs/predictors_bare.txt", "trainingType": "pctbare",
           "fit": {"nEstimators": 60, "maxFeatures": "sqrt", "nJobs": 12, "randomState": 0, "minSamplesLeaf": 1}}
 },
 "cloudFallbackTree": {"tiles": ["<every map tile in rows v06-v11>"], "manifest": ".../manifest_tree_2021.parq",
             "predictors": ".../docs/predictors_tree.txt", "trainingType": "pcttree",
             "fit": {"nEstimators": 60, "maxFeatures": "sqrt", "nJobs": 12, "randomState": 0, "minSamplesLeaf": 1}},
 "mapTiles": ["h00v08", "...", "h35v10"],
 "workers": {"metrics": 8, "split": 6, "mapJobs": 8, "cloudFallback": 8},
 "fitHosts": ["<hosts allowed to fit>"],
 "slots": {"<host name prefix>": 4},
 "memGB": {"metrics": 35, "split": 130, "fit": 90, "map": 10},
 "cleanup": {"mode": "after_map", "keepTiles": []},
 "maxBacklog": 60
}
```

- Tasks: `metrics:<tile>` -> `split:<model>` -> `fit:<model>` -> `map:<tile>`. Each machine claims tasks whose inputs are done, up to
  its `slots`, when its free memory exceeds `memGB` for that task type. Markers: `status/<task>.done | .failed`; a map is retried once,
  then recorded as failed; a metrics, split or fit failure stops the year. `status/STOP` stops every machine.
- `cleanup: after_map` deletes a tile's composites and metrics once its map and the splits that read it are done (a year writes about
  8-9 TB otherwise). `keepTiles` keeps chosen tiles.
- Per-task time, CPU, peak memory and bytes: `timing/tasks.csv`.

## The pooled bare model (a ready-made model)

The non-vegetated layer can use one model for every year, the pooled bare model (2000-2023), instead of a model fitted each year.
It is delivered as a compact forest folder with its top-N file and a SHA-256 file. In the spec it replaces the bare model's fit:

```json
  "bare": {"trainingType": "pctbare",
           "fixed": {"model": ".../pctbare_pooled_2000-2023.cf", "topN": ".../pctbare_pooled_2000-2023TopN.bin",
                     "sha256": ".../pctbare_pooled_2000-2023.sha256"}}
```

The year then has no `split:bare` or `fit:bare` task. With `"sha256"` given, `runYear run` checks every file of the model and its
top-N file against that file (format of `sha256sum`: `<hash>  <path>`, paths relative to the file's folder) before its first task, and
stops on any mismatch or unlisted file.

**Rebuilding it** (only for a new version of the model; the result should be the delivered model):

1. For each year 2000-2023, the year's bare split, with `"splitOnly": true` in `models.bare` and no `mapTiles` (the run builds the
   metrics of the bare training tiles and `split:bare`, nothing else): `<runRoot>/Python/splits/bare`.
2. A file of lines `<year><TAB><that split dir>`, then
   ```bash
   python -m modis_vcf.view.buildPooledSplit build --years years.tsv --predictors docs/predictors_bare.txt -o pooled_split --seed 0
   ```
   (keys and labels must be identical in every year; each row takes its predictors from one year, dealt evenly per tile and label bin;
   `pooled_split/pooled_split.json` records the years, seed and checksums; the command ends with `check`).
3. Fit and compact it as any model: `fitAltModel --algo rf` on `pooled_split/xTrain.parq` / `yTrain.parq` with `--trainingType
   pctbare --nEstimators 60 --maxFeatures sqrt --minSamplesLeaf 1 --randomState 0`, then `python -m modis_vcf.model.CompactForest
   <out>/pctbare.bin <name>.cf`; write its SHA-256 file with `sha256sum`.

## Known input gaps (expected; not processing failures)

Some tiles lack inputs every year because of the data, not because of a fault in a run. The code handles each case below. The
warnings it logs are expected, the tasks still end in `.done` (or, for h18v14, are not created), and none of them should be counted
as a failure, rerun, or reported as missing data.

**1. Polar night: missing MOD44C granules on rows v00 and v01.** MOD44CH / MOD44CQ are not produced for some winter dates on the
northernmost tiles. In processing year 2019 (C7 inputs, checked on disk) the gaps are:

| Tiles | Missing input dates | Composite periods affected |
|---|---|---|
| h11v01-h24v01 (14 tiles) | DOY 337 and 353 | periods 6 and 7 lose one of their three inputs each |
| h16v00-h19v00 (4 tiles, rule tiles) | DOY 305, 321, 337, 353 and DOY 001, 017 of the next year | periods 6 and 7 have no input at all |

Every processing year 2000-2023 on LAADS has the same tile counts per date (292 tiles; 288 on DOY 305, 321, 001, 017; 274 on DOY 337,
353), so the same pattern is expected every year. Other high-latitude tiles have these granules, but in polar night their pixels
have a solar zenith at or below 0.

What the code does:
- A missing granule is logged as a warning, `Substituting empty day due to: Unable to find file for MOD44CQ.A<date>.<tile>... in
  <dir>`, and the composite then logs `<n> missing input(s) in <composite>`. Both are expected on the tiles above.
- In a period where some inputs exist (v01), the missing input enters the composite as the low-sun value 16000 where every input that
  exists is 16000 (polar night), and as 10000 elsewhere (`CompositeDayFile.fillMissing`). This keeps v01 tiles in line with their
  v02 neighbours. Its masks are all bad.
- The v00 tiles are rule tiles: the high-latitude rule (ALGORITHM.md, section 7) sets every land pixel to tree 0, non-vegetated 100
  and non-tree vegetation 0, whatever the inputs. Their metrics and predictions still run, but input gaps there do not change the
  values. In a period with no input, every input is 10000 and no input is clear, so the period is a fallback period and its Quality
  and Cloud bits are set on land. A gap on a v00 tile at another date is not a failure either (2020: DOY 289 is not on LAADS for
  h16v00-h19v00).
- Inputs with a solar zenith at or below 0, or above 70, are replaced by the low-sun substitution (`--lowSun le0`, ALGORITHM.md). The
  number of pixels replaced per date is written to each metrics folder's `lowsun_counts.csv`; large counts on high-latitude tiles in
  winter are normal.

An input check before a run should accept exactly these dates on exactly these tiles, and should report any other missing granule.

**2. h18v14: no MOD44W.** MOD44W is missing for h18v14 in every collection (C5, C6, C6.1 and C7; its neighbours have it), although
MOD44C exists. Without the water mask the tile cannot be mapped. `runYear` gives it no map task. It writes the tile to
`status/skipped_no_mod44w.txt` and logs the warning `No MOD44W, map skipped: h18v14`. Products for a year therefore have one tile
fewer than the map tile list (318 of 319 in 2019). The year is complete without it, and no rerun will change that.

**3. Rows v15-v17 (Antarctica): no MOD44C.** These tiles have no input composites by design. Unlike the v00 rule tiles, they have
no metrics task: they are mapped from MOD44W only (ALGORITHM.md, section 8). A missing MOD44C there is not a gap.

## Environment variables

| Variable | Use | Default |
|---|---|---|
| `VCF_MOD44_DIR` | MOD44CH / MOD44CQ input folder (`<dir>/<year>/`) | none: required |
| `VCF_MOD44W_DIR` | MOD44W input folder (`<dir>/<year>/001/`) | none: required |
| `VCF_MOD44_COLLECTION` | collection of the MOD44CH / MOD44CQ inputs: `7` or `61` | `7` |
| `VCF_OUTPUT_COLLECTION` | collection number in the `.h5` product file name and `GranuleID` (three digits) | `071`; `007` for an operational MODAPS run |
| `VCF_WRITE_DAILIES` | diagnostics: `1` also writes the daily inputs to `1-Days` (same results; `makeMetrics.py --writeIntermediates` does the same) | `0` (dailies kept in memory) |
| `VCF_COMPACT_MMAP` | diagnostics: `0` reads a compact model into memory instead of memory-mapping it (same results, more memory per map task) | `1` |
| `VCF_LOWSUN_CHECK` | `0` switches off the check of cached dailies against the low-sun mode, as `--lowSunCheck off` | `1` |

Internal, set by the code (values set by hand are replaced): `VCF_LOWSUN_T`, `VCF_LOWSUN_LE0` carry the `--lowSun` mode to the
composite worker processes. The product rules have no switches: the low-sun substitution follows `--lowSun` (default `le0`), and
the zenith-only NDVI rule always applies on its Greenland tiles.

## Resources (2019, all 319 tiles; Intel nodes with 24 cores / 225 GB and 12 cores / 112 GB)

| Task | Count | Time per task (median) | Peak memory | Notes |
|---|---|---|---|---|
| metrics | 292 | 14-15 min (8 workers, 24-core node); 38-48 min on 12-core nodes | 33-34 GB | 21 GB written per tile, 13 GB more on cloud-fallback tiles (deleted by clean-up) |
| split | 3 | 2-6 min | 8 GB | 6 workers |
| fit | 3 | 15-25 min | 47-58 GB | 12 jobs |
| map | 292 | 6.5-7 min | 8-11 GB | 8 jobs; 2-4 per node at once |
| map (rule tile) | 27 | 10-30 s | 0.7 GB | |

The 2019 global run took about 11 h on two 24-core and three 12-core nodes.
