# Running a year (modis_vcf v1.5.2)

One processing year (see [ALGORITHM.md](ALGORITHM.md), section 2: "2019" = 2019 day 065 to 2020 day 049) goes through five steps:

```
labels ──► 1 sample ──► manifests ─┐
MOD44C ──► 2 metrics (+ V4r on blend and tree-training tiles) ──► 3 splits ──► 4 fits ──► 5 maps ──► 7 GeoTIFFs per tile
MOD44W ────────────────────────────────────────────────────────────────────────────────────┘
```

Models are refitted every year (Terra only). Steps 2 and 5 are per tile; 3 and 4 run once per model (tree, bare, partner).

There are two ways to run it:

- **The year driver** `view/runYear.py` (`model/YearRun.py`) runs all steps as a task graph from one JSON spec, on one or several
  machines sharing a run folder. It is how every product since v1.4.1 was made, and it is the reference for the order and the
  exact commands. Section A.
- **Step by step**, with the commands the driver runs (section B), when your own scheduler runs the tasks.

Set up the environment first ([BUILD.md](BUILD.md)). Input folders: `VCF_MOD44_DIR` (MOD44CH / MOD44CQ, `<dir>/<year>/*.hdf`, the
year and the next year) and `VCF_MOD44W_DIR` (MOD44W, `<dir>/<year>/001/*.hdf`).

## Step 1: training manifests (once per year)

A manifest lists every training pixel (tile-year, row, column, label) and whether it trains or tests. It comes from the label rasters
([TRAINING_LABELS.md](TRAINING_LABELS.md)):

```bash
python -m modis_vcf.view.trainingManifest sample --labels '<labels>/pcttree/VCF_training_pcttree_{tile}.7.0.1.out' \
    --tiles tree_tiles.txt --trainingType pcttree -y 2021 --index position -o manifest_tree_2021.parq --version v1.5.2-2021
python -m modis_vcf.view.trainingManifest sample --labels '<labels>/pctbare/VCF_training_pctbare_{tile}.<7.0.0 or 7.0.1>.out' ...
```

`--labels` takes one pattern; for the bare set, whose files carry two version numbers, make a folder of links with one name pattern,
or one file per tile. `tree_tiles.txt` / `bare_tiles.txt` = the tiles in the label archive (63 / 34). The tree manifest uses
`--index position`, the bare one the default `range` (as in our runs; it only affects row ids).

> **h16v02 row order.** Sampled from the raster, the 10,000 h16v02 ice points sit in tile order. Our runs appended them after all
> other rows (`--extra`, a parquet list). Same rows, same train/test parts, different order, so a bare model fitted from a fresh sample
> is statistically equivalent to ours but **not byte-identical**. To reproduce our 2019 models exactly, use the 2019 manifests in the
> label archive instead of step 1.

## Step 2: metrics (per tile)

Reads the year's 23 MOD44C composites, applies QA and the low-sun substitution, builds the 8 period composites and the metrics. On
**blend tiles** (rows v06-v11) and on the **tree training tiles** it also builds the V4r composites and metrics (the blend partner's
inputs) and the cloud counts.

```bash
python -m modis_vcf.view.runYear metrics --spec spec.json --tile h12v09
```

Output per tile: `<metrics root>/<tile>/<year>/{2-Composites, 3-Metrics, lowsun.json, lowsun_counts.csv}`; V4r tiles also
`<v4r root>/<tile>/<year>/...` and `<counts dir>/<tile>_{counts_snow.npz, v4r_used_count.npy, v4r_periods.npz, v4r_build.json}`.
(`view/makeMetrics.py` builds the standard metrics alone, without V4r.)

## Step 3: splits (per model, after the metrics of its training tiles)

```bash
# tree and bare: from the standard metrics
python -m modis_vcf.view.trainingManifest build --manifest manifest_tree_2021.parq --predictors docs/predictors_tree.txt \
    --root <metrics root> -y 2021 -o splits/tree --workers 6 --lowSun le0
python -m modis_vcf.view.trainingManifest build --manifest manifest_bare_2021.parq --predictors docs/predictors_bare.txt \
    --root <metrics root> -y 2021 -o splits/bare --workers 6 --lowSun le0
# partner: the tree manifest and predictors, from the V4r metrics, low-sun off
python -m modis_vcf.view.trainingManifest build --manifest manifest_tree_2021.parq --predictors docs/predictors_tree.txt \
    --root <v4r root> -y 2021 -o splits/partner --workers 6 --lowSun off
# check that the partner split lines up with the tree split (exit 1 = FAIL)
python -m modis_vcf.view.trainingManifest accept --partner splits/partner --standard splits/tree \
    --counts <counts dir> --root <metrics root> -y 2021 -o splits/partner_accept.json
```

Output: `xTrain / yTrain / xTest / yTest / keysTrain / keysTest .parq`.

## Step 4: fits (per model)

```bash
python -m modis_vcf.view.fitAltModel --algo rf -o models/tree --xTrainPath splits/tree/xTrain.parq \
    --yTrainPath splits/tree/yTrain.parq --trainingType pcttree --predictorFile docs/predictors_tree.txt \
    --nEstimators 60 --maxFeatures sqrt --nJobs 12 --randomState 0 --minSamplesLeaf 1
python -m modis_vcf.model.CompactForest models/tree/pcttree.bin models/tree/pcttree.cf
```

The same for `bare` (`pctbare`, `predictors_bare.txt`) and `partner` (`pcttree`, `predictors_tree.txt`, from `splits/partner`).
Output: `<type>.bin` (scikit-learn, joblib), `<type>TopN.bin` (predictor order), `<type>.cf` (compact copy, used for the maps).

## Step 5: maps (per tile, after its metrics and the fits)

```bash
python -m modis_vcf.view.vcfPredict -o maps --metricsDir <metrics root> -t h12v09 -y 2021 \
    --tc models/tree/pcttree.cf --tcn models/tree/pcttreeTopN.bin \
    --nv models/bare/pctbare.cf --nvn models/bare/pctbareTopN.bin \
    --nJobs 8 --tcAgg median --nvAgg median --lowSun le0 \
    --partner models/partner/pcttree.cf --partnerTopN models/partner/pcttreeTopN.bin \
    --partnerMetricsDir <v4r root> --counts <counts dir>          # the last two lines on blend tiles (rows v06-v11) only
```

- **`--tcAgg median --nvAgg median` are required**: the command's defaults are `mean`, which is not the product.
- BP10 is on by default. Rows v15-v17 need only `-o maps -t <tile> -y <year>` (rule tiles, no metrics, no models).
- A tile with no MOD44W file for the year cannot be mapped; the driver skips it and lists it in `status/skipped_no_mod44w.txt`.
- Output: the 7 layers and `<tile>-<year>-chain.json` (blend and BP10 pixel counts).
- After the map of a blend tile, `python -m modis_vcf.model.V4r compose ...` writes diagnostic statistics (optional; nothing reads
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
 "partner": {"tiles": ["<every map tile in rows v06-v11>"], "manifest": ".../manifest_tree_2021.parq",
             "predictors": ".../docs/predictors_tree.txt", "trainingType": "pcttree",
             "fit": {"nEstimators": 60, "maxFeatures": "sqrt", "nJobs": 12, "randomState": 0, "minSamplesLeaf": 1}},
 "mapTiles": ["h00v08", "...", "h35v10"],
 "workers": {"metrics": 8, "split": 6, "mapJobs": 8, "v4r": 8},
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

## Resources (2019, all 319 tiles; Intel nodes with 24 cores / 225 GB and 12 cores / 112 GB)

| Task | Count | Time per task (median) | Peak memory | Notes |
|---|---|---|---|---|
| metrics | 292 | 14-15 min (8 workers, 24-core node); 38-48 min on 12-core nodes | 33-34 GB | 21 GB written per tile, 13 GB more on V4r tiles (deleted by clean-up) |
| split | 3 | 2-6 min | 8 GB | 6 workers |
| fit | 3 | 15-25 min | 47-58 GB | 12 jobs |
| map | 292 | 6.5-7 min | 8-11 GB | 8 jobs; 2-4 per node at once |
| map (rule tile) | 27 | 10-30 s | 0.7 GB | |

The 2019 global run took about 11 h on two 24-core and three 12-core nodes.
