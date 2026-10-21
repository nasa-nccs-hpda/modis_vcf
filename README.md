# MODIS Vegetation Continuous Fields (modis_vcf)

Python code for the MODIS Vegetation Continuous Fields product (MOD44B, Terra): yearly percent tree cover, percent non-vegetated
(bare) and percent non-tree vegetation for every MODIS land tile on the 250 m sinusoidal grid, with SD, Quality and Cloud layers. It
replaces the operational C code (PGE61) and reads the same inputs, MOD44CH / MOD44CQ (PGE72) and MOD44W.

Version: **v1.6.0** (see `CHANGELOG.md`).

## Documentation

| Document | For |
|---|---|
| [docs/BUILD.md](docs/BUILD.md) | the conda environment, installation, input locations, checks, tests |
| [docs/ALGORITHM.md](docs/ALGORITHM.md) | what the product is and how each step computes it |
| [docs/RUNNING.md](docs/RUNNING.md) | running a year: the commands, the year driver, resources |
| [docs/TRAINING_LABELS.md](docs/TRAINING_LABELS.md) | the training labels (delivered as a separate archive) |
| [CHANGELOG.md](CHANGELOG.md) | every change since the 2026 development started, with evidence |

## Layout

| Path | What |
|---|---|
| `model/` | the algorithm: inputs and QA (`BandDayFile`, `CompositeDayFile`, `LowSun`), metrics (`Metrics`, `CMirrorMetrics`), training (`TrainingManifest`), prediction and rules (`VcfPredict`, `BlendA`, `V4r`, `RuleTile`, `CompactForest`), the year driver (`YearRun`), input locations (`Settings`) |
| `view/` | command-line entry points: `runYear`, `trainingManifest`, `fitAltModel`, `vcfPredict`, `makeMetrics` |
| `model/tests/` | unit tests |
| `docs/` | documentation and the predictor lists (`predictors_tree.txt`, `predictors_bare.txt`) |
| `environment.yml`, `environment-optional.yml` | conda environments |

Testing only, not part of the production chain: the GPU Monte Carlo predictor search (`model/MonteCarloSim*.py`,
`model/MasterTraining.py`, `model/Trial.py`, `model/TrainingType.py`, `view/monteCarloSim*.py`), the MOD09A1 input
(`model/ProductTypeMod09A.py`, `makeMetrics.py --productType`) and the XGBoost / HGB models (`fitAltModel.py --algo xgb|hgb`).

## Quick start

```bash
conda env create -f environment.yml && conda activate modis_vcf
export PYTHONPATH=<parent folder of modis_vcf>
export VCF_MOD44_DIR=<MOD44C folder> VCF_MOD44W_DIR=<MOD44W folder>
python -m modis_vcf.view.runYear run --spec spec.json          # docs/RUNNING.md
```

## Contributors

- Mark L. Carroll, mark.carroll@nasa.gov
- Roger L Gill, roger.l.gill@nasa.gov
- Savannah L Strong, savannah.strong@nasa.gov
- Jordan Alexis Caraballo-Vega, jordan.a.caraballo-vega@nasa.gov

## References

- [MODIS VCF user manual](https://modis.gsfc.nasa.gov/data/dataprod/mod44.php)
