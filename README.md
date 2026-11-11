# MODIS Vegetation Continuous Fields (modis_vcf)

Python code for the MODIS Vegetation Continuous Fields product (MOD44B, Terra): yearly percent tree cover, percent non-vegetated
(bare) and percent non-tree vegetation for every MODIS land tile on the 250 m sinusoidal grid, with SD, Quality and Cloud layers,
written as one Collection 7 HDF5 file per tile and year (`docs/RUNNING.md`). It
replaces the operational C code (PGE61) and reads the same inputs, MOD44CH / MOD44CQ (PGE72) and MOD44W.

Version: **v1.8.0** (see `CHANGELOG.md`). v1.8.0 requires the **Collection 7 MOD44CH / MOD44CQ inputs** (`.h5`):
set `VCF_MOD44_COLLECTION=7` ([docs/RUNNING.md](docs/RUNNING.md)); the code default, Collection 6.1, reproduces v1.7.1.

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
| `model/` | the algorithm: inputs and QA (`BandDayFile`, `CompositeDayFile`, `LowSun`), metrics (`Metrics`, `CMirrorMetrics`), training (`TrainingManifest`, `PooledSplit`), prediction and rules (`VcfPredict`, `CloudFallbackBlend`, `CloudFallback`, `BarePriority`, `RuleTile`, `CompactForest`), the year driver (`YearRun`), input locations (`Settings`) |
| `view/` | command-line entry points: `runYear`, `trainingManifest`, `buildPooledSplit`, `fitAltModel`, `vcfPredict`, `makeMetrics` |
| `tests/`, `pytest.ini` | unit tests (`pytest` from the repository root; docs/BUILD.md section 5) |
| `docs/` | documentation and the predictor lists (`predictors_tree.txt`, `predictors_bare.txt`) |
| `environment.yml` | conda environment |

Only the production chain is on this branch. The testing-only modules of earlier versions (the Monte Carlo predictor search on
CPU and GPU, the MOD09A1 input, the XGBoost / HGB fits and `environment-optional.yml`) are kept at tag `optional-modules-v1.8.0`.

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
