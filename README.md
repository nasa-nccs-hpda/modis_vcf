# MODIS Vegetation Continuous Fields (modis_vcf)

[![DOI](https://zenodo.org/badge/710328106.svg)](https://doi.org/10.5281/zenodo.24174799)

Python code for the MODIS Vegetation Continuous Fields product (MOD44B, Terra): yearly percent tree cover, percent non-vegetated
(bare) and percent non-tree vegetation for every MODIS land tile on the 250 m sinusoidal grid, with SD, Quality and Cloud layers,
written as one Collection 7 HDF5 file per tile and year (`docs/RUNNING.md`). It
replaces the operational C code (PGE61) and reads the same inputs, MOD44CH / MOD44CQ (PGE72) and MOD44W.

Version: **v1.9.0** (see `CHANGELOG.md`). It reads the **Collection 7 MOD44CH / MOD44CQ inputs** (`.h5`, the default collection).
The tree cover models are fitted every year. Non-vegetated uses one delivered model for every year, the **pooled bare model**
(2000-2023), which comes as a separate archive with its checksums ([docs/RUNNING.md](docs/RUNNING.md)).

**Files needed to run, not in git:** the training labels and the pooled bare model are attached to the
[v1.9.0 release](https://github.com/nasa-nccs-hpda/modis_vcf/releases/tag/v1.9.0) (`VCF_training_labels_7.0.1.tar.gz`, 83 MB; `pctbare_pooled_2000-2023_v1.9.0.tar.gz`, 1.8 GB;
each with a `.sha256`). The tree labels are needed every year (tree cover is fitted each year); the model is needed for the
non-vegetated layer. How to check and unpack them: [docs/RUNNING.md](docs/RUNNING.md) and [docs/TRAINING_LABELS.md](docs/TRAINING_LABELS.md).

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
python -m pytest                                               # unit tests, about a minute (docs/BUILD.md)
python -m modis_vcf.view.runYear run --spec spec.json          # docs/RUNNING.md
```

## Citing

Version v1.9.0 is archived on Zenodo with the DOI [10.5281/zenodo.24174799](https://doi.org/10.5281/zenodo.24174799).
Please cite that DOI when you use this code; the Zenodo record gives the citation in the usual formats.

## Contributors

- Mark L. Carroll, mark.carroll@nasa.gov
- Roger L Gill, roger.l.gill@nasa.gov
- Savannah L Strong, savannah.strong@nasa.gov
- Jordan Alexis Caraballo-Vega, jordan.a.caraballo-vega@nasa.gov

## References

- [MODIS VCF user manual](https://modis.gsfc.nasa.gov/data/dataprod/mod44.php)
