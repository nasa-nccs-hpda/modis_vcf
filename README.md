# MODIS Vegetation Continuous Fields

## Overview

Proportional estimates of cover are developed from global training data derived using 
high-resolution imagery. The training data and phenological metrics are used with a 
regression tree to derive percent cover globally. The model is then used to estimate 
areal proportions of life form, leaf type, and leaf longevity. The current V004 
collection of the yearly MODIS Vegetation Continuous Fields (VCF) product contains 
only percent tree cover, with the other layers to follow in later releases.

The data layers in the VCF product are generated on an annual basis from monthly 
composites of 500-m Surface Reflectance data. Compositing is based on the second 
darkest albedo to remove clouds and cloud shadow. For more information refer to
the [MODIS VCF user manual](https://modis.gsfc.nasa.gov/data/dataprod/mod44.php).

## Installation

The following library is intended to be used to accelerate the production 
development of MODIS VCF products. All dependencies have been housed inside
OCI Docker containers that can be directly used with Docker, Singularity, Podman,
among others.

Note: PIP installations do not include CUDA libraries for GPU support. 
Make sure NVIDIA libraries are installed locally in the system if not using conda/mamba.

## Production Container

```bash
module load singularity
singularity build --sandbox /lscratch/$USER/container/modis-vcf docker://nasanccs/modis-vcf:latest
```

## Development Container

```bash
module load singularity
singularity build --sandbox /lscratch/$USER/container/modis-vcf-dev docker://nasanccs/modis-vcf:dev
```

## Fitting and prediction

Version 1.2.0: both layers use a Random Forest of 60 trees (`max_features='sqrt'`, seed 0), fitted on the
bit-15 QA metrics, and both are combined with the MEDIAN of the trees. The non-vegetated training set adds
10,000 rows labelled 100 on the h16v02 (Greenland) ice sheet, on a regular grid at least 10 km inside the MOD44C
ice edge, so the model learns what that ice looks like with its real inputs. The tree cover model and the code
are unchanged from 1.1.0.
(Version 1.1.0 used the same recipe without the added rows. Version 1.0.0 used 30 trees, with the MEDIAN for
tree cover and the MEAN for non-vegetated.)
Tree cover uses its own predictors; non-vegetated uses the C code's 21 bare predictors
(listed in the version's predictor file).

```bash
python view/fitAltModel.py --algo rf --nEstimators 60 --maxFeatures sqrt --randomState 0 \
    --trainingType pcttree --predictorFile <predictors.txt> \
    --xTrainPath xTrain.parq --yTrainPath yTrain.parq -o <modelDir>
```

`view/vcfPredict.py` writes `Percent_Tree_Cover`,
`Percent_NonVegetated` and `Percent_NonTree_Vegetation`, plus the two SD layers
`Percent_Tree_Cover_SD` and `Percent_NonVegetated_SD`.
Each is a single-band LZW GeoTIFF. As in the historical MOD44B, the three percent layers are
uint8 (0-100, water 200, out of projection 250, nodata 253) and the SD layers are uint16.

- `--tcAgg` / `--nvAgg` set how each layer's trees are combined: `mean` (the same as
  `predict()`), `mid20` (the mean of the middle 20) or `median`. Defaults are `mean`. The
  production setting (v1.1.0 and v1.2.0) is `--tcAgg median --nvAgg median`; v1.0.0 used `--tcAgg median --nvAgg mean`.
- The SD layers are the population SD of the trees, stored as round(100 x SD), with 0 on
  water and out-of-projection. Use `--noSd` to skip them. Models without ensemble members
  (XGBoost) get no SD and support only `mean`.
- A process needs about 45 GB of memory per tile with 30-tree models and about 80 GB with the 60-tree models, and most of the run time is spent reading
  the metrics.

## Additional Projects

### Validation of MODIS VCF training with Very High Resolution Imagery

Below is the documentation to reproduce the validation of MODIS VCF training
data using Very High Resolution imagery. The following directions are meant to 
assist in reproducing these steps for a future validation sceneario. The process
is not fully automated given the need to select based on operator experience, the
best scenes to work with. This documentation is work in progress.

## Contributors

- Mark L. Carroll, mark.carroll@nasa.gov
- Roger L Gill, roger.l.gill@nasa.gov
- Savannah L Strong, savannah.strong@nasa.gov
- Jordan Alexis Caraballo-Vega, jordan.a.caraballo-vega@nasa.gov

## Contributing

Please see our guide for contributing to modis_vcf.

## References

- [MODIS VCF user manual](https://modis.gsfc.nasa.gov/data/dataprod/mod44.php)
