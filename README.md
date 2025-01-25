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
