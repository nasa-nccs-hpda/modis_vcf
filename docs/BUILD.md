# Building the environment (modis_vcf v1.8.0)

The code is pure Python. It needs no container, no GPU and no other repository.

## 1. Environment

conda-forge only (`environment.yml`, channel `nodefaults`, so the Anaconda `defaults` channel is never used):

```bash
conda env create -f environment.yml          # creates the environment "modis_vcf" (about 2.2 GB, 5 min)
conda activate modis_vcf
```

The versions are pinned to those the v1.5.x products were made with. **scikit-learn must stay 1.2.0**: fitted forests and their
predictions depend on its version. The only difference from the build our products came from is GDAL 3.8.5 instead of 3.8.4 (3.8.4 no
longer solves on conda-forge); the exactness proof below covers it. Since v1.7.1 the environment also has **h5py 3.13.0**, which
writes the Collection 7 product file. It builds on the HDF5 1.14.3 that GDAL already brings, and adding it changes no other package.
An environment built for v1.7.0 can take it in place: `conda install -n modis_vcf -c conda-forge --override-channels
--freeze-installed h5py=3.13.0 hdf5=1.14.3`.

If conda is a shared installation you cannot write to, set a writable package cache first: `export CONDA_PKGS_DIRS=<writable dir>`.

**Always run inside the activated environment** (`conda activate`, or `conda run -n modis_vcf ...` in batch jobs). Activation sets
`PROJ_DATA` and `GDAL_DATA`; calling the environment's `python` directly without it leaves PROJ unable to open its database, and map
writing fails with `PROJ: proj_create_from_name: Open of .../share/proj failed` (found in our own clean-environment test).

Optional (testing only, not used by the production chain): `conda env update -n modis_vcf -f environment-optional.yml` (xgboost).
The GPU Monte Carlo predictor search (`model/MonteCarloSimGpu.py`, `view/monteCarloSimGpu*.py`) needs NVIDIA RAPIDS (cuml, cudf,
cupy 23.12) on a GPU node and is not covered here.

## 2. Install the code

The package is imported as `modis_vcf`, so the repository folder must be named `modis_vcf`, and its parent folder must be on
`PYTHONPATH`:

```bash
git clone <repository> /opt/vcf/modis_vcf
export PYTHONPATH=/opt/vcf
```

## 3. Input locations

Set in `model/Settings.py`; override with environment variables:

| Variable | Data | Layout |
|---|---|---|
| `VCF_MOD44_COLLECTION` | which MOD44CH / MOD44CQ collection is read: `7` (**required for v1.8.0**) or `61` (default, v1.7.1) | - |
| `VCF_MOD44_DIR` | MOD44CH and MOD44CQ | Collection 7: `<dir>/<year>/MOD44C[HQ].A<year><doy>.<tile>.007.*.h5`; Collection 6.1: `<dir>/<year>/MOD44C[HQ].A<year><doy>.<tile>.061.*.hdf`; a year needs its own folder and the next year's |
| `VCF_MOD44W_DIR` | MOD44W land/water mask (Collection 6.1) | `<dir>/<year>/001/MOD44W.A<year>001.<tile>.061.*.hdf` |
| `VCF_MOD09A1_DIR` | MOD09A1 (optional input, testing only) | `<dir>/<year>/*.hdf` |
| `VCF_OUTPUT_COLLECTION` | collection number in the product file name and `GranuleID` (three digits) | `071` (default, this code's product) or e.g. `007` for an operational MODAPS run |

Command-line options (e.g. `vcfPredict.py --modisDir`) still override them.

## 4. Check

```bash
python - <<'EOF'
from osgeo import gdal
import modis_vcf.model.YearRun, modis_vcf.view.vcfPredict, modis_vcf.model.CompactForest
assert gdal.GetDriverByName('HDF4') is not None, 'GDAL has no HDF4 driver'
print('ok', gdal.__version__)
EOF
python -m modis_vcf.view.runYear --help
```

## 5. Unit tests

`model/tests/` (`python -m unittest modis_vcf.model.tests.<test_file>` with the parent of `modis_vcf` on `PYTHONPATH`). Eight test files read inputs at NCCS
paths (`/css/...`, `/explore/...`): `test_BandDayFile`, `test_CompositeDayFile`, `test_Metrics`, `test_MonteCarloSim`,
`test_ProductTypeMod09A`, `test_ProductTypeMod44`, `test_ProductTypeMod44W`, `test_VcfPredict`. Point them at your copies of those
granules, or skip them. The others run anywhere.

## 6. The Collection 7 product file and its corrections

Since v1.7.1 the map step writes one HDF5 file per tile and year, `MOD44B.A<yyyy>065.<tile>.071.<yyyydddhhmmss>.h5`
(`model/Mod44bH5.py`; collection `071` since v1.8.0, set by `VCF_OUTPUT_COLLECTION`, so the files are not confused with the
MODAPS C7 granules, which use `007`; `VersionID` and the DOI keep the Collection 7 values). Its layout matches the MODAPS C7 granules made with the C code and API-Trans (HDF-EOS5 groups, the 7 fields,
types, chunks, shuffle + gzip 4, field and root attributes, `StructMetadata.0`). No API-Trans step is needed: MODAPS confirmed it is
optional. The field values are those of the GeoTIFFs: this code's models and rules. A file for the same tile and year therefore
matches a C-code granule in layout, not in pixel values. Three things differ from the C-code granules, on purpose:

- **Corrected `valid_range` of the SD layers: 0-5000** (SD x 100 of values in 0-100, which cannot exceed 50). The C code declares
  0-275, and its own values exceed that: up to 2841 on C7 h08v05 2019.
- **Out-of-projection pixels are flagged.** The C code predicts on the fill inputs there and writes an ordinary-looking constant
  (C7 2019 h00v08 / h35v08: tree 15, non-vegetated 85 on 99.85 % of those pixels). It never uses its declared `_FillValue` 253.
  This code writes 250 there, as in the GeoTIFFs, and the three percent layers declare it as `OutOfProjection_Value = 250`, next to
  `Water_Value = 200`. `_FillValue` stays 253.
- **Run metadata are this code's own:**
  - `PGEVersion` and `LocalVersionID` give the modis_vcf version;
  - `ProcessingEnvironment` is the host;
  - `InputDataProducts` lists the MOD44CQ and MOD44CH composites, the model files and the MOD44W file actually read. With
    Collection 6.1 inputs these are `.061` `.hdf` names, where the C7 granules list C7 `.h5` names. On polar tiles fewer composites
    appear (e.g. 21 + 21 on h17v01), because only the composites that exist are listed;
  - the G-ring and bounds are copied from the first MOD44CQ, as the C code does (C7 inputs: the bounds from its root attributes
    `geospatial_lat_max` / `_lat_min` / `_lon_max` / `_lon_min`, the MODLAND tile table only as the fallback).

  Any root attribute can be set from the command line, e.g. `vcfPredict.py --h5Attr PGEVersion=7.0.2`, or through `h5Attrs` in a
  `runYear` spec. The QA attributes keep the C7 placeholders ("Passed", 100 % good quality).

`--format geotiff` or `--format both` (spec `"mapFormat"`) also writes the 7 GeoTIFFs, as before v1.7.1.

## 7. Proof that this environment gives our products

On 2026-10-20 the v1.5.2 code in this environment (no container, no other repository) reproduced the global 2019 v1.5.1 run on four
test tiles (h12v09, h17v01, h16v02, h20v09): composites byte for byte, metrics and all 7 map layers by value, the training splits of
those tiles, and the three models byte for byte. See `CHANGELOG.md` (v1.5.2).

On 2026-10-22 the v1.6.0 code in this environment (no container) reproduced the global 2019 v1.6.0 run, with its fitted models, on
h12v01 (a v01 tile, where the winter fill for missing inputs acts) and h12v09 (a blend tile): composites byte for byte (V4r composites
too), metrics and all 7 map layers by value. See `CHANGELOG.md` (v1.6.0).
