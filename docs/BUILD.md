# Building the environment (modis_vcf v1.6.0)

The code is pure Python. It needs no container, no GPU and no other repository.

## 1. Environment

conda-forge only (`environment.yml`, channel `nodefaults`, so the Anaconda `defaults` channel is never used):

```bash
conda env create -f environment.yml          # creates the environment "modis_vcf" (about 2.2 GB, 5 min)
conda activate modis_vcf
```

The versions are pinned to those the v1.5.x products were made with. **scikit-learn must stay 1.2.0**: fitted forests and their
predictions depend on its version. The only difference from the build our products came from is GDAL 3.8.5 instead of 3.8.4 (3.8.4 no
longer solves on conda-forge); the exactness proof below covers it.

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
| `VCF_MOD44_DIR` | MOD44CH and MOD44CQ (Collection 6.1) | `<dir>/<year>/MOD44C[HQ].A<year><doy>.<tile>.061.*.hdf`; a year needs its own folder and the next year's |
| `VCF_MOD44W_DIR` | MOD44W land/water mask (Collection 6.1) | `<dir>/<year>/001/MOD44W.A<year>001.<tile>.061.*.hdf` |
| `VCF_MOD09A1_DIR` | MOD09A1 (optional input, testing only) | `<dir>/<year>/*.hdf` |

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

## 6. Proof that this environment gives our products

On 2026-10-20 the v1.5.2 code in this environment (no container, no other repository) reproduced the global 2019 v1.5.1 run on four
test tiles (h12v09, h17v01, h16v02, h20v09): composites byte for byte, metrics and all 7 map layers by value, the training splits of
those tiles, and the three models byte for byte. See `CHANGELOG.md` (v1.5.2).
