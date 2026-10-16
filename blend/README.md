# blend/: the v1.4.0 steps outside the package

v1.4.0 = v1.3.0 + **LE0** (low-sun bright substitution, in `model/BandDayFile.py`, switched on with `VCF_LOWSUN_T=70 VCF_LOWSUN_LE0=1`,
with tree and bare models refit on the substituted metrics) + **blend A** (tree layer, MODIS rows v06-v11 = 30°N-30°S) + **BP10**
(all tiles). These scripts are the code that made the 2019 v1.4.0 maps (3 layers; no SD layers yet). They run as standalone
scripts with the `modis_vcf` package on the path. In v1.4.1 the V4r composite and the blend are planned to move into the package's own
composite and map steps.

## Order for one year

Since v1.4.1 every script that reads or writes a year's files takes a required `--year` (`blend_year.py`, `map_predict.py`,
`rawbare.py`, `bp10_apply.py`, `recompute_split_onepass.py`); none defaults to 2019.

1. **LE0 product** (package): metrics, training splits and model fits as v1.3.0, in low-sun mode le0 (the default since v1.4.1;
   v1.4.0 set `VCF_LOWSUN_T=70 VCF_LOWSUN_LE0=1`); then the
   maps (`view/vcfPredictQueue.py` / `VcfPredict`).
2. **Raw bare**: `rawbare.py ... --year Y` per tile (the bare model's uncapped value).
3. **V4r inputs** (blend tiles only, **low-sun mode off**: the partner model was trained without the substitution; since v1.4.1 pass
   `--lowSun off` to these scripts, because the package default is now le0; the cache guard refuses to mix modes on one root):
   `days_year.py` -> `v4r_year.py` (V4r composites + `CUR_ZS` cloud counts) -> `warm_metrics.py`.
4. **Partner model** (once per training set): `recompute_split_onepass.py` -> `accept_split.py` -> `view/fitAltModel.py` (V4R60: RF,
   60 trees, MEDIAN).
5. **Partner map**: `check_map_names.py` (for a year other than 2019) -> `map_predict.py --year Y`.
6. **Blend**: `blend_year.py --clear <LE0 maps> --partner <V4R60 maps> --counts <counts> --rawBare <raw bare>` on the blend tiles (the 2019 product also used `--mask` with the V1-V4C tree maps, which only drops pixels invalid there)
   (`gate_common.py` holds the weights: w = cloud weight (`CUR_ZS` >= 5 of 8 periods) x cover weight (ramp 20-50 on the clear tree)).
7. **BP10**: `bp10_apply.py --year Y` on every tile (blend output for the blend tiles, LE0 maps elsewhere). Where raw bare = 100 and tree is 1-9:
   tree 0, bare 100, non-tree 0.

Each script's docstring gives its arguments and its built-in checks.
