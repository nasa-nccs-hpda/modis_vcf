# Changelog: MODIS-VCF Python code (modis_vcf)

Changes to the Python MOD44B (PGE61) code since the start of the 2026 development work, newest first. Each entry says what changed,
why, and **whether product values change**. "Exact" means proven identical to the previous version (dailies and composites byte for
byte, metrics and map layers by value) on the named tiles. Commits are on branch `mcarrol2/v1.2.0` of
`nasa-nccs-hpda/modis_vcf`, which starts from `develop` at `b83e302` (2025-07-18). Evidence is in the development reports
(`docs/reports/` snapshot, and the project's `Reports/` folder for later ones).

Version names: v1.0.0 is the first product version. Later versions are cumulative. A version tag marks the code; the models of a
version are fitted with the recipe given here. Maintained with every code commit (see the end of this file).

---

## Unreleased: v1.4.1 (code tidy-up and production benchmark; in progress)

Goal: the v1.4.0 product from one package, in one driver per year, with exact speed-ups, then a benchmark against the C code (time,
memory, disk). Every item below is **exact** against v1.4.0 unless stated. v1.4.1 is tagged only when the benchmark passes.
Plan: `Reports/V141_Code_Tidyup_And_Full_Retrain_Plan.md`.

- **One driver per year** (P5, `f24f68a`, 2026-10-17). `model/YearRun.py` + `view/runYear.py`: metrics -> splits -> fits (+ compact
  copy) -> maps as a task graph shared by the nodes through a status folder (claims, slots, memory floors, fit hosts, retry of maps,
  STOP markers), with per-task timing (wall, CPU, peak RSS, bytes) in `timing/tasks.csv`. A 2019 year built from scratch reproduced
  v1.4.0's splits, models (every stored test row) and maps (4 tiles, 28 layers). Evidence: `Reports/V141_S5_P5_Results.md`.
- **Training rows built per tile in parallel** (S5, `da96b2a`, 2026-10-17). `TrainingManifest.build(workers=N)`,
  `trainingManifest.py --workers`. Byte-identical to the serial build; tree split of 63 tiles in 824 s (peak 129.5 GB with 6 workers).
- **Training samples as key manifests** (P4, `954045b`, 2026-10-16). `model/TrainingManifest.py`, `view/trainingManifest.py`: a split
  becomes a manifest of pixel keys + labels, and a split is rebuilt from a manifest and a metrics root. Keys, labels and row index equal
  v1.4.0's on every row; predictors equal on every tile with metrics on disk. Fresh-sample mode not yet included.
- **Blend A, raw-bare cap and BP10 inside the map step** (P3 + S3, `79cb19f`, 2026-10-16). `vcfPredict` makes the v1.4.0 product in
  one call (`--partner ... --counts`; BP10 on by default, `--noBp10`); `model/BlendA.py`. The raw non-vegetated value is reused from the
  same prediction (no second bare prediction). The four standalone blend scripts are deprecated (kept one version). Equal to the
  delivered v1.4.0 product on h12v09.
- **Year required everywhere** (P2, `bce3a08`, 2026-10-16). `blend/rawbare.py`, `bp10_apply.py`, `recompute_split_onepass.py`,
  `view/rfFitFromPge61.py` no longer assume 2019.
- **Low-sun substitution (LE0) is the package default** (P1, `d8c3ef9`, 2026-10-16). `model/LowSun.py`: modes `le0` (default),
  `s70`, `off` (= v1.3.0); `--lowSun`, `--lowSunCheck` on the CLIs; the `VCF_LOWSUN_*` variables still work for one version
  (deprecated). Each metrics root gets a stamp (`lowsun.json`) and per-day substitution counts (`lowsun_counts.csv`), and a root built
  in another mode is refused. **Note:** 34 of v1.4.0's 69 tiles were mapped from metrics built without the substitution (found
  2026-10-16, Q6); with P1 every new run applies it. On those tiles, wherever low-sun inputs occur (e.g. h20v09: 653 composite pixels in
  period 2019353), new maps differ from the delivered v1.4.0 maps, by design; h12v09 has no such inputs in 2019 and is unchanged.
- **Quality / Cloud layers read the stored composite masks** (S12, `b0eaf05`, 2026-10-14). No composite rebuild per map; a map no longer
  needs write access to the metrics folder. Fixed Cloud flag stored as `-cloudfb.npy` (old `-cloud.npy` files, which hold the inverted
  flag, are never read).
- **`fitAltModel --minSamplesLeaf`** (`a13d107`, 2026-10-14). Default 1, unchanged. Leaf sizes 5-100 were tested and not adopted
  (values compressed).
- **Compact random-forest models** (S9, `ac6d680`, 2026-10-14). `model/CompactForest.py`: the same trees as flat arrays with a numba
  predictor, 6.4x smaller (29 GB -> 4.6 GB); thresholds stored so every float32 input takes the same branch. Identical predictions per
  tree, MEDIAN and SD; a map 712 s / 81 GB -> 377 s / 22 GB. `VCF_COMPACT_MMAP=1` memory-maps a model.
- **Quality and Cloud layers; Cloud flag fixed** (`986892b`, 2026-10-13). `Quality-<tile>-<year>.tif` (per-period fallback) and
  `Cloud-<tile>-<year>.tif` (fallback with no clear day), period 1 in bit 0. The per-period Cloud flag was inverted and is now the C
  code's definition (equal to the C code's Cloud on every bit of 5 tiles). Cover layers unchanged.

Planned before the benchmark (not committed): per-tile clean-up of intermediates in the driver (S8, in test 2026-10-17), dailies kept
in memory (S6), V4r composites and counts in the composite pass (S7), and the blend partner trained in the year run (decision D4,
2026-10-17). Plan: `Reports/V141_S6_S7_S8_Plan.md`.

## v1.4.0 (tag on `53eb115`, 2026-10-13)

**Product change.** v1.3.0 + low-sun substitution (LE0, refit) + blend A + BP10. Maps: 2019, 69 tiles, 3 cover layers.

- **Low-sun bright substitution, LE0** (`b3a7142`). With `VCF_LOWSUN_T=70 VCF_LOWSUN_LE0=1`, Band_1-7 and Band_31 inputs with solar
  zenith above 70 or at or below 0 become 16000 (the intent of PGE61 `fix_bands.c`); models refitted on these inputs. Fixes the
  Greenland ice interior and the Iceland / h17v01 weak years. With the variables unset the code is v1.3.0.
  `VCF_ZENITH_NDVI=0` switches the zenith-only NDVI rule off. Evidence: `Low_Sun_Bright_Substitution_S70LE0_Results.md`,
  `Bundle_PreDecision_Checks_Results.md`.
- **Blend A, raw-bare cap, BP10 as standalone scripts** (`53eb115`, `blend/`). Tree cover on rows v06-v11 (30N-30S) blended with a
  partner model (V4R60) trained on V4r composites (a cloud-adapted composite), weighted by a cloud count and the clear model's cover;
  the sum cap uses the raw non-vegetated value; BP10 sets tree 0 / non-vegetated 100 where raw non-vegetated is 100 and tree is 1-9.
  Evidence: `BlendA_Full_Product_v130_Results.md`, `BlendA_Other_Years_Results.md`, `Bare_Priority_Cap_Pixel_Count.md`. Moved into the
  package in v1.4.1 (P3).

## v1.3.0 (tag on `7036b6a`, 2026-10-10)

**Product values same as v1.2.0** (speed-ups exact); one fix enables other years.

- **Parallel composites** (`f3deac7`, `7036b6a`). `Metrics.compositeWorkers`, `makeMetrics.py --workers`: a band's composite periods
  built in processes (GDAL serialises HDF4 reads within one process). Metrics 799 s -> 376 s per tile with 8 workers. Exact.
- **Optional MOD09A1 input** (`9e70feb`). MOD09A1 QA rule and cloud mask; `makeMetrics.py --productType`, `--mod09Dir`. Tested on
  h09v05 2019, not adopted as the product input. MOD44C results unchanged.
- **QA and cloud masks built once per day** (`ca764df`). Not once per band: HDF reads per tile 575 -> 253. Exact.
- **Metric GeoTIFFs written once** (`bb2552e`). No per-band `FlushCache`: metric writes 417 s -> 33 s per tile. Pixel values,
  metadata and layout identical (files ~48 KB smaller).
- **Predictor dates mapped to the metrics' year** (`0a5d21a`). `Metrics.getMetricFromRf` did not do what its comment said, so no model
  could run on another year. Identity for 2019. Used for the 2010 and 2014 products.

## v1.2.0 (tag on `45edf62`, 2026-10-07)

**Model change only** (code = v1.1.0). 10,000 non-vegetated training rows labelled 100 on the h16v02 Greenland ice sheet (regular grid
at least 10 km inside the MOD44C ice edge); non-vegetated model refitted with the v1.1.0 recipe. h16v02 interior ice below 100:
8.7% -> 0.007%; held-out labels within seed noise.

## v1.1.0 (tag on `920d06f`, 2026-10-04)

**Product change.** Random forests of 60 trees for both layers, combined with the MEDIAN of the trees (`--tcAgg median --nvAgg
median`); held-out labels better for both layers (`Candidate_60Trees_Median_Results.md`).

- **MODIS sinusoidal grid in the outputs** (`920d06f`). The GeoTIFFs had no geotransform and a WGS84-ellipsoid projection; they now
  carry the MODIS sphere and the tile's geotransform (checked against C6.1 MOD44B granules).
- **uint8 percent layers, uint16 SD layers** (`0298e4e`). As the historical MOD44B (water 200, out of projection 250, no data 253);
  values unchanged.

## v1.0.0 (tag on `16d6474`, 2026-10-02)

**The first product version**: bit-15 QA, random forests (30 trees, sqrt) for tree cover (MEDIAN) and non-vegetated (MEAN), rounded
values, SD layers, Greenland zenith-only NDVI, high-latitude rule on v00 tiles. Definition: `Product_Versions_and_Work_Map.md`.

- **Rounding and v00 rule** (`16d6474`). Values rounded (`floor(v + 0.5)`, as the C code) instead of truncated, which biased both
  layers down by about half a point. On v00 tiles every land pixel is tree 0, non-vegetated 100.
- **Zenith-only NDVI on 6 Greenland tiles** (`13773db`). NDVI set to 0 in composite periods with no day of 0 < solar zenith < 65.
- **Member pass, aggregation and SD layers** (`6684b88`, `f2d9190`, `92c7edc`, `615557d`). Each ensemble member predicted once;
  MEAN / mid20 / MEDIAN per layer (`--tcAgg`, `--nvAgg`); `Percent_Tree_Cover_SD` and `Percent_NonVegetated_SD` as round(100 x SD)
  (`--noSd` to skip). Unit tests; README section on fitting and layers.
- **C code's 21 non-vegetated predictors as the default** (`2778628`). The hand-written list had translated the C code's indices in
  the wrong layout (13 of 18 names were not the C code's predictors).
- **`fitAltModel.py`** (`8f42c83`). Random forest, XGBoost and HGB fits with the same inputs and outputs as `rfFitFromPge61.py`.
- **Predictions clamped to 0-100; layers sum to 100** (`2a37f9e`). Non-vegetated is reduced to absorb any overshoot; water and
  out-of-projection codes carried into non-tree vegetation.
- **CPU models** (`c69c32d`). cuML `RandomForestRegressor` replaced by scikit-learn models (no GPU on the processing nodes); parallel
  prediction in row blocks; `vcfPredictQueue` predicts many tiles with the models loaded once. A design change.
- **Python mirrors of the C code's derived metrics** (`e0e6478`). `CAttributes`, `CMirrorMetrics` reproduce the C code's 252 derived
  attributes exactly; metric files read once per tile.
- **QA and band fixes for C-code parity** (`1c3ac20`). Adjacency is state bit 6; the aerosol test is MOD44C state bit 15 (PGE72's
  overall QC flag; bits 7-8 overlap other fields and rejected 29-50% of good pixels); zenith cutoff 65; Bands 1-7 clamp removed (the C
  code's never fires); Band_31 clamp at 70; `BuildTraining` honours `trainingDir`.

## Documentation snapshots (no code change)

`docs/reports/` snapshots of the development reports: `335ceb2`, `fe56c5f`, `88baf42`, `0959749`, `f99704a`, `3f8097a`, `3ec71c1`,
`3e4f766`, `41556d0`, `6852581` (2026-09-30 to 2026-10-07).

---

## Keeping this file current

Every code commit on the delivery branch adds its entry here **in the same commit**, under "Unreleased" (or the version being built):
what changed, why, whether values change (exact / product change / model change), the commit's evidence report. When a version is
tagged, the "Unreleased" heading becomes the version heading with its tag commit and date.
