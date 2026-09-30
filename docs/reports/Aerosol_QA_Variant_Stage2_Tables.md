# Aerosol QA variant C (state bit 15): Stage 2 tables (auto-generated)

Generated 2026-09-29 16:59 by `scripts/stage2_summary.py` from `/explore/nobackup/people/mcarrol2/MOD44B_VCF_Development/FullRuns/aerosol_qa_variants_20260929`.
Tables only; no interpretation. Plan: `AI_code_sync_09152026/Reports/Aerosol_QA_Variant_Test_Plan.md`.

**Caveat:** MAE, bias and correlation against `v5.0.3samp` are **in-sample for B** (the models were trained
on these pixels with B metrics), so they measure how far C moves from what the models memorised, not which
screen is better.

## Stage 0: QA pass rates (% of valid pixels, pixel-weighted over available periods)

| Tile | Periods | Pass A (none) | Pass B (bits 7-8) | Pass C (bit 15) | Of A removed by B | Of A removed by C |
|---|---|---|---|---|---|---|
| h09v05 | 23 | 84.1 | 41.9 | 83.8 | 50.1 | 0.3 |
| h12v09 | 23 | 39.5 | 22.7 | 38.7 | 42.6 | 2.1 |
| h16v01 | 21 | 7.7 | 5.1 | 6.1 | 33.8 | 20.7 |

## Tree cover predicted from C metrics vs from B metrics (same models)

| Tile | Model | Tile mean B | Tile mean C | Mean C - B | SD of C - B | % pixels abs(C - B) <= 5 | % > 20 | MAE B (in-sample) | MAE C | Bias B | Bias C | Corr B | Corr C | Labels |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| h09v05 | baseline | 8.96 | 8.78 | -0.19 | 4.00 | 88.2 | 0.36 | 1.49 | 3.09 | -0.54 | -1.05 | 0.977 | 0.913 | 268003 |
| h09v05 | rf | 8.71 | 8.54 | -0.18 | 3.15 | 91.6 | 0.07 | 1.51 | 3.12 | -0.61 | -1.17 | 0.977 | 0.914 | 268003 |
| h09v05 | xgboost_tunedB | 8.05 | 7.98 | -0.07 | 1.95 | 97.2 | 0.00 | 3.65 | 3.77 | -1.37 | -1.30 | 0.882 | 0.875 | 267581 |
| h12v09 | baseline | 63.01 | 64.28 | 1.28 | 7.13 | 68.7 | 2.21 | 3.71 | 6.96 | -2.21 | -4.40 | 0.955 | 0.887 | 24541 |
| h12v09 | rf | 63.03 | 64.40 | 1.37 | 5.93 | 73.4 | 0.98 | 3.79 | 7.05 | -2.41 | -4.64 | 0.956 | 0.889 | 24541 |
| h12v09 | xgboost_tunedB | 62.76 | 64.16 | 1.41 | 4.76 | 81.7 | 0.55 | 8.76 | 8.85 | -5.91 | -6.17 | 0.855 | 0.855 | 24539 |
| h16v01 | baseline | 5.66 | 5.66 | 0.01 | 1.71 | 98.9 | 0.20 | 0.01 | 0.02 | 0.01 | 0.02 | - | - | 156403 |
| h16v01 | rf | 4.44 | 4.44 | 0.00 | 0.85 | 99.4 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | - | - | 156403 |
| h16v01 | xgboost_tunedB | 5.05 | 5.05 | -0.00 | 0.79 | 99.5 | 0.00 | 0.07 | 0.07 | 0.07 | 0.07 | - | - | 152971 |

## Metric shift B -> C at labelled pixels

Per metric file: median over its bands of the correlation between B and C, of mean abs(C - B) / SD(B),
of mean (C - B) / SD(B), and of the % of labelled pixels whose value changed. MOD44-NDVI.bin: % of all
elements changed (layout not decoded).

### h09v05

| Metric file | Bands | Median corr | Median MAE/SD | Median mean shift/SD | Median % changed |
|---|---|---|---|---|---|
| CGreenAmp15.tif | 1 | 0.924 | 0.238 | -0.125 | 92.8 |
| CGreenMax.tif | 2 | 0.846 | 0.294 | 0.066 | 75.6 |
| CGreenMean13.tif | 3 | 0.980 | 0.126 | 0.026 | 90.9 |
| CGreenMean16.tif | 3 | 0.991 | 0.094 | 0.019 | 97.5 |
| CGreenMean19.tif | 2 | 0.947 | 0.229 | 0.079 | 99.0 |
| CGreenMean46.tif | 2 | 0.799 | 0.405 | 0.084 | 91.9 |
| CSortedRank0.tif | 1 | 0.952 | 0.190 | -0.012 | 64.2 |
| CSortedRank1.tif | 1 | 0.949 | 0.203 | 0.025 | 67.4 |
| CSortedRank6.tif | 1 | 0.969 | 0.167 | 0.013 | 83.1 |
| CSortedRank7.tif | 1 | 0.969 | 0.081 | 0.060 | 56.8 |
| CThermMean13.tif | 6 | 0.991 | 0.088 | -0.006 | 80.9 |
| CThermMean16.tif | 2 | 0.982 | 0.134 | 0.058 | 93.2 |
| CThermMean19.tif | 3 | 0.993 | 0.063 | 0.038 | 98.8 |
| CUnsAmp15.tif | 2 | 0.901 | 0.279 | 0.103 | 95.4 |
| CUnsMean13.tif | 2 | 0.972 | 0.133 | 0.060 | 91.6 |
| CUnsMean16.tif | 4 | 0.993 | 0.081 | 0.016 | 97.5 |
| CUnsMean19.tif | 3 | 0.983 | 0.094 | -0.019 | 98.8 |
| CUnsMean46.tif | 2 | 0.991 | 0.075 | -0.012 | 76.3 |
| CUnsMean79.tif | 3 | 0.954 | 0.181 | -0.039 | 90.3 |
| UnsortedMonthlyBands.tif | 72 | 0.970 | 0.105 | 0.005 | 63.1 |
| MOD44-NDVI.bin | - | - | - | - | 58.0 (all elements) |

All bands of all .tif metrics (116): median corr 0.972, minimum corr 0.607, median MAE/SD 0.117, maximum MAE/SD 0.581.

### h12v09

| Metric file | Bands | Median corr | Median MAE/SD | Median mean shift/SD | Median % changed |
|---|---|---|---|---|---|
| CGreenAmp15.tif | 1 | 0.807 | 0.321 | -0.106 | 83.7 |
| CGreenMax.tif | 2 | 0.800 | 0.291 | -0.009 | 36.3 |
| CGreenMean13.tif | 3 | 0.983 | 0.076 | 0.004 | 63.4 |
| CGreenMean16.tif | 3 | 0.989 | 0.068 | 0.016 | 86.7 |
| CGreenMean19.tif | 2 | 0.911 | 0.290 | 0.101 | 91.9 |
| CGreenMean46.tif | 2 | 0.764 | 0.441 | 0.051 | 76.0 |
| CSortedRank0.tif | 1 | 0.786 | 0.463 | 0.410 | 54.4 |
| CSortedRank1.tif | 1 | 0.727 | 0.608 | 0.523 | 60.8 |
| CSortedRank6.tif | 1 | 0.945 | 0.189 | -0.011 | 55.3 |
| CSortedRank7.tif | 1 | 0.941 | 0.111 | 0.101 | 14.0 |
| CThermMean13.tif | 6 | 0.955 | 0.171 | -0.020 | 83.2 |
| CThermMean16.tif | 2 | 0.916 | 0.270 | 0.192 | 85.6 |
| CThermMean19.tif | 3 | 0.990 | 0.082 | 0.038 | 91.3 |
| CUnsAmp15.tif | 2 | 0.926 | 0.210 | -0.016 | 85.7 |
| CUnsMean13.tif | 2 | 0.958 | 0.115 | -0.012 | 24.8 |
| CUnsMean16.tif | 4 | 0.981 | 0.099 | -0.053 | 86.5 |
| CUnsMean19.tif | 3 | 0.976 | 0.108 | -0.061 | 90.2 |
| CUnsMean46.tif | 2 | 0.969 | 0.153 | -0.007 | 82.1 |
| CUnsMean79.tif | 3 | 0.933 | 0.125 | -0.075 | 36.3 |
| UnsortedMonthlyBands.tif | 72 | 0.942 | 0.097 | -0.035 | 16.1 |
| MOD44-NDVI.bin | - | - | - | - | 28.9 (all elements) |

All bands of all .tif metrics (116): median corr 0.946, minimum corr 0.636, median MAE/SD 0.106, maximum MAE/SD 0.720.

### h16v01

| Metric file | Bands | Median corr | Median MAE/SD | Median mean shift/SD | Median % changed |
|---|---|---|---|---|---|
| CGreenAmp15.tif | 1 | 0.962 | 0.012 | 0.003 | 1.0 |
| CGreenMax.tif | 2 | 0.997 | 0.002 | 0.001 | 0.8 |
| CGreenMean13.tif | 3 | 0.998 | 0.003 | 0.001 | 1.0 |
| CGreenMean16.tif | 3 | 0.998 | 0.002 | 0.001 | 1.0 |
| CGreenMean19.tif | 2 | 0.999 | 0.002 | 0.001 | 1.0 |
| CGreenMean46.tif | 2 | 0.998 | 0.002 | 0.000 | 0.4 |
| CSortedRank0.tif | 1 | 0.999 | 0.001 | 0.000 | 0.8 |
| CSortedRank1.tif | 1 | 0.999 | 0.001 | 0.001 | 0.9 |
| CSortedRank6.tif | 1 | 1.000 | 0.001 | 0.001 | 0.5 |
| CSortedRank7.tif | 1 | 0.693 | 0.009 | 0.000 | 0.1 |
| CThermMean13.tif | 6 | 0.999 | 0.002 | 0.000 | 1.1 |
| CThermMean16.tif | 2 | 0.986 | 0.005 | 0.002 | 1.1 |
| CThermMean19.tif | 3 | 0.999 | 0.003 | 0.002 | 1.1 |
| CUnsAmp15.tif | 2 | 0.982 | 0.006 | -0.000 | 1.1 |
| CUnsMean13.tif | 2 | 0.986 | 0.002 | 0.001 | 0.1 |
| CUnsMean16.tif | 4 | 0.999 | 0.003 | 0.000 | 1.1 |
| CUnsMean19.tif | 3 | 1.000 | 0.001 | 0.000 | 1.1 |
| CUnsMean46.tif | 2 | 0.983 | 0.005 | 0.001 | 1.1 |
| CUnsMean79.tif | 3 | 1.000 | 0.000 | 0.000 | 0.0 |
| UnsortedMonthlyBands.tif | 72 | 1.000 | 0.000 | 0.000 | 0.0 |
| MOD44-NDVI.bin | - | - | - | - | 5.8 (all elements) |

All bands of all .tif metrics (114): median corr 1.000, minimum corr 0.693, median MAE/SD 0.001, maximum MAE/SD 0.012.

