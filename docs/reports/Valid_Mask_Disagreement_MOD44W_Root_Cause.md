# Valid-pixel mask disagreement (open item 2): root cause found

**Status: root cause confirmed.** The MOD44W fill/unclassified region (`water_mask == 250`) that
accounts for the disagreement, pixel for pixel, on all four tiles is a genuine out-of-bounds area of
the MODIS sinusoidal projection at high latitude, not a data-quality artifact - confirmed by the
project lead's independent knowledge of the projection. The C-side mechanism was deliberately not
pursued: the project's goal is Python's own quality, not bit-for-bit agreement with C, so chasing why
C produces a value in a genuinely out-of-projection area is not useful work here. Written 2026-09-27,
in the regression-tree-methods experiment session (`Regression_Tree_Methods_RF_XGBoost_Plan.md`), as
a side investigation - no product code changed.

## Background

`README_START_HERE.md`, open item 2, from the 63-tile C-vs-Python run: "Valid-pixel masks differ on
some high-latitude tiles: C has values where Python has none on h12v01, h13v01, h21v01, h11v02.
Python's water and no-data handling there is not examined." That line had stood unexamined since
2026-09-20.

## What the figures show

Four full-tile PNGs (`figures/c_vs_python_mask_disagreement/<tile>_c_vs_python_valid_mask.png`),
built by a new script, `make_c_vs_python_mask_disagreement_figures.py`, classify every pixel of
`Percent_Tree_Cover` into four classes - neither valid, both valid, C valid only, Python valid only
- and render each tile at native 4800x4800 resolution:

| Tile | C valid, Python missing | Python valid, C missing | Both valid | Neither |
|---|---|---|---|---|
| h12v01 | 77.94% | 0.00% | 3.20% | 18.86% |
| h13v01 | 44.80% | 0.00% | 0.32% | 54.88% |
| h21v01 | 12.39% | 0.16% | 33.73% | 53.72% |
| h11v02 | 12.13% | 0.01% | 70.45% | 17.42% |

In all four tiles the "C valid, Python missing" pixels are not scattered - they form one contiguous
region bounded by a single straight diagonal line, in the same corner of the tile in every case. That
is the signature of a geometric/coordinate cause, not scattered noise from water or no-data handling.

## Root cause: MOD44W's fill code (250) accounts for the disagreement exactly

Python's tree-cover prediction reads a water mask for each tile/year from MOD44W
(`/css/modis/Collection6.1/L3/MOD44W-LandWaterMask/<year>/001/`, `VcfPredict._getMask`,
`MOD44W_DIR` in `VcfPredict.py`). That product's `water_mask` band uses 0 = land, 1 = water, and
**250 = fill / unclassified / out of projection** (paired with `seven_class == 250` and
`water_mask_QA == 10`).

Comparing the "C valid, Python missing" pixels against `water_mask` for each tile's actual 2019/day-1
MOD44W input gives an **exact match, down to the pixel count**:

| Tile | Disagreement pixels (C valid, Python missing) | `water_mask == 250` pixels |
|---|---|---|
| h12v01 | 17,957,806 | 17,957,806 |
| h13v01 | 10,320,859 | 10,320,859 |
| h21v01 | 2,855,346 | 2,855,346 |
| h11v02 | 2,794,969 | 2,794,969 |

Every single disagreement pixel in every one of the four tiles falls inside MOD44W's fill region, and
every fill-region pixel is a disagreement pixel - a complete explanation for this open item, not an
approximate correlation.

## Mechanism, and why the geometry looks the way it does

- **The MOD44W fill area is a real, expected artifact of the MODIS sinusoidal projection**, not a
  data problem to fix or a masking bug - confirmed independently (the project lead has direct
  knowledge of the projection geometry). The sinusoidal grid's tiles are rectangular in pixel space
  but the actual mapped/valid projection area within a tile shrinks at high latitude, which is exactly
  the clean straight-line boundary the four figures show: a real edge of the projection's valid
  extent, cutting across the tile at whatever angle that tile happens to sit at.
- **Correction (2026-09-27, later the same day): Python already special-cases this, correctly.** An
  earlier pass of this document read only part of `VcfPredict.py` and concluded fill "falls through
  unchanged" into a NaN prediction. Reading the full file shows `VcfPredict` already defines
  `MASK_OUT_OF_PROJ = 250` and `PRED_OUT_OF_PROJ = 250`, and `_maskPrediction` applies both the water
  rule (`mask == 1` -> 200) and the out-of-projection rule (`mask == 250` -> 250) in sequence. Checked
  directly against the real 63-tile-run output: `Percent_Tree_Cover` and `Percent_NonVegetated` do
  contain exactly 250 at exactly the disagreement pixel counts above (e.g. h12v01: 17,957,806 pixels
  at value 250, matching the disagreement count exactly) - this is a deliberate, working, unit-tested
  feature (`test_VcfPredict.py::testMaskPrediction`), not a gap. So Python is not producing "no value"
  in these pixels at all; it is correctly writing 250 to say so, and the `valid()` check used
  throughout this project's comparison scripts (`0 <= x <= 100`) correctly treats that as invalid,
  which is why it looked like a gap from the outside.
- **The real, remaining gap is one layer down.** `Percent_NonTree_Vegetation` is not predicted
  directly; `VcfPredict.runTileForYear` computes it as `max(0, 100 - Percent_Tree_Cover -
  Percent_NonVegetated)` and writes that, with no equivalent masking step. Where tree cover and
  non-veg are both 200 (water) or both 250 (out of projection), the formula produces flat **0**, not
  200 or 250 - confirmed against the real output (e.g. h12v01: 22,302,935 pixels are 0 in
  `Percent_NonTree_Vegetation`, within 133 of the 22,302,802 that are water or out-of-projection in
  `Percent_Tree_Cover`; the small remainder is ordinary land pixels where tree cover plus non-veg
  happens to reach 100 on its own). This is the actual
  code-level version of open item 4's "no 200 in non-tree vegetation over water," and it turns out to
  extend to the out-of-projection code too. See
  `NonTreeVeg_Water_OutOfProj_Code_Propagation_Plan.md` for the fix.
- **The C-side mechanism was not pursued, on purpose.** The project's goal is Python's own quality,
  not matching C pixel for pixel, so understanding why C emits a value in a genuinely out-of-projection
  area is not useful work here.

## Does this fully address the open item?

**Yes, and more cleanly than the first pass of this document said.** The disagreement is fully
explained (an exact, pixel-for-pixel match to MOD44W's fill code, in all four tiles), the fill region
is confirmed as a real sinusoidal-projection boundary, and Python's `Percent_Tree_Cover`/
`Percent_NonVegetated` layers already handle it correctly and deliberately (code 250), not as a silent
no-data gap. The one real, confirmed defect found in the course of this correction -
`Percent_NonTree_Vegetation` not propagating the water/out-of-projection codes - is a separate, small,
well-scoped fix (plan above), not evidence that the original masking was broken.
