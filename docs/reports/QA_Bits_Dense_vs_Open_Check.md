# Can MOD44C QA bits tell open land from dense forest where clouds caused the fallback? (diagnostic)

> **Diagnostic only, experimental.** Nothing here changes v1.0.0. Run 2026-10-03 on forest202-205,
> `FullRuns/qa_bits_dense_vs_open_20261003` (scripts in `code/`, outputs in `out/analysis/`).

## Question (project lead)

The V4r blends raised tree cover in cloud-caused fallback pixels on open land as much as in dense forest (`V4r_Cloud_Blend_Results.md`,
Reassessment). Can any combination of MOD44C `state` QA bits on those cloudy inputs distinguish open-land labels from dense-forest
labels, so that a raise could be targeted?

## Method

**Pixels:**
- labeled pixels (training and held-out; Landsat labels judged by bin) with label 0-20 ("open") or 61+ ("dense");
- zenith-gated cloud count (`cloud_z_count`, from `trees_counts_2x2_20261003`) ≥ 3;
- tiles v04 and south;
- **651,002 pixels (357,515 dense, 293,487 open) on 38 tiles.**
- Also reported: cloud count ≥ 1 and ≥ 5, and v00-v03.

**Inputs:**
- every MOD44CQ input of the pixel's cloud-caused fallback periods, the same periods the cloud count counts;
- each pixel's number of such periods matches its cloud count exactly (all 63 tiles).

**Fields (per input):**
- cloud value (bits 0-1: 0-3), shadow (2), cirrus (3-4: 0-3), snow (5), adjacency (6), adjacency only;
- bit 7 (aerosol low bit or BRDF flag), bit 8 (aerosol high bit or odd day), bits 12-14, bit 15 (QC);
- solar zenith ≥ 60.

Each pixel gets the share of its inputs with each value.

**Tests:**
- **(a)** each field alone;
- **(b)** every combination of bits 0-7 and 15 as a per-input signature (512 possible; those with ≥ 200 inputs reported);
- **(c)** all fields together: decision trees of depth 1-4 and gradient boosting (an upper bound), with 5-fold cross-validated AUC;
- **(d)** baseline: how well v1.0.0's prediction alone separates the same classes (held-out pixels).

AUC 0.5 means no separation; 1.0 means perfect.

## Results (v04 and south, cloud count ≥ 3)

**(a) Single fields.**
- Most fields are never set in these inputs: cloud values 2 and 3, shadow, cirrus, and bits 12-14 are all at about 0%.
- The rest differ little between open and dense:

| Field | Share of inputs, open | Share of inputs, dense | Pixel AUC |
|---|---|---|---|
| bit 8 (aerosol high or odd day) | 0.51 | 0.57 | 0.58 |
| cloud = 1 | 0.73 | 0.75 | 0.45 |
| adjacency only | 0.27 | 0.25 | 0.55 |
| zenith ≥ 60 | 0.31 | 0.25 | 0.45 |
| snow | 0.06 | 0.05 | 0.46 |
| bit 15 | 0.23 | 0.20 | 0.53 |
| bit 7 (aerosol low or BRDF) | 0.21 | 0.16 | 0.53 |

**(b) Signatures.**
- The dominant one is "cloud = 1, nothing else": 4.3 M inputs, 51% dense, AUC 0.47. It carries no information.
- The adjacency-only signatures split on bit 7: 38% dense with bit 7 set, 64% without.
- No signature reaches a pixel AUC above 0.58.

**(c) All fields together** (5-fold cross-validated AUC):

| Model | cloud ≥ 1 | cloud ≥ 3 | cloud ≥ 5 |
|---|---|---|---|
| Tree depth 1 | 0.53 | 0.61 | 0.63 |
| Tree depth 3 | 0.65 | 0.77 | 0.78 |
| Tree depth 4 | 0.68 | 0.80 | 0.82 |
| Gradient boosting | 0.80 | 0.86 | 0.87 |

- **The depth-3 tree splits only on the cloud count, the share of inputs with zenith ≥ 60, and the adjacency share.** Those describe
  where and in which season the clouds occur, not the cover.
- The folds are random pixels, and QA flags cluster in space. So these scores largely reflect location and season and are optimistic.

**(d) Baseline** (130,029 held-out pixels in the same set, 71,462 dense):

| Separating open from dense | AUC |
|---|---|
| v1.0.0 prediction alone | **0.990** |
| V4R prediction alone | 0.988 |
| QA fields alone (gradient boosting) | 0.855 |
| QA fields + v1.0.0 prediction (gradient boosting) | 0.991 |

**v00-v03:** 71,289 pixels, almost all open (2,309 dense). The high AUCs there (0.91-0.98) reflect that imbalance and the polar
setting, not a usable signal.

## Findings

1. **No QA bit, and no combination of them, separates open land from dense forest in cloudy pixels in a usable way.**
   - Single fields and signatures reach at most AUC 0.58.
   - The multi-field models reach 0.77-0.87, but only through cloud amount, low-sun share and adjacency share, i.e. location and
     season. That is optimistic, and it isn't cover information.
2. **It is not needed.** v1.0.0's own prediction already ranks open against dense almost perfectly in these pixels (AUC 0.99), and
   adding QA bits adds nothing (0.991).
3. **So the cloud problem is not that open and dense get confused.** It is that dense pixels are given values that are too low:
   - Reassessment R1: at 5+ cloud-caused periods, 44% of dense-label pixels fall one bin low;
   - yet they are still ranked above open land.
4. **Implication for a targeted fix:** use the cover information the models already have, not QA bits.
   - For example, apply a raise in cloud-caused fallback pixels only where v1.0.0 (or V4R) already indicates forest. That would avoid
     raising open land.
   - Alternatively, use a model with the cloud counts as predictors, which can learn a correction that depends on cover. That is
     being tested in `Trees_and_Counts_2x2_Plan.md`.
