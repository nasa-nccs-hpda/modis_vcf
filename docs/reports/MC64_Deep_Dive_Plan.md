# Why is the Monte Carlo 64 predictor set worse? Deep-dive plan (to-do 14; written 2026-10-04 evening, NOT approved, nothing run)

**Status:** plan only, for the project lead to review. The MC64 set is already out of the product (decision 19.3). This work is
investigation only and cannot change v1.1.0. Any fits it makes are **EXPERIMENTAL** and go under a run directory prefixed `EXP_`.

## 1. What is known

- Tree MC top 64 is **+0.28 MAE** worse than the C code's 64 on held-out labels, in both tests: the 2026-10-02 rerun (30 trees) and the
  60-tree gated test (`Monte_Carlo_Rerun_Results.md`, `Trees60_then_MC64_Gated_Results.md`). It is worse in every cover bin
  (0-20 +0.17 to 61+ +0.44) and in all 15 regions (up to +0.68 in South America 10N-10S). The gap is widest on dense labels under
  heavy cloud (+0.57). It is better on only 9 of 63 tiles. The seed noise is about 0.01-0.02.
- On the maps it agrees less with the C code, C6 and C6.1 (+0.25 to +0.35).
- Only 19 of the 64 predictors are shared with the C code's set. The ranking itself is reproducible (Spearman 0.97 against 2026-09-20).
- Non-vegetated MC top 64 is +0.11 worse (mixed against references). 6 of its predictors are shared with the C code's 21.

## 2. Hypotheses

- **H1, how the selection works.** Each trial fits an RF on **10 random predictors** and scores them by permutation importance; the
  ranking is the mean over trials (`MonteCarloSim.computeAverages`, `_getTopN`). That rewards predictors that are strong **on their own**
  in a small set. A top 64 built that way tends to be **redundant**: many near-copies of the same signal (e.g. one band's statistics
  over several periods). It also tends to miss predictors that add **complementary** information only alongside the others.
  This is a known weakness of ranking by individual importance. It predicts that MC64 is more inter-correlated and has fewer
  "effective" predictors than the C code's 64.
- **H2, cloud-sensitive metrics.** MC64 may favour statistics (single-period values, maxima or minima) that are more affected by
  fallback composites. This predicts a gap that grows with the fallback count, as seen at dense, heavily clouded labels.
- **H3, specific C-code-only predictors.** A few of the 45 C-code-only predictors (for example thermal band 31, or the dark-period
  band 7 ranks) may carry most of the advantage.
- **H4, scoring rows.** The trials use 500,000 train and 500,000 evaluation rows, a subsample with a pixel split. If the subsample
  over-represents easy (open, unclouded) pixels, the ranking may favour predictors that are good there.

## 3. Steps

**Read-only (no fits; about 1-2 h; from the existing tables, models and predictions):**
1. **Inventory:** the two 64-sets by band, statistic type and period, with what is shared and what is unique.
2. **Redundancy (H1):** the correlation matrices of both sets on the training rows. Report the mean |r| within each set, clusters at
   |r| > 0.9, and the effective number of predictors (PCA components needed for 95% of the variance).
3. **Where the gap is (H2, H4):** the held-out error difference (MC64 minus C64, same rows) by fallback count × cover bin × region,
   and the share of the MC trial rows in each fallback/cover cell compared with the held-out set.
4. **Importance in context (H3):** permutation importance on held-out rows inside the full 64-predictor models (T60 and MC64-60),
   overall and on dense cloudy pixels. Uses the saved models.

**Optional fits (EXPERIMENTAL, about 20 min per fit; only if the read-only steps point somewhere; ask first):**
5. **Swap tests:** C64 minus its top-k unique predictors, replaced by MC-only ones, and MC64 plus the top C-only predictors.
6. **Redundancy-aware selection** (if H1 holds): greedy forward selection, or ranking with correlated clusters collapsed. This is a
   test of the method, not a product candidate.

## 4. What the project lead gets

`MC64_Deep_Dive_Results.md`: which hypotheses hold, with numbers. If useful, it also says whether a future predictor search should
change method. There are no product changes.

## 5. Inputs (existing; read-only)

`FullRuns/mc_rerun_20261002` (trials, rankings, final fits, held-out predictions) and `FullRuns/trees60_mc64_gated_20261003` (T60 and
MC64-60 models, held-out scores). Fallback counts are as in the cloudy-pixel reports. Run on forest201-205 or ngaproc201/202, fully
detached.
