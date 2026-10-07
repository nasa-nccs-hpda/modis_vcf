# Tree cover on ice and polar-night land: tree rows and a bare-priority cap (plan)

**Status: WRITTEN 2026-10-07. EXPERIMENT TO BE DONE: approved in principle by the project lead as the next experiment, not scheduled
yet.** It runs after F13 unless the project lead says otherwise. Any run goes under `FullRuns/EXP_tree_ice_<date>/` (EXPERIMENTAL). It
cannot change v1.2.0.

## 1. Why

- **v00 without the rule** (`EXP_v00_norule_20261007`, in `Version_v1.2.0_Build_Results.md`):
  - On h16v00 / h17v00 interior ice, v1.2.0's bare model predicts about 100.
  - The tree model adds 1-2% tree cover on 71-82% of those pixels.
  - The sum cap then lowers bare to 98-99.
  - Tree is exactly 0 on only 10-16% of land, against C6 / C6.1 at 0.
- **The same mechanism outside v00:**
  - Up to 48,798 pixels across v1.2.0 have tree 1-5 and bare 95-99 (`v120_20261007/out/summary/cap_count.csv`).
  - Concentrations: h21v06 desert 9,656; h12v12 9,030; Iceland h17v02 7,591; h13v12 4,163; h16v01 3,913.
- **Why tree does this:**
  - The tree model has no ice rows at all.
  - Decision T3 kept the h16v02 ice rows bare-only.
  - With no training rows like these pixels, a random forest averages leaves from elsewhere, so small positive tree values are expected.
- **The project lead's view (2026-10-07):** a bare prediction of exactly 100 is probably more trustworthy than a tree prediction of 1-5%.

## 2. Variants

All variants keep v1.2.0's bare model (A10S0), code, inputs and flags. Only the tree side changes.

| Variant | Change | Kind |
|---|---|---|
| **T0** | v1.2.0 as is (control); T60 seed 0, and T60 seed 1 for the yardstick | — |
| **TR** | **Tree rows:** the same 10,000 h16v02 ice points as the bare rows (A10), label tree = 0, added to the tree split; refit T60 (seeds 0 and 1) | data-driven |
| **BP** | **Bare-priority cap:** where the raw bare prediction is 100, set tree to 0 (tree = min(tree, 100 − bare_raw)); everything else as v1.2.0 | rule (post-model) |
| **BP95** | Sensitivity of BP: also where bare_raw ≥ 95 and tree ≤ 5, set tree = 100 − bare_raw | rule (post-model) |
| **TR+BP** | Both | data-driven + rule |

**Notes:**
- BP is the mirror image of today's sum cap. Today bare gives way to tree; under BP, tree gives way when bare is 100.
- BP is close to the C code's rule C5 (the tree cap after the bare rules). The difference is that BP is triggered by the model's own bare =
  100, not by a mask.
- BP is still a post-model rule. It is tested here because the project lead expects bare = 100 to be the more trustworthy prediction. Step
  1 checks that expectation against the labels before any maps are made.

**Combined 2026-10-07 (project lead): the cold-ice rule is revisited in this experiment (variant CR + tree).** **Added 2026-10-07: a cold-ice rule (CR) was checked and NOT ADOPTED, to be revisited** (`Cold_Ice_Rule_Check_Results.md`). Cold
(band 31 ≤ 275 K in 3 of 4 late-summer periods) and bare ≥ 90 → bare 100, tree 0. It reaches 100% on the ice interior by both masks,
including v00. It leaves low-sun artefacts on warm v00 land untouched. If this plan runs, score CR + tree as a comparison variant on the
same zones.

## 3. Steps

1. **Evidence first** (read-only, about 30 min): is bare = 100 more trustworthy than tree 1-5?
   - **Rows:** held-out pixels that carry both a tree label and a bare label (the hash split is the same function, so the test keys overlap
     on tiles with both labels).
   - **Comparison:** where v1.2.0's raw bare prediction is 100, the tree label distribution and the tree prediction error.
   - **Reading:** if the tree labels there are mostly 0 while the tree model says 1-5, BP is supported by the labels; if they are not, BP
     is dropped before any maps.
   - **Also:** the same table for bare_raw ≥ 95 (BP95).
2. **Tree rows (TR):**
   - Tree predictors for the 10,000 A10 points are computed with the frozen pipeline from h16v02's v1.1.0 metric folder.
   - Gate first: 1,000 existing h16v01 tree rows recomputed must equal the stored split, as in the ice-rows run.
   - Rows go into a new tree split; the keyed hash puts about 20% in the held-out set, which is reported separately.
   - Fit T60 with seeds 0 and 1 (about 15 min each).
3. **Code for BP:**
   - One new option in a frozen copy of `VcfPredict.py` (for example `--barePriority 100` or `95`). The default is off, so v1.2.0 is
     unchanged.
   - Self-test: with the option off, the maps equal v1.2.0 byte for byte.
4. **Held-out labels:**
   - TR against T0 for tree (MEDIAN, all tree-label tiles), with the T60 seed-1 yardstick.
   - BP and BP95 applied to the held-out rows: tree and bare MAE.
   - Broken down by cover bin, by fallback count and by region (continent and latitude band).
5. **Maps (15 tiles, as in the ice-rows run, plus h21v06, h12v12 and h13v12, where the bare-near-100 cap cases cluster):**
   - T0 against TR, BP, BP95 and TR+BP.
   - Count of tree = 0 and bare = 100 on land.
   - Change beyond seed noise.
6. **v00 without the rule:** h16v00 and h17v00 for each variant.
   - **Pass for retiring the rule:** tree = 0 on at least 99% of land and bare = 100 on at least 95%, with the rule off.
   - Also: the MOD44C ice-zone scores.
7. **Write-up:** `Reports/Tree_Ice_Rows_and_Bare_Priority_Results.md`.

About 3-4 h, mostly maps.

## 4. Pass rules

- **TR (data-driven):**
  - Tree held-out labels improve or stay within seed noise, overall and on every tundra tile (h11v02, h12v02, h13v01, h21v01).
  - Tree on the ice zones moves to 0 by more than the seed change.
  - Maps outside the ice change within seed noise.
- **BP / BP95 (rule):**
  - Adopted only if step 1 supports it.
  - Tree held-out MAE improves (not merely ties), because a rule must earn its place.
  - Affected pixels are where the labels say tree is 0.
- **Retiring the v00 rule:** the v00 pass in step 6 with the smallest change that passes. The order of preference is TR, then TR+BP, then BP.
- **Version:** any adoption is a new version, named by the project lead (v1.2.1 or v1.3.0; not guessed).

## 5. Decisions (when it is scheduled)

| # | Question | Options | Suggested |
|---|---|---|---|
| R1 | Tree rows | (a) the same 10,000 A10 points (b) also h16v01/h17v01 ice | **(a)**: one change at a time |
| R2 | BP threshold | (a) 100 only, with 95 as a sensitivity (b) 100 only | **(a)** |
| R3 | Run BP if step 1 does not support it | (a) no (b) yes, for the record | **(a)** |
| R4 | v00 pass bar | (a) tree 0 on ≥ 99% and bare 100 on ≥ 95% of land (b) other | **(a)** |
| R5 | Extra map tiles | (a) add h21v06, h12v12 and h13v12 (b) the 15 tiles only | **(a)** |
