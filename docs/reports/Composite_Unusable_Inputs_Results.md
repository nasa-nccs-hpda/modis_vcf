# What the composite does with unusable inputs: results (EXP_unusable_inputs_20261005)

**Status 2026-10-05 evening: done; does not pass the project lead's rule D9 (b). Recommendation: do not adopt; v1.1.0 stays as it is.**
Run `FullRuns/EXP_unusable_inputs_20261005` (EXPERIMENTAL), 11:18-19:48 on ngaproc201/202. Plan and decisions D1-D12:
`Composite_Unusable_Inputs_Plan.md` (section 9). Adoption is the project lead's decision (D12).

**Decided 2026-10-05 evening (project lead): the full rule (D1 (a)) is NOT adopted; v1.1.0 stays as it is. Option 2 approved:** a short
test of the fill-only rule (D1 (c): only −28672 is unusable, per band; polar-night zeros and missing files as today), without retraining,
run `FullRuns/EXP_fillonly_20261005` (EXPERIMENTAL; 22 tiles, v1.1.0 models, maps against v1.1.0).
**Correction to the report text below:** option 2 is NOT expected to be a near-no-op on the maps. The −28672 fill reaches the case-2 mean in
nearly every predictor band on the 15 tiles at v ≤ 02 (up to about 770,000 land pixels in one period on h16v01). Only the 9 tiles
outside v ≤ 02 are near-no-ops.

## Bottom line

- **Tree cover: unchanged.**
  - Held-out labels: −0.001 MAE [−0.005, +0.003], against a seed yardstick of 0.002-0.005.
  - Maps on the held-out labels: +0.000.
  - Pooled maps against C6, C6.1 and the C code: within 0.03.
- **Non-vegetated: slightly worse, not better.**
  - Held-out labels: **+0.069 MAE [+0.003, +0.159]**. On the 11 rebuilt tiles with bare labels it is **+0.109**, with 8 of 11 tiles worse.
  - Maps on the held-out labels: +0.044.
  - Pooled maps: C6 −0.03 (a tie), C6.1 +0.33, the C code +0.28.
- **Where the change comes from:** the polar-night tiles h12v01, h12v02, h13v01 and h21v01 get worse by +0.31 to +0.46 MAE each.
  These are where kind B (solz ≤ 0 zeros) now becomes 10000 in the fallback. They are also the only tiles, with h16v01 and h19v09,
  whose training columns changed. Everywhere else the differences are at seed-noise size.
- **Greenland: the references disagree.**
  - Held-out labels on h16v01, scored from the maps, improve: 0.93 → 0.46.
  - Against C6, h16v01 and h17v01 get worse: bare MAE 2.7 → 5.9 and 3.0 → 7.3, and the share with tree = 0 falls from about 99.6% to 64-69%.
  - h15v02 and h16v02 get better against C6.
- **D9 (b) needs bare to improve, and it does not.** It would also fail D9 (a), "no worse", which allows a bare change of at most about 0.005.

## What was tested

- **Rule (D1-D4 as decided):** an input is unusable for a band if:
  - (A) its file is missing;
  - (B) its solz ≤ 0 (polar night: band = 0, state = 0);
  - (C) its value in that band is −28672. This is tested per band, so a fill in one band never removes another band's value.
- **What the composite does with them:**
  - Case 1 (a QA pass) also rejects −28672.
  - Case 2 (fallback) averages only the usable inputs.
  - When no usable input is left, the band gets 10000, as for a missing file today.
- **Left as they were:** low-sun inputs (D), the band-31 clamp, the Greenland zenith-only NDVI rule and the v00 rule.
- **Which tiles were rebuilt:** 24 in all. These are the 15 tiles at v ≤ 02, plus 9 of the 51 other screened tiles. Those 9 each have
  1-1,044 land pixels with kind C in band 1 or 7, at most 0.005% of land. Each was given new composites, metrics and training rows.
- **Models:** tree and bare refit with v1.1.0 settings (60 trees, seed 0, sqrt, MEDIAN for both layers). Maps were made for 69 tiles.
- **What the scores are compared against:**
  - For held-out labels: fill72 CTRL, which has the same base splits and the same fit recipe.
  - For maps: v1.1.0 (`cand60med_20261004/out/maps/CAND`), with v1.0.0 alongside.

## Checks

- **Self-test:** passed. The patched composer changes exactly the land pixels the overnight count predicted, on all 56 band-periods of
  h16v02 and h12v09. Example: h16v02 band 1 changes 34 pixels in 2019209 and 683,774 in 2019305; h12v09 changes 0.
- **Gate:** passed. With the patched code, h12v09, h11v04 and h20v06 reproduce the base tree and bare training rows exactly.
- **Screen of the 51 tiles the count did not cover:** 42 unchanged. Today's composites were reproduced exactly on all 51.
- **Splits:** keys and labels are unchanged. Training columns changed only on h12v01, h12v02, h13v01, h16v01, h21v01 (tree and bare)
  and h19v09 (tree only). On every other rebuilt tile, 0 columns changed.
- **Fits:** 2 of 2 done, 6-14 minutes each. **Maps:** 69 of 69, h15v02 on a re-run (see below). **Evaluation:** done.
- **Map consistency:** the v1.0.0 map value equals the v1.0.0 held-out prediction at 99.47% of bare held-out pixels. I did not look into
  the other 0.5%. The maps apply the non-vegetated sum cap and the held-out predictions do not, which may account for it.
- **Three stops, no data lost:**
  - Attempt 1: the metrics root did not exist. Fixed.
  - Attempt 2: the screen's scratch folder did not exist. Fixed.
  - Attempt 3: the h15v02 map ran out of memory twice while 2 other map slots were running. Re-run alone at 19:20 and succeeded.
    Because the map was missing, `map_check` and `eval` crashed the first time; both passed on the re-run.
    Details: `status/attempt3_h15v02_oom/README.txt`. About 1 h lost.

## Held-out labels (NEW − CTRL, MAE; 95% tile-bootstrap interval)

These scores come from models refit on the rebuilt rows, on the project's 20% pixel hold-out. The scores are optimistic, because
held-out pixels neighbour training pixels, but the comparison between the two models is fair. Seed yardstick: about 0.002-0.005.

| Layer | Group | n | CTRL MAE | NEW − CTRL | Tiles better / worse |
|---|---|---|---|---|---|
| Tree | all 63 tiles | 2,289,990 | 6.588 | −0.001 [−0.005, +0.003] | 37 / 21 |
| Tree | 18 rebuilt tiles | 627,408 | 5.033 | −0.000 [−0.009, +0.010] | 10 / 6 |
| Tree | 45 other tiles | 1,662,582 | 7.175 | −0.001 [−0.006, +0.003] | 27 / 15 |
| **Bare** | all 20 tiles | 2,160,056 | 3.768 | **+0.069 [+0.003, +0.159]** | 8 / 12 |
| **Bare** | 11 rebuilt tiles | 1,356,999 | 4.047 | **+0.109 [+0.007, +0.249]** | 3 / 8 |
| Bare | 9 other tiles | 803,057 | 3.296 | +0.000 [−0.003, +0.004] | 5 / 4 |

**Bare by tile (rebuilt tiles and any tile beyond ±0.01):**

| Tile | n | CTRL MAE | NEW − CTRL | Note |
|---|---|---|---|---|
| h12v02 | 149,613 | 5.70 | **+0.460** | polar night; columns changed |
| h13v01 | 7,195 | 5.42 | **+0.366** | polar night; columns changed |
| h12v01 | 16,413 | 7.89 | **+0.324** | polar night; columns changed |
| h21v01 | 214,703 | 4.74 | **+0.314** | polar night; columns changed |
| h20v09 | 3,282 | 11.81 | +0.194 | kind C only, 0 test columns changed (refit only; small n) |
| h11v02 | 260,246 | 4.81 | +0.015 | lesson-5 tile |
| h16v01 | 100,951 | 0.45 | +0.009 | Greenland; columns changed |
| h08v05, h09v05, h11v10, h19v12 | | | −0.008 to +0.006 | kind C only; seed-noise size |

**Tree on the 4 lesson-5 tiles:** h11v02 −0.024, h20v02 −0.026, h21v02 +0.030, h23v02 −0.026. These are mixed and small.

**Bare by label bin, scored from the maps (v1.1.0 → NEW):**

| Bin | n | v1.1.0 | NEW | Difference |
|---|---|---|---|---|
| 0-20 | 990,268 | 2.505 | 2.521 | +0.015 |
| 21-40 | 286,685 | 6.869 | 6.947 | +0.078 |
| 41-60 | 282,881 | 5.853 | 5.883 | +0.031 |
| 61-99 | 354,868 | 5.323 | 5.450 | +0.128 |
| 100 | 226,342 | 0.287 | 0.302 | +0.015 |

**Bare by region, scored from the maps (v1.1.0 → NEW):**
- North America 50-80N: 5.19 → 5.38 (+0.19)
- Asia 50-80N: 4.70 → 4.99 (+0.29)
- Greenland: 0.93 → 0.46 (−0.46)
- Every other region within ±0.01, except Africa 10N-10S (−0.08, n = 3,130)

**Tree by region:** every region is within ±0.02.

## Maps against C6, C6.1 and the C code (comparisons, not truth)

Pooled over 69 tiles, on the common valid mask (about 1.1 billion pixels):

| Layer | Reference | v1.0.0 | v1.1.0 | NEW | NEW − v1.1.0 |
|---|---|---|---|---|---|
| Tree | C6 | 4.749 | 4.524 | 4.537 | +0.013 |
| Tree | C6.1 | 5.752 | 5.537 | 5.569 | +0.031 |
| Tree | C code | 5.362 | 5.150 | 5.153 | +0.003 |
| Bare | C6 | 8.781 | 10.482 | 10.451 | −0.030 |
| Bare | C6.1 | 15.097 | 17.019 | 17.353 | +0.335 |
| Bare | C code | 12.399 | 14.074 | 14.352 | +0.279 |

C6.1 zeroes high-latitude pixels, so it weighs the v ≤ 02 changes differently from C6.

**v ≤ 02 tiles, bare MAE against C6 (v1.1.0 → NEW):**

| Tile | v1.1.0 → NEW | Notes |
|---|---|---|
| h15v02 | 14.5 → 10.9 | better |
| h16v02 | 1.7 → 1.1 | better; exactly-100 share 70 → 92% |
| h20v02 | 9.8 → 7.9 | better |
| h21v02 | 10.6 → 8.5 | better |
| h23v02 | 9.8 → 8.9 | better |
| h16v01 | 2.7 → 5.9 | worse; exactly-100 share 85 → 59%; tree = 0 share 99.5 → 69.5%; tree mean 0.01 → 1.0 |
| h17v01 | 3.0 → 7.3 | worse; exactly-100 share 86 → 43%; tree = 0 share 99.7 → 63.7%; tree mean 0.01 → 1.0 |
| h13v01 | 9.3 → 13.4 | worse; bare mean 21.8 → 27.1 |
| h12v01, h21v01 | 0.0-0.6 change | ties |
| h11v02, h12v02, h17v02 | 0.0-0.6 change | ties |
| h16v00, h17v00 | unchanged | v00 rule |

**The h16v02 edge:** the share of pixels 5 or more below C6 is 0.08% above the edge and 0.00% below it, against 0.04% and 0.05% for
v1.1.0. The edge does not return.

**The 9 kind-C-only tiles outside v ≤ 02:** every change against C6 and C6.1 is ≤ 0.1 point. These changes come from the refit.

## Reading

- **Polar night (kind B) drives the result.** The models were trained on the same rows, so turning fallback zeros into 10000 moves the
  training rows and the maps on a handful of Arctic tiles. On h12v01, h12v02, h13v01 and h21v01, the island and coast tiles with labels,
  bare gets worse against the labels. Against C6, the results are mixed.
- **On Greenland the two references point opposite ways.**
  - Held-out labels on h16v01 halve the map error.
  - C6 says h16v01 and h17v01 move away from 100 bare and 0 tree on the ice sheet. fill72 G moved h16v01 the same way, by less (4.8 → 5.7).
  - Part of the label gain may only mean that h16v01's training rows and its map inputs now come from the same composites. In v1.1.0
    they did not (decision 19.6, "revisit h16v01 training inputs"). I have not checked this.
  - Under the project rule, two references that disagree give no recommendation for Greenland on their own.
- **Kind C alone (the in-file −28672) is harmless, but this run cannot measure its benefit by itself.** On the 9 tiles where only kind C
  applied, every difference is at seed-noise size. The run cannot separate kind C from kind B on the v ≤ 02 tiles.
- **Kind A (missing files) behaves the same as today in 2019.** The new rule gives 10000 to a period with no usable input, which is what
  a missing file already gets. The choice therefore matters mainly for the other years (to-do 12), where it is unchanged by not adopting.

## Options for the project lead

1. **Do not adopt; keep v1.1.0 as is** (recommended under D9 (b)). The current behaviour stays in place for the other years, including
   10000 for missing files.
2. **Test a narrower rule**, D1 (c): treat only the fill −28672 (kind C) as unusable, and leave polar-night zeros as they are.
   - The evidence above suggests this would be a near-no-op on the map. It is a data-validity fix, not an accuracy gain.
   - It could be adopted under D10 (b) with only the exactness check, if you want the composites cleaned before other years.
   - About 4-5 h detached, or about 1 h as a maps-only check.
3. **Look into Greenland separately.** Why do held-out labels and C6 disagree on h16v01? This is tied to decision 19.6, the h16v01
   training inputs. No run is needed to plan it.

## Files

- **Run:** `FullRuns/EXP_unusable_inputs_20261005/`.
- **Held-out scores:** `out/heldout/{tree,bare}_NEW/`, and NEW against CTRL in `out/compare/` (`compare.csv`, `per_tile_*.csv`).
- **Map checks:** `out/map_check.csv` (rebuilt tiles: means, exactly-0/100 shares, against C6 and C6.1, the h16v02 edge).
- **Map evaluation:** `out/eval/` (`refs_pooled.csv`, `refs_per_tile.csv`, `labels_from_maps.csv`, `checks.json`).
- **Maps:** `out/maps/NEW/` (69 tiles, 3 layers). **Models:** `Python/models/NEW_{tree,bare}/`. **Rebuilt metrics:** `Python/metrics_new/`.
- **Stops:** `status/attempt{1,2,3}_*/README.txt`.
