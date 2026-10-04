# V4r only where clouds caused the fallback: one model on two composite strategies, and a cloud-gated blend (plan)

**Written 2026-10-03. Approved and run the same day (`FullRuns/v4r_blend_20261003`); results in `V4r_Cloud_Blend_Results.md`.** This is an experiment to improve v1.0.0 (names as in
`Product_Versions_and_Work_Map.md`). It follows `Cloudy_Pixels_Continuation_Results.md` (run `FullRuns/cloudy_v4r_weights_20261002`).

## Idea (project lead, 2026-10-03)

- Use v1.0.0 as it is, unless a pixel's composites fell back **because of clouds**.
- As cloud-caused fallback increases, give the V4r-based prediction more weight, up to all of it.
- The threshold and the weights are to be found by experiment.
- **Preferred if it works: one model (v1.0.0) applied to two composite strategies**, rather than two models.

## V4r applies only where clouds caused the fallback

This is already how the V4r composites in `cloudy_v4r_weights_20261002` were built (`code/v4r_compose.py`). They are reused unchanged.

**A period of a pixel is replaced only if all three hold:**
1. **The period fell back under v1.0.0's QA:** none of its three 16-day inputs passed the good-day test (cloud bits clear, shadow
   off, adjacency off, bit 15 set, 0 < zenith < 65).
2. **Every input that exists is flagged cloudy or cloud-adjacent:** `state` cloud bits 0-1 ≠ 0, or adjacency bit 6 set.
3. **At least one input file exists.**

**What replaces it:** the input with the lowest red (Band_1) among inputs that are not shadow, have valid red and NIR, NDVI ≥ 0 and
NIR ≥ 1500. All 8 bands come from that input. If no input qualifies, the fallback mean stays.

**Periods that are not replaced (they stay as v1.0.0's fallback mean):**
- periods that fell back for other reasons: zenith over 65, bit 15 not set, or shadow on a day not flagged cloudy;
- periods with any input that is not flagged cloudy.

**The cloud count used for the gate below** is the number of a pixel's 8 periods that were actually replaced (0-8). It is stored
for every pixel of all 63 tiles in `out/compose/<tile>_v4r_used_count.npy`. Unlike the fallback count, it counts only cloud-caused
fallbacks.

**For the project lead to confirm:**
- **(a)** the cloud test counts cloud-bit values 1 (cloudy), 2 (mixed) **and 3 ("not set, assumed clear")** as cloudy;
- **(b)** cloud-adjacent (bit 6) counts as cloud.

Both are part of the existing composites. Changing either means rebuilding the V4r composites and metrics for 63 tiles (about
1.5-2 h) before step 1. A count of how many replacements depend on (a) and (b) can be added to step 1 at no cost.

## Candidates (tree cover only)

| ID | Composites | Model | State |
|---|---|---|---|
| **v1.0.0** | current (fallback mean) | v1.0.0 | maps of the 7 tiles and held-out predictions exist |
| **v1.0.0-V4C** | V4r (cloud-caused fallbacks replaced) | **v1.0.0, not retrained** | **new (step 1)** |
| **V4R** | V4r | v1.0.0's rules, retrained on V4r metrics | maps of the 7 tiles and held-out predictions exist |
| **BLEND-1** | both | **one model:** v1.0.0 on both composite sets, blended by cloud count | post-processing (steps 2-3) |
| **BLEND-2** | both | two models: v1.0.0 and V4R, blended by cloud count | post-processing (steps 2-3) |
| SEED1, SEED2 | current | v1.0.0 refit with seeds 1 and 2 | exist; the run-to-run yardstick |

**v1.0.0-V4C is itself already a one-model version of the idea.** Where no period was replaced, its inputs equal v1.0.0's, so its
output must equal v1.0.0 exactly. Step 1 checks this. BLEND-1 only adds a threshold and a ramp on top.

**Blend:**
- weight w = 0 for cloud count below t₀, rising linearly to 1 at t₁;
- output = round((1 − w) × v1.0.0 + w × other).

The grid covers t₀ from 1 to 6 and t₁ from t₀ to 8. A hard switch is t₀ = t₁.

## Steps

**Step 1 (compute; about 30-45 min on ngaproc201/202, which are free).** Run v1.0.0 (`qaC_full63_retrain_20260929/Python/models/C/rf`,
MEDIAN, rounded) on the V4r metrics (`cloudy_v4r_weights_20261002/Python/metrics_v4r`), in a new run root:
- **Maps of the 7 tiles:** h11v08, h12v04, h12v09, h19v08, h19v09, h20v06, h26v06. These are the tiles that already have v1.0.0,
  V4R and seed maps.
- **Held-out predictions, all 63 tiles:** from `Python/split_v4r/xTest.parq` (same pixels and keys as the existing held-out
  predictions).
- **Acceptance checks:**
  - v1.0.0-V4C equals v1.0.0 at every pixel with cloud count 0, on the maps and on the held-out rows;
  - the held-out keys match the existing `out/heldout/predictions.parq` row for row;
  - a count of replacements that depend on cloud value 3 and on adjacency alone (for (a) and (b) above).

**Step 2 (post-processing; minutes). Blend test on the held-out labels, all 63 tiles:**
- Join each held-out pixel's cloud count from the `used_count` rasters, with an orientation check (row/column, as in the stable-set
  scoring).
- **Tuning:** split the held-out pixels in half by a fixed hash. Choose t₀ and t₁ for BLEND-1 and BLEND-2 on one half, and report
  on the other.
- **Scores:** MAE, bias and RMSE for every candidate:
  - pooled;
  - by cloud count (0, 1-2, 3-4, 5+);
  - by cover bin (0-20 / 21-40 / 41-60 / 61+);
  - for dense pixels with 5+;
  - by continent and latitude band;

  each with intervals and tiles-better counts against v1.0.0, next to the SEED1-SEED2 difference.

**Step 3 (post-processing; minutes). Blend maps of the 7 tiles,** from the step 1 maps, the existing maps and the cloud counts, with
the step 2 thresholds:
- agreement with C6, C6.1 and the C code (all three cloud-affected), pooled and by cloud count;
- the h12v09 drop pixels;
- a seam check: differences between neighbouring pixels at the t₀ and t₁ edges, against the same within v1.0.0.

**Step 4 (alongside; read-only). The sharp decreases:**
- Where V4R and v1.0.0-V4C fall well below v1.0.0, on the 7 tiles: their cloud counts, MCD12Q1 land cover, and pick properties
  (red, NIR, NDVI, date). This includes the h12v09 floodplain and eastern edge.
- The question for the blend: are the decreases in the high-count pixels that a blend would hand to V4r?
- **The northern tiles** where V4r replaces the most (h12v03 44% of land pixel-periods, h24v03, h23v03, h11v03) are not among the 7
  mapped tiles. They are covered here only through their held-out pixels.

**Step 5. Report:** `V4r_Cloud_Blend_Results.md`, and the README is updated.

## Reading the results (no automatic adoption; the project lead decides)

A blend is worth taking further if:
- at high cloud counts it beats v1.0.0 on the held-out labels by more than the seed noise;
- at low counts it is v1.0.0 (by construction);
- the maps move the same way against C6, C6.1 and the C code;
- it shows no new seams;
- it does not carry unexplained sharp decreases.

**BLEND-1 is preferred over BLEND-2 if it is within the seed noise of BLEND-2,** because it needs one model.

## Not in this plan (needed only if a blend is taken further)

- Non-vegetated: blend it as well, or only tree cover. The sum cap ties the two layers together.
- The SD layer for blended pixels.
- 63-tile maps.
- Production cost: two composite sets and two metric sets per tile, plus two models for BLEND-2. V4r compositing is an experimental
  script, not product code.

## Practicalities

- **Run root:** `FullRuns/v4r_blend_<date>/`, new.
- **Inputs:** the earlier runs are mounted read-only.
- **Code:** copies only, nothing on the delivery branch.
- **Runs:** fully detached, with status and done markers.
- **Total:** about 1-1.5 h, including the report. Only step 1 uses a compute node.
