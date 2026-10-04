# V4r only where clouds caused the fallback: one model on two composite strategies, and a cloud-gated blend (results)

**Run 2026-10-03, 08:02-08:21 (plus the control in section 2b, run afterwards), ngaproc201/202, fully detached (`FullRuns/v4r_blend_20261003`). Plan: `V4r_Cloud_Blend_Plan.md`
(approved 2026-10-03).** This is an experiment to improve v1.0.0 (names in `Product_Versions_and_Work_Map.md`). All gates and
acceptance tests passed, and no task failed. One control was added after the first results (section 2b). Nothing is adopted; the
project lead decides.

## Why the maps and the labels disagreed — correction (2026-10-03, read this first)

**Question:** on the 7-tile maps, the V4r products raise tree cover in dense cover at 5+ cloud-caused periods (+4 to +6.5 where C6 ≥
60). At the held-out dense labels pooled over all tiles, they did not. Why?

**Checks** (`code/maps_vs_labels.py`, `code/per_tile_dense.py`; outputs in `out/gated/`):
- held-out pixels cross-tabulated by label bin × C6 class and label bin × v1.0.0 class, on all v04+ tiles and on the 7 map tiles only;
- then the same per tile.

**Names:**
- V4R = the model retrained on V4r metrics;
- V1-V4C = v1.0.0's model applied to V4r composites;
- SEED1 = the seed-refit control.

All changes below are from v1.0.0, at 5+ zenith-gated cloud-caused periods.

**1. The pooled label result is dominated by one tile.** h10v04 (US northern Rockies, about 45 N) holds 6,595 of the 9,893 held-out
pixels with label ≥ 41 and 5+ periods (67%).

| Pixels with label ≥ 41, 5+ periods | Tiles | n | Mean label | v1.0.0 | V4R | V1-V4C | SEED1 |
|---|---|---|---|---|---|---|---|
| All tiles | 15 | 9,893 | 61.3 | 52.0 | −0.48 | +0.15 | −0.04 |
| **Without h10v04** | 14 | 3,298 | 68.6 | 61.1 | **+1.23 (12 of 14 tiles up)** | **+1.76 (11 of 14 up)** | +0.01 |
| h10v04 only | 1 | 6,595 | 57.7 | 47.4 | −1.33 | −0.66 | −0.06 |

- h10v04 is a snowy mountain tile. Its cloud-caused periods with adequate sun are most likely snow flagged as cloud. That matches
  `Snow_vs_Cloud_V4r_Trigger_Check.md`, where the snow bit catches only part of it.
- **Without h10v04, both V4r products raise dense-label pixels at 5+ periods, on most tiles,** while the seed control does not move.
  The raise (+1.2 to +1.8) closes about a sixth to a quarter of the 7.5-point gap between v1.0.0 and the labels there.
- **At 3-4 periods there is no raise** (−0.26 / −0.37, about half the tiles up). The effect is specific to the heaviest cloud.

**2. On the 7 map tiles, the labels agree with the maps.**
- Held-out dense labels at 5+ periods are raised there: 61+ +0.83 (V4R) / +0.63 (V1-V4C); 41-60 +2.45 / +4.29.
- So the disagreement was not maps against labels as such. It came from h10v04, which is not a map tile.

**3. The raise goes where v1.0.0 is too low, not where it is already right.** On the 7 map tiles, labels 61+ at 5+ periods, by
v1.0.0's own value:

| v1.0.0 at the pixel | n | Mean label | v1.0.0 | V4R | V1-V4C | SEED1 |
|---|---|---|---|---|---|---|
| < 30 | 44 | 68.3 | 15.4 | **+10.8** | **+12.1** | −0.2 |
| 30-59 | 288 | 68.3 | 50.9 | **+5.6** | **+7.2** | +1.5 |
| ≥ 60 | 1,115 | 79.8 | 77.1 | −0.8 | −1.5 | −0.6 |

- Over all v04+ tiles the pattern is the same but weaker: +3.8 / +6.2 where v1.0.0 < 30, and −1.3 / −2.1 where v1.0.0 ≥ 60.
- The seed control shows some of this pattern (+2.5 / −0.6). That part is regression to the mean when grouping by v1.0.0's own value.
  The V4r products go well beyond it.
- **This explains the map numbers.** The C6 ≥ 60 map pixels at 5+ periods have v1.0.0 averaging 69.6, i.e. many under-predicted
  pixels, and those get the large raise. The held-out label sites with C6 ≥ 60 are stable, homogeneous forest where v1.0.0 is already
  close to the label (76.7 against 77.4), so they have little room to rise (+0.1 to +0.4).

**4. Where v1.0.0 already matches dense labels, V4r lowers them slightly.** For example, h19v08 (v1.0.0 81.6, label 82.3: V4R −1.6,
V1-V4C −2.8) and h26v06 (−0.2 / −1.1). That is the price of the raise.

**Corrected conclusions** (they replace "V4r is not the right source of the raise" in the sections below):
- **V4r composites do raise tree cover in dense pixels that v1.0.0 under-predicts under heavy cloud (5+ periods),** on most tiles,
  in the right direction, beyond the seed control, and consistently between maps and labels on the 7 map tiles.
- **The exceptions:**
  - snowy tiles (h10v04), where "cloud" is likely snow and V4r lowers tree cover;
  - dense pixels v1.0.0 already gets right, which are lowered slightly;
  - 3-4 periods, where there is no net raise.
- **What follows for a blend rule:** gate on heavy cloud (5+ periods), on cover (protects open land), and on snow. Snow needs a test
  beyond the MOD44C snow bit, e.g. NDSI from Bands 4 and 6, or an external snow product. Without snow handling, a tile like h10v04
  loses.
- **Remaining caveats:**
  - only 14 tiles carry the "without h10v04" result, several with few pixels (20-470);
  - the held-out pixels are neighbours of training pixels;
  - excluding one tile after seeing the result is a post-hoc choice. The snow explanation needs confirming on h10v04 itself, as was
    done for h12v03.

## Reassessment (2026-10-03, after the project lead's review; partly corrected by the section above)

**The question, as the project lead framed it:**
- Where fallback is caused by clouds, C6, C6.1 and the C code are all likely too low in tree cover. Pooled MAE against them can
  therefore not judge a fix.
- What matters is **whether the blend raises tree cover where clouds caused the fallback, and whether the raise lands in the right
  places.**
- **Low-sun fallback must be excluded:** zenith test, `Snow_vs_Cloud_V4r_Trigger_Check.md`.

**Reference:** the held-out labels are Landsat 30 m from the early 2000s, at sites chosen to avoid change and shown to be mostly
stable. They are independent of MODIS clouds, and **they are judged by bin** (0-20 / 21-40 / 41-60 / 61+), not by absolute value.

**What was added** (`out/counts_zen/`, `out/reassess/`):
- **The zenith-gated cloud count (CUR_Z):** replaced periods where at least one input has 0 < zenith < 65. It is recomputed for all 63
  tiles, and the existing count is reproduced exactly as a check.
- **Fixed blends with w = CUR_Z / 8,** with nothing tuned:
  - B2Z = v1.0.0 with V4R;
  - B1Z = v1.0.0 with v1.0.0 on V4r composites (one model);
  - BSZ = v1.0.0 with a seed refit (control).
- **Held-out labels by cloud count × label bin,** and **maps (7 tiles)** by cloud count × C6 cover class. The maps report the change in
  tree cover, not error.

**Effect of the zenith test:**
- In the tropics nothing changes.
- On v00-v03 tiles, held-out pixels with 5+ cloud-caused periods fall to 157. **Gated this way, the high latitudes are almost
  entirely left as v1.0.0.**

### R1. Do the labels confirm that cloud-caused fallback makes v1.0.0 too low? Yes, in dense cover.

Tiles v04 and south, labels 61+:

| Cloud-caused periods | n | Mean label | v1.0.0 mean | v1.0.0 in the same bin / a lower bin |
|---|---|---|---|---|
| 0 | 184,112 | 74.8 | 68.2 | 0.76 / 0.24 |
| 1-2 | 195,717 | 77.8 | 73.6 | 0.85 / 0.15 |
| 3-4 | 65,764 | 76.8 | 70.7 | 0.77 / 0.23 |
| **5+** | 4,855 | 72.7 | **61.4** | **0.56 / 0.44** |

At 5+ cloud-caused periods, v1.0.0 is about 5 points further below the label than in cloud-free dense pixels, and 44% of dense-label
pixels fall a bin low.

### R2. Does the blend raise tree cover where clouds caused fallback? On the maps, yes; but not only where it should.

Maps (7 tiles), mean change from v1.0.0 at 5+ cloud-caused periods:

| C6 cover class (n) | V4R | v1.0.0 on V4r | B2Z (v1.0.0 + V4R) | B1Z (one model) | BSZ (seed control) |
|---|---|---|---|---|---|
| C6 ≥ 60 (5.7 M) | +5.6 | +6.5 | **+4.1** | **+4.7** | +0.5 |
| C6 < 30 (4.9 M) | +6.8 | +10.8 | **+5.6** | **+8.7** | 0.0 |
| All land (14.4 M) | +7.3 | +9.8 | +5.6 | +7.4 | +0.3 |

At 3-4 periods the raise is smaller: C6 ≥ 60 +0.3 to +1.0; C6 < 30 +1.0 to +3.7.

- **The V4r-based blends do raise tree cover where clouds caused fallback,** and the seed control does not. So the raise is a real
  effect of the V4r composites, not of averaging.
- **But the raise is not targeted at dense cover.** It is as large or larger where C6 is below 30: +5.6 to +10.8.
  - C6 is cloud-affected too, so a raise in C6 < 30 pixels could partly be correct where C6 itself is too low.
  - The held-out labels test this directly (R3).

### R3. Where the independent labels exist, the raise lands on open land, not on dense cover

Held-out pixels, tiles v04 and south, at 5+ cloud-caused periods:

| Label bin | n | v1.0.0: same / lower / higher bin | Mean change from v1.0.0: V4R / v1.0.0 on V4r / B2Z / B1Z / BSZ | Same / lower / higher bin, B2Z | Same / lower / higher bin, B1Z |
|---|---|---|---|---|---|
| 0-20 | 13,544 | 0.84 / 0 / 0.16 | +0.3 / +2.5 / +0.3 / +1.8 / 0.0 | 0.83 / 0 / 0.17 | **0.77 / 0 / 0.23** |
| 21-40 | 5,826 | 0.53 / 0.27 / 0.20 | −0.1 / +2.4 / 0.0 / +1.6 / +0.1 | 0.54 / 0.26 / 0.20 | 0.56 / 0.21 / 0.23 |
| 41-60 | 5,073 | 0.51 / 0.39 / 0.09 | −0.4 / +1.1 / −0.1 / +0.8 / +0.1 | 0.52 / 0.40 / 0.08 | 0.53 / 0.37 / 0.10 |
| **61+** | 4,855 | **0.56 / 0.44 / 0** | **−0.6 / −0.8 / −0.3 / −0.4 / 0.0** | 0.55 / 0.45 / 0 | 0.55 / 0.45 / 0 |

At 3-4 periods the pattern is the same: labels 0-20 rise +0.25 to +2.0 and 61+ fall −0.04 to −0.9. Labels 61+ stay at 0.76-0.77 in
the same bin.

- **At the dense-label pixels, which are exactly where R1 shows v1.0.0 too low, no V4r product raises tree cover.** They all move it
  slightly down, and the share a bin low stays at 0.44-0.46.
- **At open-land labels the V4r products raise it,** most of all the one-model versions: B1Z pushes 23% of 0-20 labels a bin up,
  against 16% for v1.0.0.
- **The seed control changes nothing,** as expected.

### Reassessed conclusions

1. **The blend does raise tree cover where clouds caused fallback,** by +4 to +5 points at 5+ periods for the gated blends. Gating by
   the zenith-tested count keeps the high latitudes out.
2. **The raise is broad, not targeted.**
   - On the maps it is as large in open land as in dense cover.
   - At independent label pixels it goes into open land and slightly away from dense cover, where v1.0.0 is too low.
   - The likely reason: the lowest-red pick is greener than the fallback mean everywhere, so the model reads more tree cover in any
     cover type.
3. **So the blend, as built, does not fix the problem it targets.** It shifts cloudy pixels up across the board rather than
   restoring dense cover.
   - Its higher agreement with C6 on the h12v09 drop pixels is consistent with this: the drop pixels are dense, and a broad raise
     helps there.
   - The labels show that the raise is not specific to dense cover.
4. **One caveat about the label sample:** only 4,855 dense held-out pixels at 5+ periods. The maps cover far more dense
   cloudy pixels (5.7 M with C6 ≥ 60), where the raise is +4 to +6.5. The labels are the independent check, but on few pixels.
5. **The earlier conclusion about BLEND-2's held-out MAE gain (ensemble averaging, sections 2a-2b) still stands,** but it was the
   wrong test for the question. This reassessment replaces the summary below as the answer to the project lead's question.

## Cover-gated blends (2026-10-03, project lead's question: blend only where v1.0.0 > 20%?)

**Method** (`code/gated.py`, outputs in `out/gated/`):
- **Weight** w = (zenith-gated V4r count / 8) × g, where g is the cover gate on v1.0.0's prediction.
- **Gates:**
  - none;
  - hard: v1.0.0 > 20, > 30, > 40;
  - ramp: from 0 at 20 to 1 at 50.
- **Blend partners:** V4R (two models), v1.0.0 on V4r composites (one model), and SEED1 (control).
- **Scoring:** held-out labels on all 63 tiles, by bin; maps of the 7 tiles.

**Maps, 5+ cloud-caused periods: mean change from v1.0.0, V4R blend**

| C6 class (n) | No gate | > 20 | > 30 | > 40 | Ramp 20-50 |
|---|---|---|---|---|---|
| C6 < 30 (4.9 M) | +5.6 | +2.6 | +1.5 | +0.6 | +1.2 |
| C6 30-59 (3.9 M) | +7.7 | +7.4 | +6.6 | +5.1 | +5.8 |
| C6 ≥ 60 (5.7 M) | +4.1 | +4.1 | +4.0 | +3.8 | +3.9 |

The one-model blend shows the same pattern but larger: C6 < 30 +8.7 → +1.7 at > 40.

**The gate does what it was meant to on the maps:** it removes most of the open-land raise (C6 < 30) and keeps the raise in dense
cover (C6 ≥ 60).

**Held-out labels at 5+ cloud-caused periods** (tiles v04 and south): same / lower / higher bin, and mean change from v1.0.0.

| Label (n) | v1.0.0 | V4R blend, no gate | V4R blend, > 20 | V4R blend, > 40 | Seed control, > 40 |
|---|---|---|---|---|---|
| 0-20 (13,544) | 0.837 / 0 / 0.163 | 0.831 / 0 / 0.169 (+0.30) | **0.866 / 0 / 0.134** (−0.19) | 0.837 / 0 / 0.163 (−0.06) | 0.837 / 0 / 0.163 (−0.04) |
| 21-40 (5,826) | 0.535 / 0.267 / 0.199 | 0.544 / 0.261 / 0.196 (+0.02) | 0.493 / 0.312 / 0.195 (−0.34) | 0.576 / 0.267 / 0.156 (−0.39) | 0.563 / 0.267 / 0.170 (−0.21) |
| 41-60 (5,073) | 0.514 / 0.394 / 0.092 | 0.518 / 0.398 / 0.084 (−0.11) | 0.518 / 0.398 / 0.084 (−0.32) | 0.463 / **0.453** / 0.084 (−0.65) | 0.479 / 0.433 / 0.089 (−0.29) |
| 61+ (4,855) | 0.558 / 0.442 / 0 | 0.549 / 0.451 / 0 (−0.28) | 0.549 / 0.451 / 0 (−0.37) | 0.548 / 0.452 / 0 (−0.54) | 0.554 / 0.446 / 0 (−0.18) |

**MAE difference from v1.0.0, all pixels:**

| Gate | V4R blend | One-model blend | Seed control |
|---|---|---|---|
| None | −0.021 | +0.043 | −0.036 |
| > 20 | −0.020 (43/3 tiles) | +0.025 | −0.031 |
| > 40 | −0.012 | +0.017 | −0.020 |
| Ramp 20-50 | −0.016 | +0.017 | −0.024 |

**Seams:** at neighbouring pixel pairs whose weights differ, the blends differ less than v1.0.0 for every gate (for example, V4R > 20:
9.03 against 9.69). So no gate edges appear.

**Reading:**
1. **The gate removes the side effect.** Open-land labels are no longer pushed up: at > 20, the share of 0-20 labels a bin too high
   falls from 0.163 to 0.134. On the maps, the C6 < 30 raise falls from +5.6 to +0.6-2.6.
2. **It does not create the wanted effect.**
   - At the independent dense labels (41-60 and 61+), the gated V4R blend lowers tree cover (−0.3 to −0.65), and more pixels fall a
     bin low: at 41-60 with the > 40 gate, 0.394 → 0.453.
   - V4R simply predicts lower than v1.0.0 at these labeled pixels, so focusing the blend on them focuses that.
3. **The maps disagree:** they show +4 at C6 ≥ 60. The labels say the raise in dense cover is not confirmed. A gate cannot resolve
   that disagreement.
4. **The small pooled MAE gains remain a two-forest averaging effect:** the gated seed control gains more than the gated V4R blend at
   every gate.
5. **One model on V4r composites stays worse with every gate.**

**Opinion (superseded in part by the correction at the top of this report):**
- A cover gate is a sound design element for any cloud correction, because it protects open land. A ramp, or a gate nearer 40,
  protects more.
- But V4R is not the right source for the raise: at the pixels where we can check independently, it does not raise dense cover.
- The cover-dependent correction learned by the counts model (C30 in `Trees_and_Counts_2x2_Results.md`) points the right way, but it
  is small.

## Earlier summary (by MAE; superseded by the reassessment above)

1. **One model on V4r composites does not work.** v1.0.0 applied to V4r composites without retraining (V1-V4C):
   - is exactly v1.0.0 where nothing was replaced (acceptance: 100%);
   - where composites were replaced, it is **clearly worse on the held-out labels**: +0.45 MAE overall, and +0.64 / +1.72 / +1.33
     at 1-2 / 3-4 / 5+ replaced periods;
   - it reads much higher than C6, C6.1 and the C code.

   So the one-model blend (BLEND-1) tunes itself to almost no V4r: weight only at 6-8 replaced periods, and no measurable change.
2. **The two-model blend (BLEND-2: v1.0.0 with the retrained V4R, weight = count/8) gains −0.023 MAE.** The gain is at 1-4 replaced
   periods (−0.05 to −0.06). At 5+ it is +0.02, with an interval spanning zero.
3. **The control shows that this gain is from averaging two models, not from V4r.** Blending v1.0.0 with a seed refit (the same
   composites, the same weights) gains about twice as much: −0.047 overall, −0.15 at 5+, −0.17 on dense pixels with 5+. **There is no
   evidence on the held-out labels that V4r composites help.**
4. **Maps (7 tiles):** the V4r-based products (V1-V4C, V4R, BLEND-2) raise tree cover strongly where many periods were replaced.
   - On the h12v09 drop pixels they move toward C6, but away from C6.1 and the C code.
   - The references disagree, so under the project's rule this is not a basis to act. As before, all three references are
     cloud-affected.
5. **Sharp decreases** (≥ 15 points below v1.0.0) are 1-4% of land.
   - **Not water or shadow:** in the tropics they are rarely near water (1-4%), and the picks there are greener than the fallback
     means they replaced.
   - **A fifth to two fifths of V4R's decreases are at pixels with no replaced period** (h11v08 28%, h19v08 24%, h19v09 41%,
     h26v06 20%). Those come from retraining, not from the composites.
   - **On h12v04 the winter picks look like snow** (red about 3,200, NDVI about 0.13), consistent with
     `Snow_vs_Cloud_V4r_Trigger_Check.md`.
6. **Side finding worth a test:** averaging v1.0.0 with a seed refit improves the held-out labels most where v1.0.0 is weakest: −0.15
   at 5+ replaced periods, better on 26 of 40 tiles. In effect that is a larger forest (60 trees), which may be a cheap improvement to
   v1.0.0 on its own.

## 1. Step 1: v1.0.0 on V4r composites; acceptance

- **Products:**
  - **V1-V4C:** v1.0.0's tree model (`qaC_full63_retrain_20260929/Python/models/C/rf`, MEDIAN, rounded) on the V4r metrics of
    `cloudy_v4r_weights_20261002`;
  - maps of the 7 tiles;
  - held-out predictions on all 63 tiles.
- **V4r replacement rule:** the existing one (`Snow_vs_Cloud_V4r_Trigger_Check.md` notes that, on northern tiles, it also fires on
  low sun).

**Acceptance (all passed, `out/acceptance.json`):**

| Check | Result |
|---|---|
| Held-out pixels with no replaced period (1,238,385): V1-V4C = v1.0.0 | 100% (h16v01 too) |
| Orientation of the count rasters against the held-out keys | count[x, y]: 100% agreement (the other orientation: 69%) |
| Maps, land pixels with no replaced period, 7 tiles: V1-V4C = v1.0.0 | 100% on every tile; non-land identical |
| Replacement counts recomputed under the existing rule = stored counts | 0 pixels differ (63 tiles) |

Where a period was replaced, V1-V4C equals v1.0.0 on only 27% of held-out pixels and on 11-14% of map land on the cloudy tiles.

**Questions (a) and (b) of the plan (which inputs count as cloudy):**
- Cloud value 3 never occurs in the inputs checked (`Snow_vs_Cloud_V4r_Trigger_Check.md`).
- **Adjacency matters:** without it, the held-out pixels with any replaced period fall from 1,051,605 to 719,976, and those with 5+
  from 32,474 to 11,861.
- Gating the blends on the count without adjacency changes little (section 2).

## 2. Step 2: blends on the held-out labels (63 tiles)

**Method:**
- **Weight:** w = 0 below t₀, rising linearly to 1 at t₁; value = round((1 − w) × v1.0.0 + w × other).
- **Tuning:** t₀ and t₁ were chosen on half A of the held-out pixels (1,145,169, by a fixed hash), from a grid of t₀ = 1-6 and
  t₁ = t₀-8, for each blend and each gate.
- **Scoring:** all scores below are on half B (1,144,821).
- **Intervals:** 95% bootstrap over tiles and over 100 × 100-pixel blocks.

**Chosen thresholds (on half A):**
- **BLEND-1** (with V1-V4C): t₀ = 6, t₁ = 8, the most restrictive choice in the grid. Same with the count without adjacency.
- **BLEND-2** (with V4R): t₀ = 1, t₁ = 8, i.e. w = count/8. Same with the count without adjacency.

### 2a. MAE difference from v1.0.0 (half B; negative = better)

| Product | All | count 0 | 1-2 | 3-4 | 5+ | Dense (label ≥ 60), 5+ | Tiles better / worse (all) |
|---|---|---|---|---|---|---|---|
| n | 1,144,821 | 620,091 | 354,152 | 154,287 | 16,291 | 3,029 | |
| V1-V4C | **+0.449** | 0 | +0.643 | +1.716 | +1.334 | +1.056 | 5 / 49 |
| V4R | +0.125 | +0.011 | +0.193 | +0.403 | +0.365 | +0.623 (interval spans 0) | 8 / 52 |
| BLEND-1 (6-8) | +0.001 | 0 | 0 | 0 | +0.066 | −0.036 (spans 0) | 7 / 22 |
| BLEND-2 (1-8) | **−0.023** [−0.035, −0.014] | 0 | −0.050 | −0.061 | +0.017 (spans 0) | +0.184 (spans 0) | 41 / 11 |
| BLEND-2, gate without adjacency (1-8) | −0.019 | 0 | −0.028 | −0.068 | −0.082 | +0.237 (spans 0) | 36 / 10 |
| SEED1 (yardstick) | −0.010 | −0.006 | −0.003 | −0.041 | −0.037 | −0.100 | 38 / 22 |
| SEED2 (yardstick) | −0.001 | +0.006 | +0.001 | −0.030 | −0.037 | −0.074 | 30 / 29 |

The count groups use the existing replacement count in every row.

### 2b. Control (added after the first results): is BLEND-2's gain from V4r, or from averaging two models?

v1.0.0 blended with each seed refit, using exactly BLEND-2's weights (w = count/8):

| Blended with | All | 1-2 | 3-4 | 5+ | Dense, 5+ | Tiles better / worse |
|---|---|---|---|---|---|---|
| V4R (= BLEND-2) | −0.023 | −0.050 | −0.061 | +0.017 | +0.184 | 41 / 11 |
| SEED1 | **−0.047** | −0.067 | −0.176 | **−0.148** | −0.172 | 45 / 7 |
| SEED2 | **−0.047** | −0.067 | −0.175 | **−0.160** | −0.164 | 47 / 5 |

All the seed-blend intervals exclude zero.

**Reading:**
- Averaging v1.0.0 with a second, equally good forest reduces error. The more weight it gets, the larger the gain, so the gain is
  largest where the weight is largest (5+ periods).
- Swapping that second forest for V4R gives half the gain overall and none at 5+.
- **So V4r's own contribution is negative on the held-out labels.** BLEND-2 looks good only because averaging two forests helps.

## 3. Step 3: blend maps (7 tiles)

The tiles are h11v08, h12v04, h12v09, h19v08, h19v09, h20v06 and h26v06. Pooled over pixels valid in every map and reference, MAE /
mean difference. C6, C6.1 and the C code are comparisons, not truth, and all three are cloud-affected in the same pixels.

| Set (n) | Reference | v1.0.0 | V1-V4C | V4R | BLEND-1 | BLEND-2 | SEED1 |
|---|---|---|---|---|---|---|---|
| All land (146.5 M) | C6 | 6.23 / +0.52 | 6.99 / +2.45 | 6.79 / +1.48 | 6.41 / +0.84 | 6.25 / +1.26 | 6.24 / +0.59 |
| | C6.1 | 6.16 / +0.54 | 7.06 / +2.47 | 6.88 / +1.50 | 6.38 / +0.86 | 6.27 / +1.27 | 6.15 / +0.61 |
| | C code | 7.01 / +1.32 | 7.99 / +3.24 | 7.73 / +2.28 | 7.22 / +1.64 | 7.18 / +2.05 | 7.01 / +1.39 |
| Dense (C6 ≥ 60), 5+ periods (5.7 M) | C6 | 7.17 / −1.56 | 7.42 / +4.97 | 7.22 / +4.08 | 6.59 / −0.20 | 6.11 / +2.54 | 7.01 / −0.96 |
| | C6.1 | 8.74 / +3.89 | 11.97 / +10.42 | 11.32 / +9.53 | 9.11 / +5.25 | 9.81 / +8.00 | 8.74 / +4.49 |
| | C code | 10.04 / +6.87 | 14.53 / +13.40 | 13.82 / +12.51 | 10.64 / +8.23 | 12.16 / +10.97 | 10.22 / +7.47 |
| All land, 5+ periods (14.5 M) | C6 | 11.00 / +5.47 | 17.00 / +15.23 | 15.02 / +12.71 | 12.75 / +8.68 | 13.37 / +11.03 | 11.05 / +5.85 |
| h12v09 drop pixels (103,011) | C6 | 19.44 / −18.86 | 6.90 / −0.11 | 7.65 / −2.66 | 15.52 / −14.68 | 8.84 / −7.31 | 18.70 / −18.08 |
| | C6.1 | 12.13 / −7.70 | 14.08 / +11.05 | 13.38 / +8.50 | 12.58 / −3.52 | 10.98 / +3.85 | 11.70 / −6.91 |
| | C code | 9.67 / −0.35 | 20.18 / +18.40 | 18.18 / +15.85 | 11.45 / +3.83 | 13.90 / +11.20 | 9.68 / +0.44 |

- **Every V4r-based product reads higher than v1.0.0, and the effect grows with the number of replaced periods.** At 5+ periods it is
  +10 to +15 points above C6 on all land.
- **On the h12v09 drop pixels:**
  - against C6, V1-V4C and V4R close the gap: −18.9 becomes −0.1 and −2.7;
  - against the C code, v1.0.0 already agrees (−0.35), and the V4r products overshoot by +16 to +18;
  - C6.1 is in between.
- **The references disagree,** so under the project's rule this is not a basis to adopt V4r.
- **The map pattern matches the held-out labels:** the V4r products push tree cover up where clouds replaced periods, and the labels
  say that makes them worse.

**Seams (`out/blend/maps/*_seams.csv`):** at neighbouring pixel pairs whose blend weights differ, the mean |difference| between
neighbours is:

| | v1.0.0 | the blend |
|---|---|---|
| BLEND-2 (39.9 M pairs) | 6.67 | 5.74 |
| BLEND-1 (5.1 M pairs) | 8.91 | 7.66 |

So the blends add no seams at the threshold edges.

## 4. Step 4: the sharp decreases (7 tiles)

Sharp decrease means product − v1.0.0 ≤ −15 on land; sharp increase means ≥ +15. Details are in `out/decreases/<tile>_decreases.json`.

| Tile | V4R decrease / increase (share of land) | V4R decreases with no replaced period | V1-V4C decrease / increase | Decreases within 2 px of water (all land) |
|---|---|---|---|---|
| h11v08 | 2.2% / 7.3% | 28% | 1.0% / 7.6% | 3% (1%) |
| h12v09 | 1.7% / 9.5% | 3% | 1.2% / 10.5% | 4% (2%) |
| h19v08 | 4.0% / 5.0% | 24% | 1.1% / 6.4% | 1% (1%) |
| h19v09 | 3.7% / 5.1% | 41% | 0.9% / 7.6% | 2% (1%) |
| h26v06 | 4.3% / 7.5% | 20% | 2.1% / 9.9% | 2% (2%) |
| h12v04 | 2.1% / 2.3% | 3% | 1.3% / 3.3% | 16% (11%) |
| h20v06 | about 0 | | 0 / 0 | |

**What the decreases are:**
- **V1-V4C's decreases** (composites only, same model) are 1-2% of land, against 3-10% increases. The change is mostly upward.
- **Tropical picks at decrease pixels look like vegetation, not water or shadow.**
  - Mean pick NDVI is 0.56-0.69, against 0.37-0.53 for the fallback means they replaced. Red is about half.
  - The decreases are rarely near water.
  - The model's response to the cleaner inputs goes down in these pixels, against the direction of the pick's greenness.
- **Adjacency-only replacements** (cloud flag only through bit 6) account for 10-40% of V4R's tropical decreases (17-53% for V1-V4C).
  That is similar to or below their share across all land on the same tiles (22-49%), so adjacency is not what drives the decreases.
- **h12v04 (northeastern US, Great Lakes)** is different. Its winter picks are bright with low NDVI (red about 3,200, NIR about
  3,900, NDVI 0.12-0.13), as expected for snow. Decreases there are more often near water (16% against 11% of land).
- **h12v09 east edge and floodplain:** V4R's decreases lean east (fifths of the tile from west to east: 11, 12, 26, 20, 30%). They are
  not concentrated at water (4%). By MCD12Q1 class they are 42% evergreen broadleaf forest, 26% savanna, 17% woody savanna and 13%
  grassland. A location-specific explanation would need the maps viewed by eye on ADAPT.

## 5. Points for the project lead

1. **The one-model approach (v1.0.0 on V4r composites) is not supported.** It is clearly worse on the held-out labels at every
   replaced count, and much higher than all three map references where many periods were replaced.
2. **The two-model blend's gain is ensemble averaging, not V4r.** A seed blend does better. On this evidence, no V4r blend is worth
   taking further.
3. **V4r, as a compositing rule, raises tree cover in cloud-replaced periods.** Only C6 agrees with that direction; C6.1, the C code
   and the held-out labels do not.
4. **Worth testing separately: a larger forest for v1.0.0.** Two 30-tree forests averaged gain −0.047 overall and −0.15 at 5+
   fallback-heavy pixels, better on 45-47 of 63 tiles. That suggests 60 trees (MEDIAN) could help v1.0.0 directly, independent of
   clouds.
   - **Cost:** fit time and model size double (about 30 GB per tree model).
   - **Test:** refit with `--nEstimators 60` and score on the same held-out pixels. That is about 20-30 min on one ngaproc node.
5. **Low sun on northern tiles** (see `Snow_vs_Cloud_V4r_Trigger_Check.md`) is mixed into these V4r results. It does not change the
   conclusions, which hold on the tropical tiles alone.

## Caveats

- Held-out scores are optimistic: the held-out pixels are a random 20% of pixels, neighbours of training pixels. There are only 3,029
  dense held-out pixels with 5+ periods in half B.
- Maps are on 7 tiles. The northern tiles where V4r fires most (h12v03, h24v03, h23v03, h11v03) are only in the held-out scores.
- Tree cover only. The non-vegetated layer and the SD layer were not part of this test.

## Where the results are

All under `FullRuns/v4r_blend_20261003/`:
- `out/acceptance.json`;
- `out/heldout/predictions.parq`;
- `out/blend/`: `heldout_grid.csv`, `heldout_scores.csv`, `heldout_diffs.csv`, `blend_choice.json`, `control_diffs.csv`, and per
  tile `maps/*_stats.csv` and `*_seams.csv`;
- `out/maps/`: V1-V4C, BLEND-1 and BLEND-2 maps of the 7 tiles;
- `out/counts/`: replacement counts under 4 cloud definitions, 63 tiles;
- `out/decreases/`;
- `code/`, `scripts/`, `logs/`, `status/`.
