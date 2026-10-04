# Cloudy pixels: retrained V4r, fallback counts and down-weighting (results)

**Status 2026-10-02 ~16:15: the run is finished (`FullRuns/cloudy_v4r_weights_20261002`, `status/ALL.done` 15:38; every gate and
acceptance test passed). The full results write-up is still to do. This file starts with the project lead's review notes, which the
write-up must follow.** Plan: `Cloudy_Pixels_Continuation_Plan.md`. Figures: `figures_cloudy_v4r_weights_20261002/` (28 figures,
BASE, BASE-C, BASE-W and V4R on 7 tiles, against C6, with the shares more than 15 points above and below C6).

**Update 2026-10-03:** the questions these notes raise were taken up in the follow-up reports, rather than in a full write-up of this
run:
- the sharp decreases: `V4r_Cloud_Blend_Results.md`, section 4 and the correction section;
- V4R's increases against the labels: the same report;
- snow and low sun: `Snow_vs_Cloud_V4r_Trigger_Check.md`;
- a cover-, cloud- and snow-gated V4r blend: `Gated_V4r_Blend_63Tile_Results.md`.

This file keeps the first-look numbers and the review notes. A full write-up of the 2026-10-02 run as such has not been done.

## Project lead review notes (2026-10-02, after the first look and the figures) — guidance for this report

1. **C6 is a comparison, not truth.** We have no true reference for these pixels. C6 is itself affected by clouds (its own fallback
   composites; `Cloudy_Fallback_Diagnosis_Results.md`), so a map moving away from C6 in cloudy forest is not by itself an error.
   The first-look wording that V4R "overcorrects" or is "too high" was too harsh: describe departures from C6 as *higher / lower than
   C6*, not as over- or under-prediction. The same holds for C6.1 and the C code, which share the cloud problem.
2. **The increases in tree cover under V4R look plausible to the project lead** (e.g. h12v09 and h11v08, where the cloud-driven
   drop pixels become dense forest). They are not to be counted against V4R just because they exceed C6.
3. **Concern: the areas with sharp decreases.** Where V4R is markedly lower than BASE or C6 (visible in the figures, e.g. along the
   h12v09 floodplain and the eastern edge), the cause needs to be understood before any conclusion. This needs more thought; it is
   an open question for the report, not a finding.
4. **No decision on V4R yet.** The report should lay out what is actually seen and what each reference can and cannot say.
5. **NGA (WorldView) data are also a comparison, not truth** (project lead, 2026-10-02). Tree canopy derived from very-high-resolution
   imagery is another remotely sensed estimate, with its own choices (thresholds, shadows, dates, sharpening) and errors. It is useful
   because it is independent of MODIS compositing and clouds, but report it as "WV-derived canopy", alongside C6, C6.1, the C code and the
   labels, never as ground truth. The same applies to AlphaEarth embeddings and any GEDI or lidar-derived values.

### What the report should therefore do (from the notes)

- Present map differences against C6, C6.1 and the C code as agreement with reference products, not as accuracy, and say each time
  that all three are cloud-affected in the same pixels.
- Give the held-out labels their real weight and limits: they are the only reference independent of the historical product, but they
  are few where clouds are worst (7,460 dense held-out pixels with 5+ fallback periods outside the high latitudes), and stage A found
  the labels themselves are associated with fallback (low labels are the most fallback-heavy).
- Separate V4R's **increases** (where, how large, under which fallback counts and land cover) from its **sharp decreases**, and
  characterise the decreases: where they are (floodplain, water edges, the eastern edge of h12v09, other tiles), their fallback
  counts, land cover (`out/strata/*_landcover.csv`), and whether the V4r picks there look unusual (e.g. darker NIR, water or shadow
  that passed the guards). Candidate checks for the project lead to choose from; none run yet.
- Keep the first-look numbers (below) as numbers, without the "over-prediction" reading.

## First-look numbers (2026-10-02, recorded before the write-up)

- **Held-out labels, all 63 tiles** (MAE difference from BASE; seed refits differ by −0.01 / 0.00):
  - BASE-C −0.03, better on 51 of 63 tiles;
  - BASE-W +0.004;
  - V4R +0.13, better on 9 of 63 tiles.
- **Dense labels (>= 60), tiles v04 and south, 5+ fallback periods (7,460 pixels):**
  - BASE-C −0.18, better on 12 of 22 tiles;
  - BASE-W +0.36, better on 8 of 22;
  - V4R +0.60, with an interval spanning zero, but better on 17 of 22 tiles.
- **h12v09 held-out MAE:** BASE 6.49, BASE-C 6.25, BASE-W 6.58, V4R 6.59. Without retraining, V4r was +1.9 here.
- **h12v09 drop pixels, mean difference from C6:**
  - BASE −18.9, BASE-C −16.6, BASE-W −19.1, V4R −2.7;
  - seed refits −18.1 and −18.4.
- **h12v09, all land:**
  - mean difference from C6: BASE +2.5, V4R +5.1;
  - share more than 15 points above C6: 5.9% → 12.2%;
  - share more than 15 points below C6: 2.5% → 1.1%.
- **Seed noise at the pixel level:** two seed refits differ by about 4 points MAE per pixel, while their tile means differ by about 0.1.

Raw results: `FullRuns/cloudy_v4r_weights_20261002/out/heldout/` (scores.csv, differences.csv) and `out/strata/` (per tile, by
fallback count, h12v09 drop/control, land cover).
