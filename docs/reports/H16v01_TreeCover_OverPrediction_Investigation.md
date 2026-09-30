# h16v01 tree-cover over-prediction: investigation, left unresolved

**See also `Zenith_Only_Fix_63Tile_Plan.md`** for the broader check this section's finding led to: the
same zenith-only fix run across 31 of 63 tiles and all three models sharing the 64-predictor list
(baseline, Random Forest, XGBoost tunedB), in progress as of 2026-09-28 16:30.

**PROJECT LEAD'S DIRECTION (2026-09-28, after the cutoff-70 retest): scope the fix to Greenland, and consider Iceland.**
h16v01 is a Greenland tile. The project lead's view is that applying this zenith-only NDVI-default fix
only to Greenland tiles is likely to be the solution. Iceland tiles should be considered for inclusion
as well. This has not been designed or tested yet. Points to check when it is:
- **None of the 10 tiles the fix hurt is in Greenland or Iceland** (h11v02, h11v03, h18v03, h18v04,
  h20v02, h20v03, h21v02, h23v02, h23v03, h24v03 are northern Canada, Europe and Russia). That is
  consistent with the direction, but it is not a test of it.
- **The C code already scopes a rule this way.** `MOD44B3.c` lines 411-421 (the "Greenland" block, MSZ
  4/30/2026) sets non-vegetated to 100 when some metrics are zero, but only for `vtile < 3 && htile < 18`.
  That box also contains h11v02, one of the hurt tiles, so a Python scope copied from C's box would
  include a tile the fix damages. The list of Greenland (and Iceland) tiles needs to be explicit.
- **The tile list is still to be written.** Iceland is about 63-67 N and 13-24 W, which should put it in
  h16v02 and possibly the edge of h15v02; confirm against the tile grid. Check which of these tiles are
  among the 63 run tiles and the 31 in the 65-degree test.

## Zenith-only NDVI-default variant (2026-09-28, later the same day) - a real candidate, not yet applied

A **zenith-only variant** of the disqualified NDVI-default fix was tested and looks promising. Instead
of zeroing NDVI for *any* QA failure (the full-QA version below, disqualified because it broke h12v09),
it zeros NDVI for a pixel/period only when **no** observation in that period had a valid solar zenith
(0-65 degrees), ignoring cloud/shadow/aerosol/adjacency entirely - on the theory that h12v09 was being
hurt by real clouds over a real forest legitimately failing QA, not by the zenith mechanism specifically.

**Step 1, h16v01 itself: resolves most of the over-prediction.** Tile-wide mean tree cover 5.66
(unfixed) -> 1.47 (zenith-only) vs. 0.85 (full-QA, disqualified) - most of the over-prediction resolved,
though not quite as completely as the full-QA version. Pixels >20%: 1,186,258 (8.2%, unfixed) -> 18,501
(0.13%, zenith-only). Script: `h16v01_make_ndvi_zenith_only.py`; launcher: `run_ndvi_zenith_detached.sh`;
output: `expA/out_ndvi0_zenith/Percent_Tree_Cover-h16v01-2019.tif`.

**Step 2, the validation on h12v09: PASS, but a clean pass, not a narrow one.** Ran the same
build-modified-NDVI -> predict -> compare-against-real-labels sequence used to disqualify the full-QA
version, this time with the zenith-only rule, against h12v09 (real training tile, mean label 76.8).
Launched detached from forest203 at 2026-09-28 15:08 (PID 1525121, parent PID 1), finished 15:23 (about
14 minutes), rc=0 at every step. **Result: the zenith-only rule zeroed 0.00% of pixels in every one of
h12v09's 8 periods** - it never fires on this tile at all, at any period. Prediction output, whole-tile
stats, and accuracy against the 24,860 real training labels are therefore bit-for-bit identical to the
real unmodified run (MAE 6.00, bias +0.16, corr 0.5607, diff mean/std exactly 0.00 everywhere). This is
not a weaker result than it sounds: it confirms the original hypothesis directly - h12v09's real QA
failures are ordinary clouds/shadow/aerosol at a normal latitude that never all exceed 65 degrees
zenith at once, so the zenith-only rule is cleanly selective for the high-latitude structural mechanism
found on h16v01 and provably does not touch a normal-latitude forested tile's real signal. Full compare
output: `expA/compare_ndvi0_zenith_h12v09.log`; per-period zero counts: `expA/build_ndvi_zenith_h12v09.log`.

**Net effect and what remains before this could be adopted as a real fix (not done, not decided):**
h16v01 tile-wide mean 5.66 -> 1.47 and pixels >20% 8.2% -> 0.13%, with h12v09 completely unaffected
(0.00% of pixels touched, identical accuracy). Only tested on these two tiles and only for 2019 - not
run across the other 61 tiles, not compared against the historical MOD44B product, and the small
residual over-prediction the fuller (disqualified) fix caught but this one doesn't (see "residual" in
the summary below) is still open. No code has been changed in the real pipeline; everything here was
read-only, in a prior session's scratchpad, against the real, unmodified `pcttree.bin` and metrics cache:
`/panfs/ccds02/nobackup/people/mcarrol2/.nccstmp/claude-267298826/-panfs-ccds02-nobackup-projects-ilab-projects-MODIS-VCF-pge61-08312026-pge61/cfb2496e-dede-49f4-9d80-ded423d651cb/scratchpad/`
(reachable by this path from a fresh session; scripts `h12v09_make_ndvi_zenith_only.py`,
`h12v09_compare_zenith.py`, driven by `run_ndvi_zenith_h12v09_detached.sh`).

**Status: root cause understood and confirmed, but the first candidate fix does not generalize and is
disqualified (2026-09-28).** Root cause: 5 of h16v01's 8 annual compositing periods are 100% unfiltered
tile-wide (zero QA-passing observations anywhere on the tile) and the rest are 68-86% unfiltered - a
structural consequence of the fixed 65-degree zenith cutoff at 69-80 degrees N, not a narrow bug. NDVI
computed from these unfiltered periods is the strongest statistical driver of the over-prediction and
has no counterpart in this tile's real training labels. A candidate fix - defaulting NDVI to 0 at any
pixel/period with no QA-passing observation - resolved the great majority of h16v01's over-prediction
(tile-wide mean 5.66 to 0.85) when tested through the real, unmodified fitted model, **but the same fix,
tested on h12v09 (a real training tile with substantial tree cover), degrades prediction quality badly**
(MAE against real labels 6.00 to 28.91; correlation 0.56 to 0.24) - it cannot tell "no real signal to
lose" (true on h16v01) from "discarding genuine green-up signal" (true on h12v09). The fix as designed
is disqualified; the underlying diagnosis stands. The originally-suspected snow/missing-file mechanism
(below) is real but explains only a minor share of the effect the (disqualified) larger fix captured. A snow-plus-extreme-solar-zenith spectral artifact (real observations show a
vegetation-mimicking red-suppressed ratio in two specific 48-day windows) was confirmed as a genuine,
causal contributor by directly re-running the real, unmodified fitted model against edited real metrics
- but it only explains about 11% of the over-prediction. A direct statistical comparison (rank-based
AUC across all 64 predictors, real over-predicted pixels vs. a large sample of correctly-predicted
pixels) found NDVI-family predictors are by far the strongest separator (AUC up to 0.90), present at
clean, non-contaminated periods throughout the year, not just the two implicated windows - while the
contaminated family barely separates the two groups at all. **Essentially none of h16v01's 158,912
real, confirmed-zero training labels reach the NDVI range the over-predicted pixels sit in** (0.0% for
the strongest predictor) - the model has no local, correctly-labeled example to learn from here, and is
extrapolating from the other 62 (more vegetated) training tiles. **Root cause of the NDVI elevation
itself, confirmed the same day: the strongest predictor's window (`Day_2019065`) is, tile-wide, 100%
unfiltered data** - solar zenith (57-93 degrees) never passes the 65-degree QA cutoff for any pixel at
this extreme latitude in early spring, so `CompositeDayFile`'s Case-2 fallback (meant for rare bad
periods) applies to the entire tile for this window, exactly like the already-known missing-file
windows, but for a structural, latitude-driven reason rather than an absent archive file - any
sufficiently high-latitude tile's early/late-season window could have the same property. This is a
training-representation gap layered on top of a genuinely unscreened-data problem, not a narrow,
patchable metrics or compositing bug confined to one tile. No code was changed. Written 2026-09-27, in the regression-tree-methods session
(`Regression_Tree_Methods_RF_XGBoost_Plan.md`, `Regression_Tree_Methods_RF_XGBoost_Results.md`);
extended 2026-09-28.

## The question

h16v01 (69-80°N, Canadian Arctic Archipelago through central/east Greenland) should realistically have
virtually no tree cover. Its actual training labels confirm this: all 158,912 real sample points in
this tile are valued exactly 0 (`v5.0.3samp`/`v5.0.3` - the same file under two directory names,
checksums identical). Yet the baseline (Bagging+DecisionTree), Random Forest, and XGBoost tunedB
models all predict non-trivial tree cover on a meaningful fraction of the tile: baseline has 1,186,258
pixels above 20% (max 72), Random Forest 127,216 (max 65), XGBoost tunedB 642,813 (max 60) - all on
land, away from water, away from any training-labeled pixel, where both C and the historical MOD44B
agree the answer is ~0. This document traces what's behind that, without reaching a resolved answer.

## What's established

1. **At the training-label locations specifically, all three models are accurate** (mean error
   0.006-0.09, over 96% exactly 0). The over-prediction is confined to the ~83-99% of the tile with no
   direct training label, where the models must generalize from the pooled 63-tile training set.
2. **Ten pixels representative of the baseline's >5% "pool"** (5th-95th percentile of its predicted
   value) were selected and traced through the 64-predictor feature set, using the same `Metrics`
   class the model itself uses. Nine of the ten show no distinct greenness signal (NDVI near-zero,
   indistinguishable from confirmed-zero training pixels) - the tenth (labeled p85 in the trace)
   shows a genuinely elevated NDVI across multiple independent metrics at once, unexplained by
   anything below and possibly a real, separate case (coastal vegetation the historical product's
   call missed, or a different surface artifact) - not investigated further.
3. **Band_31's zenith clamp** (`BandDayFile.ZENITH_CLAMP = 16000` where `solz > 70°`) fires on the
   large majority of pixels in 5 of the tile's 8 monthly periods, feeding into 10 of the 64
   predictors (`CThermMean*-Band_31`, `CGreenMean*-Band_31`, `CGreenMax-Band_31`,
   `CSortedRank{0,1}-Band_31`). **Checked directly against C's real `metrics.bin` for this tile
   (not just Python's mirror) and ruled out as the explanation**: C sees the identical degeneracy at
   the same pixels and still predicts correctly. Since it doesn't differentiate C from Python, it
   can't explain why Python specifically over-predicts. Best remaining guess, **not confirmed**: a
   training-representation gap (C's tree-cover model is calibrated over a longer operational history;
   Python's models here were trained from scratch on one pooled year, 2019, across 63 tiles, which may
   under-represent this input pattern) - this was not tested.
4. **Broadened to check all 64 predictors' Python-delivered-vs-C-real divergence at the same 10
   pixels** (not just the Band_31 ones). The largest, most systematic divergence: `Band_1`/`Band_2`
   at `Day_2019305` and `Day_2019353` - Python reads ~3333-3374 at every sampled pixel; C reads
   **exactly 0** at every one. This propagates into `CUnsMean79-Band_1/2` (the mean of periods 4 and
   5, which include day 305) and is plausibly why the baseline over-predicts more aggressively than
   RF or tunedB, though that link was not confirmed quantitatively.
5. **Traced the ~3333 mechanism to `CompositeDayFile._computeComposite()`'s two-case fallback**, and
   confirmed it is a specific, narrower manifestation of **already-documented, already-analyzed
   project history** - not a new bug:
   - Days 305 and 321 (2019) show the exact `Band1_raw=0, solz=0, state=0` signature that
     `Polar_Night_Zero_Observation_Finding_h12v02.md` identified and traced to the upstream MOD44CQ
     compositor's `obscov==0` ("zero MODIS observations this period") condition, confirmed at day 321
     for all 10 sampled pixels and at day 305 for 6 of 10 (the other 4 have real, genuinely
     high-zenith readings, `solz` 89-92°).
   - Day 337 (2019) has no source file at all for this tile (confirmed against the actual archive
     directory listing), triggering `CompositeDayFile`'s missing-file substitute
     (`SUBSTITUTE_PIXEL_VALUE = 10000`).
   - **C's validity check is zenith-only** (`zenith < 65°` in `fix_bands.c`); a placeholder's
     `solz=0` trivially passes it, so C accepts the fake zero as a genuine low-zenith observation -
     giving its window average ~0, matching the real `metrics.bin` value exactly.
   - **Python's `createQaMask()` additionally requires `solz > 0`**, correctly rejecting the
     placeholder. That is **already on record as the scientifically correct side** of this
     disagreement (`Polar_Night_Zero_Observation_Finding_h12v02.md`: "already endorsed as Python
     being the more correct side"). This investigation is not revisiting that conclusion.
   - **The narrower, more specific residual this investigation adds**: rejecting the placeholder
     leaves zero valid sub-periods, which trips the Case-2 fallback (average all three raw values
     regardless of QA) - and that fallback does not distinguish "a day with real, if degraded, data"
     from "a day whose file never existed at all." Day 337's fabricated 10000 gets averaged in on
     equal footing with the real (if placeholder-zero) readings from 305/321, landing the result near
     3333 instead of near 0. The math reproduces every sampled pixel's value exactly, including the
     small pixel-to-pixel spread from day 305's real-vs-placeholder split.

## What's explicitly not established

1. **Whether this Case-2-fallback nuance actually drives the model's over-prediction**, or is a
   correctly-identified but low-impact artifact. Not checked against `Feature_Importance_Part_A_Results.md`
   or any other measure of how much these specific predictors matter to the fitted model.
2. **Whether "fixing" this would even be correct.** C's 0 is not ground truth either -
   `Polar_Night_Zero_Observation_Finding_h12v02.md` says so explicitly ("C's 0 is arguably the less
   physically meaningful number of the two here"). Both C's 0 and Python's ~3333 are synthetic
   fallback values for a period with no reliable observation; neither is a real reflectance
   measurement. There is no confirmed "correct" value to converge on here, only two different wrong
   defaults.
3. **The p85 pixel's elevated NDVI signal** (point 2 above) - not explained, not investigated beyond
   noting it looks different from the other 9.
4. **Why Random Forest is consistently the least aggressive over-predictor** of the three models
   (fewest high-value pixels, lowest max) - not investigated; could be the feature-subsampling at each
   split, could be something else.
5. No code changes were proposed or made from this investigation. It does not affect or call into
   question the two fixes implemented and validated earlier in this session (`Percent_NonVegetated`/
   `Percent_NonTree_Vegetation` code propagation and the prediction clamp,
   `NonTreeVeg_Water_OutOfProj_Code_Propagation_Plan.md`) - those are separate, resolved, and
   unaffected by anything here.

## Method note: reused validated project tooling rather than re-deriving

Per the project lead's instruction mid-investigation, this used `CAttributes.py`/`CMirrorMetrics.py`
and the exact `metrics.bin` record format (`row`, `col`, 324×int16) from the already-validated
`validate_cmirror_vs_c.py` (see `OptionC_ImplementationB_Validation_Results.md`) to read C's real
per-pixel metrics, rather than reconstructing the byte layout independently. The `Metrics` class's
`getMetricFromRf()` was used to pull Python's actual delivered predictor values, the same code path
the real prediction pipeline uses.

## Where things are

C's real metrics for h16v01: `/explore/nobackup/people/mcarrol2/MOD44B_VCF_Development/VCF_outputs/Metrics/h16v01.2019.metrics.bin`
(read-only, shared project data, not modified). The one-off scripts used to query it and trace the 10
pixels were written to this session's scratchpad and were not preserved - the findings above are
complete enough to reproduce them (predictor names, C column resolution via `CAttributes.familyColumn`,
pixel row/col values are all given in this document and the conversation it summarizes).

## Follow-up (2026-09-28): the snow-bit hypothesis, checked against real data

**Hypothesis tested** (the project lead's): the over-prediction is associated with high-solar-zenith
observations over snow, where snow's reflectance reads anomalously low and ends up simulating
vegetation; `SUBSTITUTE_PIXEL_VALUE = 10000` was meant to counteract this for missing days by reading
"bright/snow," but evidently has not worked in these cases.

**QA-bit mapping used, and its provenance.** The archived MOD44CQ `state` SDS is not raw MOD09
`state_1km` passed through — PGE72's `add_SDS_250m.c` repacks it at write time
(`C_code/pge61/MOD_PR44B_Metrics/C_vs_Python_Code_Differences.md`, finding 2), moving original bits
12-14 (snow/adjacency/brdf) to new bits 5-7. That document's adjacency (bit 6) and aerosol (bits 7-8)
reads were tested against a real MOD09GA granule and decided as canonical (2026-09-14/15) - already
what `ProductTypeMod44.createQaMask()` and C's `fix_bands.c` both use. **Snow was never part of that
contested finding** (neither C's B1 nor Python's QA currently reads it), so it was not put through
that same real-granule pixel test. This session used **bit 5** for snow: it falls out of the same
repack-formula hand-decode that produced the validated adjacency/aerosol answers, and is independently
corroborated by `MOD_PR44B3/fix_bands.c:212`'s own `snow = state & 32` - two independent sources
agreeing, but not itself pixel-validated the way adjacency/aerosol were. A sanity check against bit 12
(the MOD09-standard position, wrongly assumed at first) gave physically implausible results (0% snow
tile-wide even on a fully-observed Arctic summer granule, ~50% "fire") and was discarded; bit 5 gives
plausible numbers (20-23% snow on h16v01 in July; 93% on h12v02's real observations on the already-
studied polar-night day 337).

**What was checked, read-only, against the real archive** (`/css/modis/Collection6.1/L3/MOD44B-VCF/dev`,
via GDAL on forest203) **and the real 63-tile-run prediction** (`Percent_Tree_Cover-h16v01-2019.tif`):

1. **The two windows already implicated** (`Day_2019305` = raw days 305, 321, missing-337;
   `Day_2019353` = missing-353, 001, 017) were recomputed at full tile scale, not just the 10 sampled
   pixels. Confirmed: **99.9% of all 1,186,258 over-predicted (>20%) pixels** have both `Band_1` and
   `Band_2` Case-2 composite values within +/-200 of exactly 3333, in both windows - the arithmetic
   signature of "two near-zero real/placeholder days plus one fabricated 10000, divided by three,"
   reproduced at full scale, not an artifact of the earlier hand-picked sample.
2. **Within the over-predicted mask, on day 2019-305** (real, `obscov>0` observations, solar zenith
   84-93 degrees): **84.6% carry the snow bit.** Snow-flagged pixels there read `Band_1` (red) mean 11,
   `Band_2` (NIR) mean 45 - a roughly 4:1 NIR:red ratio. The non-snow-flagged minority in the same
   zone, at the same zenith, reads `Band_1` mean 56, `Band_2` mean 76 - a roughly 1.4:1 ratio. A pure
   illumination-geometry effect (less downwelling light at grazing incidence) would dim both bands by
   about the same fraction and leave the ratio roughly constant with zenith; the sharp difference in
   ratio between snow-flagged and non-snow-flagged pixels at the *same* zenith is the specific,
   quantified evidence for the hypothesis - a vegetation-mimicking red-suppression tied to the snow
   flag, not generic low-light noise.
3. **On days 2020-001 and 2020-017**, the real (`obscov>0`) observations feeding the over-predicted
   zone are almost universally snow-flagged (99.6% and 100%). Day 2020-017 additionally showed a small
   number of large-magnitude negative `Band_1`/`Band_2` values coexisting with `obscov>0` (mean -905/
   -704 against a median of 11/15) - a real, separate data-quality wrinkle (an apparent fill/sentinel
   value leaking into nominally-valid observations), not yet explained, flagged here rather than
   pursued.
4. **On the near-zero real readings specifically**: day 305's real (non-placeholder) observations in
   the over-predicted zone already read `Band_1` ~= 11-17 on their own, almost as low as the `obscov==0`
   upstream placeholder's exact zero. So the `SUBSTITUTE_PIXEL_VALUE = 10000` "bright/snow" fallback is
   not being diluted by genuinely bright, legitimate snow reflectance - it is being diluted by real
   observations that are *themselves* already anomalously dark under this specific
   snow-plus-extreme-zenith condition. Whether that darkness is itself a retrieval artifact (this
   session's evidence, e.g. the red-vs-NIR ratio split, points that way) or a real physical floor of
   very little downwelling light at ~90 degrees zenith was not resolved either way.

**Net assessment (superseded in part by the causal test below).** The hypothesis is well supported by
data at full tile scale, not just a plausible story: extreme-zenith, snow-flagged real observations
show a distinctive, vegetation-mimicking red-suppressed spectral ratio, and the pixels that end up
over-predicted are overwhelmingly exactly this population. What remained unconfirmed was the last link
in the chain - that this specific mechanism, as opposed to something else, actually drives the fitted
model's over-prediction. That link was tested directly the same day; see below. No code was changed.

**If the snow-bit thread specifically is picked up again:** check whether the snow-flagged
red-suppression pattern found here on h16v01 recurs on other tiles/periods where it would matter (i.e.,
is this tile-specific or a general high-solar-zenith-plus-snow artifact the model should be guarded
against generally), and, if it is confirmed to matter, consider whether `CompositeDayFile`'s Case-2
fallback should treat a QA-mask-driven zero (real snow/zenith rejection) and a totally-missing source
file differently, rather than averaging both into the same raw mean alongside the 10000 substitute.
(Superseded in practice by the tile-wide NDVI-default fix further below, which addresses this and much
more at once - kept here for the record.)

**Where things are (snow-bit follow-up).** Read-only scripts written to this session's scratchpad (not
preserved in the repo): one decodes the MOD44C-repacked `state` QA bits and cross-tabulates the snow
bit against the over-predicted mask and `Band_1`/`Band_2` at the four real-data days in the two
implicated windows; a second recomputes the Case-2 fallback composite at full tile scale for both
windows and checks it against the over-predicted mask. Both reused the real archive at
`/css/modis/Collection6.1/L3/MOD44B-VCF/dev` and the real 63-tile-run prediction output
(`FullRuns/full63_c_vs_py_20260919-0806/Python/predictions/Percent_Tree_Cover-h16v01-2019.tif`);
nothing was written to either. Run via `singularity exec` on forest203 (the `modis-vcf-latest`
container), binding `/css/modis` read-only per the project's existing launcher convention
(`code/validation/pilot_metrics_node.sh`).

## Causal test (2026-09-28, same day): fixing the composite-fallback dilution alone recovers only ~11%

**Design.** Ran the real, unmodified prediction code path (`predict_tree_cover_model.py` ->
`VcfPredict._runPrediction` -> the already-fitted `pcttree.bin`, unchanged) twice against h16v01,
through `singularity exec` on forest203, following the project's existing launcher convention
(`code/validation/run_predict_model_node.sh`):

1. **Sanity check**, real metrics bound read-only, no changes: reproduced the real 63-tile-run
   `Percent_Tree_Cover-h16v01-2019.tif` **exactly**, pixel for pixel (100.000% identical, max abs diff
   0) - confirms the harness is faithful to the real pipeline before trusting the counterfactual below.
2. **Counterfactual**, a scratch copy of h16v01's real `2-Composites` cache (192 files, 6.8 GB) with
   only 4 files replaced: the `Band_1`/`Band_2` composites for the two implicated windows
   (`Day_2019305`, `Day_2019353`), recomputed to drop the fabricated `SUBSTITUTE_PIXEL_VALUE=10000`
   from the Case-2 average entirely and use only the real days (`mean(real_305, real_321)` and
   `mean(real_001, real_017)`, floored at 0 to exclude a separately-flagged fill/sentinel leak on day
   2020-017 from dominating the mean). All 60 derived-metric cache files (`3-Metrics/*.tif`, the NDVI
   cache) were deleted so they recompute fresh from the edited composites through the real
   `CMirrorMetrics.py`/`Metrics.py` code - this also correctly propagates the fix through the
   sorted/reordered families (`CGreenMean*`, `CThermMean*`, `CSortedRank*`), which a hand-picked
   predictor edit would have missed, since a contaminated period can sort to any rank depending on its
   NDVI/Band_31 value.

**Result, within the real run's 1,186,258 over-predicted (>20%) pixels:**

| | mean | median | max | fraction >20 |
|---|---|---|---|---|
| Real (unmodified) | 33.94 | 35.0 | 72 | 100% |
| Cleaned composites | 30.05 | 31.0 | 69 | 84.6% |

Fixing the fabricated-substitute dilution alone drops the mean by 3.9 points (about 11% relative) and
moves only 1.5% of pixels down to a near-zero (<=5) prediction. **84.6% of the over-predicted area is
still over-predicted after the fix.** This is a real, reproducible, non-zero effect - not nothing - but
it is not close to explaining most of the phenomenon.

**Interpretation.** The composite-fallback dilution (averaging a fabricated 10000 in with real
readings) is a genuine, minor contributor, not the dominant one. The "cleaned" composite still carries
whatever the real, snow-flagged, extreme-zenith `Band_1`/`Band_2` readings themselves contain - this
experiment only removed the fabricated missing-file value from the average, it left the real (if
red-suppressed) observations exactly as they were. That the model's output barely moved suggests the
earlier-found spectral anomaly (snow-flagged pixels reading a ~4:1 NIR:red ratio versus ~1.4:1 for
non-snow at the same zenith) is doing most of the work **on its own**, independent of whether a
fabricated substitute is blended in - i.e., the real observations under this snow-plus-extreme-zenith
condition are themselves the primary driver, not an artifact of the missing-file handling. This
reframes which fix candidates are worth pursuing: option 2a (stop blending the fabricated substitute
into Case-2) is validated as correct and worth keeping, but it will not by itself resolve the
over-prediction; something closer to option 3 (treat snow-flagged, high-zenith real observations as
untrustworthy too, not just missing days) is now the better-supported next thing to test.

**Not yet done:** the natural follow-up is the same experiment extended one step further - also
exclude/zero the real snow-flagged, high-zenith (`solz` in the high-80s/90s) observations within the
two windows (not just the fabricated substitute), and see whether that closes more of the remaining
84.6%. This reuses the same harness and costs about the same (minutes, no retrain).

## Follow-up experiment, same day: does a stronger "bright/snow" substitute help?

**Question** (the project lead's): `SUBSTITUTE_PIXEL_VALUE = 10000` was designed to read as bright/snow,
but the Case-2 average dilutes it to about 1/3 strength (~3333) by blending it with the real days. Does
using the full, undiluted value help, as the original design intended? Also raised: excluding/NaN-ing
the contaminated predictors was considered and rejected up front, since `pcttree.bin`
(`BaggingRegressor(DecisionTreeRegressor)`) cannot accept NaN input (`predict()` raises `ValueError:
Input contains NaN`, not a NaN output) - not tested further because it isn't viable without an
imputation step the current code doesn't have.

**Design.** A third variant, same harness: `MOD44-h16v01-2019{305,353}-{Band_1,Band_2}.bin` replaced
with a **flat 10000 across every pixel of the raster**, not filtered by a freshly-recomputed per-pixel
QA mask. This is a reasonable simplification specifically for these two windows: solar zenith was
already confirmed (see above) to run 84-94 degrees across the *entire* tile on all four real days
feeding these windows, and the zero-observation placeholder pixels fail QA on `solz>0` too - so no
pixel anywhere in h16v01 can pass QA on any of these four days, meaning Case-2 already applied
tile-wide in the real data for this window (not verified pixel-by-pixel for this specific write, e.g.
water pixels weren't individually checked, but it doesn't matter for those - `VcfPredict` overwrites
them with the 200/250 codes downstream regardless of the composite value).

**Result**, within the same 1,186,258-pixel over-predicted mask:

| | mean | median | max | fraction <=5 | fraction >20 |
|---|---|---|---|---|---|
| Real (diluted ~3333) | 33.94 | 35.0 | 72 | 0.0% | 100% |
| Cleaned (real-only avg, ~100-130) | 30.05 | 31.0 | 69 | 1.5% | 84.6% |
| Bright (flat, undiluted 10000) | 33.98 | 35.0 | 71 | 0.3% | 90.3% |

**The undiluted bright value is statistically indistinguishable from the real, diluted one** (mean
33.98 vs 33.94, median identical, mean difference +0.04) - going all the way to 10000 instead of the
diluted ~3333 barely moves the prediction at all, and if anything is very slightly worse by the `>20`
count (90.3% vs the clean variant's 84.6%). Only pushing the value down toward what the real
(if red-suppressed) observations actually read (the "clean" variant, ~100-130) produced any real
movement, and even that was a modest 11%.

**Interpretation.** The model's response to these predictors is flat across the whole high range
(roughly 3,300 to 10,000) - whatever these predictors are doing, both a diluted and an undiluted
"bright" reading route to essentially the same output. This rules out "make the substitute brighter"
as a useful fix direction on its own: the original design intent (dilution weakening an otherwise-
correct bright signal) does not hold up - a full-strength bright signal fails the same way. The 11%
recovered in the "clean" experiment is the only lever these two windows' `Band_1`/`Band_2` values have
shown so far, and it came from going low, not from going bright. This further supports the earlier
reframing: the real driver is more likely the intrinsic snow-plus-extreme-zenith spectral artifact
(and/or other predictors/periods not touched by this experiment) rather than anything about how the
missing-file substitute is chosen or diluted. NaN was considered and rejected up front: `pcttree.bin`
is `BaggingRegressor(DecisionTreeRegressor)`, which raises `ValueError: Input contains NaN` rather than
producing a NaN or otherwise graceful output - not viable without an imputation step the pipeline does
not have.

## Decision-path tracing (2026-09-28, same day): does one tree's routing point to a single culprit metric?

**Question.** With the ablation experiments only explaining a modest slice of the phenomenon, can
tracing individual pixels through the fitted model's actual tree structure identify which predictors
are really deciding these predictions - more direct evidence than importance rankings (which average
over the whole 63-tile training set) or ablation (which only tests the two windows already suspected)?

**Method.** `pcttree.bin` is confirmed unmodified: `BaggingRegressor` of 30 unpruned
`DecisionTreeRegressor`s (`max_depth` reported by `tree.get_depth()` = 90, 3.5 million leaves in the
one tree inspected first), no per-tree feature subsampling (`estimators_features_` covers all 64
predictors for every tree - `bootstrap_features` is off). For a given pixel, `sklearn`'s
`tree_.feature`/`tree_.threshold`/`children_left`/`children_right` arrays were walked from the root to
the pixel's leaf, printing the predictor, threshold, the pixel's real value (from the unmodified
metrics cache, the same `getMetricFromRf()` code path `VcfPredict` uses), and which branch was taken.

**First pass: one tree (index 0), 10 pixels** spanning deciles of the over-predicted value distribution
(21 to 72, real row/col given so this is reproducible, unlike the original investigation's unpreserved
10-pixel sample):
- 4 of 9 successfully-traced pixels (1 hit a reproducibility anomaly, below) pass directly through the
  contaminated `Day_2019305`/`Day_2019353`/`CUnsMean79` predictors at their real (diluted ~3333) values,
  confirming they are genuinely used as splits, not just theoretically implicated.
- **But the thresholds tested are far below the contaminated value** (e.g. `Band_1-Day_2019353 >
  710.5` against an actual value of 3333) - a coarse "is there any elevated signal" gate, not a precise
  read of magnitude. This is the split-level explanation for the bright-vs-diluted null result above:
  both 3333 and 10000 clear the same low bar and fall down the same branch.
- **The single most over-predicted pixel of the ten (row 609, col 857, predicted 72%) never touches
  any contaminated predictor in this tree at all** - its path runs through `CThermMean19-Band_31`,
  `CGreenAmp15-NDVI`, `Band_7-Day_2019257`, `Band_1-Day_2019161`, `NDVI-Day_2019065`. Same for the
  pixel at row 4415 (predicted 25%).
- One pixel (row 159, col 1737) produced an empty path (a leaf value of 0.00 with zero splits
  printed) - not reproduced in the larger second pass below (0 of 15,000 traces degenerate there), so
  treated as an unexplained, apparently isolated artifact of that specific pixel/tree pairing, not
  chased further.

**Second pass: generalizing across 500 randomly-sampled over-predicted pixels (fixed seed) and all 30
trees** (15,000 pixel-tree traces total, reusing the same 64-predictor matrix so the added cost was
pure tree traversal, not new metric reads):

- **"Does predictor X appear anywhere in the path" is too weak a test for a 90-deep tree**: the
  contaminated family appears somewhere in 94.4% of traces on average per pixel, Band_31 in 89.9% -
  a 90-node-deep path touches nearly every predictor family somewhere, so mere presence says little.
- **The last split before the leaf - the most locally-discriminating decision - is far more
  informative, and it is diffuse, not concentrated**: the single most common last-split predictor
  (`CGreenMean16-NDVI`) accounts for only 3.53% of the 15,000 traces. No predictor dominates the final
  decision.
- Aggregated to family level by last split: the contaminated family decides 13.8% of a pixel's trees on
  average, Band_31 14.1% - both real, neither dominant.
- Classifying each of the 500 pixels by which family decides more than 20% of its 30 trees: **16.8%
  contaminated-dominant, 10.4% Band_31-dominant, 1.2% both, and 71.6% neither** - for most
  over-predicted pixels, the final decision comes from ordinary daylight-period predictors
  (`Band_2-Day_2020033`, `CUnsMean46-Band_2`, `NDVI-Day_2019257`, `Band_1-Day_2019209`, `Band_2-
  Day_2019161`, `Band_1-Day_2019113`, `CThermMean13-Band_2/Band_6`, `CGreenMean16-NDVI`, and others),
  not the missing-file windows and not Band_31.
- **Severity does not track which family dominates**: mean predicted value is 34.4 for
  contaminated-dominant pixels, 31.3 for Band_31-dominant, 34.9 for neither - essentially the same
  across all three groups.

**Net assessment.** There is no single "smoking gun" metric, nor even a dominant family, behind this
tile's over-prediction. The snow-plus-extreme-zenith mechanism is real (both the spectral evidence and
the causal ablation confirm it), but it is the deciding factor for only about a sixth of the
over-predicted population, and Band_31 metrics (real, influential, recurring near the top of their
range in the first-pass traces) for about a tenth. For most pixels, the model is routing them toward
nonzero tree cover using an assortment of ordinary, uncontaminated, daylight-period predictors -
consistent with the earlier, unconfirmed "training-representation gap" guess (point 3 of the original
investigation): this pixel's overall 64-predictor profile resembles low/moderate tree-cover training
examples from elsewhere in the pooled 63-tile training set closely enough, on many predictors at once,
that the ensemble extrapolates a nonzero value - not a narrow, patchable bug in one part of the
pipeline. This reframes what a "fix" would need to be: less a compositing or QA correction, more a
training-data or model-confidence question (see the fix-candidate list above, especially option 5).

**Where things are.** Read-only scripts in this session's scratchpad (not preserved in the repo): one
traces a single tree for a hand-picked decile sample of 10 pixels; a second generalizes to 500 randomly
sampled pixels (seed 0) across all 30 trees, tabulating predictor usage by presence-anywhere,
last-split, and last-5-splits. Both reuse the real, unmodified `pcttree.bin`/`pcttreeTopN.bin` and the
real metrics cache, read-only, via the same `Metrics.getMetricFromRf()` path `VcfPredict` uses.

## Methodology check (2026-09-28, same day): is the fitted model itself unreliable here, making the ablation experiments' conclusions untrustworthy?

**The question** (the project lead's): every ablation experiment above (the "clean"/"bright"
counterfactuals, the decision-path traces) queries the *same, already-fitted* `pcttree.bin` with
*edited* inputs. If that model was itself trained on the same kind of contaminated metrics, two
different failure modes become possible - the model might have learned something wrong from
contaminated training rows, or the edited inputs used in these experiments might fall into a region of
feature space the tree never saw during training, where a fitted tree's output is not a trustworthy
answer to "what would a correctly-trained model predict." Either would mean these experiments' modest
effect sizes were measuring an artifact of querying an ill-fitted region, not a real answer.

**Direct test.** h16v01's own real training labels
(`v5.0.3samp/h16v01.samp.bin`, 158,912 pixels, all labeled exactly 0 - the same file referenced
throughout this investigation) were checked against the contaminated-composite footprint (`Band_1` and
`Band_2` within +/-200 of 3333 in either window, the same test used in the full-tile Case-2
quantification). Because the missing-file condition affects nearly the *entire* tile, not just the
over-predicted area, **100% of the 158,912 labeled training pixels fall inside the contaminated
footprint** (tile-wide: 99.8% of all pixels do, over-predicted or not). So the model was not just
exposed to this contamination pattern during training - every single real, human-labeled example of
this tile the model ever saw carried it.

**And at those exact 158,912 labeled-and-contaminated pixels, the real 63-tile-run prediction is
overwhelmingly correct**: mean 3.17, median 0.0, only 1.6% predicted above 20%. The model, queried with
the real (equally contaminated) inputs at real training locations, gets it right 98.4% of the time.

**This substantially deflates the "contaminated model" concern, and sharpens the earlier finding rather
than undermining it.** The model is demonstrably not confused or broken by the ~3333 pattern in
general - it correctly predicts near-zero at the overwhelming majority of real examples that carry
it. Since virtually the *whole tile* shares this same contamination but only about 5% of it is
over-predicted, the contamination cannot be sufficient on its own to explain the over-prediction -
something else must differ between the 95%+ the model handles correctly (including all 158,912 labeled
points) and the roughly 5% it does not. That "something else" is consistent with, and now better
supported than, the decision-path finding above: for most over-predicted pixels, ordinary daylight-
period predictors decide the outcome, not the shared contamination. It also means the ablation
experiments were not querying wildly off-distribution territory - the model handles this exact input
pattern routinely and (mostly) correctly, so the ~11% effect size measured earlier is more likely a
real, if partial, signal than an artifact of an ill-fitted tree region.

**What this does not settle.** This checks the model's behavior at the sparse labeled footprint (0.69%
of the tile), not at the specific unlabeled pixels that are actually over-predicted - by definition
those differ from the labeled points in some way, and this check does not identify how. It also does
not rule out subtler miscalibration near the leaf boundaries closest to 3333, only that the model's
*aggregate* behavior at real, contaminated, correctly-labeled examples is sound. The fully decisive
test - retraining (at least partially) on cleaned metrics and comparing - was not done; it is the
natural, more expensive next step if this finding needs to be pushed further (see the fix-candidate
list's option 5). That "some way" the over-predicted pixels differ from the labeled points is answered
directly below.

## What actually distinguishes over-predicted from correctly-predicted pixels (2026-09-28, same day): NDVI, not the contamination

**Question.** Rather than continuing to reason indirectly from tree routing or ablation, compare the
64 predictors' real values directly between the over-predicted pixels (n=1,186,258, "incorrect") and a
large sample of confidently-correct pixels elsewhere in the tile (predicted <=5%, n=500,000 of
10,423,048 available, seed 0, "correct") - a rank-based separation measure (AUC via Mann-Whitney,
robust to the C-mirror metrics' truncated integer scales) across all 64 predictors, using the real,
unmodified metrics cache.

**Result: NDVI-family predictors dominate the ranking, not the contaminated family and not Band_31.**
Of the top 8 predictors by separation, 7 are NDVI metrics, with AUC 0.85-0.90 (over 0.5 means
over-predicted pixels read higher):

| Predictor | AUC | mean, over-predicted | mean, correct |
|---|---|---|---|
| `UnsortedMonthlyBands-NDVI-Day_2019065` | 0.904 | 182.3 | 19.1 |
| `CThermMean19-NDVI` | 0.893 | 95.7 | 21.0 |
| `CGreenMean16-NDVI` | 0.892 | 147.8 | 53.1 |
| `CUnsMean13-NDVI` | 0.887 | 86.4 | 6.7 |
| `CThermMean16-NDVI` | 0.878 | 88.2 | 12.4 |
| `CUnsMean16-NDVI` | 0.875 | 88.0 | 13.4 |
| `CGreenMean13-NDVI` | 0.875 | 210.0 | 88.1 |
| `UnsortedMonthlyBands-NDVI-Day_2019257` | 0.851 | 104.3 | 22.7 |

**Critically, this elevation shows up at clean, non-contaminated periods** - day 065 (early spring),
day 257 (September), day 209 and day 2020033 also rank highly - not just the two implicated windows.
Band_31 metrics form a distinct, secondary cluster (AUC 0.61-0.81, over-predicted pixels consistently
warmer). **The contaminated family, by contrast, barely separates the two groups at all, and mostly in
the *opposite* direction**: `CUnsMean79-Band_1` AUC 0.139 (over-predicted pixels average *lower*, 2248
vs 4278), `UnsortedMonthlyBands-Band_1-Day_2019305` AUC 0.149 (3336 vs 3463, nearly identical). The
over-predicted area is, in absolute reflectance, generally *darker* than the rest of the tile across
most bands and periods (consistent with exposed rock/terrain rather than bright glacier ice) - but its
NDVI (a ratio, not an absolute brightness) is disproportionately high anyway.

**Follow-up: does h16v01's own real training data include any example of this?** Checked the same 5
NDVI predictors at the 158,912 real labeled pixels (all confirmed label 0) against the over-predicted
group's distribution:

| Predictor | over-predicted median | correct-sample median | labeled(training) median | training pixels exceeding the over-predicted median |
|---|---|---|---|---|
| `UnsortedMonthlyBands-NDVI-Day_2019065` | 171.0 | -20.0 | -24.0 | **0.0%** |
| `UnsortedMonthlyBands-NDVI-Day_2019257` | 98.0 | 0.0 | -17.0 | 0.7% |
| `CThermMean19-NDVI` | 72.0 | 11.0 | -1.0 | 1.9% |
| `CGreenMean16-NDVI` | 117.0 | 38.0 | 23.0 | 1.9% |
| `CUnsMean13-NDVI` | 76.0 | -17.0 | -9.0 | 1.7% |

**Essentially none of h16v01's real, confirmed-zero training examples reach the NDVI range the
over-predicted pixels sit in** - for the single strongest predictor, literally zero of 158,912 labeled
pixels do. This is a training-representation gap, precisely localized: the model has never seen a real,
correctly-labeled example, from this tile, of "elevated NDVI and genuinely zero tree cover." It must be
relying entirely on pooled data from the other 62 (mostly lower-latitude, more vegetated) training
tiles to interpret an NDVI value in this range - and elsewhere in that pool, elevated NDVI presumably
does mean real tree cover, so the model extrapolates accordingly here too.

**This resolves the "is the model contaminated" question from the section above, in more specific
terms than the aggregate check could.** The model is not confused by the shared Day_305/353
contamination (it handles that correctly at labeled points, which also happen to have low NDVI); it is
under-informed about what elevated NDVI means in this specific, unlabeled part of the tile, because
nothing in its training data - from this tile - tells it otherwise. This also revises a claim in the
original 2026-09-27 investigation (point 2): "nine of the ten sampled pixels show no distinct greenness
signal" was true of that small, hand-picked sample, but does not hold at population scale - the
over-predicted area's NDVI elevation is real, large, and consistent across thousands of pixels and
multiple clean periods.

**Net effect on the fix-candidate list above.** This points more specifically at option 5
(training-side): not a general "better high-latitude representation" but specifically **labeled
examples of elevated-NDVI, genuinely-non-tree terrain** (exposed rock, lichen, moss, sparse tundra
vegetation - all plausible real covers for the Canadian Arctic Archipelago) are what is missing.
Options 1-4 (compositing/QA/output-masking fixes) address the smaller, already-quantified ~11% slice
from the missing-file mechanism; they would not touch this larger, NDVI-driven majority.

**Not yet done:** identify what this over-predicted area's surface actually is (a land-cover base map
or the figures already in `figures/` could show whether it is exposed rock, a specific geological unit,
or something else); check whether the same "elevated NDVI, unlabeled, genuinely non-tree" pattern
recurs on other high-latitude tiles, which would tell us whether this is an h16v01-specific labeling gap
or a broader one worth fixing at the training-data level across the pipeline.

## Root cause of the `Day_2019065` NDVI elevation (2026-09-28, same day): confirmed unfiltered data, for a new, more systemic reason

**Hypothesis tested** (the project lead's): is the elevated NDVI at `Day_2019065` - the single strongest
predictor found above - caused by unfiltered bad data, rather than a real surface signal? Window
`Day_2019065` covers raw days 065, 081, 097 (confirmed from `ProductType.__init__`'s
`range(yearOneStartDay, yearOneEndDay, dayStep)` = `range(65, 354, 16)`, taking the first three) - all
real 2019 archive files exist for h16v01 on these days (unlike the already-studied `Day_2019305`/`353`
windows), so this is a structurally different situation from the missing-file/`obscov==0` mechanism
already found.

**Checked directly against the real archive** (state QA bits, solar zenith, `obscov`, raw `Band_1`/
`Band_2`), at the over-predicted pixels and a 200,000-pixel sample of correctly-predicted pixels:

- **Zero pixels in the entire tile pass real QA on any of the three days.** Solar zenith runs 57-93
  degrees (mean 75-87) on all three days, for both groups - Case-1 (`createQaMask`'s `solz < 65`) can
  never succeed. The per-pixel QA-passing-day count confirms it: of 1,186,258 over-predicted pixels,
  1,186,256 have **zero** of the three days passing QA (2 pixels have one); of 200,000 sampled
  correctly-predicted pixels, **all 200,000** have zero. **This window is Case-2 (the raw,
  QA-blind fallback average) tile-wide** - the same fallback mechanism already documented for
  `Day_2019305`/`353`, but triggered here by ordinary, expected high solar zenith at 69-80 degrees N in
  early spring, not by a missing archive file. **This confirms the hypothesis directly: none of this
  window's composite values, for any pixel, have ever been quality-screened at all** - `createQaMask`'s
  cloud/shadow/aerosol/adjacency checks are computed but never actually gate anything here, because
  Case-2 uses the raw per-day values "regardless of the QA" by design (`CompositeDayFile.
  _computeComposite`, Case 2).
- **The over-predicted and correctly-predicted groups differ sharply in raw brightness, not in how
  filtered their data is** (neither group's data is filtered at all): correctly-predicted pixels read
  bright (`Band_1`/`Band_2` medians ~4100-9100, ~50-90% reflectance - consistent with bright late-winter
  snow/ice, the dominant cover on this tile), while over-predicted pixels read very dark (`Band_1`/
  `Band_2` medians ~200-410, only 2-4% reflectance) with `Band_2` consistently somewhat higher than
  `Band_1` - the same disproportionate NIR-over-red pattern as the deep-winter windows, now confirmed to
  recur here too, for an entirely unfiltered composite.
- **Cloud-flag rates do not explain the difference, and go the "wrong" way to support a simple
  cloud-contamination story**: over-predicted pixels actually show *lower* cloud-flag rates (26-38%)
  than the correctly-predicted sample (53-88%) on all three days. Since the cloud flag is never used to
  exclude anything here (Case-2 ignores it), this likely just reflects real differences in cloud cover
  between the two groups' locations that week, not a filtering artifact either way.

**Net assessment: the hypothesis is confirmed, and it generalizes the investigation's central finding.**
This is not a cloud-contamination or sensor-glitch event local to one window - it is a **structural
consequence of the fixed 65-degree zenith QA cutoff at extreme high latitude**: at 69-80 degrees N in
early March through early April, the sun genuinely does not rise high enough for *any* observation to
pass QA, so `CompositeDayFile`'s Case-2 fallback (built for the rare case of an occasional bad period)
instead applies to this entire, otherwise-ordinary window, tile-wide, blending unscreened
cloud/aerosol/shadow/snow-mixed raw readings with no quality control whatsoever. The earlier finding
that this window's NDVI has no counterpart in h16v01's real training labels still stands, but this adds
the mechanism: it is not necessarily that this specific terrain has a real, persistent vegetation-like
spectral signature - it may equally be an artifact of the atmosphere/geometry at extreme solar zenith,
exactly as already established for the deep-winter windows, just triggered by ordinary spring
illumination geometry rather than a missing file. **This likely is not unique to `Day_2019065`, to
h16v01, or even to this specific mechanism's two previously-known trigger conditions (missing file,
`obscov==0`) - any composite window, on any sufficiently high-latitude tile, whose raw days all exceed
65 degrees solar zenith will fall into the same tile-wide, completely unfiltered Case-2 fallback.** Not
yet checked: which other windows/tiles in the 63-tile set have this same "zero QA-passing days,
tile-wide" property, and whether they show the same NDVI-elevation-with-no-training-counterpart pattern.

**Where things are.** A read-only script in this session's scratchpad (not preserved in the repo) reads
the three raw `Day_2019065` archive granules directly, decodes the MOD44C-repacked state bits (the same
corrected mapping as the earlier snow-bit work), computes `createQaMask`'s real pass/fail logic per
pixel per day, and compares the over-predicted and correctly-predicted groups' solar zenith, QA
sub-flags, and raw band values.

## The tile-wide scope of the problem, and a fix that works (2026-09-28, same day)

**Scanning all 8 annual composite windows for h16v01** (same per-pixel Case-1/Case-2 test as above,
tile-wide, all 23,040,000 pixels) found the `Day_2019065` situation is not an outlier - it is close to
the median case for this tile:

| Window | Case-1 coverage (>=1 real QA-passing day) |
|---|---|
| `Day_2019065` | 0.53% |
| `Day_2019113` | 14.05% |
| `Day_2019161` | 28.47% |
| `Day_2019209` | 32.03% |
| `Day_2019257` | **0.00%** |
| `Day_2019305` | **0.00%** |
| `Day_2019353` | **0.00%** |
| `Day_2020033` | **0.00%** |

**Five of the eight annual periods are 100% Case-2 (fully unfiltered) tile-wide, and the three
"best" periods only reach 14-32% real coverage even at the height of summer.** This tile essentially
never gets a clean, QA-passing MODIS observation for most of the year - a far larger structural problem
than the two windows examined earlier.

### The fix: NDVI defaulted to 0 wherever a pixel's period had no QA-passing observation

**Design** (the project lead's proposal, implemented per-pixel rather than per-window): read the
per-pixel, per-period Case-1/Case-2 flag `CompositeDayFile._computeComposite()` already computes and
saves (`-qa.npy` files in `2-Composites/`, confirmed here to exactly reproduce the archive-based QA scan
above) - a signal that exists on disk and, as established earlier in this investigation, is never read
by `Metrics.py`. Where that flag says Case-2 (no real observation passed QA), set NDVI to 0 for that
specific pixel and period; leave every Case-1 (real) NDVI value untouched, including in the
partially-covered windows (113, 161, 209), where only the individual pixels that failed QA that period
are touched. This is exactly the rule the Python code already anticipated and left commented out in
`Metrics.getNdvi()` (see above), generalized from the single `ZENITH_CLAMP` case it named to the real
Case-1/Case-2 flag.

**Tested through the real, unmodified pipeline** (same harness as the "clean"/"bright" experiments):
2-Composites bound read-only to the real, unmodified data (no 6.8 GB copy needed, since only the NDVI
cache differs); the modified NDVI cube replaces `3-Metrics/MOD44-NDVI.bin`; every other cached derived
metric was left to recompute fresh, so the fix correctly propagates through the NDVI-based period
reordering the `CGreenMean*`/`CGreenMax`/`CGreenAmp15` families use, not just the NDVI values
themselves. The already-fitted `pcttree.bin` was not changed.

**Result, all four variants side by side, within the 1,186,258-pixel over-predicted mask:**

| Variant | mean | median | max | fraction <=5 | fraction >20 |
|---|---|---|---|---|---|
| Real (unmodified) | 33.94 | 35.0 | 72 | 0.0% | 100.0% |
| Clean (real-only avg, two windows only) | 30.05 | 31.0 | 69 | 1.5% | 84.6% |
| Bright (flat 10000, two windows only) | 33.98 | 35.0 | 71 | 0.3% | 90.3% |
| **NDVI default (per-pixel, all 8 windows)** | **2.52** | **2.0** | **54** | **96.8%** | **0.4%** |

**And, critically, checked tile-wide, not just within the mask that originally defined "over-predicted"**
(all 14,387,266 valid land pixels, true label 0 everywhere in this tile):

| | mean | fraction >20% |
|---|---|---|
| Real | 5.658 | 8.245% |
| NDVI default | 0.848 | 0.047% |

**This is a strict, large improvement, not a partial one.** Mean prediction across the whole tile drops
by 85%, and the fraction of the tile predicted above 20% tree cover drops by more than 99% in relative
terms (8.245% to 0.047%). The over-predicted mask's mean drops from 33.94 to 2.52 - a 93% reduction,
against the 11% and ~0% the two earlier, narrower fixes achieved. 96.8% of the pixels that were wrongly
predicted above 20% now land at 5% or below. A small residual remains (0.4% of the mask, 0.047% of the
tile still above 20%) - not fully explained; plausibly the periods with partial real coverage (113, 161,
209) still carrying some genuine but misleading signal, or a mechanism this experiment did not touch.

**Important caveat, checked the same day, and confirmed to be disqualifying as designed: this is a
strict improvement on h16v01 specifically because its true tree cover is 0 everywhere**, so there was
no real seasonal green-up signal this rule could suppress by mistake there. See "Validation on a real
tree-cover tile" below - the same fix, tested on h12v09, is not safe to adopt as-is.

## Validation on a real tree-cover tile (2026-09-28, same day): the fix as designed fails badly

**Tile chosen:** h12v09, confirmed to be one of the 63 training tiles (its metrics are precomputed in
the run root, it was predicted in the 63-tile run, and it has a real training-label file) with
substantial genuine tree cover: 24,860 labeled pixels, mean label 76.8, 95% labeled above 20%, 88%
above 50% - the real-signal counterpart h16v01 could not provide.

**Same 8-window QA scan, repeated for h12v09**: better than h16v01 (a lower-latitude tile), but still
far from fully filtered - Case-1 coverage ranges 8.7% (`Day_2020033`) to 79.0% (`Day_2019209`), meaning
21-91% of this tile is Case-2 in every one of its 8 periods too. The identical NDVI-default fix was
built and tested through the same real, unmodified pipeline and model.

**Result: the fix substantially degrades prediction quality on this tile.**

| | mean (whole tile) | median |
|---|---|---|
| Real | 63.01 | 70.0 |
| NDVI default | 40.97 | 39.0 |

**Against h12v09's own 24,860 real training labels** (mean label 76.8):

| | MAE | bias | correlation |
|---|---|---|---|
| Real | 6.00 | +0.16 | 0.5607 |
| NDVI default | **28.91** | **-23.37** | **0.2397** |

MAE nearly quintuples; correlation with ground truth collapses from a moderate 0.56 to a weak 0.24. Broken down by true-label bin, the damage is concentrated exactly where it matters most - the high-cover
pixels that make up the bulk of this tile's labels:

| True label | n | MAE, real | MAE, NDVI default |
|---|---|---|---|
| 0-20 | 1,213 | 36.85 | 38.40 |
| 20-50 | 1,643 | 14.89 | 22.72 |
| 50-80 | 3,824 | 8.04 | 27.47 |
| **80-100** | **18,180 (73% of labels)** | **2.71** | **29.14** |

At the pixels with the strongest, most confident real tree-cover signal (80-100%, and the large
majority of this tile's labeled points), the fix turns a near-perfect prediction (MAE 2.71) into one
about as wrong as a random guess within the 0-100 range (MAE 29.14).

**Conclusion: the NDVI-default rule, exactly as designed and exactly as it dramatically helped h16v01,
is not a viable general fix.** It cannot distinguish "this Case-2 period has no real seasonal signal to
lose" (true on h16v01, where every period's true state is near-zero reflectance year-round) from "this
Case-2 period is quietly discarding genuine, strong green-up signal" (true on h12v09, where 21-91% of
every period is Case-2 and real forest is present). A blanket zero-default is safe only on a tile with
no real vegetation signal to begin with - which is not a property the pipeline can know in advance
without already knowing the answer. **This does not overturn the h16v01 diagnosis** (the unfiltered-data
structural problem and the resulting NDVI elevation with no training counterpart are still real,
confirmed findings) - it overturns the specific proposed remedy. A viable fix would need to preserve
real signal where it exists rather than discard it uniformly: candidates worth testing instead include
temporal carry-forward from the nearest QA-passing period (raised earlier in this investigation as an
alternative to a constant substitute, for a different mechanism, but the same idea applies here), or a
confidence-weighted approach the model can learn from rather than a hard override at the metrics stage.
No code was changed in the real pipeline; both this result and the h16v01 result remain scratchpad
experiments.

**Where things are.** Scripts in this session's scratchpad (not preserved in the repo): one scans all 8
windows' tile-wide Case-1 coverage; one builds the modified NDVI cube from the real cache and the
already-saved per-period QA flags; a launcher mirrors the project's existing prediction-testing
convention (`code/validation/run_predict_model_node.sh`), binding the real `2-Composites` read-only and
only substituting the modified NDVI cache.

### Checked: does the earlier 65-vs-70 zenith-cutoff decision explain this?

**Question** (the project lead's): finding 1 of `C_vs_Python_Code_Differences.md` records a resolved
debate over the zenith QA cutoff (C's 65 degrees vs. Python's original, higher default) that was
decided in favor of 65 degrees on 2026-09-14/15. Given today's finding that 65 degrees leaves
`Day_2019065` completely unfiltered at this tile, could that earlier decision have been wrong, and would
70 degrees have caught real, usable observations here?

**Rerun with `ZENITH_CUTOFF = 70`, otherwise identical.** No: QA-pass rate is still 0.00% for both
groups on all three days. Of 1,186,258 over-predicted pixels, 3 gain a QA-passing day (against 2 at 65
degrees); of 200,000 correctly-predicted pixels, 1 gains one (against 0). The minimum solar zenith
across these large samples is 57-69 degrees, but the bulk of the distribution sits at a 75-87 degree
mean - far beyond either threshold. **The 65-vs-70 choice does not explain or affect this window's
unfiltered-data problem**: the actual observations are so far past any reasonable zenith cutoff that a
5-degree shift changes essentially nothing (4 pixels total, out of 1.4 million checked). This closes off
that alternative explanation and reinforces the structural read above - no plausible zenith cutoff
would let this window's early-spring, extreme-high-latitude observations pass real QA.

## Summary and current state (2026-09-28, end of session)

**What is established, in order of how much of the phenomenon each explains:**

1. The dominant mechanism: 5 of h16v01's 8 annual compositing periods are 100% unfiltered tile-wide
   (zero QA-passing observations anywhere on the tile), and the other 3 reach only 14-32% real
   coverage even at peak summer - a structural consequence of the fixed 65-degree zenith QA cutoff at
   69-80 degrees N, confirmed not to be an artifact of the 65-vs-70 cutoff debate specifically. NDVI
   computed from these unfiltered composites is the strongest statistical separator of over-predicted
   from correctly-predicted pixels (AUC up to 0.90), with no counterpart anywhere in this tile's real
   training labels.
2. A candidate fix - defaulting NDVI to 0, per pixel, wherever the underlying period had no QA-passing
   observation, using a flag the pipeline already computes but never reads - resolved the great majority
   of h16v01's over-prediction when tested through the real, unmodified fitted model (tile-wide mean
   5.66 to 0.85; fraction predicted above 20% from 8.2% to 0.05%). **Validated on h12v09 (a real
   training tile, mean label 76.8) and disqualified**: MAE against real labels goes from 6.00 to 28.91,
   correlation from 0.56 to 0.24 - the rule cannot distinguish a tile with no real signal to lose from
   one with genuine signal it would discard. The diagnosis (point 1) stands; this specific remedy does
   not generalize and should not be adopted as-is. A signal-preserving alternative (e.g. temporal
   carry-forward from the nearest QA-passing period, rather than a constant default) is the natural next
   candidate.
3. A secondary, real, causally-confirmed but minor mechanism: snow-flagged, extreme-solar-zenith real
   observations show a vegetation-mimicking red-suppressed spectral ratio, and the missing-file
   composite-dilution artifact this investigation started from explains about 11% of the effect on its
   own - both are now understood as special cases of the broader unfiltered-data problem in point 1,
   not separate bugs.
4. Decision-path tracing found no single "smoking gun" predictor or family - over-prediction is decided
   by a diffuse mix of ordinary predictors for most pixels, consistent with points 1-2 rather than a
   narrow, patchable bug.

**What remains open:** a signal-preserving alternative to the disqualified NDVI-default rule (temporal
carry-forward from the nearest QA-passing period is the leading candidate, not yet tested); identifying
the over-predicted area's actual surface type; checking whether other high-latitude tiles show the same
unfiltered-period pattern; the small residual (0.4% of the mask, 0.047% of the tile) the NDVI-default
fix did not resolve even on h16v01. No code has been changed in the real pipeline anywhere in this
investigation; everything above was tested read-only, in this session's scratchpad, against the real,
unmodified `pcttree.bin` and the real metrics cache.
