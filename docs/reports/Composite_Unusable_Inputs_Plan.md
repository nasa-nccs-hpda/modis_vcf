# What the composite does with unusable inputs: plan (fill values, polar-night zeros, missing files)

**Status: written 2026-10-04 evening; awaiting the project lead's decisions (section "Decision points"). Nothing has been run
beyond the read-only fill-value screen.** Requested by the project lead after decision 7 of README item 19 (fill values first) and the
discussion that followed: settle what the composite does when a pixel has no usable observation **once**, for all kinds of bad
input, before running other years (to-do 12). The history below is kept in full so that the choices can be revisited.

## 1. The question

`CompositeDayFile._computeComposite` builds each period's composite per pixel and band from that period's inputs:

- **Case 1 (QA pass):** the mean of the inputs that pass `createQaMask` (cloud, shadow and adjacency clear, bit 15 good, 0 < solz < 65).
- **Case 2 (fallback):** if no input passes, the mean of **all** inputs, with no test of any kind.

Several kinds of input that are not observations reach these means as if they were numbers. Each has been handled, or not handled, on
its own so far. **The main question:** what should a period's composite be when a pixel has no usable observation, and which inputs
count as unusable? Every year that is run uses this rule, so it should be settled before to-do 12.

## 2. The kinds of unusable input, and what happens to each today

| Kind | What the input holds | Case 1 (QA path) today | Case 2 (fallback) today | Where it occurs | Evidence |
|---|---|---|---|---|---|
| **A. Missing file** | no file for that date | rejected (qa = 0 by construction) | **10000 enters the mean**, all bands | wherever a file is absent (rare in 2019; the 2001/2005/2015/2023 year ends need files from the private repository) | `CompositeDayFile.SUBSTITUTE_PIXEL_VALUE`; copies PGE61 |
| **B. Polar-night "no observation"** | band = 0, solz = 0, state = 0, obscov = 0 | rejected (solz > 0 test) | **zeros enter the mean**, pulling every band toward 0 | 13 tiles at v ≤ 02, Nov-Feb periods only | `Polar_Night_Zero_Observation_Finding_h12v02.md`; `Fallback_Solz_LE0_Plan.md` screen |
| **C. In-file fill −28672** | −28672 in one band | **band 5: passes QA** (83% of its fill on h11v04 and h16v02); other bands: rejected | **−28672 enters the mean** (one of three inputs pulls it to about −9500) | band 5: every tile tested; bands 1-4, 6, 7: high-latitude tiles only (Nov-Jan, ice) | `Fill_Value_Screen_Results.md` |
| **D. Low-sun observation** | a real but dark or unreliable reading, solz ≥ 65 (twilight to 93°) | rejected (solz < 65 test) | **enters the mean** | high latitudes; ice sheet in Apr-Jun (midnight sun, solz 86-90°) and twilight in Nov-Dec | `Fallback_Solz_LE0_Plan.md` (revision note and correction) |
| **E. Band 31 above solz 70** | clamped to 16000 before the mean | rejected (solz) | 16000 enters the mean | high latitudes | `BandDayFile.VALUE_CLAMP_ZENITH_CUTOFF`; matches the C code |

Downstream:
- **NDVI** is computed from the composite bands; when bands 1 and 2 are both 10000, NDVI is 0 (`Metrics.py`, comment near line 396).
- On the 6 Greenland tiles, the adopted **zenith-only NDVI rule** sets NDVI to 0 in a period with no input at 0 < solz < 65
  (`Metrics.ZENITH_ONLY_NDVI_TILES`). It changes NDVI only, never the bands.
- **The v00 rule** sets every v00 land pixel to tree 0 / bare 100 / non-tree vegetation 0 after prediction, so the v00 tiles' maps
  do not depend on their composites (their composites still matter if they become training tiles).
- **The C code** does not predict a pixel whose band 1 is −28672 in any of its 23 inputs (value −3 in all layers;
  `C_Code_Bare_Rules_Inventory.md`, row 0). Python has no equivalent.

## 3. History (oldest first; kept in full)

1. **PGE61 / C code behaviour, inherited by Python.** Missing inputs are replaced with 10000 and both QA arrays get 0; case 2 takes
   the mean of everything. The C code's band 1-7 zenith clamp (> 70 → 16000) is dead code (it reads solz before it is filled);
   only band 31's clamp runs. Python copies this (`Zenith_Value_Clamp_Root_Cause_Full_Picture.md`).
2. **2026-09-27: the bands 1-7 clamp removed from Python** to match the C code's real behaviour; band 31 keeps it.
3. **2026-09-29/30: bit 15 replaces the bits 7-8 aerosol test in the QA mask** (decision: B and C tie, a tie favours C). The README
   noted then: "look at h16v01 (snow): C's baseline model is +3.8 MAE there".
4. **Polar-night finding (h12v02):** solz ≤ 0 inputs are true "no observation" days (all zeros), not the documented fill values
   (`Polar_Night_Zero_Observation_Finding_h12v02.md`).
5. **Zenith-only NDVI fix, 63 tiles** (`Zenith_Only_Fix_63Tile_Plan.md` / `_Results.md`): NDVI = 0 where a period has no input at
   0 < solz < 65. Applied everywhere, it **worsened agreement with the labels on 10 high-latitude tiles**: h11v02, h11v03, h18v03,
   h18v04, h20v02, h20v03, h21v02, h23v02, h23v03, h24v03. The cutoff-70 retest did not remove that
   (`Zenith_Only_Fix_Cutoff70_Retest_Results.md`). **Lesson: scope by explicit tiles and evaluate per tile;** the damage most
   likely came from the upper bound (real low-sun data).
6. **2026-10-02: zenith-only NDVI adopted for the 6 Greenland tiles only**, Iceland excluded (`Greenland_Iceland_Scoped_Fix_v2_Results.md`;
   commit `13773db`). h15v03 has no MOD44C files and is out.
7. **2026-10-02: v00 rule adopted** (knowledge-based, not evaluated; commit `16d6474`).
8. **2026-10-04: h16v02 straight edge traced** to the polar-night input boundary in DOY 353 (`Bare_Mechanisms_Results.md`, point 7).
   Above the edge the December input is all zeros (kind B); below it some pixels have a twilight observation at solz 93° with
   band 7 = −28672 (kinds C and D together).
9. **2026-10-04: solz ≤ 0 screen** (`FullRuns/solz_screen_20261004`): kind B in the fallback is confined to 13 tiles at v ≤ 02 and the
   Nov-Feb periods; the QA path is never affected. Plan `Fallback_Solz_LE0_Plan.md`: leave solz ≤ 0 out of the fallback mean; where
   every present input is ≤ 0, keep today's all-input mean (recommended) or treat it as missing (10000); **project lead to choose
   (never chosen: superseded by item 10).** The same plan recorded the fill values as a side finding.
10. **2026-10-04: the plan revised into the fill72 experiment** (approved as an experiment): usable = 0 < solz < 72 and file present;
    a period with no usable input takes the pixel's nearest valid period. **Result: fails the pass rule, not adopted**
    (`Fallback_Fill72_Results.md`): bare not clearly helped; tree cover rises on the ice sheet (tree = 0 share 99.7 → 94.1% on h16v01).
    Lessons: MEDIAN alone removes the h16v02 edge and most ice bare < 100; filling with "summer-like" inputs makes ice look slightly
    vegetated. **Pass rule (project lead): bare helped and tree cover unchanged or better.**
11. **2026-10-04: MEDIAN for both layers, 60 trees = v1.1.0** (README items 13, 17, 18).
12. **2026-10-04 evening: fill-value screen** (`Fill_Value_Screen_Results.md`, 5 tiles): see kind C above. No product predictor set
    uses band 5 (tree 64, non-vegetated 21, MC64).
13. **2026-10-04 evening: decision 7** (README item 19): fill values first, then other years, then Quality/Cloud + HDF. The project
    lead then asked for this plan covering all kinds of unusable input together.

What has **not** been tried: leaving kinds A-C out of the means with a constant substitute where nothing usable remains; a fill-value
test in the QA path; any change to kind A.

## 4. Proposed approach (subject to the decision points)

**Recommended definition:** an input is unusable for a band if its file is missing (A), its solz ≤ 0 (B), or its value in that band
is −28672 (C). Low-sun observations (D) stay usable in the fallback, as today, because the upper bound has its own known issues
(lesson 5) and the project lead asked for it to be left alone.

**Recommended rule:**
- **Case 1:** also reject −28672 in every band (this only matters for band 5, which the product does not use; it cleans the metric pool).
- **Case 2:** take the mean of the usable inputs only.
- **No usable input at all:** use the constant 10000 in that band, as a missing file is treated today. The models already see 10000
  for missing inputs in training, NDVI from it is 0 (the same value the Greenland rule gives), and no new kind of value enters the metrics.
- **Everything else unchanged:** the QA test, band 31's clamp (E), the Greenland zenith-only NDVI rule, the v00 rule.

**Why one rule for A-C:** none of them is a measurement. Today the composite averages 10000s, zeros and −28672 depending on which
kind happens to be present, so the same physical situation ("nothing seen") gets three different values.

## 5. Decision points (for the project lead)

| # | Decision | Options | Recommendation |
|---|---|---|---|
| **D1** | Which inputs count as unusable | (a) A, B, C; (b) A, B, C + D (solz ≥ 65 or ≥ 72 in the fallback); (c) C only (narrowest fix) | **(a).** D is real data with known upper-bound problems (lesson 5); test it separately if at all. |
| **D2** | Fallback mean | (a) usable inputs only; (b) today's all-input mean | **(a).** |
| **D3** | Value when nothing usable remains | (a) constant 10000 (as missing files); (b) today's all-input mean (kept for B in the earlier solz plan); (c) nearest valid period (fill72; **tested, failed**); (d) no-data in the metrics (needs imputation or no prediction; new behaviour for the models); (e) 0 | **(a).** One value for "nothing seen"; already in training data; NDVI 0 matches the Greenland rule. (b) keeps −28672 averages where all inputs are fill. |
| **D4** | Fill test in the QA path (case 1) | (a) all bands; (b) not at all (product unaffected) | **(a).** No product effect today; keeps band 5 metrics sane for future predictor searches (MC) and any later use. |
| **D5** | Where it applies | (a) every tile, evaluated per tile, scoped by an explicit list only if some tiles get worse; (b) an explicit list from the start (the 15 v ≤ 02 tiles) | **(a).** Kinds A-C are data validity, not a physical judgement; lesson 5 is respected by per-tile evaluation and leaving D alone. |
| **D6** | Greenland zenith-only NDVI rule | (a) keep as is; (b) replace with the new rule on its 6 tiles | **(a).** Adopted and evaluated; fill72 showed replacing it was not helpful. |
| **D7** | The C code's −3 rule (no prediction where band 1 is fill) | (a) do not adopt; (b) adopt | **(a).** It leaves holes; the goal is Python quality, not C parity. Recorded for completeness. |
| **D8** | Retraining | (a) rebuild training rows on affected tiles, refit T60 and B60 (seed 0), maps for 69 tiles; (b) apply at prediction only, keep v1.1.0 models | **(a).** Train and predict on the same composites; (b) is a fallback if the change is negligible (see D10). |
| **D9** | Pass rule | (a) "no worse": each layer within the seed yardstick (0.002-0.005 MAE) overall and per tile, with no tile worse beyond it; (b) fill72's rule: bare helped and tree unchanged or better | **(a)** for a data-validity fix, which is expected to change little; the lead may prefer (b). |
| **D10** | Gate after the count (step 1) | (a) proceed to the full test whatever the count; (b) if the predictor values affected are below a threshold (e.g. < 0.1% of land per tile outside v00), adopt as a code fix with only the one-tile exactness check and a maps-only comparison | **(b), with the threshold set by the lead.** Saves about 4 h if the effect is negligible. |
| **D11** | Order with other work | (a) other years (to-do 12) wait until this is decided; the summer-snow screen (decision 5) runs alongside (read-only, independent); (b) run the years now and redo later | **(a).** |
| **D12** | Version and delivery if adopted | a version name (ask; v1.1.1 would be a guess), a commit on the Git_delivery branch (the project lead pushes) | **Ask at adoption.** Do not tag without the lead's OK. |

## 6. Run plan (after the decisions; fully detached, adapt-detached-run skill)

**Step 1, composite-level count (read-only; about 1 h on one node).**
- Tiles: the 15 at v ≤ 02 (h11v02, h12v01, h12v02, h13v01, h15v02, h16v00, h16v01, h16v02, h17v00, h17v01, h17v02, h20v02, h21v01,
  h21v02, h23v02; 13 of them have solz ≤ 0 fallback inputs in the solz screen, h11v02 and h13v01 almost none), plus h11v04, h20v06 and h12v09 as controls.
- Using the pipeline's own period definitions (`_getDaysToFind`) and QA mask, classify every land pixel, band and period
  (land from the v1.1.0 maps: not 200/250):
  case 1 clean; case 1 with fill; case 2 all present inputs observations; case 2 with any of A, B, C, D; case 2 with nothing usable.
- Report per tile for the **predictor bands only** (tree: 1, 2, 6, 7, 31, NDVI; non-vegetated: 2, 3, 4, 6, 7, 31, NDVI), and how
  many predictor values the rule would change, split by kind.
- **Decision gate D10.**

**Step 2, self-test (minutes).**
- h12v09 (no affected inputs expected): rebuilt composites equal the stored ones byte for byte.
- h16v02: only pixels and periods flagged in step 1 change.

**Step 3, composites and metrics** for the affected tiles (needed-bands builder, about 7.5 min per tile).

**Step 4, training tables** for the affected labelled tiles; replace their rows in copies of the tree and bare splits.
Acceptance: keys and labels identical; unaffected tiles identical cell for cell; on affected tiles only columns from changed periods differ.

**Step 5, refit** T60 and B60, seed 0 (about 20 min).

**Step 6, maps for 69 tiles** with v1.1.0 settings (60 trees, MEDIAN both layers; about 2 h on ngaproc201/202 + forest201-205).

**Step 7, evaluation** against v1.1.0 (`cand60med_20261004` maps / `v110_sd_20261004`), with the vcf-eval-toolkit:
- held-out labels by bin and region, per tile, with the 15 v ≤ 02 tiles and the 4 lesson-5 tiles (h11v02, h20v02, h21v02, h23v02) on their own lines;
- maps against C6, C6.1 and the C code (comparisons, not truth);
- the h16v02 edge and the Greenland ice (exactly 100 share, tree = 0 share);
- seed yardstick 0.002-0.005.

Then `Composite_Unusable_Inputs_Results.md`.

**Total:** about 1 h for step 1; about 4-5 h more for steps 2-7 if D10 sends it on. The nodes are free (v110_sd finished about
17:55 on 2026-10-04). Disk: about 60-80 GB.

## 7. Expected outcome

- Kind C in the predictor bands and kind B are confined to the high-latitude tiles, mostly ice and polar-night land, where MEDIAN and
  the v00 rule already decide much of the outcome. **Small map changes outside v ≤ 02 are expected only through the refit.**
- Kind A has no effect on 2019 if no files are missing, but the rule matters for the years with missing year-end files.
- If D3 (a) is chosen, polar-night pixels that today average to zeros will read 10000 in the affected bands; this is the change most
  likely to move the Greenland and Arctic maps, and the per-tile evaluation is there to catch it.

## 8. Files

- Screen: `Fill_Value_Screen_Results.md`; script `Reports/fill_count.py`.
- Earlier plans and results: listed in section 3.
- Code: `Python_code/modis_vcf/model/CompositeDayFile.py` (`_computeComposite`, `SUBSTITUTE_PIXEL_VALUE`),
  `ProductTypeMod44.py` (`createQaMask`), `BandDayFile.py` (band 31 clamp), `Metrics.py` (NDVI, zenith-only rule).
  Changes go in a copy in the run folder; the delivery branch is changed only after adoption.
