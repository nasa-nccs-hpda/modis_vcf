# Decisions waiting for the project lead (written 2026-10-04 evening, for 2026-10-05)

Nothing is running. Every run under `FullRuns/` from 2026-10-03/04 is finished, with no failures. Suggested defaults are in **bold**.

## A. Delivery and push (from the office)

**Status 2026-10-05 evening: A1, A2 and A3 are DONE** (the project lead did the Git work; verified 2026-10-05 with `git ls-remote origin`). On GitHub: `mcarrol2/c-python-parity-fixes` = `0959749`, `mcarrol2/v1.1.0` = `3ec71c1`, tags `v1.0.0` -> `16d6474` and `v1.1.0` -> `920d06f`; `develop` still `b83e302`. **Still open: A4 and A5.**

1. **DONE 2026-10-05: tag `v1.1.0` is on `920d06f`** (uint8/uint16 + MODIS grid), locally and on GitHub. *Original item:* **Move the tag `v1.1.0`** (local; nothing is pushed yet). It is on `1ac9bc7` (Int16 output). You decided that v1.1.0 = uint8 output.
   The suggestion is to put it on `920d06f`, which has uint8/uint16 output and the MODIS sinusoidal grid. Run this in the clone:
   ```bash
   cd /explore/nobackup/people/mcarrol2/MOD44B_VCF_Development/AI_code_sync_09152026/Git_delivery/modis_vcf
   git tag -f -a v1.1.0 -m "v1.1.0: 60 trees and MEDIAN for both layers; uint8/uint16 output; MODIS sinusoidal grid" 920d06f
   ```
   (The session's permission guard blocked Claude from moving it.) If you want the docs snapshot commit (A3) to be the tagged commit,
   use the branch head instead of `920d06f`.
2. **DONE 2026-10-05: pushed** (both branches and both tags on GitHub, matching the clone). *Original item:* **Push** (`Git_Delivery_Clone_modis_vcf.md`, "To push"; nothing is on GitHub yet), then check with `git ls-remote origin`:
   ```bash
   git push origin mcarrol2/c-python-parity-fixes
   git push -u origin mcarrol2/v1.1.0
   git push origin v1.0.0 v1.1.0
   ```
3. **DONE 2026-10-05: pushed with the branch** (no WV2 report on GitHub; the WV2 mentions in the two documents named below went up as they were). *Original item:* **The docs snapshot** on `mcarrol2/v1.1.0` was refreshed on 2026-10-04 evening (see `Git_Delivery_Clone_modis_vcf.md`). The WV2
   report is left out (NGA rule; decision 19.1), and another user's username is redacted. Check that you are happy with it before you push.
   **Note:** `Product_Versions_and_Work_Map.md` (validation table) and `README_START_HERE.md` (report list) still *mention* the WV2
   work: a one-line description, the scene count, and the name of the footprint index table (`nga_wv02_m1bs_index_2019_20261002`). They
   contain no imagery or values, and the same lines were already in the `0959749` snapshot. Your CLAUDE.md keeps footprint metadata on
   ADAPT unless you say otherwise, so decide whether to strip these lines from `docs/reports` before pushing.
4. **CLOSED 2026-10-07 (project lead): superseded by the v1.2.0 build** (`Version_v1.2.0_Build_Plan.md`: 69 tiles in uint8/uint16 with the MODIS grid, code at `920d06f`). Rewriting the v1.1.0 maps is needed only if v1.1.0 itself must be distributed. *Original item:* **Rewrite the v1.1.0 maps already on disk?** `FullRuns/cand60med_20261004` and `v110_sd_20261004` are Int16 with no geotransform.
   The code now writes uint8/uint16 with the MODIS grid. Options: (a) **leave them as they are (they are evaluation maps) and use the new
   writer for every new map**; (b) rewrite the 69 tiles × 5 layers in the new format (values are unchanged; about 1-2 h detached).
5. **Open the pull request into `develop`** for rlgill after the push (to-do 8).

## B. Composite unusable inputs (`Composite_Unusable_Inputs_Plan.md`, section 5)

6. **CLOSED 2026-10-07 (project lead): option 1, do not adopt; no further test needed.** Option 2 (fill-only, D1 (c)) was run (`EXP_fillonly_20261005`) and failed D9 (b). Option 3 (Greenland separately) became E0, and the ice now has a data-driven fix (h16v02 ice rows, v1.2.0). Current behaviour, including 10000 for missing files, stays for the other years, so closing B unblocks D8 (decision 19.7(b)). Not tested: fill-only with a retrain. *Before:* **DECIDED 2026-10-05 evening: not adopted; fill-only short test running (`EXP_fillonly_20261005`).** **RESULT 2026-10-05 evening: fails D9 (b)** (bare +0.069 MAE, from the polar-night tiles; tree unchanged). **Decide: do not adopt (recommended), test the narrower kind-C-only rule, or look into Greenland separately** (`Composite_Unusable_Inputs_Results.md`, Options). *Before:* **D1-D12** — ALL DECIDED 2026-10-05 (see `Composite_Unusable_Inputs_Plan.md` section 9; D9 = stricter rule):

**Background.** When a pixel has no clean observation in a 16-day period, the composite falls back to the **mean of all inputs**. Some
of those inputs are not real data:
- **A. Missing file:** the substitute value 10000 enters the mean (rare in 2019; matters for the other years).
- **B. Polar-night zeros:** band = 0, solz = 0, state = 0 enter the mean and pull every band toward 0 (13 tiles at v <= 02, Nov-Feb).
- **C. In-file fill −28672:** one fill input pulls the mean to about −9500 (bands 1-4, 6, 7: high-latitude ice in Nov-Jan; band 5
  also passes QA on every tile, but no product predictor uses band 5).
- **D. Low-sun readings (solz >= 65):** real but dark/unreliable observations.

The plan proposes one rule for these before the other years are run. Answer each line with (a)/(b)/... or "default".

| # | Question | Options | Suggested (default) | Why |
|---|---|---|---|---|
| D1 | Which inputs count as unusable? | (a) A, B, C; (b) A, B, C + D; (c) C only | **(a)** | D is real data; changing it hurt some tiles before (fill72) |
| D2 | What does the fallback mean average? | (a) usable inputs only; (b) all inputs, as today | **(a)** | That is the point of the fix |
| D3 | Value when no usable input is left | (a) 10000, as for missing files; (b) today's all-input mean; (c) nearest valid period; (d) no-data in the metrics; (e) 0 | **(a)** | One "nothing seen" value, already in the training data, gives NDVI 0; (c) failed in fill72 |
| D4 | Also catch −28672 in the clean-observation (QA) path? | (a) all bands; (b) no | **(a)** | No product effect today; keeps band 5 sane for future predictor searches |
| D5 | Where does it apply? | (a) every tile, checked per tile; (b) a fixed list of 15 tiles at v <= 02 | **(a)** | A data-validity fix, not a regional rule |
| D6 | Greenland zenith-only NDVI rule | (a) keep; (b) replace with the new rule | **(a)** | Already adopted and tested |
| D7 | Copy the C code's "no prediction where band 1 is fill" (output −3)? | (a) no; (b) yes | **(a)** | It leaves holes in the map |
| D8 | Retrain? | (a) rebuild the affected training rows, refit T60/B60, 69 maps; (b) apply only at prediction | **(a)** | Train and predict on the same composites; (b) only if the change is negligible (D10) |
| D9 | Pass rule | (a) "no worse": overall and every tile within seed noise (0.002-0.005 MAE); (b) bare better and tree no worse | **(a)** | A validity fix is expected to change little |
| D10 | Gate after the 1-hour count (step 1) | (a) always run the full test (about 4-5 h); (b) if few values change, make it a code fix with only a maps check | **(b)** | Saves about 4 h. **You set the threshold** (suggested: < 0.1% of land per tile outside v00) |
| D11 | Order with other work | (a) this first, other years after; (b) run the years now, redo later | **(a)** | Avoids redoing the years |
| D12 | Version name if adopted | name at adoption (v1.1.1 would be a guess) | **ask at adoption** | No tag without your OK |

**Step 1 (the read-only count) is RUNNING OVERNIGHT, approved 2026-10-04 evening and tagged EXPERIMENTAL:**
`FullRuns/EXP_unusable_inputs_count_20261004` (see its `EXPERIMENTAL.md`). For 18 tiles (the 15 at v <= 02 and h11v04, h20v06, h12v09),
it gives, per band and period, the land pixels whose composite would change under the plan's recommended rule, by kind (A missing,
B polar night, C fill; D low sun counted but not changed). It also gives, per tile, the land pixels with any changed tree or bare
predictor input (`out/<tile>_summary.csv`). Gate: h12v09's replay of today's rule must equal the stored composites exactly. Use these
numbers to set the D10 threshold. **Then answer: "D1-D12 default, D10 threshold X%".** Full detail: `Composite_Unusable_Inputs_Plan.md`.

## C. Summer-snow ice rule (`SummerSnow_Rule_Screen_Results.md`)

7. **CLOSED 2026-10-06: (a) drop the rule** (project lead). It is an override rule, against the 2026-10-05 direction (data-driven fix
   for the ice sheet, no rules or masks); it was never in v1.1.0 or any product code (checked: no summer-snow logic at tag `v1.1.0`).
   The h16v02 gap is now addressed by `Ice_Training_Rows_h16v02_Plan.md`. *Original item:* **The screen is not clean as specified.** With June-August everywhere, the rule fires on h12v12 (Patagonian Andes in southern winter)
   and would remove about 2.5 M tree cover points. With the local summer (HEMI) it hardly touches trees (92,579 pixels outside
   Greenland/Iceland, 90% on h11v02 ice caps), but the few bare labels there side with v1.1.0 (h11v02 5.70 vs 7.69 MAE; h08v05 3.7 vs 42).
   The gain is a few bare points on the Greenland ice sheet (h16v02), where v1.1.0 already reads 99.4-99.9.
   Options: (a) **drop the rule**; (b) HEMI rule on the Greenland/Iceland tiles only; (c) HEMI everywhere at 0.5 after a look at the
   h11v02/h08v05 labels.

## D. Work order and planned work

8. **Other years** (to-do 12, `Additional_Years_Plan.md`, about 16-20 h): approve to run after B is settled (decision 19.7(b)).
9. **`Quality`/`Cloud` layers and HDF output** (to-do 11, `Quality_Cloud_Layers_and_HDF_Output_Plan.md`): priority; writer (`pyhdf`
   or a C packer); format (**HDF-EOS2** recommended; confirm with MODAPS / LP DAAC).

## F. Next priority (added 2026-10-07 by the project lead)

13. **DRAFT DONE 2026-10-07, awaiting review: `AICoE_10142026/Process_Overview_Figures.md` (fig0-fig7).** **Figures of the whole process (2026-09 to now): NEXT PRIORITY after the v1.2.0 build completes.** Figures showing the path from the
    C code / historical MOD44B and rlgill's Python, through the parity fixes, v0.0.0, v1.0.0, v1.1.0 and v1.2.0, with the experiments
    tried and why each was adopted or dropped.
    **Scope set by the project lead 2026-10-07:** audience = management and developers. Start from the starting point: operational C code
    that could not be built locally (toolkit dependencies) and an incomplete Python build (it could not be checked against the C code because the C code could not be run).
    Figures: how many experiments were run; compute used (rough hours, order of magnitude); the result to date with the problems solved.
    Must show: "mask solutions" (rules and overrides) replaced by data-driven ones; limitations found in the MOD44C input.
    Folder: `Reports/AICoE_10142026/`. Include the C build challenge: every fix needed just to build the toolkits and C code. Plan: `Process_Figures_Plan.md`.

14. **Experiment to do (2026-10-07, not scheduled; after F13): `Tree_Ice_Rows_and_Bare_Priority_Plan.md`.** Tree rows (tree = 0) on the
    h16v02 ice points, and a bare-priority cap (raw bare = 100 -> tree 0), tested first against the held-out labels; goal: retire the v00
    rule with data. Decisions R1-R5 when scheduled.

15. **NOT ADOPTED, to be revisited (project lead, 2026-10-07): cold-ice rule (CR) for the ice-sheet interior.** Bare 90-99 and band 31
    ≤ 275 K in at least 3 of 4 late-summer periods → bare 100; on v00 also tree → 0. It takes the interior to 100% by both the MOD44C
    mask and MCD12 permanent ice, on h16v02 and on v00. It is redundant with v1.2.0 on h16v02. It does not touch low-sun artefacts on
    warm v00 land (non-tree vegetation up to 84; judged not real). It is a rule. Revisit with the tree-side experiment (item 14) and
    the v00 low-sun problem. Write-up: `Cold_Ice_Rule_Check_Results.md`.

## E0. Greenland ice sheet (`Ice_Sheet_Diagnosis_Plan.md`, written 2026-10-05 late evening)

- **DECIDED 2026-10-05 late evening: E1 approved; E3 yes, with the MCD12 counts as a reference, not a definitive ice-sheet boundary.** *Before:* **E1:** approve the read-only diagnosis (about 1-2 h, after the fill-only test). (E2 withdrawn: the summer-snow rule was only tested, never adopted; nothing to decide.)
  **E3:** the fill-only test is judged on the same ice-sheet counts.

- **Diagnosis DONE** (`Ice_Sheet_Diagnosis_Results.md`). **`Band31_Unclamped_Plan.md` APPROVED 2026-10-05 late evening with U1-U6 as recommended;
  adoption later only with care (the clamp was a design decision; caution note in the plan).** Decisions U1-U6 (suggested: U1 (a) everywhere,
  U2 (a) keep QA as is, U3 (a) retrain, U4 (a) separate from fill-only, U5 D9 (b), U6 ask at adoption).

- **ANSWERED 2026-10-07 (project lead): band 31 unclamped NOT adopted** (clamp stays). It **failed** D9 (b) on tree cover (`Band31_Unclamped_Results.md`: do not adopt,
  the clamp stays, no version name). The ice is fixed instead by the h16v02 ice training rows (`Ice_Training_Rows_h16v02_Results.md`: A10
  kept, h12v02 accepted as seed noise; going into **v1.2.0**). Still open, no decision needed now: Iceland (h17v02) is seed-sensitive, and
  the MCD12-only cells on h16v02 are of uncertain cover (the NGA edge list is on ADAPT for a look).

## E. Still under investigation (no decision needed now; to plan when you choose)

10. The MC 64 deep dive (to-do 14): why it is +0.28 MAE worse. **Plan written: `MC64_Deep_Dive_Plan.md`** (read-only steps
    about 1-2 h; optional EXPERIMENTAL fits; approve or change).
    **Status 2026-10-07:** still awaiting approval; nothing run. Tree-cover only, unaffected by v1.2.0.
11. Cloud blend A (v1.0.1): why it helps open labels and hurts dense ones in the cloudiest pixels (decision 19.2).
    **Status 2026-10-07:** no plan written, nothing run since 2026-10-03. Tree-cover only, so v1.2.0 (bare change) does not affect it.
    **Revisit later (project lead, 2026-10-07): review the decision not to adopt the cloud variants (V1-V4C, V4R, BLEND-2, v1.0.1).**
    **Goal: improve dense-forest predictions in cloudy areas without really hurting predictions elsewhere.**
    What the review found (2026-10-07, from the reports; nothing run):
    - Higher forest values were not what counted against them. Where labels show dense forest under heavy cloud, v1.0.0 is 7-9 points
      low, so a raise there lowers the error.
    - The reason differed by stage:
      - **Whole-tile V4r** (V1-V4C, V4R): worse on clear pixels too (h12v09 MAE 6.84 -> 8.70; V4R +0.13 over all labels).
      - **Ungated blends:** raised open labels (0-20), e.g. B1Z +1.8; the pooled dense result was skewed by snowy h10v04. Without
        h10v04, dense labels went up +1.2 / +1.8 on 11-12 of 14 tiles.
      - **Gated v1.0.1:** open land (-0.04) and snow (h10v04 -0.92 -> -0.20) are under control, but dense labels rose only +0.3
        (n.s.), about 5% of the gap, against +4 on the maps.
    - The labels may be a weak test of this fix. Dense label sites are stable forest where v1.0.0 is already close (76.7 vs 77.4).
      The large raises land on pixels v1.0.0 under-predicts, where the few labels do agree (+11 to +12 where v1.0.0 < 30,
      +5.6 to +7.2 where 30-59; n = 44 / 288 on the 7 map tiles).
    - **Suggested first step (read-only, short):** score v1.0.1 and the gate variants on dense-label pixels that v1.0.0
      under-predicts, on all tiles, with the seed refit as the control. Also report open labels (<= 40) and clear pixels, to show
      the harm elsewhere. Then plan any gate or weight changes.
    Sources: `V4r_Cloud_Blend_Results.md` (correction section, R3), `Gated_V4r_Blend_63Tile_Results.md`, `V4r_Darkest_Red_h12v09_Results.md`;
    figures `AICoE_10142026/fig8a-c`.
12. Revisit the h16v01 training inputs at the next retrain (decision 19.6).
    **CLOSED 2026-10-07 (project lead):** answered by `Ice_Sheet_Diagnosis_Results.md` (H3): 25-32% of h16v01's stored predictor cells differ from
    the v1.1.0 metric folder, but the bare model gives the same answer from either (mean 99.998, 0.015% below 100 from both; tree 0).
    Only a consistency item for a future full rebuild of the training tables; v1.2.0 does not touch h16v01's rows.

## Done this evening (for reference)

- `920d06f` on `mcarrol2/v1.1.0`: output GeoTIFFs carry the MODIS sinusoidal grid (R = 6371007.181 m). The corner and pixel size match
  C6.1 MOD44B within 1 µm on h08v04, h16v01 and h31v11. 9/9 tests pass.
- The workspace copy `Python_code/modis_vcf` is synced to `920d06f`. Only the notebook `view/editLabeledTraining.ipynb` differs; it has
  local edits that were never on the branch, and it was left alone.
- Predictor sets: the C code's sets are kept for both layers (decision recorded in README item 9).
- The summer-snow screen was run (`FullRuns/summersnow_screen_20261004`, about 9 min).
