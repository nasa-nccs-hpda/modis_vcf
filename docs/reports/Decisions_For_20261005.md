# Decisions waiting for the project lead (written 2026-10-04 evening, for 2026-10-05)

Nothing is running. Every run under `FullRuns/` from 2026-10-03/04 is finished, with no failures. Suggested defaults are in **bold**.

## A. Delivery and push (from the office)

1. **Move the tag `v1.1.0`** (local; nothing is pushed yet). It is on `1ac9bc7` (Int16 output). You decided that v1.1.0 = uint8 output.
   The suggestion is to put it on `920d06f`, which has uint8/uint16 output and the MODIS sinusoidal grid. Run this in the clone:
   ```bash
   cd /explore/nobackup/people/mcarrol2/MOD44B_VCF_Development/AI_code_sync_09152026/Git_delivery/modis_vcf
   git tag -f -a v1.1.0 -m "v1.1.0: 60 trees and MEDIAN for both layers; uint8/uint16 output; MODIS sinusoidal grid" 920d06f
   ```
   (The session's permission guard blocked Claude from moving it.) If you want the docs snapshot commit (A3) to be the tagged commit,
   use the branch head instead of `920d06f`.
2. **Push** (`Git_Delivery_Clone_modis_vcf.md`, "To push"; nothing is on GitHub yet), then check with `git ls-remote origin`:
   ```bash
   git push origin mcarrol2/c-python-parity-fixes
   git push -u origin mcarrol2/v1.1.0
   git push origin v1.0.0 v1.1.0
   ```
3. **The docs snapshot** on `mcarrol2/v1.1.0` was refreshed on 2026-10-04 evening (see `Git_Delivery_Clone_modis_vcf.md`). The WV2
   report is left out (NGA rule; decision 19.1), and another user's username is redacted. Check that you are happy with it before you push.
   **Note:** `Product_Versions_and_Work_Map.md` (validation table) and `README_START_HERE.md` (report list) still *mention* the WV2
   work: a one-line description, the scene count, and the name of the footprint index table (`nga_wv02_m1bs_index_2019_20261002`). They
   contain no imagery or values, and the same lines were already in the `0959749` snapshot. Your CLAUDE.md keeps footprint metadata on
   ADAPT unless you say otherwise, so decide whether to strip these lines from `docs/reports` before pushing.
4. **Rewrite the v1.1.0 maps already on disk?** `FullRuns/cand60med_20261004` and `v110_sd_20261004` are Int16 with no geotransform.
   The code now writes uint8/uint16 with the MODIS grid. Options: (a) **leave them as they are (they are evaluation maps) and use the new
   writer for every new map**; (b) rewrite the 69 tiles × 5 layers in the new format (values are unchanged; about 1-2 h detached).
5. **Open the pull request into `develop`** for rlgill after the push (to-do 8).

## B. Composite unusable inputs (`Composite_Unusable_Inputs_Plan.md`, section 5)

6. **D1-D12.** The suggested defaults are D1 (a) A+B+C, not low-sun; D2 (a) usable inputs only; D3 (a) 10000 when nothing is usable;
   D4 (a) fill test in all bands; D5 (a) every tile, evaluated per tile; D6 (a) keep the Greenland zenith rule; D7 (a) do not adopt the
   C code's −3; D8 (a) retrain T60/B60; D9 (a) "no worse" pass rule; D10 (b) gate after the count; D11 (a) this goes before other years;
   D12 ask at adoption.
   **What you need to supply: the D10 threshold** (e.g. < 0.1% of land per tile outside v00 → code fix only). Step 1 is a read-only
   count, about 1 h.

## C. Summer-snow ice rule (`SummerSnow_Rule_Screen_Results.md`)

7. **The screen is not clean as specified.** With June-August everywhere, the rule fires on h12v12 (Patagonian Andes in southern winter)
   and would remove about 2.5 M tree cover points. With the local summer (HEMI) it hardly touches trees (92,579 pixels outside
   Greenland/Iceland, 90% on h11v02 ice caps), but the few bare labels there side with v1.1.0 (h11v02 5.70 vs 7.69 MAE; h08v05 3.7 vs 42).
   The gain is a few bare points on the Greenland ice sheet (h16v02), where v1.1.0 already reads 99.4-99.9.
   Options: (a) **drop the rule**; (b) HEMI rule on the Greenland/Iceland tiles only; (c) HEMI everywhere at 0.5 after a look at the
   h11v02/h08v05 labels.

## D. Work order and planned work

8. **Other years** (to-do 12, `Additional_Years_Plan.md`, about 16-20 h): approve to run after B is settled (decision 19.7(b)).
9. **`Quality`/`Cloud` layers and HDF output** (to-do 11, `Quality_Cloud_Layers_and_HDF_Output_Plan.md`): priority; writer (`pyhdf`
   or a C packer); format (**HDF-EOS2** recommended; confirm with MODAPS / LP DAAC).

## E. Still under investigation (no decision needed now; to plan when you choose)

10. The MC 64 deep dive (to-do 14): why it is +0.28 MAE worse.
11. Cloud blend A (v1.0.1): why it helps open labels and hurts dense ones in the cloudiest pixels (decision 19.2).
12. Revisit the h16v01 training inputs at the next retrain (decision 19.6).

## Done this evening (for reference)

- `920d06f` on `mcarrol2/v1.1.0`: output GeoTIFFs carry the MODIS sinusoidal grid (R = 6371007.181 m). The corner and pixel size match
  C6.1 MOD44B within 1 µm on h08v04, h16v01 and h31v11. 9/9 tests pass.
- The workspace copy `Python_code/modis_vcf` is synced to `920d06f`. Only the notebook `view/editLabeledTraining.ipynb` differs; it has
  local edits that were never on the branch, and it was left alone.
- Predictor sets: the C code's sets are kept for both layers (decision recorded in README item 9).
- The summer-snow screen was run (`FullRuns/summersnow_screen_20261004`, about 9 min).
