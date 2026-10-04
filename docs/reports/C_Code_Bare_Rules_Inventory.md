# The C code's post-model rules for non-vegetated: inventory and evidence (step 4 of `Bare_Where_C6_Is_100_Investigation_Plan.md`)

**Written 2026-10-04. Desk work plus a metadata read; nothing was run on the product.**
- **Code:** `pge61/MOD_PR44B3/MOD44B3.c` (current v7 checkout), lines 354-500.
- **Earlier listing:** `Bare_Static_Model_Pixel_Trace.md`, "There is no separate bare module".
- **Label evidence:** `Bare_C_Regions_Investigation_Results.md` and `Bare_Mechanisms_Results.md`.

C6, C6.1 and the C code are comparisons, not truth. The held-out labels are the independent check.

## Which code made the references (resolves the open "v6.0.3 conflict")

| Product (2019) | `PGEVERSION` in the HDF metadata | High-latitude rule it carries |
|---|---|---|
| C6 (`.006`, produced 2020-04-07/08) | **6.0.4** | Narrowed: h < 18 and v < 3, slot-11 test only. The v7 code comment says its change is "consistent with PGE61 V6.0.4". |
| C6.1 (`.061`, produced 2022-10-10) | **6.1.4** | Every v < 3 tile (the v6.0.3-type rule) |
| C code run (2026-09-19) | v7 checkout | Same as 6.0.4 (h < 18, v < 3) |

**What this explains:**
- **C6 on h20v02, h21v01, h21v02 and h23v02** has bare at 10-17, not 100: those tiles are outside the h < 18 box.
- **C6.1 is 100 there,** and is very far from the labels at high latitude:
  - Asia 50-80N: MAE 89.0 against the labels;
  - North America 50-80N: MAE 81.9;
  - label bin 0-20 overall: MAE 51.9.

The earlier note that the rebuilt v6.0.3 mask "does not match C6" is therefore expected, not a conflict.

## The rules, in the order the C code applies them

| # | Rule (C code) | Where | Python now | What the labels say | Status |
|---|---|---|---|---|---|
| 0 | **Fill:** a pixel whose band-1 input is −28672 in any of its 23 inputs gets no prediction (−3, all layers) | everywhere | No equivalent. Missing inputs enter the means as 10000 and are predicted; out-of-projection comes from the water mask (250). | Not studied | Note only |
| 1 | **Model:** mean of 30 static M5P models, rounded | everywhere | Retrained RF; 60 trees, MEDIAN (decided 2026-10-04) | Retrained RF is far closer to the labels than C6 in every bin except 100 (`Bare_Mechanisms_Results.md`). The static models may expect a half-year-shifted calendar (unresolved). | Replaced |
| 2 | **Sum cap, first:** if tree + bare > 100, bare = 100 − tree | everywhere | Same direction (bare absorbs the overshoot, never tree) | Where C6 = 100 the cap cuts bare on ≤ 0.01% of pixels | Kept |
| 3 | **High-latitude rule:** bare = 100 if the slot-11 (DOY 241) snow flag is 0, or slots 11-13 (DOY 241-273) have no clear observation | h < 18, v < 3 (v6.0.4 / v7); every v < 3 (v6.1.4) | v00 tiles only: tree 0, bare 100 (decided 2026-10-02) | Against forcing 100: h11v02 labels 61, h12v02 81, h12v01/h13v01 23-52, and 74% of below-90 Arctic gap pixels are MCD12Q1 grassland. Neutral on h16v01: labels 98.6, and the model is right there (MAE 0.16; better with median). | Not adopted beyond v00 |
| 4 | **Sahara "volcanic soils and varnish" stretch:** bare ≥ 55 → 2 × bare − 55, capped at 100; hand-drawn exemptions on h22/h23 v05/v06 | h16-h23, v05-v08 | None | Against: on h18v07 the held-out MAE goes from 7.3 to 11.1 and the bias from −0.6 to +8.5. Neutral on h20v06/h21v06, where the labels are about 100 and the model is within 0.1 either way. It brings maps closer to C6, because C6 carries it. | Not adopted |
| 5 | **Tree cap, after rules 3-4:** tree = 100 − bare where the sum exceeds 100 | everywhere | No equivalent. Python never lowers tree, except on v00 tiles (tree 0). | In C, rules 3-4 raise bare and so push tree down on those tiles. Python tree is unaffected there. | Note for C/Python tree comparisons |
| 6 | **Non-tree vegetation** = 100 − tree − bare, kept within 0-100 | everywhere | Same | — | Kept |
| 7 | **SD:** 100 × SD of the 30 model predictions | everywhere | SD of the RF trees (when written) | — | — |

## Reading for comparisons

- **On the tiles where rules 3 and 4 apply, much of the difference between C6 and the Python product is the rules, not the models:**
  - rule 3: h11v02, h12v01, h12v02, h13v01, h16v01, and the Greenland tiles;
  - rule 4: h17v05, h18v07, h19v08, h20v06, h20v08, h21v05, h21v06.

  For C6.1, rule 3 covers every v < 3 tile.
- **Where held-out labels exist inside these areas, they do not support either rule.** The exceptions are labels that are already
  100, where the rule does nothing.
- **No new rule is recommended.** If the project lead wants the product to look closer to C6 in these areas for user continuity,
  that is a policy choice. It would raise disagreement with the labels.
