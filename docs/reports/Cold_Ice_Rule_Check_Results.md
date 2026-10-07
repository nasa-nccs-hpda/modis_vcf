# Cold-ice rule for the ice-sheet interior: check results

**Status 2026-10-07: NOT ADOPTED — to be revisited (project lead).** This was a read-only check, with no maps or models written. Nothing
in v1.2.0 changes. The rule is a post-model rule, which goes against the project lead's aim of eliminating rules. It was checked
because it targets one physical case narrowly. Run folder: `FullRuns/EXP_cold_ice_rule_check_20261007/` (scripts in `code/`, tables in
`out/`, marked EXPERIMENTAL).

## 1. How this came up

1. **F13 figure.** The h16v02 figure (`AICoE_10142026/fig7_h16v02_bare_v110_to_v120.png`) was redrawn by class so that bare < 100
   is easy to see. In v1.1.0, a large 90-99 area covers the ice sheet; v1.2.0 removes almost all of it.
2. **The sum cap note.** The project lead's earlier note on a different sum cap rule is the bare-priority cap (BP) in
   `Tree_Ice_Rows_and_Bare_Priority_Plan.md`. **On h16v02 the sum cap is not the cause.**
   - 100% of v1.1.0's bare 90-99 pixels have tree = 0, so the bare model itself predicts 90-99.
   - In v1.2.0 the figure is 99.5%.
   - So BP would not change h16v02. The cap matters on the v00 tiles (with the v00 rule off) and on about 48,800 pixels elsewhere
     with tree 1-5 and bare 95-99.
   - Options discussed: the current rule (tree wins), BP, proportional scaling, an uncertainty-based cap using the SD layers, and tree
     training rows (TR).
3. **The project lead's question.** The goal is the **middle of the ice sheet at 100% bare**. Would a rule "bare 90-99 and cold in
   summer → bare 100" capture those pixels? Combined with pushing tree cover down, would it also handle the v00 tiles?
4. **A correction during the discussion.** The first answer pooled whole tiles and so mixed ice-free coast into a question about
   the ice interior. The project lead asked for results by the ice masks used before: the ad hoc MOD44C ice mask and the MCD12
   permanent ice. Section 3 gives those.

## 2. The rule tested

- **Cold:** band 31 ≤ 275 K in at least 3 of the 4 late-summer MOD44C periods (2019177, 193, 209, 225).
  - This is the thermal part of the MOD44C ice-extent test (`Ice_Extent_MOD44C_Reference_Results.md`).
  - Melting ice cannot be warmer than 273.15 K; the extra 2 K allows for mixed pixels.
  - It reads the raw MOD44C band 31, not the product's clamped predictor.
- **CR (bare only):** cold and bare 90-99 → bare = 100.
- **CR + tree:** cold and bare ≥ 90 → bare = 100 and tree = 0.
  - On the v00 tiles the tree part is necessary. Tree is 1-2 on most ice pixels there, so the existing sum cap would pull bare back
    to 98-99.
- **Zones:** from the June-blue reference masks (`EXP_ice_extent_june_20261006`, all ice bodies), the masks used for the ice rows and
  for `EXP_v00_norule_20261007`.
  - The zone sizes match that run's own tables.
  - A first pass used the earlier `EXP_ice_extent_mod44c_20261006` masks. Its zones differ (for example, h16v00 ambiguous is 2,721
    pixels there against 246,724), so it is superseded and kept in `out/` for the record only.
  - Zone definitions:

| Zone | Definition |
|---|---|
| MOD44C interior | MOD44C ice more than 2 km inside its edge |
| MOD44C edge | MOD44C ice within 2 km of its edge |
| MCD12 > 10 km inside | MCD12 class 15 (permanent ice) more than 10 km inside its edge |
| MCD12 ambiguous | MCD12 ice that MOD44C cannot confirm (too few usable or bright periods) |
| MCD12 only | MCD12 ice that MOD44C says is not ice |

## 3. Results by zone

### h16v02 (bare < 100, share of the zone)

| Zone | Pixels | Cold | v1.1.0 | v1.1.0 + CR | v1.2.0 | v1.2.0 + CR |
|---|---|---|---|---|---|---|
| MOD44C interior | 1.13 M | 100% | 8.7% | **0.0%** | 0.01% | **0.0%** |
| MOD44C edge | 4.38 M | 100% | 26.8% | 0.01% | 0.11% | 0.01% |
| MCD12 > 10 km inside | 6.78 M | 100% | 25.0% | **0.0%** | 0.04% | **0.0%** |
| MCD12 ambiguous | 2.47 M | 97% | 27.3% | 1.8% | 0.84% | 0.67% |
| MCD12 only | 0.03 M | 0% | 60.7% | 60.7% | 37.8% | 37.8% |

- **CR takes the interior to 100% by both masks.** Every interior pixel is cold, and all interior bare 90-99 pixels are caught.
- **On v1.1.0, CR does what the ice training rows did** (interior 8.7% → 0.0%, against v1.2.0's 0.007-0.01%). It would be a rule-based
  route to the result v1.2.0 already reaches with data.
- **On v1.2.0, CR adds almost nothing on h16v02** (0.01% → 0.0%; 0.04% → 0.0%).
- Tree is already 0 here, so the bare-only form is enough.

### v00 tiles (v1.2.0 with the v00 rule off, `EXP_v00_norule_20261007`)

| Zone | h16v00: cold | h16v00: bare < 100 / tree > 0 | h16v00: after CR + tree | h17v00: cold | h17v00: bare < 100 / tree > 0 | h17v00: after CR + tree |
|---|---|---|---|---|---|---|
| MOD44C interior | 100% | 81.6% / 81.6% | **bare 100 and tree 0 on 100%** | 100% | 71.4% / 71.4% | **100% / 100%** |
| MOD44C edge | 100% | 78.2% / 78.2% | 100% / 100% | 100% | 74.4% / 74.4% | 99.98% / 99.98% |
| MCD12 > 10 km inside | 100% | 84.0% / 84.0% | **100% / 100%** | 100% | 72.8% / 72.8% | **100% / 100%** |
| MCD12 ambiguous | 8.6% | 96.0% / 94.6% | about 10% at 100 | 7.1% | 96.6% / 94.4% | about 9% at 100 |
| MCD12 only | 0% | 99.7% / 96.6% | unchanged | 0% | 99.8% / 96.0% | unchanged |

- **On the v00 ice, CR + tree gives bare 100 and tree 0 on 100% of the interior,** by both masks.
- **The MCD12 ambiguous and MCD12-only zones are warm in summer, so CR does not reach them.** Their July band-31 mean is about
  282-287 K, so this is not melting ice. They make up about 36% (h16v00) and 27% (h17v00) of the MCD12 ice area. Whether they should
  be 100 depends on MCD12's ice extent, not on the interior.

### Non-tree vegetation on h16v00 (v00 rule off; CR + tree does not change it)

On the ice, tree + bare = 100 already (bare 98-99, tree 1-2), so non-tree vegetation is 0 there before and after. CR moves 1-2 points
from tree to bare. Off the ice, nothing changes.

| Zone | Pixels | Median | 95th pct | Max | > 20 |
|---|---|---|---|---|---|
| MOD44C interior and edge; MCD12 > 10 km | 0.69 M | 0 | 0 | 0-15 | 0% |
| MCD12 ambiguous (warm) | 247 k | 0 | 14 | 57 | 0.9% |
| MCD12 only (warm) | 108 k | 8 | 17 | 60 | 3.2% |
| Not ice in either mask | 870 k | 8 | 26 | 84 | 11.6% |
| **All land** | 1.89 M | 0 | 22 | 84 | 5.7% |

- **High values are rare:**

  | Non-tree vegetation | Pixels | Share of land |
  |---|---|---|
  | > 30 | 20,809 | 1.1% |
  | > 50 | 2,971 | 0.16% |
  | > 70 | 75 | 0.004% |

- **Where the pixels above 50 are:**
  - none in either ice mask;
  - all in the southern fifth of the tile, near 80°N, on ice-free north Greenland land;
  - median 3.5 km from water;
  - scattered: 1,067 clusters, half the pixels in clusters of 1-4;
  - bare median 43 and tree median 1 there.
- C6 is 0 at all of them, but only because the C code's high-latitude rule forces v00 land to bare 100.
- **Project lead's judgment (2026-10-07): these high non-tree vegetation values are almost certainly low-sun artefacts, not real
  vegetation.** This is the same low-sun and polar-night input problem as before (2 of 8 periods have no data on v00 tiles).

## 4. Conclusions

1. **For the stated goal, CR works.** It takes the ice-sheet interior to 100% bare by both the MOD44C mask and MCD12 permanent ice, on
   h16v02 and on the v00 tiles. On v00 that needs CR + tree.
2. **It is narrower than the v00 rule.** It fires only where the surface is below melting in summer and bare is already predicted at
   90 or more. The v00 rule acts on whole tiles.
3. **It does not solve the v00 tiles on its own.** The v00 rule also hides low-sun artefacts on warm, ice-free land (non-tree
   vegetation up to 84, tree 1-2). CR does not touch those. Retiring the v00 rule needs a fix for that land too.
4. **On h16v02 it is redundant with v1.2.0**, whose ice training rows reach the same interior result with data.

**Decision (project lead, 2026-10-07): not adopted; to be revisited.** It goes against the aim of eliminating rules. Revisit it
together with the tree-side experiment (`Tree_Ice_Rows_and_Bare_Priority_Plan.md`) and the low-sun / polar-night problem on v00 land.

## 5. Before any adoption (when revisited)

1. **Where else CR fires.** Count cold, bare 90-99 pixels on all 69 tiles. Mountain glaciers and high plateaus could pass the cold
   test. The ice masks exist only for Greenland and Iceland, so the thermal count needs computing for the others (read-only).
2. **Data-driven alternative first.** Tree rows (tree = 0) on the ice (TR in the plan). The low-sun land on v00 may also need rows or
   an input fix. Compare these against CR + tree on the zones above.
3. **Low-sun artefacts on warm v00 land.** Decide the target for ice-free land north of about 80°N: low-sun vegetation, and tree 1-2.
   Use an outside reference such as ESA WorldCover or MCD12 land cover; there are no labels on v00 tiles.
4. **Rule bookkeeping.** If adopted, CR is a new post-model rule (P3 in `AICoE_10142026/rules_c_vs_python.csv`). Any version name is
   set by the project lead.
