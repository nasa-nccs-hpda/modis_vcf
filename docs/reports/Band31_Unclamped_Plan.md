# Keep band 31's real values (no solar-zenith clamp): plan

**Status: APPROVED by the project lead 2026-10-05 late evening, with the recommended options U1-U6 ("move forward with the plan").**
Evidence: `Ice_Sheet_Diagnosis_Results.md`.

> **Caution (project lead, 2026-10-05):** the clamp was clearly a design decision at some point. Someone wrote it deliberately (in
> fix_bands.c, for bands 1-7 and band 31), even if it only ever took effect on band 31 and its reason is not recorded. Undoing it is a
> choice that must be adopted with care: we do not know what it was meant to protect against (for example, unreliable thermal readings
> from night or twilight passes, a sensor or calibration issue at low sun, or making winter values uniform for the C models). Before
> any adoption: look for the reason (C code history and comments, PGE61 documentation, the original developers if reachable), check
> the unclamped band-31 values for implausible readings (the distribution by latitude band and period, and values outside a
> physical range), and adopt only if the evaluation is clearly better under D9 (b) and nothing in that search argues against it.

## 1. The question

`BandDayFile._getRaster` replaces Band_31 with 16000 wherever the input's solar zenith is above 70° (`VALUE_CLAMP_ZENITH_CUTOFF`). The rule
comes from the C code (fix_bands.c), where it only takes effect on Band_31. It has no physical basis there:
- band 31 (about 11 µm) is an **emissive** band, and the sun's angle does not affect the reading;
- 16000 is 160 K if the 0.01 K scale applies, colder than any measured surface on Earth.

On the Greenland ice the clamp turns real summer readings of about 25,000-26,000 into 16000. The model has almost no training rows with that
value in June, so the pixels land with Canadian Arctic tundra rows. This is the main difference between interior ice pixels at 100 and those
just below it (h16v02: 1.9 million pixels).

**Question:** with band 31's real values kept everywhere, and the models retrained, does the product get better? This is a pipeline
correction, not a new rule. It departs from the C code's behaviour, which is in line with the project's goal (Python quality, not C parity).

## 2. The change

- **One line of behaviour:** Band_31 keeps its raw value whatever the solar zenith. Bands 1-7 are already unclamped (since 2026-09-27).
- **Unchanged:**
  - the QA test (it still requires 0 < solz < 65 for a clean input, also for band 31; see U2);
  - compositing (case 1 and case 2, and 10000 for a missing file);
  - −28672 handling, which stays as in v1.1.0 unless the fill-only test is adopted first (see U4);
  - the Greenland zenith-only NDVI rule;
  - the v00 rule;
  - the predictor sets;
  - the model recipe (60 trees, seed 0, sqrt, MEDIAN for both layers).
- **What it changes downstream:**
  - every predictor built from band 31: raw `Band_31-Day_*`, `CSortedRank*-Band_31`, `CGreen*-Band_31`, `CTherm*-Band_31`;
  - **every `CTherm*` predictor of every band**, because CTherm ranks whole periods by band 31;
  - the predictors use only the composites, so nothing else changes.

## 3. Run plan (fully detached, adapt-detached-run skill; EXPERIMENTAL run `EXP_b31_unclamped_<date>`)

**Step 1, count (read-only; about 1 h).**
- For all 69 tiles and 8 periods, count the land pixels whose band-31 composite changes. A composite changes if at least one of its inputs
  has solz > 70 and the input enters the mean that is used:
  - a case-1 input passes QA only if solz < 65, so it is never clamped;
  - **in practice the change is in case 2 (fallback)**, where every input enters the mean.
- Report each tile's share of land changed, by period. This defines the rebuild tiles; we expect high latitudes, plus mid-latitude winter
  periods (h27v04: 29% of training rows clamped in 2019353).
- **Cheap check:** this count, with the self-test and the gate below, comes before any long step.

**Step 2, self-test (minutes).**
- **h16v02:** the rebuilt band-31 composites equal the diagnosis's unclamped rebuild (`thermal_check.py`) at all 14,267 sampled pixels in
  all 8 periods.
- **Unaffected pixels and bands:** byte-identical to v1.1.0, including bands 1-7 everywhere.

**Step 3, gate (fatal).** On one tile with no clamped input (a tropical tile, chosen from step 1, e.g. h12v09), the patched code reproduces
the base tree and bare training rows exactly.

**Step 4, composites and metrics** for the rebuild tiles. These are band 31 plus the derived metrics; bands 1-7 composites are reused unchanged.

**Step 5, training rows:**
- Recompute the band-31-dependent columns of the tree and bare splits on the rebuild tiles that have labels.
- **Acceptance:** keys and labels identical; other tiles identical cell for cell; on rebuild tiles only band-31 and CTherm columns differ.

**Step 6, refit** tree and bare with the v1.1.0 recipe (seed 0), about 20 min.

**Step 7, maps** for all 69 tiles (v1.1.0 settings), about 2 h on ngaproc201/202 with 2 map slots per node, plus free forest nodes.

**Step 8, evaluation** (vcf-eval-toolkit), against v1.1.0:
- **Held-out labels:** fill72 CTRL as control (same base split, same recipe; seed yardstick 0.002-0.005); by cover bin, by region
  (continent × latitude band) and by tile; rebuild tiles separately; the 4 lesson-5 tiles on their own lines.
- **Maps against C6, C6.1 and the C code:** comparisons, not truth; increases and decreases looked at separately.
- **The Greenland ice (reference, not truth):**
  - the same MCD12-selected counts as the baseline (tree > 0, bare < 100, < 90, by tile);
  - the h16v02 and h16v01 interior and margin groups;
  - the share of h16v02's leaf neighbours that come from h16v01 ice (re-run `diag_train.py` with the new models).
- **Mid-latitude winter tiles:** the tiles where step 1 finds clamping outside the Arctic get a line of their own.
- **Pass rule D9 (b):** bare improves and tree cover is unchanged or better.

**Time:** about 1 h for step 1, then about 6-8 h for steps 2-8, depending on how many tiles step 1 sends to rebuild (about 27 min per tile
for composites and metrics in the 2026-10-05 runs). Nodes: ngaproc201/202 once `EXP_fillonly_20261005` ends (about 01:00), plus free forest
nodes. Disk: about 60-100 GB.

## 4. What to expect

- **Ice sheet:** interior pixels just below 100 on h16v02 and h16v01 should move to 100, if the clamp is the main cause. That may be only
  partly so: h16v02's ice may still draw on Canadian tundra rows, because its low-sun fallback composites differ from h16v01's. If so, the
  next lever is southern ice-sheet training labels, which needs a label-source decision.
- **Margin pixels:** little change expected; they are probably ground.
- **Elsewhere:** changes where step 1 finds clamping. Winter thermal values at mid- and high latitudes stop being 16000 and become real cold
  readings. The models are retrained on the same rows, so this should be neutral or helpful; the evaluation will show it.

## 5. Decisions for the project lead

| # | Decision | Options | Recommendation |
|---|---|---|---|
| **U1** | Scope | (a) everywhere the clamp fires; (b) Greenland tiles only | **(a).** A physical correction, not a regional rule; per-tile evaluation shows any tile that gets worse. |
| **U2** | The solz < 65 test in band 31's QA | (a) keep it (one change at a time); (b) also drop it for band 31, so clean thermal inputs pass case 1 regardless of sun angle | **(a)** for this test. (b) is a natural follow-up: the QA's cloud and state tests still apply, but the sun-angle test is irrelevant to an emissive band. |
| **U3** | Retrain | (a) yes, refit both layers on the recomputed rows; (b) prediction only | **(a).** The training rows carry the clamp too (94,834 h16v01 ice rows in July). |
| **U4** | Order with the fill-only test | (a) run on v1.1.0 composites, separately; (b) on top of the fill-only composites if adopted | **(a).** One change at a time; the fill-only result is due about 01:00. |
| **U5** | Pass rule | D9 (b), as for every recent experiment, with the ice counts as a reference | **D9 (b).** |
| **U6** | Version if adopted | ask at adoption | **Ask; do not tag.** |

## 6. Files (when run)

- Run: `FullRuns/EXP_b31_unclamped_<date>/` with EXPERIMENTAL.md.
- Code: v1.1.0 code with `BandDayFile` patched (no band-31 clamp); helpers from `EXP_unusable_inputs_20261005` (`warm_metrics.py`,
  `recompute_split.py`, `fitAltModel.py`, `heldout_eval.py`, `compare_variants.py`, `map_check.py`, `eval_maps.py`); and from
  `EXP_icesheet_diag_20261005` (`thermal_check.py` for the self-test, `diag_train.py` and `ice_counts.py` for the ice evaluation).
- Results: `Reports/Band31_Unclamped_Results.md`.
