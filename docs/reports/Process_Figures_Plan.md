# Figures of the whole MODIS-VCF process (open item F13): plan

**Status: APPROVED 2026-10-07 (P1-P5 as suggested); started the same day, in parallel with the v1.2.0 build (read-only, no nodes).**
**Update 2026-10-07 late morning:** inventory accepted by the project lead ("fine for now"); figures 0-6 regenerated after the v1.2.0
build; fig7 (one tile through every version, h12v02 + h09v05 + h16v02) added at the lead's request; narrative written:
`AICoE_10142026/Process_Overview_Figures.md`. Awaiting the lead's review.
**DONE 2026-10-07 afternoon:** fig8 (h12v09 cloud tests) and fig9 (before/after v0.0.0) added; fig0a holiday break collapsed. Project lead: "figures are good for now". It is the next priority after the v1.2.0 build (`FullRuns/v120_20261007`).
Scope was set by the project lead on 2026-10-07; see `Decisions_For_20261005.md` item 13.

## 1. What the figures must say

- **Audience:** management and developers. Each figure gets a one-line takeaway for management and enough labels and numbers for
  developers.
- **Starting point:**
  - The operational C code (PGE61) could not be built locally. Its toolkits (SDP Toolkit, HDF4, HDF-EOS2/5, api-trans, an embedded Java
    VM) are old, partly undocumented and no longer packaged.
  - The Python conversion (`modis_vcf`, develop `b83e302`) was incomplete, and it could not be checked because the C code could not be
    run.
- **The path:**
  - C build from source, 2026-09-03 to 09-12.
  - C-vs-Python parity and bug fixes.
  - The versions: v0.0.0, v1.0.0, v1.1.0, v1.2.0.
  - The experiments tried around them.
- **Magnitude:** how many experiments, and roughly how many compute hours.
- **Result to date:** the problems solved, and what is still open.
- **Two required messages:**
  - "Mask solutions" (rules and overrides) were tested and replaced by data-driven fixes.
  - Limitations of the MOD44C input were identified.

## 2. Figure set (proposed)

**Folder (project lead 2026-10-07):** every figure and table goes in `Reports/AICoE_10142026/`.

0. **What it took to build the C code (for novice and intermediate developers):**
   - A tally of every fix and revision needed just to build and run it, in four groups:
     - **Toolkit stack:** 10+ source packages built by hand into `~/.local` with no root access (ncompress, libtirpc, zlib, JPEG, SZIP,
       HDF4, HDF5, HDF-EOS2/5, SDP Toolkit, api-trans, JDK 8, Weka 3.6.14). Each has its own patch or workaround: no `.tar.Z` support,
       Sun RPC missing from glibc, HDF-EOS2 prototype changes, a dead leap-second path, LD_LIBRARY_PATH.
     - **Runtime configuration** that operations had always supplied: PCF logical IDs, the blank-line PCF trap, path-field conventions,
       the missing ODL MCF, the missing `header.bare.arff`, the processing year.
     - **Weka version archaeology:** `-P 6`, old serialized models, the class path, the gdb artefact.
     - **Latent C bugs** exposed by actually running the code: 5 in `MOD44B3.c`, `create_output_file.c`, `fix_bands.c` and `MOD44B2.c`,
       including two silent failures.
   - Shown as a "dependency wall": the chain from source tarball to a 63-tile run, with each obstacle marked. Each obstacle is labelled by
     what makes it hard: undocumented; fails silently; needs operations-only paths; needs old library versions.
   - Source: `C_to_Python_Conversion_Assessment.md`. The inventory step adds to it from the build trees in `~/.local/src` (retries and
     stale builds), the C source changes and any build notes, so the count is complete and not from memory.

1. **Timeline:** one row per phase (C build, parity, versions, experiments by theme), 2026-09-03 to today. Milestones are dated and
   labelled with their version names.
2. **Experiments run:**
   - Count by week and by theme: parity and bug fixes; model and aggregation; clouds and composites; bare and ice; QA and inputs;
     validation and assessment.
   - Each experiment marked adopted, dropped, or still under investigation.
3. **Compute magnitude:**
   - Node-hours by theme and cumulative, with a rough conversion to CPU-hours.
   - Two kinds of source:
     - Measured from the status files: 27 runs, about 250 node-lane hours.
     - Estimated from log timestamps: 28 older-style runs and the work before `FullRuns` (C build runs, `Python_code` tests).
   - Labelled "rough estimate, order of magnitude".
4. **Mask solutions vs data-driven solutions:**
   - Each rule tried: Sahara stretch, high-latitude box, summer-snow rule, fill72, unusable-inputs rule, fill-only, band 31 unclamped.
   - For each: what it was meant to fix, the result, and the data-driven fix that replaced it (MEDIAN of 60 trees, the h16v02 ice
     training rows).
   - Headline numbers, for example h16v02 interior ice below 100: 8.7% to 0.007%.
5. **MOD44C limitations found:**
   - Each item with its evidence: the bit-15 aerosol state (and the historical MOD44B aerosol caveat); the band-31 solar-zenith clamp on an
     emissive band; low-sun fallback means on ice (solz about 86 in June); polar-night zeros and the −28672 fill entering the fallback
     mean; 250 m and 500 m bands carrying different observations on h16v02 ice; cloud-fallback contamination; Terra band-5 detectors.
   - For each, whether it is handled in the product, worked around, or left upstream (PGE72).
6. **Result to date:**
   - A scorecard from v0.0.0 through v1.2.0 (held-out MAE for tree and bare; agreement with C6 and C6.1 and the C code) and a "problems
     solved / still open" list.
   - Map panels before and after where they help: h16v01 tree, and h16v02 bare from v1.1.0 to v1.2.0.
   - C6 is described as a comparison, not truth.

## 3. How

1. **Inventory first** (about 1 h, read-only), written to `AICoE_10142026/inventory.csv` (and `AICoE_10142026/c_build_fixes.csv` for figure 0):
   - One row per run in `FullRuns/` plus the earlier work: date, theme, question, outcome, node-hours (measured or estimated), source
     report.
   - **You review this table before any figure is made**, because the counts and hours will be quoted to management.
2. **Figures** (about 2 h): matplotlib PNGs, readable in both slides and documents, using one colour per theme across all figures.
   Saved to `Reports/AICoE_10142026/`.
3. **Narrative:** `Reports/AICoE_10142026/Process_Overview_Figures.md`, one short paragraph per figure. It stays local; nothing goes online without your
   OK.

No compute runs. No NGA imagery or anything derived from it; NGA appears only as a count of index metadata, if at all.

## 4. Decisions

| # | Question | Options | Suggested |
|---|---|---|---|
| P1 | Start date | (a) 2026-09-03, the C toolkit build (b) earlier, including 2025 groundwork (`metrics_test_05072025`) | **(a)**, with 2025 as one "before" note |
| P2 | What counts as an experiment | (a) one per run folder (b) one per tested variant (fits, rules, configurations) inside runs | **(a) and (b)**: runs as the headline, variants as a second bar |
| P3 | Compute unit | (a) node-hours (b) CPU-hours (c) both | **(c)**: node-hours measured, CPU-hours as an upper bound (12 or 24 cores per node) |
| P4 | Format | (a) PNG + markdown narrative (b) also a 16:9 slide set | **(a)**; (b) later from the same PNGs if wanted |
| P5 | Include AI-assistant context | (a) yes, a one-line note that the build and conversion were done in VS Code with AI coding assistants (as in `C_to_Python_Conversion_Assessment.md`) (b) no | **(a)** |
