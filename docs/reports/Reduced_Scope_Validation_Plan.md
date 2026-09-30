# Reduced-Scope Plan: Validate Findings 1–2 Fix via Metrics-Only Regen on h09v05 + h12v02

Status: **executed 2026-09-15; results in `Phase5_6_Before_After_Summary.md`** (originally written as a plan, 2026-09-15).

Companion to `Sync_Plan_and_Model_Retraining.md` and
`Sync_Plan_and_Model_Retraining_Alternate.md` (in `pge61` repo root) and
`MOD_PR44B_Metrics/C_vs_Python_Code_Differences.md` /
`HOWTO_C_vs_Python_Comparison.md`. This plan narrows those to the cheapest
slice that still answers "did the Finding 1–2 sync narrow the C/Python gap":
apply the two decided code fixes, regenerate **metrics only** (no model
retraining) for **two tiles only** (h09v05, h12v02), and compare against the
existing baseline reports. Phases 4–6 (retrain, full compare, writeup) of the
original plan are explicitly deferred, not part of this pass.

## Execution environment

**Every command in this plan — builds, `gen_metrics` runs, the Singularity
Python runs, and `compare_c_python_metrics.py` — must run on compute node
`forest203`, not a login node (e.g. `adaptlogin201`/`202`).** This matches
existing practice for this project's VCF batch/compare work
(`HOWTO_C_vs_Python_Comparison.md` states the same for the original
procedure) and matters concretely here: the C build/run needs
`LD_LIBRARY_PATH=/explore/nobackup/people/mcarrol2/.local/lib` for its HDF4
libraries, which an interactive login shell on forest203 picks up
automatically via shell profile but a non-interactive `ssh host 'command'`
invocation (e.g. from a login node) does not — this caused a failure once
already in the original comparison work. Detach every long-running step
(`nohup`/`setsid ... disown`, as shown in Phases 3 and 4) so a dropped
connection can't kill a multi-tens-of-minutes run mid-way.

## Why these two tiles, and what they will and won't tell you

- **h09v05** is the clean signal for Findings 1 and 2: Finding 6 (Python's
  disabled masking-sentinel filter) stays latent there, so every metric in
  the comparison table is a genuine, uncontaminated read on whether the
  zenith-cutoff and QA-bit fixes moved the numbers. This tile is the primary
  "did it work" check.
- **h12v02** is the tile where Finding 6 was discovered — it will confirm
  the fix doesn't regress anything, but **its comparison table will still be
  dominated by Finding 6**, which this plan does not fix (out of scope,
  targeted-fix approach only). At h12v02, watch specifically
  `BandReflMaxGreenness`/`BandReflMaxTemp` (rank-selected, structurally
  insulated from Finding 6) for improvement — the `BandReflMax`/`Min`/
  `Median`/`AmpBandRefl` rows will likely look similar to the existing
  baseline report regardless of whether Findings 1–2 are fixed correctly.
- **Known gap in this reduced scope:** three other tiles in the 63-tile
  batch (h18v03, h13v01, h16v01) show numeric anomalies their own reports
  flag as *not* explained by Findings 1–6 — unconfirmed, undiagnosed. Using
  only h09v05/h12v02 will not surface or rule out whatever is happening at
  those tiles. Acceptable for a quick "is the sync directionally correct"
  check; not a substitute for the full batch re-run before calling Findings
  1–2 closed.

## Grounding — exact code changes (verified 2026-09-15, re-verify before editing)

**C (`fix_bands.c`, both `MOD_PR44B1/` and `MOD_PR44B3/` copies, identical
change at the same line in both):**
```c
// line 186 (unchanged — already canonical bit 6):
adjacency = (arrays[j][i * 4800 + ii] >> 6) & 1;
// line 187 (change):
aerosol = (arrays[j][i * 4800 + ii] >> 13) & 3;   // OLD
aerosol = (arrays[j][i * 4800 + ii] >> 7) & 3;    // NEW (bits 7-8, mask 0x180)
```

**Python (`model/BandDayFile.py`):**
```python
# line 17 (change):
ZENITH_CUTOFF = 70   # OLD — note: code-differences doc says the prior
                      # session found 72; live grep on 2026-09-15 shows 70.
                      # Re-grep immediately before editing — this value has
                      # drifted once already and may have moved again.
ZENITH_CUTOFF = 65   # NEW (canonical, matches C's PCF value)
```

**Python (`model/ProductTypeMod44.py`, `createQaMask`, lines 75–76):**
```python
adjacency = state & 8192        # OLD (bit 13, confirmed always-0 post-repack)
adjacency = state & 64          # NEW (bit 6)

aerosol = (state & 192) >> 6    # OLD (bits 6-7)
aerosol = (state & 384) >> 7    # NEW (bits 7-8, mask 0x180 = 384)
```

Do **not** touch `ProductTypeMod09A.py` / `ProductTypeMod09G.py`'s own
`createQaMask` copies — Findings 1–2 are specifically about MOD44C's
already-repacked `state` SDS, not MOD09's.

## Directory layout (under `AI_code_sync_09152026/`)

```
C_code/pge61/                    ← curated copy of pge61, original structure preserved
Python_code/modis_vcf/           ← full copy of modis_vcf, original structure preserved
Python_code/core/                ← full copy of core (modis_vcf's sibling dep), same as HOWTO's bind-mount
Metrics/h09v05/{C,Python,compare}/
Metrics/h12v02/{C,Python,compare}/
Model_outputs/                   ← reserved, unused this pass (Phase 4/5 deferred)
Reports/                         ← this doc; post-run writeup goes here too
```

Baseline ("before") artifacts are **not copied** — referenced read-only in
place at `/explore/nobackup/people/mcarrol2/MOD44B_VCF_Development/VCF_outputs/`
(`Metrics/h09v05.2019.metrics.bin`, `PythonMetrics/h09v05/2019/3-Metrics/*.tif`,
same for h12v02) and their existing writeups
(`pge61/MOD_PR44B_Metrics/C_vs_Python_VCF_Metrics_Comparison_{h09v05,h12v02}.md`).

## Phase 0 — Copy code (not data), preserving original repo structure

**C side.** `pge61` is 15G total, almost entirely disposable: `build_context/
runtime/intermediate/` (15G — this is B1's own *output* location from past
runs, i.e. old `.fix.out`/`weka_training.*.arff` files; confirmed by finding
`Band_1.2019.h09v05.fix.out` there directly) is safe to exclude — it holds
prior *output*, not a dependency, and B1 will recreate it fresh when rerun.
`v5.0.3samp/` (1.8G) **must be kept** — `build_context/pipeline_logs/b1/
h09v05.log` shows the prior h09v05 B1 run reading
`v5.0.3samp/h09v05.samp.bin` directly, i.e. it's a real input, not disposable
sample data.

```bash
mkdir -p /explore/nobackup/people/mcarrol2/MOD44B_VCF_Development/AI_code_sync_09152026/C_code
rsync -a \
  --exclude='.git/' \
  --exclude='build_context/runtime/intermediate/' \
  --exclude='build_context/pipeline_logs/' \
  /panfs/ccds02/nobackup/projects/ilab/projects/MODIS-VCF/pge61_08312026/pge61/ \
  /explore/nobackup/people/mcarrol2/MOD44B_VCF_Development/AI_code_sync_09152026/C_code/pge61/
```

**Python side.** `modis_vcf` (2.8G, mostly `data/` at 2.6G) and `core` (15M)
— copy both in full; `data/` is referenced by several unit tests
(`test_BandDayFile.py`, `test_CompositeDayFile.py`, `test_Metrics.py`,
`test_ProductTypeMod09A.py`), and 2.8G is trivial against 2.0P free on the
target filesystem, so no need to curate it out.

```bash
mkdir -p /explore/nobackup/people/mcarrol2/MOD44B_VCF_Development/AI_code_sync_09152026/Python_code
rsync -a --exclude='.git/' \
  /explore/nobackup/people/rlgill/innovation-lab-repositories/modis_vcf/ \
  /explore/nobackup/people/mcarrol2/MOD44B_VCF_Development/AI_code_sync_09152026/Python_code/modis_vcf/
rsync -a --exclude='.git/' \
  /explore/nobackup/people/rlgill/innovation-lab-repositories/core/ \
  /explore/nobackup/people/mcarrol2/MOD44B_VCF_Development/AI_code_sync_09152026/Python_code/core/
```

Both copies exclude `.git` entirely (per standing project rule: never
commit/push in rlgill's repos, and don't carry his remote into a working
copy) — these are plain directory trees from here on, not clones.

**Repoint the two hardcoded-path harness scripts in the C copy.**
`C_code/pge61/build_context/scripts/run_full_pipeline.py` and
`generate_tile_pcf.py` both hardcode
`WORKSPACE = Path("/panfs/.../pge61_08312026/pge61")` (the **original**
path) — confirmed by grep. Left unedited, running them from the copy would
silently read/write the original rlgill-owned tree. Edit `WORKSPACE` in both
copied scripts to point at
`.../AI_code_sync_09152026/C_code/pge61` before using them.

## Phase 1 — Apply Finding 1 fix (Python copy only)

Edit `Python_code/modis_vcf/model/BandDayFile.py` line 17 as shown above
(70 → 65, after re-confirming the live value). No C change (C's PCF-driven
`zenith_cutoff` is already 65 — confirm by checking the PCF used for the B1
run in Phase 3, not the `.pcf` template value assumed here).

## Phase 2 — Apply Finding 2 fix (both C copies + Python copy)

Edit the three files/lines shown in Grounding above:
`C_code/pge61/MOD_PR44B1/fix_bands.c:187`,
`C_code/pge61/MOD_PR44B3/fix_bands.c:187`,
`Python_code/modis_vcf/model/ProductTypeMod44.py:75-76`.

## Phase 3 — Regenerate metrics (C side)

Unlike a pure recompile, Finding 2's fix lives in `fix_bands.c`, which runs
during B1's masking/QA step and writes new `.fix.out` intermediate files —
`gen_metrics` only consumes `.fix.out` + unmodified `make_metrics.c`, so the
fix has **no effect** until B1 itself is rerun with the edited code to
produce fresh `.fix.out` files. This is more than the original HOWTO's Step
1 assumed (it treated `.fix.out` as a pre-existing prerequisite).

3a. **Build the full MOD_PR44B1 executable** from the edited copy (not just
`fix_bands.o` standalone — it's linked into the full B1 target per
`MOD_PR44B1.mk`'s `OBJ` list):
```bash
cd .../C_code/pge61
export PGE_WS=$PWD   # overrides build_context/env.sh's default, which points at the original repo
. build_context/env.sh
cd MOD_PR44B1 && make -f MOD_PR44B1.mk   # confirm exact target name/flags against the .mk before running
```
Do the same for `MOD_PR44B3` only if you decide to also validate B3
(prediction) — not needed for a metrics-only check, since `gen_metrics`
reads B1's `.fix.out` directly.

3b. **Regenerate `.fix.out` for h09v05 and h12v02 only.** `run_full_pipeline.py`
as written has no tile filter — it loads every tile in `tile_audit.json` for
whichever `--only-stage` is given, so running it as-is would rebuild all 63
tiles' B1 stage, defeating the point of scoping to two tiles. Instead, use
`generate_tile_pcf.py --tile <tile> --pge b1` (it already supports
single-tile mode) to produce a PCF for just h09v05 and just h12v02, then
invoke `bin/MOD_PR44B1.exe <generated-pcf>` directly for each, rather than
going through the 63-tile harness. **Confirm at execution time**: exact
`generate_tile_pcf.py` output path/filename pattern and `MOD_PR44B1.exe`'s
argument form (PCF path vs. env var) — not fully traced in this planning
pass, quick to check by reading the script and one existing
`build_context/pipeline_logs/b1/*.log` for the invocation pattern.

Confirm success the same way the original B1 run did — check
`build_context/runtime/intermediate/` for fresh
`Band_*.<year>.<tile>.fix.out`, `state.<year>.<tile>.fix.out`, etc. for both
tiles, then copy/move them into
`Metrics/<tile>/C/` (keep the run's own output location as scratch, land the
result in the new workspace's `Metrics/` tree for a clean record).

3c. **Build and run `gen_metrics`** against the fresh `.fix.out` (same
command as `HOWTO_C_vs_Python_Comparison.md` Step 1, from the copy):
```bash
cd .../C_code/pge61/MOD_PR44B_Metrics
gcc -O2 -o gen_metrics gen_metrics.c make_metrics.c -lm
export LD_LIBRARY_PATH=/explore/nobackup/people/mcarrol2/.local/lib:$LD_LIBRARY_PATH
for TILE in h09v05 h12v02; do
  WATER=$(ls /css/modis/Collection6.1/L3/MOD44W-LandWaterMask/2019/001/MOD44W.A2019001.${TILE}.061.*.hdf)
  nohup ./gen_metrics $TILE 2019 "$WATER" \
    > .../Metrics/${TILE}/C/gen_metrics.${TILE}.log 2>&1 &
done
```
(Point `gen_metrics`'s output — normally hardcoded relative to its own
`VCF_outputs/Metrics/` convention, confirm/adjust the output path in
`gen_metrics.c` or move the `.bin` after — into
`Metrics/<tile>/C/<tile>.2019.metrics.bin`.)

## Phase 3.5 — Instrument `gen_metrics`/`make_metrics` for full-coverage validation

**Independent of Findings 1–2** — this phase can run immediately against the
**existing baseline** `.fix.out` files (no need to wait for Phase 1–2's fix
or Phase 3's B1 rebuild), and answers "does Python correctly represent the
original C metrics" more directly and more completely than the 7-of-24-metric
comparison in Phase 5. Do it on the copied
`C_code/pge61/MOD_PR44B_Metrics/gen_metrics.c` /`make_metrics.c`, producing a
**second, richer output format alongside** (not replacing) the existing
324-value-per-record `.bin`, so `compare_c_python_metrics.py` and the
existing 63-tile baseline reports keep working unmodified.

Three additions, all cheap because the C code already computes this data
internally and currently discards it:

**a. QA/cloud degraded-period bitmasks.** `gen_metrics.c:114` already calls
`make_metrics(&QA, &cloud)` and gets back two 8-bit masks (one bit per
period) — `QA` bit `n` set means period `n` had every sub-observation masked
and fell back to an unfiltered raw average (`make_metrics.c:85`); `cloud`
bit `n` set means that fallback also found no cloud-free sub-observation
(`:102`). Currently thrown away — `gen_metrics.c:118-119` writes only `row,
col, metrics[324]`. Add `QA`/`cloud` (2 bytes) to each output record. Gives
a pixel-exact, C-ground-truth signal for which periods C itself considers
degraded — directly testable against Finding 6 (does Python's
sentinel-contaminated period line up with the same pixels/periods, instead
of only inferring this from aggregate tile-wide masked-fraction
percentages) and makes Finding 5's QA-coverage comparison pixel-exact
instead of the ad hoc band-level percentage measurement done for h09v05.

**b. Full NDVI-rank and thermal-rank sorted arrays.** After the
greenest-sort (`make_metrics.c:159-166`) and thermal-sort (`:199-206`),
`metrics_save[k][0..7]` holds every band's value at every rank, fully
computed in memory — but only rank 0 plus six fixed-window means ever reach
`metrics[]`. This is the one addition that requires editing `make_metrics.c`
itself (not just `gen_metrics.c`): serialize the full 8-value array in both
rank orders (144 additional shorts: 2 orders × 8 ranks × 9 bands). Closes
the metric-mapping gap from 7/24 to most of Python's 24 metrics being
exactly comparable — unlocks `Greenest{3,6,8}MeanBandRefl`,
`Warmest{3,6,8}MeanBandRefl`, `Lowest{3,6,8}MeanBandRefl`,
`BandReflMin/MedianGreenness`, `BandReflMin/MedianTemp`,
`AmpGreenestBandRefl`, `AmpWarmestBandRefl`, `TempMeanGreenest3`.

**c. Raw 23-period per-band values.** `array[0..8][23]` (populated via
`fread` at `gen_metrics.c:109`) holds the un-averaged raw composites before
`make_metrics()` collapses them into C's 3-period rolling-average 8-period
scheme. Export the 8 values at Python's exact `_daysSought` indices
(0, 3, 6, 9, 12, 15, 18, 21 of 23) per band (72 additional shorts) — enough
to recompute Python's exact single-fixed-day aggregation directly from C's
own input pixels, isolating "does Python's arithmetic match" from "does
Python's period-selection design match C's windowed-average design"
(currently only a qualitative structural finding, not a quantified one).

**Run against:** the existing baseline `.fix.out` for h09v05 and h12v02
first — fast, available today, doesn't wait on any other phase in this
plan. Optionally extend to more/all 63 tiles later; this is orthogonal to
the two-tile scope reduction elsewhere in this plan. If Phase 3's B1 rebuild
proceeds, also rerun against the post-fix `.fix.out` so the richer
comparison reflects Findings 1–2 too, not just the baseline code.

**Output:** `Metrics/<tile>/C/<tile>.2019.metrics_extended.bin`, alongside
the standard `.bin` Phase 3c already produces there.

**New unknown:** `compare_c_python_metrics.py` only knows the 324-value
format and the 7-metric mapping — a companion script (or an extended mode
of the same one) is needed to read the new fields and map them to the
remaining 17 Python metrics plus the QA-bitmask agreement check. Not
written yet; straightforward given the mapping above, but not a small
"just add two fields" change once the comparison side is included.

## Phase 4 — Regenerate metrics (Python side)

Same as `HOWTO_C_vs_Python_Comparison.md` Step 2, mechanically unchanged
except the two bind-mounts now point at the edited copies instead of
rlgill's checkout:

```bash
for TILE in h09v05 h12v02; do
  setsid nohup singularity exec \
    --bind /css/modis:/css/modis \
    --bind /explore/nobackup/people/mcarrol2:/explore/nobackup/people/mcarrol2 \
    --bind .../Python_code/modis_vcf:/usr/local/ilab/modis_vcf \
    --bind .../Python_code/core:/usr/local/ilab/core \
    /explore/nobackup/projects/ilab/containers/modis-vcf-latest \
    python3 /usr/local/ilab/modis_vcf/view/makeMetrics.py \
      -o .../Metrics/${TILE}/Python -y 2019 -t ${TILE} \
    > .../Metrics/${TILE}/Python/${TILE}.2019.makeMetrics.log 2>&1 < /dev/null &
  disown
done
```
Remember the autofs bind-mount gotcha from the HOWTO: bind at
`/css/modis` and `/explore/nobackup/people/mcarrol2`, never at `/css` or
`/explore` themselves. ~45–50 min/tile, so both can run in parallel/detached
as shown; expect ~1 hour wall time total for this phase.

Confirm completion: 24 `.tif`s present under
`Metrics/<tile>/Python/<tile>/2019/3-Metrics/`, log ends cleanly.

## Phase 5 — Compare, before vs. after

Copy `compare_c_python_metrics.py` (already in the C_code copy via
`MOD_PR44B_Metrics/`) and repoint its two hardcoded paths (lines 32–33) at
the new `Metrics/<tile>/C/<tile>.2019.metrics.bin` and
`Metrics/<tile>/Python/<tile>/2019/3-Metrics` instead of the shared
`VCF_outputs/` location, then run it per tile, capturing output:

```bash
cd .../C_code/pge61/MOD_PR44B_Metrics
singularity exec --bind /explore/nobackup/people/mcarrol2:/explore/nobackup/people/mcarrol2 \
  /explore/nobackup/projects/ilab/containers/modis-vcf-latest \
  python3 compare_c_python_metrics.py h09v05 > .../Metrics/h09v05/compare/comparison.h09v05.after.txt
# same for h12v02
```

**Before/after diff**: place the existing baseline table (reproduce by
rerunning the same script unmodified against the original `VCF_outputs/`
paths, or simply copy the numbers already in
`C_vs_Python_VCF_Metrics_Comparison_{h09v05,h12v02}.md` §3) alongside the new
`comparison.<tile>.after.txt` in `Metrics/<tile>/compare/`, and diff the
`bias`/`mae`/`rmse`/`corr`/`exact%` columns row by row.

**What "the fix worked" looks like:**
- h09v05: bias/RMSE shrink and `corr`/`exact%` rise across most of the 7×9
  rows, especially the ones the interpretation section tied directly to
  Finding 5's mask-coverage gap (`BandReflMaxGreenness`/`BandReflMaxTemp`
  had the worst scatter at baseline — 0.948–0.96 — since they're most
  sensitive to which periods each side calls "clean").
- h12v02: `BandReflMaxGreenness`/`BandReflMaxTemp` rows improve similarly;
  the Finding-6-affected rows (`BandReflMax` reflectance bands at 0 valid
  pixels, `BandReflMin`/`Median`/`AmpBandRefl`'s large bias) are expected to
  look about the same as baseline — that's not a failure of this fix, it's
  Finding 6 being out of scope.

**Optional, higher-value but not scripted anywhere today:** rerun the
Finding-5 QA-mask-coverage measurement (period-by-period "clean" agreement
between C and Python, `Band_1`, land pixels only — see
`C_vs_Python_Code_Differences.md` §2 of the h09v05 report for the exact
method) against the new fix_bands/QA-bit code. This is the most direct
signal that Findings 1–2 changed pixel-selection behavior as intended, but
it was ad hoc analysis in the original session, not a saved script — would
need to be written fresh (a straightforward adaptation of the existing
method, not a new investigation). **If Phase 3.5 is done**, this measurement
should use its per-pixel `QA`/`cloud` bitmasks directly instead of
re-deriving validity from `state` bits — pixel-exact rather than inferred.

**If Phase 3.5's extended output is available**, also run the companion
comparison it requires (see that phase's "new unknown") — this extends
coverage from the 7 metrics above to most of Python's 24, which is a
materially stronger answer to "does Python correctly represent the original
metrics" than the base Phase 5 table alone.

## Phase 6 — Write up

Short note in `Reports/`, not a full tile-report rewrite: before/after
tables for both tiles, whether the fix direction matched expectations, and
an explicit statement that h18v03/h13v01/h16v01's unexplained anomalies
remain unaddressed by this pass.

## Phase 7 — Final cleanup: sync reports back to the original working directory

**Not to be done now — a note for the final cleanup step before considering
this whole effort complete, once all validation (including any Finding 6
work) is finished.**

Everything in this plan deliberately operates inside the isolated copy at
`AI_code_sync_09152026/` rather than the original repo at
`pge61_08312026/pge61/`, per standing project rule (the original repo and
`modis_vcf` are owned by rlgill — no commits/pushes there, code changes get
made in a copy first). One consequence: every report and findings document
produced or updated *during* this validation pass currently exists **only**
in the copy, not in the original working directory — including
`execution_log.md`, `Phase5_6_Before_After_Summary.md`,
`Finding6_Fix_Plan_h12v02.md`, and the in-place updates to
`MOD_PR44B_Metrics/C_vs_Python_Code_Differences.md` (the new "Quantified on
h12v02" Finding 6 note). Left as-is, this analysis is stranded in a
throwaway copy and won't be visible to anyone working from the original
repo.

Before finishing: sync these **reports/documentation only** (not the code
edits themselves — those are a separate, more sensitive decision that
should involve the code owner, per this doc's own Recommendation 1) back
into the original working directory. This needs a deliberate pass, not a
blind copy — diff first, since the original's `C_vs_Python_Code_Differences.md`
and the two `C_vs_Python_VCF_Metrics_Comparison_*.md` files may have moved
on independently since this copy was made, and writing into rlgill's
directory tree warrants the same care this session already applied before
writing into any shared location.

## Remaining unknowns before this is executable

- **`generate_tile_pcf.py`'s exact output path and `MOD_PR44B1.exe`'s
  invocation form** (Phase 3b) — not fully traced this session, but the
  single-tile mode (`--tile`) already exists, so this is a quick check, not
  a redesign.
- **`gen_metrics`'s exact output path convention** — need to confirm
  whether it's hardcoded relative to a `VCF_outputs/Metrics/` sibling
  directory or takes an output path argument, before wiring Phase 3c's
  redirection into `Metrics/<tile>/C/`.
- **Re-verify `BandDayFile.ZENITH_CUTOFF`'s current value and the PCF's
  `zenith_cutoff`** immediately before editing (Phase 1) — the documented
  value (72) and the live-grepped value (70) disagree, meaning this file
  has changed at least once since the finding was first written; treat this
  plan's snapshot as stale by execution time, not authoritative.
