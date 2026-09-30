# Zenith value-clamp: full picture before deciding on a fix

Status: investigation only, no code changed (resolved 2026-09-17: the clamp was removed for Bands 1-7 and added for Band_31; see `Bands17_Fix_Validation_Results.md`). Builds on
`Single_Pixel_Trace_h12v02_row50_col2264.txt`, which found that C's
`fix_bands.c:210` zenith>70 reflectance-value clamp never actually fires
for Band_1-7 in real output, because it reads `arrayz` (solar zenith)
before that array has been populated for this invocation (Band_1-7 are
processed earlier in the band loop than `solar_zenith` itself). This
document checks what that means for the earlier "zenith-gap fix"
(`BandDayFile.VALUE_CLAMP_ZENITH_CUTOFF = 70`), which was declared
"confirmed working" on h09v05.

## Why h09v05 looked like a clean confirmation, but wasn't testing the
   right thing

Two things were checked when that fix was validated, and both, on
closer look, were structurally blind to the actual bug:

**1. `zenith_gap_diagnostic.py`'s own results
(`Reports/zenith_gap_diagnostic_results.txt`)** only ever measured the
narrow `65 < solz <= 70` band ("the gap zone" between the QA cutoff and
the old shared clamp threshold). After the fix, Python's clamp
threshold became exactly 70, so by construction **zero** samples in a
`<=70` range can ever exceed 70 and get clamped - the reported "0.00%
clamped, matches C's 0.00%" result for both tiles is close to a
tautology. It says nothing about whether Python's clamp at `solz > 70`
(the actual clamp that fires in production, on the actual population
that matters) matches C's real behavior there. It doesn't, per the
single-pixel trace: C never clamps Band_1-7 at any zenith, `>70`
included.

**2. The full-pipeline comparison declaring h09v05 at exact parity**
was real and correct as a comparison, but h09v05 essentially never
exercises this code path at all. Checked directly this session (pure
local read of the already-cached `Python_zenithfix` 1-Days rasters,
tile-wide, all 23 raw days, no container needed):

```
day       pixels clamped to 16000 (of 23,040,000)
2019065-2019305   0            (14 of 22 days: exactly zero)
2019321           14           (0.0001%)
2019337           487          (0.0021%)
2019353           620          (0.0027%)
2020001           103          (0.0004%)
2020017           18           (0.0001%)
2020033-2020049   0
```

At most 620 pixels out of 23 million on h09v05's worst day. That's not
"the fix correctly reproduces C's clamping behavior" - it's "the fix's
mechanism is essentially never invoked on this tile, so it can't
diverge from C no matter what it does." The "exact parity" result is
real but doesn't validate the clamp logic; it's consistent with the
clamp mattering approximately zero times on this tile either way.

h12v02, by contrast (same check, done implicitly via the single-pixel
trace and the earlier full-tile sample in this session), has entire
raw days where a large fraction of the tile is clamped - the traced
pixel alone had 10 of its 23 raw days pinned at 16000, spanning day
273 through day 049 of the following year. h12v02 is a high-latitude
tile; h09v05 is not. The fix's real-world effect is concentrated almost
entirely on exactly the tile it was supposed to fix, for the wrong
reason: not because it correctly restores parity there, but because
it's the only tile where its (incorrect) behavior has enough
opportunity to fire and do damage.

## The full per-band picture (not just Band_1)

Re-checked how the clamp/QA-skip logic is actually wired per band,
since the fix's correctness question is different for each:

- **Band_1 - Band_7** (`b=0..6` in `fix_bands.c`'s loop, processed
  before `b=7`/`solar_zenith`): C's `>70` value clamp is dead code
  (confirmed). Python's `VALUE_CLAMP_ZENITH_CUTOFF=70` fix makes these
  bands clamp for real, using real zenith - this is the mismatch
  driving h12v02's `BandReflMax/Min/Median/AmpBandRefl` catastrophe,
  and by the same mechanism would affect any other tile with a
  meaningful population of `solz > 70` raw days.
- **Band_31** (`b=9`, processed *after* `b=7`, so `arrayz` is real by
  then): C's `>70` clamp is **real and does fire** for this band.
  Python's `CompositeDayFile._getRaster()` explicitly sets
  `qa = False` for `BAND31` (`self.bandName == ProductType.BAND31`),
  which skips `BandDayFile._getRaster()`'s entire clamp block - Python
  currently applies **no** zenith-based value substitution at all to
  Band_31, in either direction. This is the opposite-direction
  mismatch from Bands 1-7: here C clamps and Python doesn't, whereas
  for Bands 1-7 Python clamps and C doesn't. This was flagged as an
  aside in the earlier per-metric derivation report but not
  investigated further at the time; it's now confirmed relevant to the
  same root cause.
- **NDVI**: `Metrics.getNdvi()` computes NDVI from
  `getBandCube(BAND1)` and `getBandCube(BAND2)` - i.e. from the same
  over-clamped Band_1/Band_2 composite cubes as any other metric built
  on those two bands. NDVI inherits Bands 1-7's over-clamping issue by
  construction; it isn't a separate code path.
- **state / cloud QA masks**: unaffected either way. `createQaMask()`/
  `createCloudMask()` read `solz` directly (not through the
  reflectance-value clamp path at all), and that logic already matches
  C's real, working `b=10`/`b=11` behavior (confirmed in the Findings
  1-2 work and re-confirmed in the single-pixel trace's `state.fix.out`
  column).

## Clarification: this is not a verdict that Python is right and C has
   been wrong all along

Asked directly whether this adds up to "Python makes the metrics
correctly and C has been wrong all along" - no, and it's worth being
precise about why not, since several statements above (e.g. "matching
C means Python should not clamp these") can read that way out of
context.

What was actually found is narrower and mixed, not a one-sided verdict:

- **On the `>70` value clamp (the dominant driver of the traced
  mismatch)**: C has a genuine coding defect - the clamp reads a
  solar-zenith array before it's populated, for Bands 1-7, so it never
  fires. That means C is not doing what *its own code* appears to say
  it should do for those bands. Python's fix implemented that apparent
  intent literally and, having no equivalent bug, actually executes it.
  That is a statement about C failing to match its own design, and
  about Python matching the design-as-written rather than C's
  behavior-as-shipped - it is not evidence that clamping at `solz > 70`
  is the scientifically correct thing to do at all. Nothing in this
  investigation independently validates the clamp's design intent
  against ground truth or an algorithm specification; it only compares
  two implementations against each other and against what the C source
  literally says.
- **Which one Python's fix should target - C's real (buggy) production
  behavior, or C's apparent design intent - is an open decision, not
  settled by this investigation.** This whole effort has been framed as
  validating Python against C as the trusted, flight-heritage
  reference. By that framing, matching C means reproducing its bug
  (Bands 1-7 never clamp), which is what the bullet points below
  assume. But it is equally valid to conclude C's bug should be fixed
  on the C side and Python's current behavior is the one to keep -
  that is a product/science decision about intended behavior, most
  appropriately made by whoever owns the C algorithm's design (per this
  project's convention, the code owner), not something to infer
  unilaterally from reading the source.
- **`Band_31` cuts the other way, with no such ambiguity**: C's clamp
  genuinely fires there as designed, and Python currently skips all
  zenith-based value handling for `Band_31`. There, Python simply does
  not match C, full stop - no bug-vs-intent question involved.
- **The `solz <= 0` edge case is separate from all of the above and was
  already reviewed and settled in an earlier session**: the user
  previously confirmed C's lack of a lower zenith bound is a known,
  accepted limitation, and Python's stricter exclusion there was
  already endorsed as correct. That piece of the picture is not in
  question the way the `>70` clamp is.

## What this means for deciding a fix (not decided here)

The earlier fix was a correct reading of what `fix_bands.c:210` says,
applied uniformly, but the code's *real* runtime behavior is
band-dependent because of where each band sits relative to `b==7` in
the loop. The options below are framed around *matching C's real,
shipped behavior*, per this project's stated goal of validating Python
against C - not because that is necessarily the scientifically "right"
answer (see the clarification above):
  - Bands 1-7 (and NDVI, derived from them): C never clamps -> matching
    C's real behavior means Python should not clamp these on
    `solz > 70` either. (Alternative: fix C's bug instead, and leave
    Python as-is - a decision for the code owner, not assumed here.)
  - Band_31: C does clamp -> Python currently doesn't clamp it at all,
    which is its own, separate discrepancy worth fixing at the same
    time, in the opposite direction. No bug-vs-intent ambiguity here.
  - The `solz <= 0` clamp (a separate, pre-existing mechanism from the
    same code block, not introduced by the zenith-gap fix) also only
    affects Bands 1-7 today, and its interaction with the `>70` fallback
    contamination seen in the single-pixel trace (windows 5/6) would
    also go away if Bands 1-7 stop value-clamping on `solz` altogether,
    since the fallback would then use the real (if still QA-excluded)
    raw value instead of a forced 16000.

No change has been made. This is the background needed to decide
whether to: (a) remove the `solz > 70` (and possibly `solz <= 0`)
value-clamp for Bands 1-7 entirely, (b) add the equivalent real clamp
to Band_31, or (c) something narrower - that decision is next, not
part of this investigation, and per the clarification above it should
account for whether C's bug or Python's current behavior is the one
that should actually change.
