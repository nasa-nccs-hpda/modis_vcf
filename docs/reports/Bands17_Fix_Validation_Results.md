# Bands 1-7 value-clamp removal: validation results (both tiles)

Status: **fix applied and validated. Confirmed successful.** Supersedes the
"not yet decided" status in `BandReflMax_Bug_Investigation_Plan.md` and the
open question in `Zenith_Value_Clamp_Root_Cause_Full_Picture.md`.

## What changed

`BandDayFile.py`'s `_getRaster()` had both zenith-based value substitutions
(`solz <= 0 -> 16000`, `solz > 70 -> 16000`) removed entirely, since the
single-pixel trace (`Single_Pixel_Trace_h12v02_row50_col2264.txt`) confirmed
C's equivalent clamp (`fix_bands.c:210`) never executes for Bands 1-7 in real
production (reads the solar-zenith array before it's populated for these
bands). Scope: Bands 1-7 only, per instruction - Band_31 and the QA-mask
`solz>0` check were deliberately left untouched. Before committing to the
~45-50 min full-tile run, the fix was hand-verified and then automated-
pixel-verified (via the real, edited code path) against the already-traced
h12v02 pixel (row 50, col 2264) - both predicted `BandReflMax` would go
from 16000 to the correct 5954, matching C exactly.

## Full-tile validation (both tiles, `Python_bands17fix/`, vs. each tile's
C `metrics.bin`)

**h09v05** (already near-perfect pre-fix, per the earlier bias~3.7 baseline):
`BandReflMax`/`BandReflMin`/`AmpBandRefl` for **all of Band_1-7 and Band_31**
now bias=0.000, corr=1.0000, **100.00% exact** - full parity, no regression.
`BandReflMedian` likewise (bias 0.25, rounding-level, 100% exact).

**h12v02** (the catastrophic tile - `BandReflMax` was 0 valid pixels
pre-fix):
- **`BandReflMax` Band_1-7: bias=0.000, corr=1.0000, 100.00% exact** - the
  dominant failure is completely resolved, exactly as the single-pixel
  hand/automated verification predicted.
- **`BandReflMedian` Band_1-7: 99.99-100.00% exact**, bias ~0.25-0.30 -
  effectively perfect.
- **`BandReflMin`/`AmpBandRefl` Band_1-7: ~87% exact**, bias in the
  single-digit-to-teens range (was previously off by thousands, pinned at
  the 16000 sentinel). This residual is the already-understood, separate,
  out-of-scope mechanism: C's no-lower-zenith-bound QA flag accepts a
  genuine no-observation placeholder day (`solz=0`, confirmed via `obscov`
  in `Polar_Night_Zero_Observation_Finding_h12v02.md`) as a single "valid"
  day, while Python's stricter `solz>0` check correctly rejects it and
  falls back to a 3-day average pulling in two real, if zenith-degraded,
  readings - exactly the magnitude and direction predicted in the pixel
  trace, not a new/surprise divergence.
- **Everything touching `Band_31` or `NDVI`** (`BandReflMax`-Band_31,
  `BandReflMin`-Band_31 [bias 9148, 0% exact], `BandReflMaxGreenness`,
  `BandReflMaxTemp`, `TempMeanWarmest3`) remains meaningfully off. This is
  the separate, already-flagged, deliberately-untouched issue: C's clamp
  *does* fire for Band_31 (processed after `b=7` in `fix_bands.c`'s loop),
  but Python's `CompositeDayFile` still sets `applyQa=False` for `Band_31`,
  applying no zenith-based value handling at all to it. Not part of this
  fix's scope; a separate follow-up decision.

## Conclusion

The fix does exactly what the investigation predicted: it resolves the
dominant Bands 1-7 catastrophe on h12v02 to exact parity, leaves h09v05
unregressed (net slight improvement, in fact - was already alone at ~99.9%
pre-fix, edge cases now fully closed), and leaves precisely the two already-
identified, separate issues (the `solz<=0`/no-observation QA-mask residual
on Min/Amp, and the un-clamped `Band_31`) as the only remaining gaps - both
already understood, already attributed, and both explicitly out of scope
for this change.

**Next decision points, not part of this fix:** the separately-discussed
plan to run a scoped, two-tile (h09v05+h12v02) B1(already done)->B2->B3
weka retrain in the sync copy of `pge61`, to check whether these corrected
metrics train a sensible model - plan written up in conversation, not yet
executed.

## Addendum: Band_31 clamp added (follow-up, same session)

`BandDayFile._getRaster()` now applies the `solz > 70 -> 16000` clamp to
`Band_31` specifically (only band besides 1-7 that goes through this
method) - matching C's real behavior, since `Band_31` is `b=9` in
`fix_bands.c`'s band loop, processed *after* `b=7`/`solar_zenith`, so
unlike Bands 1-7 the clamp is not dead code there. Bands 1-7 remain
unclamped (unchanged from above). Hand/pixel-verified first
(`BandReflMax`(Band_31) for the traced pixel: C=28999, post-fix
Python=28999, exact match), then full-tile validated
(`Python_bands17_band31fix/`, `comparison.<tile>.bands17_band31fix.after.txt`):

- **h09v05**: Band_31 now bias=0.000/100.00% exact on
  `BandReflMax`/`Min`/`Median`/`BandReflMaxTemp`/`AmpBandRefl` (full
  parity; `BandReflMaxGreenness` unchanged at 97.67%, NDVI-driven, not
  part of this fix).
- **h12v02**: `BandReflMax`-Band_31 bias 11.395/98.95% exact ->
  **0.000/100.00% exact**. `BandReflMin`/`AmpBandRefl`-Band_31 collapsed
  from the huge ~9,148/9,137 outlier bias (0.00% exact) down to **~1,386
  bias, 87.01% exact** - landing at essentially the same residual rate as
  Bands 1-7's Min/Amp (87%), confirming this hits the same already-
  understood `solz<=0`/no-observation mechanism, not a new problem.

Both Bands 1-7 and Band_31 value-clamp fixes are now applied and
validated. The only remaining known gap is the shared `solz<=0` QA-mask
residual (already understood, already endorsed as Python being the more
correct side) and NDVI-touching metrics (separate, pre-existing, out of
scope).
