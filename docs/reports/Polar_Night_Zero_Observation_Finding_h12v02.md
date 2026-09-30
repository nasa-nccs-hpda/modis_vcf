# Why C reads exactly 0 for h12v02's winter high-solzen periods (windows 5/6, row 50 col 2264) — confirmed root cause, not chance

> **Outcome (2026-09-20 note):** the Bands 1-7 clamp fix this report calls "planned" was applied on 2026-09-17. As predicted here, the polar-night zero-observation mechanism remains as a small residual on `BandReflMin` and `AmpBandRefl`; see `Bands17_Fix_Validation_Results.md`. It is a documented difference, not an open bug.

Status: investigation only, no code changed. Extends
`Single_Pixel_Trace_h12v02_row50_col2264.txt`, which flagged day
337/353's `Band_1=0, solz=0, state=0` reading as "likely a genuine
no-data/no-coverage day" without confirming it. This report confirms
that directly against the raw archive granules and explains why
Python's corresponding window values (142, 31) come out greater than
zero while C's (0, 0) do not, even after the planned Bands 1-7
value-clamp fix is applied (this is an unrelated, separate mechanism
from that fix, and is expected to remain as a small residual
afterward).

## The question

For the traced pixel, windows 5 (raw days 305, 321, 337) and 6 (raw
days 353, 001, 017) are the two windows spanning Northern Hemisphere
winter at this high-latitude tile. C's real, persisted `metrics.bin`
value for both windows is exactly **0**. That exactness, on two
separate windows, was conspicuous enough to warrant checking whether it
was coincidence or a real, systematic data condition.

## What was checked

Read the raw `MOD44CQ.A2019337.h12v02` and `A2019353.h12v02` granules
directly via GDAL (not simulated), including each SDS's own documented
fill/no-data value and the `obscov` (observation coverage) band, which
neither `fix_bands.c` nor `BandDayFile.py` currently reads at all.

| SDS | documented fill value | pixel (50,2264), day 337 | pixel (50,2264), day 161 (summer, for contrast) |
|---|---|---|---|
| Band_1 | 36864 | **0** | 744 |
| solar_zenith | 255 | **0** | 46 |
| state | 65535 | **0** | 34432 |
| obscov | 0 | **0** | 32 |

The documented fill values for `Band_1`/`solar_zenith`/`state` are
36864/255/65535 — not 0. Yet all three read exactly 0 at this pixel on
both winter days, immediately ruling out "coincidentally a real value
of 0."

## The explanation: `obscov`

`obscov`'s own documented fill value genuinely is 0, and it means "zero
MODIS swath observations were available to composite into this grid
cell for this 16-day period." Tile-wide counts confirm this isn't
isolated to one pixel:

- **Day 337 (Dec 3):** 1,012,585 of 23,040,000 pixels (4.39%) have
  `obscov==0`, and every one of those pixels also has `Band_1==0`,
  `Band_2==0`, `solar_zenith==0`, `state==0` — a perfect match, not a
  partial overlap.
- **Day 353 (Dec 19, closer to winter solstice):** the same perfect
  match, now on 3,386,897 pixels (14.70%) — over 3x the fraction,
  growing smoothly as the pole tips further from the sun.
- **Day 161 (summer):** no zeros anywhere; real reflectance, real
  angle, real state, real nonzero obscov count.

**Conclusion: this is a real, systematic, physically-expected
no-observation condition (polar night → no usable solar-reflective
retrieval at this latitude/date), not chance.** It originates entirely
upstream, in the MOD44CQ Level-3 compositor: when `obscov==0`, the
compositor appears to simply never write a value into `Band_1`/
`solar_zenith`/`state` for that cell, leaving them at 0 rather than
each field's own documented fill sentinel — a minor inconsistency in
the granule-generation step, but not something either `fix_bands.c` or
`BandDayFile.py` creates or controls; both are just consumers of an
already-zeroed upstream input.

## Why C's window value is exactly 0 but Python's (post-fix) is not

Window 5's three raw days:

| day | raw Band_1 | raw solz | what it is |
|---|---|---|---|
| 305 | 316 | 86° | real reading, genuinely high zenith |
| 321 | 112 | 89° | real reading, genuinely high zenith |
| 337 | 0 | 0° | the no-observation placeholder (confirmed above) |

**C's path:** C's validity test only checks `zenith < 65`
(`fix_bands.c`'s `zenith_cutoff`). Days 305/321 fail that (86°, 89° >
65°) — correctly excluded, for the right reason. Day 337's zenith is
0°, which trivially satisfies `< 65`, so C accepts it as the *sole*
"valid" day and takes the Case-1 average = just that one value = **0**.
C ends up using only the fake placeholder, believing it to be a
legitimately low-zenith good observation.

**Python's path:** Python's QA mask (`createQaMask()`) additionally
requires `solz > 0`, which correctly rejects day 337 as suspect/no-data.
With zero days passing full validity, Python falls into the Case-2
fallback — average all three raw days regardless of QA — which now
includes the two real readings from 305 and 321 alongside 337's zero:
`(316 + 112 + 0) / 3 = 142.67 -> 142`. Window 6 (353→0 placeholder,
plus real days 001/017 at 37/56) works out the same way, giving 31
instead of C's 0.

So the nonzero Python value is not a bug: it's the direct consequence
of Python correctly refusing to treat the no-observation day as valid,
which pushes it into a 3-day average that (correctly) pulls in two
genuinely real, zenith-degraded-but-real reflectance readings that C's
window never uses at all. C's 0 is arguably the less physically
meaningful number of the two here — a single fake placeholder standing
in for the whole period — while Python's value is a real (if
zenith-degraded) average of actual measured reflectance.

## Implication for the current investigation

This does not change the Bands 1-7 value-clamp fix plan (that fix
targets a separate mechanism: the `>70` and `<=0` value substitutions
in `BandDayFile._getRaster()`, versus this finding which is about the
`solz>0` QA-*validity* threshold in `createQaMask()`, never touched by
that fix). It does two things:

1. **Confirms, with much stronger evidence than before, that the
   earlier-endorsed project decision to keep Python's `solz>0`
   QA-mask exclusion is scientifically correct** — `solz<=0` here isn't
   an ambiguous edge case, it's a directly-confirmed "zero valid
   observations" flag from the upstream product's own `obscov` field.
2. **Explains, and validates as expected/legitimate, the small
   residual gap** (C=0 vs Python≈31-142 on these two windows) that the
   hand-verification of the Bands 1-7 fix showed would remain on
   `BandReflMin`/`AmpBandRefl` for the traced pixel even after that fix
   is applied. That residual is not a new bug to chase — it is this
   mechanism, already understood and already endorsed as Python being
   the more correct side.

No code changed. This is background only, in support of the Bands 1-7
fix decision already in progress.
