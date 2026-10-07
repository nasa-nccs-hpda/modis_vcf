# Is the recurring problem "MOD44C composites and polar night"? A review of the reports

**Written 2026-10-06 at the project lead's request, as a reference.** Read-only review of existing reports; no run.

## Verdict

**Partly confirmed, but "polar night" is the wrong name.** The common thread is **low-sun inputs at high latitude reaching the composites
through the fallback mean, because almost nothing on snow or ice passes QA**. That includes polar night, but the evidence points more to
**summer near-horizon (midnight-sun) observations** than to the dark season.

Restated to fit the evidence:

> On snow and ice nothing passes QA, so every composite is a fallback mean. That mean mixes in the low-sun observations MOD44C selects:
> midnight-sun evening passes in summer, and zero-observation polar-night inputs in winter. The summer ones do the most damage, because
> they make the reflectance-based predictors unreliable on the ice. Rules that remove or replace inputs have not helped, because the
> models were trained on the same inputs. What is left is upstream (why MOD44C picks an 86° observation when noon ones exist) or in the
> training rows (too few ice rows outside h16v01).

## Confirmed

- **The polar-night data condition is real** (`Polar_Night_Zero_Observation_Finding_h12v02.md`): where MOD44C had no observation (`obscov`
  = 0), the bands, solar zenith and state are written as 0, not as their fill values; 4.4% of h12v02 on 2019337, 14.7% on 2019353.
- **It reaches our composites only through the fallback mean** (`Fallback_Solz_LE0_Plan.md`, screen): solz ≤ 0 never enters the QA path;
  in the fallback it does, on 13 tiles at v ≤ 02 and only in the November-February periods.
- **On the ice every period is a fallback** (`Bare_Mechanisms_Results.md` point 6; `Ice_Sheet_Diagnosis_Results.md` point 4): PGE72 sets
  state bit 15 = 0 for snow by construction, about 14 of 23 inputs have solz outside 0-65°, so the ice predictions rest on cloud- and
  low-sun-including means.

## Refuted: polar night as the main cause

- **The harmful inputs are mostly summer, low-sun ones.** MOD44C selects dark evening passes at solz 86-90° in April-July beside bright
  noon ones at 45-54° (`Fallback_Solz_LE0_Plan.md`, revision note; `Ice_Sheet_Diagnosis_Results.md`: 86° in June on h16v02, 86-87° in
  July on h16v01 / h17v01). On 2026-10-06 the interior-ice composites for June-August had non-physical reflectances: NDSI about 0.3,
  band 6 about 0.35, red near 0.03 in late August (`Ice_Extent_MOD44C_Reference_Plan.md`, revision note). The sun is up all day then.
- **Two fixes aimed at polar night did not help.**
  - `Fallback_Fill72_Results.md`: nearest-valid-period fill. Greenland changes at seed-noise size, tree cover rose on the ice, bare
    worse on the island tiles (+0.058). Not adopted.
  - `Composite_Unusable_Inputs_Results.md`: polar-night zeros treated as unusable made bare **worse** on exactly the polar-night tiles
    h12v01, h12v02, h13v01, h21v01 (+0.31 to +0.46 MAE each). Not adopted. The models were trained on the same zeros and handle them
    consistently; removing them breaks that.
- **v1.1.0 is right on most of the ice anyway.** On the MOD44C ice reference, interior ice is at 100 on 95.8-100% of pixels on five tiles,
  all from fallback composites (`Ice_Extent_MOD44C_Reference_Results.md`). The inputs hurt where the training does not match: h16v02's ice
  draws 54% of its leaf neighbours from h12v02 tundra and 3.6% from h16v01 ice (`Ice_Sheet_Diagnosis_Results.md` point 3).

## Other recurring causes that are not polar night

- **Band-31 clamp** (pipeline rule on solz > 70): removing it did not fix the ice and broke the interior of h16v01 and h17v01
  (`Ice_Extent_MOD44C_Reference_Results.md`).
- **MEAN against MEDIAN of the trees:** most of the 90-99 band and the h16v02 straight edge; MEDIAN fixed them (`Bare_Mechanisms_Results.md`,
  `Fallback_Fill72_Results.md`).
- **MCD12 as the ice reference:** its edge is too far out; most of h17v01's "misses" (`Ice_Extent_MOD44C_Reference_Results.md`).
- **Training-row geography** on h16v02 (above).
- **The MOD44C state word** repacked by PGE72, with the aerosol bit collision (`MOD44C_PGE72_Investigation/`).
- **Cloud fallback in mid-latitude dense forest** (`Cloudy_Fallback_Diagnosis_Results.md`): the same fallback mechanism, unrelated to sun angle.

## Open, upstream

Why MOD44C picks a dark 86° evening observation when bright noon observations exist in the same 16-day period. This belongs to the PGE72
investigation; no PGE61-side rule has been able to touch it.
