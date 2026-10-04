# Keep solar zenith ≤ 0 inputs out of the fallback composite mean: screen result and plan

**RESULT 2026-10-04 16:00: experiment run; fails the pass rule; not adopted. See `Fallback_Fill72_Results.md`.** Written 2026-10-04; revised the same day after reviewing the earlier zenith work (section "Review against earlier work"). Status: screen done; plan awaiting approval.** Requested by the project lead: apply the existing `solz <= 0`
exclusion to the fallback path as well. Only the ≤ 0 part; not the upper cutoff (≥ 65), which has other known issues.

> **REVISED 2026-10-04, later the same day (supersedes the "solz ≤ 0 only" change below).** The project lead approved items 1-4 as an
> **experiment**:
> 1. **Usable input:** 0 < solz < 72. Earlier work (outside these reports) found MOD09 values usable only below 72°. A missing input
>    file counts as unusable.
> 2. **Fill rule:** where a pixel's period has **no** usable input, replace the composite (all bands) with that pixel's nearest valid
>    period in time.
>    - The year wraps.
>    - Ties take the mean of the two neighbouring periods.
>    - Mixed periods (some usable inputs) are unchanged.
> 3. **Scope screen first** (read-only); the project lead then chooses an explicit tile list.
> 4. **The fill replaces the Greenland zenith-only NDVI rule** on its 6 tiles in the test.
>
> **Pass rule (project lead):** bare is helped **and** tree cover is unchanged or better. Anything that helps bare but makes tree
> cover worse cannot be approved.
>
> **Why the revision:** the traces show dark low-sun observations (solz 86-90°, band 1 about 100-400) chosen by MOD44C on the ice
> sheet in April-June (midnight sun), alongside bright noon ones (solz 45-54°, band 1 about 9,500). The problem is not only polar
> night.
>
> The rebuild (item 5) waits for the screen and the tile list.

## Why

The QA mask (`ProductTypeMod44.createQaMask`) already rejects inputs with solz ≤ 0, so the **QA-pass mean** never uses them.
When a period has **no** QA-pass input, `CompositeDayFile._computeComposite` takes the mean of **all** inputs, with no zenith test. So
polar-night inputs (solz = 0, unreliable data; see memory `modis-solz-le-0-known-edge-case`) enter those composites.

This affects both layers, since they share the composites. On h16v02 it produces the straight edge at row 651/652
(`Bare_Mechanisms_Results.md`, point 7).

## Screen (done; `FullRuns/solz_screen_20261004`, read-only)

**Measure:** for each tile and period, the share of land where the period is a fallback **and** at least one present input has
solz ≤ 0.

**Result:**
- **Confined to 13 of 69 tiles, all at v ≤ 02, and only the November-February periods** (2019305, 2019353; 2020033 on the v00
  tiles).
- **The QA path is never affected:** 0.00% on every tile. The rule already works there.

| Tile | Fallback (all periods) | Fallback with a solz ≤ 0 input (mean over 8 periods) | Worst period | Every present input ≤ 0 |
|---|---|---|---|---|
| h17v01 | 97.5% | 22.7% | 2019353: 97.8% | 12.1% |
| h16v01 | 97.5% | 19.8% | 2019353: 90.5% | 9.1% |
| h21v01 | 70.3% | 12.9% | 2019353: 80.2% | 0.8% |
| h17v00 | 94.2% | 9.3% | 2020033: 74.1% | 0 |
| h16v00 | 93.4% | 7.3% | 2020033: 58.5% | 0 |
| h16v02 | 97.9% | 5.0% | 2019353: 32.1% | 0 |
| h17v02 | 73.6% | 3.0% | 2019353: 16.4% | 0 |
| h12v01 | 73.9% | 2.8% | 2019353: 22.1% | 0.04% |
| h15v02 | 82.7% | 2.4% | 2019353: 14.8% | 0 |
| h21v02 | 63.1% | 2.3% | 2019353: 14.9% | 0 |
| h12v02 | 59.9% | 2.0% | 2019353: 13.0% | 0 |
| h23v02 | 58.5% | 1.9% | 2019353: 12.6% | 0 |
| h20v02 | 63.3% | 1.7% | 2019353: 11.2% | 0 |
| h11v02, h13v01 | — | ≤ 0.05% | — | — |

h16v00 and h17v00 are set to tree 0 / bare 100 by the v00 rule, so their maps do not change. Their composites still matter if they
ever become training tiles.

## Review against earlier work (2026-10-04)

**What was adopted before:**
- The QA mask's `0 < solz < 65`, on the QA path only.
- The **zenith-only NDVI fix**, which sets NDVI to 0 in a period with no day at `0 < solz < 65`. It is scoped by an explicit list
  to the 6 Greenland tiles (`Metrics.ZENITH_ONLY_NDVI_TILES`) and changes NDVI only, never the bands.
- The Bands 1-7 value clamp was removed; Band 31 keeps a clamp to 16000 above 70°.

**Lessons that apply here:**
1. **Scope by explicit tiles; evaluate per tile.** Applied to every tile, the zenith-only rule worsened agreement with the labels on
   10 high-latitude tiles (`Zenith_Only_Fix_63Tile_Results.md`): h11v02, h11v03, h18v03, h18v04, h20v02, h20v03, h21v02, h23v02,
   h23v03, h24v03. The cutoff-70 retest did not remove that.
   - Four of them (**h11v02, h20v02, h21v02, h23v02**) are in this screen's 13 tiles.
   - That rule used both bounds. The damage most likely came from the upper bound (real low-sun data), which this plan leaves alone,
     as the project lead directed. Still, these four tiles get their own evaluation lines.
2. **Inputs with solz ≤ 0 are true "no observation" days.** `Polar_Night_Zero_Observation_Finding_h12v02.md` shows band = 0,
   solz = 0, state = 0 and obscov = 0 there, not the documented fill values. Today those zeros enter the fallback mean and pull every
   band toward 0. Leaving them out is the clean case, and it is exactly what the QA path already does.
3. **The all-zero case has a precedent.** Where a period has no usable zenith at all, the adopted Greenland rule sets NDVI to a
   "no signal" default (0) rather than inventing a value. Keeping today's all-input mean there (all zeros, or 10000 for missing
   files) is the minimal, consistent choice, and is now the recommendation.
4. **Per band:** the QA path tests each band's own input file (CQ for bands 1-2, CH for 3-7 and 31). The fallback exclusion will do
   the same.

**Correction to the expected outcome (traces, `bare_mech_20261004/out/trace/h16v02_edge`):** the h16v02 edge is **not** only a
solz ≤ 0 effect.
- **Above the edge,** the December input (DOY 353) is all zeros: no observation.
- **Below it,** some pixels have a **twilight observation at solz 93°**, with band 7 = −28672 (the fill value) and band 31 clamped to
  16000. The "solz about 37°" in `Bare_Mechanisms_Results.md` was the average of 0, 0, 0, 93 and 93.

So this change removes the zeros above the edge but keeps the twilight inputs below it (they are > 0). **The edge will change but is
not expected to disappear**; removing it would need the upper bound or a fill-value test.

**Side finding, not in scope:** in-file fill values (−28672) enter both the QA-pass and the fallback means. The pipeline substitutes
10000 only for missing files. The C code instead drops such pixels from prediction when band 1 has −28672 (its −3 rule). This is
worth its own check later.

## The change

In the fallback (case 2) mean, leave out inputs with solz ≤ 0. Missing inputs keep today's behaviour (10000 enters the mean), so
this change touches only zenith.

- **Case: every present input has solz ≤ 0** (up to 12% of land on h17v01 and 9% on h16v01, in the polar-night periods).
  - **Recommended:** keep today's all-input mean there, so no new kind of missing value enters the metrics. This is consistent
    with the Greenland NDVI precedent (lesson 3).
  - **Alternative:** treat it as missing, the way an absent input is (10000).
  - **Project lead to choose.**
- **Everything else is unchanged:** the QA path, the zenith-only NDVI fix on the 6 Greenland tiles, and the upper cutoff.
- **Where the change lives:** a copy of the frozen code in the run folder (`CompositeDayFile._computeComposite`), with the solz
  ≤ 0 mask kept beside each input. The delivery branch is not touched.

## Run plan (ngaproc202 + forest201-205; fully detached)

1. **Self-test (minutes):**
   - on h12v09 (no affected periods) the rebuilt composites must equal the stored ones byte for byte;
   - on h16v02, only fallback pixels with a solz ≤ 0 input may change, and only in 2019305/2019353.
2. **Composites and metrics** for the 13 affected tiles. The metric families the 60-tree tree and bare models need are about
   7.5 min per tile with the needed-bands builder: about 30 min.
3. **Training tables** for the affected training tiles with labels:
   - h11v02, h12v01, h12v02, h13v01, h16v01, h20v02, h21v01, h21v02 and h23v02 for tree cover;
   - the bare-label subset of those.

   Replace those rows in copies of the tree and bare splits. Acceptance:
   - keys and labels are identical;
   - unaffected tiles are identical cell for cell;
   - on affected tiles, only columns from the changed periods differ.
4. **Refit** T60 and B60 (seed 0) on the new splits: about 20 min.
5. **Maps for 69 tiles** with 60 trees and MEDIAN for both layers (the step-1 candidate settings): about 2 h.
6. **Evaluation** against the step-1 candidate (`cand60med_20261004`):
   - held-out labels by bin and region, with the affected tiles separately;
   - maps against C6, C6.1 and the C code;
   - the h16v02 edge (share at least 5 under C6 above and below row 651);
   - seed yardstick: 0.002-0.005 from earlier seeds.

   Then `Fallback_Solz_LE0_Results.md`.

**Total:** about 4-5 h of compute, starting after the step-1 candidate maps finish (the nodes are busy until then). Disk: about
60 GB (13 tiles of metrics, two splits, two models).

## Expected outcome and decision rule

- **Expected:** the h16v02 edge changes but does not disappear (see the correction above). Changes are small and confined to the 13 tiles. Small knock-on changes
  elsewhere come only through the refit.
- **Adopt if:**
  - held-out MAE on the affected tiles is no worse (within the seed yardstick), **per tile**, with h11v02, h20v02, h21v02 and
    h23v02 reported on their own lines;
  - nothing gets worse elsewhere beyond the yardstick.

  If only some tiles benefit, scope it by an explicit tile list, as the Greenland NDVI fix is. The decision is the project lead's.
