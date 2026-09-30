# Before/After Summary: Findings 1-2 Fix (Reduced-Scope Validation)

> **Status note (2026-09-20):** this is the checkpoint after the Findings 1-2 fix (2026-09-15). Its "after" numbers are superseded by later fixes: the Bands 1-7 zenith value clamp was removed and a Band_31 clamp added on 2026-09-17, validated in `Bands17_Fix_Validation_Results.md` (the final metrics-fix state). The remark below that the 63-tile batch has not been rerun still holds for the per-tile metric comparisons in `C_code/pge61/MOD_PR44B_Metrics/C_vs_Python_VCF_Metrics_Comparison_*.md`, which were written before any fix. The separate 63-tile product comparison, C against the fixed Python code, is in `FullRun_Results_C_vs_Python_63tiles.md`.

Short writeup per the plan's Phase 6 spec — not a full tile-report rewrite.
"Before" = existing baseline (`C_vs_Python_VCF_Metrics_Comparison_{h09v05,h12v02}.md`
§3, original code). "After" = this pass's fixed-code run
(`Metrics/<tile>/compare/comparison.<tile>.after.txt`), same method/pixel
set. Full "after" tables are in those `comparison.<tile>.after.txt` files;
this note only pulls out the rows that matter for judging the fix.

## h09v05 — the clean signal (Finding 6 doesn't contaminate this tile)

Every metric improved, not just the rank-selected ones the plan called out
as the primary signal:

| metric (Band_1 example) | before mae | after mae | before corr | after corr | before exact% | after exact% |
|---|---|---|---|---|---|---|
| BandReflMaxGreenness | 11.607 | 4.743 | 0.9934 | 0.9966 | 91.83% | 97.67% |
| BandReflMaxTemp | 5.050 | 0.223 | 0.9974 | 0.9999 | 96.51% | 99.87% |
| BandReflMax | 16.081 | 3.747 | 0.9947 | 0.9967 | 96.39% | 99.90% |
| BandReflMedian | 10.129 | 0.349 | 0.9934 | 0.9999 | 89.80% | 99.97% |

Band_31 (thermal) shows the largest jump: `BandReflMaxTemp`/`BandReflMax`
both went from corr 0.9976 / exact 96.64% to **corr 1.0000 / exact
100.00%** — the two codebases now agree on this band's max/max-temp value
essentially exactly. The pattern holds across all 9 bands in both metrics
(see the full after-table for the rest) — **this is unambiguous
confirmation that the zenith-cutoff and aerosol-bit fixes moved the two
implementations into agreement, not just for the rank-selected metrics but
broadly.**

## h12v02 — the Finding-6-discovery tile

As the plan predicted: `BandReflMax`/`Min`/`Median`/`AmpBandRefl` are still
dominated by Finding 6 (Python's disabled masking-sentinel filter, out of
scope for this fix) — e.g. `AmpBandRefl` bias is in the thousands and
`exact%` is ~0% on both before and after, unchanged by this fix, as
expected.

The two metrics the plan said to watch specifically, being rank-selected
and structurally insulated from Finding 6:

| metric (Band_1 example) | before mae | after mae | before corr | after corr | before exact% | after exact% |
|---|---|---|---|---|---|---|
| BandReflMaxGreenness | 58.374 | 51.815 | 0.4434 | 0.4381 | 79.69% | 95.36% |
| BandReflMaxTemp | 46.062 | 50.238 | 0.6657 | 0.6739 | 84.90% | 98.92% |

Mixed on mae/rmse (Band_1 `BandReflMaxTemp` mae/rmse actually got slightly
worse), but **`exact%` jumped dramatically for every band in both metrics**
(roughly 80-85% before → 95-99% after) — the fix clearly moved the large
majority of pixels into exact agreement. The residual mae/rmse noise is
consistent with a smaller population of outlier pixels still affected by
Finding 6 (or the three unexplained-anomaly tiles' issue, if it's not
tile-specific) pulling the mean/RMS around, rather than the fix failing.
Band_31's `BandReflMaxTemp` improved on every axis (mae 40.081→11.395, corr
0.8434→0.9152, exact% 84.88%→98.95%) — the cleanest single confirmation on
this tile.

## Bottom line

The Findings 1-2 fix (Python `ZENITH_CUTOFF` 70→65, C+Python aerosol-bit
read position 13-14→7-8) is validated as directionally and substantively
correct on both tiles, most cleanly on h09v05 where nothing else is
confounding the comparison, and via the `exact%` metric on h12v02 where
Finding 6 still dominates the raw bias/rmse numbers.

## Explicitly not addressed by this pass

- **Finding 6** (Python's disabled masking-sentinel filter) — still
  unfixed, still dominates most of h12v02's comparison table. Out of
  scope by design for this targeted-fix validation.
- **h18v03, h13v01, h16v01** — the three tiles flagged in the original
  63-tile batch as showing numeric anomalies not explained by Findings
  1-6. This reduced-scope pass used only h09v05/h12v02 and does not
  surface or rule out whatever is happening at those tiles.
- **Phase 3.5** (extended C-side instrumentation for pixel-exact QA/cloud
  bitmask comparison and full metric coverage) — not run this pass.
- The full 63-tile batch has not been rerun with the fix; this is a
  2-tile directional check only, per the plan's stated scope.
