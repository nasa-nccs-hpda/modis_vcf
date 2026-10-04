# Git delivery of the modis_vcf changes (prepared 2026-09-21, committed 2026-09-30, push by the project lead)

## Status (2026-09-30)

**Pushed 2026-09-30 by the project lead:** `origin/mcarrol2/c-python-parity-fixes` = `335ceb2` (checked with `git ls-remote`). Next: a pull request into `develop` for rlgill. Before the push, the branch was committed locally in the delivery clone. The cluster has no GitHub credentials: the SSH key is rejected (`Permission denied (publickey)`), there is no HTTPS token and no `gh`. The project lead pushes from a terminal with their own credentials (see "To push" below).

- **Clone:** `/explore/nobackup/people/mcarrol2/MOD44B_VCF_Development/AI_code_sync_09152026/Git_delivery/modis_vcf/`
- **Origin:** `https://github.com/nasa-nccs-hpda/modis_vcf`. rlgill's checkout at `/explore/nobackup/people/rlgill/innovation-lab-repositories/modis_vcf` is a clone of the same repository and was not touched.
- **Base:** `origin/develop` at `b83e302` ("Quick utility", 2025-07-18). This was re-checked with `git ls-remote` on 2026-09-30, and develop has not moved.
- **Sparse checkout:** only `model/` and `view/` are checked out. `data/` (2.7 GB) is not.
- **Author:** `Mark L. Carroll <marklcar73@gmail.com>`, set in this clone only (`git config`, not `--global`).
- **The workspace copy `Python_code/modis_vcf` matches the branch** for all 16 files (checked byte for byte). The files it replaced are in `Python_code/_backup_before_sync_20260930/`.

## Commits (oldest first)

| Commit | What | Evidence |
|---|---|---|
| `1c3ac20` | QA mask: adjacency is `state` bit 6; the aerosol test is replaced by `state` bit 15 (PGE72's overall QC flag). Zenith cutoff 65; Bands 1-7 value clamp removed; Band_31 clamp added. `BuildTraining` honors `trainingDir`. | `Bands17_Fix_Validation_Results.md`, `Aerosol_QA_C_Full63_Retrain_Results.md` |
| `e0e6478` | New `CAttributes.py`, `CMirrorMetrics.py`, `CEquivalentMetrics.py` (Python mirrors of C's 252 derived metrics); `Metrics.py` read cache. | `OptionC_ImplementationB_Validation_Results.md` |
| `c69c32d` | cuML `RandomForestRegressor` replaced by scikit-learn `BaggingRegressor` (**design change**, separate so it can be reverted alone); threaded, chunked prediction; new `view/vcfPredictQueue.py`. | `Parallel_CPU_Use_Assessment.md`, 63-tile run |
| `2a37f9e` | `VcfPredict`: clamp to 0-100, layers sum to 100, codes 200/250 on all layers; tests. | `NonTreeVeg_Water_OutOfProj_Code_Propagation_Plan.md` |
| `8f42c83` | New `view/fitAltModel.py` (RF, XGBoost, HGB fits; XGBoost tunedB settings in its docstring). | `Regression_Tree_Methods_RF_XGBoost_Results.md` |
| `335ceb2` | `docs/reports/`: copies of the 51 `Reports/*.md` documents as of 2026-09-30, plus a short README. Another cluster user's name is replaced by "another user" in the copies. | |

`c69c32d` holds `VcfPredict.py` and `test_VcfPredict.py` as they were in the 63-tile run's code copy. `2a37f9e` adds the 2026-09-27 fix on top, so each commit matches code that actually ran.

## Checks done

- All 16 changed Python files compile (`py_compile`).
- `model/tests/test_ProductTypeMod44.py`: 13 passed, run in the `modis-vcf-latest` container on forest203 (NumPy 1.24.4).
- Bit-15 mask spot check: bit 15 alone passes; bit 15 with the adjacency bit fails; the old bits 7-8 no longer reject.
- `state` is read as int16, so bit 15 is its sign bit. The mask widens to int32 before testing, so it also works under NumPy 2, where `int16 & 32768` raises `OverflowError`.
- **Not run:** the other unit tests. `test_VcfPredict.py` and `test_MonteCarloSim.py` need real models and tile data, and `test_MonteCarloSim.py` was out of date before these changes. The code in `c69c32d` is what the 63-tile, regression-tree and aerosol QA runs used. The bit-15 line is what the qaC run used. The final `VcfPredict` output fix was validated on real RF and XGBoost output on 2026-09-27.

## Deliberately left out

- `view/editLabeledTraining.ipynb` and `view/stateDump2.py`: rlgill's own work.
- The `modis_water` shim and `core`: workspace-only or unchanged.
- The C change (`fix_bands.c`, aerosol `>> 13` to `>> 7`): it belongs to the pge61 repository and implements the bits 7-8 screen, which has since been shown to be wrong. Do not apply it.
- Run harness scripts (`FullRuns/*/code`, `scripts`) and all data and models.

## To push (project lead, in a terminal)

```bash
cd /explore/nobackup/people/mcarrol2/MOD44B_VCF_Development/AI_code_sync_09152026/Git_delivery/modis_vcf
git fetch origin && git log --oneline HEAD..origin/develop    # empty = develop has not moved
git log --oneline origin/develop..HEAD                        # the 6 commits
git push -u origin mcarrol2/c-python-parity-fixes             # prompts for GitHub username + token over HTTPS
```

If you want a different author email on GitHub, run this before pushing:
`git config user.email <address> && git rebase -r origin/develop --exec 'git commit --amend --no-edit --reset-author'`.
Then open a pull request into `develop` for rlgill to review.

## Cleanup

`rm -rf .../AI_code_sync_09152026/Git_delivery` removes the clone. Nothing else was created for the delivery.

## 2026-10-04: docs snapshot and v1.1.0

**On `mcarrol2/c-python-parity-fixes`:**
- `0959749` "Refresh docs/reports snapshot (2026-10-04)": 44 new and 4 updated reports. One username is replaced by "another
  user". `WV2_Reference_Scenes_Provenance.md` is left out (NGA rule; pending the project lead's OK).

The branch is now 10 commits ahead of GitHub.

**New branch `mcarrol2/v1.1.0`** (from `0959749`):
- `1ac9bc7` "Document v1.1.0": the README now gives 60 trees and MEDIAN for both layers. Production flags: `--tcAgg median --nvAgg
  median`.
- **The code is unchanged**: v1.1.0 is defined by the models and the flags. See `FullRuns/v110_sd_20261004/manifests/VERSION_v1.1.0.md`.
- Local annotated tag `v1.1.0`. The tag `v1.0.0` is also only local.

**To push (the project lead; the cluster has no GitHub credentials):**
```bash
cd /explore/nobackup/people/mcarrol2/MOD44B_VCF_Development/AI_code_sync_09152026/Git_delivery/modis_vcf
git push origin mcarrol2/c-python-parity-fixes
git push -u origin mcarrol2/v1.1.0
git push origin v1.0.0 v1.1.0
```

**2026-10-04 evening:** the project lead's push attempt failed. `git ls-remote origin` afterwards: `mcarrol2/c-python-parity-fixes` still at
`335ceb2`, no `mcarrol2/v1.1.0`, no `v1.*` tags, so nothing was partly pushed. The local branches and tags are unchanged; the same three
commands apply. Planned from the office on 2026-10-05.

**2026-10-04, 18:25: new commit `0298e4e` on `mcarrol2/v1.1.0`:** percent layers written as uint8 (nodata 253), SD layers as uint16
(project lead's decision). Checked on h08v04: values identical to the v1.1.0 maps. The branch is now 2 commits past
`0959749`. The tag `v1.1.0` still points at `1ac9bc7`. To move it to this commit before pushing (only if the project lead wants it):
`git tag -f -a v1.1.0 0298e4e -m "v1.1.0: 60 trees and MEDIAN for both layers; uint8/uint16 output"`. The push commands are unchanged.


**2026-10-04 evening: new commits on `mcarrol2/v1.1.0`:**
- `920d06f` "Write the MODIS sinusoidal grid into the output GeoTIFFs": geotransform from the tile ID and the MODIS sphere
  (R = 6371007.181 m) instead of no geotransform and a WGS84-ellipsoid string. Matches C6.1 MOD44B on h08v04, h16v01, h31v11 within 1 µm.
  New test `testModisSinusoidalGrid`; 9/9 writer and member tests pass.
- A docs snapshot commit (the commit after `920d06f`): the new and changed reports since `0959749`, with the same rules (WV2 report
  left out; one username replaced by "another user").
- The tag `v1.1.0` still points at `1ac9bc7`. The project lead decided v1.1.0 = uint8 output; moving the tag was blocked by the
  session's permission guard, so the project lead moves it (`Decisions_For_20261005.md`, A1). The push commands above are unchanged.
