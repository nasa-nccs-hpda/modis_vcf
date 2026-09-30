# Git delivery of the modis_vcf changes (prepared 2026-09-21, committed 2026-09-30, push by the project lead)

## Status (2026-09-30)

The Python changes are **committed locally** on branch `mcarrol2/c-python-parity-fixes` in the delivery clone. The branch is **not pushed**. The cluster has no GitHub credentials: the SSH key is rejected (`Permission denied (publickey)`), there is no HTTPS token and no `gh`. The project lead pushes from a terminal with their own credentials (see "To push" below).

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
| `241623f` | New `view/fitAltModel.py` (RF, XGBoost, HGB fits; XGBoost tunedB settings in its docstring). | `Regression_Tree_Methods_RF_XGBoost_Results.md` |
| (last) | `docs/`: copies of the `Reports/*.md` documents as of 2026-09-30. | |

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
