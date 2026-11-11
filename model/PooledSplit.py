"""Pooled bare split: the training table of the pooled bare model (2000-2023).

The pooled bare model is one non-vegetated model used for every year.  It is fitted on the bare training rows of many years at
once, so that it does not depend on the spectral quirks of one year.  The labels do not change from year to year (the same pixels,
the same label rasters), so every year's bare split has the same rows in the same order; only the predictors (the year's metrics)
differ.

build() makes one split from the per-year splits:

- every year's Train part must have the same keys (tile, row x, column y) and the same labels, in the same order (fatal
  otherwise);
- each row takes all its predictors from one year.  The years are assigned per stratum (tile x label bin 0-20 / 21-40 / 41-60 /
  61-99 / 100): the stratum's rows are shuffled and dealt to the years in turn, from a random start, so every year gets the same
  share of every stratum (+-1 row).  Random generator: numpy default_rng(seed), seed 0 for the delivered model;
- the keys and labels are written as they are (the first year's keys file).

Outputs in <out>: xTrain.parq, yTrain.parq, keysTrain.parq (the inputs of view/fitAltModel.py), assign.parq (the keys plus the
year each row was taken from) and pooled_split.json (years, seed, predictors, rows per year, SHA-256 of the four output files and
of the source files).  check() re-reads the sources and fails unless sampled rows equal their year's row and every stratum is
even.

The delivered model was made as: build (years 2000-2023, the 21 bare predictors, seed 0) -> view/fitAltModel.py --algo rf
--nEstimators 60 --maxFeatures sqrt --minSamplesLeaf 1 --randomState 0 --trainingType pctbare -> model/CompactForest.py.
Command line: view/buildPooledSplit.py.
"""
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

KEY_TILE = 'tid-year'                  # '<tile>-<year>' in the split key files; the tile is its first 6 characters
LABEL_BINS = [-1, 20, 40, 60, 99, 100]  # label bins of the strata: 0-20, 21-40, 41-60, 61-99, 100
FILES = ('xTrain.parq', 'yTrain.parq', 'keysTrain.parq', 'assign.parq')
MANIFEST = 'pooled_split.json'


# ----------------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------------
def _bins(labels: np.ndarray) -> np.ndarray:
    return pd.cut(labels, LABEL_BINS, labels=False)


def _strata(keys: pd.DataFrame, labels: np.ndarray) -> np.ndarray:
    return keys[KEY_TILE].astype(str).str[:6].to_numpy() + '_' + _bins(labels).astype(str)


def _sha256(path: Path) -> str:

    h = hashlib.sha256()

    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1 << 24), b''):
            h.update(block)

    return h.hexdigest()


def readYears(tsv: Path) -> dict:
    """{year: split dir} from a file of lines '<year><TAB><split dir>'."""

    lines = [l.split('\t') for l in Path(tsv).read_text().split('\n') if l.strip()]
    return {int(y): Path(d.strip()) for y, d in lines}


# ----------------------------------------------------------------------------
# build
# ----------------------------------------------------------------------------
def build(splits: dict, predictors: list, out: Path, seed: int = 0, logger=None) -> dict:
    """Write the pooled split of the per-year bare splits ({year: split dir}) to out (which must not exist).  Returns the
    manifest written to out/pooled_split.json."""

    out = Path(out)
    out.mkdir(parents=True)
    years = sorted(splits)
    first = splits[years[0]]
    k0 = pd.read_parquet(first / 'keysTrain.parq')
    y0 = pd.read_parquet(first / 'yTrain.parq')

    # ---
    # Same rows in every year: tile, x, y and label, in the same order.
    # ---
    tile0 = k0[KEY_TILE].astype(str).str[:6].to_numpy()

    for y in years[1:]:

        k = pd.read_parquet(splits[y] / 'keysTrain.parq')
        yy = pd.read_parquet(splits[y] / 'yTrain.parq')

        if len(k) != len(k0) or not np.array_equal(k[KEY_TILE].astype(str).str[:6].to_numpy(), tile0) or \
           not all(np.array_equal(k[c].to_numpy(), k0[c].to_numpy()) for c in ('x', 'y')):
            raise ValueError('Pooled split: the keys of %d differ from those of %d.' % (y, years[0]))

        if not np.array_equal(yy.iloc[:, 0].to_numpy(), y0.iloc[:, 0].to_numpy()):
            raise ValueError('Pooled split: the labels of %d differ from those of %d.' % (y, years[0]))

    # ---
    # Year of each row: per stratum, shuffle and deal to the years from a random start.
    # ---
    labels = y0.iloc[:, 0].to_numpy()
    strata = _strata(k0, labels)
    rng = np.random.default_rng(seed)
    yearOf = np.empty(len(k0), np.int16)
    Y = np.array(years, np.int16)
    n = len(years)

    for _, idx in pd.Series(np.arange(len(k0))).groupby(strata):
        i = rng.permutation(idx.to_numpy())
        yearOf[i] = Y[(np.arange(len(i)) + rng.integers(0, n)) % n]

    # ---
    # Predictors: start from the first year, replace each other year's rows.
    # ---
    x = pd.read_parquet(first / 'xTrain.parq', columns=predictors)

    for y in years[1:]:
        m = yearOf == y
        xy = pd.read_parquet(splits[y] / 'xTrain.parq', columns=predictors)
        x.loc[m, predictors] = xy.loc[m, predictors].to_numpy()
        del xy

    x.to_parquet(out / 'xTrain.parq', index=False)
    y0.to_parquet(out / 'yTrain.parq', index=False)
    k0.to_parquet(out / 'keysTrain.parq', index=False)
    k0.assign(year=yearOf).to_parquet(out / 'assign.parq', index=False)

    perYear = {int(k): int(v) for k, v in pd.Series(yearOf).value_counts().sort_index().items()}
    manifest = dict(format='PooledSplit 1', years=years, seed=seed, predictors=list(predictors), rows=len(k0),
                    rowsPerYear=perYear, labelBins=LABEL_BINS,
                    sha256={f: _sha256(out / f) for f in FILES},
                    sources={str(y): dict(dir=str(splits[y]),
                                          sha256={f: _sha256(splits[y] / f)
                                                  for f in ('xTrain.parq', 'yTrain.parq', 'keysTrain.parq')})
                             for y in years})
    (out / MANIFEST).write_text(json.dumps(manifest, indent=1))

    if logger:
        logger.info('Pooled split: %d rows over %d years; per year %s' % (len(k0), n, perYear))

    return manifest


# ----------------------------------------------------------------------------
# check
# ----------------------------------------------------------------------------
def check(out: Path, splits: dict, sample: int = 20000, logger=None) -> dict:
    """Fail (ValueError) unless sampled rows of the pooled split equal their year's row in the source splits, every stratum is
    dealt evenly (+-1 row) and the output files match the SHA-256 in pooled_split.json.  Returns a summary."""

    out = Path(out)
    manifest = json.loads((out / MANIFEST).read_text())
    predictors = manifest['predictors']

    for f in FILES:
        if _sha256(out / f) != manifest['sha256'][f]:
            raise ValueError('Pooled split: ' + f + ' does not match its SHA-256 in ' + MANIFEST)

    x = pd.read_parquet(out / 'xTrain.parq')
    a = pd.read_parquet(out / 'assign.parq')
    i = np.sort(np.random.default_rng(1).choice(len(x), min(sample, len(x)), replace=False))
    bad = []

    for y in manifest['years']:
        j = i[a.year.to_numpy()[i] == y]
        s = pd.read_parquet(splits[y] / 'xTrain.parq', columns=predictors).iloc[j]
        if not np.array_equal(s.to_numpy(), x.iloc[j][predictors].to_numpy(), equal_nan=True):
            bad.append(y)

    labels = pd.read_parquet(out / 'yTrain.parq').iloc[:, 0].to_numpy()
    counts = pd.crosstab(_strata(a, labels), a.year)
    spread = int((counts.max(axis=1) - counts.min(axis=1)).max())
    summary = dict(sampled=len(i), yearsWithDifferences=bad, maxStratumSpread=spread)

    if logger:
        logger.info('Pooled split check: ' + json.dumps(summary))

    if bad or spread > 1:
        raise ValueError('Pooled split check failed: ' + json.dumps(summary))

    return summary
